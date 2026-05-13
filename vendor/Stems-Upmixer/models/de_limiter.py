from __future__ import annotations

import math
import pathlib
import sys
import types
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DeLimiterConfig:
    sample_rate: int = 44100
    channels: int = 2
    condition_dim: int = 0
    encoder_dim: int = 128
    bottleneck_dim: int = 128
    hidden_dim: int = 256
    encoder_kernel_size: int = 64
    encoder_stride: int = 32
    tcn_kernel_size: int = 3
    blocks: int = 5
    repeats: int = 2
    mask_init: float = 0.96


@dataclass
class DeLimiterProcessStats:
    input_sample_rate: int
    model_sample_rate: int
    chunks: int
    mix: float
    peak_before: float
    peak_after: float
    peak_scale: float


class GlobalLayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = (x - mean).pow(2).mean(dim=(1, 2), keepdim=True)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class TcnBlock(nn.Module):
    def __init__(
        self, bottleneck_dim: int, hidden_dim: int, kernel_size: int, dilation: int
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(bottleneck_dim, hidden_dim, 1),
            nn.PReLU(hidden_dim),
            GlobalLayerNorm(hidden_dim),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_dim,
            ),
            nn.PReLU(hidden_dim),
            GlobalLayerNorm(hidden_dim),
        )
        self.residual = nn.Conv1d(hidden_dim, bottleneck_dim, 1)
        self.skip = nn.Conv1d(hidden_dim, bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(x)
        return x + self.residual(hidden), self.skip(hidden)


class DeLimiterNet(nn.Module):
    """Conv-TasNet style SGI de-limiter.

    The network predicts a waveform-rate gain curve in [0, 1] and applies it
    directly to the input waveform. This follows the sample-wise gain inversion
    idea from the De-limiter paper while keeping phase unchanged.
    """

    def __init__(self, config: DeLimiterConfig | dict[str, Any] | None = None) -> None:
        super().__init__()
        if config is None:
            config = DeLimiterConfig()
        if isinstance(config, dict):
            config = DeLimiterConfig(**config)
        self.config = config
        self.encoder = nn.Conv1d(
            config.channels,
            config.encoder_dim,
            config.encoder_kernel_size,
            stride=config.encoder_stride,
            bias=False,
        )
        self.encoder_norm = GlobalLayerNorm(config.encoder_dim)
        self.bottleneck = nn.Conv1d(config.encoder_dim, config.bottleneck_dim, 1)
        self.condition_proj = (
            nn.Linear(config.condition_dim, config.bottleneck_dim)
            if int(config.condition_dim) > 0
            else None
        )
        blocks: list[nn.Module] = []
        for _repeat in range(config.repeats):
            for block_index in range(config.blocks):
                blocks.append(
                    TcnBlock(
                        config.bottleneck_dim,
                        config.hidden_dim,
                        config.tcn_kernel_size,
                        dilation=2**block_index,
                    )
                )
        self.tcn = nn.ModuleList(blocks)
        self.mask_pre = nn.Sequential(
            nn.PReLU(config.bottleneck_dim),
            nn.Conv1d(config.bottleneck_dim, config.encoder_dim, 1),
        )
        self.decoder = nn.ConvTranspose1d(
            config.encoder_dim,
            config.channels,
            config.encoder_kernel_size,
            stride=config.encoder_stride,
        )
        self._init_mask_bias(config.mask_init)

    def _init_mask_bias(self, value: float) -> None:
        value = min(max(float(value), 1e-4), 1.0 - 1e-4)
        bias = math.log(value / (1.0 - value))
        decoder_bias = self.decoder.bias
        if decoder_bias is None:
            raise RuntimeError("DeLimiter decoder must be created with bias=True.")
        nn.init.constant_(decoder_bias, bias)

    def _right_pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        length = int(x.shape[-1])
        kernel = int(self.config.encoder_kernel_size)
        stride = int(self.config.encoder_stride)
        if length <= kernel:
            padded_length = kernel
        else:
            frames = math.ceil((length - kernel) / stride) + 1
            padded_length = (frames - 1) * stride + kernel
        pad = max(0, padded_length - length)
        if pad:
            x = F.pad(x, (0, pad))
        return x, pad

    def estimate_gain(
        self, x: torch.Tensor, condition: torch.Tensor | None = None
    ) -> torch.Tensor:
        padded, pad = self._right_pad(x)
        encoded = F.relu(self.encoder(padded))
        hidden = self.bottleneck(self.encoder_norm(encoded))
        if self.condition_proj is not None:
            if condition is None:
                condition = torch.zeros(
                    x.shape[0],
                    int(self.config.condition_dim),
                    dtype=x.dtype,
                    device=x.device,
                )
            hidden = hidden + self.condition_proj(
                condition.to(dtype=hidden.dtype)
            ).unsqueeze(-1)
        skips: list[torch.Tensor] = []
        for block in self.tcn:
            hidden, skip = block(hidden)
            skips.append(skip)
        masked_basis = self.mask_pre(torch.stack(skips, dim=0).sum(dim=0))
        gain = torch.sigmoid(self.decoder(masked_basis))
        gain = gain[..., : padded.shape[-1]]
        if pad:
            gain = gain[..., :-pad]
        return gain

    def forward(
        self,
        x: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        return_gain: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        gain = self.estimate_gain(x, condition=condition)
        output = x * gain
        if return_gain:
            return output, gain
        return output


def checkpoint_config(checkpoint: dict[str, Any]) -> DeLimiterConfig:
    config = checkpoint.get("config") or checkpoint.get("model_config") or {}
    if isinstance(config, DeLimiterConfig):
        return config
    return DeLimiterConfig(**dict(config))


def _install_pathlib_local_pickle_compat() -> None:
    if "pathlib._local" in sys.modules:
        return
    module = types.ModuleType("pathlib._local")
    for name in dir(pathlib):
        setattr(module, name, getattr(pathlib, name))

    class PosixPath(pathlib.PurePosixPath):
        pass

    class WindowsPath(pathlib.PureWindowsPath):
        pass

    setattr(module, "PosixPath", PosixPath)
    setattr(module, "WindowsPath", WindowsPath)
    sys.modules["pathlib._local"] = module


def _safetensors_metadata_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.suffix.lower() in {".yaml", ".yml"}:
        return checkpoint_path
    return checkpoint_path.with_suffix(".yaml")


def _load_safetensors_checkpoint(
    checkpoint_path: Path,
    *,
    device: str | torch.device,
) -> tuple[dict[str, torch.Tensor], DeLimiterConfig]:
    try:
        import yaml
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError(
            "safetensors checkpoints require safetensors and PyYAML to be installed"
        ) from exc

    metadata_path = _safetensors_metadata_path(checkpoint_path)
    if not metadata_path.exists():
        raise RuntimeError(f"safetensors metadata YAML does not exist: {metadata_path}")
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
    if not isinstance(metadata, dict):
        raise RuntimeError(f"safetensors metadata YAML is invalid: {metadata_path}")
    weights_name = metadata.get("weights")
    weights_path = checkpoint_path
    if checkpoint_path.suffix.lower() in {".yaml", ".yml"}:
        if not weights_name:
            raise RuntimeError(
                f"safetensors metadata YAML is missing weights: {metadata_path}"
            )
        weights_path = metadata_path.parent / str(weights_name)
    if weights_name and Path(str(weights_name)).name != weights_path.name:
        raise RuntimeError(
            f"safetensors metadata points at {weights_name}, not {weights_path.name}"
        )
    if not weights_path.exists():
        raise RuntimeError(f"safetensors weights do not exist: {weights_path}")
    return load_file(str(weights_path), device=str(device)), checkpoint_config(metadata)


def load_de_limiter_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> DeLimiterNet:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() in {".safetensors", ".yaml", ".yml"}:
        state, config = _load_safetensors_checkpoint(checkpoint_path, device=device)
        model = DeLimiterNet(config).to(device)
        model.load_state_dict(state)
        model.eval()
        return model

    _install_pathlib_local_pickle_compat()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint_config(checkpoint if isinstance(checkpoint, dict) else {})
    model = DeLimiterNet(config).to(device)
    state = (
        checkpoint.get("model_state_dict")
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    if state is None:
        state = checkpoint.get("state_dict")
    if state is None:
        raise RuntimeError(
            f"checkpoint does not contain model weights: {checkpoint_path}"
        )
    model.load_state_dict(state)
    model.eval()
    return model


def de_limiter_checkpoint_payload(
    model: DeLimiterNet,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "bs-roformer-kai-sgi-de-limiter-v1",
        "config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    return payload


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def de_limiter_condition_vector(
    name: str | None,
    condition_dim: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    if int(condition_dim) <= 0:
        return None
    names = ("mix", "vocals", "drums", "bass", "guitar", "piano", "other")
    label = (name or "mix").strip().lower()
    index = names.index(label) if label in names else 0
    vector = torch.zeros(int(condition_dim), dtype=torch.float32, device=device)
    if index < int(condition_dim):
        vector[index] = 1.0
    return vector


def _resample_audio(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if int(source_rate) == int(target_rate):
        return audio
    from scipy import signal

    ratio = Fraction(int(target_rate), int(source_rate)).limit_denominator(1000)
    return signal.resample_poly(
        audio, ratio.numerator, ratio.denominator, axis=0
    ).astype(np.float32, copy=False)


def _adapt_channels_for_model(
    audio: np.ndarray, channels: int
) -> tuple[np.ndarray, int]:
    original_channels = int(audio.shape[1])
    if original_channels == channels:
        return audio, original_channels
    if original_channels == 1 and channels == 2:
        return np.repeat(audio, 2, axis=1), original_channels
    if original_channels > channels:
        return audio[:, :channels], original_channels
    pad = np.zeros((audio.shape[0], channels - original_channels), dtype=audio.dtype)
    return np.concatenate([audio, pad], axis=1), original_channels


def _restore_original_channels(audio: np.ndarray, original_channels: int) -> np.ndarray:
    if audio.shape[1] == original_channels:
        return audio
    if original_channels == 1:
        return audio.mean(axis=1, keepdims=True)
    if audio.shape[1] > original_channels:
        return audio[:, :original_channels]
    pad = np.zeros(
        (audio.shape[0], original_channels - audio.shape[1]), dtype=audio.dtype
    )
    return np.concatenate([audio, pad], axis=1)


@torch.inference_mode()
def apply_de_limiter_to_array(
    audio: np.ndarray,
    sample_rate: int,
    checkpoint_path: str | Path,
    *,
    condition_name: str | None = None,
    mix: float = 1.0,
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
    device: str | torch.device = "auto",
    peak_limit: float = 0.98,
) -> tuple[np.ndarray, DeLimiterProcessStats]:
    device_obj = _resolve_device(device)
    model = load_de_limiter_checkpoint(checkpoint_path, device=device_obj)
    return apply_de_limiter_model_to_array(
        audio,
        sample_rate,
        model,
        condition_name=condition_name,
        mix=mix,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        device=device_obj,
        peak_limit=peak_limit,
    )


@torch.inference_mode()
def apply_de_limiter_model_to_array(
    audio: np.ndarray,
    sample_rate: int,
    model: DeLimiterNet,
    *,
    condition_name: str | None = None,
    mix: float = 1.0,
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
    device: str | torch.device = "auto",
    peak_limit: float = 0.98,
) -> tuple[np.ndarray, DeLimiterProcessStats]:
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.asarray(audio, dtype=np.float32)
    input_length = int(audio.shape[0])
    input_channels = int(audio.shape[1])
    input_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    model_rate = int(model.config.sample_rate)
    work = _resample_audio(audio, sample_rate, model_rate)
    work, original_channels = _adapt_channels_for_model(
        work, int(model.config.channels)
    )

    chunk_samples = max(
        int(model_rate * max(float(chunk_seconds), 0.25)),
        int(model.config.encoder_kernel_size),
    )
    overlap_samples = int(model_rate * max(float(overlap_seconds), 0.0))
    overlap_samples = min(overlap_samples, max(0, chunk_samples - 1))
    hop_samples = max(1, chunk_samples - overlap_samples)
    total = int(work.shape[0])
    device_obj = _resolve_device(device)
    model = model.to(device_obj).eval()
    condition = de_limiter_condition_vector(
        condition_name, int(model.config.condition_dim), device=device_obj
    )
    if condition is not None:
        condition = condition.unsqueeze(0)

    if total == 0:
        processed = work.copy()
        chunks = 0
    else:
        accum = np.zeros_like(work, dtype=np.float32)
        weights = np.zeros((total, 1), dtype=np.float32)
        chunks = 0
        for start in range(0, total, hop_samples):
            end = min(total, start + chunk_samples)
            chunk = work[start:end]
            if chunk.shape[0] < int(model.config.encoder_kernel_size):
                pad = int(model.config.encoder_kernel_size) - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad), (0, 0)), mode="constant")
            tensor = torch.from_numpy(chunk.T[None]).to(device_obj)
            estimate = model(tensor, condition=condition)
            estimate_np = estimate[0].detach().cpu().numpy().T[: end - start]
            weight = np.ones((end - start, 1), dtype=np.float32)
            if overlap_samples > 0:
                fade = min(overlap_samples, end - start)
                if start > 0 and fade > 1:
                    weight[:fade, 0] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
                if end < total and fade > 1:
                    weight[-fade:, 0] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
            accum[start:end] += estimate_np * weight
            weights[start:end] += weight
            chunks += 1
            if end >= total:
                break
        processed = accum / np.maximum(weights, 1e-8)

    mix = min(max(float(mix), 0.0), 1.0)
    processed = work * (1.0 - mix) + processed * mix
    processed = _restore_original_channels(processed, original_channels)
    processed = _resample_audio(processed, model_rate, sample_rate)
    processed = processed[:input_length]
    processed = _restore_original_channels(processed, input_channels)
    peak_after = float(np.max(np.abs(processed))) if processed.size else 0.0
    peak_scale = 1.0
    if peak_limit > 0.0 and peak_after > float(peak_limit):
        peak_scale = float(peak_limit) / max(peak_after, 1e-12)
        processed *= peak_scale
        peak_after = float(np.max(np.abs(processed))) if processed.size else 0.0

    return processed.astype(np.float32, copy=False), DeLimiterProcessStats(
        input_sample_rate=int(sample_rate),
        model_sample_rate=model_rate,
        chunks=chunks,
        mix=mix,
        peak_before=input_peak,
        peak_after=peak_after,
        peak_scale=peak_scale,
    )
