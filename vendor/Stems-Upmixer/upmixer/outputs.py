"""Render targets, loudness measurement, cover extraction, and output files."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


from .filters import fold_714_to_51_filter, fold_714_to_stereo_filter, stereo_fold_temporal_filter
from .models import (
    CHANNELS_714,
    MULTIBAND_RANGES,
    REPO_ROOT,
    AutomationLanes,
    MultibandItem,
    MultibandStats,
    StemDeLimiterStemDecision,
    StemStats,
    StereoEnvelopeMatchResult,
    VolumeStats,
)
from .utils import (
    append_filter_complex_arg,
    clamp,
    db_to_amp,
    fmt,
    print_elapsed,
    read_audio_float,
    run,
    safe_label,
    volumedetect_complex,
)

NATIVE_RENDER_TRUE_PEAK_MARGIN_DB = 0.5


def extract_cover(
    ffmpeg: str, input_file: Path | None, output_path: Path
) -> Path | None:
    if input_file is None or not input_file.exists():
        return None
    proc = run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_file),
            "-map",
            "0:v:0",
            "-frames:v",
            "1",
            "-update",
            "1",
            str(output_path),
        ],
        check=False,
        capture=True,
    )
    if (
        proc.returncode != 0
        or not output_path.exists()
        or output_path.stat().st_size == 0
    ):
        return None
    return output_path


def analyze_rendered_fold_numpy(
    name: str, path: Path, lfe_fold_gain: float
) -> StemStats:
    from scipy import signal

    audio_raw, sample_rate = read_audio_float(path, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] < len(CHANNELS_714):
        raise RuntimeError(
            f"Expected 7.1.4 audio with {len(CHANNELS_714)} channels, got {audio.shape[1]}"
        )
    audio = audio[:, : len(CHANNELS_714)]
    folded = _fold_714_to_stereo_native(audio, lfe_fold_gain)

    def filtered_stats(kind: str, freq: float | tuple[float, float]) -> VolumeStats:
        sos = signal.butter(4, freq, btype=kind, fs=sample_rate, output="sos")
        filtered = cast(AudioArray, signal.sosfilt(sos, folded, axis=0))
        return _volume_stats_from_array(filtered)

    mono = 0.5 * (folded[:, 0] + folded[:, 1])
    side = 0.5 * (folded[:, 0] - folded[:, 1])
    return StemStats(
        name=name,
        path=path,
        full=_volume_stats_from_array(folded),
        low=filtered_stats("lowpass", 180.0),
        mid=filtered_stats("bandpass", (180.0, 2500.0)),
        high=filtered_stats("highpass", 2500.0),
        mono=_volume_stats_from_array(mono),
        side=_volume_stats_from_array(side),
    )


def analyze_rendered_fold_ffmpeg(
    ffmpeg: str, name: str, path: Path, lfe_fold_gain: float
) -> StemStats:
    def folded_volumedetect(label: str, audio_filter: str) -> VolumeStats:
        folded_label = f"{label}_fold"
        chain = f"{audio_filter},volumedetect" if audio_filter else "volumedetect"
        filter_complex = ";".join(
            [
                fold_714_to_stereo_filter(
                    "[0:a]", f"[{folded_label}]", lfe_fold_gain, limiter=False
                ),
                f"[{folded_label}]{chain}[{label}]",
            ]
        )
        return volumedetect_complex(ffmpeg, path, filter_complex, label)

    return StemStats(
        name=name,
        path=path,
        full=folded_volumedetect("actual_fold_full", ""),
        low=folded_volumedetect("actual_fold_low", "lowpass=f=180"),
        mid=folded_volumedetect("actual_fold_mid", "highpass=f=180,lowpass=f=2500"),
        high=folded_volumedetect("actual_fold_high", "highpass=f=2500"),
        mono=folded_volumedetect("actual_fold_mono", "pan=mono|c0=0.5*FL+0.5*FR"),
        side=folded_volumedetect("actual_fold_side", "pan=mono|c0=0.5*FL-0.5*FR"),
    )


def analyze_rendered_fold(
    ffmpeg: str,
    path: Path,
    lfe_fold_gain: float,
    backend: str,
) -> tuple[StemStats, str]:
    if backend in {"auto", "numpy"}:
        try:
            return analyze_rendered_fold_numpy(
                "actual_fold", path, lfe_fold_gain
            ), "numpy"
        except Exception as exc:
            if backend == "numpy":
                raise
            print(
                f"post-render fold analysis unavailable with numpy: {exc}. Falling back to ffmpeg.",
                file=sys.stderr,
            )
    return analyze_rendered_fold_ffmpeg(
        ffmpeg, "actual_fold", path, lfe_fold_gain
    ), "ffmpeg"


def multiband_stereo_audio_numpy(
    path: Path, lfe_fold_gain: float
) -> tuple[AudioArray, int]:
    import numpy as np

    audio_raw, sample_rate = read_audio_float(path, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] >= len(CHANNELS_714):
        audio = audio[:, : len(CHANNELS_714)]
        return _fold_714_to_stereo_native(audio, lfe_fold_gain), int(sample_rate)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, int(sample_rate)


def multiband_rms_db(data: AudioArray) -> float:
    import numpy as np

    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    return 20.0 * math.log10(max(rms, 1e-12))


def _volume_stats_from_array(data: AudioArray) -> VolumeStats:
    import numpy as np

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    peak_db = 20.0 * math.log10(max(peak, 1e-12))
    rms_db = 20.0 * math.log10(max(rms, 1e-12))
    return VolumeStats(mean_db=rms_db, max_db=peak_db)


def multiband_filter_numpy(
    data: AudioArray, sample_rate: int, low_hz: float, high_hz: float
) -> AudioArray:
    from scipy import signal

    nyquist = sample_rate * 0.5
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.98)
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    return cast(AudioArray, signal.sosfilt(sos, data, axis=0))


def analyze_multiband_numpy(
    name: str, path: Path, lfe_fold_gain: float
) -> MultibandStats:
    audio, sample_rate = multiband_stereo_audio_numpy(path, lfe_fold_gain)
    rows: list[MultibandItem] = []
    for band, low_hz, high_hz in MULTIBAND_RANGES:
        full_band = multiband_filter_numpy(audio, sample_rate, low_hz, high_hz)
        mono_band = 0.5 * (full_band[:, 0] + full_band[:, 1])
        side_band = 0.5 * (full_band[:, 0] - full_band[:, 1])
        rows.append(
            MultibandItem(
                band=band,
                full_db=multiband_rms_db(full_band),
                side_minus_mono_db=multiband_rms_db(side_band)
                - multiband_rms_db(mono_band),
            )
        )
    return MultibandStats(name=name, rows=tuple(rows))


def print_multiband_fold_analysis(
    bed_path: Path,
    *,
    lfe_fold_gain: float,
    guard_stats: StemStats | None,
    style_stats: StemStats | None,
) -> None:
    references = []
    if guard_stats is not None:
        references.append(("guard", guard_stats.path))
    if style_stats is not None:
        references.append(("style", style_stats.path))
    if not references:
        return
    try:
        actual = analyze_multiband_numpy("actual", bed_path, lfe_fold_gain)
        reference_rows = [
            analyze_multiband_numpy(label, path, lfe_fold_gain)
            for label, path in references
        ]
    except Exception as exc:
        print(f"Post-render 8-band analysis unavailable: {exc}", flush=True)
        return

    print("Post-render 8-band fold analysis:", flush=True)
    for reference in reference_rows:
        print(f"vs {reference.name}:", flush=True)
        print("band        full   ref  drift   S/M    ref  drift", flush=True)
        for actual_item, reference_item in zip(actual.rows, reference.rows, strict=False):
            full_drift = actual_item.full_db - reference_item.full_db
            width_drift = (
                actual_item.side_minus_mono_db - reference_item.side_minus_mono_db
            )
            print(
                f"{actual_item.band:<9} "
                f"{actual_item.full_db:>6.1f} "
                f"{reference_item.full_db:>6.1f} "
                f"{full_drift:>+6.1f} "
                f"{actual_item.side_minus_mono_db:>6.1f} "
                f"{reference_item.side_minus_mono_db:>6.1f} "
                f"{width_drift:>+6.1f}",
                flush=True,
            )


def filename_number_token(value: float) -> str:
    if abs(value) < 0.005:
        return "0"
    sign = "m" if value < 0.0 else "p"
    body = f"{abs(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"{sign}{body}"


def stereo_loudness_suffix(
    normalize: str,
    target_i: float,
    target_tp: float,
    *,
    envelope_match: bool = False,
) -> str:
    root = "stereo_envmatch" if envelope_match else "stereo"
    if normalize == "off":
        return root
    mode = "" if normalize == "loudnorm" else f"_{normalize}"
    return f"{root}{mode}_lufs_{filename_number_token(target_i)}_tp_{filename_number_token(target_tp)}"


def surround_loudness_suffix(normalize: str, target_i: float, target_tp: float) -> str:
    if normalize == "off":
        return "upmix5.1"
    return f"upmix5.1_lufs_{filename_number_token(target_i)}_tp_{filename_number_token(target_tp)}"


def load_de_limiter_module():
    import importlib

    repo_root = REPO_ROOT.parent
    for path in (REPO_ROOT, repo_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
    return importlib.import_module("models.de_limiter")


def load_apply_de_limiter_to_array():
    return load_de_limiter_module().apply_de_limiter_to_array


def print_de_limiter_stats(label: str, stats: Any, start_time: float) -> None:
    print(
        f"{label}: "
        f"chunks={stats.chunks} "
        f"sr={stats.input_sample_rate}->{stats.model_sample_rate}->{stats.input_sample_rate} "
        f"mix={stats.mix:.2f} "
        f"peak={stats.peak_before:.4f}->{stats.peak_after:.4f} "
        f"peak_scale={stats.peak_scale:.4f}",
        flush=True,
    )
    print_elapsed(label, start_time)


def _parse_loudnorm_json_object(stderr: str) -> dict[str, Any]:
    start = stderr.find("{")
    end = stderr.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("ffmpeg loudnorm output did not include JSON measurements")
    data = json.loads(stderr[start : end + 1])
    if not isinstance(data, dict):
        raise RuntimeError("ffmpeg loudnorm JSON measurements were not an object")
    return data


def parse_loudnorm_json(stderr: str) -> dict[str, str]:
    data = _parse_loudnorm_json_object(stderr)
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"ffmpeg loudnorm JSON is missing: {', '.join(missing)}")
    return {key: str(data[key]) for key in required}


def parse_loudnorm_output_loudness(stderr: str) -> dict[str, float]:
    data = _parse_loudnorm_json_object(stderr)
    required = ("output_i", "output_tp", "output_lra", "output_thresh")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(
            f"ffmpeg loudnorm output JSON is missing: {', '.join(missing)}"
        )
    return {
        "integrated_lufs": round(float(data["output_i"]), 4),
        "true_peak_dbtp": round(float(data["output_tp"]), 4),
        "lra_lu": round(float(data["output_lra"]), 4),
        "threshold_lufs": round(float(data["output_thresh"]), 4),
    }


def loudnorm_filter(
    input_label: str,
    output_label: str,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
    measurements: dict[str, str] | None,
    print_format: str,
    channel_layout: str = "stereo",
) -> str:
    args = [
        f"I={fmt(target_i)}",
        f"TP={fmt(target_tp)}",
        f"LRA={fmt(target_lra)}",
    ]
    if measurements is not None:
        args.extend(
            [
                f"measured_I={measurements['input_i']}",
                f"measured_TP={measurements['input_tp']}",
                f"measured_LRA={measurements['input_lra']}",
                f"measured_thresh={measurements['input_thresh']}",
                f"offset={measurements['target_offset']}",
                "linear=true",
            ]
        )
    args.append(f"print_format={print_format}")
    return f"{input_label}loudnorm={':'.join(args)},aresample=48000,aformat=channel_layouts={channel_layout}{output_label}"


def source_stereo_filter(input_label: str, output_label: str, *, limiter: bool) -> str:
    filter_text = f"{input_label}aresample=48000,aformat=channel_layouts=stereo"
    if limiter:
        filter_text += ",alimiter=limit=0.98"
    return f"{filter_text}{output_label}"


def apply_de_limiter_wav(
    input_path: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    condition_name: str | None,
    mix: float,
    chunk_seconds: float,
    overlap_seconds: float,
    device: str,
    peak_limit: float,
    label: str = "Stereo SGI de-limiter",
) -> None:
    import soundfile as sf

    apply_de_limiter_to_array = load_apply_de_limiter_to_array()

    audio_raw, sample_rate = sf.read(input_path, always_2d=True, dtype="float32")
    audio = cast(AudioArray, audio_raw)
    timer = time.perf_counter()
    processed, stats = apply_de_limiter_to_array(
        audio,
        int(sample_rate),
        checkpoint_path,
        condition_name=condition_name,
        mix=mix,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
        device=device,
        peak_limit=peak_limit,
    )
    sf.write(output_path, processed, int(sample_rate), subtype="PCM_24")
    print_de_limiter_stats(label, stats, timer)


def apply_stem_de_limiter_to_stems(
    stems: dict[str, Path],
    checkpoint_path: Path,
    output_dir: Path,
    *,
    mix: float,
    chunk_seconds: float,
    overlap_seconds: float,
    device: str,
    peak_limit: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: dict[str, Path] = {}
    print(f"Stem SGI de-limiter: checkpoint={checkpoint_path}", flush=True)
    import soundfile as sf

    de_limiter_module = load_de_limiter_module()
    device_obj = de_limiter_module._resolve_device(device)
    model = de_limiter_module.load_de_limiter_checkpoint(
        checkpoint_path, device=device_obj
    )
    model = model.to(device_obj).eval()
    print(f"Stem SGI de-limiter: device={device_obj}", flush=True)
    for stem, path in stems.items():
        output_path = output_dir / f"{safe_label(stem)}_sgi_delimited.wav"
        audio_raw, sample_rate = sf.read(path, always_2d=True, dtype="float32")
        audio = cast(AudioArray, audio_raw)
        timer = time.perf_counter()
        result, stats = de_limiter_module.apply_de_limiter_model_to_array(
            audio,
            int(sample_rate),
            model,
            condition_name=stem,
            mix=mix,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
            device=device_obj,
            peak_limit=peak_limit,
        )
        sf.write(output_path, result, int(sample_rate), subtype="PCM_24")
        print_de_limiter_stats(f"Stem SGI de-limiter {stem}", stats, timer)
        processed[stem] = output_path
    return processed


def prepare_stem_de_limiter_dir(
    output_dir: Path, slug: str, *, keep_stems: bool
) -> Path:
    if not keep_stems:
        return Path(tempfile.mkdtemp(prefix=f"{slug}_stem_sgi_", dir=output_dir))

    keep_dir = output_dir / "SGI_Stems"
    keep_dir.mkdir(parents=True, exist_ok=True)
    for path in keep_dir.glob("*_sgi_delimited.wav"):
        path.unlink(missing_ok=True)
    return keep_dir


def prune_stem_de_limiter_candidate_files(
    candidate_stems: dict[str, Path],
    stem_decisions: tuple[StemDeLimiterStemDecision, ...],
    *,
    guard_accepted: bool,
) -> None:
    applied_stems = {
        decision.stem
        for decision in stem_decisions
        if guard_accepted and decision.status == "applied"
    }
    for stem, path in candidate_stems.items():
        if stem not in applied_stems:
            path.unlink(missing_ok=True)


def measure_stereo_loudnorm(
    ffmpeg: str,
    input_path: Path,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> dict[str, str]:
    timer = time.perf_counter()
    proc = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(input_path),
            "-af",
            loudnorm_filter(
                "",
                "",
                target_i=target_i,
                target_tp=target_tp,
                target_lra=target_lra,
                measurements=None,
                print_format="json",
            ),
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    print_elapsed("stereo loudnorm measure", timer)
    return parse_loudnorm_json(proc.stderr or "")


def measure_51_loudnorm(
    ffmpeg: str,
    bed_path: Path,
    lfe_fold_gain: float,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
    label: str = "51_loudnorm_measure",
) -> dict[str, str]:
    first_pass_filter = ";".join(
        [
            fold_714_to_51_filter("[0:a]", "[fold]", lfe_fold_gain, limiter=False),
            loudnorm_filter(
                "[fold]",
                "[norm]",
                target_i=target_i,
                target_tp=target_tp,
                target_lra=target_lra,
                measurements=None,
                print_format="json",
                channel_layout="5.1(side)",
            ),
        ]
    )
    temp_paths: list[Path] = []
    first_pass_cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(bed_path)]
    append_filter_complex_arg(
        first_pass_cmd, first_pass_filter, temp_paths, label=label
    )
    first_pass_cmd.extend(["-map", "[norm]", "-f", "null", "-"])
    timer = time.perf_counter()
    try:
        first_pass = run(first_pass_cmd, capture=True)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed(f"{label} loudnorm measure", timer)
    return parse_loudnorm_json(first_pass.stderr or "")


def native_loudnorm_binary() -> Path | None:
    env_path = os.environ.get("STEMS_UPMIXER_LOUDNORM_BIN", "").strip()
    candidates = []
    if env_path:
        candidates.append(Path(env_path))
    native_root = REPO_ROOT / "native_loudnorm" / "target"
    candidates.extend(
        [
            native_root / "release" / "stems-upmixer-loudnorm",
            native_root / "debug" / "stems-upmixer-loudnorm",
        ]
    )
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    if not env_path:
        return build_native_loudnorm_binary(native_root / "release" / "stems-upmixer-loudnorm")
    return None


def build_native_loudnorm_binary(expected_path: Path) -> Path | None:
    manifest = REPO_ROOT / "native_loudnorm" / "Cargo.toml"
    if not manifest.is_file() or shutil.which("cargo") is None:
        return None
    timer = time.perf_counter()
    proc = run(
        [
            "cargo",
            "build",
            "--release",
            "--quiet",
            "--manifest-path",
            str(manifest),
        ],
        check=False,
        capture=True,
    )
    print_elapsed("native loudnorm build", timer)
    if proc.returncode != 0:
        if proc.stderr:
            print(proc.stderr.strip(), file=sys.stderr, flush=True)
        return None
    if expected_path.exists() and os.access(expected_path, os.X_OK):
        return expected_path
    return None


def measure_51_loudnorm_native(
    bed_path: Path,
    lfe_fold_gain: float,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> dict[str, str] | None:
    binary = native_loudnorm_binary()
    if binary is None:
        return None
    proc = run(
        [
            str(binary),
            "--bed",
            str(bed_path),
            "--lfe-fold-gain",
            fmt(lfe_fold_gain),
            "--target-i",
            fmt(target_i),
            "--target-tp",
            fmt(target_tp),
            "--target-lra",
            fmt(target_lra),
        ],
        capture=True,
    )
    data = json.loads(proc.stdout or "{}")
    required = ("input_i", "input_tp", "input_lra", "input_thresh")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(
            f"native loudnorm output is missing: {', '.join(missing)}"
        )
    measurements = {key: str(data[key]) for key in required}
    if "target_offset" in data:
        measurements["target_offset"] = str(data["target_offset"])
    return measurements


def render_51_wav_native(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    lfe_fold_gain: float,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> dict[str, float] | None:
    if normalize != "loudnorm":
        render_51_fold_wav(ffmpeg, bed_path, output_path, lfe_fold_gain)
        return None

    binary = native_loudnorm_binary()
    if binary is None:
        raise RuntimeError(
            "native 5.1 WAV render requested but stems-upmixer-loudnorm was not found"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timer = time.perf_counter()
    try:
        proc = run(
            [
                str(binary),
                "--bed",
                str(bed_path),
                "--lfe-fold-gain",
                fmt(lfe_fold_gain),
                "--target-i",
                fmt(target_i),
                "--target-tp",
                fmt(target_tp),
                "--target-lra",
                fmt(target_lra),
                "--render-wav",
                str(output_path),
            ],
            capture=True,
        )
    finally:
        print_elapsed("5.1 WAV native render", timer)
    data = json.loads(proc.stdout or "{}")
    required = ("render_i", "render_tp", "render_lra", "render_thresh")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(
            f"native 5.1 WAV render output is missing: {', '.join(missing)}"
        )
    return {
        "integrated_lufs": float(data["render_i"]),
        "true_peak_dbtp": float(data["render_tp"]) + NATIVE_RENDER_TRUE_PEAK_MARGIN_DB,
        "lra_lu": float(data["render_lra"]),
        "threshold_lufs": float(data["render_thresh"]),
    }


def stereo_linear_gain_db(
    measurements: dict[str, str], target_i: float, target_tp: float
) -> float:
    loudness_gain = float(target_i) - float(measurements["input_i"])
    peak_gain = float(target_tp) - float(measurements["input_tp"])
    return min(loudness_gain, peak_gain)


def _write_pcm24(path: Path, audio: AudioArray, sample_rate: int) -> None:
    import numpy as np
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.asarray(audio, dtype=np.float32),
        int(sample_rate),
        subtype="PCM_24",
    )


def _read_native_fold_source(
    path: Path, *, min_channels: int
) -> tuple[AudioArray, int] | None:
    audio, sample_rate = read_audio_float(path, dtype="float64")
    if int(sample_rate) != 48000 or audio.shape[1] < min_channels:
        return None
    return audio, int(sample_rate)


def _fold_714_to_stereo_native(audio: AudioArray, lfe_fold_gain: float) -> AudioArray:
    import numpy as np

    channels = {channel: audio[:, index] for index, channel in enumerate(CHANNELS_714)}
    left = (
        0.92 * channels["FL"]
        + 0.7071 * channels["FC"]
        + 0.50 * channels["BL"]
        + 0.55 * channels["SL"]
        + 0.35 * channels["TFL"]
        + 0.25 * channels["TBL"]
        + float(lfe_fold_gain) * channels["LFE"]
    )
    right = (
        0.92 * channels["FR"]
        + 0.7071 * channels["FC"]
        + 0.50 * channels["BR"]
        + 0.55 * channels["SR"]
        + 0.35 * channels["TFR"]
        + 0.25 * channels["TBR"]
        + float(lfe_fold_gain) * channels["LFE"]
    )
    return cast(AudioArray, np.stack([left, right], axis=1))


def _fold_714_to_51_native(audio: AudioArray, lfe_fold_gain: float) -> AudioArray:
    import numpy as np

    channels = {channel: audio[:, index] for index, channel in enumerate(CHANNELS_714)}
    folded = np.stack(
        [
            (
                0.92 * channels["FL"]
                + 0.24 * channels["FC"]
                + 0.20 * channels["BL"]
                + 0.18 * channels["TFL"]
                + 0.10 * channels["TBL"]
            ),
            (
                0.92 * channels["FR"]
                + 0.24 * channels["FC"]
                + 0.20 * channels["BR"]
                + 0.18 * channels["TFR"]
                + 0.10 * channels["TBR"]
            ),
            0.78 * channels["FC"] + 0.05 * channels["FL"] + 0.05 * channels["FR"],
            float(lfe_fold_gain) * channels["LFE"],
            (
                0.78 * channels["SL"]
                + 0.40 * channels["BL"]
                + 0.20 * channels["TFL"]
                + 0.34 * channels["TBL"]
            ),
            (
                0.78 * channels["SR"]
                + 0.40 * channels["BR"]
                + 0.20 * channels["TFR"]
                + 0.34 * channels["TBR"]
            ),
        ],
        axis=1,
    )
    return cast(AudioArray, folded)


def _fold_51_to_stereo_native(audio: AudioArray, lfe_fold_gain: float) -> AudioArray:
    import numpy as np

    fl = audio[:, 0]
    fr = audio[:, 1]
    fc = audio[:, 2]
    lfe = audio[:, 3]
    sl = audio[:, 4]
    sr = audio[:, 5]
    left = 0.92 * fl + 0.7071 * fc + 0.55 * sl + float(lfe_fold_gain) * lfe
    right = 0.92 * fr + 0.7071 * fc + 0.55 * sr + float(lfe_fold_gain) * lfe
    return cast(AudioArray, np.stack([left, right], axis=1))


def _render_native_fold_wav(
    input_path: Path,
    output_path: Path,
    *,
    min_channels: int,
    fold,
    lfe_fold_gain: float,
    label: str,
) -> bool:
    timer = time.perf_counter()
    try:
        source = _read_native_fold_source(input_path, min_channels=min_channels)
        if source is None:
            return False
        audio, sample_rate = source
        _write_pcm24(output_path, fold(audio, lfe_fold_gain), sample_rate)
    except Exception as exc:
        print(
            f"{label} native render unavailable: {exc}. Falling back to FFmpeg.",
            flush=True,
        )
        return False
    print_elapsed(f"{label} native render", timer)
    return True


def render_stereo_fold_wav(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    lfe_fold_gain: float,
    automation_lanes: AutomationLanes | None,
) -> None:
    if automation_lanes is None and _render_native_fold_wav(
        bed_path,
        output_path,
        min_channels=len(CHANNELS_714),
        fold=_fold_714_to_stereo_native,
        lfe_fold_gain=lfe_fold_gain,
        label="stereo fold WAV",
    ):
        return

    filter_complex = stereo_fold_temporal_filter(
        "[0:a]",
        "[a]",
        lfe_fold_gain,
        automation_lanes,
        prefix="stereo_fold_wav",
        limiter=False,
    )
    temp_paths: list[Path] = []
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(bed_path),
    ]
    append_filter_complex_arg(cmd, filter_complex, temp_paths, label="stereo_fold_wav")
    cmd.extend(["-map", "[a]", "-c:a", "pcm_s24le", str(output_path)])
    timer = time.perf_counter()
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("stereo fold WAV render", timer)


def _read_stereo_float(path: Path) -> tuple[AudioArray, int]:
    import numpy as np

    audio_raw, sample_rate = read_audio_float(path, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, int(sample_rate)


def _window_rms_db(
    audio: AudioArray,
    sample_rate: int,
    centers_sec: Iterable[float],
    *,
    window_sec: float,
) -> AudioArray:
    import numpy as np

    centers = np.asarray(tuple(float(center) for center in centers_sec), dtype=np.float64)
    if centers.size == 0:
        return cast(AudioArray, np.asarray([], dtype=np.float64))
    if len(audio) <= 0:
        return cast(AudioArray, np.full(centers.shape, -240.0, dtype=np.float64))

    half = max(1, int(sample_rate * max(window_sec, 0.001) * 0.5))
    center_indices = (centers * float(sample_rate)).astype(np.int64)
    starts = np.maximum(0, center_indices - half)
    ends = np.minimum(len(audio), center_indices + half)
    audio64 = np.asarray(audio, dtype=np.float64)
    sample_power = np.mean(audio64 * audio64, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power, dtype=np.float64)))
    counts = np.maximum(ends - starts, 1)
    mean_power = (cumulative[ends] - cumulative[starts]) / counts
    rms = np.sqrt(np.maximum(mean_power, 0.0))
    return cast(AudioArray, 20.0 * np.log10(np.maximum(rms, 1e-12)))


def _smooth_gain_curve(values: AudioArray, smooth_steps: int) -> AudioArray:
    import numpy as np

    if smooth_steps <= 1 or len(values) <= 2:
        return values
    width = max(1, int(smooth_steps))
    kernel = np.ones(width, dtype=np.float64) / float(width)
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = cast(AudioArray, np.pad(values, (pad_left, pad_right), mode="edge"))
    return cast(AudioArray, np.convolve(padded, kernel, mode="valid"))


def apply_stereo_envelope_match(
    reference_path: Path,
    stereo_path: Path,
    output_path: Path,
    *,
    window_sec: float,
    hop_sec: float,
    smooth_sec: float,
    strength: float,
    max_boost_db: float,
    max_cut_db: float,
) -> StereoEnvelopeMatchResult:
    import numpy as np
    import soundfile as sf

    reference_audio, reference_rate = _read_stereo_float(reference_path)
    stereo_audio, stereo_rate = _read_stereo_float(stereo_path)
    duration = len(stereo_audio) / float(stereo_rate)
    hop_sec = max(0.1, float(hop_sec))
    window_sec = max(hop_sec, float(window_sec))
    centers = np.arange(
        hop_sec * 0.5, max(duration, hop_sec * 0.5), hop_sec, dtype=np.float64
    )
    if len(centers) == 0 or centers[-1] < duration:
        centers = np.append(centers, duration)

    reference_db = _window_rms_db(
        reference_audio, reference_rate, centers, window_sec=window_sec
    )
    stereo_db = _window_rms_db(
        stereo_audio, stereo_rate, centers, window_sec=window_sec
    )
    active = np.maximum(reference_db, stereo_db) > -60.0
    if not np.any(active):
        gain_db = np.zeros_like(centers)
        median_delta = 0.0
    else:
        delta = reference_db - stereo_db
        median_delta = float(np.median(delta[active]))
        gain_db = (delta - median_delta) * clamp(float(strength), 0.0, 1.0)
        gain_db = np.clip(gain_db, -abs(float(max_cut_db)), abs(float(max_boost_db)))
        smooth_steps = max(1, int(round(max(0.0, float(smooth_sec)) / hop_sec)))
        gain_db = _smooth_gain_curve(gain_db, smooth_steps)
        gain_db = np.clip(gain_db, -abs(float(max_cut_db)), abs(float(max_boost_db)))

    sample_times = np.arange(len(stereo_audio), dtype=np.float64) / float(stereo_rate)
    interp_centers = np.concatenate(([0.0], centers, [duration]))
    interp_gain_db = (
        np.concatenate(([gain_db[0]], gain_db, [gain_db[-1]]))
        if len(gain_db)
        else np.zeros_like(interp_centers)
    )
    sample_gain = np.interp(sample_times, interp_centers, interp_gain_db)
    matched = stereo_audio * np.power(10.0, sample_gain[:, None] / 20.0)
    peak = float(np.max(np.abs(matched))) if matched.size else 0.0
    if peak > 0.98:
        matched *= 0.98 / peak
    sf.write(output_path, matched, stereo_rate, subtype="PCM_24")

    return StereoEnvelopeMatchResult(
        windows=int(len(centers)),
        median_delta_db=median_delta,
        min_gain_db=float(np.min(gain_db)) if len(gain_db) else 0.0,
        max_gain_db=float(np.max(gain_db)) if len(gain_db) else 0.0,
        mean_abs_gain_db=float(np.mean(np.abs(gain_db))) if len(gain_db) else 0.0,
        strength=clamp(float(strength), 0.0, 1.0),
    )


def render_stereo_source_flac(
    ffmpeg: str,
    source_path: Path,
    output_path: Path,
    tags: dict[str, str],
    title: str,
    compression_level: int,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
    comment: str,
) -> None:
    if normalize == "loudnorm":
        measurements = measure_stereo_loudnorm(
            ffmpeg,
            source_path,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
        )
        filter_complex = loudnorm_filter(
            "[0:a]",
            "[a]",
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
            measurements=measurements,
            print_format="summary",
        )
        output_comment = (
            f"{comment}; normalized to {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
        )
    elif normalize == "linear":
        measurements = measure_stereo_loudnorm(
            ffmpeg,
            source_path,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
        )
        gain_db = stereo_linear_gain_db(measurements, target_i, target_tp)
        gain = db_to_amp(gain_db)
        print(
            f"Stereo linear normalization: gain={gain_db:+.2f} dB "
            f"(input_i={float(measurements['input_i']):+.2f} LUFS, input_tp={float(measurements['input_tp']):+.2f} dBTP)",
            flush=True,
        )
        filter_complex = f"[0:a]aresample=48000,volume={fmt(gain)},aformat=channel_layouts=stereo,alimiter=limit=0.98[a]"
        output_comment = f"{comment}; linear peak-safe gain toward {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
    else:
        filter_complex = source_stereo_filter("[0:a]", "[a]", limiter=True)
        output_comment = comment

    temp_paths: list[Path] = []
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
    ]
    append_filter_complex_arg(
        cmd, filter_complex, temp_paths, label="source_stereo_flac"
    )
    cmd.extend(
        [
            "-map",
            "[a]",
            "-c:a",
            "flac",
            "-compression_level",
            str(compression_level),
            "-metadata",
            f"title={title}",
            "-metadata",
            f"comment={output_comment}",
            str(output_path),
        ]
    )
    for source_key, flac_key in (
        ("artist", "artist"),
        ("album", "album"),
        ("date", "date"),
    ):
        value = tags.get(source_key)
        if value:
            cmd[-1:-1] = ["-metadata", f"{flac_key}={value}"]
    timer = time.perf_counter()
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("source stereo FLAC render", timer)


def render_51_flac(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    tags: dict[str, str],
    title: str,
    lfe_fold_gain: float,
    compression_level: int,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
    loudnorm_measurements: dict[str, str] | None = None,
) -> dict[str, float] | None:
    render_loudness: dict[str, float] | None = None
    if normalize == "loudnorm":
        measurements = loudnorm_measurements or measure_51_loudnorm(
            ffmpeg,
            bed_path,
            lfe_fold_gain,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
            label="51_loudnorm_measure",
        )
        filter_complex = ";".join(
            [
                fold_714_to_51_filter("[0:a]", "[fold]", lfe_fold_gain, limiter=False),
                loudnorm_filter(
                    "[fold]",
                    "[a]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=measurements,
                    print_format="json",
                    channel_layout="5.1(side)",
                ),
            ]
        )
        comment = (
            f"5.1 fold-down normalized to {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
        )
    else:
        filter_complex = fold_714_to_51_filter("[0:a]", "[a]", lfe_fold_gain)
        comment = "5.1 fold-down from upmix bed"

    temp_paths: list[Path] = []
    capture_loudnorm_output = normalize == "loudnorm"
    cmd = [ffmpeg, "-y", "-hide_banner"]
    if capture_loudnorm_output:
        cmd.append("-nostats")
    else:
        cmd.extend(["-loglevel", "error"])
    cmd.extend(["-i", str(bed_path)])
    append_filter_complex_arg(cmd, filter_complex, temp_paths, label="51_flac")
    cmd.extend(
        [
            "-map",
            "[a]",
            "-c:a",
            "flac",
            "-compression_level",
            str(compression_level),
            "-metadata",
            f"title={title}",
            "-metadata",
            f"comment={comment}",
        ]
    )
    for source_key, flac_key in (
        ("artist", "artist"),
        ("album", "album"),
        ("date", "date"),
    ):
        value = tags.get(source_key)
        if value:
            cmd.extend(["-metadata", f"{flac_key}={value}"])
    cmd.append(str(output_path))
    timer = time.perf_counter()
    try:
        proc = run(cmd, capture=capture_loudnorm_output)
        if capture_loudnorm_output:
            try:
                render_loudness = parse_loudnorm_output_loudness(proc.stderr or "")
            except Exception as exc:
                print(
                    f"5.1 FLAC render loudness capture unavailable: {exc}",
                    flush=True,
                )
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("5.1 FLAC render", timer)
    return render_loudness


def render_stereo_flac(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    tags: dict[str, str],
    title: str,
    reference_path: Path | None,
    lfe_fold_gain: float,
    automation_lanes: AutomationLanes | None,
    compression_level: int,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
    envelope_match: str,
    envelope_window_sec: float,
    envelope_hop_sec: float,
    envelope_smooth_sec: float,
    envelope_strength: float,
    envelope_max_boost_db: float,
    envelope_max_cut_db: float,
    de_limiter_checkpoint: Path | None = None,
    de_limiter_mix: float = 1.0,
    de_limiter_chunk_seconds: float = 8.0,
    de_limiter_overlap_seconds: float = 1.0,
    de_limiter_device: str = "auto",
    de_limiter_peak_limit: float = 0.98,
) -> None:
    render_timer = time.perf_counter()
    use_envelope_match = (
        envelope_match != "off"
        and reference_path is not None
        and reference_path.exists()
    )
    if use_envelope_match:
        assert reference_path is not None
        fold_path = output_path.with_suffix(".fold.tmp.wav")
        matched_path = output_path.with_suffix(".envmatch.tmp.wav")
        delimited_path = output_path.with_suffix(".delimit.tmp.wav")
        try:
            render_stereo_fold_wav(
                ffmpeg, bed_path, fold_path, lfe_fold_gain, automation_lanes
            )
            envelope_timer = time.perf_counter()
            result = apply_stereo_envelope_match(
                reference_path,
                fold_path,
                matched_path,
                window_sec=envelope_window_sec,
                hop_sec=envelope_hop_sec,
                smooth_sec=envelope_smooth_sec,
                strength=envelope_strength,
                max_boost_db=envelope_max_boost_db,
                max_cut_db=envelope_max_cut_db,
            )
            print_elapsed("stereo envelope match", envelope_timer)
            print(
                "Stereo envelope match: "
                f"windows={result.windows} "
                f"median_delta={result.median_delta_db:+.2f}dB "
                f"gain={result.min_gain_db:+.2f}..{result.max_gain_db:+.2f}dB "
                f"mean_abs={result.mean_abs_gain_db:.2f}dB "
                f"strength={result.strength:.2f}",
                flush=True,
            )
            source_path = matched_path
            de_limiter_note = ""
            if de_limiter_checkpoint is not None:
                apply_de_limiter_wav(
                    matched_path,
                    delimited_path,
                    de_limiter_checkpoint,
                    condition_name="mix",
                    mix=de_limiter_mix,
                    chunk_seconds=de_limiter_chunk_seconds,
                    overlap_seconds=de_limiter_overlap_seconds,
                    device=de_limiter_device,
                    peak_limit=de_limiter_peak_limit,
                )
                source_path = delimited_path
                de_limiter_note = " and SGI de-limiter"
            render_stereo_source_flac(
                ffmpeg,
                source_path,
                output_path,
                tags,
                title,
                compression_level,
                normalize,
                target_i,
                target_tp,
                target_lra,
                (
                    "Stereo fold-down from upmix bed with temporal mastering safety "
                    f"and envelope match to original{de_limiter_note}"
                    if automation_lanes is not None
                    else f"Stereo fold-down from upmix bed with envelope match to original{de_limiter_note}"
                ),
            )
        finally:
            fold_path.unlink(missing_ok=True)
            matched_path.unlink(missing_ok=True)
            delimited_path.unlink(missing_ok=True)
            print_elapsed("stereo FLAC render", render_timer)
        return

    if de_limiter_checkpoint is not None:
        fold_path = output_path.with_suffix(".fold.tmp.wav")
        delimited_path = output_path.with_suffix(".delimit.tmp.wav")
        try:
            render_stereo_fold_wav(
                ffmpeg, bed_path, fold_path, lfe_fold_gain, automation_lanes
            )
            apply_de_limiter_wav(
                fold_path,
                delimited_path,
                de_limiter_checkpoint,
                condition_name="mix",
                mix=de_limiter_mix,
                chunk_seconds=de_limiter_chunk_seconds,
                overlap_seconds=de_limiter_overlap_seconds,
                device=de_limiter_device,
                peak_limit=de_limiter_peak_limit,
            )
            comment = "Stereo fold-down from upmix bed with SGI de-limiter"
            if automation_lanes is not None:
                comment += "; temporal mastering safety for 2ch fold-down"
            render_stereo_source_flac(
                ffmpeg,
                delimited_path,
                output_path,
                tags,
                title,
                compression_level,
                normalize,
                target_i,
                target_tp,
                target_lra,
                comment,
            )
        finally:
            fold_path.unlink(missing_ok=True)
            delimited_path.unlink(missing_ok=True)
            print_elapsed("stereo FLAC render", render_timer)
        return

    if normalize == "loudnorm":
        first_pass_filter = ";".join(
            [
                stereo_fold_temporal_filter(
                    "[0:a]",
                    "[fold]",
                    lfe_fold_gain,
                    automation_lanes,
                    prefix="stereo_loudnorm_measure",
                    limiter=False,
                ),
                loudnorm_filter(
                    "[fold]",
                    "[norm]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=None,
                    print_format="json",
                ),
            ]
        )
        temp_paths: list[Path] = []
        first_pass_cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(bed_path)]
        append_filter_complex_arg(
            first_pass_cmd,
            first_pass_filter,
            temp_paths,
            label="stereo_loudnorm_measure",
        )
        first_pass_cmd.extend(["-map", "[norm]", "-f", "null", "-"])
        try:
            first_pass = run(first_pass_cmd, capture=True)
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)
        measurements = parse_loudnorm_json(first_pass.stderr or "")
        filter_complex = ";".join(
            [
                stereo_fold_temporal_filter(
                    "[0:a]",
                    "[fold]",
                    lfe_fold_gain,
                    automation_lanes,
                    prefix="stereo_loudnorm_render",
                    limiter=False,
                ),
                loudnorm_filter(
                    "[fold]",
                    "[a]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=measurements,
                    print_format="summary",
                ),
            ]
        )
        comment = f"Stereo fold-down normalized to {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
    elif normalize == "linear":
        first_pass_filter = ";".join(
            [
                stereo_fold_temporal_filter(
                    "[0:a]",
                    "[fold]",
                    lfe_fold_gain,
                    automation_lanes,
                    prefix="stereo_linear_measure",
                    limiter=False,
                ),
                loudnorm_filter(
                    "[fold]",
                    "[norm]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=None,
                    print_format="json",
                ),
            ]
        )
        temp_paths: list[Path] = []
        first_pass_cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(bed_path)]
        append_filter_complex_arg(
            first_pass_cmd, first_pass_filter, temp_paths, label="stereo_linear_measure"
        )
        first_pass_cmd.extend(["-map", "[norm]", "-f", "null", "-"])
        try:
            first_pass = run(first_pass_cmd, capture=True)
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)
        measurements = parse_loudnorm_json(first_pass.stderr or "")
        gain_db = stereo_linear_gain_db(measurements, target_i, target_tp)
        print(
            f"Stereo linear normalization: gain={gain_db:+.2f} dB "
            f"(input_i={float(measurements['input_i']):+.2f} LUFS, input_tp={float(measurements['input_tp']):+.2f} dBTP)",
            flush=True,
        )
        filter_complex = ";".join(
            [
                stereo_fold_temporal_filter(
                    "[0:a]",
                    "[fold]",
                    lfe_fold_gain,
                    automation_lanes,
                    prefix="stereo_linear_render",
                    limiter=False,
                ),
                f"[fold]volume={fmt(db_to_amp(gain_db))},aresample=48000,aformat=channel_layouts=stereo,alimiter=limit=0.98[a]",
            ]
        )
        comment = f"Stereo fold-down linear peak-safe gain toward {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
    else:
        filter_complex = stereo_fold_temporal_filter(
            "[0:a]",
            "[a]",
            lfe_fold_gain,
            automation_lanes,
            prefix="stereo_direct",
            limiter=True,
        )
        comment = "Stereo fold-down from upmix bed"

    if automation_lanes is not None:
        comment += "; temporal mastering safety for 2ch fold-down"

    temp_paths: list[Path] = []
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(bed_path),
    ]
    append_filter_complex_arg(cmd, filter_complex, temp_paths, label="stereo_flac")
    cmd.extend(
        [
            "-map",
            "[a]",
            "-c:a",
            "flac",
            "-compression_level",
            str(compression_level),
            "-metadata",
            f"title={title}",
            "-metadata",
            f"comment={comment}",
            str(output_path),
        ]
    )
    for source_key, flac_key in (
        ("artist", "artist"),
        ("album", "album"),
        ("date", "date"),
    ):
        value = tags.get(source_key)
        if value:
            cmd[-1:-1] = ["-metadata", f"{flac_key}={value}"]
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("stereo FLAC render", render_timer)


def render_apple_tv(
    ffmpeg: str,
    bed_path: Path,
    input_file: Path | None,
    output_path: Path,
    duration: float,
    bitrate: str,
    tags: dict[str, str],
    title: str,
    lfe_fold_gain: float,
    normalize: str,
    target_i: float,
    target_tp: float,
    target_lra: float,
    loudnorm_measurements: dict[str, str] | None = None,
) -> None:
    render_timer = time.perf_counter()
    measurements: dict[str, str] | None = None
    if normalize == "loudnorm":
        measurements = loudnorm_measurements or measure_51_loudnorm(
            ffmpeg,
            bed_path,
            lfe_fold_gain,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
            label="apple_tv_loudnorm_measure",
        )

    cover_path = extract_cover(
        ffmpeg, input_file, output_path.with_suffix(".cover.jpg")
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if cover_path is not None:
        cmd.extend(
            ["-f", "image2", "-loop", "1", "-framerate", "1", "-i", str(cover_path)]
        )
        video_filter = (
            "[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v]"
        )
    else:
        cmd.extend(["-f", "lavfi", "-i", "color=c=black:s=1920x1080:r=1"])
        video_filter = "[0:v]format=yuv420p[v]"
    cmd.extend(["-i", str(bed_path)])
    if normalize == "loudnorm":
        audio_filter = ";".join(
            [
                fold_714_to_51_filter("[1:a]", "[fold]", lfe_fold_gain, limiter=False),
                loudnorm_filter(
                    "[fold]",
                    "[a]",
                    target_i=target_i,
                    target_tp=target_tp,
                    target_lra=target_lra,
                    measurements=measurements,
                    print_format="summary",
                    channel_layout="5.1(side)",
                ),
            ]
        )
        audio_title = f"Dolby Digital Plus 5.1 normalized {fmt(target_i)} LUFS / {fmt(target_tp)} dBTP"
    else:
        audio_filter = fold_714_to_51_filter("[1:a]", "[a]", lfe_fold_gain)
        audio_title = "Dolby Digital Plus 5.1 Auto placement"
    temp_paths: list[Path] = []
    append_filter_complex_arg(
        cmd, f"{video_filter};{audio_filter}", temp_paths, label="apple_tv"
    )
    cmd.extend(
        [
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-profile:v",
            "high",
            "-level",
            "4.1",
            "-preset",
            "medium",
            "-tune",
            "stillimage",
            "-crf",
            "20",
            "-r",
            "24",
            "-c:a",
            "eac3",
            "-b:a",
            bitrate,
            "-tag:a",
            "ec-3",
            "-metadata:s:a:0",
            f"title={audio_title}",
            "-metadata",
            f"title={title}",
        ]
    )
    for source_key, mp4_key in (
        ("artist", "artist"),
        ("album", "album"),
        ("date", "date"),
    ):
        value = tags.get(source_key)
        if value:
            cmd.extend(["-metadata", f"{mp4_key}={value}"])
    cmd.extend(["-movflags", "+faststart", "-brand", "mp42", str(output_path)])
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("Apple TV render", render_timer)


def render_51_fold_wav(
    ffmpeg: str,
    bed_path: Path,
    output_path: Path,
    lfe_fold_gain: float,
) -> None:
    if _render_native_fold_wav(
        bed_path,
        output_path,
        min_channels=len(CHANNELS_714),
        fold=_fold_714_to_51_native,
        lfe_fold_gain=lfe_fold_gain,
        label="5.1 fold WAV",
    ):
        return

    temp_paths: list[Path] = []
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(bed_path)]
    append_filter_complex_arg(
        cmd,
        fold_714_to_51_filter("[0:a]", "[a]", lfe_fold_gain, limiter=False),
        temp_paths,
        label="51_fold_wav",
    )
    cmd.extend(["-map", "[a]", "-c:a", "pcm_s24le", str(output_path)])
    timer = time.perf_counter()
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("5.1 fold WAV render", timer)


def render_51_to_stereo_fold_wav(
    ffmpeg: str,
    input_path: Path,
    output_path: Path,
    lfe_fold_gain: float,
) -> None:
    if _render_native_fold_wav(
        input_path,
        output_path,
        min_channels=6,
        fold=_fold_51_to_stereo_native,
        lfe_fold_gain=lfe_fold_gain,
        label="5.1 to stereo fold WAV",
    ):
        return

    lfe_left = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    lfe_right = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    filter_complex = (
        "[0:a]pan=stereo|"
        f"FL=0.92*FL+0.7071*FC+0.55*SL{lfe_left}|"
        f"FR=0.92*FR+0.7071*FC+0.55*SR{lfe_right},"
        "aresample=48000,aformat=channel_layouts=stereo[a]"
    )
    temp_paths: list[Path] = []
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_path)]
    append_filter_complex_arg(
        cmd, filter_complex, temp_paths, label="51_to_stereo_fold_wav"
    )
    cmd.extend(["-map", "[a]", "-c:a", "pcm_s24le", str(output_path)])
    timer = time.perf_counter()
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("5.1 to stereo fold WAV render", timer)


__all__ = [
    "extract_cover",
    "analyze_rendered_fold_numpy",
    "analyze_rendered_fold_ffmpeg",
    "analyze_rendered_fold",
    "multiband_stereo_audio_numpy",
    "multiband_rms_db",
    "multiband_filter_numpy",
    "analyze_multiband_numpy",
    "print_multiband_fold_analysis",
    "filename_number_token",
    "stereo_loudness_suffix",
    "surround_loudness_suffix",
    "parse_loudnorm_json",
    "parse_loudnorm_output_loudness",
    "loudnorm_filter",
    "source_stereo_filter",
    "apply_de_limiter_wav",
    "apply_stem_de_limiter_to_stems",
    "prepare_stem_de_limiter_dir",
    "prune_stem_de_limiter_candidate_files",
    "measure_stereo_loudnorm",
    "measure_51_loudnorm",
    "measure_51_loudnorm_native",
    "native_loudnorm_binary",
    "stereo_linear_gain_db",
    "render_stereo_fold_wav",
    "_read_stereo_float",
    "_window_rms_db",
    "_smooth_gain_curve",
    "apply_stereo_envelope_match",
    "render_stereo_source_flac",
    "render_51_flac",
    "render_51_wav_native",
    "render_stereo_flac",
    "render_apple_tv",
    "render_51_fold_wav",
    "render_51_to_stereo_fold_wav",
]
