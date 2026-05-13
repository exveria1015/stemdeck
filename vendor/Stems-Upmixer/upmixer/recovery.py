"""Brickwall recovery planning and transient restoration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


from .models import (
    BrickwallRecoveryItem,
    BrickwallRecoveryResult,
    StemStats,
    TemporalRemasterResult,
    WindowStats,
)
from .temporal import temporal_diagnostic_summary, weighted_mean
from .utils import clamp, db_to_amp, safe_label


def brickwall_summary_value(
    temporal_result: TemporalRemasterResult | None,
    key: str,
    stat: str,
    default: float,
) -> float:
    if temporal_result is None or not temporal_result.intents:
        return default
    summary = temporal_diagnostic_summary(temporal_result.intents)
    row = summary.get(key) or {}
    try:
        return float(row.get(stat, default))
    except (TypeError, ValueError):
        return default


def brickwall_recovery_pressure(
    stats: list[StemStats],
    temporal_result: TemporalRemasterResult | None,
) -> float:
    peak_pressure = brickwall_summary_value(
        temporal_result, "peak_pressure", "p95", 0.0
    )
    crest = brickwall_summary_value(
        temporal_result, "crest_factor_db", "loudness_weighted_mean", 10.0
    )
    short_loudness = brickwall_summary_value(
        temporal_result, "short_lufs_estimate", "active_mean", -18.0
    )
    low_crest = clamp((9.5 - crest) / 4.5, 0.0, 1.0)
    loud = clamp((short_loudness + 14.0) / 6.0, 0.0, 1.0)
    if not temporal_result or not temporal_result.intents:
        max_peak = max((item.full.max_db for item in stats), default=-120.0)
        mean_rms = (
            weighted_mean(
                [item.full.mean_db for item in stats],
                [db_to_amp(item.full.mean_db) for item in stats],
            )
            if stats
            else -120.0
        )
        peak_pressure = clamp((max_peak + 3.0) / 5.0, 0.0, 1.0)
        loud = clamp((mean_rms + 18.0) / 10.0, 0.0, 1.0)
    return clamp(0.58 * peak_pressure + 0.26 * low_crest + 0.16 * loud, 0.0, 1.0)


def brickwall_mode_scale(mode: str, pressure: float) -> tuple[float, str]:
    if mode == "off":
        return 0.0, "disabled"
    if mode == "auto" and pressure < 0.34:
        return 0.0, f"brickwall pressure {pressure:.2f} below auto threshold"
    scale = {
        "auto": 0.70,
        "conservative": 0.55,
        "natural": 0.78,
        "open": 1.0,
    }.get(mode, 0.0)
    return scale, f"brickwall pressure {pressure:.2f}"


def stem_recovery_base(stem: str) -> tuple[float, float, float, float, float]:
    if stem == "drums":
        return 0.85, 0.24, 45.0, 9000.0, 0.95
    if stem == "bass":
        return 0.34, 0.08, 70.0, 850.0, 0.70
    if stem == "vocals":
        return 0.22, 0.05, 120.0, 5200.0, 0.72
    if stem == "piano":
        return 0.42, 0.09, 120.0, 9000.0, 0.82
    if stem == "guitar":
        return 0.38, 0.08, 120.0, 8200.0, 0.80
    return 0.20, 0.05, 180.0, 7000.0, 0.62


def stem_recovery_activity(window: WindowStats | None) -> float:
    if window is None:
        return 0.55
    active = clamp((float(window.active_ratio) - 0.01) / 0.34, 0.0, 1.0)
    sustain = clamp(float(window.sustain_score), 0.0, 1.0)
    return clamp(active * (0.64 + 0.36 * sustain), 0.0, 1.0)


def brickwall_hf_guard(
    temporal_result: TemporalRemasterResult | None,
) -> tuple[float, float, float]:
    presence = brickwall_summary_value(
        temporal_result, "presence_harshness_score", "p95", 0.0
    )
    sibilance = brickwall_summary_value(temporal_result, "sibilance_score", "p95", 0.0)
    phase = brickwall_summary_value(temporal_result, "phase_risk", "p95", 0.0)
    return clamp(presence, 0.0, 1.0), clamp(sibilance, 0.0, 1.0), clamp(phase, 0.0, 1.0)


def butter_band_audio(
    audio: AudioArray, sample_rate: int, low_hz: float, high_hz: float
) -> AudioArray:
    from scipy import signal

    nyquist = float(sample_rate) * 0.5
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.92)
    if high <= low * 1.05:
        return audio.copy()
    if low <= 5.0:
        sos = signal.butter(3, high, btype="lowpass", fs=sample_rate, output="sos")
    elif high >= nyquist * 0.90:
        sos = signal.butter(3, low, btype="highpass", fs=sample_rate, output="sos")
    else:
        sos = signal.butter(
            3, (low, high), btype="bandpass", fs=sample_rate, output="sos"
        )
    try:
        return cast(AudioArray, signal.sosfiltfilt(sos, audio, axis=0))
    except ValueError:
        return cast(AudioArray, signal.sosfilt(sos, audio, axis=0))


def transient_mask(
    audio: AudioArray,
    sample_rate: int,
    *,
    fast_ms: float,
    slow_ms: float,
    release_ms: float,
) -> AudioArray:
    import numpy as np
    from scipy import ndimage

    power = np.mean(audio * audio, axis=1) if audio.ndim == 2 else audio * audio
    fast = max(1, int(sample_rate * fast_ms * 0.001))
    slow = max(fast + 1, int(sample_rate * slow_ms * 0.001))
    release = max(1, int(sample_rate * release_ms * 0.001))
    fast_env = np.sqrt(
        ndimage.uniform_filter1d(power, size=fast, mode="nearest") + 1e-12
    )
    slow_env = np.sqrt(
        ndimage.uniform_filter1d(power, size=slow, mode="nearest") + 1e-12
    )
    mask = np.clip((fast_env / np.maximum(slow_env, 1e-8) - 1.0) / 1.35, 0.0, 1.0)
    return cast(
        AudioArray,
        ndimage.uniform_filter1d(mask, size=release, mode="nearest").astype(np.float64),
    )


def side_scaled_delta(delta: AudioArray, side_scale: float) -> AudioArray:
    import numpy as np

    if delta.ndim != 2 or delta.shape[1] < 2:
        return delta
    mid = 0.5 * (delta[:, 0] + delta[:, 1])
    side = 0.5 * (delta[:, 0] - delta[:, 1]) * clamp(float(side_scale), 0.0, 1.0)
    return cast(AudioArray, np.stack([mid + side, mid - side], axis=1))


def apply_stem_transient_recovery(
    stem: str,
    input_path: Path,
    output_path: Path,
    *,
    attack_db: float,
    sustain_cut_db: float,
    low_hz: float,
    high_hz: float,
    side_scale: float,
) -> tuple[float, float]:
    import numpy as np
    import soundfile as sf

    audio_raw, sample_rate = sf.read(input_path, always_2d=True, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0
    band = butter_band_audio(audio, int(sample_rate), low_hz, high_hz)
    mask = transient_mask(
        band,
        int(sample_rate),
        fast_ms=4.0 if stem in {"drums", "piano"} else 6.0,
        slow_ms=72.0 if stem in {"drums", "bass"} else 95.0,
        release_ms=18.0 if stem == "drums" else 30.0,
    )[:, None]
    attack_gain = db_to_amp(max(0.0, attack_db)) - 1.0
    sustain_cut = 1.0 - db_to_amp(-max(0.0, sustain_cut_db))
    candidate = audio * (1.0 - sustain_cut * (1.0 - mask)) + band * attack_gain * mask
    delta = side_scaled_delta(candidate - audio, side_scale)
    recovered = audio + delta
    allowed_peak = max(peak_before * db_to_amp(0.45), peak_before + 1e-6)
    peak_after = float(np.max(np.abs(recovered))) if recovered.size else 0.0
    if peak_after > allowed_peak:
        delta *= allowed_peak / max(peak_after, 1e-8)
        recovered = audio + delta
        peak_after = float(np.max(np.abs(recovered))) if recovered.size else 0.0
    sf.write(
        output_path,
        recovered.astype(np.float32, copy=False),
        int(sample_rate),
        subtype="PCM_24",
    )
    return peak_before, peak_after


def apply_brickwall_recovery(
    stems: dict[str, Path],
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    temporal_result: TemporalRemasterResult | None,
    *,
    mode: str,
    strength: float,
    output_dir: Path,
) -> BrickwallRecoveryResult:
    pressure = brickwall_recovery_pressure(stats, temporal_result)
    mode_scale, reason = brickwall_mode_scale(mode, pressure)
    if mode_scale <= 0.0:
        return BrickwallRecoveryResult(
            False, mode, pressure, dict(stems), reason=reason
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    presence_guard, sibilance_guard, phase_guard = brickwall_hf_guard(temporal_result)
    stat_by_name = {item.name: item for item in stats}
    recovered_stems = dict(stems)
    items: list[BrickwallRecoveryItem] = []
    for stem, path in stems.items():
        if stem not in stat_by_name:
            continue
        base_attack, base_sustain, low_hz, high_hz, base_side_scale = (
            stem_recovery_base(stem)
        )
        activity = stem_recovery_activity(windows.get(stem))
        stem_strength = clamp(
            mode_scale * clamp(float(strength), 0.0, 1.2) * pressure * activity,
            0.0,
            1.0,
        )
        guards: list[str] = []
        if stem == "vocals" and sibilance_guard >= 0.25:
            reduction = 1.0 - 0.55 * sibilance_guard
            stem_strength *= reduction
            high_hz = min(high_hz, 4200.0)
            guards.append("sibilance")
        if stem in {"guitar", "piano", "other", "drums"} and presence_guard >= 0.28:
            stem_strength *= 1.0 - 0.38 * presence_guard
            high_hz = min(high_hz, 6500.0 if stem != "drums" else 7800.0)
            guards.append("presence")
        side_scale = base_side_scale * (1.0 - 0.55 * phase_guard)
        if phase_guard >= 0.28:
            guards.append("fold_phase")
        attack_db = base_attack * stem_strength
        sustain_cut_db = base_sustain * stem_strength
        if attack_db < 0.03 and sustain_cut_db < 0.01:
            items.append(
                BrickwallRecoveryItem(
                    stem,
                    False,
                    attack_db,
                    sustain_cut_db,
                    side_scale,
                    0.0,
                    0.0,
                    tuple(guards),
                )
            )
            continue
        output_path = output_dir / f"{safe_label(stem)}_brickwall_recovery.wav"
        peak_before, peak_after = apply_stem_transient_recovery(
            stem,
            path,
            output_path,
            attack_db=attack_db,
            sustain_cut_db=sustain_cut_db,
            low_hz=low_hz,
            high_hz=high_hz,
            side_scale=side_scale,
        )
        recovered_stems[stem] = output_path
        items.append(
            BrickwallRecoveryItem(
                stem=stem,
                enabled=True,
                attack_db=attack_db,
                sustain_cut_db=sustain_cut_db,
                side_scale=side_scale,
                peak_before=peak_before,
                peak_after=peak_after,
                guards=tuple(guards),
            )
        )
    enabled = any(item.enabled for item in items)
    return BrickwallRecoveryResult(
        enabled,
        mode,
        pressure,
        recovered_stems if enabled else dict(stems),
        items,
        reason,
    )


def print_brickwall_recovery_result(result: BrickwallRecoveryResult) -> None:
    if not result.enabled:
        if result.mode != "off":
            print(f"Brickwall recovery skipped: {result.reason}", flush=True)
        return
    print(
        f"Brickwall recovery: mode={result.mode} pressure={result.pressure:.2f} ({result.reason})",
        flush=True,
    )
    for item in result.items:
        if not item.enabled:
            continue
        guard_text = f" guards={','.join(item.guards)}" if item.guards else ""
        print(
            f"  {item.stem:<9} attack=+{item.attack_db:.2f}dB "
            f"sustain=-{item.sustain_cut_db:.2f}dB "
            f"side={item.side_scale:.2f} "
            f"peak={item.peak_before:.3f}->{item.peak_after:.3f}{guard_text}",
            flush=True,
        )


__all__ = [
    "brickwall_summary_value",
    "brickwall_recovery_pressure",
    "brickwall_mode_scale",
    "stem_recovery_base",
    "stem_recovery_activity",
    "brickwall_hf_guard",
    "butter_band_audio",
    "transient_mask",
    "side_scaled_delta",
    "apply_stem_transient_recovery",
    "apply_brickwall_recovery",
    "print_brickwall_recovery_result",
]
