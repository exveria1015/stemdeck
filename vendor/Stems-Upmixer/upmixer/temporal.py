"""Temporal mastering diagnostics, automation lanes, and report data."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

from .models import (
    MULTIBAND_RANGES,
    AutomationLanes,
    MasteringIntentDecision,
    MasteringPrincipleScore,
    MixMultibandProfile,
    StemStats,
    TemporalDiagnosticFrame,
    TemporalFrame,
    TemporalRemasterResult,
    WindowStats,
)
from .placement import stem_energy_shares, temporal_duck_fragility_score
from .stems import (
    band_fractions,
    bright_score,
    stem_range_energies_numpy,
    vocal_leak_risk,
    wide_score,
)
from .utils import clamp, linear_energy, median_float, read_audio_float

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


def percentile_float(values: Iterable[float], percentile: float) -> float:
    sorted_values = sorted(float(value) for value in values)
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = clamp(float(percentile), 0.0, 100.0) * 0.01 * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def window_relative_level(window: WindowStats, index: int) -> float:
    if not window.envelope_db:
        return 0.0
    value = window.envelope_db[min(index, len(window.envelope_db) - 1)]
    floor = min(window.p50_db - 9.0, window.p10_db + 3.0)
    ceiling = max(window.p90_db, window.p50_db + 6.0, floor + 6.0)
    return clamp((value - floor) / max(ceiling - floor, 1e-6), 0.0, 1.0)


def smooth_automation_lane(
    values: Iterable[float], *, hop_sec: float, attack_sec: float, release_sec: float
) -> tuple[float, ...]:
    raw = [clamp(float(value), 0.0, 1.0) for value in values]
    if not raw:
        return ()
    attack_steps = max(1.0, float(attack_sec) / max(hop_sec, 1e-6))
    release_steps = max(1.0, float(release_sec) / max(hop_sec, 1e-6))
    attack_alpha = 1.0 - math.exp(-1.0 / attack_steps)
    release_alpha = 1.0 - math.exp(-1.0 / release_steps)
    smoothed = [raw[0]]
    current = raw[0]
    for target in raw[1:]:
        alpha = attack_alpha if target > current else release_alpha
        current += (target - current) * alpha
        smoothed.append(clamp(current, 0.0, 1.0))
    return tuple(smoothed)


def neutral_temporal_diagnostics(count: int) -> tuple[TemporalDiagnosticFrame, ...]:
    return tuple(TemporalDiagnosticFrame() for _index in range(max(0, count)))


def build_mix_multiband_profile(stats: list[StemStats]) -> MixMultibandProfile:
    energies = {band: 0.0 for band, _low, _high in MULTIBAND_RANGES}
    for item in stats:
        try:
            item_energies = stem_range_energies_numpy(item, MULTIBAND_RANGES)
        except Exception:
            low, mid, high = band_fractions(item)
            full = linear_energy(item.full.mean_db)
            item_energies = {
                "sub": full * low * 0.30,
                "bass": full * low * 0.42,
                "lowmid": full * low * 0.28,
                "body": full * mid * 0.32,
                "mid": full * mid * 0.38,
                "presence": full * mid * 0.30,
                "air": full * high * 0.66,
                "top": full * high * 0.34,
            }
        for band in energies:
            energies[band] += max(0.0, item_energies.get(band, 0.0))

    total = max(sum(energies.values()), 1e-24)
    fractions = {band: energy / total for band, energy in energies.items()}
    low_share = fractions.get("sub", 0.0) + fractions.get("bass", 0.0)
    body_share = (
        fractions.get("lowmid", 0.0)
        + fractions.get("body", 0.0)
        + fractions.get("mid", 0.0)
    )
    presence_share = fractions.get("presence", 0.0) + 0.65 * fractions.get("air", 0.0)
    air_share = fractions.get("air", 0.0) + fractions.get("top", 0.0)
    return MixMultibandProfile(
        fractions=fractions,
        low_pressure=clamp((low_share - 0.13) / 0.24, 0.0, 1.0),
        body_congestion=clamp((body_share - 0.42) / 0.34, 0.0, 1.0),
        presence_pressure=clamp((presence_share - 0.15) / 0.26, 0.0, 1.0),
        air_opportunity=clamp((air_share - 0.055) / 0.18, 0.0, 1.0),
    )


def print_mix_multiband_profile(profile: MixMultibandProfile) -> None:
    if not profile.fractions:
        return
    strongest = sorted(
        profile.fractions.items(), key=lambda item: item[1], reverse=True
    )[:4]
    text = " ".join(f"{band}={value * 100.0:.1f}%" for band, value in strongest)
    print(
        "Mix 8-band profile: "
        f"low={profile.low_pressure:.2f} "
        f"body={profile.body_congestion:.2f} "
        f"presence={profile.presence_pressure:.2f} "
        f"air={profile.air_opportunity:.2f} "
        f"bands={text}",
        flush=True,
    )


def read_stem_mix_audio(stats: list[StemStats]) -> tuple[AudioArray, int]:
    import numpy as np
    from scipy import signal

    mix: AudioArray | None = None
    mix_rate: int | None = None
    for item in stats:
        audio_raw, sample_rate = read_audio_float(item.path, dtype="float32")
        audio = cast(AudioArray, audio_raw)
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        elif audio.shape[1] > 2:
            audio = audio[:, :2]
        if mix_rate is None:
            mix_rate = int(sample_rate)
        elif int(sample_rate) != mix_rate:
            divisor = math.gcd(mix_rate, int(sample_rate))
            resampled = signal.resample_poly(
                audio,
                mix_rate // divisor,
                int(sample_rate) // divisor,
                axis=0,
            )
            audio = cast(AudioArray, resampled.astype(np.float32, copy=False))
        if mix is None:
            mix = cast(AudioArray, np.zeros((len(audio), 2), dtype=np.float32))
        elif len(audio) > len(mix):
            extended = cast(AudioArray, np.zeros((len(audio), 2), dtype=np.float32))
            extended[: len(mix)] = mix
            mix = extended
        mix[: len(audio)] += audio[:, :2]
    if mix is None or mix_rate is None:
        raise RuntimeError("no stems available for temporal diagnostics")
    return mix, mix_rate


def temporal_window(
    data: AudioArray, sample_rate: int, center_sec: float, window_sec: float
) -> AudioArray:
    center = int(round(float(center_sec) * sample_rate))
    half = max(1, int(round(max(float(window_sec), 0.001) * sample_rate * 0.5)))
    start = max(0, center - half)
    end = min(len(data), center + half)
    return data[start:end]


def window_bounds(
    length: int, sample_rate: int, centers: tuple[float, ...], window_sec: float
):
    import numpy as np

    if not centers:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
        )
    center_indices = np.rint(np.asarray(centers, dtype=np.float64) * sample_rate).astype(
        np.int64
    )
    half = max(1, int(round(max(float(window_sec), 0.001) * sample_rate * 0.5)))
    max_index = int(length)
    starts = np.clip(center_indices - half, 0, max_index)
    ends = np.clip(center_indices + half, 0, max_index)
    return starts, ends


def audio_rms_db(data: AudioArray) -> float:
    import numpy as np

    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    return 20.0 * math.log10(max(rms, 1e-12))


def audio_peak_db(
    data: AudioArray, sample_rate: int | None = None, *, oversample: int = 1
) -> float:
    import numpy as np

    if not data.size:
        return -120.0
    if sample_rate is not None and oversample > 1:
        try:
            from scipy import signal

            data = cast(AudioArray, signal.resample_poly(data, oversample, 1, axis=0))
        except Exception:
            pass
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return 20.0 * math.log10(max(peak, 1e-12))


def loudness_lufs_estimate(data: AudioArray) -> float:
    # This is a short-window RMS/K-weight proxy, not a formal integrated EBU R128 pass.
    return audio_rms_db(data) - 0.691


def band_filter_audio(
    data: AudioArray, sample_rate: int, low_hz: float, high_hz: float
) -> AudioArray:
    from scipy import signal

    nyquist = sample_rate * 0.5
    low = max(1.0, float(low_hz))
    high = min(float(high_hz), nyquist * 0.96)
    if low >= nyquist * 0.96:
        return cast(AudioArray, data * 0.0)
    if high <= low:
        high = min(nyquist * 0.96, low * 1.2)
    if high <= low:
        return cast(AudioArray, data * 0.0)
    sos = signal.butter(4, (low, high), btype="bandpass", fs=sample_rate, output="sos")
    return cast(AudioArray, signal.sosfilt(sos, data, axis=0))


def window_loudness_values(
    data: AudioArray, sample_rate: int, centers: tuple[float, ...], window_sec: float
) -> list[float]:
    return [value - 0.691 for value in window_rms_db_values(data, sample_rate, centers, window_sec)]


def window_rms_db_values(
    data: AudioArray, sample_rate: int, centers: tuple[float, ...], window_sec: float
) -> list[float]:
    import numpy as np

    starts, ends = window_bounds(len(data), sample_rate, centers, window_sec)
    if len(starts) == 0:
        return []
    if data.size == 0:
        return [-240.0 for _center in centers]
    data64 = np.asarray(data, dtype=np.float64)
    if data64.ndim == 1:
        sample_power = data64 * data64
    else:
        sample_power = np.mean(data64 * data64, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power, dtype=np.float64)))
    counts = np.maximum(ends - starts, 1)
    mean_power = (cumulative[ends] - cumulative[starts]) / counts
    rms = np.sqrt(np.maximum(mean_power, 0.0))
    return [float(value) for value in 20.0 * np.log10(np.maximum(rms, 1e-12))]


def window_peak_db_values(
    data: AudioArray,
    sample_rate: int,
    centers: tuple[float, ...],
    window_sec: float,
    *,
    oversample: int = 1,
) -> list[float]:
    if oversample <= 1:
        import numpy as np

        starts, ends = window_bounds(len(data), sample_rate, centers, window_sec)
        if len(starts) == 0:
            return []
        data64 = np.asarray(data, dtype=np.float64)
        if data64.size == 0:
            return [-120.0 for _center in centers]
        if data64.ndim == 1:
            sample_peak = np.abs(data64)
        else:
            sample_peak = np.max(np.abs(data64), axis=1)
        return [
            float(
                20.0
                * math.log10(
                    max(float(np.max(sample_peak[start:end])), 1e-12)
                )
            )
            if end > start
            else -120.0
            for start, end in zip(starts, ends)
        ]
    return [
        audio_peak_db(
            temporal_window(data, sample_rate, center, window_sec),
            sample_rate,
            oversample=oversample,
        )
        for center in centers
    ]


def stereo_correlation(data: AudioArray) -> float:
    import numpy as np

    if data.size == 0:
        return 1.0
    if len(data.shape) == 1 or data.shape[1] < 2:
        return 1.0
    left = data[:, 0] - float(np.mean(data[:, 0]))
    right = data[:, 1] - float(np.mean(data[:, 1]))
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-12:
        return 1.0
    return clamp(float(np.dot(left, right) / denom), -1.0, 1.0)


def mono_fold_loss_db(data: AudioArray) -> float:
    import numpy as np

    if data.size == 0:
        return 0.0
    if len(data.shape) == 1 or data.shape[1] < 2:
        return 0.0
    stereo_rms = float(
        np.sqrt(np.mean((data[:, 0] * data[:, 0] + data[:, 1] * data[:, 1]) * 0.5))
    )
    mono = 0.5 * (data[:, 0] + data[:, 1])
    mono_rms = float(np.sqrt(np.mean(mono * mono)))
    return 20.0 * math.log10(max(mono_rms, 1e-12) / max(stereo_rms, 1e-12))


def window_correlation_values(
    data: AudioArray, sample_rate: int, centers: tuple[float, ...], window_sec: float
) -> list[float]:
    import numpy as np

    starts, ends = window_bounds(len(data), sample_rate, centers, window_sec)
    if len(starts) == 0:
        return []
    data64 = np.asarray(data, dtype=np.float64)
    if data64.size == 0 or data64.ndim == 1 or data64.shape[1] < 2:
        return [1.0 for _center in centers]
    left = data64[:, 0]
    right = data64[:, 1]
    ones = np.ones(len(left), dtype=np.float64)
    cumul_n = np.concatenate(([0.0], np.cumsum(ones)))
    cumul_l = np.concatenate(([0.0], np.cumsum(left)))
    cumul_r = np.concatenate(([0.0], np.cumsum(right)))
    cumul_l2 = np.concatenate(([0.0], np.cumsum(left * left)))
    cumul_r2 = np.concatenate(([0.0], np.cumsum(right * right)))
    cumul_lr = np.concatenate(([0.0], np.cumsum(left * right)))
    count = np.maximum(cumul_n[ends] - cumul_n[starts], 1.0)
    sum_l = cumul_l[ends] - cumul_l[starts]
    sum_r = cumul_r[ends] - cumul_r[starts]
    cov = (cumul_lr[ends] - cumul_lr[starts]) - (sum_l * sum_r / count)
    var_l = (cumul_l2[ends] - cumul_l2[starts]) - (sum_l * sum_l / count)
    var_r = (cumul_r2[ends] - cumul_r2[starts]) - (sum_r * sum_r / count)
    denom = np.sqrt(np.maximum(var_l, 0.0) * np.maximum(var_r, 0.0))
    values = np.divide(cov, denom, out=np.ones_like(cov), where=denom > 1e-12)
    return [clamp(float(value), -1.0, 1.0) for value in values]


def window_mono_loss_values(
    data: AudioArray, sample_rate: int, centers: tuple[float, ...], window_sec: float
) -> list[float]:
    import numpy as np

    starts, ends = window_bounds(len(data), sample_rate, centers, window_sec)
    if len(starts) == 0:
        return []
    data64 = np.asarray(data, dtype=np.float64)
    if data64.size == 0 or data64.ndim == 1 or data64.shape[1] < 2:
        return [0.0 for _center in centers]
    left = data64[:, 0]
    right = data64[:, 1]
    stereo_power = (left * left + right * right) * 0.5
    mono = 0.5 * (left + right)
    mono_power = mono * mono
    stereo_cum = np.concatenate(([0.0], np.cumsum(stereo_power, dtype=np.float64)))
    mono_cum = np.concatenate(([0.0], np.cumsum(mono_power, dtype=np.float64)))
    counts = np.maximum(ends - starts, 1)
    stereo_rms = np.sqrt(np.maximum((stereo_cum[ends] - stereo_cum[starts]) / counts, 0.0))
    mono_rms = np.sqrt(np.maximum((mono_cum[ends] - mono_cum[starts]) / counts, 0.0))
    ratios = np.maximum(mono_rms, 1e-12) / np.maximum(stereo_rms, 1e-12)
    return [float(value) for value in 20.0 * np.log10(ratios)]


def local_loudness_range(values: list[float], index: int, *, radius: int) -> float:
    start = max(0, index - radius)
    end = min(len(values), index + radius + 1)
    active = [value for value in values[start:end] if value > -70.0]
    if len(active) < 2:
        return 0.0
    return percentile_float(active, 95.0) - percentile_float(active, 10.0)


def compute_temporal_mix_diagnostics(
    stats: list[StemStats],
    *,
    frame_count: int,
    hop_sec: float,
    mix_audio: AudioArray | None = None,
    sample_rate: int | None = None,
) -> tuple[TemporalDiagnosticFrame, ...]:
    if frame_count <= 0:
        return ()
    try:
        if mix_audio is None or sample_rate is None:
            mix, sample_rate = read_stem_mix_audio(stats)
        else:
            mix = mix_audio
        centers = tuple((index + 0.5) * hop_sec for index in range(frame_count))
        momentary_lufs = window_loudness_values(mix, sample_rate, centers, 0.400)
        short_lufs = window_loudness_values(mix, sample_rate, centers, 3.000)
        true_peak = window_peak_db_values(
            mix, sample_rate, centers, 0.400, oversample=4
        )
        crest = [peak - loudness for peak, loudness in zip(true_peak, short_lufs)]
        lra_radius = max(2, int(round(15.0 / max(hop_sec, 0.05))))

        low_band = band_filter_audio(mix, sample_rate, 45.0, 160.0)
        low_rms = window_rms_db_values(low_band, sample_rate, centers, 0.800)
        low_peak = window_peak_db_values(
            low_band, sample_rate, centers, 0.400, oversample=1
        )
        low_punch_scores = [
            clamp(((peak - rms) - 7.0) / 12.0, 0.0, 1.0)
            for peak, rms in zip(low_peak, low_rms)
        ]
        del low_band

        body_band = band_filter_audio(mix, sample_rate, 300.0, 1500.0)
        body_db = window_rms_db_values(body_band, sample_rate, centers, 1.500)
        del body_band
        presence_band = band_filter_audio(mix, sample_rate, 2500.0, 5000.0)
        presence_db = window_rms_db_values(presence_band, sample_rate, centers, 1.500)
        del presence_band
        sibilance_band = band_filter_audio(mix, sample_rate, 5000.0, 8000.0)
        sibilance_db = window_rms_db_values(sibilance_band, sample_rate, centers, 1.500)
        del sibilance_band
        air_band = band_filter_audio(mix, sample_rate, 8000.0, 12000.0)
        air_db = window_rms_db_values(air_band, sample_rate, centers, 1.500)
        del air_band

        mono_corr = window_correlation_values(mix, sample_rate, centers, 1.500)
        mono_loss = window_mono_loss_values(mix, sample_rate, centers, 1.500)
        low_corr_band = band_filter_audio(mix, sample_rate, 20.0, 180.0)
        low_corr = window_correlation_values(low_corr_band, sample_rate, centers, 1.500)
        del low_corr_band
        mid_corr_band = band_filter_audio(mix, sample_rate, 180.0, 2500.0)
        mid_corr = window_correlation_values(mid_corr_band, sample_rate, centers, 1.500)
        del mid_corr_band
        high_corr_band = band_filter_audio(mix, sample_rate, 2500.0, 12000.0)
        high_corr = window_correlation_values(
            high_corr_band, sample_rate, centers, 1.500
        )
        del high_corr_band

        diagnostics: list[TemporalDiagnosticFrame] = []
        previous_momentary = momentary_lufs[0] if momentary_lufs else -120.0
        for index in range(frame_count):
            rise_db = (
                max(0.0, momentary_lufs[index] - previous_momentary)
                if index > 0
                else 0.0
            )
            transient_score = clamp(
                0.55 * ((crest[index] - 8.5) / 10.5) + 0.45 * (rise_db / 6.0), 0.0, 1.0
            )
            low_punch = low_punch_scores[index]
            punch_score = clamp(0.58 * transient_score + 0.42 * low_punch, 0.0, 1.0)
            presence_score = clamp(
                (presence_db[index] - body_db[index] + 5.5) / 12.0, 0.0, 1.0
            )
            sibilance_score = clamp(
                (sibilance_db[index] - body_db[index] + 4.0) / 12.0, 0.0, 1.0
            )
            air_score = clamp((air_db[index] - body_db[index] + 2.5) / 12.0, 0.0, 1.0)
            min_corr = min(
                mono_corr[index], low_corr[index], mid_corr[index], high_corr[index]
            )
            phase_risk = clamp(
                0.58 * ((-mono_loss[index] - 1.5) / 7.0)
                + 0.42 * ((0.28 - min_corr) / 0.78),
                0.0,
                1.0,
            )
            peak_pressure = clamp((true_peak[index] + 3.0) / 5.0, 0.0, 1.0)
            diagnostics.append(
                TemporalDiagnosticFrame(
                    momentary_lufs=momentary_lufs[index],
                    short_lufs=short_lufs[index],
                    loudness_range_db=local_loudness_range(
                        short_lufs, index, radius=lra_radius
                    ),
                    true_peak_dbtp=true_peak[index],
                    crest_factor_db=max(0.0, crest[index]),
                    transient_score=transient_score,
                    punch_score=punch_score,
                    low_punch_score=low_punch,
                    mono_correlation=mono_corr[index],
                    mono_fold_loss_db=mono_loss[index],
                    low_correlation=low_corr[index],
                    mid_correlation=mid_corr[index],
                    high_correlation=high_corr[index],
                    phase_risk=phase_risk,
                    peak_pressure=peak_pressure,
                    presence_harshness_score=presence_score,
                    sibilance_score=sibilance_score,
                    air_harshness_score=air_score,
                )
            )
            previous_momentary = momentary_lufs[index]
        return tuple(diagnostics)
    except Exception as exc:
        print(
            f"Temporal diagnostics unavailable: {exc}. Using neutral mastering diagnostics.",
            file=sys.stderr,
        )
        return neutral_temporal_diagnostics(frame_count)


def temporal_frame_state(
    *,
    density: float,
    vocal_presence: float,
    rhythm_drive: float,
    harmonic_density: float,
    chorus_space: float,
    breakdown_air: float,
    solo_score: float,
) -> str:
    if chorus_space >= 0.48 and density >= 0.58:
        return "wide_chorus"
    if (
        vocal_presence >= 0.56
        and vocal_presence >= max(rhythm_drive, harmonic_density) * 0.70
    ):
        return "vocal_focus"
    if solo_score >= 0.56 and vocal_presence <= 0.48:
        return "solo_feature"
    if breakdown_air >= 0.34 and rhythm_drive <= 0.42:
        return "breakdown"
    if rhythm_drive >= 0.62:
        return "rhythm_drive"
    if density <= 0.28:
        return "sparse"
    return "bed"


def top_intent(scores: dict[str, float]) -> tuple[str, float]:
    if not scores:
        return "preserve_original", 0.0
    intent, score = max(scores.items(), key=lambda item: item[1])
    return intent, clamp(score, 0.0, 1.0)


def peak_pressure_guard_score(diagnostic: TemporalDiagnosticFrame) -> float:
    peak_pressure = clamp(float(diagnostic.peak_pressure), 0.0, 1.0)
    peak_component = clamp((peak_pressure - 0.48) / 0.30, 0.0, 1.0)
    true_peak_component = clamp(
        (float(diagnostic.true_peak_dbtp) + 0.60) / 2.20, 0.0, 1.0
    )
    loudness_component = clamp((float(diagnostic.short_lufs) + 10.5) / 4.0, 0.0, 1.0)
    return clamp(
        0.58 * peak_component + 0.27 * true_peak_component + 0.15 * loudness_component,
        0.0,
        1.0,
    )


def decide_mastering_intent(
    *,
    time_sec: float,
    state: str,
    density: float,
    vocal_presence: float,
    rhythm_drive: float,
    harmonic_density: float,
    side_energy: float,
    air_energy: float,
    artifact_risk: float,
    chorus_space_opportunity: float,
    breakdown_air_opportunity: float,
    low_anchor_opportunity: float,
    mid_competition: float,
    low_overlap: float,
    solo_scores: dict[str, float],
    multiband_profile: MixMultibandProfile | None = None,
    diagnostics: TemporalDiagnosticFrame | None = None,
) -> MasteringIntentDecision:
    solo_peak = max(solo_scores.values()) if solo_scores else 0.0
    diagnostic = diagnostics or TemporalDiagnosticFrame()
    multiband = multiband_profile or MixMultibandProfile({}, 0.0, 0.0, 0.0, 0.0)
    peak_guard = peak_pressure_guard_score(diagnostic)
    hf_harshness = max(
        diagnostic.presence_harshness_score,
        diagnostic.sibilance_score,
        diagnostic.air_harshness_score,
    )
    vocal_clarity = clamp(
        vocal_presence * (0.45 + 0.55 * mid_competition) * (0.58 + 0.42 * density),
        0.0,
        1.0,
    )
    low_end_control = clamp(
        low_anchor_opportunity * (0.58 + 0.42 * low_overlap) * (0.62 + 0.38 * density),
        0.0,
        1.0,
    )
    low_end_control = clamp(
        low_end_control + 0.13 * multiband.low_pressure * (0.45 + 0.55 * rhythm_drive),
        0.0,
        1.0,
    )
    midrange_clarity = clamp(
        (0.35 + 0.65 * density)
        * (0.42 * vocal_presence + 0.36 * harmonic_density + 0.22 * solo_peak)
        * (0.45 + 0.55 * mid_competition),
        0.0,
        1.0,
    )
    midrange_clarity = clamp(
        midrange_clarity + 0.11 * multiband.body_congestion * (0.42 + 0.58 * density),
        0.0,
        1.0,
    )
    midrange_clarity = clamp(
        midrange_clarity + 0.14 * peak_guard * (0.40 + 0.60 * density),
        0.0,
        1.0,
    )
    harshness_control = clamp(
        0.58 * artifact_risk
        + 0.32 * max(0.0, air_energy - 0.54) / 0.46
        + 0.10 * max(0.0, side_energy - 0.72) / 0.28,
        0.0,
        1.0,
    )
    harshness_control = clamp(
        harshness_control + 0.24 * hf_harshness + 0.10 * diagnostic.peak_pressure,
        0.0,
        1.0,
    )
    harshness_control = clamp(
        harshness_control
        + 0.16 * multiband.presence_pressure * (0.45 + 0.55 * air_energy),
        0.0,
        1.0,
    )
    fold_translation_safety = clamp(
        0.52 * low_end_control
        + 0.25 * harshness_control
        + 0.23 * max(0.0, side_energy - 0.62) / 0.38,
        0.0,
        1.0,
    )
    fold_translation_safety = clamp(
        fold_translation_safety
        + 0.24 * diagnostic.phase_risk
        + 0.08 * diagnostic.peak_pressure,
        0.0,
        1.0,
    )
    fold_translation_safety = clamp(
        fold_translation_safety + 0.10 * peak_guard,
        0.0,
        1.0,
    )
    space_opportunity = clamp(
        max(chorus_space_opportunity, breakdown_air_opportunity, solo_peak * 0.55)
        * (1.0 - 0.38 * harshness_control)
        * (1.0 - 0.18 * fold_translation_safety),
        0.0,
        1.0,
    )
    space_opportunity = clamp(
        space_opportunity
        * (1.0 - 0.14 * multiband.presence_pressure)
        * (1.0 - 0.22 * peak_guard)
        + 0.08
        * breakdown_air_opportunity
        * multiband.air_opportunity
        * (1.0 - harshness_control),
        0.0,
        1.0,
    )
    punch_preservation = clamp(
        rhythm_drive * (1.0 - 0.30 * low_end_control) * (0.70 + 0.30 * density),
        0.0,
        1.0,
    )
    punch_preservation = clamp(
        punch_preservation * (0.78 + 0.22 * diagnostic.punch_score)
        + 0.18 * rhythm_drive * diagnostic.punch_score,
        0.0,
        1.0,
    )
    original_intent_preservation = clamp(
        0.34
        + 0.28 * max(vocal_presence, rhythm_drive)
        + 0.22 * fold_translation_safety
        + 0.16 * harshness_control,
        0.0,
        1.0,
    )

    principles = MasteringPrincipleScore(
        vocal_clarity=vocal_clarity,
        low_end_control=low_end_control,
        midrange_clarity=midrange_clarity,
        harshness_control=harshness_control,
        punch_preservation=punch_preservation,
        space_opportunity=space_opportunity,
        original_intent_preservation=original_intent_preservation,
        fold_translation_safety=fold_translation_safety,
    )

    vocal_protect = clamp(
        max(
            vocal_clarity * 0.92,
            vocal_presence * 0.42 if vocal_presence >= 0.52 else 0.0,
        ),
        0.0,
        1.0,
    )
    mid_decongest = clamp(
        max(vocal_clarity * 0.70, midrange_clarity * 0.62)
        + 0.10 * peak_guard * (0.45 + 0.55 * density),
        0.0,
        1.0,
    )
    low_tighten = clamp(
        max(low_end_control * 0.86, chorus_space_opportunity * rhythm_drive * 0.36),
        0.0,
        1.0,
    )
    fold_safety = clamp(fold_translation_safety, 0.0, 1.0)
    harshness_guard = clamp(harshness_control, 0.0, 1.0)
    chorus_space = clamp(
        chorus_space_opportunity
        * (1.0 - 0.48 * harshness_guard)
        * (1.0 - 0.24 * fold_safety)
        * (1.0 - 0.10 * vocal_protect),
        0.0,
        1.0,
    )
    chorus_space = clamp(chorus_space * (1.0 - 0.34 * peak_guard), 0.0, 1.0)
    breakdown_air = clamp(
        breakdown_air_opportunity
        * (1.0 - 0.68 * harshness_guard)
        * (1.0 - 0.15 * fold_safety),
        0.0,
        1.0,
    )
    breakdown_air = clamp(breakdown_air * (1.0 - 0.26 * peak_guard), 0.0, 1.0)
    feature_opportunity = {
        stem: clamp(value, 0.0, 1.0)
        for stem, value in solo_scores.items()
        if value >= 0.04
    }
    solo_lift = {
        stem: clamp(
            value
            * (1.0 - 0.52 * vocal_presence)
            * (1.0 - 0.34 * harshness_guard)
            * (1.0 - 0.42 * peak_guard),
            0.0,
            1.0,
        )
        for stem, value in feature_opportunity.items()
    }

    intent_scores = {
        "reduce_harshness": harshness_control * 1.10,
        "tighten_low_end": low_end_control * 1.05,
        "preserve_vocal": vocal_clarity * 1.02,
        "decongest_midrange": midrange_clarity,
        "open_chorus": chorus_space * (0.82 + 0.18 * density),
        "feature_solo": max(solo_lift.values(), default=0.0),
        "restore_air": breakdown_air * 0.95,
        "preserve_punch": punch_preservation * 0.72,
        "preserve_original": original_intent_preservation * 0.45,
    }
    intent, confidence = top_intent(intent_scores)

    reasons: list[str] = []
    if vocal_clarity >= 0.40:
        reasons.append("vocal clarity risk from active vocal plus mid competition")
    if low_end_control >= 0.40:
        reasons.append("low-end control prioritized while rhythm/bass energy is active")
    if midrange_clarity >= 0.40:
        reasons.append("dense midrange overlap suggests light decongestion")
    if harshness_control >= 0.36:
        reasons.append("air/side/fragility risk limits top and space expansion")
    if diagnostic.peak_pressure >= 0.45:
        reasons.append(
            "short-term loudness or true-peak pressure limits additional gain"
        )
    if peak_guard >= 0.35:
        reasons.append(
            "peak-pressure guard restrains space and favors midrange decongestion"
        )
    if diagnostic.punch_score >= 0.35 and rhythm_drive >= 0.30:
        reasons.append("transient/punch activity favors preserving attack")
    if diagnostic.phase_risk >= 0.35:
        reasons.append("phase/mono fold risk constrains width")
    if hf_harshness >= 0.38:
        reasons.append("presence/sibilance energy asks for a high-frequency guard")
    if chorus_space >= 0.28:
        reasons.append("density and rhythm support a restrained chorus-space lift")
    if breakdown_air >= 0.20:
        reasons.append("low rhythm density leaves room for gentle air")
    if max(feature_opportunity.values(), default=0.0) >= 0.18:
        reasons.append("non-vocal stem has a sustained feature opportunity")
    if fold_translation_safety >= 0.34:
        reasons.append("fold-down safety constrains width and low-end movement")
    if not reasons:
        reasons.append("preserve original balance with minimal temporal motion")

    return MasteringIntentDecision(
        time_sec=time_sec,
        state=state,
        principles=principles,
        intent=intent,
        confidence=confidence,
        actions={
            "vocal_protect": vocal_protect,
            "low_tighten": low_tighten,
            "mid_decongest": mid_decongest,
            "chorus_space": chorus_space,
            "breakdown_air": breakdown_air,
            "low_anchor": low_anchor_opportunity,
            "harshness_guard": harshness_guard,
            "fold_safety": fold_safety,
            "space_open": max(chorus_space, breakdown_air),
            "peak_pressure_guard": peak_guard,
        },
        solo_lift=solo_lift,
        reasons=tuple(reasons),
        feature_opportunity=feature_opportunity,
        intent_scores={
            key: clamp(value, 0.0, 1.0) for key, value in intent_scores.items()
        },
        diagnostics=diagnostic,
    )


def build_temporal_remaster(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    *,
    mode: str,
    profile: str,
    strength: float,
    hop_ms: float,
    step_sec: float,
    multiband_profile: MixMultibandProfile | None = None,
) -> TemporalRemasterResult:
    if mode == "off":
        return TemporalRemasterResult(False, reason="disabled", profile=profile)

    usable_windows = {
        name: window for name, window in windows.items() if window.envelope_db
    }
    if not usable_windows:
        return TemporalRemasterResult(
            False, reason="no window envelopes", profile=profile
        )
    frame_count = min(len(window.envelope_db) for window in usable_windows.values())
    if frame_count < 4:
        return TemporalRemasterResult(
            False, reason="not enough window frames", profile=profile
        )

    by_name = {item.name: item for item in stats}
    hop_sec = max(0.05, float(hop_ms) / 1000.0)
    times = tuple(index * hop_sec for index in range(frame_count))
    relative_levels = {
        name: tuple(
            window_relative_level(window, index) for index in range(frame_count)
        )
        for name, window in usable_windows.items()
    }

    mix_db_values = []
    for index in range(frame_count):
        energy = 0.0
        for window in usable_windows.values():
            energy += linear_energy(window.envelope_db[index])
        mix_db_values.append(10.0 * math.log10(max(energy, 1e-24)))
    density_floor = percentile_float(mix_db_values, 20.0)
    density_ceiling = max(percentile_float(mix_db_values, 85.0), density_floor + 3.0)
    densities = [
        clamp(
            (value - density_floor) / max(density_ceiling - density_floor, 1e-6),
            0.0,
            1.0,
        )
        for value in mix_db_values
    ]

    bright_weights = {name: bright_score(item) for name, item in by_name.items()}
    wide_weights = {name: wide_score(item) for name, item in by_name.items()}
    mid_weights = {name: band_fractions(item)[1] for name, item in by_name.items()}
    low_weights = {name: band_fractions(item)[0] for name, item in by_name.items()}
    fragility_weights = {
        name: temporal_duck_fragility_score(item, windows.get(name))
        for name, item in by_name.items()
    }
    if "bass" in by_name and "drums" in by_name:
        low_overlap = math.sqrt(
            max(low_weights.get("bass", 0.0), 0.0)
            * max(low_weights.get("drums", 0.0), 0.0)
        )
    else:
        low_overlap = 0.0
    harmonic_stems = tuple(
        stem for stem in ("guitar", "piano", "other") if stem in relative_levels
    )
    solo_stems = tuple(
        stem for stem in ("guitar", "piano", "other", "bass") if stem in relative_levels
    )
    profile_space_scale = {"conservative": 0.76, "natural": 1.0, "open": 1.16}.get(
        profile, 1.0
    )
    profile_guard_scale = {"conservative": 1.12, "natural": 1.0, "open": 0.94}.get(
        profile, 1.0
    )

    frames: list[TemporalFrame] = []
    intents: list[MasteringIntentDecision] = []
    raw_vocal_protect: list[float] = []
    raw_chorus_space: list[float] = []
    raw_breakdown_air: list[float] = []
    raw_low_anchor: list[float] = []
    raw_low_tighten: list[float] = []
    raw_mid_decongest: list[float] = []
    raw_harshness_guard: list[float] = []
    raw_fold_safety: list[float] = []
    raw_solo: dict[str, list[float]] = {stem: [] for stem in solo_stems}
    diagnostic_frames = compute_temporal_mix_diagnostics(
        stats,
        frame_count=frame_count,
        hop_sec=hop_sec,
    )

    for index in range(frame_count):
        diagnostics = (
            diagnostic_frames[index]
            if index < len(diagnostic_frames)
            else TemporalDiagnosticFrame()
        )
        density = densities[index]
        vocal = (
            relative_levels.get("vocals", (0.0,) * frame_count)[index]
            if "vocals" in relative_levels
            else 0.0
        )
        drums = (
            relative_levels.get("drums", (0.0,) * frame_count)[index]
            if "drums" in relative_levels
            else 0.0
        )
        bass = (
            relative_levels.get("bass", (0.0,) * frame_count)[index]
            if "bass" in relative_levels
            else 0.0
        )
        rhythm = clamp(
            math.sqrt(max(drums, 0.0) * max(bass if bass > 0.0 else drums * 0.45, 0.0)),
            0.0,
            1.0,
        )

        harmonic_values = [relative_levels[stem][index] for stem in harmonic_stems]
        if harmonic_values:
            harmonic_peak = max(harmonic_values)
            harmonic_mean = sum(harmonic_values) / len(harmonic_values)
            harmonic = clamp(0.62 * harmonic_peak + 0.38 * harmonic_mean, 0.0, 1.0)
        else:
            harmonic = 0.0

        weighted_side = 0.0
        weighted_air = 0.0
        active_weight = 0.0
        artifact = 0.0
        for stem, levels in relative_levels.items():
            level = levels[index]
            if level <= 0.0:
                continue
            active_weight += level
            weighted_side += level * wide_weights.get(stem, 0.0)
            weighted_air += level * bright_weights.get(stem, 0.0)
            if stem not in {"vocals", "bass", "drums"}:
                artifact = max(artifact, level * fragility_weights.get(stem, 0.0))
        side_energy = clamp(weighted_side / max(active_weight, 1e-6), 0.0, 1.0)
        air_energy = clamp(weighted_air / max(active_weight, 1e-6), 0.0, 1.0)
        diagnostic_hf_risk = max(
            diagnostics.presence_harshness_score,
            diagnostics.sibilance_score,
            diagnostics.air_harshness_score,
        )
        artifact_risk = clamp(
            0.55 * artifact
            + 0.30 * max(0.0, air_energy - 0.55)
            + 0.15 * max(0.0, side_energy - 0.70)
            + 0.16 * diagnostic_hf_risk
            + 0.08 * diagnostics.phase_risk,
            0.0,
            1.0,
        )

        solo_scores: dict[str, float] = {}
        for stem in solo_stems:
            level = relative_levels[stem][index]
            competitor = max(
                vocal,
                rhythm,
                *(
                    relative_levels[other][index]
                    for other in solo_stems
                    if other != stem
                ),
                0.0,
            )
            dominance = clamp((level - competitor + 0.20) / 0.55, 0.0, 1.0)
            solo_scores[stem] = clamp(
                level * dominance * (1.0 - 0.70 * vocal), 0.0, 1.0
            )

        solo_peak = max(solo_scores.values()) if solo_scores else 0.0
        mid_competition = 0.0
        for stem in harmonic_stems:
            mid_competition = max(
                mid_competition,
                relative_levels[stem][index] * mid_weights.get(stem, 0.0),
            )
        mid_competition = clamp(0.72 * mid_competition + 0.28 * harmonic, 0.0, 1.0)
        chorus_opportunity = clamp(
            density
            * (0.45 + 0.55 * rhythm)
            * (0.45 + 0.55 * harmonic)
            * (0.60 + 0.40 * side_energy)
            * (1.0 - 0.45 * artifact_risk),
            0.0,
            1.0,
        )
        breakdown_opportunity = clamp(
            (1.0 - rhythm)
            * (0.30 + 0.70 * harmonic)
            * (0.35 + 0.65 * air_energy)
            * (1.0 - 0.62 * artifact_risk)
            * (0.55 + 0.45 * (1.0 - density)),
            0.0,
            1.0,
        )
        low_anchor = clamp(rhythm * (0.45 + 0.55 * density), 0.0, 1.0)
        section_state = temporal_frame_state(
            density=density,
            vocal_presence=vocal,
            rhythm_drive=rhythm,
            harmonic_density=harmonic,
            chorus_space=chorus_opportunity,
            breakdown_air=breakdown_opportunity,
            solo_score=solo_peak,
        )
        intent_decision = decide_mastering_intent(
            time_sec=times[index],
            state=section_state,
            density=density,
            vocal_presence=vocal,
            rhythm_drive=rhythm,
            harmonic_density=harmonic,
            side_energy=side_energy,
            air_energy=air_energy,
            artifact_risk=artifact_risk,
            chorus_space_opportunity=chorus_opportunity,
            breakdown_air_opportunity=breakdown_opportunity,
            low_anchor_opportunity=low_anchor,
            mid_competition=mid_competition,
            low_overlap=low_overlap,
            solo_scores=solo_scores,
            multiband_profile=multiband_profile,
            diagnostics=diagnostics,
        )

        frames.append(
            TemporalFrame(
                time_sec=times[index],
                mix_energy_db=mix_db_values[index],
                vocal_presence=vocal,
                rhythm_drive=rhythm,
                harmonic_density=harmonic,
                side_energy=side_energy,
                air_energy=air_energy,
                artifact_risk=artifact_risk,
                section_state=section_state,
                diagnostics=diagnostics,
            )
        )
        intents.append(intent_decision)
        raw_vocal_protect.append(
            intent_decision.actions["vocal_protect"] * profile_guard_scale
        )
        raw_chorus_space.append(
            intent_decision.actions["chorus_space"] * profile_space_scale
        )
        raw_breakdown_air.append(
            intent_decision.actions["breakdown_air"] * profile_space_scale
        )
        raw_low_anchor.append(low_anchor)
        raw_low_tighten.append(
            intent_decision.actions["low_tighten"] * profile_guard_scale
        )
        raw_mid_decongest.append(
            intent_decision.actions["mid_decongest"] * profile_guard_scale
        )
        raw_harshness_guard.append(
            intent_decision.actions["harshness_guard"] * profile_guard_scale
        )
        raw_fold_safety.append(
            intent_decision.actions["fold_safety"] * profile_guard_scale
        )
        for stem in raw_solo:
            raw_solo[stem].append(
                intent_decision.solo_lift.get(stem, 0.0) * profile_space_scale
            )

    stable_states = stabilized_temporal_states(
        times,
        tuple(frame.section_state for frame in frames),
        min_sec=max(6.0, float(step_sec) * 2.0),
    )
    for frame, intent_decision, stable_state in zip(frames, intents, stable_states):
        frame.section_state = stable_state
        intent_decision.state = stable_state

    lane_strength = clamp(float(strength), 0.0, 1.0)

    def scaled(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(clamp(value * lane_strength, 0.0, 1.0) for value in values)

    lanes = AutomationLanes(
        times=times,
        step_sec=max(1.0, float(step_sec)),
        vocal_protect=scaled(
            smooth_automation_lane(
                raw_vocal_protect, hop_sec=hop_sec, attack_sec=1.0, release_sec=4.5
            )
        ),
        chorus_space=scaled(
            smooth_automation_lane(
                raw_chorus_space, hop_sec=hop_sec, attack_sec=5.0, release_sec=12.0
            )
        ),
        breakdown_air=scaled(
            smooth_automation_lane(
                raw_breakdown_air, hop_sec=hop_sec, attack_sec=6.0, release_sec=14.0
            )
        ),
        low_anchor=scaled(
            smooth_automation_lane(
                raw_low_anchor, hop_sec=hop_sec, attack_sec=0.8, release_sec=3.5
            )
        ),
        low_tighten=scaled(
            smooth_automation_lane(
                raw_low_tighten, hop_sec=hop_sec, attack_sec=0.8, release_sec=3.5
            )
        ),
        mid_decongest=scaled(
            smooth_automation_lane(
                raw_mid_decongest, hop_sec=hop_sec, attack_sec=1.4, release_sec=5.5
            )
        ),
        harshness_guard=scaled(
            smooth_automation_lane(
                raw_harshness_guard, hop_sec=hop_sec, attack_sec=1.2, release_sec=6.0
            )
        ),
        fold_safety=scaled(
            smooth_automation_lane(
                raw_fold_safety, hop_sec=hop_sec, attack_sec=1.5, release_sec=6.5
            )
        ),
        solo_feature={
            stem: scaled(
                smooth_automation_lane(
                    values, hop_sec=hop_sec, attack_sec=2.5, release_sec=7.0
                )
            )
            for stem, values in raw_solo.items()
        },
    )
    if not any(
        max(lane, default=0.0) >= 0.05
        for lane in (
            lanes.vocal_protect,
            lanes.chorus_space,
            lanes.breakdown_air,
            lanes.low_tighten,
            lanes.mid_decongest,
            lanes.harshness_guard,
            lanes.fold_safety,
        )
    ):
        return TemporalRemasterResult(
            False,
            frames=tuple(frames),
            lanes=lanes,
            intents=tuple(intents),
            reason="lanes below threshold",
            strength=lane_strength,
            profile=profile,
        )
    return TemporalRemasterResult(
        True,
        frames=tuple(frames),
        lanes=lanes,
        intents=tuple(intents),
        reason="auto",
        strength=lane_strength,
        profile=profile,
    )


def temporal_lane_peak(values: tuple[float, ...]) -> float:
    return max(values, default=0.0)


def temporal_lane_mean(values: tuple[float, ...]) -> float:
    return sum(values) / len(values) if values else 0.0


def stabilized_temporal_states(
    times: tuple[float, ...],
    states: tuple[str, ...],
    *,
    min_sec: float,
) -> tuple[str, ...]:
    if not states or len(states) != len(times):
        return states
    if len(states) < 3:
        return states
    hop_sec = max(times[1] - times[0], 0.05) if len(times) > 1 else 0.5
    min_frames = max(2, int(round(float(min_sec) / hop_sec)))
    current = list(states)

    for _pass in range(4):
        runs: list[tuple[int, int, str]] = []
        start = 0
        for index in range(1, len(current)):
            if current[index] != current[start]:
                runs.append((start, index, current[start]))
                start = index
        runs.append((start, len(current), current[start]))
        changed = False
        for run_index, (start, end, state) in enumerate(runs):
            length = end - start
            if length >= min_frames:
                continue
            previous_run = runs[run_index - 1] if run_index > 0 else None
            next_run = runs[run_index + 1] if run_index + 1 < len(runs) else None
            if previous_run is None and next_run is None:
                continue
            if previous_run is None:
                assert next_run is not None
                replacement = next_run[2]
            elif next_run is None:
                replacement = previous_run[2]
            else:
                previous_length = previous_run[1] - previous_run[0]
                next_length = next_run[1] - next_run[0]
                if previous_run[2] == next_run[2]:
                    replacement = previous_run[2]
                else:
                    replacement = (
                        previous_run[2]
                        if previous_length >= next_length
                        else next_run[2]
                    )
            if replacement == state:
                continue
            for index in range(start, end):
                current[index] = replacement
            changed = True
        if not changed:
            break
    return tuple(current)


def temporal_state_sections(
    frames: tuple[TemporalFrame, ...], *, min_sec: float
) -> list[tuple[float, float, str]]:
    frame_count = len(frames)
    if frame_count == 0:
        return []
    sections: list[tuple[float, float, str]] = []
    start = frames[0].time_sec
    state = frames[0].section_state
    for previous, current in zip(frames, frames[1:]):
        if current.section_state == state:
            continue
        sections.append((start, current.time_sec, state))
        start = current.time_sec
        state = current.section_state
    last_frame = frames[frame_count - 1]
    if frame_count > 1:
        previous_frame = frames[frame_count - 2]
        step = max(last_frame.time_sec - previous_frame.time_sec, 0.5)
    else:
        step = 0.5
    sections.append((start, last_frame.time_sec + step, state))

    merged: list[tuple[float, float, str]] = []
    for start_sec, end_sec, section_state in sections:
        if merged and end_sec - start_sec < min_sec:
            prev_start, _prev_end, prev_state = merged[-1]
            merged[-1] = (prev_start, end_sec, prev_state)
        else:
            merged.append((start_sec, end_sec, section_state))
    consolidated: list[tuple[float, float, str]] = []
    for start_sec, end_sec, section_state in merged:
        if consolidated and consolidated[-1][2] == section_state:
            prev_start, _prev_end, _prev_state = consolidated[-1]
            consolidated[-1] = (prev_start, end_sec, section_state)
        else:
            consolidated.append((start_sec, end_sec, section_state))
    return consolidated


def print_temporal_remaster_result(result: TemporalRemasterResult) -> None:
    if not result.enabled or result.lanes is None:
        if result.reason not in {"disabled", ""}:
            print(f"Temporal remaster skipped: {result.reason}", flush=True)
        return
    lanes = result.lanes
    print("Temporal mastering:", flush=True)
    print(
        f"  strength={result.strength:.2f} "
        f"profile={result.profile} "
        f"frames={len(lanes.times)} "
        f"step={lanes.step_sec:.1f}s "
        f"vocal={temporal_lane_peak(lanes.vocal_protect):.2f}/{temporal_lane_mean(lanes.vocal_protect):.2f} "
        f"chorus={temporal_lane_peak(lanes.chorus_space):.2f}/{temporal_lane_mean(lanes.chorus_space):.2f} "
        f"breakdown={temporal_lane_peak(lanes.breakdown_air):.2f}/{temporal_lane_mean(lanes.breakdown_air):.2f} "
        f"low={temporal_lane_peak(lanes.low_tighten):.2f}/{temporal_lane_mean(lanes.low_tighten):.2f} "
        f"mid={temporal_lane_peak(lanes.mid_decongest):.2f}/{temporal_lane_mean(lanes.mid_decongest):.2f} "
        f"harsh={temporal_lane_peak(lanes.harshness_guard):.2f}/{temporal_lane_mean(lanes.harshness_guard):.2f}",
        flush=True,
    )
    if result.intents:
        intent_counts: dict[str, int] = {}
        for intent in result.intents:
            intent_counts[intent.intent] = intent_counts.get(intent.intent, 0) + 1
        top_intents = sorted(
            intent_counts.items(), key=lambda item: item[1], reverse=True
        )[:4]
        text = " ".join(f"{intent}={count}" for intent, count in top_intents)
        print(f"  intents: {text}", flush=True)
    solo_peaks = sorted(
        (
            (stem, temporal_lane_peak(values), temporal_lane_mean(values))
            for stem, values in lanes.solo_feature.items()
            if temporal_lane_peak(values) >= 0.05
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    if solo_peaks:
        text = " ".join(
            f"{stem}={peak:.2f}/{mean:.2f}" for stem, peak, mean in solo_peaks[:3]
        )
        print(f"  solo lanes: {text}", flush=True)
    diagnostic_summary = temporal_diagnostic_summary(result.intents)
    if diagnostic_summary:
        short_loudness = diagnostic_summary.get("short_lufs_estimate", {})
        true_peak = diagnostic_summary.get("true_peak_dbtp_estimate", {})
        punch = diagnostic_summary.get("punch_score", {})
        phase = diagnostic_summary.get("phase_risk", {})
        sibilance = diagnostic_summary.get("sibilance_score", {})
        mono_loss = diagnostic_summary.get("mono_fold_loss_db", {})
        print(
            "  diagnostics: "
            f"shortLUFS={short_loudness.get('max', -120.0):+.1f}/{short_loudness.get('active_mean', short_loudness.get('mean', -120.0)):+.1f} "
            f"TP={true_peak.get('max', -120.0):+.1f}dBTP "
            f"punch={punch.get('peak', 0.0):.2f}/{punch.get('mean', 0.0):.2f} "
            f"phaseRisk={phase.get('p95', 0.0):.2f}/{phase.get('peak', 0.0):.2f} "
            f"monoLoss={mono_loss.get('min', 0.0):+.1f}dB "
            f"sibilance={sibilance.get('p95', 0.0):.2f}/{sibilance.get('peak', 0.0):.2f}",
            flush=True,
        )
    sections = temporal_state_sections(result.frames, min_sec=6.0)[:6]
    if sections:
        text = " ".join(
            f"{start:.0f}-{end:.0f}s:{state}" for start, end, state in sections
        )
        print(f"  sections: {text}", flush=True)


def select_temporal_mastering_profile(
    requested: str,
    stats: list[StemStats],
    windows: dict[str, WindowStats],
) -> tuple[str, str]:
    if requested != "auto":
        return requested, "manual"

    by_name = {item.name: item for item in stats}
    vocals = by_name.get("vocals")
    vocal_window = windows.get("vocals")
    leak_risk = 0.0
    if vocals is not None and vocal_window is not None:
        for item in stats:
            if item.name in {"vocals", "bass", "drums"}:
                continue
            leak_risk = max(
                leak_risk,
                vocal_leak_risk(item, vocals, windows.get(item.name), vocal_window),
            )

    shares = stem_energy_shares(stats)
    active_weights: dict[str, float] = {}
    for item in stats:
        window = windows.get(item.name)
        active_ratio = window.active_ratio if window is not None else 0.5
        active_weights[item.name] = active_ratio * shares.get(item.name, 0.0)
    weight_total = max(sum(active_weights.values()), 1e-9)
    weighted_wide = (
        sum(wide_score(item) * active_weights[item.name] for item in stats)
        / weight_total
    )
    weighted_bright = (
        sum(bright_score(item) * active_weights[item.name] for item in stats)
        / weight_total
    )
    max_fragility = max(
        (temporal_duck_fragility_score(item, windows.get(item.name)) for item in stats),
        default=0.0,
    )
    vocal_share = shares.get("vocals", 0.0)

    if leak_risk >= 0.32 or max_fragility >= 0.58 or weighted_bright >= 0.72:
        return (
            "conservative",
            f"auto leak={leak_risk:.2f} fragility={max_fragility:.2f} bright={weighted_bright:.2f}",
        )
    if (
        weighted_wide >= 0.54
        and weighted_bright <= 0.62
        and max_fragility <= 0.42
        and leak_risk <= 0.22
        and vocal_share <= 0.48
    ):
        return (
            "open",
            f"auto width={weighted_wide:.2f} bright={weighted_bright:.2f} fragility={max_fragility:.2f}",
        )
    return (
        "natural",
        f"auto balanced leak={leak_risk:.2f} width={weighted_wide:.2f} fragility={max_fragility:.2f}",
    )


def print_temporal_profile_selection(
    requested: str, selected: str, reason: str
) -> None:
    if requested == "auto":
        print(f"Temporal mastering profile: auto -> {selected} ({reason})", flush=True)


def mastering_principles_to_dict(
    principles: MasteringPrincipleScore,
) -> dict[str, float]:
    return {
        "vocal_clarity": principles.vocal_clarity,
        "low_end_control": principles.low_end_control,
        "midrange_clarity": principles.midrange_clarity,
        "harshness_control": principles.harshness_control,
        "punch_preservation": principles.punch_preservation,
        "space_opportunity": principles.space_opportunity,
        "original_intent_preservation": principles.original_intent_preservation,
        "fold_translation_safety": principles.fold_translation_safety,
    }


def temporal_diagnostics_to_dict(
    diagnostics: TemporalDiagnosticFrame,
) -> dict[str, float]:
    return {
        "momentary_lufs_estimate": diagnostics.momentary_lufs,
        "short_lufs_estimate": diagnostics.short_lufs,
        "local_loudness_range_db": diagnostics.loudness_range_db,
        "true_peak_dbtp_estimate": diagnostics.true_peak_dbtp,
        "crest_factor_db": diagnostics.crest_factor_db,
        "transient_score": diagnostics.transient_score,
        "punch_score": diagnostics.punch_score,
        "low_punch_score": diagnostics.low_punch_score,
        "mono_correlation": diagnostics.mono_correlation,
        "mono_fold_loss_db": diagnostics.mono_fold_loss_db,
        "low_correlation": diagnostics.low_correlation,
        "mid_correlation": diagnostics.mid_correlation,
        "high_correlation": diagnostics.high_correlation,
        "phase_risk": diagnostics.phase_risk,
        "peak_pressure": diagnostics.peak_pressure,
        "presence_harshness_score": diagnostics.presence_harshness_score,
        "sibilance_score": diagnostics.sibilance_score,
        "air_harshness_score": diagnostics.air_harshness_score,
    }


def average_named_values(items: Iterable[dict[str, float]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in items:
        for key, value in item.items():
            totals[key] = totals.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return {key: round(totals[key] / max(counts[key], 1), 4) for key in sorted(totals)}


def temporal_intent_step_sec(intents: tuple[MasteringIntentDecision, ...]) -> float:
    if len(intents) < 2:
        return 0.0
    deltas = [
        right.time_sec - left.time_sec
        for left, right in zip(intents, intents[1:])
        if right.time_sec > left.time_sec
    ]
    return median_float(deltas) if deltas else 0.0


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 1e-12:
        return sum(values) / len(values) if values else 0.0
    return sum(value * weight for value, weight in zip(values, weights)) / total


def loudness_weights(rows: list[dict[str, float]]) -> list[float]:
    loudness = [float(row.get("short_lufs_estimate", -120.0)) for row in rows]
    active = [value for value in loudness if value > -70.0]
    if not active:
        return [0.0 for _value in loudness]
    ceiling = max(active)
    return [
        10.0 ** ((value - ceiling) / 10.0) if value > -70.0 else 0.0
        for value in loudness
    ]


def active_db_values(values: list[float], *, floor_db: float = -70.0) -> list[float]:
    return [value for value in values if value > floor_db]


def temporal_diagnostic_summary(
    intents: tuple[MasteringIntentDecision, ...],
) -> dict[str, dict[str, float]]:
    if not intents:
        return {}
    rows = [temporal_diagnostics_to_dict(intent.diagnostics) for intent in intents]
    keys = sorted({key for row in rows for key in row})
    summary: dict[str, dict[str, float]] = {}
    step_sec = temporal_intent_step_sec(intents)
    weights = loudness_weights(rows)
    for key in keys:
        values = [float(row.get(key, 0.0)) for row in rows]
        if key in {
            "momentary_lufs_estimate",
            "short_lufs_estimate",
            "true_peak_dbtp_estimate",
            "crest_factor_db",
            "local_loudness_range_db",
        }:
            active_values = (
                active_db_values(values)
                if key in {"momentary_lufs_estimate", "short_lufs_estimate"}
                else values
            )
            percentile_source = active_values or values
            summary[key] = {
                "min": round(min(values), 4),
                "mean": round(sum(values) / len(values), 4),
                "max": round(max(values), 4),
                "p10": round(percentile_float(percentile_source, 10.0), 4),
                "p95": round(percentile_float(percentile_source, 95.0), 4),
            }
            if key in {"momentary_lufs_estimate", "short_lufs_estimate"}:
                summary[key]["absolute_min"] = round(min(values), 4)
                summary[key]["active_min"] = (
                    round(min(active_values), 4)
                    if active_values
                    else round(min(values), 4)
                )
                summary[key]["active_mean"] = (
                    round(sum(active_values) / len(active_values), 4)
                    if active_values
                    else round(sum(values) / len(values), 4)
                )
        elif key in {
            "mono_correlation",
            "low_correlation",
            "mid_correlation",
            "high_correlation",
            "mono_fold_loss_db",
        }:
            summary[key] = {
                "min": round(min(values), 4),
                "mean": round(sum(values) / len(values), 4),
                "p05": round(percentile_float(values, 5.0), 4),
                "p95": round(percentile_float(values, 95.0), 4),
                "loudness_weighted_mean": round(weighted_mean(values, weights), 4),
            }
        else:
            summary[key] = {
                "mean": round(sum(values) / len(values), 4),
                "p95": round(percentile_float(values, 95.0), 4),
                "peak": round(max(values), 4),
            }
            if key in {
                "phase_risk",
                "peak_pressure",
                "presence_harshness_score",
                "sibilance_score",
                "air_harshness_score",
                "punch_score",
                "transient_score",
                "low_punch_score",
            }:
                summary[key]["loudness_weighted_mean"] = round(
                    weighted_mean(values, weights), 4
                )
                summary[key]["duration_over_0_5_sec"] = round(
                    sum(
                        step_sec
                        for value, weight in zip(values, weights)
                        if value >= 0.5 and weight > 0.0
                    ),
                    3,
                )
    return summary


def peak_pressure_true_peak_guard(
    base_target_tp: float,
    result: TemporalRemasterResult,
) -> tuple[float, str]:
    if not result.enabled or not result.intents:
        return base_target_tp, ""
    summary = temporal_diagnostic_summary(result.intents)
    pressure = summary.get("peak_pressure", {})
    true_peak = summary.get("true_peak_dbtp_estimate", {})
    weighted_pressure = float(
        pressure.get("loudness_weighted_mean", pressure.get("mean", 0.0))
    )
    pressure_p95 = float(pressure.get("p95", 0.0))
    pressure_duration = float(pressure.get("duration_over_0_5_sec", 0.0))
    true_peak_max = float(true_peak.get("max", -120.0))
    guard = max(
        clamp((weighted_pressure - 0.54) / 0.24, 0.0, 1.0),
        clamp((pressure_p95 - 0.64) / 0.24, 0.0, 1.0),
        clamp((true_peak_max + 0.40) / 2.20, 0.0, 1.0),
    )
    if pressure_duration < 12.0 and true_peak_max <= 0.10:
        guard *= 0.45
    if guard < 0.08:
        return base_target_tp, ""
    extra_headroom = 0.18 + 0.42 * guard
    desired_target = -1.0 - extra_headroom
    guarded_target = min(float(base_target_tp), desired_target)
    if abs(guarded_target - float(base_target_tp)) < 0.01:
        return base_target_tp, ""
    reason = (
        f"pressure={weighted_pressure:.2f}/{pressure_p95:.2f} "
        f"TPmax={true_peak_max:+.2f}dBTP "
        f"hotDuration={pressure_duration:.1f}s"
    )
    return round(guarded_target, 2), reason


def ordered_named_scores(scores: dict[str, float]) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def primary_secondary_intents(intent_scores: dict[str, float]) -> tuple[str, list[str]]:
    ordered = ordered_named_scores(intent_scores)
    if not ordered:
        return "preserve_original", []
    primary, primary_score = ordered[0]
    secondary_threshold = max(0.20, primary_score * 0.55)
    secondary = [name for name, score in ordered[1:] if score >= secondary_threshold][
        :3
    ]
    if not secondary:
        secondary = [name for name, score in ordered[1:] if score >= 0.12][:2]
    return primary, secondary


def applied_render_actions(
    actions: dict[str, float],
    applied_solo_lift: dict[str, float],
) -> dict[str, float]:
    chorus = actions.get("chorus_space", 0.0)
    breakdown = actions.get("breakdown_air", 0.0)
    low = actions.get("low_tighten", 0.0)
    low_anchor = actions.get("low_anchor", 0.0)
    mid = actions.get("mid_decongest", 0.0)
    harshness = actions.get("harshness_guard", 0.0)
    fold = actions.get("fold_safety", 0.0)
    feature_peak = max(applied_solo_lift.values(), default=0.0)
    space_value = clamp(
        0.78 * chorus + 0.52 * breakdown + 0.42 * feature_peak, 0.0, 1.0
    )
    space_value *= 1.0 - 0.45 * harshness
    space_value *= 1.0 - 0.24 * fold
    stereo_low = max(low, fold * 0.55, chorus * low_anchor * 0.28)
    stereo_mid = max(mid, fold * 0.18)
    stereo_high = max(harshness, fold * 0.34)
    return {
        "space_bed_gain_db": round(space_value * 1.25, 3),
        "feature_space_gain_db": round(feature_peak * 0.525, 3),
        "drum_low_tighten_db": round(-low * 0.55, 3),
        "mid_decongest_db": round(-mid * 0.75, 3),
        "stereo_low_tighten_db": round(-stereo_low * 0.45, 3),
        "stereo_mid_decongest_db": round(-stereo_mid * 0.50, 3),
        "stereo_high_guard_db": round(-stereo_high * 0.38, 3),
    }


def classify_segment_reasons(
    principles: dict[str, float],
    actions: dict[str, float],
    diagnostics: dict[str, float],
    feature_opportunity: dict[str, float],
    applied_solo_lift: dict[str, float],
) -> tuple[list[str], list[str], list[str]]:
    hf_guard_score = max(
        diagnostics.get("presence_harshness_score", 0.0),
        diagnostics.get("sibilance_score", 0.0),
        diagnostics.get("air_harshness_score", 0.0),
    )
    candidates = [
        (
            "vocal clarity risk from active vocal plus mid competition",
            principles.get("vocal_clarity", 0.0),
        ),
        (
            "low-end control prioritized while rhythm/bass energy is active",
            max(
                principles.get("low_end_control", 0.0), actions.get("low_tighten", 0.0)
            ),
        ),
        (
            "dense midrange overlap suggests light decongestion",
            max(
                principles.get("midrange_clarity", 0.0),
                actions.get("mid_decongest", 0.0),
            ),
        ),
        (
            "air/side/fragility risk limits top and space expansion",
            max(
                principles.get("harshness_control", 0.0),
                actions.get("harshness_guard", 0.0),
            ),
        ),
        (
            "short-term loudness or true-peak pressure limits additional gain",
            diagnostics.get("peak_pressure", 0.0),
        ),
        (
            "peak-pressure guard restrains space and favors midrange decongestion",
            actions.get("peak_pressure_guard", 0.0),
        ),
        (
            "transient/punch activity favors preserving attack",
            diagnostics.get("punch_score", 0.0),
        ),
        (
            "phase/mono fold risk constrains width",
            diagnostics.get("phase_risk", 0.0),
        ),
        (
            "presence/sibilance energy asks for a high-frequency guard",
            hf_guard_score,
        ),
        (
            "low rhythm density leaves room for gentle air",
            actions.get("breakdown_air", 0.0),
        ),
        (
            "density and rhythm support a restrained chorus-space lift",
            actions.get("chorus_space", 0.0),
        ),
        (
            "non-vocal stem has a sustained feature opportunity",
            max(feature_opportunity.values(), default=0.0),
        ),
        (
            "fold-down safety constrains width and low-end movement",
            max(
                principles.get("fold_translation_safety", 0.0),
                actions.get("fold_safety", 0.0),
            ),
        ),
    ]
    ranked = sorted(candidates, key=lambda item: item[1], reverse=True)
    primary = [text for text, score in ranked if score >= 0.40][:3]
    secondary = [text for text, score in ranked if 0.22 <= score < 0.40][:5]

    suppressed: list[str] = []
    space_opportunity = principles.get("space_opportunity", 0.0)
    space_open = actions.get(
        "space_open",
        max(actions.get("chorus_space", 0.0), actions.get("breakdown_air", 0.0)),
    )
    if space_opportunity >= 0.30 and space_open <= space_opportunity * 0.68:
        suppressed.append(
            "space expansion opportunity was constrained by vocal, fold, or harshness guard"
        )
    feature_peak = max(feature_opportunity.values(), default=0.0)
    applied_feature_peak = max(applied_solo_lift.values(), default=0.0)
    if feature_peak >= 0.20 and applied_feature_peak <= feature_peak * 0.62:
        suppressed.append(
            "feature opportunity was held back by vocal, pressure, or harshness guard"
        )
    if (
        actions.get("chorus_space", 0.0) >= 0.20
        and actions.get("fold_safety", 0.0) >= 0.25
    ):
        suppressed.append("chorus width was limited for fold-down translation")
    if (
        hf_guard_score >= 0.30
        and actions.get("breakdown_air", 0.0)
        <= principles.get("space_opportunity", 0.0) * 0.55
    ):
        suppressed.append(
            "air restoration was restrained by presence or sibilance risk"
        )
    if not primary and not secondary:
        secondary.append("preserve original balance with minimal temporal motion")
    return primary, secondary, suppressed[:4]


def fallback_primary_reason(primary_intent: str, confidence: float) -> str:
    if confidence < 0.45:
        return "low-confidence segment; preserving original balance"
    return {
        "preserve_vocal": "vocal preservation is the strongest available intent",
        "tighten_low_end": "low-end control is the strongest available intent",
        "decongest_midrange": "midrange cleanup is the strongest available intent",
        "reduce_harshness": "high-frequency guard is the strongest available intent",
        "feature_solo": "feature support is the strongest available intent",
        "open_chorus": "chorus space is the strongest available intent",
        "restore_air": "air restoration is the strongest available intent",
        "preserve_punch": "punch preservation is the strongest available intent",
    }.get(primary_intent, "preserve original balance with minimal temporal motion")


def temporal_report_substate(intent: MasteringIntentDecision) -> str:
    actions = intent.actions
    principles = intent.principles
    if intent.state == "vocal_focus":
        if actions.get("chorus_space", 0.0) >= 0.30:
            return "vocal_chorus"
        if (
            actions.get("mid_decongest", 0.0) >= 0.32
            or actions.get("low_tighten", 0.0) >= 0.38
        ):
            return "vocal_focus_dense"
        if (
            actions.get("breakdown_air", 0.0) >= 0.18
            and actions.get("low_tighten", 0.0) < 0.30
        ):
            return "vocal_focus_sparse"
        return "vocal_focus_balanced"
    if intent.state == "wide_chorus":
        if actions.get("low_tighten", 0.0) >= actions.get("chorus_space", 0.0):
            return "wide_chorus_low_control"
        return "wide_chorus_open"
    if intent.state == "rhythm_drive":
        if actions.get("low_tighten", 0.0) >= 0.35:
            return "rhythm_drive_low_control"
        return "rhythm_drive"
    if intent.state == "solo_feature":
        if max(intent.feature_opportunity.values(), default=0.0) >= 0.30:
            return "solo_feature_prominent"
        return "solo_feature"
    if principles.harshness_control >= 0.36:
        return f"{intent.state}_guarded"
    return intent.state


def should_split_temporal_subsegment(
    anchor: MasteringIntentDecision, current: MasteringIntentDecision
) -> bool:
    thresholds = {
        "chorus_space": 0.15,
        "low_tighten": 0.15,
        "vocal_protect": 0.12,
        "mid_decongest": 0.12,
        "fold_safety": 0.14,
    }
    if temporal_report_substate(anchor) != temporal_report_substate(current):
        return True
    if abs(current.confidence - anchor.confidence) >= 0.15:
        return True
    return any(
        abs(current.actions.get(key, 0.0) - anchor.actions.get(key, 0.0)) >= threshold
        for key, threshold in thresholds.items()
    )


def temporal_subsegment_ranges(
    segment_intents: list[MasteringIntentDecision],
    segment_end_sec: float,
    *,
    min_sec: float,
    max_segments: int,
) -> list[tuple[float, float, str, list[MasteringIntentDecision]]]:
    if not segment_intents:
        return []
    ranges: list[tuple[float, float, str, list[MasteringIntentDecision]]] = []
    start_index = 0
    anchor = segment_intents[0]
    for index, current in enumerate(segment_intents[1:], start=1):
        start_time = segment_intents[start_index].time_sec
        if current.time_sec - start_time < min_sec:
            continue
        if not should_split_temporal_subsegment(anchor, current):
            continue
        items = segment_intents[start_index:index]
        if items:
            ranges.append(
                (start_time, current.time_sec, temporal_report_substate(anchor), items)
            )
        start_index = index
        anchor = current
    items = segment_intents[start_index:]
    if items:
        ranges.append(
            (
                items[0].time_sec,
                segment_end_sec,
                temporal_report_substate(anchor),
                items,
            )
        )

    merged: list[tuple[float, float, str, list[MasteringIntentDecision]]] = []
    for start_sec, end_sec, state, items in ranges:
        if merged and end_sec - start_sec < min_sec * 0.55:
            prev_start, _prev_end, prev_state, prev_items = merged[-1]
            merged[-1] = (prev_start, end_sec, prev_state, prev_items + items)
        else:
            merged.append((start_sec, end_sec, state, items))
    if len(merged) <= max_segments:
        return merged

    step = math.ceil(len(merged) / max_segments)
    capped: list[tuple[float, float, str, list[MasteringIntentDecision]]] = []
    for index in range(0, len(merged), step):
        group = merged[index : index + step]
        start_sec = group[0][0]
        end_sec = group[-1][1]
        items = [
            intent
            for _start, _end, _state, group_items in group
            for intent in group_items
        ]
        state_counts: dict[str, int] = {}
        for _start, _end, state, group_items in group:
            state_counts[state] = state_counts.get(state, 0) + len(group_items)
        state = max(state_counts.items(), key=lambda item: item[1])[0]
        capped.append((start_sec, end_sec, state, items))
    return capped


def temporal_segment_report_entry(
    *,
    start_sec: float,
    end_sec: float,
    state: str,
    segment_intents: list[MasteringIntentDecision],
    include_subsegments: bool,
) -> dict:
    principles = average_named_values(
        mastering_principles_to_dict(intent.principles) for intent in segment_intents
    )
    actions = average_named_values(intent.actions for intent in segment_intents)
    diagnostics = average_named_values(
        temporal_diagnostics_to_dict(intent.diagnostics) for intent in segment_intents
    )
    feature_opportunity = average_named_values(
        intent.feature_opportunity for intent in segment_intents
    )
    applied_solo_lift = average_named_values(
        intent.solo_lift for intent in segment_intents
    )
    intent_scores = average_named_values(
        intent.intent_scores for intent in segment_intents
    )
    primary_intent, secondary_intents = primary_secondary_intents(intent_scores)
    primary_reasons, secondary_reasons, suppressed_reasons = classify_segment_reasons(
        principles,
        actions,
        diagnostics,
        feature_opportunity,
        applied_solo_lift,
    )
    confidence = round(
        sum(intent.confidence for intent in segment_intents) / len(segment_intents), 4
    )
    if not primary_reasons:
        primary_reasons = [fallback_primary_reason(primary_intent, confidence)]
    entry = {
        "start": round(start_sec, 3),
        "end": round(end_sec, 3),
        "state": state,
        "macro_state": segment_intents[0].state if segment_intents else state,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
        "confidence": confidence,
        "intent_scores": intent_scores,
        "principles": principles,
        "actions": actions,
        "diagnostics": diagnostics,
        "applied": applied_render_actions(actions, applied_solo_lift),
        "feature_opportunity": feature_opportunity,
        "applied_solo_lift": applied_solo_lift,
        "primary_reasons": primary_reasons,
        "secondary_reasons": secondary_reasons,
        "suppressed_reasons": suppressed_reasons,
    }
    if include_subsegments and end_sec - start_sec >= 36.0:
        subsegments = temporal_subsegment_ranges(
            segment_intents,
            end_sec,
            min_sec=10.0,
            max_segments=16,
        )
        if len(subsegments) > 1:
            entry["subsegments"] = [
                temporal_segment_report_entry(
                    start_sec=sub_start,
                    end_sec=sub_end,
                    state=sub_state,
                    segment_intents=sub_items,
                    include_subsegments=False,
                )
                for sub_start, sub_end, sub_state, sub_items in subsegments
            ]
    return entry


def diagnostic_units() -> dict[str, str]:
    return {
        "momentary_lufs_estimate": "LUFS_estimate",
        "short_lufs_estimate": "LUFS_estimate",
        "local_loudness_range_db": "dB",
        "true_peak_dbtp_estimate": "dBTP_estimate",
        "crest_factor_db": "dB",
        "transient_score": "score_0_1",
        "punch_score": "score_0_1",
        "low_punch_score": "score_0_1",
        "mono_correlation": "correlation_minus1_1",
        "mono_fold_loss_db": "dB",
        "low_correlation": "correlation_minus1_1",
        "mid_correlation": "correlation_minus1_1",
        "high_correlation": "correlation_minus1_1",
        "phase_risk": "score_0_1",
        "peak_pressure": "score_0_1",
        "presence_harshness_score": "score_0_1",
        "sibilance_score": "score_0_1",
        "air_harshness_score": "score_0_1",
    }


def temporal_mastering_report_data(
    result: TemporalRemasterResult,
    *,
    validation_summary: dict | None = None,
    de_limiter_summary: dict | None = None,
) -> dict:
    lanes = result.lanes
    lane_summary = {}
    if lanes is not None:
        lane_summary = {
            "vocal_protect": {
                "peak": temporal_lane_peak(lanes.vocal_protect),
                "mean": temporal_lane_mean(lanes.vocal_protect),
            },
            "chorus_space": {
                "peak": temporal_lane_peak(lanes.chorus_space),
                "mean": temporal_lane_mean(lanes.chorus_space),
            },
            "breakdown_air": {
                "peak": temporal_lane_peak(lanes.breakdown_air),
                "mean": temporal_lane_mean(lanes.breakdown_air),
            },
            "low_tighten": {
                "peak": temporal_lane_peak(lanes.low_tighten),
                "mean": temporal_lane_mean(lanes.low_tighten),
            },
            "mid_decongest": {
                "peak": temporal_lane_peak(lanes.mid_decongest),
                "mean": temporal_lane_mean(lanes.mid_decongest),
            },
            "harshness_guard": {
                "peak": temporal_lane_peak(lanes.harshness_guard),
                "mean": temporal_lane_mean(lanes.harshness_guard),
            },
            "fold_safety": {
                "peak": temporal_lane_peak(lanes.fold_safety),
                "mean": temporal_lane_mean(lanes.fold_safety),
            },
        }

    segments = []
    sections = temporal_state_sections(result.frames, min_sec=6.0)
    for start_sec, end_sec, state in sections:
        segment_intents = [
            intent
            for intent in result.intents
            if start_sec <= intent.time_sec < end_sec
        ]
        if not segment_intents:
            continue
        segments.append(
            temporal_segment_report_entry(
                start_sec=start_sec,
                end_sec=end_sec,
                state=state,
                segment_intents=segment_intents,
                include_subsegments=True,
            )
        )

    return {
        "schema_version": 4,
        "mode": "temporal_mastering",
        "enabled": result.enabled,
        "reason": result.reason,
        "profile": result.profile,
        "strength": result.strength,
        "frames": len(result.frames),
        "lane_summary": lane_summary,
        "diagnostic_units": diagnostic_units(),
        "diagnostic_summary": temporal_diagnostic_summary(result.intents),
        "validation_summary": validation_summary or {},
        "de_limiter_summary": de_limiter_summary or {},
        "segments": segments,
    }


def write_temporal_mastering_report(
    path: Path,
    result: TemporalRemasterResult,
    *,
    validation_summary: dict | None = None,
    de_limiter_summary: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            temporal_mastering_report_data(
                result,
                validation_summary=validation_summary,
                de_limiter_summary=de_limiter_summary,
            ),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Temporal mastering report: {path}", flush=True)


__all__ = [
    "percentile_float",
    "window_relative_level",
    "smooth_automation_lane",
    "neutral_temporal_diagnostics",
    "build_mix_multiband_profile",
    "print_mix_multiband_profile",
    "read_stem_mix_audio",
    "temporal_window",
    "audio_rms_db",
    "audio_peak_db",
    "loudness_lufs_estimate",
    "band_filter_audio",
    "window_loudness_values",
    "window_rms_db_values",
    "window_peak_db_values",
    "stereo_correlation",
    "mono_fold_loss_db",
    "window_correlation_values",
    "window_mono_loss_values",
    "local_loudness_range",
    "compute_temporal_mix_diagnostics",
    "temporal_frame_state",
    "top_intent",
    "peak_pressure_guard_score",
    "decide_mastering_intent",
    "build_temporal_remaster",
    "temporal_lane_peak",
    "temporal_lane_mean",
    "stabilized_temporal_states",
    "temporal_state_sections",
    "print_temporal_remaster_result",
    "select_temporal_mastering_profile",
    "print_temporal_profile_selection",
    "mastering_principles_to_dict",
    "temporal_diagnostics_to_dict",
    "average_named_values",
    "temporal_intent_step_sec",
    "weighted_mean",
    "loudness_weights",
    "active_db_values",
    "temporal_diagnostic_summary",
    "peak_pressure_true_peak_guard",
    "ordered_named_scores",
    "primary_secondary_intents",
    "applied_render_actions",
    "classify_segment_reasons",
    "fallback_primary_reason",
    "temporal_report_substate",
    "should_split_temporal_subsegment",
    "temporal_subsegment_ranges",
    "temporal_segment_report_entry",
    "diagnostic_units",
    "temporal_mastering_report_data",
    "write_temporal_mastering_report",
]
