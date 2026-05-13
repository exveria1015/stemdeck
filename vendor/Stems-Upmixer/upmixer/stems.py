"""Stem discovery, stem analysis, and stem-level signal metrics."""

from __future__ import annotations

import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


from .models import (
    BAND_NAMES,
    BASS_DRUM_GUARD_RANGES,
    CHANNELS_714,
    MULTIBAND_RANGES,
    SILENT_STEM_ACTIVE_RATIO,
    SILENT_STEM_P90_DB,
    SILENT_STEM_RMS_DB,
    STEM_EXTENSIONS,
    StemStats,
    VolumeStats,
    WindowStats,
)
from .utils import clamp, linear_energy, volumedetect, volumedetect_complex

_STANDARD_RANGE_SPECS = tuple(
    dict.fromkeys(
        (str(name), float(low_hz), float(high_hz))
        for name, low_hz, high_hz in (*MULTIBAND_RANGES, *BASS_DRUM_GUARD_RANGES)
    )
)
_STANDARD_RANGE_KEYS = set(_STANDARD_RANGE_SPECS)


def resolve_stem_dir(job_dir: Path | None, stem_dir: Path | None) -> Path:
    if stem_dir is not None:
        return stem_dir.resolve()
    if job_dir is None:
        raise SystemExit("Either --job-dir or --stem-dir is required.")
    output_root = job_dir / "output"
    if not output_root.exists():
        raise SystemExit(f"Could not find output directory: {output_root}")
    children = sorted(path for path in output_root.iterdir() if path.is_dir())
    if len(children) == 1:
        return children[0].resolve()
    if not children:
        raise SystemExit(f"No stem directories found under {output_root}")
    names = "\n".join(f"  - {path}" for path in children)
    raise SystemExit(
        f"Multiple stem directories found. Pass --stem-dir explicitly:\n{names}"
    )


def resolve_input_file(
    job_dir: Path | None, explicit_input: Path | None
) -> Path | None:
    if explicit_input is not None:
        return explicit_input.resolve()
    if job_dir is None:
        return None
    input_root = job_dir / "input"
    if not input_root.exists():
        return None
    candidates = sorted(path for path in input_root.iterdir() if path.is_file())
    return candidates[0].resolve() if candidates else None


def find_stems(stem_dir: Path, requested: Iterable[str]) -> dict[str, Path]:
    stems: dict[str, Path] = {}
    for stem in requested:
        for ext in STEM_EXTENSIONS:
            candidate = stem_dir / f"{stem}{ext}"
            if candidate.exists():
                stems[stem] = candidate.resolve()
                break
    if not stems:
        raise SystemExit(f"No requested stems found in {stem_dir}")
    return stems


def analyze_stem_ffmpeg(ffmpeg: str, name: str, path: Path) -> StemStats:
    full = volumedetect(ffmpeg, path, "volumedetect")
    low = volumedetect(ffmpeg, path, "lowpass=f=180,volumedetect")
    mid = volumedetect(ffmpeg, path, "highpass=f=180,lowpass=f=2500,volumedetect")
    high = volumedetect(ffmpeg, path, "highpass=f=2500,volumedetect")
    mono = volumedetect_complex(
        ffmpeg,
        path,
        "[0:a]pan=mono|c0=0.5*FL+0.5*FR,volumedetect[mid]",
        "mid",
    )
    side = volumedetect_complex(
        ffmpeg,
        path,
        "[0:a]pan=mono|c0=0.5*FL-0.5*FR,volumedetect[side]",
        "side",
    )
    return StemStats(
        name=name,
        path=path,
        full=full,
        low=low,
        mid=mid,
        high=high,
        mono=mono,
        side=side,
    )


def analyze_stem_numpy(name: str, path: Path) -> StemStats:
    import numpy as np
    import soundfile as sf
    from scipy import signal

    audio_raw, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    def stats_from(data: AudioArray) -> VolumeStats:
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
        peak_db = 20.0 * math.log10(max(peak, 1e-12))
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        return VolumeStats(mean_db=rms_db, max_db=peak_db)

    def filtered_stats(kind: str, freq: float | tuple[float, float]) -> VolumeStats:
        sos = signal.butter(4, freq, btype=kind, fs=sample_rate, output="sos")
        filtered = cast(AudioArray, signal.sosfilt(sos, audio, axis=0))
        return stats_from(filtered)

    mono = 0.5 * (audio[:, 0] + audio[:, 1])
    side = 0.5 * (audio[:, 0] - audio[:, 1])
    return StemStats(
        name=name,
        path=path,
        full=stats_from(audio),
        low=filtered_stats("lowpass", 180.0),
        mid=filtered_stats("bandpass", (180.0, 2500.0)),
        high=filtered_stats("highpass", 2500.0),
        mono=stats_from(mono),
        side=stats_from(side),
    )


def analyze_stem(
    ffmpeg: str, name: str, path: Path, backend: str
) -> tuple[StemStats, str]:
    if backend in {"auto", "numpy"}:
        try:
            return analyze_stem_numpy(name, path), "numpy"
        except Exception as exc:
            if backend == "numpy":
                raise
            print(
                f"numpy analysis unavailable for {name}: {exc}. Falling back to ffmpeg.",
                file=sys.stderr,
            )
    return analyze_stem_ffmpeg(ffmpeg, name, path), "ffmpeg"


def fallback_window_stats(stats: StemStats) -> WindowStats:
    return WindowStats(
        name=stats.name,
        active_ratio=0.5,
        p10_db=stats.full.mean_db - 12.0,
        p50_db=stats.full.mean_db - 3.0,
        p90_db=stats.full.mean_db + 3.0,
        sustain_score=0.5,
        envelope_db=(),
    )


def analyze_window_stats_numpy(
    name: str,
    path: Path,
    *,
    window_ms: float,
    hop_ms: float,
) -> WindowStats:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    window = max(1, int(sample_rate * window_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    if len(audio) <= window:
        starts = np.asarray([0], dtype=np.int64)
        ends = np.asarray([len(audio)], dtype=np.int64)
    else:
        starts = np.arange(0, len(audio) - window + 1, hop, dtype=np.int64)
        ends = starts + window
    audio64 = np.asarray(audio, dtype=np.float64)
    sample_power = (
        np.mean(audio64 * audio64, axis=1)
        if len(audio64)
        else np.asarray([], dtype=np.float64)
    )
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power, dtype=np.float64)))
    counts = np.maximum(ends - starts, 1)
    mean_power = (cumulative[ends] - cumulative[starts]) / counts
    rms_values = np.sqrt(np.maximum(mean_power, 0.0))
    envelope = 20.0 * np.log10(np.maximum(rms_values, 1e-12))
    p10, p50, p90 = [float(np.percentile(envelope, value)) for value in (10, 50, 90)]
    active_threshold = max(p90 - 24.0, p50 - 9.0, -60.0)
    active_ratio = float(np.mean(envelope >= active_threshold))
    sustain_score = clamp(1.0 - ((p90 - p50) / 18.0), 0.0, 1.0)
    return WindowStats(
        name=name,
        active_ratio=active_ratio,
        p10_db=p10,
        p50_db=p50,
        p90_db=p90,
        sustain_score=sustain_score,
        envelope_db=tuple(float(value) for value in envelope),
    )


def analyze_window_stats(
    name: str,
    path: Path,
    stem_stats: StemStats,
    *,
    window_ms: float,
    hop_ms: float,
) -> WindowStats:
    try:
        return analyze_window_stats_numpy(
            name, path, window_ms=window_ms, hop_ms=hop_ms
        )
    except Exception as exc:
        print(
            f"window analysis unavailable for {name}: {exc}. Using summary stats.",
            file=sys.stderr,
        )
        return fallback_window_stats(stem_stats)


def silent_stem_reason(stats: StemStats, window: WindowStats | None) -> str | None:
    if window is None:
        return None
    if (
        stats.full.mean_db <= SILENT_STEM_RMS_DB
        and window.p90_db <= SILENT_STEM_P90_DB
        and window.active_ratio <= SILENT_STEM_ACTIVE_RATIO
    ):
        return (
            f"rms={stats.full.mean_db:.1f}dB "
            f"p90={window.p90_db:.1f}dB "
            f"active={window.active_ratio:.2f}"
        )
    return None


def filter_silent_stems(
    stems: dict[str, Path],
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    *,
    mode: str,
) -> tuple[dict[str, Path], list[StemStats], dict[str, WindowStats]]:
    if mode == "off":
        return stems, stats, windows

    kept_stats: list[StemStats] = []
    removed: list[tuple[str, str]] = []
    for item in stats:
        reason = silent_stem_reason(item, windows.get(item.name))
        if reason is None:
            kept_stats.append(item)
        else:
            removed.append((item.name, reason))

    if not removed:
        return stems, stats, windows
    if not kept_stats:
        raise SystemExit(
            "All requested stems appear silent; refusing to render an empty upmix."
        )

    for stem, reason in removed:
        print(f"Ignoring silent stem {stem}: {reason}", flush=True)

    kept_names = {item.name for item in kept_stats}
    filtered_stems = {name: path for name, path in stems.items() if name in kept_names}
    filtered_windows = {
        name: window for name, window in windows.items() if name in kept_names
    }
    return filtered_stems, kept_stats, filtered_windows


def envelope_correlation(left: WindowStats | None, right: WindowStats | None) -> float:
    if left is None or right is None or not left.envelope_db or not right.envelope_db:
        return 0.0
    count = min(len(left.envelope_db), len(right.envelope_db))
    if count < 3:
        return 0.0
    import numpy as np

    left_values = np.asarray(left.envelope_db[:count], dtype=np.float64)
    right_values = np.asarray(right.envelope_db[:count], dtype=np.float64)
    left_values -= float(np.mean(left_values))
    right_values -= float(np.mean(right_values))
    denom = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
    if denom <= 1e-12:
        return 0.0
    return clamp(float(np.dot(left_values, right_values) / denom), -1.0, 1.0)


def band_overlap_score(left: StemStats, right: StemStats) -> float:
    left_bands = band_fractions(left)
    right_bands = band_fractions(right)
    return clamp(
        sum(
            min(left_value, right_value)
            for left_value, right_value in zip(left_bands, right_bands)
        ),
        0.0,
        1.0,
    )


def activity_overlap_score(
    left: WindowStats | None, right: WindowStats | None
) -> float:
    if left is None or right is None:
        return 0.5
    return math.sqrt(max(left.active_ratio, 0.0) * max(right.active_ratio, 0.0))


def placement_collision_score(
    left: StemStats | None,
    right: StemStats | None,
    left_window: WindowStats | None,
    right_window: WindowStats | None,
) -> float:
    if left is None or right is None:
        return 0.0
    activity = activity_overlap_score(left_window, right_window)
    temporal = activity * (
        0.75 * max(0.0, envelope_correlation(left_window, right_window))
        + 0.25 * activity
    )
    spectral = band_overlap_score(left, right)
    width = math.sqrt(wide_score(left) * wide_score(right))
    spread = 0.35 + 0.65 * width
    return clamp(temporal * (0.60 * spectral + 0.40 * spread), 0.0, 1.0)


def vocal_leak_risk(
    stats: StemStats,
    vocals: StemStats | None,
    windows: WindowStats | None,
    vocal_windows: WindowStats | None,
) -> float:
    if vocals is None:
        return 0.0
    _low_frac, mid_frac, _high_frac = band_fractions(stats)
    narrow_mid = clamp(0.65 * mid_frac + 0.35 * (1.0 - wide_score(stats)), 0.0, 1.0)
    return placement_collision_score(stats, vocals, windows, vocal_windows) * narrow_mid


def pan_expr(values: dict[str, str]) -> str:
    parts = [f"{channel}={values.get(channel, '0*FL')}" for channel in CHANNELS_714]
    return "pan=7.1.4|" + "|".join(parts)


def wide_score(stats: StemStats) -> float:
    return clamp((stats.width_db + 14.0) / 14.0, 0.0, 1.0)


def bright_score(stats: StemStats) -> float:
    return clamp((stats.high_vs_mid_db + 13.0) / 10.0, 0.0, 1.0)


def low_light_score(stats: StemStats) -> float:
    return clamp((2.0 - stats.low_vs_mid_db) / 16.0, 0.0, 1.0)


def band_fractions(stats: StemStats) -> tuple[float, float, float]:
    low = linear_energy(stats.low.mean_db)
    mid = linear_energy(stats.mid.mean_db)
    high = linear_energy(stats.high.mean_db)
    total = max(low + mid + high, 1e-24)
    return low / total, mid / total, high / total


def stats_band_energies(stats: StemStats) -> dict[str, float]:
    return {
        "low": linear_energy(stats.low.mean_db),
        "mid": linear_energy(stats.mid.mean_db),
        "high": linear_energy(stats.high.mean_db),
    }


def stats_band_fraction_map(stats: StemStats) -> dict[str, float]:
    return dict(zip(BAND_NAMES, band_fractions(stats)))


def selected_overlap_bands(overlaps: dict[str, float]) -> tuple[str, ...]:
    if not overlaps:
        return ()
    max_overlap = max(overlaps.values())
    if max_overlap < 0.08:
        return ()
    selected = [
        band
        for band in BAND_NAMES
        if overlaps.get(band, 0.0) >= max(0.08, max_overlap * 0.64)
    ]
    if len(selected) == 3 and max_overlap < 0.34:
        selected = sorted(selected, key=lambda band: overlaps[band], reverse=True)[:2]
        selected.sort(key=lambda band: BAND_NAMES.index(band))
    return tuple(selected)


def stem_range_energies_numpy(
    stats: StemStats, ranges: tuple[tuple[str, float, float], ...]
) -> dict[str, float]:
    try:
        stat = stats.path.stat()
    except OSError:
        return dict(_stem_range_energies_uncached(stats.path, ranges))
    normalized_ranges = tuple(
        (str(name), float(low_hz), float(high_hz))
        for name, low_hz, high_hz in ranges
    )
    if all(item in _STANDARD_RANGE_KEYS for item in normalized_ranges):
        standard = dict(
            _stem_standard_range_energies_cached(
                str(stats.path),
                int(stat.st_mtime_ns),
                int(stat.st_size),
            )
        )
        return {name: standard.get(name, 0.0) for name, _low, _high in normalized_ranges}
    return dict(
        _stem_range_energies_cached(
            str(stats.path),
            int(stat.st_mtime_ns),
            int(stat.st_size),
            normalized_ranges,
        )
    )


@lru_cache(maxsize=128)
def _stem_standard_range_energies_cached(
    path_text: str,
    _mtime_ns: int,
    _size: int,
) -> tuple[tuple[str, float], ...]:
    return _stem_range_energies_uncached(Path(path_text), _STANDARD_RANGE_SPECS)


@lru_cache(maxsize=256)
def _stem_range_energies_cached(
    path_text: str,
    _mtime_ns: int,
    _size: int,
    ranges: tuple[tuple[str, float, float], ...],
) -> tuple[tuple[str, float], ...]:
    return _stem_range_energies_uncached(Path(path_text), ranges)


def _stem_range_energies_uncached(
    path: Path, ranges: tuple[tuple[str, float, float], ...]
) -> tuple[tuple[str, float], ...]:
    import numpy as np
    import soundfile as sf
    from scipy import signal

    audio_raw, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    audio = cast(AudioArray, audio_raw)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]

    energies: list[tuple[str, float]] = []
    for name, low_hz, high_hz in ranges:
        high = min(float(high_hz), sample_rate * 0.49)
        sos = signal.butter(
            4, (float(low_hz), high), btype="bandpass", fs=sample_rate, output="sos"
        )
        filtered = cast(AudioArray, signal.sosfilt(sos, audio, axis=0))
        energy = float(np.mean(filtered * filtered)) if filtered.size else 0.0
        energies.append((name, energy))
    return tuple(energies)


__all__ = [
    "resolve_stem_dir",
    "resolve_input_file",
    "find_stems",
    "analyze_stem_ffmpeg",
    "analyze_stem_numpy",
    "analyze_stem",
    "fallback_window_stats",
    "analyze_window_stats_numpy",
    "analyze_window_stats",
    "silent_stem_reason",
    "filter_silent_stems",
    "envelope_correlation",
    "band_overlap_score",
    "activity_overlap_score",
    "placement_collision_score",
    "vocal_leak_risk",
    "pan_expr",
    "wide_score",
    "bright_score",
    "low_light_score",
    "band_fractions",
    "stats_band_energies",
    "stats_band_fraction_map",
    "selected_overlap_bands",
    "stem_range_energies_numpy",
]
