"""Rendered-output validation and temporal feedback tuning."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from .mix_tuning import tonal_ratios
from .models import (
    TEMPORAL_FEEDBACK_WARN_MIN_LANE_DELTA,
    TEMPORAL_FEEDBACK_WATCH_MIN_LANE_DELTA,
    AutomationLanes,
    TemporalFeedbackResult,
    TemporalRemasterResult,
)
from .outputs import (
    _read_stereo_float,
    parse_loudnorm_json,
    render_51_fold_wav,
    render_51_to_stereo_fold_wav,
    render_stereo_fold_wav,
)
from .stems import analyze_stem_numpy, stats_band_energies
from .temporal import (
    audio_peak_db,
    audio_rms_db,
    band_filter_audio,
    loudness_lufs_estimate,
    mono_fold_loss_db,
    percentile_float,
    primary_secondary_intents,
    stereo_correlation,
    temporal_window,
)
from .utils import clamp, ffprobe_json, fmt, print_elapsed, run

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any

T = TypeVar("T")
R = TypeVar("R")


def validation_worker_count(item_count: int) -> int:
    if item_count <= 1:
        return 1
    raw = os.environ.get("STEMS_UPMIXER_VALIDATION_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
        return max(1, min(item_count, requested))
    return max(1, min(item_count, 4, os.cpu_count() or 1))


def map_validation_tasks(
    items: Iterable[T], worker: Callable[[T], R], *, max_workers: int
) -> list[R]:
    ordered = list(items)
    if max_workers <= 1 or len(ordered) <= 1:
        return [worker(item) for item in ordered]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, item) for item in ordered]
        return [future.result() for future in futures]


def measure_audio_loudness(
    ffmpeg: str,
    path: Path,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
) -> dict[str, float]:
    proc = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            "loudnorm="
            f"I={fmt(target_i)}:"
            f"TP={fmt(target_tp)}:"
            f"LRA={fmt(target_lra)}:"
            "print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    measurements = parse_loudnorm_json(proc.stderr or "")
    return {
        "integrated_lufs": round(float(measurements["input_i"]), 4),
        "true_peak_dbtp": round(float(measurements["input_tp"]), 4),
        "lra_lu": round(float(measurements["input_lra"]), 4),
        "threshold_lufs": round(float(measurements["input_thresh"]), 4),
    }


def coerce_loudness(loudness: Mapping[str, float]) -> dict[str, float]:
    return {
        "integrated_lufs": round(float(loudness["integrated_lufs"]), 4),
        "true_peak_dbtp": round(float(loudness["true_peak_dbtp"]), 4),
        "lra_lu": round(float(loudness["lra_lu"]), 4),
        "threshold_lufs": round(float(loudness["threshold_lufs"]), 4),
    }


def media_stream_summary(ffprobe: str, path: Path) -> dict:
    try:
        data = ffprobe_json(ffprobe, path)
    except Exception as exc:
        return {"path": str(path), "probe_error": str(exc)}
    audio_stream = next(
        (
            stream
            for stream in data.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        {},
    )
    duration = (data.get("format") or {}).get("duration") or audio_stream.get(
        "duration"
    )
    return {
        "path": str(path),
        "codec": audio_stream.get("codec_name", ""),
        "sample_rate": int(audio_stream["sample_rate"])
        if str(audio_stream.get("sample_rate", "")).isdigit()
        else None,
        "channels": audio_stream.get("channels"),
        "channel_layout": audio_stream.get("channel_layout", ""),
        "duration_sec": round(float(duration), 4) if duration is not None else None,
    }


def validation_loudness_status(
    loudness: dict[str, float], target_i: float, target_tp: float
) -> tuple[str, list[str]]:
    warnings: list[str] = []
    integrated = loudness.get("integrated_lufs", -120.0)
    true_peak = loudness.get("true_peak_dbtp", -120.0)
    if true_peak > target_tp + 0.30:
        warnings.append("true peak exceeds target margin")
    if integrated > target_i + 1.00:
        warnings.append("integrated loudness is above musical target range")
    if integrated < target_i - 6.00:
        warnings.append("integrated loudness is far below target")
    return ("pass" if not warnings else "warn"), warnings


def stereo_validation_metrics(path: Path) -> dict:
    audio, _sample_rate = _read_stereo_float(path)
    stats = analyze_stem_numpy("validation", path)
    low_mid, high_mid = tonal_ratios(stats_band_energies(stats))
    correlation = stereo_correlation(audio)
    mono_loss = mono_fold_loss_db(audio)
    phase_risk = clamp(
        0.58 * ((-mono_loss - 1.5) / 7.0) + 0.42 * ((0.28 - correlation) / 0.78),
        0.0,
        1.0,
    )
    return {
        "rms_db": round(stats.full.mean_db, 4),
        "sample_peak_dbfs": round(stats.full.max_db, 4),
        "low_mid_db": round(low_mid, 4),
        "high_mid_db": round(high_mid, 4),
        "side_minus_mid_db": round(stats.width_db, 4),
        "stereo_correlation": round(correlation, 4),
        "mono_fold_loss_db": round(mono_loss, 4),
        "phase_risk": round(phase_risk, 4),
    }


def validation_translation_status(metrics: dict[str, float]) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if metrics.get("sample_peak_dbfs", -120.0) >= -0.10:
        warnings.append("stereo fold-down reaches 0 dBFS headroom")
    if metrics.get("phase_risk", 0.0) >= 0.40:
        warnings.append("phase risk is high after fold-down")
    if metrics.get("mono_fold_loss_db", 0.0) <= -4.5:
        warnings.append("mono fold loses more than 4.5 dB")
    if metrics.get("stereo_correlation", 1.0) < 0.10:
        warnings.append("stereo correlation is low")
    if metrics.get("side_minus_mid_db", -99.0) > -1.0:
        warnings.append("side energy is unusually high versus mono")
    return ("pass" if not warnings else "warn"), warnings


def temporal_validation_context(
    result: TemporalRemasterResult | None, time_sec: float
) -> dict:
    if result is None or not result.intents:
        return {}
    nearest = min(result.intents, key=lambda item: abs(item.time_sec - time_sec))
    primary_intent, secondary_intents = primary_secondary_intents(nearest.intent_scores)
    return {
        "state": nearest.state,
        "primary_intent": primary_intent,
        "secondary_intents": secondary_intents,
    }


def temporal_consistency_centers(
    duration_sec: float, window_sec: float, hop_sec: float
) -> tuple[float, ...]:
    if duration_sec <= 0.0:
        return ()
    start = max(0.0, window_sec * 0.5)
    if duration_sec <= window_sec:
        return (duration_sec * 0.5,)
    centers: list[float] = []
    current = start
    last = max(start, duration_sec - window_sec * 0.5)
    while current <= last + 1e-6:
        centers.append(current)
        current += hop_sec
    if centers and duration_sec - centers[-1] > hop_sec * 0.75:
        centers.append(duration_sec - window_sec * 0.5)
    return tuple(max(0.0, min(duration_sec, center)) for center in centers)


def window_side_minus_mid_db(data: AudioArray) -> float:
    import numpy as np

    if data.size == 0:
        return 0.0
    if len(data.shape) == 1 or data.shape[1] < 2:
        return -120.0
    mid = 0.5 * (data[:, 0] + data[:, 1])
    side = 0.5 * (data[:, 0] - data[:, 1])
    mid_rms = float(np.sqrt(np.mean(mid * mid))) if mid.size else 0.0
    side_rms = float(np.sqrt(np.mean(side * side))) if side.size else 0.0
    return 20.0 * math.log10(max(side_rms, 1e-12) / max(mid_rms, 1e-12))


def temporal_phase_risk(correlation: float, mono_loss_db: float) -> float:
    return clamp(
        0.58 * ((-mono_loss_db - 1.5) / 7.0) + 0.42 * ((0.28 - correlation) / 0.78),
        0.0,
        1.0,
    )


def temporal_consistency_rows(
    path: Path, *, window_sec: float, hop_sec: float
) -> list[dict]:

    audio, sample_rate = _read_stereo_float(path)
    duration_sec = len(audio) / max(sample_rate, 1)
    centers = temporal_consistency_centers(duration_sec, window_sec, hop_sec)
    if not centers:
        return []

    low_band = band_filter_audio(audio, sample_rate, 20.0, 180.0)
    mid_band = band_filter_audio(audio, sample_rate, 180.0, 2500.0)
    high_band = band_filter_audio(audio, sample_rate, 2500.0, 12000.0)
    rows: list[dict] = []
    for center in centers:
        frame = temporal_window(audio, sample_rate, center, window_sec)
        low = temporal_window(low_band, sample_rate, center, window_sec)
        mid = temporal_window(mid_band, sample_rate, center, window_sec)
        high = temporal_window(high_band, sample_rate, center, window_sec)
        low_db = audio_rms_db(low)
        mid_db = audio_rms_db(mid)
        high_db = audio_rms_db(high)
        correlation = stereo_correlation(frame)
        mono_loss = mono_fold_loss_db(frame)
        rows.append(
            {
                "time_sec": round(float(center), 3),
                "loudness_lufs_estimate": round(loudness_lufs_estimate(frame), 4),
                "sample_peak_dbfs": round(
                    audio_peak_db(frame, sample_rate, oversample=1), 4
                ),
                "low_mid_db": round(low_db - mid_db, 4),
                "high_mid_db": round(high_db - mid_db, 4),
                "side_minus_mid_db": round(window_side_minus_mid_db(frame), 4),
                "stereo_correlation": round(correlation, 4),
                "mono_fold_loss_db": round(mono_loss, 4),
                "phase_risk": round(temporal_phase_risk(correlation, mono_loss), 4),
            }
        )
    del low_band, mid_band, high_band
    return rows


def temporal_row_values(rows: list[dict], key: str) -> list[float]:
    return [float(row.get(key, 0.0)) for row in rows]


def step_peak(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return max(
        abs(values[index] - values[index - 1]) for index in range(1, len(values))
    )


def step_p95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return percentile_float(
        [abs(values[index] - values[index - 1]) for index in range(1, len(values))],
        95.0,
    )


def temporal_trajectory_summary(rows: list[dict]) -> dict:
    if not rows:
        return {}
    loudness = temporal_row_values(rows, "loudness_lufs_estimate")
    low_mid = temporal_row_values(rows, "low_mid_db")
    high_mid = temporal_row_values(rows, "high_mid_db")
    width = temporal_row_values(rows, "side_minus_mid_db")
    correlation = temporal_row_values(rows, "stereo_correlation")
    mono_loss = temporal_row_values(rows, "mono_fold_loss_db")
    phase = temporal_row_values(rows, "phase_risk")
    peak = temporal_row_values(rows, "sample_peak_dbfs")
    return {
        "window_count": len(rows),
        "loudness_step_peak_db": round(step_peak(loudness), 4),
        "loudness_step_p95_db": round(step_p95(loudness), 4),
        "low_mid_step_peak_db": round(step_peak(low_mid), 4),
        "low_mid_step_p95_db": round(step_p95(low_mid), 4),
        "high_mid_step_peak_db": round(step_peak(high_mid), 4),
        "high_mid_step_p95_db": round(step_p95(high_mid), 4),
        "width_step_peak_db": round(step_peak(width), 4),
        "width_step_p95_db": round(step_p95(width), 4),
        "loudness_range_db": round(max(loudness) - min(loudness), 4),
        "sample_peak_max_dbfs": round(max(peak), 4),
        "stereo_correlation_min": round(min(correlation), 4),
        "mono_fold_loss_min_db": round(min(mono_loss), 4),
        "phase_risk_peak": round(max(phase), 4),
        "phase_risk_p95": round(percentile_float(phase, 95.0), 4),
    }


def temporal_shape_delta_summary(
    render_rows: list[dict], source_rows: list[dict]
) -> dict:
    if not render_rows or not source_rows:
        return {}
    import numpy as np

    count = min(len(render_rows), len(source_rows))
    metrics = {
        "loudness_lufs_estimate": "loudness_shape_delta",
        "low_mid_db": "low_mid_delta",
        "high_mid_db": "high_mid_delta",
        "side_minus_mid_db": "width_delta",
        "mono_fold_loss_db": "mono_fold_loss_delta",
    }
    summary: dict[str, float | str] = {}
    warning = False
    for source_key, out_key in metrics.items():
        render_values = np.asarray(
            temporal_row_values(render_rows[:count], source_key), dtype=np.float64
        )
        source_values = np.asarray(
            temporal_row_values(source_rows[:count], source_key), dtype=np.float64
        )
        delta = render_values - source_values
        delta -= float(np.median(delta))
        abs_delta = np.abs(delta)
        p95 = float(np.percentile(abs_delta, 95.0)) if abs_delta.size else 0.0
        peak = float(np.max(abs_delta)) if abs_delta.size else 0.0
        summary[f"{out_key}_p95_db"] = round(p95, 4)
        summary[f"{out_key}_peak_db"] = round(peak, 4)
        if p95 >= (2.2 if source_key == "loudness_lufs_estimate" else 2.6):
            warning = True
    summary["status"] = "warn" if warning else "pass"
    return summary


def temporal_row_float(row: dict | None, key: str, default: float) -> float:
    if row is None:
        return default
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def temporal_row_low_audibility(row: dict | None) -> bool:
    loudness = temporal_row_float(row, "loudness_lufs_estimate", -120.0)
    peak = temporal_row_float(row, "sample_peak_dbfs", -120.0)
    return loudness <= -42.0 and peak <= -30.0


def temporal_row_terminal_low_energy(index: int, rows: list[dict]) -> bool:
    if not rows or index < max(0, len(rows) - 2):
        return False
    loudness = temporal_row_float(rows[index], "loudness_lufs_estimate", -120.0)
    active_loudness = [
        temporal_row_float(row, "loudness_lufs_estimate", -120.0)
        for row in rows
        if temporal_row_float(row, "loudness_lufs_estimate", -120.0) > -70.0
    ]
    if not active_loudness:
        return True
    return loudness <= max(active_loudness) - 8.0 or loudness <= -38.0


def temporal_consistency_events(
    rows: list[dict],
    source_rows: list[dict],
    result: TemporalRemasterResult | None,
) -> list[dict]:
    if len(rows) < 2:
        return []
    source_count = len(source_rows)
    step_checks = [
        ("loudness_lufs_estimate", "loudness_jump", 2.4, 3.8, "delta_lufs"),
        ("low_mid_db", "low_balance_shift", 2.2, 3.6, "delta_db"),
        ("high_mid_db", "presence_balance_shift", 2.0, 3.4, "delta_db"),
        ("side_minus_mid_db", "width_shift", 2.8, 4.2, "delta_db"),
    ]
    events: list[dict] = []
    for index in range(1, len(rows)):
        start = float(rows[index - 1]["time_sec"])
        end = float(rows[index]["time_sec"])
        for key, event_type, watch_threshold, warn_threshold, delta_name in step_checks:
            delta = float(rows[index].get(key, 0.0)) - float(
                rows[index - 1].get(key, 0.0)
            )
            source_delta = 0.0
            if source_count > index:
                source_delta = float(source_rows[index].get(key, 0.0)) - float(
                    source_rows[index - 1].get(key, 0.0)
                )
            excess = abs(delta) - abs(source_delta)
            if abs(delta) < watch_threshold or (source_rows and excess < 1.5):
                continue
            if source_rows:
                status = "warn" if excess >= 3.8 else "watch"
            else:
                status = "warn" if abs(delta) >= warn_threshold else "watch"
            events.append(
                {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "type": event_type,
                    delta_name: round(delta, 4),
                    "source_delta": round(source_delta, 4) if source_rows else None,
                    "excess_over_source": round(excess, 4) if source_rows else None,
                    "status": status,
                    "context": temporal_validation_context(result, end),
                }
            )
    for index, row in enumerate(rows):
        phase = temporal_row_float(row, "phase_risk", 0.0)
        mono_loss = temporal_row_float(row, "mono_fold_loss_db", 0.0)
        correlation = temporal_row_float(row, "stereo_correlation", 1.0)
        if phase >= 0.42 or mono_loss <= -4.5 or correlation < 0.10:
            source_row = source_rows[index] if source_count > index else None
            if temporal_row_low_audibility(row) and (
                source_row is None or temporal_row_low_audibility(source_row)
            ):
                continue
            source_phase = temporal_row_float(source_row, "phase_risk", 0.0)
            source_mono_loss = temporal_row_float(source_row, "mono_fold_loss_db", 0.0)
            source_correlation = temporal_row_float(
                source_row, "stereo_correlation", 1.0
            )
            phase_excess = max(0.0, phase - source_phase)
            mono_loss_excess = max(0.0, source_mono_loss - mono_loss)
            correlation_excess = max(0.0, source_correlation - correlation)
            source_inherited = bool(
                source_row is not None
                and (
                    source_phase >= 0.38
                    or source_mono_loss <= -3.8
                    or source_correlation < 0.18
                )
                and phase_excess <= 0.18
                and mono_loss_excess <= 1.25
                and correlation_excess <= 0.20
            )
            terminal_low_energy = temporal_row_terminal_low_energy(index, rows)
            low_audibility = (
                temporal_row_float(row, "loudness_lufs_estimate", -120.0) <= -35.0
            )
            status = "warn" if phase >= 0.55 or mono_loss <= -6.0 else "watch"
            if source_inherited or low_audibility or terminal_low_energy:
                status = "watch"
            event = {
                "start": round(float(row["time_sec"]), 3),
                "end": round(float(row["time_sec"]), 3),
                "type": "terminal_translation_risk"
                if terminal_low_energy
                else "translation_risk",
                "phase_risk": round(phase, 4),
                "mono_fold_loss_db": round(mono_loss, 4),
                "stereo_correlation": round(correlation, 4),
                "loudness_lufs_estimate": round(
                    temporal_row_float(row, "loudness_lufs_estimate", -120.0),
                    4,
                ),
                "sample_peak_dbfs": round(
                    temporal_row_float(row, "sample_peak_dbfs", -120.0), 4
                ),
                "status": status,
                "context": temporal_validation_context(result, float(row["time_sec"])),
            }
            if source_row is not None:
                event.update(
                    {
                        "source_phase_risk": round(source_phase, 4),
                        "source_mono_fold_loss_db": round(source_mono_loss, 4),
                        "source_stereo_correlation": round(source_correlation, 4),
                        "render_added_phase_risk": round(phase_excess, 4),
                        "render_added_mono_loss_db": round(mono_loss_excess, 4),
                        "render_added_correlation_loss": round(correlation_excess, 4),
                    }
                )
            if source_inherited:
                event["source_inherited"] = True
                event["feedback_weight"] = 0.18 if terminal_low_energy else 0.35
            elif low_audibility:
                event["feedback_weight"] = 0.25
            elif terminal_low_energy:
                event["feedback_weight"] = 0.18
            if terminal_low_energy:
                event["audibility_context"] = "terminal_low_energy"
            events.append(event)
    events.sort(
        key=lambda item: (
            0 if item.get("status") == "warn" else 1,
            -abs(
                float(
                    item.get("delta_lufs")
                    or item.get("delta_db")
                    or item.get("phase_risk")
                    or 0.0
                )
            ),
        )
    )
    return events[:16]


def build_temporal_consistency_validation(
    *,
    checked_output: str,
    render_path: Path,
    source_path: Path | None,
    result: TemporalRemasterResult | None,
    window_sec: float = 8.0,
    hop_sec: float = 4.0,
) -> dict:
    try:
        render_rows = temporal_consistency_rows(
            render_path, window_sec=window_sec, hop_sec=hop_sec
        )
        source_rows = (
            temporal_consistency_rows(
                source_path, window_sec=window_sec, hop_sec=hop_sec
            )
            if source_path is not None and source_path.exists()
            else []
        )
        summary = temporal_trajectory_summary(render_rows)
        source_delta = temporal_shape_delta_summary(render_rows, source_rows)
        events = temporal_consistency_events(render_rows, source_rows, result)
        status = "pass"
        if source_delta.get("status") == "warn" or any(
            event.get("status") == "warn" for event in events
        ):
            status = "warn"
        elif events:
            status = "watch"
        return {
            "schema_version": 1,
            "status": status,
            "checked_output": checked_output,
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "trajectory_summary": summary,
            "source_render_delta": source_delta,
            "events": events,
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "checked_output": checked_output,
            "error": str(exc),
        }


def validate_audio_output(
    *,
    ffmpeg: str,
    ffprobe: str,
    path: Path,
    target_i: float,
    target_tp: float,
    target_lra: float,
    stereo_path: Path | None = None,
    precomputed_loudness: Mapping[str, float] | None = None,
) -> dict:
    if precomputed_loudness is not None:
        loudness = coerce_loudness(precomputed_loudness)
        loudness_source = "render_loudnorm"
    else:
        loudness = measure_audio_loudness(
            ffmpeg,
            path,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
        )
        loudness_source = "validation_measure"
    loudness_status, loudness_warnings = validation_loudness_status(
        loudness, target_i, target_tp
    )
    entry = {
        "file": media_stream_summary(ffprobe, path),
        "target": {
            "integrated_lufs": target_i,
            "true_peak_dbtp": target_tp,
            "lra_lu": target_lra,
        },
        "loudness": loudness,
        "loudness_source": loudness_source,
        "loudness_status": loudness_status,
        "warnings": list(loudness_warnings),
    }
    if stereo_path is not None and stereo_path.exists():
        try:
            metrics = stereo_validation_metrics(stereo_path)
            translation_status, translation_warnings = validation_translation_status(
                metrics
            )
            entry["translation"] = metrics
            entry["translation_status"] = translation_status
            entry["warnings"].extend(translation_warnings)
        except Exception as exc:
            entry["translation_status"] = "unavailable"
            entry["warnings"].append(
                f"stereo translation validation unavailable: {exc}"
            )
    entry["status"] = "pass" if not entry["warnings"] else "warn"
    return entry


def validation_overall_status(outputs: dict[str, dict]) -> str:
    if not outputs:
        return "skipped"
    if any(output.get("status") == "warn" for output in outputs.values()):
        return "warn"
    if all(output.get("status") == "pass" for output in outputs.values()):
        return "pass"
    return "partial"


def validation_summary_status(
    outputs: dict[str, dict], temporal_consistency: dict | None
) -> str:
    base = validation_overall_status(outputs)
    temporal_status = (temporal_consistency or {}).get("status")
    if temporal_status == "warn":
        return "warn"
    if base == "pass" and temporal_status == "watch":
        return "watch"
    return base


def build_mastering_validation_summary(
    *,
    ffmpeg: str,
    ffprobe: str,
    output_dir: Path,
    source_path: Path | None,
    bed_path: Path,
    stereo_path: Path,
    surround_path: Path,
    apple_path: Path,
    render_stereo: bool,
    render_surround: bool,
    render_apple_tv: bool,
    stereo_target_i: float,
    stereo_target_tp: float,
    stereo_target_lra: float,
    surround_target_i: float,
    surround_target_tp: float,
    surround_target_lra: float,
    stereo_lfe_fold_gain: float,
    surround_lfe_fold_gain: float,
    automation_lanes: AutomationLanes | None,
    temporal_result: TemporalRemasterResult | None,
    precomputed_loudness: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    outputs: dict[str, dict] = {}
    temporal_consistency: dict = {}
    precomputed_loudness = precomputed_loudness or {}
    temporal_candidate: tuple[str, Path] | None = None
    temp_dir = output_dir / ".validation_tmp"
    temp_paths: list[Path] = []
    validation_tasks: list[tuple[str, Callable[[], dict]]] = []
    surround_key = (
        "surround_flac" if surround_path.suffix.lower() == ".flac" else "surround_wav"
    )
    timer = time.perf_counter()
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        has_actual_output = (
            stereo_path.exists() or surround_path.exists() or apple_path.exists()
        )
        if stereo_path.exists():
            validation_tasks.append(
                (
                    "stereo_flac",
                    lambda path=stereo_path: validate_audio_output(
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        path=path,
                        target_i=stereo_target_i,
                        target_tp=stereo_target_tp,
                        target_lra=stereo_target_lra,
                        stereo_path=path,
                        precomputed_loudness=precomputed_loudness.get("stereo_flac"),
                    ),
                )
            )
            temporal_candidate = ("stereo_flac", stereo_path)
        elif bed_path.exists() and (render_stereo or not has_actual_output):
            preview = temp_dir / "bed_stereo_fold_validation.wav"
            render_stereo_fold_wav(
                ffmpeg, bed_path, preview, stereo_lfe_fold_gain, automation_lanes
            )
            temp_paths.append(preview)
            validation_tasks.append(
                (
                    "bed_stereo_fold_preview",
                    lambda path=preview: validate_audio_output(
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        path=path,
                        target_i=stereo_target_i,
                        target_tp=stereo_target_tp,
                        target_lra=stereo_target_lra,
                        stereo_path=path,
                    ),
                )
            )
            temporal_candidate = ("bed_stereo_fold_preview", preview)

        if surround_path.exists():
            folded = temp_dir / f"{surround_key}_stereo_fold_validation.wav"
            render_51_to_stereo_fold_wav(
                ffmpeg, surround_path, folded, stereo_lfe_fold_gain
            )
            temp_paths.append(folded)
            validation_tasks.append(
                (
                    surround_key,
                    lambda path=surround_path, stereo=folded: validate_audio_output(
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        path=path,
                        target_i=surround_target_i,
                        target_tp=surround_target_tp,
                        target_lra=surround_target_lra,
                        stereo_path=stereo,
                        precomputed_loudness=precomputed_loudness.get(surround_key)
                        or precomputed_loudness.get("surround_flac"),
                    ),
                )
            )
            if temporal_candidate is None:
                temporal_candidate = (f"{surround_key}_stereo_fold", folded)
        elif bed_path.exists() and (render_surround or not has_actual_output):
            preview_51 = temp_dir / "bed_5_1_fold_validation.wav"
            preview_stereo = temp_dir / "bed_5_1_stereo_fold_validation.wav"
            render_51_fold_wav(ffmpeg, bed_path, preview_51, surround_lfe_fold_gain)
            render_51_to_stereo_fold_wav(
                ffmpeg, preview_51, preview_stereo, stereo_lfe_fold_gain
            )
            temp_paths.extend([preview_51, preview_stereo])
            validation_tasks.append(
                (
                    "bed_5_1_fold_preview",
                    lambda path=preview_51, stereo=preview_stereo: validate_audio_output(
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        path=path,
                        target_i=surround_target_i,
                        target_tp=surround_target_tp,
                        target_lra=surround_target_lra,
                        stereo_path=stereo,
                        precomputed_loudness=precomputed_loudness.get("apple_tv"),
                    ),
                )
            )
            if temporal_candidate is None:
                temporal_candidate = ("bed_5_1_stereo_fold_preview", preview_stereo)

        if apple_path.exists():
            folded = temp_dir / "apple_tv_stereo_fold_validation.wav"
            render_51_to_stereo_fold_wav(
                ffmpeg, apple_path, folded, stereo_lfe_fold_gain
            )
            temp_paths.append(folded)
            validation_tasks.append(
                (
                    "apple_tv",
                    lambda path=apple_path, stereo=folded: validate_audio_output(
                        ffmpeg=ffmpeg,
                        ffprobe=ffprobe,
                        path=path,
                        target_i=surround_target_i,
                        target_tp=surround_target_tp,
                        target_lra=surround_target_lra,
                        stereo_path=stereo,
                    ),
                )
            )
            if temporal_candidate is None:
                temporal_candidate = ("apple_tv_stereo_fold", folded)

        parallel_tasks: list[tuple[str, str, Callable[[], dict]]] = [
            ("output", name, task) for name, task in validation_tasks
        ]
        if temporal_candidate is not None:
            parallel_tasks.append(
                (
                    "temporal",
                    "temporal_consistency",
                    lambda candidate=temporal_candidate: build_temporal_consistency_validation(
                        checked_output=candidate[0],
                        render_path=candidate[1],
                        source_path=source_path,
                        result=temporal_result,
                    ),
                )
            )

        workers = validation_worker_count(len(parallel_tasks))
        if workers > 1:
            names = ", ".join(name for _kind, name, _task in parallel_tasks)
            print(f"Validation workers: {workers} ({names})", flush=True)
        for kind, name, result in map_validation_tasks(
            parallel_tasks,
            lambda item: (item[0], item[1], item[2]()),
            max_workers=workers,
        ):
            if kind == "temporal":
                temporal_consistency = result
            else:
                outputs[name] = result
    except Exception as exc:
        return {
            "schema_version": 1,
            "status": "unavailable",
            "error": str(exc),
            "outputs": outputs,
            "temporal_consistency": temporal_consistency,
        }
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        print_elapsed("validation summary", timer)

    return {
        "schema_version": 1,
        "status": validation_summary_status(outputs, temporal_consistency),
        "outputs": outputs,
        "temporal_consistency": temporal_consistency,
    }


def print_validation_summary(summary: dict) -> None:
    if not summary:
        return
    print(f"Validation summary: {summary.get('status', 'unknown')}", flush=True)
    for name, output in (summary.get("outputs") or {}).items():
        loudness = output.get("loudness") or {}
        translation = output.get("translation") or {}
        text = (
            f"  {name}: {output.get('status', 'unknown')} "
            f"I={loudness.get('integrated_lufs', 0.0):+.2f}LUFS "
            f"TP={loudness.get('true_peak_dbtp', 0.0):+.2f}dBTP"
        )
        if translation:
            text += (
                f" corr={translation.get('stereo_correlation', 0.0):+.2f} "
                f"monoLoss={translation.get('mono_fold_loss_db', 0.0):+.2f}dB "
                f"phase={translation.get('phase_risk', 0.0):.2f}"
            )
        warnings = output.get("warnings") or []
        if warnings:
            text += f" warnings={len(warnings)}"
        print(text, flush=True)
    temporal = summary.get("temporal_consistency") or {}
    if temporal:
        trajectory = temporal.get("trajectory_summary") or {}
        source_delta = temporal.get("source_render_delta") or {}
        text = (
            f"  temporal_consistency: {temporal.get('status', 'unknown')} "
            f"checked={temporal.get('checked_output', '')} "
            f"loudStep={trajectory.get('loudness_step_peak_db', 0.0):.2f}dB "
            f"lowStep={trajectory.get('low_mid_step_peak_db', 0.0):.2f}dB "
            f"widthStep={trajectory.get('width_step_peak_db', 0.0):.2f}dB "
            f"phasePeak={trajectory.get('phase_risk_peak', 0.0):.2f}"
        )
        if source_delta:
            text += f" sourceDelta={source_delta.get('status', 'unknown')}"
        events = temporal.get("events") or []
        if events:
            text += f" events={len(events)}"
        print(text, flush=True)


def temporal_feedback_event_amount(
    event: dict, *, strength: float, base: float
) -> float:
    status_scale = 1.0 if event.get("status") == "warn" else 0.62
    delta = abs(
        float(
            event.get("excess_over_source")
            or event.get("delta_db")
            or event.get("delta_lufs")
            or 0.0
        )
    )
    try:
        feedback_weight = float(event.get("feedback_weight", 1.0))
    except (TypeError, ValueError):
        feedback_weight = 1.0
    return clamp(
        float(strength)
        * base
        * status_scale
        * clamp(feedback_weight, 0.0, 1.0)
        * (1.0 + min(delta, 6.0) / 12.0),
        0.0,
        0.32,
    )


def temporal_feedback_profile(
    times: tuple[float, ...],
    events: list[dict],
    event_types: set[str],
    *,
    strength: float,
    base: float,
    radius_sec: float,
) -> tuple[float, ...]:
    if not times or not events:
        return tuple(0.0 for _time in times)
    profile = [0.0 for _time in times]
    radius = max(1.0, float(radius_sec))
    for event in events:
        if event.get("type") not in event_types:
            continue
        amount = temporal_feedback_event_amount(event, strength=strength, base=base)
        center = 0.5 * (float(event.get("start", 0.0)) + float(event.get("end", 0.0)))
        for index, time_sec in enumerate(times):
            distance = abs(float(time_sec) - center)
            if distance <= radius:
                profile[index] = max(profile[index], amount * (1.0 - distance / radius))
    return tuple(clamp(value, 0.0, 0.40) for value in profile)


def dampen_temporal_lane(
    values: tuple[float, ...], profile: tuple[float, ...]
) -> tuple[float, ...]:
    if not values or not profile:
        return values
    count = min(len(values), len(profile))
    damped = [
        clamp(values[index] * (1.0 - profile[index]), 0.0, 1.0)
        for index in range(count)
    ]
    damped.extend(values[count:])
    return tuple(damped)


def boost_temporal_lane(
    values: tuple[float, ...], profile: tuple[float, ...]
) -> tuple[float, ...]:
    if not values or not profile:
        return values
    count = min(len(values), len(profile))
    boosted = [
        clamp(values[index] + (1.0 - values[index]) * profile[index], 0.0, 1.0)
        for index in range(count)
    ]
    boosted.extend(values[count:])
    return tuple(boosted)


def lane_max_delta(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    count = min(len(left), len(right))
    if count <= 0:
        return 0.0
    return max(abs(left[index] - right[index]) for index in range(count))


def tune_temporal_feedback(
    lanes: AutomationLanes | None,
    validation_summary: dict,
    *,
    strength: float,
) -> TemporalFeedbackResult:
    if lanes is None:
        return TemporalFeedbackResult(False, None, reason="no automation lanes")
    temporal = validation_summary.get("temporal_consistency") or {}
    events = list(temporal.get("events") or [])
    if not events or temporal.get("status") not in {"watch", "warn"}:
        return TemporalFeedbackResult(
            False, lanes, reason="no temporal feedback events"
        )

    width_types = {"width_shift", "translation_risk"}
    low_types = {"low_balance_shift", "loudness_jump"}
    presence_types = {"presence_balance_shift", "translation_risk"}
    loudness_types = {"loudness_jump"}
    motion_types = width_types | low_types | presence_types | loudness_types
    translation_types = {"translation_risk", "terminal_translation_risk"}

    strength = clamp(float(strength), 0.0, 1.0)
    width_profile = temporal_feedback_profile(
        lanes.times, events, width_types, strength=strength, base=0.18, radius_sec=18.0
    )
    low_profile = temporal_feedback_profile(
        lanes.times, events, low_types, strength=strength, base=0.14, radius_sec=14.0
    )
    presence_profile = temporal_feedback_profile(
        lanes.times,
        events,
        presence_types,
        strength=strength,
        base=0.16,
        radius_sec=14.0,
    )
    motion_profile = temporal_feedback_profile(
        lanes.times, events, motion_types, strength=strength, base=0.12, radius_sec=16.0
    )
    translation_profile = temporal_feedback_profile(
        lanes.times,
        events,
        translation_types | width_types,
        strength=strength,
        base=0.10,
        radius_sec=18.0,
    )

    chorus_space = dampen_temporal_lane(
        dampen_temporal_lane(lanes.chorus_space, width_profile), motion_profile
    )
    breakdown_air = dampen_temporal_lane(
        dampen_temporal_lane(lanes.breakdown_air, presence_profile), motion_profile
    )
    low_tighten = dampen_temporal_lane(lanes.low_tighten, low_profile)
    mid_decongest = dampen_temporal_lane(lanes.mid_decongest, presence_profile)
    harsh_profile = temporal_feedback_profile(
        lanes.times,
        events,
        presence_types,
        strength=strength,
        base=0.06,
        radius_sec=12.0,
    )
    harshness_guard = dampen_temporal_lane(lanes.harshness_guard, harsh_profile)
    fold_safety = boost_temporal_lane(lanes.fold_safety, translation_profile)
    solo_feature = {
        stem: dampen_temporal_lane(values, motion_profile)
        for stem, values in lanes.solo_feature.items()
    }

    new_lanes = AutomationLanes(
        times=lanes.times,
        step_sec=lanes.step_sec,
        vocal_protect=lanes.vocal_protect,
        chorus_space=chorus_space,
        breakdown_air=breakdown_air,
        low_anchor=lanes.low_anchor,
        low_tighten=low_tighten,
        mid_decongest=mid_decongest,
        harshness_guard=harshness_guard,
        fold_safety=fold_safety,
        solo_feature=solo_feature,
    )
    max_delta = max(
        lane_max_delta(lanes.chorus_space, new_lanes.chorus_space),
        lane_max_delta(lanes.breakdown_air, new_lanes.breakdown_air),
        lane_max_delta(lanes.low_tighten, new_lanes.low_tighten),
        lane_max_delta(lanes.mid_decongest, new_lanes.mid_decongest),
        lane_max_delta(lanes.harshness_guard, new_lanes.harshness_guard),
        lane_max_delta(lanes.fold_safety, new_lanes.fold_safety),
        max(
            (
                lane_max_delta(lanes.solo_feature.get(stem, ()), values)
                for stem, values in new_lanes.solo_feature.items()
            ),
            default=0.0,
        ),
    )
    has_warn_event = temporal.get("status") == "warn" or any(
        event.get("status") == "warn" for event in events
    )
    min_delta = (
        TEMPORAL_FEEDBACK_WARN_MIN_LANE_DELTA
        if has_warn_event
        else TEMPORAL_FEEDBACK_WATCH_MIN_LANE_DELTA
    )
    changed = max_delta >= min_delta
    return TemporalFeedbackResult(
        changed,
        new_lanes if changed else lanes,
        event_count=len(events),
        width_events=sum(1 for event in events if event.get("type") == "width_shift"),
        low_events=sum(
            1 for event in events if event.get("type") == "low_balance_shift"
        ),
        presence_events=sum(
            1 for event in events if event.get("type") == "presence_balance_shift"
        ),
        loudness_events=sum(
            1 for event in events if event.get("type") == "loudness_jump"
        ),
        translation_events=sum(
            1 for event in events if event.get("type") in translation_types
        ),
        max_damping=max_delta,
        reason=temporal.get("status", "unknown"),
    )


def print_temporal_feedback_result(
    result: TemporalFeedbackResult, pass_index: int
) -> None:
    if not result.changed:
        return
    print(
        f"Temporal feedback pass {pass_index}: "
        f"events={result.event_count} "
        f"width={result.width_events} "
        f"low={result.low_events} "
        f"presence={result.presence_events} "
        f"loudness={result.loudness_events} "
        f"translation={result.translation_events} "
        f"max_lane_delta={result.max_damping:.3f}",
        flush=True,
    )


__all__ = [
    "measure_audio_loudness",
    "media_stream_summary",
    "validation_loudness_status",
    "stereo_validation_metrics",
    "validation_translation_status",
    "temporal_validation_context",
    "temporal_consistency_centers",
    "window_side_minus_mid_db",
    "temporal_phase_risk",
    "temporal_consistency_rows",
    "temporal_row_values",
    "step_peak",
    "step_p95",
    "temporal_trajectory_summary",
    "temporal_shape_delta_summary",
    "temporal_row_float",
    "temporal_row_low_audibility",
    "temporal_row_terminal_low_energy",
    "temporal_consistency_events",
    "build_temporal_consistency_validation",
    "validate_audio_output",
    "validation_overall_status",
    "validation_summary_status",
    "build_mastering_validation_summary",
    "print_validation_summary",
    "temporal_feedback_event_amount",
    "temporal_feedback_profile",
    "dampen_temporal_lane",
    "boost_temporal_lane",
    "lane_max_delta",
    "tune_temporal_feedback",
    "print_temporal_feedback_result",
]
