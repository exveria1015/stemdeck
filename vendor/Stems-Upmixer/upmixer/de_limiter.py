"""Stem de-limiter metrics, planning, guard checks, and reports."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import (
    DEFAULT_STEM_DE_LIMITER_MIX,
    DEFAULT_STEM_DE_LIMITER_REMASTER_MIX,
    DEFAULT_STEM_DE_LIMITER_REPAIR_MIX,
    STEM_ORDER,
    StemDeLimiterGuardResult,
    StemDeLimiterMetrics,
    StemDeLimiterPlan,
    StemDeLimiterStemDecision,
    StemStats,
    TemporalDiagnosticFrame,
    VolumeStats,
)
from .temporal import (
    audio_peak_db,
    audio_rms_db,
    compute_temporal_mix_diagnostics,
    loudness_weights,
    mono_fold_loss_db,
    percentile_float,
    read_stem_mix_audio,
    temporal_diagnostics_to_dict,
    weighted_mean,
)
from .utils import clamp, read_audio_float

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


def placeholder_stem_stats(name: str, path: Path) -> StemStats:
    silent = VolumeStats(-120.0, -120.0)
    return StemStats(
        name=name,
        path=path,
        full=silent,
        low=silent,
        mid=silent,
        high=silent,
        mono=silent,
        side=silent,
    )


def stem_metric_stats_from_paths(paths: dict[str, Path]) -> list[StemStats]:
    return [placeholder_stem_stats(name, path) for name, path in paths.items()]


def crest_100ms_p95_db(data: AudioArray, sample_rate: int) -> float:
    import numpy as np

    window = max(1, int(round(sample_rate * 0.100)))
    hop = max(1, int(round(sample_rate * 0.050)))
    if len(data) < window:
        rms_db = audio_rms_db(data)
        return (
            audio_peak_db(data, sample_rate, oversample=1) - rms_db
            if rms_db > -100.0
            else 0.0
        )
    data64 = np.asarray(data, dtype=np.float64)
    if data64.ndim == 1:
        sample_power = data64 * data64
        sample_peak = np.abs(data64)
    else:
        sample_power = np.mean(data64 * data64, axis=1)
        sample_peak = np.max(np.abs(data64), axis=1)
    starts = np.arange(0, len(data64) - window + 1, hop, dtype=np.int64)
    ends = starts + window
    cumulative = np.concatenate(([0.0], np.cumsum(sample_power, dtype=np.float64)))
    rms = np.sqrt(np.maximum((cumulative[ends] - cumulative[starts]) / window, 0.0))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    values = [
        float(20.0 * math.log10(max(float(np.max(sample_peak[start:end])), 1e-12)) - rms_value)
        for start, end, rms_value in zip(starts, ends, rms_db, strict=False)
        if rms_value > -100.0
    ]
    return percentile_float(values, 95.0) if values else 0.0


def sample_peak_density_pct(data: AudioArray, threshold: float) -> float:
    import numpy as np

    if not getattr(data, "size", 0):
        return 0.0
    return float(np.mean(np.abs(data) > float(threshold)) * 100.0)


def diagnostic_score_summary(
    diagnostics: tuple[TemporalDiagnosticFrame, ...],
    key: str,
) -> dict[str, float]:
    if not diagnostics:
        return {"loudness_weighted_mean": 0.0, "p95": 0.0, "peak": 0.0}
    rows = [temporal_diagnostics_to_dict(item) for item in diagnostics]
    weights = loudness_weights(rows)
    values = [float(row.get(key, 0.0)) for row in rows]
    return {
        "loudness_weighted_mean": weighted_mean(values, weights),
        "p95": percentile_float(values, 95.0),
        "peak": max(values) if values else 0.0,
    }


def _normalized_stereo_audio(path: Path) -> tuple[AudioArray, int]:
    import numpy as np

    audio_raw, sample_rate = read_audio_float(path, dtype="float32")
    audio = np.asarray(audio_raw, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return audio, int(sample_rate)


def _mix_cached_audio(items: list[tuple[str, AudioArray, int]]) -> tuple[AudioArray, int]:
    import numpy as np
    from scipy import signal

    mix: AudioArray | None = None
    mix_rate: int | None = None
    for _name, audio_raw, sample_rate in items:
        audio = audio_raw
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
            audio = resampled.astype(np.float32, copy=False)
        if mix is None:
            mix = np.zeros((len(audio), 2), dtype=np.float32)
        elif len(audio) > len(mix):
            extended = np.zeros((len(audio), 2), dtype=np.float32)
            extended[: len(mix)] = mix
            mix = extended
        mix[: len(audio)] += audio[:, :2]
    if mix is None or mix_rate is None:
        raise RuntimeError("no stems available for stem de-limiter metrics")
    return mix, mix_rate


def _metrics_from_mix(
    label: str,
    paths: dict[str, Path],
    mix: AudioArray,
    sample_rate: int,
    *,
    hop_sec: float,
) -> StemDeLimiterMetrics:
    stats = stem_metric_stats_from_paths(paths)
    duration_sec = len(mix) / max(sample_rate, 1)
    frame_count = max(1, int(math.ceil(duration_sec / max(float(hop_sec), 0.05))))
    diagnostics = compute_temporal_mix_diagnostics(
        stats,
        frame_count=frame_count,
        hop_sec=float(hop_sec),
        mix_audio=mix,
        sample_rate=sample_rate,
    )
    transient = diagnostic_score_summary(diagnostics, "transient_score")
    punch = diagnostic_score_summary(diagnostics, "punch_score")
    presence = diagnostic_score_summary(diagnostics, "presence_harshness_score")
    sibilance = diagnostic_score_summary(diagnostics, "sibilance_score")
    phase = diagnostic_score_summary(diagnostics, "phase_risk")
    mono_loss_values = [item.mono_fold_loss_db for item in diagnostics]
    sample_peak = audio_peak_db(mix, sample_rate, oversample=1)
    true_peak = audio_peak_db(mix, sample_rate, oversample=4)
    rms_db = audio_rms_db(mix)
    mono_loss = mono_fold_loss_db(mix)
    return StemDeLimiterMetrics(
        label=label,
        true_peak_dbtp=true_peak,
        sample_peak_dbfs=sample_peak,
        rms_dbfs=rms_db,
        crest_factor_db=sample_peak - rms_db,
        crest_100ms_p95_db=crest_100ms_p95_db(mix, sample_rate),
        transient_lwmean=transient["loudness_weighted_mean"],
        transient_p95=transient["p95"],
        punch_lwmean=punch["loudness_weighted_mean"],
        punch_p95=punch["p95"],
        presence_harshness_lwmean=presence["loudness_weighted_mean"],
        presence_harshness_p95=presence["p95"],
        sibilance_lwmean=sibilance["loudness_weighted_mean"],
        sibilance_p95=sibilance["p95"],
        phase_risk_lwmean=phase["loudness_weighted_mean"],
        phase_risk_p95=phase["p95"],
        mono_fold_loss_db=mono_loss,
        mono_fold_loss_min_db=min(mono_loss_values) if mono_loss_values else mono_loss,
        peak_density_gt_095_pct=sample_peak_density_pct(mix, 0.95),
        peak_density_gt_090_pct=sample_peak_density_pct(mix, 0.90),
    )


def _local_screen_metrics_from_mix(
    label: str,
    mix: AudioArray,
    sample_rate: int,
) -> StemDeLimiterMetrics:
    sample_peak = audio_peak_db(mix, sample_rate, oversample=1)
    true_peak = audio_peak_db(mix, sample_rate, oversample=4)
    rms_db = audio_rms_db(mix)
    mono_loss = mono_fold_loss_db(mix)
    crest = sample_peak - rms_db
    return StemDeLimiterMetrics(
        label=label,
        true_peak_dbtp=true_peak,
        sample_peak_dbfs=sample_peak,
        rms_dbfs=rms_db,
        crest_factor_db=crest,
        crest_100ms_p95_db=crest,
        transient_lwmean=0.0,
        transient_p95=0.0,
        punch_lwmean=0.0,
        punch_p95=0.0,
        presence_harshness_lwmean=0.0,
        presence_harshness_p95=0.0,
        sibilance_lwmean=0.0,
        sibilance_p95=0.0,
        phase_risk_lwmean=0.0,
        phase_risk_p95=0.0,
        mono_fold_loss_db=mono_loss,
        mono_fold_loss_min_db=mono_loss,
        peak_density_gt_095_pct=sample_peak_density_pct(mix, 0.95),
        peak_density_gt_090_pct=sample_peak_density_pct(mix, 0.90),
    )


class StemDeLimiterMetricsCache:
    """Per-run audio and metrics cache for SGI candidate screening."""

    def __init__(self) -> None:
        self._audio: dict[Path, tuple[AudioArray, int]] = {}
        self._full_metrics: dict[
            tuple[float, tuple[tuple[str, str], ...]], StemDeLimiterMetrics
        ] = {}
        self._local_metrics: dict[
            tuple[tuple[tuple[str, str], ...]], StemDeLimiterMetrics
        ] = {}

    @staticmethod
    def _paths_key(paths: dict[str, Path]) -> tuple[tuple[str, str], ...]:
        return tuple((name, str(path.resolve())) for name, path in paths.items())

    def audio(self, path: Path) -> tuple[AudioArray, int]:
        resolved = path.resolve()
        cached = self._audio.get(resolved)
        if cached is None:
            cached = _normalized_stereo_audio(resolved)
            self._audio[resolved] = cached
        return cached

    def mix(self, paths: dict[str, Path]) -> tuple[AudioArray, int]:
        return _mix_cached_audio(
            [(name, *self.audio(path)) for name, path in paths.items()]
        )

    def full(
        self,
        label: str,
        paths: dict[str, Path],
        *,
        hop_sec: float = 4.0,
    ) -> StemDeLimiterMetrics:
        key = (round(float(hop_sec), 6), self._paths_key(paths))
        cached = self._full_metrics.get(key)
        if cached is None:
            mix, sample_rate = self.mix(paths)
            cached = _metrics_from_mix(
                label,
                paths,
                mix,
                sample_rate,
                hop_sec=float(hop_sec),
            )
            self._full_metrics[key] = cached
        return replace(cached, label=label)

    def local(
        self,
        label: str,
        paths: dict[str, Path],
    ) -> StemDeLimiterMetrics:
        key = (self._paths_key(paths),)
        cached = self._local_metrics.get(key)
        if cached is None:
            mix, sample_rate = self.mix(paths)
            cached = _local_screen_metrics_from_mix(label, mix, sample_rate)
            self._local_metrics[key] = cached
        return replace(cached, label=label)


def stem_de_limiter_metrics_from_paths(
    label: str,
    paths: dict[str, Path],
    *,
    hop_sec: float = 4.0,
    cache: StemDeLimiterMetricsCache | None = None,
) -> StemDeLimiterMetrics:
    if cache is not None:
        return cache.full(label, paths, hop_sec=hop_sec)
    stats = stem_metric_stats_from_paths(paths)
    mix, sample_rate = read_stem_mix_audio(stats)
    return _metrics_from_mix(label, paths, mix, sample_rate, hop_sec=hop_sec)


def stem_de_limiter_metrics_to_dict(
    metrics: StemDeLimiterMetrics | None,
) -> dict[str, float | str]:
    if metrics is None:
        return {}
    return {
        "label": metrics.label,
        "true_peak_dbtp": round(metrics.true_peak_dbtp, 4),
        "sample_peak_dbfs": round(metrics.sample_peak_dbfs, 4),
        "rms_dbfs": round(metrics.rms_dbfs, 4),
        "crest_factor_db": round(metrics.crest_factor_db, 4),
        "crest_100ms_p95_db": round(metrics.crest_100ms_p95_db, 4),
        "transient_lwmean": round(metrics.transient_lwmean, 4),
        "transient_p95": round(metrics.transient_p95, 4),
        "punch_lwmean": round(metrics.punch_lwmean, 4),
        "punch_p95": round(metrics.punch_p95, 4),
        "presence_harshness_lwmean": round(metrics.presence_harshness_lwmean, 4),
        "presence_harshness_p95": round(metrics.presence_harshness_p95, 4),
        "sibilance_lwmean": round(metrics.sibilance_lwmean, 4),
        "sibilance_p95": round(metrics.sibilance_p95, 4),
        "phase_risk_lwmean": round(metrics.phase_risk_lwmean, 4),
        "phase_risk_p95": round(metrics.phase_risk_p95, 4),
        "mono_fold_loss_db": round(metrics.mono_fold_loss_db, 4),
        "mono_fold_loss_min_db": round(metrics.mono_fold_loss_min_db, 4),
        "peak_density_gt_095_pct": round(metrics.peak_density_gt_095_pct, 4),
        "peak_density_gt_090_pct": round(metrics.peak_density_gt_090_pct, 4),
    }


def stem_de_limiter_metric_deltas(
    before: StemDeLimiterMetrics | None,
    after: StemDeLimiterMetrics | None,
) -> dict[str, float]:
    before_dict = stem_de_limiter_metrics_to_dict(before)
    after_dict = stem_de_limiter_metrics_to_dict(after)
    deltas: dict[str, float] = {}
    for key, after_value in after_dict.items():
        before_value = before_dict.get(key)
        if isinstance(before_value, (int, float)) and isinstance(
            after_value, (int, float)
        ):
            deltas[key] = round(float(after_value) - float(before_value), 4)
    return deltas


def stem_de_limiter_stem_decision_to_dict(decision: StemDeLimiterStemDecision) -> dict:
    return {
        "stem": decision.stem,
        "status": decision.status,
        "reasons": list(decision.reasons),
        "warnings": list(decision.warnings),
        "source_path": str(decision.source_path),
        "output_path": str(decision.output_path),
        "source_metrics": stem_de_limiter_metrics_to_dict(decision.source_metrics),
        "output_metrics": stem_de_limiter_metrics_to_dict(decision.output_metrics),
        "metric_deltas": stem_de_limiter_metric_deltas(
            decision.source_metrics, decision.output_metrics
        ),
    }


def stem_de_limiter_summary_data(
    plan: StemDeLimiterPlan | None,
    guard: StemDeLimiterGuardResult | None,
) -> dict:
    if plan is None:
        return {}
    accepted = None if guard is None else guard.accepted
    output_metrics = None if guard is None else guard.output_metrics
    if not plan.enabled:
        status = "disabled"
    elif accepted is True:
        status = "applied"
    elif accepted is False:
        status = "rejected"
    else:
        status = "not_run"
    stem_decisions = () if guard is None else guard.stem_decisions
    return {
        "requested_mode": plan.requested_mode,
        "selected_mode": plan.selected_mode,
        "status": status,
        "enabled": plan.enabled,
        "applied": status == "applied",
        "mix": round(plan.mix, 4),
        "reason": plan.reason,
        "checkpoint": "" if plan.checkpoint is None else str(plan.checkpoint),
        "accepted": accepted,
        "guard_reasons": [] if guard is None else list(guard.reasons),
        "guard_warnings": [] if guard is None else list(guard.warnings),
        "source_metrics": stem_de_limiter_metrics_to_dict(plan.source_metrics),
        "output_metrics": stem_de_limiter_metrics_to_dict(output_metrics),
        "metric_deltas": stem_de_limiter_metric_deltas(
            plan.source_metrics, output_metrics
        ),
        "applied_stems": [
            item.stem for item in stem_decisions if item.status == "applied"
        ],
        "rejected_stems": [
            item.stem for item in stem_decisions if item.status == "rejected"
        ],
        "skipped_stems": [
            item.stem for item in stem_decisions if item.status == "skipped"
        ],
        "stem_decisions": [
            stem_de_limiter_stem_decision_to_dict(item) for item in stem_decisions
        ],
    }


def choose_stem_de_limiter_plan(
    requested_mode: str,
    source_metrics: StemDeLimiterMetrics,
    checkpoint: Path | None,
    mix_override: float | None,
) -> StemDeLimiterPlan:
    requested = (
        requested_mode
        if requested_mode in {"off", "auto", "safe", "remaster", "repair", "sections2"}
        else "off"
    )
    if requested == "off":
        return StemDeLimiterPlan(
            requested, "off", False, 0.0, "disabled", checkpoint, source_metrics
        )
    if requested == "safe":
        return StemDeLimiterPlan(
            requested,
            "safe",
            False,
            0.0,
            "safe mode preserves closure",
            checkpoint,
            source_metrics,
        )
    if requested == "sections2":
        mix = (
            DEFAULT_STEM_DE_LIMITER_MIX if mix_override is None else float(mix_override)
        )
        return StemDeLimiterPlan(
            requested,
            "sections2",
            True,
            clamp(mix, 0.0, 1.0),
            "manual sections2 preset",
            checkpoint,
            source_metrics,
        )
    if requested == "remaster":
        mix = (
            DEFAULT_STEM_DE_LIMITER_REMASTER_MIX
            if mix_override is None
            else float(mix_override)
        )
        return StemDeLimiterPlan(
            requested,
            "remaster",
            True,
            clamp(mix, 0.0, 1.0),
            "manual remaster preset",
            checkpoint,
            source_metrics,
        )
    if requested == "repair":
        mix = (
            DEFAULT_STEM_DE_LIMITER_REPAIR_MIX
            if mix_override is None
            else float(mix_override)
        )
        return StemDeLimiterPlan(
            requested,
            "repair",
            True,
            clamp(mix, 0.0, 1.0),
            "manual repair preset",
            checkpoint,
            source_metrics,
        )

    artifact_guard = (
        source_metrics.presence_harshness_p95 >= 0.45
        or source_metrics.sibilance_p95 >= 0.45
        or source_metrics.phase_risk_p95 >= 0.45
    )
    brickwall_risk = (
        source_metrics.true_peak_dbtp >= -0.3
        and source_metrics.sample_peak_dbfs >= -0.5
        and source_metrics.crest_factor_db <= 9.5
    )
    dense_peak_risk = (
        source_metrics.peak_density_gt_095_pct >= 0.20
        or source_metrics.peak_density_gt_090_pct >= 0.75
    )
    moderate_risk = (
        source_metrics.true_peak_dbtp >= -1.0
        or source_metrics.sample_peak_dbfs >= -1.0
        or source_metrics.crest_factor_db <= 10.3
        or dense_peak_risk
    )
    if artifact_guard and not brickwall_risk:
        return StemDeLimiterPlan(
            requested,
            "safe",
            False,
            0.0,
            "artifact guard: harshness/sibilance/phase risk is already high",
            checkpoint,
            source_metrics,
        )
    if brickwall_risk and dense_peak_risk:
        mix = (
            DEFAULT_STEM_DE_LIMITER_REPAIR_MIX
            if mix_override is None
            else float(mix_override)
        )
        return StemDeLimiterPlan(
            requested,
            "repair",
            True,
            clamp(mix, 0.0, 1.0),
            "brickwall repair: high TP/sample peak, low crest, dense peaks",
            checkpoint,
            source_metrics,
        )
    if brickwall_risk or moderate_risk:
        default_mix = (
            DEFAULT_STEM_DE_LIMITER_REMASTER_MIX
            if source_metrics.true_peak_dbtp >= -0.3
            else 0.50
        )
        mix = default_mix if mix_override is None else float(mix_override)
        return StemDeLimiterPlan(
            requested,
            "remaster",
            True,
            clamp(mix, 0.0, 1.0),
            "remaster: mild peak pressure or reduced crest",
            checkpoint,
            source_metrics,
        )
    return StemDeLimiterPlan(
        requested,
        "safe",
        False,
        0.0,
        "safe: no brickwall recovery needed",
        checkpoint,
        source_metrics,
    )


def ordered_stem_names(stems: dict[str, Path]) -> list[str]:
    return sorted(
        stems,
        key=lambda name: (
            STEM_ORDER.index(name) if name in STEM_ORDER else len(STEM_ORDER),
            name,
        ),
    )


def stem_de_limiter_local_risk_reasons(
    stem: str, metrics: StemDeLimiterMetrics
) -> list[str]:
    reasons: list[str] = []
    if metrics.rms_dbfs <= -75.0:
        return reasons
    if metrics.true_peak_dbtp >= -0.50:
        reasons.append(f"local true peak is high at {metrics.true_peak_dbtp:+.2f} dBTP")
    if metrics.sample_peak_dbfs >= -1.00:
        reasons.append(
            f"local sample peak is high at {metrics.sample_peak_dbfs:+.2f} dBFS"
        )
    if metrics.peak_density_gt_095_pct >= 0.05:
        reasons.append(
            f"local peak density >0.95 is {metrics.peak_density_gt_095_pct:.3f}%"
        )
    if metrics.peak_density_gt_090_pct >= 0.20:
        reasons.append(
            f"local peak density >0.90 is {metrics.peak_density_gt_090_pct:.3f}%"
        )
    crest_floor = 8.5 if stem == "bass" else 9.5
    if metrics.crest_factor_db <= crest_floor:
        reasons.append(f"local crest is low at {metrics.crest_factor_db:.2f} dB")
    return reasons


def evaluate_stem_de_limiter_stem_decision(
    stem: str,
    before: StemDeLimiterMetrics,
    after: StemDeLimiterMetrics,
    *,
    source_path: Path,
    output_path: Path,
    selected_mode: str,
) -> StemDeLimiterStemDecision:
    warnings: list[str] = []
    reasons = stem_de_limiter_local_risk_reasons(stem, before)
    true_peak_delta = after.true_peak_dbtp - before.true_peak_dbtp
    sample_peak_delta = after.sample_peak_dbfs - before.sample_peak_dbfs
    crest_delta = after.crest_factor_db - before.crest_factor_db
    transient_delta = after.transient_lwmean - before.transient_lwmean
    punch_delta = after.punch_lwmean - before.punch_lwmean
    presence_delta = after.presence_harshness_p95 - before.presence_harshness_p95
    sibilance_delta = after.sibilance_p95 - before.sibilance_p95
    phase_delta = after.phase_risk_p95 - before.phase_risk_p95
    mono_loss_min_delta = after.mono_fold_loss_min_db - before.mono_fold_loss_min_db
    rms_delta = after.rms_dbfs - before.rms_dbfs
    peak_density_095_delta = (
        after.peak_density_gt_095_pct - before.peak_density_gt_095_pct
    )
    peak_density_090_delta = (
        after.peak_density_gt_090_pct - before.peak_density_gt_090_pct
    )

    if before.rms_dbfs <= -75.0:
        return StemDeLimiterStemDecision(
            stem,
            "skipped",
            ("stem is near silent",),
            (),
            before,
            after,
            source_path,
            output_path,
        )
    if after.rms_dbfs <= -75.0 and before.rms_dbfs > -60.0:
        warnings.append("candidate made an active stem nearly silent")
    if true_peak_delta > 0.15:
        warnings.append(f"true peak worsened by {true_peak_delta:+.2f} dB")
    if sample_peak_delta > 0.15:
        warnings.append(f"sample peak worsened by {sample_peak_delta:+.2f} dB")
    presence_limit = 0.03 if stem in {"vocals", "guitar", "piano", "other"} else 0.05
    if presence_delta > presence_limit:
        warnings.append(f"presence harshness worsened by {presence_delta:+.3f}")
    sibilance_limit = 0.02 if stem == "vocals" else 0.05
    if sibilance_delta > sibilance_limit:
        warnings.append(f"sibilance worsened by {sibilance_delta:+.3f}")
    if phase_delta > 0.05:
        warnings.append(f"phase risk worsened by {phase_delta:+.3f}")
    if mono_loss_min_delta < -0.50:
        warnings.append(
            f"mono fold loss minimum worsened by {mono_loss_min_delta:+.2f} dB"
        )
    if crest_delta < -0.25:
        warnings.append(f"crest factor fell by {crest_delta:+.2f} dB")
    if transient_delta < -0.02 and punch_delta < -0.02:
        warnings.append("transient and punch scores both fell")
    if stem == "bass" and rms_delta > 1.50:
        warnings.append(f"bass RMS rose by {rms_delta:+.2f} dB")
    if rms_delta < -3.00 and transient_delta < 0.01 and punch_delta < 0.005:
        warnings.append(
            f"RMS fell {rms_delta:+.2f} dB without enough transient/punch recovery"
        )

    improvements: list[str] = []
    if true_peak_delta <= -0.15:
        improvements.append(f"true peak improved by {true_peak_delta:+.2f} dB")
    if sample_peak_delta <= -0.20:
        improvements.append(f"sample peak improved by {sample_peak_delta:+.2f} dB")
    if peak_density_095_delta <= -0.05:
        improvements.append(
            f"peak density >0.95 improved by {peak_density_095_delta:+.3f} pct"
        )
    if peak_density_090_delta <= -0.10:
        improvements.append(
            f"peak density >0.90 improved by {peak_density_090_delta:+.3f} pct"
        )
    if crest_delta >= 0.05:
        improvements.append(f"crest improved by {crest_delta:+.2f} dB")
    if transient_delta >= 0.01:
        improvements.append(f"transient score improved by {transient_delta:+.3f}")
    if punch_delta >= 0.005:
        improvements.append(f"punch score improved by {punch_delta:+.3f}")

    has_local_risk = bool(reasons)
    risk_gate = has_local_risk or selected_mode in {"repair", "sections2"}
    if improvements:
        reasons.extend(improvements)
    if warnings:
        status = "rejected"
    elif not risk_gate:
        status = "skipped"
        reasons.append("no local limiter pressure detected")
    elif not improvements:
        status = "skipped"
        reasons.append("no meaningful local de-limiter improvement detected")
    else:
        status = "applied"
    return StemDeLimiterStemDecision(
        stem,
        status,
        tuple(reasons),
        tuple(warnings),
        before,
        after,
        source_path,
        output_path,
    )


def select_stem_de_limiter_candidates(
    stems: dict[str, Path],
    candidate_stems: dict[str, Path],
    *,
    selected_mode: str,
    mix: float,
    hop_sec: float,
    metrics_cache: StemDeLimiterMetricsCache | None = None,
) -> tuple[dict[str, Path], tuple[StemDeLimiterStemDecision, ...]]:
    cache = metrics_cache or StemDeLimiterMetricsCache()
    selected: dict[str, Path] = dict(stems)
    decisions: list[StemDeLimiterStemDecision] = []
    for stem in ordered_stem_names(stems):
        source_path = stems[stem]
        output_path = candidate_stems.get(stem)
        stem_hop_sec = max(float(hop_sec), 8.0)
        before = cache.local(f"{stem}_source", {stem: source_path})
        local_risk_reasons = stem_de_limiter_local_risk_reasons(stem, before)
        if output_path is None or before.rms_dbfs <= -75.0:
            reason = (
                "candidate was not generated"
                if output_path is None
                else "stem is near silent"
            )
            decisions.append(
                StemDeLimiterStemDecision(
                    stem,
                    "skipped",
                    (reason,),
                    (),
                    before,
                    before,
                    source_path,
                    source_path,
                )
            )
            continue
        if not local_risk_reasons and selected_mode not in {"repair", "sections2"}:
            decisions.append(
                StemDeLimiterStemDecision(
                    stem,
                    "skipped",
                    ("no local limiter pressure detected",),
                    (),
                    before,
                    before,
                    source_path,
                    source_path,
                )
            )
            continue
        before = cache.full(f"{stem}_source", {stem: source_path}, hop_sec=stem_hop_sec)
        after = cache.full(
            f"{stem}_sgi_{selected_mode}_mix{mix:.2f}",
            {stem: output_path},
            hop_sec=stem_hop_sec,
        )
        decision = evaluate_stem_de_limiter_stem_decision(
            stem,
            before,
            after,
            source_path=source_path,
            output_path=output_path,
            selected_mode=selected_mode,
        )
        decisions.append(decision)
        if decision.status == "applied":
            selected[stem] = output_path
    return selected, tuple(decisions)


def evaluate_stem_de_limiter_guard(
    before: StemDeLimiterMetrics,
    after: StemDeLimiterMetrics,
    *,
    selected_mode: str,
    stem_decisions: tuple[StemDeLimiterStemDecision, ...] = (),
) -> StemDeLimiterGuardResult:
    warnings: list[str] = []
    reasons: list[str] = []
    true_peak_delta = after.true_peak_dbtp - before.true_peak_dbtp
    crest_delta = after.crest_factor_db - before.crest_factor_db
    transient_delta = after.transient_lwmean - before.transient_lwmean
    punch_delta = after.punch_lwmean - before.punch_lwmean
    presence_delta = after.presence_harshness_p95 - before.presence_harshness_p95
    sibilance_delta = after.sibilance_p95 - before.sibilance_p95
    phase_delta = after.phase_risk_p95 - before.phase_risk_p95
    mono_loss_min_delta = after.mono_fold_loss_min_db - before.mono_fold_loss_min_db
    rms_delta = after.rms_dbfs - before.rms_dbfs
    peak_density_delta = after.peak_density_gt_090_pct - before.peak_density_gt_090_pct

    if presence_delta > 0.05:
        warnings.append(f"presence harshness worsened by {presence_delta:+.3f}")
    if sibilance_delta > 0.05:
        warnings.append(f"sibilance worsened by {sibilance_delta:+.3f}")
    if phase_delta > 0.05:
        warnings.append(f"phase risk worsened by {phase_delta:+.3f}")
    if mono_loss_min_delta < -0.50:
        warnings.append(
            f"mono fold loss minimum worsened by {mono_loss_min_delta:+.2f} dB"
        )
    if crest_delta < -0.05:
        warnings.append(f"crest factor fell by {crest_delta:+.2f} dB")
    if transient_delta < -0.01 and punch_delta < -0.01:
        warnings.append("transient and punch scores both fell")
    if rms_delta < -2.50 and transient_delta < 0.015 and punch_delta < 0.010:
        warnings.append(
            f"RMS fell {rms_delta:+.2f} dB without enough transient/punch recovery"
        )
    if true_peak_delta > 0.15:
        warnings.append(f"true peak worsened by {true_peak_delta:+.2f} dB")
    if after.true_peak_dbtp > 0.20 and not (
        before.true_peak_dbtp > 0.20 and true_peak_delta <= -0.20
    ):
        warnings.append(f"true peak remains unsafe at {after.true_peak_dbtp:+.2f} dBTP")

    if true_peak_delta <= -0.30:
        reasons.append(f"true peak improved by {true_peak_delta:+.2f} dB")
    if peak_density_delta <= -0.25:
        reasons.append(f"peak density improved by {peak_density_delta:+.2f} pct")
    if crest_delta >= 0.05:
        reasons.append(f"crest improved by {crest_delta:+.2f} dB")
    if transient_delta >= 0.01:
        reasons.append(f"transient score improved by {transient_delta:+.3f}")
    if punch_delta >= 0.005:
        reasons.append(f"punch score improved by {punch_delta:+.3f}")

    if selected_mode == "repair":
        improved = true_peak_delta <= -0.30 and (
            crest_delta >= 0.05 or transient_delta >= 0.01 or punch_delta >= 0.005
        )
    else:
        improved = bool(reasons) or true_peak_delta <= -0.20
    if not improved:
        warnings.append("no meaningful de-limiter improvement detected")
    return StemDeLimiterGuardResult(
        accepted=not warnings and improved,
        reasons=tuple(reasons),
        warnings=tuple(warnings),
        output_metrics=after,
        stem_decisions=stem_decisions,
    )


def print_stem_de_limiter_plan(plan: StemDeLimiterPlan) -> None:
    source = plan.source_metrics
    metric_text = ""
    if source is not None:
        metric_text = (
            f" TP={source.true_peak_dbtp:+.2f}dBTP"
            f" peak={source.sample_peak_dbfs:+.2f}dBFS"
            f" crest={source.crest_factor_db:.2f}dB"
            f" density90={source.peak_density_gt_090_pct:.3f}%"
        )
    if plan.enabled:
        print(
            f"Stem SGI de-limiter decision: requested={plan.requested_mode} "
            f"selected={plan.selected_mode} mix={plan.mix:.2f}; {plan.reason}.{metric_text}",
            flush=True,
        )
    else:
        print(
            f"Stem SGI de-limiter decision: requested={plan.requested_mode} "
            f"selected={plan.selected_mode} disabled; {plan.reason}.{metric_text}",
            flush=True,
        )


def print_stem_de_limiter_guard(result: StemDeLimiterGuardResult) -> None:
    status = "accepted" if result.accepted else "rejected"
    print(f"Stem SGI de-limiter guard: {status}", flush=True)
    for decision in result.stem_decisions:
        print(
            f"  stem {decision.status}: {decision.stem} "
            f"TP={decision.source_metrics.true_peak_dbtp:+.2f}->{decision.output_metrics.true_peak_dbtp:+.2f}dBTP "
            f"crest={decision.source_metrics.crest_factor_db:.2f}->{decision.output_metrics.crest_factor_db:.2f}dB",
            flush=True,
        )
    for reason in result.reasons:
        print(f"  ok: {reason}", flush=True)
    for warning in result.warnings:
        print(f"  reject: {warning}", flush=True)


def write_stem_de_limiter_report(
    path: Path,
    plan: StemDeLimiterPlan,
    guard: StemDeLimiterGuardResult | None,
) -> None:
    payload = stem_de_limiter_summary_data(plan, guard)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


__all__ = [
    "placeholder_stem_stats",
    "stem_metric_stats_from_paths",
    "crest_100ms_p95_db",
    "sample_peak_density_pct",
    "diagnostic_score_summary",
    "StemDeLimiterMetricsCache",
    "stem_de_limiter_metrics_from_paths",
    "stem_de_limiter_metrics_to_dict",
    "stem_de_limiter_metric_deltas",
    "stem_de_limiter_stem_decision_to_dict",
    "stem_de_limiter_summary_data",
    "choose_stem_de_limiter_plan",
    "ordered_stem_names",
    "stem_de_limiter_local_risk_reasons",
    "evaluate_stem_de_limiter_stem_decision",
    "select_stem_de_limiter_candidates",
    "evaluate_stem_de_limiter_guard",
    "print_stem_de_limiter_plan",
    "print_stem_de_limiter_guard",
    "write_stem_de_limiter_report",
]
