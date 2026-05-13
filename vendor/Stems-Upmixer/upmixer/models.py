"""Shared constants and dataclasses for the upmix pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

STEM_ORDER = ("vocals", "drums", "bass", "guitar", "piano", "other")

STEM_EXTENSIONS = (".wav", ".flac", ".aiff", ".aif")

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_STEM_DE_LIMITER_CHECKPOINT = (
    REPO_ROOT / "weights" / "de_limiter_best.safetensors"
)

DEFAULT_STEM_DE_LIMITER_MIX = 0.65

DEFAULT_STEM_DE_LIMITER_REMASTER_MIX = 0.65

DEFAULT_STEM_DE_LIMITER_REPAIR_MIX = 0.75

CHANNELS_714 = (
    "FL",
    "FR",
    "FC",
    "LFE",
    "BL",
    "BR",
    "SL",
    "SR",
    "TFL",
    "TFR",
    "TBL",
    "TBR",
)

VOL_RE = re.compile(r"(mean|max)_volume:\s*(-?inf|[-+]?\d+(?:\.\d+)?)\s*dB")

PAN_TERM_RE = re.compile(r"([-+]?\d+(?:\.\d+)?)\*([A-Z]+)")

FRONT_CHANNELS = {"FL", "FR", "FC"}

SPACE_CHANNELS = {"BL", "BR", "SL", "SR", "TFL", "TFR", "TBL", "TBR"}

AMBIENCE_CHANNELS = {"BL", "BR", "TFL", "TFR", "TBL", "TBR"}

SIDE_AMBIENCE_STEMS = set()

TOP_CHANNELS = {"TFL", "TFR", "TBL", "TBR"}

STEREO_FOLD_WEIGHTS = {
    "FL": (1.0, 0.0),
    "FR": (0.0, 1.0),
    "FC": (0.7071, 0.7071),
    "BL": (0.50, 0.0),
    "BR": (0.0, 0.50),
    "SL": (0.55, 0.0),
    "SR": (0.0, 0.55),
    "TFL": (0.35, 0.0),
    "TFR": (0.0, 0.35),
    "TBL": (0.25, 0.0),
    "TBR": (0.0, 0.25),
    "LFE": (0.0, 0.0),
}

BAND_NAMES = ("low", "mid", "high")

BASS_DRUM_GUARD_RANGES = (
    ("sub", 20.0, 60.0),
    ("kick", 60.0, 120.0),
    ("upper_bass", 120.0, 180.0),
)

BASS_DRUM_CROSSOVER_BANDS = ("sub", "kick", "upper_bass", "mid", "high")

SILENT_STEM_RMS_DB = -80.0

SILENT_STEM_P90_DB = -75.0

SILENT_STEM_ACTIVE_RATIO = 0.01

FILTER_COMPLEX_SCRIPT_CHAR_THRESHOLD = 24000

FILTER_COMPLEX_SCRIPT_NODE_THRESHOLD = 180

FILTER_COMPLEX_WARN_CHAR_THRESHOLD = 16000

FILTER_COMPLEX_WARN_NODE_THRESHOLD = 120

POST_RENDER_EFFECTIVE_GAIN_DB = 0.03

POST_RENDER_EFFECTIVE_EQ_DB = 0.075

POST_RENDER_EFFECTIVE_PLACEMENT_DELTA = 0.006

TEMPORAL_FEEDBACK_WARN_MIN_LANE_DELTA = 0.006

TEMPORAL_FEEDBACK_WATCH_MIN_LANE_DELTA = 0.035

MULTIBAND_RANGES = (
    ("sub", 20.0, 60.0),
    ("bass", 60.0, 120.0),
    ("lowmid", 120.0, 300.0),
    ("body", 300.0, 700.0),
    ("mid", 700.0, 1500.0),
    ("presence", 1500.0, 3500.0),
    ("air", 3500.0, 8000.0),
    ("top", 8000.0, 16000.0),
)

MULTIBAND_EQ_SPECS = {
    "sub": (42.0, 1.05),
    "bass": (85.0, 1.00),
    "lowmid": (190.0, 1.10),
    "body": (480.0, 1.05),
    "mid": (1150.0, 1.00),
    "presence": (2600.0, 1.10),
    "air": (6200.0, 1.20),
    "top": (10500.0, 1.20),
}

MULTIBAND_EQ_CANDIDATES = {
    "sub": (("bass", "drums", "other"), {"bass": 1.8, "drums": 1.0, "other": 0.35}),
    "bass": (("bass", "drums", "other"), {"bass": 1.6, "drums": 1.1, "other": 0.40}),
    "lowmid": (
        ("bass", "drums", "guitar", "piano", "other"),
        {"bass": 1.2, "drums": 0.9, "guitar": 0.75, "piano": 0.75, "other": 0.75},
    ),
    "body": (
        ("vocals", "guitar", "piano", "other", "drums"),
        {"vocals": 1.2, "guitar": 1.0, "piano": 0.9, "other": 0.9, "drums": 0.45},
    ),
    "mid": (
        ("vocals", "guitar", "piano", "other"),
        {"vocals": 1.15, "guitar": 1.0, "piano": 0.9, "other": 0.9},
    ),
    "presence": (
        ("vocals", "guitar", "piano", "other", "drums"),
        {"vocals": 0.9, "guitar": 1.05, "piano": 0.9, "other": 1.05, "drums": 0.55},
    ),
    "air": (
        ("guitar", "piano", "other", "drums", "vocals"),
        {"guitar": 1.0, "piano": 0.9, "other": 1.05, "drums": 0.75, "vocals": 0.35},
    ),
    "top": (
        ("drums", "guitar", "other", "piano"),
        {"drums": 0.9, "guitar": 0.9, "other": 1.0, "piano": 0.75},
    ),
}


@dataclass
class VolumeStats:
    mean_db: float
    max_db: float


@dataclass
class StemStats:
    name: str
    path: Path
    full: VolumeStats
    low: VolumeStats
    mid: VolumeStats
    high: VolumeStats
    mono: VolumeStats
    side: VolumeStats

    @property
    def width_db(self) -> float:
        return self.side.mean_db - self.mono.mean_db

    @property
    def low_vs_mid_db(self) -> float:
        return self.low.mean_db - self.mid.mean_db

    @property
    def high_vs_mid_db(self) -> float:
        return self.high.mean_db - self.mid.mean_db


@dataclass
class WindowStats:
    name: str
    active_ratio: float
    p10_db: float
    p50_db: float
    p90_db: float
    sustain_score: float
    envelope_db: tuple[float, ...]


@dataclass
class Placement:
    stem: str
    role: str
    pan: dict[str, str]
    lfe_amount: float = 0.0
    lfe_cutoff_hz: int = 80


@dataclass
class BassDrumCollisionBand:
    name: str
    low_hz: float
    high_hz: float
    overlap_score: float
    drums_share: float
    duck_db: float


@dataclass
class FocusDuckTarget:
    stem: str
    bands: tuple[str, ...]
    overlap_score: float
    duck_db: float
    bass_drum_bands: tuple[BassDrumCollisionBand, ...] = ()
    temporal_protection: float = 0.0


@dataclass
class PriorityDuckTarget:
    stem: str
    protector: str
    bands: tuple[str, ...]
    overlap_score: float
    duck_db: float


@dataclass
class TemporalDuckGuardItem:
    stem: str
    active_ratio: float
    dynamics_db: float
    sustain_score: float
    fragility_score: float
    protection: float
    duck_db_before: float
    duck_db_after: float


@dataclass
class TemporalDuckGuardResult:
    items: list[TemporalDuckGuardItem]


@dataclass
class TemporalDiagnosticFrame:
    momentary_lufs: float = -120.0
    short_lufs: float = -120.0
    loudness_range_db: float = 0.0
    true_peak_dbtp: float = -120.0
    crest_factor_db: float = 0.0
    transient_score: float = 0.0
    punch_score: float = 0.0
    low_punch_score: float = 0.0
    mono_correlation: float = 1.0
    mono_fold_loss_db: float = 0.0
    low_correlation: float = 1.0
    mid_correlation: float = 1.0
    high_correlation: float = 1.0
    phase_risk: float = 0.0
    peak_pressure: float = 0.0
    presence_harshness_score: float = 0.0
    sibilance_score: float = 0.0
    air_harshness_score: float = 0.0


@dataclass
class TemporalFrame:
    time_sec: float
    mix_energy_db: float
    vocal_presence: float
    rhythm_drive: float
    harmonic_density: float
    side_energy: float
    air_energy: float
    artifact_risk: float
    section_state: str
    diagnostics: TemporalDiagnosticFrame = field(
        default_factory=TemporalDiagnosticFrame
    )


@dataclass
class AutomationLanes:
    times: tuple[float, ...]
    step_sec: float
    vocal_protect: tuple[float, ...]
    chorus_space: tuple[float, ...]
    breakdown_air: tuple[float, ...]
    low_anchor: tuple[float, ...]
    low_tighten: tuple[float, ...]
    mid_decongest: tuple[float, ...]
    harshness_guard: tuple[float, ...]
    fold_safety: tuple[float, ...]
    solo_feature: dict[str, tuple[float, ...]]


@dataclass
class MasteringPrincipleScore:
    vocal_clarity: float
    low_end_control: float
    midrange_clarity: float
    harshness_control: float
    punch_preservation: float
    space_opportunity: float
    original_intent_preservation: float
    fold_translation_safety: float


@dataclass
class MasteringIntentDecision:
    time_sec: float
    state: str
    principles: MasteringPrincipleScore
    intent: str
    confidence: float
    actions: dict[str, float]
    solo_lift: dict[str, float]
    reasons: tuple[str, ...]
    feature_opportunity: dict[str, float]
    intent_scores: dict[str, float]
    diagnostics: TemporalDiagnosticFrame = field(
        default_factory=TemporalDiagnosticFrame
    )


@dataclass
class TemporalRemasterResult:
    enabled: bool
    frames: tuple[TemporalFrame, ...] = ()
    lanes: AutomationLanes | None = None
    intents: tuple[MasteringIntentDecision, ...] = ()
    reason: str = ""
    strength: float = 0.0
    profile: str = "natural"


@dataclass
class MasteringReferenceResult:
    gains: dict[str, float]
    adjustments: dict[str, float]
    before_ratios: tuple[float, float]
    after_ratios: tuple[float, float]
    guard_ratios: tuple[float, float] | None
    style_ratios: tuple[float, float] | None
    guard_tolerance_db: float
    style_strength: float


@dataclass
class PostRenderFoldAnalysis:
    actual: StemStats
    predicted_ratios: tuple[float, float] | None
    guard: StemStats | None
    style: StemStats | None
    guard_tolerance_db: float


@dataclass
class PostRenderCorrectionStep:
    iteration: int
    target_name: str
    low_mid_drift_db: float
    high_mid_drift_db: float
    width_drift_db: float
    rms_drift_db: float
    gain_adjustments: dict[str, float]
    placement_scales: dict[str, tuple[float, float]]
    eq_adjustments: dict[str, tuple[StemEqBand, ...]] = field(default_factory=dict)
    multiband_offset_db: float | None = None
    multiband_drifts_db: dict[str, float] = field(default_factory=dict)


@dataclass
class PostRenderCorrectionResult:
    gains: dict[str, float]
    placements: dict[str, Placement]
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]]
    step: PostRenderCorrectionStep


@dataclass
class StereoEnvelopeMatchResult:
    windows: int
    median_delta_db: float
    min_gain_db: float
    max_gain_db: float
    mean_abs_gain_db: float
    strength: float


@dataclass
class BrickwallRecoveryItem:
    stem: str
    enabled: bool
    attack_db: float
    sustain_cut_db: float
    side_scale: float
    peak_before: float
    peak_after: float
    guards: tuple[str, ...] = ()


@dataclass
class BrickwallRecoveryResult:
    enabled: bool
    mode: str
    pressure: float
    stems: dict[str, Path]
    items: list[BrickwallRecoveryItem] = field(default_factory=list)
    reason: str = ""


@dataclass
class StemDeLimiterMetrics:
    label: str
    true_peak_dbtp: float
    sample_peak_dbfs: float
    rms_dbfs: float
    crest_factor_db: float
    crest_100ms_p95_db: float
    transient_lwmean: float
    transient_p95: float
    punch_lwmean: float
    punch_p95: float
    presence_harshness_lwmean: float
    presence_harshness_p95: float
    sibilance_lwmean: float
    sibilance_p95: float
    phase_risk_lwmean: float
    phase_risk_p95: float
    mono_fold_loss_db: float
    mono_fold_loss_min_db: float
    peak_density_gt_095_pct: float
    peak_density_gt_090_pct: float


@dataclass
class StemDeLimiterPlan:
    requested_mode: str
    selected_mode: str
    enabled: bool
    mix: float
    reason: str
    checkpoint: Path | None = None
    source_metrics: StemDeLimiterMetrics | None = None


@dataclass
class StemDeLimiterStemDecision:
    stem: str
    status: str
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    source_metrics: StemDeLimiterMetrics
    output_metrics: StemDeLimiterMetrics
    source_path: Path
    output_path: Path


@dataclass
class StemDeLimiterGuardResult:
    accepted: bool
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    output_metrics: StemDeLimiterMetrics
    stem_decisions: tuple[StemDeLimiterStemDecision, ...] = ()


@dataclass
class MultibandItem:
    band: str
    full_db: float
    side_minus_mono_db: float


@dataclass
class MultibandStats:
    name: str
    rows: tuple[MultibandItem, ...]


@dataclass
class StemEqBand:
    name: str
    frequency_hz: float
    width_octaves: float
    gain_db: float


@dataclass
class StemPresenceItem:
    stem: str
    source_share: float
    fold_share_before: float
    fold_delta_db: float
    active_ratio: float
    target_floor_db: float
    adjustment_db: float
    reason: str


@dataclass
class StemPresenceResult:
    gains: dict[str, float]
    items: list[StemPresenceItem]


@dataclass
class DrumDominanceResult:
    gains: dict[str, float]
    source_share: float
    fold_share_before: float
    fold_delta_db: float
    active_ratio: float
    trim_db: float
    reason: str


@dataclass
class BassDrumCollisionGuardResult:
    enabled: bool
    collision_score: float
    low_overlap_score: float
    active_overlap_score: float
    duck_db: float
    reason: str
    bands: tuple[BassDrumCollisionBand, ...] = ()


@dataclass
class BassFoldSupportResult:
    enabled: bool
    source_share: float
    fold_share_before: float
    fold_delta_db: float
    target_floor_db: float
    support_coeff: float
    fold_power_before: float
    fold_power_after: float
    reason: str


@dataclass
class FocusDecision:
    mode: str
    stem: str | None
    shares: dict[str, float]
    reason: str
    support_stem: str | None = None
    strength: float = 0.0
    solo_ratio: float = 0.0
    score: float = 0.0
    vocal_anchor: str | None = None
    vocal_duck_db: float = 0.0
    focus_gain_db: float = 0.0
    bed_duck_db: float = 0.0
    focus_weights: dict[str, float] = field(default_factory=dict)
    bed_duck_targets: dict[str, FocusDuckTarget] = field(default_factory=dict)


@dataclass
class MixMultibandProfile:
    fractions: dict[str, float]
    low_pressure: float
    body_congestion: float
    presence_pressure: float
    air_opportunity: float


@dataclass
class VocalLeakageItem:
    stem: str
    leak_score: float
    envelope_correlation: float
    band_overlap: float
    center_score: float
    space_scale: float
    top_scale: float
    presence_trim_db: float
    reason: str


@dataclass
class VocalLeakageResult:
    items: list[VocalLeakageItem]


@dataclass
class TemporalFeedbackResult:
    changed: bool
    lanes: AutomationLanes | None
    event_count: int = 0
    width_events: int = 0
    low_events: int = 0
    presence_events: int = 0
    loudness_events: int = 0
    translation_events: int = 0
    max_damping: float = 0.0
    reason: str = ""


__all__ = [
    "STEM_ORDER",
    "STEM_EXTENSIONS",
    "REPO_ROOT",
    "DEFAULT_STEM_DE_LIMITER_CHECKPOINT",
    "DEFAULT_STEM_DE_LIMITER_MIX",
    "DEFAULT_STEM_DE_LIMITER_REMASTER_MIX",
    "DEFAULT_STEM_DE_LIMITER_REPAIR_MIX",
    "CHANNELS_714",
    "VOL_RE",
    "PAN_TERM_RE",
    "FRONT_CHANNELS",
    "SPACE_CHANNELS",
    "AMBIENCE_CHANNELS",
    "SIDE_AMBIENCE_STEMS",
    "TOP_CHANNELS",
    "STEREO_FOLD_WEIGHTS",
    "BAND_NAMES",
    "BASS_DRUM_GUARD_RANGES",
    "BASS_DRUM_CROSSOVER_BANDS",
    "SILENT_STEM_RMS_DB",
    "SILENT_STEM_P90_DB",
    "SILENT_STEM_ACTIVE_RATIO",
    "FILTER_COMPLEX_SCRIPT_CHAR_THRESHOLD",
    "FILTER_COMPLEX_SCRIPT_NODE_THRESHOLD",
    "FILTER_COMPLEX_WARN_CHAR_THRESHOLD",
    "FILTER_COMPLEX_WARN_NODE_THRESHOLD",
    "POST_RENDER_EFFECTIVE_GAIN_DB",
    "POST_RENDER_EFFECTIVE_EQ_DB",
    "POST_RENDER_EFFECTIVE_PLACEMENT_DELTA",
    "TEMPORAL_FEEDBACK_WARN_MIN_LANE_DELTA",
    "TEMPORAL_FEEDBACK_WATCH_MIN_LANE_DELTA",
    "MULTIBAND_RANGES",
    "MULTIBAND_EQ_SPECS",
    "MULTIBAND_EQ_CANDIDATES",
    "VolumeStats",
    "StemStats",
    "WindowStats",
    "Placement",
    "BassDrumCollisionBand",
    "FocusDuckTarget",
    "PriorityDuckTarget",
    "TemporalDuckGuardItem",
    "TemporalDuckGuardResult",
    "TemporalDiagnosticFrame",
    "TemporalFrame",
    "AutomationLanes",
    "MasteringPrincipleScore",
    "MasteringIntentDecision",
    "TemporalRemasterResult",
    "MasteringReferenceResult",
    "PostRenderFoldAnalysis",
    "PostRenderCorrectionStep",
    "PostRenderCorrectionResult",
    "StereoEnvelopeMatchResult",
    "BrickwallRecoveryItem",
    "BrickwallRecoveryResult",
    "StemDeLimiterMetrics",
    "StemDeLimiterPlan",
    "StemDeLimiterStemDecision",
    "StemDeLimiterGuardResult",
    "MultibandItem",
    "MultibandStats",
    "StemEqBand",
    "StemPresenceItem",
    "StemPresenceResult",
    "DrumDominanceResult",
    "BassDrumCollisionGuardResult",
    "BassFoldSupportResult",
    "FocusDecision",
    "MixMultibandProfile",
    "VocalLeakageItem",
    "VocalLeakageResult",
    "TemporalFeedbackResult",
]
