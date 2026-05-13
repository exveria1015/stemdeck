"""Argument parsing and top-level orchestration."""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from .de_limiter import (
    StemDeLimiterMetricsCache,
    choose_stem_de_limiter_plan,
    evaluate_stem_de_limiter_guard,
    print_stem_de_limiter_guard,
    print_stem_de_limiter_plan,
    select_stem_de_limiter_candidates,
    stem_de_limiter_metrics_from_paths,
    stem_de_limiter_summary_data,
    write_stem_de_limiter_report,
)
from .filters import lfe_fold_gain_for_mode, render_714
from .mix_tuning import (
    apply_focus_gain,
    apply_vocal_leakage_eq_guard,
    compute_stem_gains,
    post_render_correction_changed,
    post_render_correction_target,
    predicted_fold_band_energies,
    print_bass_drum_collision_guard,
    print_bass_fold_support_result,
    print_drum_dominance_result,
    print_mastering_reference_result,
    print_post_render_correction_step,
    print_post_render_fold_analysis,
    print_stem_gains,
    print_stem_presence_result,
    print_vocal_leakage_guard,
    priority_masking_targets,
    tonal_ratios,
    tune_bass_drum_collision_guard,
    tune_bass_fold_support,
    tune_drum_dominance_gains,
    tune_mastering_reference_gains,
    tune_post_render_correction,
    tune_stem_presence_gains,
    tune_vocal_leakage_placements,
)
from .models import (
    DEFAULT_STEM_DE_LIMITER_CHECKPOINT,
    MULTIBAND_RANGES,
    STEM_ORDER,
    AutomationLanes,
    BrickwallRecoveryResult,
    MasteringReferenceResult,
    MultibandStats,
    PostRenderFoldAnalysis,
    PriorityDuckTarget,
    StemDeLimiterGuardResult,
    StemStats,
    TemporalFeedbackResult,
    TemporalRemasterResult,
    WindowStats,
)
from .outputs import (
    analyze_multiband_numpy,
    analyze_rendered_fold,
    apply_stem_de_limiter_to_stems,
    filename_number_token,
    measure_51_loudnorm,
    measure_51_loudnorm_native,
    prepare_stem_de_limiter_dir,
    print_multiband_fold_analysis,
    prune_stem_de_limiter_candidate_files,
    render_51_flac,
    render_51_wav_native,
    render_apple_tv,
    render_stereo_flac,
    render_stereo_fold_wav,
    stereo_loudness_suffix,
    surround_loudness_suffix,
)
from .placement import (
    apply_focus_to_placements,
    apply_temporal_duck_guard,
    compute_stem_aware_eq,
    decide_focus_stem,
    decide_placement,
    focus_duck_targets,
    gain_suffix,
    print_stem_aware_eq,
    print_temporal_duck_guard_result,
    sanitized_focus_label,
)
from .recovery import apply_brickwall_recovery, print_brickwall_recovery_result
from .reporting import (
    print_analysis,
    print_collision_analysis,
    print_focus_analysis,
    print_priority_masking_analysis,
    print_window_analysis,
)
from .stems import (
    analyze_stem,
    analyze_window_stats,
    filter_silent_stems,
    find_stems,
    resolve_input_file,
    resolve_stem_dir,
    stem_range_energies_numpy,
)
from .temporal import (
    build_mix_multiband_profile,
    build_temporal_remaster,
    peak_pressure_true_peak_guard,
    print_mix_multiband_profile,
    print_temporal_profile_selection,
    print_temporal_remaster_result,
    select_temporal_mastering_profile,
    write_temporal_mastering_report,
)
from .utils import (
    clamp,
    media_duration,
    media_tags,
    parse_presence_priority,
    parse_stem_subset,
    print_elapsed,
    safe_filename,
    safe_label,
)
from .validation import (
    build_mastering_validation_summary,
    build_temporal_consistency_validation,
    print_temporal_feedback_result,
    print_validation_summary,
    tune_temporal_feedback,
)

T = TypeVar("T")
R = TypeVar("R")


def analysis_worker_count(backend: str, item_count: int) -> int:
    if item_count <= 1:
        return 1
    raw = os.environ.get("STEMS_UPMIXER_ANALYSIS_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
        return max(1, min(item_count, requested))
    default_limit = 2 if backend == "ffmpeg" else 4
    return max(1, min(item_count, default_limit, os.cpu_count() or 1))


def map_ordered_parallel(
    items: Iterable[T], worker: Callable[[T], R], *, max_workers: int
) -> list[R]:
    ordered = list(items)
    if max_workers <= 1 or len(ordered) <= 1:
        return [worker(item) for item in ordered]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(worker, item) for item in ordered]
        return [future.result() for future in futures]


def render_worker_count(item_count: int) -> int:
    if item_count <= 1:
        return 1
    raw = os.environ.get("STEMS_UPMIXER_RENDER_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError:
            requested = 1
        return max(1, min(item_count, requested))
    return max(1, min(item_count, 2, os.cpu_count() or 1))


def run_render_tasks(tasks: list[tuple[str, Callable[[], Any]]]) -> list[Any]:
    workers = render_worker_count(len(tasks))
    if workers > 1:
        names = ", ".join(name for name, _task in tasks)
        print(f"Render workers: {workers} ({names})", flush=True)
    return map_ordered_parallel(
        [task for _name, task in tasks],
        lambda task: task(),
        max_workers=workers,
    )


def collect_render_loudness(
    tasks: list[tuple[str, Callable[[], Any]]], results: list[Any]
) -> dict[str, dict[str, float]]:
    loudness: dict[str, dict[str, float]] = {}
    for (name, _task), result in zip(tasks, results, strict=False):
        if name in {"5.1 FLAC", "5.1 WAV"} and isinstance(result, dict):
            key = "surround_wav" if name == "5.1 WAV" else "surround_flac"
            loudness[key] = {
                "integrated_lufs": float(result["integrated_lufs"]),
                "true_peak_dbtp": float(result["true_peak_dbtp"]),
                "lra_lu": float(result["lra_lu"]),
                "threshold_lufs": float(result["threshold_lufs"]),
            }
    return loudness


def prewarm_stem_range_energy_cache(
    stats: list[StemStats], *, max_workers: int
) -> None:
    if not stats:
        return
    timer = time.perf_counter()

    def warm(item: StemStats) -> bool:
        try:
            stem_range_energies_numpy(item, MULTIBAND_RANGES)
        except Exception:
            return False
        return True

    warmed = sum(
        1
        for ok in map_ordered_parallel(stats, warm, max_workers=max_workers)
        if ok
    )
    if warmed:
        print(
            f"Stem range energy cache warm-up: {warmed}/{len(stats)} stems",
            flush=True,
        )
        print_elapsed("stem range energy cache warm-up", timer)


def evaluate_temporal_feedback_preflight(
    *,
    ffmpeg: str,
    output_dir: Path,
    bed_path: Path,
    source_path: Path | None,
    lfe_fold_gain: float,
    lanes: AutomationLanes,
    temporal_result: TemporalRemasterResult,
    strength: float,
) -> TemporalFeedbackResult | None:
    timer = time.perf_counter()
    temp_dir = output_dir / ".feedback_tmp"
    preview = temp_dir / "temporal_feedback_preflight_stereo.wav"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        print("Temporal feedback preflight: rendering stereo fold preview.", flush=True)
        render_stereo_fold_wav(ffmpeg, bed_path, preview, lfe_fold_gain, lanes)
        temporal_consistency = build_temporal_consistency_validation(
            checked_output="preflight_stereo_fold",
            render_path=preview,
            source_path=source_path,
            result=temporal_result,
        )
        events = temporal_consistency.get("events") or []
        print(
            "Temporal feedback preflight: "
            f"{temporal_consistency.get('status', 'unknown')} events={len(events)}",
            flush=True,
        )
        return tune_temporal_feedback(
            lanes,
            {"temporal_consistency": temporal_consistency},
            strength=strength,
        )
    except Exception as exc:
        print(f"Temporal feedback preflight unavailable: {exc}", flush=True)
        return None
    finally:
        preview.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        print_elapsed("temporal feedback preflight", timer)


def measure_surround_loudnorm(
    args: argparse.Namespace,
    bed_path: Path,
    lfe_fold_gain: float,
    *,
    target_i: float,
    target_tp: float,
    target_lra: float,
    label: str,
) -> dict[str, str]:
    if args.loudnorm_measure_backend == "native":
        measurements = measure_51_loudnorm_native(
            bed_path,
            lfe_fold_gain,
            target_i=target_i,
            target_tp=target_tp,
            target_lra=target_lra,
        )
        if measurements is None:
            raise RuntimeError(
                "native loudnorm measurement requested but stems-upmixer-loudnorm was not found"
            )
        print(f"{label}: native loudnorm measurement", flush=True)
        return measurements
    print(f"{label}: ffmpeg strict loudnorm measurement", flush=True)
    return measure_51_loudnorm(
        args.ffmpeg,
        bed_path,
        lfe_fold_gain,
        target_i=target_i,
        target_tp=target_tp,
        target_lra=target_lra,
        label=label,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze separated stems and create a simple music-oriented 7.1.4 / Apple TV upmix."
    )
    parser.add_argument(
        "--job-dir",
        type=Path,
        help="WebUI job directory containing input/ and output/.",
    )
    parser.add_argument(
        "--stem-dir",
        type=Path,
        help="Directory containing stem files such as vocals.wav.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Original input audio, used for cover art and metadata.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for upmix outputs. Default: <job-dir>/upmix.",
    )
    parser.add_argument(
        "--stems",
        default=",".join(STEM_ORDER),
        help="Comma-separated stem names to use. Default: vocals,drums,bass,guitar,piano,other.",
    )
    parser.add_argument(
        "--lfe-mode",
        choices=("light", "normal", "off"),
        default="light",
        help="LFE strategy. light treats LFE as subtle additive support. Default: light.",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=("auto", "numpy", "ffmpeg"),
        default="auto",
        help="Stem analysis backend. auto uses numpy/soundfile/scipy when available, then ffmpeg. Default: auto.",
    )
    parser.add_argument(
        "--silent-stem-filter",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Drop stems that are effectively silent before focus, placement, and rendering. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--mix-profile",
        choices=("auto", "front51"),
        default="auto",
        help="Spatial placement profile. front51 keeps the main image forward and moves synth-like other rearward. Default: auto.",
    )
    parser.add_argument(
        "--focus-stem-mode",
        choices=("auto", "manual", "off"),
        default="auto",
        help=(
            "Main/support stem focus mode. auto uses vocals as an anchor when present, then picks the "
            "best non-vocal/non-drum support stem by energy and solo sections; manual uses --focus-stem; "
            "off disables focus. Default: auto."
        ),
    )
    parser.add_argument(
        "--focus-stem",
        choices=STEM_ORDER,
        help="Stem to use when --focus-stem-mode manual is selected.",
    )
    parser.add_argument(
        "--focus-vocal-min-share",
        type=float,
        default=0.05,
        help="Minimum energy share for auto mode to treat vocals as present. Default: 0.05.",
    )
    parser.add_argument(
        "--presence-priority",
        default="",
        help=(
            "Comma-separated stem priority for masking ducking, highest first, e.g. other,guitar,piano. "
            "Default: disabled."
        ),
    )
    parser.add_argument(
        "--priority-duck",
        choices=("auto", "off"),
        default="auto",
        help="Enable light band ducking from higher-priority stems to lower-priority stems. Default: auto.",
    )
    parser.add_argument(
        "--priority-duck-max-db",
        type=float,
        default=0.8,
        help="Maximum approximate priority masking duck amount. Default: 0.8 dB.",
    )
    parser.add_argument(
        "--temporal-duck-guard",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Use stem window envelopes to soften focus ducking on sparse/transient stems. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--temporal-duck-guard-strength",
        type=float,
        default=0.75,
        help="Strength for --temporal-duck-guard from 0.0 to 1.0. Default: 0.75.",
    )
    parser.add_argument(
        "--temporal-remaster",
        choices=("auto", "off"),
        default="off",
        help=(
            "Create slow section-aware automation lanes from stem envelopes and apply subtle "
            "space-bed motion. Default: off."
        ),
    )
    parser.add_argument(
        "--temporal-remaster-strength",
        type=float,
        default=0.75,
        help="Strength for --temporal-remaster from 0.0 to 1.0. Default: 0.75.",
    )
    parser.add_argument(
        "--temporal-remaster-step-sec",
        type=float,
        default=4.0,
        help="Hold time for rendered temporal automation points. Default: 4.0 seconds.",
    )
    parser.add_argument(
        "--temporal-mastering",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Use mastering-principle intent rules before temporal automation. "
            "This is the richer successor to --temporal-remaster. Default: auto."
        ),
    )
    parser.add_argument(
        "--temporal-mastering-profile",
        choices=("auto", "conservative", "natural", "open"),
        default="auto",
        help="Temporal mastering behavior profile. auto selects per song. Default: auto.",
    )
    parser.add_argument(
        "--temporal-mastering-strength",
        type=float,
        default=0.65,
        help="Strength for --temporal-mastering from 0.0 to 1.0. Default: 0.65.",
    )
    parser.add_argument(
        "--temporal-mastering-report",
        type=Path,
        help="Optional JSON report path for temporal mastering segments, principles, reasons, and actions.",
    )
    parser.add_argument(
        "--temporal-feedback",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Use temporal validation events to damp risky automation and re-render once. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--temporal-feedback-preflight",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Run a stereo preview temporal check before final FLAC/MP4 rendering, "
            "so feedback can be applied before expensive outputs. Default: auto."
        ),
    )
    parser.add_argument(
        "--temporal-feedback-max-passes",
        type=int,
        default=1,
        help="Maximum temporal feedback re-render passes. Default: 1.",
    )
    parser.add_argument(
        "--temporal-feedback-strength",
        type=float,
        default=0.55,
        help="Strength for temporal validation feedback from 0.0 to 1.0. Default: 0.55.",
    )
    parser.add_argument(
        "--brickwall-recovery",
        choices=("auto", "off", "conservative", "natural", "open"),
        default="off",
        help=(
            "SGI-inspired stem-aware transient recovery for brickwall-limited sources. "
            "auto acts only when peak pressure/low crest suggest over-limiting. Default: off."
        ),
    )
    parser.add_argument(
        "--brickwall-recovery-strength",
        type=float,
        default=0.75,
        help="Strength for --brickwall-recovery from 0.0 to 1.2. Default: 0.75.",
    )
    parser.add_argument(
        "--brickwall-recovery-keep-stems",
        action="store_true",
        help="Keep the temporary recovered stem WAVs for inspection.",
    )
    parser.add_argument(
        "--stem-de-limiter",
        choices=("off", "auto", "safe", "remaster", "repair", "sections2"),
        default="off",
        help=(
            "Stem SGI de-limiter mode. auto chooses safe/remaster/repair from source metrics "
            "and rejects the processed stems if artifact guards fail. safe disables it, remaster "
            "defaults to mix 0.45, repair defaults to mix 0.75, sections2 is the manual preset. "
            "Default: off."
        ),
    )
    parser.add_argument(
        "--stem-de-limiter-checkpoint",
        type=Path,
        help=(
            "Optional stem-conditioned SGI checkpoint applied to each separated stem before "
            "brickwall recovery and upmix rendering."
        ),
    )
    parser.add_argument(
        "--stem-de-limiter-mix",
        type=float,
        default=None,
        help=(
            "Override parallel mix for stem SGI de-limiter. Defaults: sections2/remaster=0.45, "
            "repair=0.75, auto chooses from selected mode."
        ),
    )
    parser.add_argument(
        "--stem-de-limiter-device",
        default="auto",
        help="Device for stem SGI inference. Default: auto.",
    )
    parser.add_argument(
        "--stem-de-limiter-chunk-sec",
        type=float,
        default=8.0,
        help="Stem SGI chunk length. Default: 8.",
    )
    parser.add_argument(
        "--stem-de-limiter-overlap-sec",
        type=float,
        default=1.0,
        help="Stem SGI overlap. Default: 1.",
    )
    parser.add_argument(
        "--stem-de-limiter-peak-limit",
        type=float,
        default=0.98,
        help="Stem SGI peak safety. Default: 0.98.",
    )
    parser.add_argument(
        "--stem-de-limiter-keep-stems",
        action="store_true",
        help="Keep accepted stem SGI WAVs under output_dir/SGI_Stems.",
    )
    parser.add_argument(
        "--space-bed",
        choices=("auto", "off"),
        default="off",
        help=(
            "Add a short delayed/allpass ambience layer to eligible surround/top sends while keeping "
            "the direct placement intact. Default: off."
        ),
    )
    parser.add_argument(
        "--space-bed-stems",
        default="guitar,piano,other,drums",
        help="Comma-separated stems eligible for --space-bed auto. Default: guitar,piano,other,drums.",
    )
    parser.add_argument(
        "--space-bed-strength",
        type=float,
        default=0.85,
        help="Ambience path strength from 0.0 to 1.0. Default: 0.85.",
    )
    parser.add_argument(
        "--window-ms",
        type=float,
        default=1000.0,
        help="RMS window size for analysis-driven static placement. Default: 1000 ms.",
    )
    parser.add_argument(
        "--hop-ms",
        type=float,
        default=500.0,
        help="RMS hop size for analysis-driven static placement. Default: 500 ms.",
    )
    parser.add_argument(
        "--vocal-gain-db",
        type=float,
        default=None,
        help=(
            "Additional gain applied only to vocals before upmix placement. "
            "Default: 0.0 dB."
        ),
    )
    parser.add_argument(
        "--stem-gain-mode",
        choices=("off", "spatial", "harman"),
        default="off",
        help=(
            "Optional automatic per-stem gain. spatial compensates for wide/height placement; "
            "harman adds a conservative Harman-inspired tonal tilt. Default: off."
        ),
    )
    parser.add_argument(
        "--stem-gain-limit-db",
        type=float,
        default=1.5,
        help="Maximum automatic per-stem gain adjustment before vocal gain is added. Default: 1.5 dB.",
    )
    parser.add_argument(
        "--stem-aware-eq",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Apply a conservative per-stem EQ lift to wide non-vocal stems before placement. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--stem-aware-eq-strength",
        type=float,
        default=0.75,
        help="Strength for --stem-aware-eq, from 0.0 to 1.0. Default: 0.75.",
    )
    parser.add_argument(
        "--stem-aware-eq-limit-db",
        type=float,
        default=1.2,
        help="Maximum EQ boost per stem band for --stem-aware-eq. Default: 1.2 dB.",
    )
    parser.add_argument(
        "--bass-fold-support",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Add a small front low-anchor support to bass when its predicted stereo fold share is too low. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--bass-fold-support-target-db",
        type=float,
        default=-3.5,
        help="Minimum target bass fold share relative to source share. Default: -3.5 dB.",
    )
    parser.add_argument(
        "--bass-fold-support-max-coeff",
        type=float,
        default=0.12,
        help="Maximum front-channel coefficient added by --bass-fold-support. Default: 0.12.",
    )
    parser.add_argument(
        "--reference-match",
        choices=("off", "original"),
        default="off",
        help="Deprecated alias for --reference-guard original. Default: off.",
    )
    parser.add_argument(
        "--reference-guard",
        choices=("auto", "off", "original"),
        default="auto",
        help=(
            "Use the original mix as a guardrail instead of an exact target. "
            "auto enables original guard when an input file is available. Default: auto."
        ),
    )
    parser.add_argument(
        "--reference-match-limit-db",
        type=float,
        default=1.0,
        help="Maximum per-stem gain adjustment used by reference guard/style reference. Default: 1.0 dB.",
    )
    parser.add_argument(
        "--reference-guard-tolerance-db",
        type=float,
        default=1.0,
        help="Allowed tonal drift from the original guard before corrections are applied. Default: 1.0 dB.",
    )
    parser.add_argument(
        "--style-reference",
        type=Path,
        help="External mastered song used as a tonal style attractor. It does not override the original guard.",
    )
    parser.add_argument(
        "--style-reference-strength",
        type=float,
        default=0.35,
        help="How strongly --style-reference pulls tonal balance, from 0.0 to 1.0. Default: 0.35.",
    )
    parser.add_argument(
        "--stem-presence-guard",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Keep non-drum stems from becoming too quiet in the predicted stereo fold-down. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--stem-presence-guard-limit-db",
        type=float,
        default=0.6,
        help="Maximum per-stem gain added by --stem-presence-guard. Default: 0.6 dB.",
    )
    parser.add_argument(
        "--drum-dominance-guard",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Trim drums when they become too dominant in the predicted stereo fold-down. "
            "Default: auto."
        ),
    )
    parser.add_argument(
        "--drum-dominance-guard-limit-db",
        type=float,
        default=0.35,
        help="Maximum drum trim applied by --drum-dominance-guard. Default: 0.35 dB.",
    )
    parser.add_argument(
        "--bass-drum-guard",
        choices=("auto", "off"),
        default="auto",
        help=(
            "When bass is the focus stem, lightly duck the drum low band from bass activity "
            "if bass and kick overlap in the stereo fold. Default: auto."
        ),
    )
    parser.add_argument(
        "--bass-drum-guard-max-db",
        type=float,
        default=1.2,
        help="Maximum approximate drum low-band duck from --bass-drum-guard. Default: 1.2 dB.",
    )
    parser.add_argument(
        "--post-render-analysis",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Analyze the rendered 7.1.4 bed folded to stereo after rendering. auto runs when "
            "reference guard/style reference is active. Default: auto."
        ),
    )
    parser.add_argument(
        "--post-render-correct",
        choices=("auto", "off"),
        default="auto",
        help=(
            "After the first 7.1.4 render, analyze the actual stereo fold and re-render with small "
            "stem/space corrections when it drifts from the reference guard or style reference. Default: auto."
        ),
    )
    parser.add_argument(
        "--post-render-correct-iterations",
        type=int,
        default=2,
        help="Maximum post-render correction passes. Default: 2.",
    )
    parser.add_argument(
        "--post-render-correct-tolerance-db",
        type=float,
        default=0.55,
        help="Tonal drift allowed before post-render correction acts. Default: 0.55 dB.",
    )
    parser.add_argument(
        "--post-render-correct-limit-db",
        type=float,
        default=0.7,
        help="Maximum cumulative stem gain change from post-render correction. Default: 0.7 dB.",
    )
    parser.add_argument(
        "--post-render-correct-space-limit",
        type=float,
        default=0.06,
        help="Maximum cumulative rear/top coefficient reduction per correction pass. Default: 0.06.",
    )
    parser.add_argument(
        "--post-render-correct-eq",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Use relative 8-band post-render drift for small per-stem EQ corrections when "
            "--post-render-correct is active. Default: auto."
        ),
    )
    parser.add_argument(
        "--post-render-correct-eq-limit-db",
        type=float,
        default=0.8,
        help="Maximum cumulative post-render EQ change per stem band. Default: 0.8 dB.",
    )
    parser.add_argument(
        "--master-gain",
        type=float,
        default=0.84,
        help="Final 7.1.4 gain before limiting. Default: 0.84.",
    )
    parser.add_argument(
        "--flac-compression-level",
        type=int,
        default=8,
        help="Compression level for FFmpeg FLAC outputs. Default: 8.",
    )
    parser.add_argument(
        "--skip-flac",
        action="store_true",
        help="Do not render the main 5.1 surround output.",
    )
    parser.add_argument(
        "--skip-stereo",
        action="store_true",
        help="Do not render the stereo fold-down FLAC output.",
    )
    parser.add_argument(
        "--stereo-normalize",
        choices=("loudnorm", "linear", "off"),
        default="linear",
        help=(
            "Stereo fold-down normalization. loudnorm uses EBU R128 dynamic normalization; "
            "linear applies peak-safe static gain and preserves macro dynamics. Default: linear."
        ),
    )
    parser.add_argument(
        "--stereo-loudness-i",
        type=float,
        default=-14.0,
        help="Stereo fold-down integrated loudness target in LUFS. Default: -14.0.",
    )
    parser.add_argument(
        "--stereo-loudness-tp",
        type=float,
        default=-1.0,
        help="Stereo fold-down true peak target in dBTP. Positive values are treated as headroom, so 1.0 means -1.0. Default: -1.0.",
    )
    parser.add_argument(
        "--stereo-loudness-lra",
        type=float,
        default=11.0,
        help="Stereo fold-down loudness range target for loudnorm. Default: 11.0.",
    )
    parser.add_argument(
        "--stereo-lfe-fold-gain",
        type=float,
        default=0.0,
        help="How much LFE is folded into the stereo output. Default: 0.0 because LFE is additive support.",
    )
    parser.add_argument(
        "--stereo-envelope-match",
        choices=("auto", "off"),
        default="auto",
        help=(
            "Match the stereo fold-down macro envelope to the original input before stereo normalization. "
            "Only affects the stereo output. Default: auto."
        ),
    )
    parser.add_argument(
        "--stereo-envelope-window-sec",
        type=float,
        default=12.0,
        help="Window length used by --stereo-envelope-match. Default: 12 seconds.",
    )
    parser.add_argument(
        "--stereo-envelope-hop-sec",
        type=float,
        default=3.0,
        help="Hop length used by --stereo-envelope-match. Default: 3 seconds.",
    )
    parser.add_argument(
        "--stereo-envelope-smooth-sec",
        type=float,
        default=24.0,
        help="Smoothing length for the stereo envelope gain curve. Default: 24 seconds.",
    )
    parser.add_argument(
        "--stereo-envelope-strength",
        type=float,
        default=0.75,
        help="Envelope matching strength from 0.0 to 1.0. Default: 0.75.",
    )
    parser.add_argument(
        "--stereo-envelope-max-boost-db",
        type=float,
        default=2.5,
        help="Maximum stereo envelope boost. Default: 2.5 dB.",
    )
    parser.add_argument(
        "--stereo-envelope-max-cut-db",
        type=float,
        default=1.5,
        help="Maximum stereo envelope cut. Default: 1.5 dB.",
    )
    parser.add_argument(
        "--de-limiter-checkpoint",
        type=Path,
        help="Optional SGI de-limiter checkpoint applied to the final stereo fold before stereo normalization.",
    )
    parser.add_argument(
        "--de-limiter-mix",
        type=float,
        default=1.0,
        help="Parallel mix for --de-limiter-checkpoint, 0=input and 1=de-limited. Default: 1.0.",
    )
    parser.add_argument(
        "--de-limiter-device",
        default="auto",
        help="Device for --de-limiter-checkpoint. Default: auto.",
    )
    parser.add_argument(
        "--de-limiter-chunk-sec",
        type=float,
        default=8.0,
        help="Inference chunk length. Default: 8.",
    )
    parser.add_argument(
        "--de-limiter-overlap-sec",
        type=float,
        default=1.0,
        help="Inference overlap. Default: 1.",
    )
    parser.add_argument(
        "--de-limiter-peak-limit",
        type=float,
        default=0.98,
        help="Peak safety after SGI inference. Default: 0.98.",
    )
    parser.add_argument(
        "--surround-normalize",
        choices=("loudnorm", "off"),
        default="loudnorm",
        help="Final 5.1 loudness normalization for surround and Apple TV outputs. Default: loudnorm.",
    )
    parser.add_argument(
        "--surround-render-backend",
        choices=("native-wav", "ffmpeg-flac"),
        default="native-wav",
        help=(
            "Main 5.1 output renderer. native-wav writes a PCM WAV directly with the Rust loudnorm path; "
            "ffmpeg-flac keeps the previous FLAC reference path. Default: native-wav."
        ),
    )
    parser.add_argument(
        "--surround-loudness-i",
        type=float,
        default=-14.0,
        help="5.1 integrated loudness target in LUFS. Default: -14.0.",
    )
    parser.add_argument(
        "--surround-loudness-tp",
        type=float,
        default=-2.5,
        help=(
            "5.1 true peak target in dBTP. Positive values are treated as headroom, so 2.5 means -2.5. "
            "Default: -2.5 to leave stereo downmix headroom."
        ),
    )
    parser.add_argument(
        "--surround-loudness-lra",
        type=float,
        default=11.0,
        help="5.1 loudness range target for loudnorm. Default: 11.0.",
    )
    parser.add_argument(
        "--loudnorm-measure-backend",
        choices=("native", "ffmpeg"),
        default="native",
        help=(
            "Backend for the shared 5.1 loudnorm measurement pass. "
            "native uses the Rust measurer by default; ffmpeg is the strict reference mode. Default: native."
        ),
    )
    parser.add_argument(
        "--apple-tv-bitrate",
        default="768k",
        help="E-AC-3 bitrate for Apple TV MP4. Default: 768k.",
    )
    parser.add_argument(
        "--skip-apple-tv",
        action="store_true",
        help="Do not render the Apple TV E-AC-3 MP4 output.",
    )
    parser.add_argument(
        "--skip-bed",
        action="store_true",
        help="Do not keep the intermediate 7.1.4 WAV after derived outputs.",
    )
    parser.add_argument("--title", help="Override output title metadata.")
    parser.add_argument(
        "--ffmpeg", default="ffmpeg", help="ffmpeg executable. Default: ffmpeg."
    )
    parser.add_argument(
        "--ffprobe", default="ffprobe", help="ffprobe executable. Default: ffprobe."
    )
    return parser.parse_args()


def main() -> None:
    total_timer = time.perf_counter()
    args = parse_args()
    job_dir = args.job_dir.resolve() if args.job_dir else None
    stem_dir = resolve_stem_dir(job_dir, args.stem_dir)
    input_file = resolve_input_file(job_dir, args.input)
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (job_dir / "upmix" if job_dir else stem_dir / "upmix")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    de_limiter_checkpoint = (
        args.de_limiter_checkpoint.resolve()
        if args.de_limiter_checkpoint is not None
        else None
    )
    if de_limiter_checkpoint is not None and not de_limiter_checkpoint.exists():
        raise SystemExit(
            f"--de-limiter-checkpoint does not exist: {de_limiter_checkpoint}"
        )
    de_limiter_mix = clamp(float(args.de_limiter_mix), 0.0, 1.0)
    stem_de_limiter_checkpoint = (
        args.stem_de_limiter_checkpoint.resolve()
        if args.stem_de_limiter_checkpoint is not None
        else None
    )
    stem_de_limiter_requested = str(args.stem_de_limiter)
    if stem_de_limiter_requested == "off" and stem_de_limiter_checkpoint is not None:
        stem_de_limiter_requested = "sections2"
    if (
        stem_de_limiter_requested not in {"off", "safe"}
        and stem_de_limiter_checkpoint is None
    ):
        stem_de_limiter_checkpoint = DEFAULT_STEM_DE_LIMITER_CHECKPOINT.resolve()
    if (
        stem_de_limiter_checkpoint is not None
        and not stem_de_limiter_checkpoint.exists()
    ):
        raise SystemExit(
            f"--stem-de-limiter-checkpoint does not exist: {stem_de_limiter_checkpoint}"
        )
    stem_de_limiter_mix_override = (
        None
        if args.stem_de_limiter_mix is None
        else clamp(float(args.stem_de_limiter_mix), 0.0, 1.0)
    )

    requested = [stem.strip() for stem in args.stems.split(",") if stem.strip()]
    stems = find_stems(stem_dir, requested)
    space_bed_strength = clamp(float(args.space_bed_strength), 0.0, 1.0)
    temporal_mode = (
        args.temporal_mastering
        if args.temporal_mastering != "off"
        else args.temporal_remaster
    )
    temporal_profile_request = (
        args.temporal_mastering_profile
        if args.temporal_mastering != "off"
        else "natural"
    )
    temporal_profile = temporal_profile_request
    temporal_strength = (
        float(args.temporal_mastering_strength)
        if args.temporal_mastering != "off"
        else float(args.temporal_remaster_strength)
    )
    space_bed_mode = args.space_bed
    if temporal_mode != "off" and space_bed_mode == "off":
        space_bed_mode = "auto"
    ignored = sorted(
        path.stem
        for path in stem_dir.iterdir()
        if path.is_file() and path.stem == "instrumental"
    )
    if ignored:
        print(
            "Ignoring instrumental stem to avoid double-counting derived mix-minus-vocals audio.",
            flush=True,
        )

    analysis_timer = time.perf_counter()
    stem_names = list(stems)
    analysis_workers = analysis_worker_count(args.analysis_backend, len(stem_names))
    if analysis_workers > 1:
        print(f"Analysis workers: {analysis_workers}", flush=True)
    analysis_results = map_ordered_parallel(
        stem_names,
        lambda stem: analyze_stem(args.ffmpeg, stem, stems[stem], args.analysis_backend),
        max_workers=analysis_workers,
    )
    stats = [item for item, _backend in analysis_results]
    backends = {backend for _item, backend in analysis_results}
    print(f"Analysis backend: {', '.join(sorted(backends))}", flush=True)
    stats_by_name = {item.name: item for item in stats}
    window_stats: dict[str, WindowStats] = {}
    if args.mix_profile in {"auto", "front51"}:
        window_results = map_ordered_parallel(
            stats,
            lambda item: (
                item.name,
                analyze_window_stats(
                    item.name,
                    stems[item.name],
                    item,
                    window_ms=float(args.window_ms),
                    hop_ms=float(args.hop_ms),
                ),
            ),
            max_workers=analysis_workers,
        )
        window_stats = dict(window_results)
        stems, stats, window_stats = filter_silent_stems(
            stems,
            stats,
            window_stats,
            mode=args.silent_stem_filter,
        )
        stats_by_name = {item.name: item for item in stats}
        print_window_analysis(stats, window_stats)
        print_collision_analysis(stats, window_stats)
    prewarm_stem_range_energy_cache(stats, max_workers=analysis_workers)
    mix_multiband_profile = build_mix_multiband_profile(stats)
    print_mix_multiband_profile(mix_multiband_profile)
    temporal_profile, temporal_profile_reason = select_temporal_mastering_profile(
        temporal_profile_request,
        stats,
        window_stats,
    )
    print_temporal_profile_selection(
        temporal_profile_request, temporal_profile, temporal_profile_reason
    )
    presence_priority = parse_presence_priority(str(args.presence_priority), stems)
    space_bed_stems = parse_stem_subset(
        str(args.space_bed_stems), stems, option="--space-bed-stems"
    )
    focus_decision = decide_focus_stem(
        stats,
        window_stats,
        mode=args.focus_stem_mode,
        manual_stem=args.focus_stem,
        vocal_min_share=clamp(float(args.focus_vocal_min_share), 0.0, 1.0),
    )
    focus_decision.bed_duck_targets = focus_duck_targets(
        stats, window_stats, focus_decision
    )
    priority_duck_targets: dict[str, list[PriorityDuckTarget]] = {}
    if args.priority_duck != "off" and presence_priority:
        priority_duck_targets = priority_masking_targets(
            stats,
            window_stats,
            presence_priority,
            max_duck_db=max(0.0, float(args.priority_duck_max_db)),
        )
    priority_sidechain_duration = (
        media_duration(args.ffprobe, next(iter(stems.values()))) + 0.05
        if priority_duck_targets
        else None
    )
    placements = {
        item.name: decide_placement(
            item, args.lfe_mode, args.mix_profile, window_stats, stats_by_name
        )
        for item in stats
    }
    placements = apply_focus_to_placements(placements, focus_decision)
    placements, bass_fold_support_result = tune_bass_fold_support(
        stats,
        placements,
        window_stats,
        mode=args.bass_fold_support,
        target_floor_db=float(args.bass_fold_support_target_db),
        max_coeff=float(args.bass_fold_support_max_coeff),
    )
    bass_drum_guard_result = tune_bass_drum_collision_guard(
        stats,
        window_stats,
        placements,
        focus_decision,
        mode=args.bass_drum_guard,
        max_duck_db=float(args.bass_drum_guard_max_db),
    )
    placements, vocal_leakage_result = tune_vocal_leakage_placements(
        stats, placements, window_stats
    )
    temporal_duck_guard_result = apply_temporal_duck_guard(
        stats,
        window_stats,
        focus_decision,
        mode=args.temporal_duck_guard,
        strength=float(args.temporal_duck_guard_strength),
    )
    temporal_remaster_result = build_temporal_remaster(
        stats,
        window_stats,
        mode=temporal_mode,
        profile=temporal_profile,
        strength=temporal_strength,
        hop_ms=float(args.hop_ms),
        step_sec=float(args.temporal_remaster_step_sec),
        multiband_profile=mix_multiband_profile,
    )
    automation_lanes = (
        temporal_remaster_result.lanes if temporal_remaster_result.enabled else None
    )
    print_focus_analysis(focus_decision)
    print_temporal_duck_guard_result(temporal_duck_guard_result)
    print_temporal_remaster_result(temporal_remaster_result)
    print_bass_fold_support_result(bass_fold_support_result)
    print_bass_drum_collision_guard(bass_drum_guard_result)
    print_priority_masking_analysis(presence_priority, priority_duck_targets)
    print_vocal_leakage_guard(vocal_leakage_result)
    print_analysis(stats, placements)
    print_elapsed("analysis and decision", analysis_timer)

    source_title = args.title
    tags = media_tags(args.ffprobe, input_file)
    if source_title is None:
        source_title = tags.get("title") or stem_dir.name
    slug = safe_filename(source_title)
    stem_de_limiter_dir: Path | None = None
    stem_de_limiter_enabled = False
    stem_de_limiter_report_path = output_dir / f"{slug}_stem_de_limiter_decision.json"
    stem_de_limiter_source_paths = (
        {"input": input_file} if input_file is not None else dict(stems)
    )
    stem_de_limiter_metrics_cache = StemDeLimiterMetricsCache()
    stem_de_limiter_source_metrics = stem_de_limiter_metrics_from_paths(
        "input" if input_file is not None else "stem_sum",
        stem_de_limiter_source_paths,
        hop_sec=float(args.temporal_remaster_step_sec),
        cache=stem_de_limiter_metrics_cache,
    )
    stem_de_limiter_plan = choose_stem_de_limiter_plan(
        stem_de_limiter_requested,
        stem_de_limiter_source_metrics,
        stem_de_limiter_checkpoint,
        stem_de_limiter_mix_override,
    )
    print_stem_de_limiter_plan(stem_de_limiter_plan)
    stem_de_limiter_guard: StemDeLimiterGuardResult | None = None
    if stem_de_limiter_plan.enabled and stem_de_limiter_plan.checkpoint is not None:
        stem_de_limiter_dir = prepare_stem_de_limiter_dir(
            output_dir,
            slug,
            keep_stems=bool(args.stem_de_limiter_keep_stems),
        )
        candidate_stems = apply_stem_de_limiter_to_stems(
            stems,
            stem_de_limiter_plan.checkpoint,
            stem_de_limiter_dir,
            mix=stem_de_limiter_plan.mix,
            chunk_seconds=float(args.stem_de_limiter_chunk_sec),
            overlap_seconds=float(args.stem_de_limiter_overlap_sec),
            device=str(args.stem_de_limiter_device),
            peak_limit=float(args.stem_de_limiter_peak_limit),
        )
        selected_candidate_stems, stem_de_limiter_stem_decisions = (
            select_stem_de_limiter_candidates(
                stems,
                candidate_stems,
                selected_mode=stem_de_limiter_plan.selected_mode,
                mix=stem_de_limiter_plan.mix,
                hop_sec=float(args.temporal_remaster_step_sec),
                metrics_cache=stem_de_limiter_metrics_cache,
            )
        )
        candidate_metrics = stem_de_limiter_metrics_from_paths(
            f"stem_sgi_{stem_de_limiter_plan.selected_mode}_mix{stem_de_limiter_plan.mix:.2f}_stem_aware",
            selected_candidate_stems,
            hop_sec=float(args.temporal_remaster_step_sec),
            cache=stem_de_limiter_metrics_cache,
        )
        stem_de_limiter_guard = evaluate_stem_de_limiter_guard(
            stem_de_limiter_source_metrics,
            candidate_metrics,
            selected_mode=stem_de_limiter_plan.selected_mode,
            stem_decisions=stem_de_limiter_stem_decisions,
        )
        print_stem_de_limiter_guard(stem_de_limiter_guard)
        prune_stem_de_limiter_candidate_files(
            candidate_stems,
            stem_de_limiter_stem_decisions,
            guard_accepted=stem_de_limiter_guard.accepted,
        )
        if stem_de_limiter_guard.accepted:
            stems = selected_candidate_stems
            stem_de_limiter_enabled = True
        elif not args.stem_de_limiter_keep_stems:
            shutil.rmtree(stem_de_limiter_dir, ignore_errors=True)
            stem_de_limiter_dir = None
    write_stem_de_limiter_report(
        stem_de_limiter_report_path, stem_de_limiter_plan, stem_de_limiter_guard
    )
    brickwall_recovery_dir: Path | None = None
    brickwall_recovery_result = BrickwallRecoveryResult(
        False,
        args.brickwall_recovery,
        0.0,
        dict(stems),
        reason="disabled",
    )
    if args.brickwall_recovery != "off":
        brickwall_recovery_dir = Path(
            tempfile.mkdtemp(prefix=f"{slug}_brickwall_recovery_", dir=output_dir)
        )
        brickwall_recovery_result = apply_brickwall_recovery(
            stems,
            stats,
            window_stats,
            temporal_remaster_result,
            mode=args.brickwall_recovery,
            strength=float(args.brickwall_recovery_strength),
            output_dir=brickwall_recovery_dir,
        )
        print_brickwall_recovery_result(brickwall_recovery_result)
        if brickwall_recovery_result.enabled:
            stems = brickwall_recovery_result.stems
        elif (
            brickwall_recovery_dir is not None
            and not args.brickwall_recovery_keep_stems
        ):
            shutil.rmtree(brickwall_recovery_dir, ignore_errors=True)
            brickwall_recovery_dir = None
    temporal_report_path = (
        args.temporal_mastering_report.resolve()
        if args.temporal_mastering_report is not None
        else (
            output_dir / f"{slug}_temporal_mastering_intent.json"
            if args.temporal_mastering != "off"
            else None
        )
    )
    if temporal_report_path is not None and temporal_remaster_result.lanes is not None:
        write_temporal_mastering_report(
            temporal_report_path,
            temporal_remaster_result,
            de_limiter_summary=stem_de_limiter_summary_data(
                stem_de_limiter_plan, stem_de_limiter_guard
            ),
        )
    vocal_gain_db = 0.0 if args.vocal_gain_db is None else float(args.vocal_gain_db)
    stem_gain_db = compute_stem_gains(
        stats,
        placements,
        mode=args.stem_gain_mode,
        vocal_gain_db=vocal_gain_db,
        limit_db=float(args.stem_gain_limit_db),
    )
    stem_eq_bands = compute_stem_aware_eq(
        stats,
        mode=args.stem_aware_eq,
        strength=float(args.stem_aware_eq_strength),
        limit_db=float(args.stem_aware_eq_limit_db),
    )
    stem_eq_bands = apply_vocal_leakage_eq_guard(stem_eq_bands, vocal_leakage_result)
    print_stem_aware_eq(stem_eq_bands)
    stem_gain_db = apply_focus_gain(stem_gain_db, focus_decision)
    presence_result = tune_stem_presence_gains(
        stats,
        placements,
        window_stats,
        stem_gain_db,
        focus_decision,
        mode=args.stem_presence_guard,
        limit_db=float(args.stem_presence_guard_limit_db),
    )
    stem_gain_db = presence_result.gains
    print_stem_presence_result(presence_result)
    drum_result = tune_drum_dominance_gains(
        stats,
        placements,
        window_stats,
        stem_gain_db,
        focus_decision,
        mode=args.drum_dominance_guard,
        limit_db=float(args.drum_dominance_guard_limit_db),
    )
    stem_gain_db = drum_result.gains
    print_drum_dominance_result(drum_result)
    reference_suffix = ""
    reference_guard = (
        "original"
        if (
            args.reference_guard == "original"
            or args.reference_match == "original"
            or (
                args.reference_guard == "auto"
                and input_file is not None
                and input_file.exists()
            )
        )
        else "off"
    )
    guard_stats: StemStats | None = None
    style_stats: StemStats | None = None
    reference_result: MasteringReferenceResult | None = None
    if reference_guard == "original":
        if input_file is None:
            raise SystemExit(
                "--reference-guard original requires --input or a WebUI job input file."
            )
        guard_stats, guard_backend = analyze_stem(
            args.ffmpeg, "original_guard", input_file, args.analysis_backend
        )
        print(f"Reference guard analysis backend: {guard_backend}", flush=True)
        reference_suffix += "_guard_original"
    if args.style_reference is not None:
        style_path = args.style_reference.resolve()
        if not style_path.exists():
            raise SystemExit(f"--style-reference does not exist: {style_path}")
        style_stats, style_backend = analyze_stem(
            args.ffmpeg, "style_reference", style_path, args.analysis_backend
        )
        print(f"Style reference analysis backend: {style_backend}", flush=True)
        reference_suffix += f"_style_{safe_filename(style_path.stem)[:32]}"
    if guard_stats is not None or style_stats is not None:
        reference_result = tune_mastering_reference_gains(
            stats,
            placements,
            stem_gain_db,
            guard_stats=guard_stats,
            style_stats=style_stats,
            style_strength=float(args.style_reference_strength),
            guard_tolerance_db=float(args.reference_guard_tolerance_db),
            limit_db=float(args.reference_match_limit_db),
        )
        stem_gain_db = reference_result.gains
        print_mastering_reference_result(reference_result)
    print_stem_gains(stem_gain_db)
    post_render_target = post_render_correction_target(guard_stats, style_stats)
    post_render_correction_enabled = (
        args.post_render_correct != "off"
        and post_render_target is not None
        and int(args.post_render_correct_iterations) > 0
    )
    if (
        args.post_render_correct != "off"
        and post_render_target is None
        and (
            args.reference_guard == "original"
            or args.reference_match == "original"
            or args.style_reference is not None
        )
    ):
        print(
            "Post-render correction skipped: requires --reference-guard original or --style-reference.",
            flush=True,
        )
    stem_gain_suffix = (
        f"_stemgain_{args.stem_gain_mode}" if args.stem_gain_mode != "off" else ""
    )
    stem_eq_suffix = "_stemeq" if stem_eq_bands else ""
    focus_suffix = (
        f"_focus_{args.focus_stem_mode}_{sanitized_focus_label(focus_decision)}"
        if focus_decision.stem is not None and args.focus_stem_mode != "off"
        else ""
    )
    priority_suffix = (
        f"_priority_{'_'.join(safe_label(stem) for stem in presence_priority)}"
        if presence_priority
        else ""
    )
    space_bed_suffix = (
        "_spacebed" if space_bed_mode != "off" and space_bed_stems else ""
    )
    if temporal_remaster_result.enabled and args.temporal_mastering != "off":
        temporal_remaster_suffix = f"_tmaster_{safe_label(temporal_profile)}"
    else:
        temporal_remaster_suffix = (
            "_temporal" if temporal_remaster_result.enabled else ""
        )
    temporal_duck_suffix = "_tempduck" if temporal_duck_guard_result.items else ""
    bass_fold_support_suffix = "_bassfold" if bass_fold_support_result.enabled else ""
    bass_drum_guard_suffix = "_bassdrumguard" if bass_drum_guard_result.enabled else ""
    stem_de_limiter_suffix = "_stemsgi" if stem_de_limiter_enabled else ""
    brickwall_recovery_suffix = (
        "_brickwallrec" if brickwall_recovery_result.enabled else ""
    )
    post_render_suffix = "_postcorr" if post_render_correction_enabled else ""
    profile_root = "front51" if args.mix_profile == "front51" else "auto_placement"
    profile = (
        f"{profile_root}_lfe_{args.lfe_mode}"
        f"{gain_suffix('vocal', vocal_gain_db)}"
        f"{stem_gain_suffix}{stem_eq_suffix}{focus_suffix}{priority_suffix}{space_bed_suffix}{temporal_remaster_suffix}"
        f"{temporal_duck_suffix}{bass_fold_support_suffix}{bass_drum_guard_suffix}{stem_de_limiter_suffix}"
        f"{brickwall_recovery_suffix}"
        f"{reference_suffix}{post_render_suffix}"
    )
    bed_path = output_dir / f"{slug}_7.1.4_{profile}.wav"
    surround_target_i = float(args.surround_loudness_i)
    surround_target_tp = -abs(float(args.surround_loudness_tp))
    guarded_surround_tp, surround_tp_reason = peak_pressure_true_peak_guard(
        surround_target_tp,
        temporal_remaster_result,
    )
    if surround_tp_reason:
        print(
            f"Peak-pressure TP guard: surround TP {surround_target_tp:+.2f}->{guarded_surround_tp:+.2f} dBTP "
            f"({surround_tp_reason})",
            flush=True,
        )
    surround_target_tp = guarded_surround_tp
    surround_suffix = surround_loudness_suffix(
        args.surround_normalize, surround_target_i, surround_target_tp
    )
    flac_path = output_dir / f"{slug}_{profile}_{surround_suffix}.flac"
    surround_output_path = (
        output_dir / f"{slug}_{profile}_{surround_suffix}.wav"
        if args.surround_render_backend == "native-wav"
        else flac_path
    )
    surround_output_label = (
        "5.1 WAV" if args.surround_render_backend == "native-wav" else "5.1 FLAC"
    )
    stereo_target_i = float(args.stereo_loudness_i)
    stereo_target_tp = -abs(float(args.stereo_loudness_tp))
    guarded_stereo_tp, stereo_tp_reason = peak_pressure_true_peak_guard(
        stereo_target_tp,
        temporal_remaster_result,
    )
    if stereo_tp_reason:
        print(
            f"Peak-pressure TP guard: stereo TP {stereo_target_tp:+.2f}->{guarded_stereo_tp:+.2f} dBTP "
            f"({stereo_tp_reason})",
            flush=True,
        )
    stereo_target_tp = guarded_stereo_tp
    stereo_envelope_match_enabled = (
        args.stereo_envelope_match != "off"
        and input_file is not None
        and input_file.exists()
    )
    stereo_suffix = stereo_loudness_suffix(
        args.stereo_normalize,
        stereo_target_i,
        stereo_target_tp,
        envelope_match=stereo_envelope_match_enabled,
    )
    if de_limiter_checkpoint is not None:
        stereo_suffix += f"_sgi_delimit_mix_{filename_number_token(de_limiter_mix)}"
    stereo_path = output_dir / f"{slug}_{profile}_{stereo_suffix}.flac"
    apple_path = output_dir / f"{slug}_{profile}_{surround_suffix}_ddp5.1_apple_tv.mp4"

    print(f"Rendering 7.1.4 WAV: {bed_path}", flush=True)
    lfe_fold_gain = lfe_fold_gain_for_mode(args.lfe_mode, args.mix_profile)
    stereo_lfe_fold_gain = max(0.0, float(args.stereo_lfe_fold_gain))
    latest_actual_fold_stats: StemStats | None = None
    latest_fold_backend: str | None = None
    render_714(
        args.ffmpeg,
        stems,
        placements,
        bed_path,
        args.master_gain,
        stem_gain_db,
        stem_eq_bands,
        focus_decision,
        priority_duck_targets,
        priority_sidechain_duration,
        space_bed_mode,
        space_bed_stems,
        space_bed_strength,
        automation_lanes,
        native_enabled=args.skip_apple_tv,
    )

    post_render_base_gains = dict(stem_gain_db)
    post_render_base_eq_bands = dict(stem_eq_bands)
    if post_render_correction_enabled and post_render_target is not None:
        target_name, target_stats = post_render_target
        target_multiband_stats: MultibandStats | None = None
        if args.post_render_correct_eq != "off":
            try:
                target_multiband_stats = analyze_multiband_numpy(
                    target_name, target_stats.path, stereo_lfe_fold_gain
                )
            except Exception as exc:
                print(f"Post-render 8-band correction unavailable: {exc}", flush=True)
        for correction_iteration in range(
            1, max(0, int(args.post_render_correct_iterations)) + 1
        ):
            latest_actual_fold_stats, latest_fold_backend = analyze_rendered_fold(
                args.ffmpeg,
                bed_path,
                stereo_lfe_fold_gain,
                args.analysis_backend,
            )
            print(
                f"Post-render correction analysis backend: {latest_fold_backend}",
                flush=True,
            )
            actual_multiband_stats: MultibandStats | None = None
            if target_multiband_stats is not None:
                try:
                    actual_multiband_stats = analyze_multiband_numpy(
                        "actual", bed_path, stereo_lfe_fold_gain
                    )
                except Exception as exc:
                    print(
                        f"Post-render 8-band correction unavailable: {exc}", flush=True
                    )
                    target_multiband_stats = None
            correction_result = tune_post_render_correction(
                stats,
                placements,
                stem_gain_db,
                post_render_base_gains,
                latest_actual_fold_stats,
                stem_eq_bands,
                post_render_base_eq_bands,
                target_name=target_name,
                target_stats=target_stats,
                actual_multiband_stats=actual_multiband_stats,
                target_multiband_stats=target_multiband_stats,
                iteration=correction_iteration,
                tolerance_db=float(args.post_render_correct_tolerance_db),
                gain_limit_db=float(args.post_render_correct_limit_db),
                placement_scale_limit=float(args.post_render_correct_space_limit),
                eq_limit_db=float(args.post_render_correct_eq_limit_db)
                if args.post_render_correct_eq != "off"
                else 0.0,
            )
            print_post_render_correction_step(correction_result.step)
            if not post_render_correction_changed(correction_result.step):
                break
            stem_gain_db = correction_result.gains
            placements = correction_result.placements
            stem_eq_bands = correction_result.stem_eq_bands
            print_stem_gains(stem_gain_db)
            print(
                f"Re-rendering 7.1.4 WAV after post-render correction {correction_iteration}: {bed_path}",
                flush=True,
            )
            render_714(
                args.ffmpeg,
                stems,
                placements,
                bed_path,
                args.master_gain,
                stem_gain_db,
                stem_eq_bands,
                focus_decision,
                priority_duck_targets,
                priority_sidechain_duration,
                space_bed_mode,
                space_bed_stems,
                space_bed_strength,
                automation_lanes,
                native_enabled=args.skip_apple_tv,
            )
            latest_actual_fold_stats = None
            latest_fold_backend = None

    if args.post_render_analysis != "off" and (
        guard_stats is not None or style_stats is not None
    ):
        if latest_actual_fold_stats is None:
            latest_actual_fold_stats, latest_fold_backend = analyze_rendered_fold(
                args.ffmpeg,
                bed_path,
                stereo_lfe_fold_gain,
                args.analysis_backend,
            )
        print(f"Post-render fold analysis backend: {latest_fold_backend}", flush=True)
        predicted_ratios = tonal_ratios(
            predicted_fold_band_energies(stats, placements, stem_gain_db)
        )
        print_post_render_fold_analysis(
            PostRenderFoldAnalysis(
                actual=latest_actual_fold_stats,
                predicted_ratios=predicted_ratios,
                guard=guard_stats,
                style=style_stats,
                guard_tolerance_db=float(args.reference_guard_tolerance_db),
            )
        )
        print_multiband_fold_analysis(
            bed_path,
            lfe_fold_gain=stereo_lfe_fold_gain,
            guard_stats=guard_stats,
            style_stats=style_stats,
        )

    max_feedback_passes = max(0, int(args.temporal_feedback_max_passes))
    feedback_pass_start = 1
    if (
        args.temporal_feedback_preflight != "off"
        and args.temporal_feedback != "off"
        and max_feedback_passes > 0
        and automation_lanes is not None
        and temporal_remaster_result.lanes is not None
    ):
        feedback_result = evaluate_temporal_feedback_preflight(
            ffmpeg=args.ffmpeg,
            output_dir=output_dir,
            bed_path=bed_path,
            source_path=input_file,
            lfe_fold_gain=stereo_lfe_fold_gain,
            lanes=automation_lanes,
            temporal_result=temporal_remaster_result,
            strength=float(args.temporal_feedback_strength),
        )
        if feedback_result is not None:
            if feedback_result.changed and feedback_result.lanes is not None:
                print_temporal_feedback_result(feedback_result, 1)
                automation_lanes = feedback_result.lanes
                temporal_remaster_result.lanes = automation_lanes
                print(
                    f"Re-rendering 7.1.4 WAV after temporal feedback preflight 1: {bed_path}",
                    flush=True,
                )
                render_714(
                    args.ffmpeg,
                    stems,
                    placements,
                    bed_path,
                    args.master_gain,
                    stem_gain_db,
                    stem_eq_bands,
                    focus_decision,
                    priority_duck_targets,
                    priority_sidechain_duration,
                    space_bed_mode,
                    space_bed_stems,
                    space_bed_strength,
                    automation_lanes,
                    native_enabled=args.skip_apple_tv,
                )
                feedback_pass_start = 2
            elif feedback_result.event_count > 0:
                print(
                    "Temporal feedback preflight: skipped "
                    f"events={feedback_result.event_count} "
                    f"max_lane_delta={feedback_result.max_damping:.3f} below effective threshold",
                    flush=True,
                )

    shared_surround_loudnorm_measurements: dict[str, str] | None = None
    needs_shared_surround_loudnorm = args.surround_normalize == "loudnorm" and (
        not args.skip_apple_tv
        or (not args.skip_flac and args.surround_render_backend == "ffmpeg-flac")
    )
    if needs_shared_surround_loudnorm:
        print("Measuring shared 5.1 loudnorm pass for surround outputs.", flush=True)
        shared_surround_loudnorm_measurements = measure_surround_loudnorm(
            args,
            bed_path,
            lfe_fold_gain,
            target_i=surround_target_i,
            target_tp=surround_target_tp,
            target_lra=float(args.surround_loudness_lra),
            label="surround_shared_loudnorm_measure",
        )

    render_tasks: list[tuple[str, Callable[[], Any]]] = []
    if not args.skip_flac:
        def render_surround_output() -> dict[str, float] | None:
            if args.surround_render_backend == "native-wav":
                print(f"Rendering 5.1 WAV: {surround_output_path}", flush=True)
                return render_51_wav_native(
                    args.ffmpeg,
                    bed_path,
                    surround_output_path,
                    lfe_fold_gain,
                    args.surround_normalize,
                    surround_target_i,
                    surround_target_tp,
                    float(args.surround_loudness_lra),
                )
            print(f"Rendering 5.1 FLAC: {surround_output_path}", flush=True)
            return render_51_flac(
                args.ffmpeg,
                bed_path,
                surround_output_path,
                tags,
                source_title,
                lfe_fold_gain,
                int(args.flac_compression_level),
                args.surround_normalize,
                surround_target_i,
                surround_target_tp,
                float(args.surround_loudness_lra),
                shared_surround_loudnorm_measurements,
            )

        render_tasks.append((surround_output_label, render_surround_output))

    if not args.skip_stereo:
        def render_stereo_output() -> None:
            print(f"Rendering stereo FLAC: {stereo_path}", flush=True)
            render_stereo_flac(
                args.ffmpeg,
                bed_path,
                stereo_path,
                tags,
                source_title,
                input_file,
                stereo_lfe_fold_gain,
                automation_lanes,
                int(args.flac_compression_level),
                args.stereo_normalize,
                stereo_target_i,
                stereo_target_tp,
                float(args.stereo_loudness_lra),
                args.stereo_envelope_match,
                float(args.stereo_envelope_window_sec),
                float(args.stereo_envelope_hop_sec),
                float(args.stereo_envelope_smooth_sec),
                float(args.stereo_envelope_strength),
                float(args.stereo_envelope_max_boost_db),
                float(args.stereo_envelope_max_cut_db),
                de_limiter_checkpoint,
                de_limiter_mix,
                float(args.de_limiter_chunk_sec),
                float(args.de_limiter_overlap_sec),
                str(args.de_limiter_device),
                float(args.de_limiter_peak_limit),
            )

        render_tasks.append(("stereo FLAC", render_stereo_output))

    if not args.skip_apple_tv:
        def render_apple_tv_output() -> None:
            duration = media_duration(args.ffprobe, bed_path)
            print(f"Rendering Apple TV MP4: {apple_path}", flush=True)
            render_apple_tv(
                args.ffmpeg,
                bed_path,
                input_file,
                apple_path,
                duration,
                args.apple_tv_bitrate,
                tags,
                source_title,
                lfe_fold_gain,
                args.surround_normalize,
                surround_target_i,
                surround_target_tp,
                float(args.surround_loudness_lra),
                shared_surround_loudnorm_measurements,
            )

        render_tasks.append(("Apple TV MP4", render_apple_tv_output))
    render_loudness = collect_render_loudness(
        render_tasks, run_render_tasks(render_tasks)
    )

    if temporal_report_path is not None and temporal_remaster_result.lanes is not None:
        validation_summary = build_mastering_validation_summary(
            ffmpeg=args.ffmpeg,
            ffprobe=args.ffprobe,
            output_dir=output_dir,
            source_path=input_file,
            bed_path=bed_path,
            stereo_path=stereo_path,
            surround_path=surround_output_path,
            apple_path=apple_path,
            render_stereo=not args.skip_stereo,
            render_surround=not args.skip_flac,
            render_apple_tv=not args.skip_apple_tv,
            stereo_target_i=stereo_target_i,
            stereo_target_tp=stereo_target_tp,
            stereo_target_lra=float(args.stereo_loudness_lra),
            surround_target_i=surround_target_i,
            surround_target_tp=surround_target_tp,
            surround_target_lra=float(args.surround_loudness_lra),
            stereo_lfe_fold_gain=stereo_lfe_fold_gain,
            surround_lfe_fold_gain=lfe_fold_gain,
            automation_lanes=automation_lanes,
            temporal_result=temporal_remaster_result,
            precomputed_loudness=render_loudness,
        )
        print_validation_summary(validation_summary)
        if args.temporal_feedback != "off" and automation_lanes is not None:
            for feedback_pass in range(
                feedback_pass_start, max_feedback_passes + 1
            ):
                feedback_result = tune_temporal_feedback(
                    automation_lanes,
                    validation_summary,
                    strength=float(args.temporal_feedback_strength),
                )
                if not feedback_result.changed or feedback_result.lanes is None:
                    if feedback_result.event_count > 0:
                        print(
                            f"Temporal feedback pass {feedback_pass}: skipped "
                            f"events={feedback_result.event_count} "
                            f"max_lane_delta={feedback_result.max_damping:.3f} below effective threshold",
                            flush=True,
                        )
                    break
                print_temporal_feedback_result(feedback_result, feedback_pass)
                automation_lanes = feedback_result.lanes
                temporal_remaster_result.lanes = automation_lanes
                print(
                    f"Re-rendering 7.1.4 WAV after temporal feedback {feedback_pass}: {bed_path}",
                    flush=True,
                )
                render_714(
                    args.ffmpeg,
                    stems,
                    placements,
                    bed_path,
                    args.master_gain,
                    stem_gain_db,
                    stem_eq_bands,
                    focus_decision,
                    priority_duck_targets,
                    priority_sidechain_duration,
                    space_bed_mode,
                    space_bed_stems,
                    space_bed_strength,
                    automation_lanes,
                    native_enabled=args.skip_apple_tv,
                )
                feedback_surround_loudnorm_measurements: dict[str, str] | None = None
                needs_feedback_surround_loudnorm = (
                    args.surround_normalize == "loudnorm"
                    and (
                        not args.skip_apple_tv
                        or (
                            not args.skip_flac
                            and args.surround_render_backend == "ffmpeg-flac"
                        )
                    )
                )
                if needs_feedback_surround_loudnorm:
                    print(
                        f"Measuring shared 5.1 loudnorm pass for surround outputs after temporal feedback {feedback_pass}.",
                        flush=True,
                    )
                    feedback_surround_loudnorm_measurements = measure_surround_loudnorm(
                        args,
                        bed_path,
                        lfe_fold_gain,
                        target_i=surround_target_i,
                        target_tp=surround_target_tp,
                        target_lra=float(args.surround_loudness_lra),
                        label=f"surround_feedback{feedback_pass}_loudnorm_measure",
                    )
                feedback_render_tasks: list[tuple[str, Callable[[], Any]]] = []
                if not args.skip_flac:
                    def render_feedback_surround_output(
                        feedback_pass: int = feedback_pass,
                        loudnorm_measurements: dict[str, str]
                        | None = feedback_surround_loudnorm_measurements,
                    ) -> dict[str, float] | None:
                        if args.surround_render_backend == "native-wav":
                            print(
                                f"Re-rendering 5.1 WAV after temporal feedback {feedback_pass}: {surround_output_path}",
                                flush=True,
                            )
                            return render_51_wav_native(
                                args.ffmpeg,
                                bed_path,
                                surround_output_path,
                                lfe_fold_gain,
                                args.surround_normalize,
                                surround_target_i,
                                surround_target_tp,
                                float(args.surround_loudness_lra),
                            )
                        print(
                            f"Re-rendering 5.1 FLAC after temporal feedback {feedback_pass}: {surround_output_path}",
                            flush=True,
                        )
                        return render_51_flac(
                            args.ffmpeg,
                            bed_path,
                            surround_output_path,
                            tags,
                            source_title,
                            lfe_fold_gain,
                            int(args.flac_compression_level),
                            args.surround_normalize,
                            surround_target_i,
                            surround_target_tp,
                            float(args.surround_loudness_lra),
                            loudnorm_measurements,
                        )

                    feedback_render_tasks.append(
                        (surround_output_label, render_feedback_surround_output)
                    )
                if not args.skip_stereo:
                    def render_feedback_stereo_output(
                        feedback_pass: int = feedback_pass,
                        lanes: AutomationLanes | None = automation_lanes,
                    ) -> None:
                        print(
                            f"Re-rendering stereo FLAC after temporal feedback {feedback_pass}: {stereo_path}",
                            flush=True,
                        )
                        render_stereo_flac(
                            args.ffmpeg,
                            bed_path,
                            stereo_path,
                            tags,
                            source_title,
                            input_file,
                            stereo_lfe_fold_gain,
                            lanes,
                            int(args.flac_compression_level),
                            args.stereo_normalize,
                            stereo_target_i,
                            stereo_target_tp,
                            float(args.stereo_loudness_lra),
                            args.stereo_envelope_match,
                            float(args.stereo_envelope_window_sec),
                            float(args.stereo_envelope_hop_sec),
                            float(args.stereo_envelope_smooth_sec),
                            float(args.stereo_envelope_strength),
                            float(args.stereo_envelope_max_boost_db),
                            float(args.stereo_envelope_max_cut_db),
                            de_limiter_checkpoint,
                            de_limiter_mix,
                            float(args.de_limiter_chunk_sec),
                            float(args.de_limiter_overlap_sec),
                            str(args.de_limiter_device),
                            float(args.de_limiter_peak_limit),
                        )

                    feedback_render_tasks.append(
                        ("stereo FLAC", render_feedback_stereo_output)
                    )
                if not args.skip_apple_tv:
                    def render_feedback_apple_tv_output(
                        feedback_pass: int = feedback_pass,
                        loudnorm_measurements: dict[str, str]
                        | None = feedback_surround_loudnorm_measurements,
                    ) -> None:
                        duration = media_duration(args.ffprobe, bed_path)
                        print(
                            f"Re-rendering Apple TV MP4 after temporal feedback {feedback_pass}: {apple_path}",
                            flush=True,
                        )
                        render_apple_tv(
                            args.ffmpeg,
                            bed_path,
                            input_file,
                            apple_path,
                            duration,
                            args.apple_tv_bitrate,
                            tags,
                            source_title,
                            lfe_fold_gain,
                            args.surround_normalize,
                            surround_target_i,
                            surround_target_tp,
                            float(args.surround_loudness_lra),
                            loudnorm_measurements,
                        )

                    feedback_render_tasks.append(
                        ("Apple TV MP4", render_feedback_apple_tv_output)
                    )
                render_loudness = collect_render_loudness(
                    feedback_render_tasks, run_render_tasks(feedback_render_tasks)
                )
                validation_summary = build_mastering_validation_summary(
                    ffmpeg=args.ffmpeg,
                    ffprobe=args.ffprobe,
                    output_dir=output_dir,
                    source_path=input_file,
                    bed_path=bed_path,
                    stereo_path=stereo_path,
                    surround_path=surround_output_path,
                    apple_path=apple_path,
                    render_stereo=not args.skip_stereo,
                    render_surround=not args.skip_flac,
                    render_apple_tv=not args.skip_apple_tv,
                    stereo_target_i=stereo_target_i,
                    stereo_target_tp=stereo_target_tp,
                    stereo_target_lra=float(args.stereo_loudness_lra),
                    surround_target_i=surround_target_i,
                    surround_target_tp=surround_target_tp,
                    surround_target_lra=float(args.surround_loudness_lra),
                    stereo_lfe_fold_gain=stereo_lfe_fold_gain,
                    surround_lfe_fold_gain=lfe_fold_gain,
                    automation_lanes=automation_lanes,
                    temporal_result=temporal_remaster_result,
                    precomputed_loudness=render_loudness,
                )
                print_validation_summary(validation_summary)
        write_temporal_mastering_report(
            temporal_report_path,
            temporal_remaster_result,
            validation_summary=validation_summary,
            de_limiter_summary=stem_de_limiter_summary_data(
                stem_de_limiter_plan, stem_de_limiter_guard
            ),
        )

    if args.skip_bed and (
        not args.skip_flac or not args.skip_stereo or not args.skip_apple_tv
    ):
        bed_path.unlink(missing_ok=True)
    if brickwall_recovery_dir is not None and not args.brickwall_recovery_keep_stems:
        shutil.rmtree(brickwall_recovery_dir, ignore_errors=True)
    if stem_de_limiter_dir is not None and not args.stem_de_limiter_keep_stems:
        shutil.rmtree(stem_de_limiter_dir, ignore_errors=True)

    print("Done.", flush=True)
    print_elapsed("total", total_timer)
    if bed_path.exists():
        print(f"7.1.4: {bed_path}", flush=True)
    if not args.skip_flac:
        print(f"{surround_output_label}: {surround_output_path}", flush=True)
    if not args.skip_stereo:
        print(f"Stereo FLAC: {stereo_path}", flush=True)
    if not args.skip_apple_tv:
        print(f"Apple TV: {apple_path}", flush=True)


__all__ = ["parse_args", "main"]
