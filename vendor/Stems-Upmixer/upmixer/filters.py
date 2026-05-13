"""FFmpeg filter graph construction for 7.1.4, 5.1, and stereo folds."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .models import (
    AMBIENCE_CHANNELS,
    BAND_NAMES,
    BASS_DRUM_CROSSOVER_BANDS,
    CHANNELS_714,
    PAN_TERM_RE,
    REPO_ROOT,
    SIDE_AMBIENCE_STEMS,
    STEM_ORDER,
    AutomationLanes,
    FocusDecision,
    FocusDuckTarget,
    Placement,
    PriorityDuckTarget,
    StemEqBand,
)
from .placement import (
    focus_label,
    gain_filter,
    ordered_focus_weights,
    should_duck_for_focus,
    stem_eq_filter,
)
from .stems import pan_expr
from .utils import append_filter_complex_arg, clamp, db_to_amp, fmt, print_elapsed, run, safe_label

_NATIVE_714_BIN: Path | None | bool = None
_NATIVE_714_INPUT_CHANNELS = {"FL": 0, "FR": 1}
_NATIVE_714_OUTPUT_CHANNELS = {
    channel: index for index, channel in enumerate(CHANNELS_714)
}


def native_714_binary() -> Path | None:
    global _NATIVE_714_BIN
    if _NATIVE_714_BIN is False:
        return None
    if isinstance(_NATIVE_714_BIN, Path):
        return _NATIVE_714_BIN

    env_path = os.environ.get("STEMDECK_NATIVE_AUDIO_BIN", "").strip()
    exe_name = "stemdeck-native-audio.exe" if sys.platform.startswith("win") else "stemdeck-native-audio"
    stemdeck_root = REPO_ROOT.parent.parent
    manifest = stemdeck_root / "native_audio" / "Cargo.toml"
    expected = stemdeck_root / "native_audio" / "target" / "release" / exe_name
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(expected)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            _NATIVE_714_BIN = path.resolve()
            return _NATIVE_714_BIN

    if env_path or not manifest.is_file() or shutil.which("cargo") is None:
        _NATIVE_714_BIN = False
        return None

    timer = time.perf_counter()
    proc = subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--manifest-path", str(manifest)],
        capture_output=True,
        text=True,
    )
    print_elapsed("native 7.1.4 helper build", timer)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        if detail:
            print(detail, file=sys.stderr, flush=True)
        _NATIVE_714_BIN = False
        return None
    if expected.is_file() and os.access(expected, os.X_OK):
        _NATIVE_714_BIN = expected.resolve()
        return _NATIVE_714_BIN
    _NATIVE_714_BIN = False
    return None


def native_714_unsupported_reason(
    *,
    input_order: list[str],
    placements: dict[str, Placement],
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    automation_lanes: AutomationLanes | None,
) -> str | None:
    return None


def _native_714_pan_rows_from_exprs(pan: dict[str, str]) -> list[list[float]]:
    rows = [[0.0, 0.0] for _channel in CHANNELS_714]
    for output_channel, expression in pan.items():
        output_index = _NATIVE_714_OUTPUT_CHANNELS.get(output_channel)
        if output_index is None:
            raise ValueError(f"unsupported 7.1.4 output channel: {output_channel}")
        for raw_coeff, input_channel in PAN_TERM_RE.findall(expression):
            input_index = _NATIVE_714_INPUT_CHANNELS.get(input_channel)
            if input_index is None:
                raise ValueError(
                    f"unsupported native 7.1.4 pan input channel: {input_channel}"
                )
            rows[output_index][input_index] += float(raw_coeff)
    return rows


def _native_714_pan_rows(placement: Placement) -> list[list[float]]:
    return _native_714_pan_rows_from_exprs(placement.pan)


def _native_714_gain_segments(
    times: tuple[float, ...],
    values: tuple[float, ...],
    *,
    step_sec: float,
    max_gain_db: float,
    min_gain_db: float = 0.05,
) -> list[dict[str, float | int]]:
    segments = sampled_automation_segments(
        times, values, step_sec=step_sec, max_gain_db=max_gain_db
    )
    if not segments or max(abs(gain_db) for _start, _end, gain_db in segments) < max(
        0.0, float(min_gain_db)
    ):
        return []
    sample_rate = 48000.0
    return [
        {
            "start_frame": max(0, int(round(start_sec * sample_rate))),
            "end_frame": max(0, int(round(end_sec * sample_rate))),
            "gain": db_to_amp(gain_db),
        }
        for start_sec, end_sec, gain_db in segments
        if end_sec > start_sec
    ]


def _native_714_low_tighten_segments(
    stem: str, lanes: AutomationLanes | None
) -> list[dict[str, float | int]]:
    if lanes is None or stem != "drums":
        return []
    return _native_714_gain_segments(
        lanes.times,
        lanes.low_tighten,
        step_sec=lanes.step_sec,
        max_gain_db=-0.55,
        min_gain_db=0.12,
    )


def _native_714_mid_decongest_segments(
    stem: str, lanes: AutomationLanes | None
) -> list[dict[str, float | int]]:
    if lanes is None or stem not in {"guitar", "piano", "other"}:
        return []
    return _native_714_gain_segments(
        lanes.times,
        temporal_mid_decongest_values(stem, lanes),
        step_sec=lanes.step_sec,
        max_gain_db=-0.75,
        min_gain_db=0.15,
    )


def _native_714_space_bed_plan(
    stem: str,
    placement: Placement,
    *,
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    space_bed_strength: float,
    automation_lanes: AutomationLanes | None,
) -> dict[str, object] | None:
    if not should_use_space_bed(stem, placement, space_bed_mode, space_bed_stems):
        return None
    _direct_pan, space_pan = split_direct_space_pan(placement, stem)
    if not pan_has_signal(space_pan):
        return None
    strength = clamp(space_bed_strength, 0.0, 1.0)
    left_delay, right_delay = space_bed_delay_pair_ms(stem, strength)
    wet_gain = 0.22 + 0.12 * strength if stem == "drums" else 0.68 + 0.38 * strength
    max_gain_db = 0.75 if stem == "drums" else 1.25
    automation = (
        _native_714_gain_segments(
            automation_lanes.times,
            temporal_space_bed_values(stem, automation_lanes),
            step_sec=automation_lanes.step_sec,
            max_gain_db=max_gain_db,
        )
        if automation_lanes is not None
        else []
    )
    return {
        "pan": _native_714_pan_rows_from_exprs(space_pan),
        "left_delay_ms": left_delay,
        "right_delay_ms": right_delay,
        "wet_gain": wet_gain,
        "allpass_mix": 0.35 + 0.25 * strength,
        "automation": automation,
    }


def _native_714_compressor_plan(
    *,
    threshold: float,
    ratio: float,
    attack_ms: float,
    release_ms: float,
    mix: float,
    link: str,
) -> dict[str, float | str]:
    return {
        "threshold": float(threshold),
        "ratio": float(ratio),
        "attack_ms": float(attack_ms),
        "release_ms": float(release_ms),
        "mix": float(mix),
        "link": link,
    }


def _native_714_focus_duck_compressor(
    decision: FocusDecision,
) -> dict[str, float | str]:
    strength = clamp(decision.strength, 0.0, 1.0)
    return _native_714_compressor_plan(
        threshold=0.110 - 0.025 * strength,
        ratio=1.05 + 0.45 * strength,
        attack_ms=35.0,
        release_ms=170.0 + 40.0 * strength,
        mix=0.18 + 0.17 * strength,
        link="maximum",
    )


def _native_714_focus_bed_compressor(
    stem: str, decision: FocusDecision, target: FocusDuckTarget
) -> dict[str, float | str]:
    strength = clamp(decision.strength, 0.0, 1.0)
    protection = clamp(target.temporal_protection, 0.0, 1.0)
    duck_boost = max(0.0, focus_target_duck_scale(decision, target) - 1.0)
    release = (260.0 + 220.0 * strength + 70.0 * duck_boost) * (
        1.0 - 0.30 * protection
    )
    return _native_714_compressor_plan(
        threshold=0.125 - 0.025 * strength - 0.012 * duck_boost + 0.018 * protection,
        ratio=(
            (
                1.05
                + 0.50
                * strength
                * clamp(0.50 + target.overlap_score, 0.50, 1.25)
            )
            * (1.0 + 0.18 * duck_boost)
            * (1.0 - 0.16 * protection)
        ),
        attack_ms=32.0 + 38.0 * protection,
        release_ms=max(110.0, release),
        mix=focus_bed_duck_mix(stem, decision, target),
        link="average",
    )


def _native_714_priority_duck_compressor(
    target: PriorityDuckTarget,
) -> dict[str, float | str]:
    strength = clamp(target.duck_db / 0.8, 0.0, 1.0)
    return _native_714_compressor_plan(
        threshold=0.135 - 0.030 * strength,
        ratio=1.05
        + 0.55 * strength * clamp(0.65 + target.overlap_score, 0.65, 1.35),
        attack_ms=28.0,
        release_ms=220.0 + 180.0 * strength,
        mix=0.14 + 0.26 * strength,
        link="average",
    )


def _native_714_sidechain_sources(
    stems: dict[str, Path], weighted_stems: list[tuple[str, float]]
) -> list[dict[str, object]]:
    return [
        {"path": str(stems[stem]), "gain": float(weight)}
        for stem, weight in weighted_stems
        if stem in stems and abs(weight) >= 1.0e-9
    ]


def _native_714_focus_sidechain(
    input_order: list[str],
    stems: dict[str, Path],
    focus_decision: FocusDecision,
) -> tuple[list[dict[str, object]], float]:
    weighted = [
        (stem, weight)
        for stem, weight in ordered_focus_weights(focus_decision)
        if stem in input_order and weight >= 0.05
    ]
    sources = _native_714_sidechain_sources(stems, weighted)
    if len(sources) <= 1:
        return sources, 1.0
    weight_power = sum(weight * weight for _stem, weight in weighted)
    return sources, 1.0 / max(math.sqrt(weight_power), 1e-6)


def _native_714_priority_sidechain_filter(bands: tuple[str, ...]) -> str:
    if bands == ("low",):
        return "low180"
    if bands == ("mid",):
        return "mid180_2500"
    if bands == ("high",):
        return "high2500"
    if bands == ("low", "mid"):
        return "low2500"
    if bands == ("mid", "high"):
        return "high180"
    return ""


def _native_714_ducking_ops(
    *,
    stem: str,
    input_order: list[str],
    stems: dict[str, Path],
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
) -> list[dict[str, object]]:
    ops: list[dict[str, object]] = []
    focus_stem = focus_decision.stem
    vocal_anchor = focus_decision.vocal_anchor
    if (
        focus_stem is not None
        and vocal_anchor is not None
        and focus_decision.vocal_duck_db > 0.0
        and focus_stem != vocal_anchor
        and stem == focus_stem
        and vocal_anchor in stems
    ):
        ops.append(
            {
                "mode": "full",
                "sidechains": _native_714_sidechain_sources(
                    stems, [(vocal_anchor, 1.0)]
                ),
                "normalizer": 1.0,
                "sidechain_filter": "",
                "compressor": _native_714_focus_duck_compressor(focus_decision),
            }
        )

    focus_sources, focus_normalizer = _native_714_focus_sidechain(
        input_order, stems, focus_decision
    )
    if (
        focus_sources
        and focus_decision.bed_duck_db > 0.0
        and stem in focus_decision.bed_duck_targets
        and should_duck_for_focus(stem, focus_decision)
    ):
        target = focus_decision.bed_duck_targets[stem]
        if focus_decision.stem == "bass" and stem == "drums" and target.bass_drum_bands:
            bands = []
            for guard_band in target.bass_drum_bands:
                band_target = FocusDuckTarget(
                    stem=target.stem,
                    bands=("low",),
                    overlap_score=guard_band.overlap_score,
                    duck_db=guard_band.duck_db,
                )
                sidechain_boost = 1.15 + 0.35 * max(
                    0.0, focus_target_duck_scale(focus_decision, band_target) - 1.0
                )
                bands.append(
                    {
                        "band": guard_band.name,
                        "sidechain_gain": sidechain_boost,
                        "compressor": _native_714_focus_bed_compressor(
                            stem, focus_decision, band_target
                        ),
                    }
                )
            if bands:
                ops.append(
                    {
                        "mode": "bands5",
                        "sidechains": focus_sources,
                        "normalizer": focus_normalizer,
                        "sidechain_filter": "",
                        "bands": bands,
                    }
                )
        else:
            bands = []
            for band in BAND_NAMES:
                if band not in target.bands:
                    continue
                sidechain_gain = 1.0
                if focus_decision.stem == "bass" and band == "low":
                    sidechain_gain = 1.25 + 0.45 * max(
                        0.0, focus_target_duck_scale(focus_decision, target) - 1.0
                    )
                bands.append(
                    {
                        "band": band,
                        "sidechain_gain": sidechain_gain,
                        "compressor": _native_714_focus_bed_compressor(
                            stem, focus_decision, target
                        ),
                    }
                )
            if bands:
                ops.append(
                    {
                        "mode": "bands3",
                        "sidechains": focus_sources,
                        "normalizer": focus_normalizer,
                        "sidechain_filter": "",
                        "bands": bands,
                    }
                )

    for target in priority_duck_targets.get(stem, ()):
        if target.protector not in stems:
            continue
        ops.append(
            {
                "mode": "full",
                "sidechains": _native_714_sidechain_sources(
                    stems, [(target.protector, 1.0)]
                ),
                "normalizer": 1.0,
                "sidechain_filter": _native_714_priority_sidechain_filter(
                    target.bands
                ),
                "compressor": _native_714_priority_duck_compressor(target),
            }
        )
    return ops


def _native_714_plan(
    *,
    input_order: list[str],
    stems: dict[str, Path],
    placements: dict[str, Placement],
    output_path: Path,
    master_gain: float,
    stem_gain_db: dict[str, float],
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    space_bed_strength: float,
    automation_lanes: AutomationLanes | None,
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
) -> dict[str, object]:
    return {
        "output": str(output_path),
        "master_gain": float(master_gain),
        "stems": [
            {
                "path": str(stems[stem]),
                "gain": db_to_amp(stem_gain_db.get(stem, 0.0)),
                "pan": _native_714_pan_rows(placements[stem]),
                "lfe_amount": float(placements[stem].lfe_amount),
                "lfe_cutoff_hz": float(placements[stem].lfe_cutoff_hz),
                "eq": [
                    {
                        "frequency_hz": float(band.frequency_hz),
                        "width_octaves": float(band.width_octaves),
                        "gain_db": float(band.gain_db),
                    }
                    for band in stem_eq_bands.get(stem, ())
                    if abs(band.gain_db) >= 0.05
                ],
                "low_tighten": _native_714_low_tighten_segments(
                    stem, automation_lanes
                ),
                "mid_decongest": _native_714_mid_decongest_segments(
                    stem, automation_lanes
                ),
                "space_bed": _native_714_space_bed_plan(
                    stem,
                    placements[stem],
                    space_bed_mode=space_bed_mode,
                    space_bed_stems=space_bed_stems,
                    space_bed_strength=space_bed_strength,
                    automation_lanes=automation_lanes,
                ),
                "ducking": _native_714_ducking_ops(
                    stem=stem,
                    input_order=input_order,
                    stems=stems,
                    focus_decision=focus_decision,
                    priority_duck_targets=priority_duck_targets,
                ),
            }
            for stem in input_order
        ],
    }


def render_714_native(
    *,
    input_order: list[str],
    stems: dict[str, Path],
    placements: dict[str, Placement],
    output_path: Path,
    master_gain: float,
    stem_gain_db: dict[str, float],
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    space_bed_strength: float,
    automation_lanes: AutomationLanes | None,
    strict: bool,
) -> bool:
    reason = native_714_unsupported_reason(
        input_order=input_order,
        placements=placements,
        focus_decision=focus_decision,
        priority_duck_targets=priority_duck_targets,
        space_bed_mode=space_bed_mode,
        space_bed_stems=space_bed_stems,
        automation_lanes=automation_lanes,
    )
    if reason is not None:
        if strict:
            raise RuntimeError(f"native 7.1.4 render requested but unsupported: {reason}")
        print(f"7.1.4 native render unavailable: {reason}. Falling back to FFmpeg.", flush=True)
        return False

    binary = native_714_binary()
    if binary is None:
        if strict:
            raise RuntimeError("native 7.1.4 render requested but helper was not found")
        return False

    try:
        plan = _native_714_plan(
            input_order=input_order,
            stems=stems,
            placements=placements,
            output_path=output_path,
            master_gain=master_gain,
            stem_gain_db=stem_gain_db,
            stem_eq_bands=stem_eq_bands,
            space_bed_mode=space_bed_mode,
            space_bed_stems=space_bed_stems,
            space_bed_strength=space_bed_strength,
            automation_lanes=automation_lanes,
            focus_decision=focus_decision,
            priority_duck_targets=priority_duck_targets,
        )
    except Exception as exc:
        if strict:
            raise RuntimeError(f"native 7.1.4 plan failed: {exc}") from exc
        print(f"7.1.4 native render unavailable: {exc}. Falling back to FFmpeg.", flush=True)
        return False

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        suffix=".json",
        prefix="stemdeck_714_plan_",
        delete=False,
    ) as handle:
        plan_path = Path(handle.name)
        json.dump(plan, handle)
        handle.write("\n")
    timer = time.perf_counter()
    try:
        run([str(binary), "render714", "--plan", str(plan_path)])
        print_elapsed("7.1.4 native render", timer)
        return True
    except SystemExit as exc:
        output_path.unlink(missing_ok=True)
        if strict:
            raise RuntimeError(f"native 7.1.4 render failed: {exc}") from exc
        print(
            f"7.1.4 native render failed: {exc}. Falling back to FFmpeg.",
            flush=True,
        )
        return False
    finally:
        plan_path.unlink(missing_ok=True)


def focus_duck_filter(decision: FocusDecision) -> str:
    strength = clamp(decision.strength, 0.0, 1.0)
    threshold = 0.110 - 0.025 * strength
    ratio = 1.05 + 0.45 * strength
    release = 170.0 + 40.0 * strength
    mix = 0.18 + 0.17 * strength
    return (
        "sidechaincompress="
        f"threshold={fmt(threshold)}:"
        f"ratio={fmt(ratio)}:"
        "attack=35:"
        f"release={fmt(release)}:"
        "knee=6:"
        "link=maximum:"
        "detection=rms:"
        f"mix={fmt(mix)}"
    )


def focus_target_duck_scale(decision: FocusDecision, target: FocusDuckTarget) -> float:
    nominal_duck = max(decision.bed_duck_db, 0.25)
    return clamp(target.duck_db / nominal_duck, 0.75, 1.65)


def focus_bed_duck_mix(
    stem: str, decision: FocusDecision, target: FocusDuckTarget
) -> float:
    strength = clamp(decision.strength, 0.0, 1.0)
    protection = clamp(target.temporal_protection, 0.0, 1.0)
    base_mix = (0.14 + 0.20 * strength) * clamp(
        0.55 + 0.65 * target.overlap_score, 0.35, 1.0
    )
    duck_boost = max(0.0, focus_target_duck_scale(decision, target) - 1.0)
    base_mix *= 1.0 + 0.22 * duck_boost
    base_mix *= 1.0 - 0.42 * protection
    if stem == "drums":
        return base_mix * 0.55
    if stem == "bass":
        return base_mix * 0.70
    return base_mix


def focus_bed_duck_filter(
    stem: str, decision: FocusDecision, target: FocusDuckTarget
) -> str:
    strength = clamp(decision.strength, 0.0, 1.0)
    protection = clamp(target.temporal_protection, 0.0, 1.0)
    duck_boost = max(0.0, focus_target_duck_scale(decision, target) - 1.0)
    threshold = 0.125 - 0.025 * strength - 0.012 * duck_boost + 0.018 * protection
    ratio = (
        (1.05 + 0.50 * strength * clamp(0.50 + target.overlap_score, 0.50, 1.25))
        * (1.0 + 0.18 * duck_boost)
        * (1.0 - 0.16 * protection)
    )
    attack = 32.0 + 38.0 * protection
    release = (260.0 + 220.0 * strength + 70.0 * duck_boost) * (1.0 - 0.30 * protection)
    release = max(110.0, release)
    return (
        "sidechaincompress="
        f"threshold={fmt(threshold)}:"
        f"ratio={fmt(ratio)}:"
        f"attack={fmt(attack)}:"
        f"release={fmt(release)}:"
        "knee=6:"
        "link=average:"
        "detection=rms:"
        f"mix={fmt(focus_bed_duck_mix(stem, decision, target))}"
    )


def priority_duck_filter(target: PriorityDuckTarget) -> str:
    strength = clamp(target.duck_db / 0.8, 0.0, 1.0)
    threshold = 0.135 - 0.030 * strength
    ratio = 1.05 + 0.55 * strength * clamp(0.65 + target.overlap_score, 0.65, 1.35)
    release = 220.0 + 180.0 * strength
    mix = 0.14 + 0.26 * strength
    return (
        "sidechaincompress="
        f"threshold={fmt(threshold)}:"
        f"ratio={fmt(ratio)}:"
        "attack=28:"
        f"release={fmt(release)}:"
        "knee=6:"
        "link=average:"
        "detection=rms:"
        f"mix={fmt(mix)}"
    )


def priority_sidechain_filters(bands: tuple[str, ...]) -> list[str]:
    if bands == ("low",):
        return ["lowpass=f=180"]
    if bands == ("mid",):
        return ["highpass=f=180", "lowpass=f=2500"]
    if bands == ("high",):
        return ["highpass=f=2500"]
    if bands == ("low", "mid"):
        return ["lowpass=f=2500"]
    if bands == ("mid", "high"):
        return ["highpass=f=180"]
    return []


def append_bass_drum_multiband_duck(
    parts: list[str],
    *,
    source_label: str,
    gain: str,
    label: str,
    stem: str,
    focus_sidechain_source: str,
    decision: FocusDecision,
    target: FocusDuckTarget,
) -> str:
    target_band_labels = {
        band: f"{label}_{band}_bdguard_target" for band in BASS_DRUM_CROSSOVER_BANDS
    }
    focus_band_labels = {
        band: f"{label}_{band}_bdguard_focus" for band in BASS_DRUM_CROSSOVER_BANDS
    }
    parts.append(
        f"{source_label}aresample=48000{gain},acrossover=split=60 120 180 2500:order=4th"
        + "".join(f"[{target_band_labels[band]}]" for band in BASS_DRUM_CROSSOVER_BANDS)
    )
    parts.append(
        f"{focus_sidechain_source}aresample=48000,acrossover=split=60 120 180 2500:order=4th"
        + "".join(f"[{focus_band_labels[band]}]" for band in BASS_DRUM_CROSSOVER_BANDS)
    )

    guard_bands = {band.name: band for band in target.bass_drum_bands}
    output_labels: list[str] = []
    for band in BASS_DRUM_CROSSOVER_BANDS:
        target_band_label = target_band_labels[band]
        focus_band_label = focus_band_labels[band]
        guard_band = guard_bands.get(band)
        if guard_band is None:
            output_labels.append(f"[{target_band_label}]")
            parts.append(f"[{focus_band_label}]anullsink")
            continue

        band_target = FocusDuckTarget(
            stem=target.stem,
            bands=("low",),
            overlap_score=guard_band.overlap_score,
            duck_db=guard_band.duck_db,
        )
        sidechain_label = f"{label}_{band}_bdguard_sc"
        ducked_label = f"{label}_{band}_bdguard_ducked"
        sidechain_boost = 1.15 + 0.35 * max(
            0.0, focus_target_duck_scale(decision, band_target) - 1.0
        )
        parts.append(
            f"[{focus_band_label}]volume={fmt(sidechain_boost)},apad[{sidechain_label}]"
        )
        parts.append(
            f"[{target_band_label}][{sidechain_label}]"
            f"{focus_bed_duck_filter(stem, decision, band_target)}[{ducked_label}]"
        )
        output_labels.append(f"[{ducked_label}]")

    ducked_full_label = f"{label}_bdguarded"
    parts.append(
        "".join(output_labels)
        + f"amix=inputs={len(output_labels)}:normalize=0:duration=first:dropout_transition=0[{ducked_full_label}]"
    )
    return f"[{ducked_full_label}]"


def append_overlap_band_duck(
    parts: list[str],
    *,
    source_label: str,
    gain: str,
    label: str,
    stem: str,
    focus_sidechain_source: str,
    decision: FocusDecision,
) -> str:
    target = decision.bed_duck_targets[stem]
    if decision.stem == "bass" and stem == "drums" and target.bass_drum_bands:
        return append_bass_drum_multiband_duck(
            parts,
            source_label=source_label,
            gain=gain,
            label=label,
            stem=stem,
            focus_sidechain_source=focus_sidechain_source,
            decision=decision,
            target=target,
        )

    target_band_labels = {band: f"{label}_{band}_duckband" for band in BAND_NAMES}
    focus_band_labels = {band: f"{label}_{band}_focusband" for band in BAND_NAMES}
    parts.append(
        f"{source_label}aresample=48000{gain},acrossover=split=180 2500:order=4th"
        + "".join(f"[{target_band_labels[band]}]" for band in BAND_NAMES)
    )
    parts.append(
        f"{focus_sidechain_source}acrossover=split=180 2500:order=4th"
        + "".join(f"[{focus_band_labels[band]}]" for band in BAND_NAMES)
    )

    output_labels: list[str] = []
    for band in BAND_NAMES:
        target_band_label = target_band_labels[band]
        focus_band_label = focus_band_labels[band]
        if band not in target.bands:
            output_labels.append(f"[{target_band_label}]")
            parts.append(f"[{focus_band_label}]anullsink")
            continue
        sidechain_label = f"{label}_{band}_focus_sc"
        ducked_label = f"{label}_{band}_ducked"
        filters = ["volume=1"]
        if decision.stem == "bass" and band == "low":
            sidechain_boost = 1.25 + 0.45 * max(
                0.0, focus_target_duck_scale(decision, target) - 1.0
            )
            filters.append(f"volume={fmt(sidechain_boost)}")
        filters.append("apad")
        parts.append(f"[{focus_band_label}]{','.join(filters)}[{sidechain_label}]")
        parts.append(
            f"[{target_band_label}][{sidechain_label}]"
            f"{focus_bed_duck_filter(stem, decision, target)}[{ducked_label}]"
        )
        output_labels.append(f"[{ducked_label}]")

    ducked_full_label = f"{label}_bedducked"
    parts.append(
        "".join(output_labels)
        + f"amix=inputs=3:normalize=0:duration=first:dropout_transition=0[{ducked_full_label}]"
    )
    return f"[{ducked_full_label}]"


def append_priority_masking_duck(
    parts: list[str],
    *,
    source_label: str,
    gain: str,
    label: str,
    target: PriorityDuckTarget,
    sidechain_source: str,
    sidechain_pad_duration: float | None,
) -> str:
    protector_label = safe_label(target.protector)
    preduck_label = f"{label}_prio_{protector_label}_preduck"
    sidechain_label = f"{label}_prio_{protector_label}_sc"
    ducked_label = f"{label}_prio_{protector_label}_ducked"
    sidechain_filters = priority_sidechain_filters(target.bands)
    sidechain_filters.append("volume=1")
    if sidechain_pad_duration is not None:
        sidechain_filters.append(
            f"apad=whole_dur={fmt(max(0.001, sidechain_pad_duration))}"
        )
    parts.append(f"{source_label}aresample=48000{gain}[{preduck_label}]")
    parts.append(f"{sidechain_source}{','.join(sidechain_filters)}[{sidechain_label}]")
    parts.append(
        f"[{preduck_label}][{sidechain_label}]{priority_duck_filter(target)}[{ducked_label}]"
    )
    return f"[{ducked_label}]"


def pan_has_signal(values: dict[str, str]) -> bool:
    for expr in values.values():
        for raw_coeff, _input_channel in PAN_TERM_RE.findall(expr):
            if abs(float(raw_coeff)) >= 1e-9:
                return True
    return False


def space_bed_channels(stem: str) -> set[str]:
    channels = set(AMBIENCE_CHANNELS)
    if stem in SIDE_AMBIENCE_STEMS:
        channels.update({"SL", "SR"})
    return channels


def split_direct_space_pan(
    placement: Placement, stem: str
) -> tuple[dict[str, str], dict[str, str]]:
    ambience_channels = space_bed_channels(stem)
    direct = {
        channel: expr
        for channel, expr in placement.pan.items()
        if channel not in ambience_channels
    }
    space = {
        channel: expr
        for channel, expr in placement.pan.items()
        if channel in ambience_channels
    }
    return direct, space


def should_use_space_bed(
    stem: str, placement: Placement, mode: str, stems: tuple[str, ...]
) -> bool:
    if mode == "off" or stem not in stems:
        return False
    _direct_pan, space_pan = split_direct_space_pan(placement, stem)
    return pan_has_signal(space_pan)


def space_bed_delay_pair_ms(stem: str, strength: float) -> tuple[float, float]:
    base = {
        "drums": 12.0,
        "guitar": 24.0,
        "piano": 30.0,
        "other": 36.0,
    }.get(stem, 26.0)
    spread = 10.0 + 6.0 * clamp(strength, 0.0, 1.0)
    return base, base + spread


def space_bed_filter_chain(stem: str, strength: float) -> str:
    strength = clamp(strength, 0.0, 1.0)
    left_delay, right_delay = space_bed_delay_pair_ms(stem, strength)
    wet_gain = 0.22 + 0.12 * strength if stem == "drums" else 0.68 + 0.38 * strength
    allpass_mix = 0.35 + 0.25 * strength
    return ",".join(
        [
            "pan=stereo|FL=0.5*FL-0.5*FR|FR=0.5*FR-0.5*FL",
            f"adelay={fmt(left_delay)}|{fmt(right_delay)}",
            f"allpass=f=720:width_type=h:width=420:mix={fmt(allpass_mix)}",
            f"allpass=f=1650:width_type=h:width=900:mix={fmt(allpass_mix * 0.75)}",
            f"volume={fmt(wet_gain)}",
        ]
    )


def sampled_automation_segments(
    times: tuple[float, ...],
    values: tuple[float, ...],
    *,
    step_sec: float,
    max_gain_db: float,
) -> list[tuple[float, float, float]]:
    if not times or not values or abs(max_gain_db) <= 0.0:
        return []
    count = min(len(times), len(values))
    hop_sec = times[1] - times[0] if len(times) > 1 else step_sec
    hop_sec = max(hop_sec, 0.05)
    sample_step = max(1, int(round(max(step_sec, hop_sec) / hop_sec)))
    while math.ceil(count / sample_step) > 120:
        sample_step *= 2
    indices = list(range(0, count, sample_step))
    if indices[-1] != count - 1:
        indices.append(count - 1)

    segments: list[tuple[float, float, float]] = []
    for offset, index in enumerate(indices):
        start = 0.0 if offset == 0 else times[index]
        if offset + 1 < len(indices):
            end = times[indices[offset + 1]]
        else:
            end = times[count - 1] + max(step_sec, hop_sec)
        gain_db = round(clamp(values[index], 0.0, 1.0) * max_gain_db / 0.10) * 0.10
        if segments and abs(segments[-1][2] - gain_db) < 1e-9:
            prev_start, _prev_end, prev_gain = segments[-1]
            segments[-1] = (prev_start, end, prev_gain)
        else:
            segments.append((start, end, gain_db))
    return segments


def automation_volume_filter(
    times: tuple[float, ...],
    values: tuple[float, ...],
    *,
    step_sec: float,
    max_gain_db: float,
    min_gain_db: float = 0.05,
) -> str:
    segments = sampled_automation_segments(
        times, values, step_sec=step_sec, max_gain_db=max_gain_db
    )
    if not segments or max(abs(gain_db) for _start, _end, gain_db in segments) < max(
        0.0, float(min_gain_db)
    ):
        return ""
    expression = fmt(db_to_amp(segments[-1][2]))
    for start_sec, end_sec, gain_db in reversed(segments[:-1]):
        gain = fmt(db_to_amp(gain_db))
        expression = f"if(between(t\\,{fmt(start_sec)}\\,{fmt(end_sec)})\\,{gain}\\,{expression})"
    return f",volume={expression}:eval=frame"


def temporal_space_bed_values(stem: str, lanes: AutomationLanes) -> tuple[float, ...]:
    values: list[float] = []
    solo_values = lanes.solo_feature.get(stem, ())
    for index, chorus in enumerate(lanes.chorus_space):
        breakdown = (
            lanes.breakdown_air[index] if index < len(lanes.breakdown_air) else 0.0
        )
        solo = solo_values[index] if index < len(solo_values) else 0.0
        vocal_protect = (
            lanes.vocal_protect[index] if index < len(lanes.vocal_protect) else 0.0
        )
        low_tighten = (
            lanes.low_tighten[index] if index < len(lanes.low_tighten) else 0.0
        )
        mid_decongest = (
            lanes.mid_decongest[index] if index < len(lanes.mid_decongest) else 0.0
        )
        harshness = (
            lanes.harshness_guard[index] if index < len(lanes.harshness_guard) else 0.0
        )
        fold_safety = (
            lanes.fold_safety[index] if index < len(lanes.fold_safety) else 0.0
        )
        if stem == "drums":
            value = 0.35 * chorus * (1.0 - 0.40 * low_tighten)
        elif stem == "vocals":
            value = 0.18 * chorus
        else:
            value = 0.78 * chorus + 0.52 * breakdown + 0.42 * solo
            value *= 1.0 - 0.18 * vocal_protect
            value *= 1.0 - 0.24 * mid_decongest
        value *= 1.0 - 0.45 * harshness
        value *= 1.0 - 0.24 * fold_safety
        values.append(clamp(value, 0.0, 1.0))
    return tuple(values)


def temporal_space_bed_volume_filter(stem: str, lanes: AutomationLanes | None) -> str:
    if lanes is None:
        return ""
    values = temporal_space_bed_values(stem, lanes)
    max_gain_db = 0.75 if stem == "drums" else 1.25
    return automation_volume_filter(
        lanes.times, values, step_sec=lanes.step_sec, max_gain_db=max_gain_db
    )


def temporal_mid_decongest_values(
    stem: str, lanes: AutomationLanes
) -> tuple[float, ...]:
    weight = {"guitar": 0.85, "piano": 0.80, "other": 1.0}.get(stem, 0.0)
    if weight <= 0.0:
        return ()
    solo_values = lanes.solo_feature.get(stem, ())
    values: list[float] = []
    for index, mid in enumerate(lanes.mid_decongest):
        vocal_protect = (
            lanes.vocal_protect[index] if index < len(lanes.vocal_protect) else 0.0
        )
        fold_safety = (
            lanes.fold_safety[index] if index < len(lanes.fold_safety) else 0.0
        )
        solo = solo_values[index] if index < len(solo_values) else 0.0
        value = mid * weight * (0.55 + 0.45 * vocal_protect) * (1.0 - 0.58 * solo)
        value = max(value, fold_safety * weight * 0.18)
        values.append(clamp(value, 0.0, 1.0))
    return tuple(values)


def append_temporal_mid_decongest(
    parts: list[str],
    *,
    source_label: str,
    gain: str,
    label: str,
    stem: str,
    lanes: AutomationLanes | None,
) -> str:
    if lanes is None or stem not in {"guitar", "piano", "other"}:
        return source_label
    values = temporal_mid_decongest_values(stem, lanes)
    decongest_filter = automation_volume_filter(
        lanes.times,
        values,
        step_sec=lanes.step_sec,
        max_gain_db=-0.75,
        min_gain_db=0.15,
    )
    if not decongest_filter:
        return source_label
    low_label = f"{label}_intent_low"
    mid_label = f"{label}_intent_mid"
    high_label = f"{label}_intent_high"
    ducked_mid_label = f"{label}_intent_mid_decongest"
    out_label = f"{label}_intent_decongested"
    parts.append(
        f"{source_label}aresample=48000{gain},acrossover=split=700 3500:order=4th"
        f"[{low_label}][{mid_label}][{high_label}]"
    )
    parts.append(f"[{mid_label}]{decongest_filter.lstrip(',')}[{ducked_mid_label}]")
    parts.append(
        f"[{low_label}][{ducked_mid_label}][{high_label}]"
        f"amix=inputs=3:normalize=0:duration=first:dropout_transition=0[{out_label}]"
    )
    return f"[{out_label}]"


def append_temporal_low_tighten(
    parts: list[str],
    *,
    source_label: str,
    gain: str,
    label: str,
    stem: str,
    lanes: AutomationLanes | None,
) -> str:
    if lanes is None or stem != "drums":
        return source_label
    low_filter = automation_volume_filter(
        lanes.times,
        lanes.low_tighten,
        step_sec=lanes.step_sec,
        max_gain_db=-0.55,
        min_gain_db=0.12,
    )
    if not low_filter:
        return source_label
    low_label = f"{label}_intent_lowtight_low"
    high_label = f"{label}_intent_lowtight_high"
    tight_low_label = f"{label}_intent_lowtight_ducked"
    out_label = f"{label}_intent_lowtight"
    parts.append(
        f"{source_label}aresample=48000{gain},acrossover=split=180:order=4th"
        f"[{low_label}][{high_label}]"
    )
    parts.append(f"[{low_label}]{low_filter.lstrip(',')}[{tight_low_label}]")
    parts.append(
        f"[{tight_low_label}][{high_label}]"
        f"amix=inputs=2:normalize=0:duration=first:dropout_transition=0[{out_label}]"
    )
    return f"[{out_label}]"


def stereo_temporal_low_tighten_values(lanes: AutomationLanes) -> tuple[float, ...]:
    values: list[float] = []
    for index, low_tighten in enumerate(lanes.low_tighten):
        fold_safety = (
            lanes.fold_safety[index] if index < len(lanes.fold_safety) else 0.0
        )
        chorus = lanes.chorus_space[index] if index < len(lanes.chorus_space) else 0.0
        low_anchor = lanes.low_anchor[index] if index < len(lanes.low_anchor) else 0.0
        value = max(low_tighten, fold_safety * 0.55, chorus * low_anchor * 0.28)
        values.append(clamp(value, 0.0, 1.0))
    return tuple(values)


def stereo_temporal_mid_decongest_values(lanes: AutomationLanes) -> tuple[float, ...]:
    values: list[float] = []
    solo_lanes = tuple(lanes.solo_feature.values())
    for index, mid_decongest in enumerate(lanes.mid_decongest):
        vocal_protect = (
            lanes.vocal_protect[index] if index < len(lanes.vocal_protect) else 0.0
        )
        fold_safety = (
            lanes.fold_safety[index] if index < len(lanes.fold_safety) else 0.0
        )
        solo_peak = max(
            (lane[index] for lane in solo_lanes if index < len(lane)), default=0.0
        )
        value = max(mid_decongest * (0.55 + 0.45 * vocal_protect), fold_safety * 0.18)
        value *= 1.0 - 0.30 * solo_peak
        values.append(clamp(value, 0.0, 1.0))
    return tuple(values)


def stereo_temporal_harshness_guard_values(lanes: AutomationLanes) -> tuple[float, ...]:
    values: list[float] = []
    for index, harshness_guard in enumerate(lanes.harshness_guard):
        fold_safety = (
            lanes.fold_safety[index] if index < len(lanes.fold_safety) else 0.0
        )
        breakdown = (
            lanes.breakdown_air[index] if index < len(lanes.breakdown_air) else 0.0
        )
        value = max(harshness_guard, fold_safety * 0.34)
        value *= 1.0 - 0.18 * breakdown
        values.append(clamp(value, 0.0, 1.0))
    return tuple(values)


def append_stereo_temporal_mastering(
    parts: list[str],
    *,
    source_label: str,
    output_label: str,
    lanes: AutomationLanes | None,
    prefix: str,
) -> None:
    if lanes is None:
        parts.append(
            f"{source_label}aresample=48000,aformat=channel_layouts=stereo{output_label}"
        )
        return

    low_filter = automation_volume_filter(
        lanes.times,
        stereo_temporal_low_tighten_values(lanes),
        step_sec=lanes.step_sec,
        max_gain_db=-0.45,
        min_gain_db=0.12,
    )
    mid_filter = automation_volume_filter(
        lanes.times,
        stereo_temporal_mid_decongest_values(lanes),
        step_sec=lanes.step_sec,
        max_gain_db=-0.50,
        min_gain_db=0.12,
    )
    high_filter = automation_volume_filter(
        lanes.times,
        stereo_temporal_harshness_guard_values(lanes),
        step_sec=lanes.step_sec,
        max_gain_db=-0.38,
        min_gain_db=0.10,
    )
    if not low_filter and not mid_filter and not high_filter:
        parts.append(
            f"{source_label}aresample=48000,aformat=channel_layouts=stereo{output_label}"
        )
        return

    low_label = f"{prefix}_tm_low"
    lowmid_label = f"{prefix}_tm_lowmid"
    mid_label = f"{prefix}_tm_mid"
    high_label = f"{prefix}_tm_high"
    parts.append(
        f"{source_label}aresample=48000,aformat=channel_layouts=stereo,"
        f"acrossover=split=180 700 3500:order=4th"
        f"[{low_label}][{lowmid_label}][{mid_label}][{high_label}]"
    )

    mix_labels = [
        f"[{low_label}]",
        f"[{lowmid_label}]",
        f"[{mid_label}]",
        f"[{high_label}]",
    ]
    for band_label, band_filter, suffix in (
        (low_label, low_filter, "lowtight"),
        (mid_label, mid_filter, "midclear"),
        (high_label, high_filter, "highguard"),
    ):
        if not band_filter:
            continue
        filtered_label = f"{prefix}_tm_{suffix}"
        parts.append(f"[{band_label}]{band_filter.lstrip(',')}[{filtered_label}]")
        mix_index = {"lowtight": 0, "midclear": 2, "highguard": 3}[suffix]
        mix_labels[mix_index] = f"[{filtered_label}]"

    parts.append(
        "".join(mix_labels)
        + f"amix=inputs=4:normalize=0:duration=first:dropout_transition=0{output_label}"
    )


def stereo_fold_temporal_filter(
    input_label: str,
    output_label: str,
    lfe_fold_gain: float,
    automation_lanes: AutomationLanes | None,
    *,
    prefix: str,
    limiter: bool,
) -> str:
    if automation_lanes is None:
        return fold_714_to_stereo_filter(
            input_label, output_label, lfe_fold_gain, limiter=limiter
        )

    folded_label = f"{prefix}_fold_raw"
    temporal_label = f"{prefix}_temporal"
    parts = [
        fold_714_to_stereo_filter(
            input_label, f"[{folded_label}]", lfe_fold_gain, limiter=False
        )
    ]
    temporal_output = f"[{temporal_label}]" if limiter else output_label
    append_stereo_temporal_mastering(
        parts,
        source_label=f"[{folded_label}]",
        output_label=temporal_output,
        lanes=automation_lanes,
        prefix=prefix,
    )
    if limiter:
        parts.append(f"[{temporal_label}]alimiter=limit=0.98{output_label}")
    return ";".join(parts)


def build_714_filter(
    input_order: list[str],
    placements: dict[str, Placement],
    master_gain: float,
    stem_gain_db: dict[str, float],
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
    priority_sidechain_duration: float | None,
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    space_bed_strength: float,
    automation_lanes: AutomationLanes | None,
) -> str:
    parts: list[str] = []
    mix_labels: list[str] = []
    focus_stem = focus_decision.stem
    vocal_anchor = focus_decision.vocal_anchor
    focus_sidechain_stems = [
        (stem, weight)
        for stem, weight in ordered_focus_weights(focus_decision)
        if stem in input_order and weight >= 0.05
    ]
    use_vocal_duck = (
        focus_stem is not None
        and vocal_anchor is not None
        and focus_decision.vocal_duck_db > 0.0
        and focus_stem != vocal_anchor
        and focus_stem in input_order
        and vocal_anchor in input_order
    )
    duck_target_names = [
        stem for stem in input_order if stem in focus_decision.bed_duck_targets
    ]
    use_focus_bed_duck = (
        bool(focus_sidechain_stems)
        and focus_decision.bed_duck_db > 0.0
        and bool(duck_target_names)
    )
    split_roles: dict[str, list[str]] = {stem: ["main"] for stem in input_order}
    if use_vocal_duck and vocal_anchor is not None:
        split_roles[vocal_anchor].append("vocal_sc")
    if use_focus_bed_duck:
        for stem, _weight in focus_sidechain_stems:
            split_roles[stem].append("focus_sc")
    priority_role_keys: dict[tuple[str, int], str] = {}
    for target_stem, target_list in priority_duck_targets.items():
        if target_stem not in input_order:
            continue
        for target_index, target in enumerate(target_list):
            if target.protector not in input_order:
                continue
            role = f"priority_sc_{safe_label(target_stem)}_{target_index}"
            split_roles.setdefault(target.protector, ["main"]).append(role)
            priority_role_keys[(target_stem, target_index)] = (
                f"{target.protector}:{role}"
            )

    source_labels: dict[str, str] = {}
    sidechain_labels: dict[str, str] = {}
    for index, stem in enumerate(input_order):
        roles = split_roles[stem]
        if len(roles) == 1:
            source_labels[stem] = f"[{index}:a]"
            continue
        label_base = safe_label(stem)
        role_labels = [f"{label_base}_{role}_src" for role in roles]
        parts.append(
            f"[{index}:a]aresample=48000,asplit={len(role_labels)}"
            + "".join(f"[{label}]" for label in role_labels)
        )
        for role, role_label in zip(roles, role_labels, strict=True):
            if role == "main":
                source_labels[stem] = f"[{role_label}]"
            else:
                sidechain_labels[f"{stem}:{role}"] = f"[{role_label}]"

    vocal_sidechain_label = (
        sidechain_labels.get(f"{vocal_anchor}:vocal_sc", "")
        if vocal_anchor is not None
        else ""
    )
    focus_sidechain_labels: dict[str, str] = {}
    if use_focus_bed_duck:
        weighted_focus_labels: list[str] = []
        weight_power = 0.0
        for stem, weight in focus_sidechain_stems:
            weighted_label = f"focusgrp_{safe_label(stem)}_weighted_sc"
            parts.append(
                f"{sidechain_labels[f'{stem}:focus_sc']}volume={fmt(weight)}[{weighted_label}]"
            )
            weighted_focus_labels.append(f"[{weighted_label}]")
            weight_power += weight * weight
        if len(weighted_focus_labels) == 1:
            focus_sc_source = weighted_focus_labels[0]
        else:
            focus_group_label = f"{safe_label(focus_label(focus_decision))}_focusgrp_sc"
            normalizer = 1.0 / max(math.sqrt(weight_power), 1e-6)
            parts.append(
                "".join(weighted_focus_labels)
                + f"amix=inputs={len(weighted_focus_labels)}:normalize=0:duration=first:dropout_transition=0,"
                f"volume={fmt(normalizer)}[{focus_group_label}]"
            )
            focus_sc_source = f"[{focus_group_label}]"
        if len(duck_target_names) == 1:
            focus_sidechain_labels[duck_target_names[0]] = focus_sc_source
        else:
            target_labels = {
                target_stem: f"{safe_label(focus_label(focus_decision))}_{safe_label(target_stem)}_bed_sc_src"
                for target_stem in duck_target_names
            }
            parts.append(
                f"{focus_sc_source}asplit={len(duck_target_names)}"
                + "".join(f"[{label}]" for label in target_labels.values())
            )
            focus_sidechain_labels = {
                target_stem: f"[{label}]"
                for target_stem, label in target_labels.items()
            }

    for stem in input_order:
        placement = placements[stem]
        gain = gain_filter(stem_gain_db.get(stem, 0.0)) + stem_eq_filter(
            stem_eq_bands.get(stem, ())
        )
        label = safe_label(stem)
        stem_label = f"{label}_bed"
        source_label = source_labels[stem]
        if use_vocal_duck and stem == focus_stem:
            preduck_label = f"{label}_preduck"
            vocal_sc_label = f"{label}_vocal_sc_apad"
            ducked_label = f"{label}_ducked"
            parts.append(f"{source_label}aresample=48000{gain}[{preduck_label}]")
            parts.append(f"{vocal_sidechain_label}apad[{vocal_sc_label}]")
            parts.append(
                f"[{preduck_label}][{vocal_sc_label}]{focus_duck_filter(focus_decision)}[{ducked_label}]"
            )
            source_label = f"[{ducked_label}]"
            gain = ""
        if use_focus_bed_duck and should_duck_for_focus(stem, focus_decision):
            source_label = append_overlap_band_duck(
                parts,
                source_label=source_label,
                gain=gain,
                label=label,
                stem=stem,
                focus_sidechain_source=focus_sidechain_labels[stem],
                decision=focus_decision,
            )
            gain = ""
        for target_index, priority_target in enumerate(
            priority_duck_targets.get(stem, ())
        ):
            sidechain_key = priority_role_keys.get((stem, target_index))
            if sidechain_key is None:
                continue
            priority_sidechain_source = sidechain_labels[sidechain_key]
            source_label = append_priority_masking_duck(
                parts,
                source_label=source_label,
                gain=gain,
                label=label,
                target=priority_target,
                sidechain_source=priority_sidechain_source,
                sidechain_pad_duration=priority_sidechain_duration,
            )
            gain = ""
        tightened_label = append_temporal_low_tighten(
            parts,
            source_label=source_label,
            gain=gain,
            label=label,
            stem=stem,
            lanes=automation_lanes,
        )
        if tightened_label != source_label:
            source_label = tightened_label
            gain = ""
        decongested_label = append_temporal_mid_decongest(
            parts,
            source_label=source_label,
            gain=gain,
            label=label,
            stem=stem,
            lanes=automation_lanes,
        )
        if decongested_label != source_label:
            source_label = decongested_label
            gain = ""
        use_space_bed = should_use_space_bed(
            stem, placement, space_bed_mode, space_bed_stems
        )
        _direct_pan, space_pan = split_direct_space_pan(placement, stem)
        if placement.lfe_amount > 0:
            full_label = f"{label}_full"
            low_label = f"{label}_low"
            lfe_label = f"{label}_lfe"
            if use_space_bed:
                space_src_label = f"{label}_space_src"
                space_label = f"{label}_space"
                parts.append(
                    f"{source_label}aresample=48000{gain},asplit=3"
                    f"[{full_label}][{space_src_label}][{low_label}]"
                )
                parts.append(f"[{full_label}]{pan_expr(placement.pan)}[{stem_label}]")
                mix_labels.append(f"[{stem_label}]")
                if pan_has_signal(space_pan):
                    space_chain = space_bed_filter_chain(stem, space_bed_strength)
                    space_chain += temporal_space_bed_volume_filter(
                        stem, automation_lanes
                    )
                    parts.append(
                        f"[{space_src_label}]{space_chain},"
                        f"{pan_expr(space_pan)}[{space_label}]"
                    )
                    mix_labels.append(f"[{space_label}]")
            else:
                parts.append(
                    f"{source_label}aresample=48000{gain},asplit=2[{full_label}][{low_label}]"
                )
                parts.append(f"[{full_label}]{pan_expr(placement.pan)}[{stem_label}]")
                mix_labels.append(f"[{stem_label}]")
            lfe_pan = {
                "LFE": f"{fmt(placement.lfe_amount)}*FL+{fmt(placement.lfe_amount)}*FR",
            }
            parts.append(
                f"[{low_label}]lowpass=f={placement.lfe_cutoff_hz},{pan_expr(lfe_pan)}[{lfe_label}]"
            )
            mix_labels.append(f"[{lfe_label}]")
        elif use_space_bed:
            direct_src_label = f"{label}_direct_src"
            space_src_label = f"{label}_space_src"
            space_label = f"{label}_space"
            parts.append(
                f"{source_label}aresample=48000{gain},asplit=2[{direct_src_label}][{space_src_label}]"
            )
            parts.append(f"[{direct_src_label}]{pan_expr(placement.pan)}[{stem_label}]")
            mix_labels.append(f"[{stem_label}]")
            if pan_has_signal(space_pan):
                space_chain = space_bed_filter_chain(stem, space_bed_strength)
                space_chain += temporal_space_bed_volume_filter(stem, automation_lanes)
                parts.append(
                    f"[{space_src_label}]{space_chain},"
                    f"{pan_expr(space_pan)}[{space_label}]"
                )
                mix_labels.append(f"[{space_label}]")
        else:
            parts.append(
                f"{source_label}aresample=48000{gain},{pan_expr(placement.pan)}[{stem_label}]"
            )
            mix_labels.append(f"[{stem_label}]")
    parts.append(
        "".join(mix_labels)
        + f"amix=inputs={len(mix_labels)}:normalize=0:duration=first:dropout_transition=0,"
        f"volume={fmt(master_gain)},alimiter=limit=0.98[out]"
    )
    return ";".join(parts)


def render_714(
    ffmpeg: str,
    stems: dict[str, Path],
    placements: dict[str, Placement],
    output_path: Path,
    master_gain: float,
    stem_gain_db: dict[str, float],
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    focus_decision: FocusDecision,
    priority_duck_targets: dict[str, list[PriorityDuckTarget]],
    priority_sidechain_duration: float | None,
    space_bed_mode: str,
    space_bed_stems: tuple[str, ...],
    space_bed_strength: float,
    automation_lanes: AutomationLanes | None,
    native_enabled: bool = True,
) -> None:
    input_order = [stem for stem in STEM_ORDER if stem in stems] + [
        stem for stem in stems if stem not in STEM_ORDER
    ]
    backend = os.environ.get("STEMS_UPMIXER_RENDER_714_BACKEND", "auto").strip().lower() or "auto"
    if backend not in {"auto", "native", "ffmpeg"}:
        backend = "auto"
    if not native_enabled and backend == "native":
        raise RuntimeError(
            "native 7.1.4 render requested but this output path requires the FFmpeg renderer"
        )
    if native_enabled and backend != "ffmpeg" and render_714_native(
        input_order=input_order,
        stems=stems,
        placements=placements,
        output_path=output_path,
        master_gain=master_gain,
        stem_gain_db=stem_gain_db,
        stem_eq_bands=stem_eq_bands,
        focus_decision=focus_decision,
        priority_duck_targets=priority_duck_targets,
        space_bed_mode=space_bed_mode,
        space_bed_stems=space_bed_stems,
        space_bed_strength=space_bed_strength,
        automation_lanes=automation_lanes,
        strict=backend == "native",
    ):
        return

    filter_complex = build_714_filter(
        input_order,
        placements,
        master_gain,
        stem_gain_db,
        stem_eq_bands,
        focus_decision,
        priority_duck_targets,
        priority_sidechain_duration,
        space_bed_mode,
        space_bed_stems,
        space_bed_strength,
        automation_lanes,
    )
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    for stem in input_order:
        cmd.extend(["-i", str(stems[stem])])
    temp_paths: list[Path] = []
    append_filter_complex_arg(cmd, filter_complex, temp_paths, label="7.1.4_render")
    cmd.extend(
        [
            "-map",
            "[out]",
            "-c:a",
            "pcm_s24le",
            "-channel_layout",
            "7.1.4",
            str(output_path),
        ]
    )
    timer = time.perf_counter()
    try:
        run(cmd)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
        print_elapsed("7.1.4 render", timer)


def fold_714_to_51_filter(
    input_label: str,
    output_label: str,
    lfe_fold_gain: float,
    *,
    limiter: bool = True,
) -> str:
    filter_text = (
        f"{input_label}pan=5.1(side)|"
        "FL=0.92*FL+0.24*FC+0.20*BL+0.18*TFL+0.10*TBL|"
        "FR=0.92*FR+0.24*FC+0.20*BR+0.18*TFR+0.10*TBR|"
        "FC=0.78*FC+0.05*FL+0.05*FR|"
        f"LFE={fmt(lfe_fold_gain)}*LFE|"
        "SL=0.78*SL+0.40*BL+0.20*TFL+0.34*TBL|"
        "SR=0.78*SR+0.40*BR+0.20*TFR+0.34*TBR,"
        "aresample=48000,aformat=channel_layouts=5.1(side)"
    )
    if limiter:
        filter_text += ",alimiter=limit=0.98"
    return f"{filter_text}{output_label}"


def lfe_fold_gain_for_mode(lfe_mode: str, mix_profile: str) -> float:
    if lfe_mode == "off":
        return 0.0
    if mix_profile == "front51":
        return 0.65 if lfe_mode == "light" else 0.85
    return 0.45 if lfe_mode == "light" else 0.8


def fold_714_to_stereo_filter(
    input_label: str, output_label: str, lfe_fold_gain: float, *, limiter: bool
) -> str:
    lfe_left = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    lfe_right = f"+{fmt(lfe_fold_gain)}*LFE" if lfe_fold_gain > 0.0 else ""
    filter_text = (
        f"{input_label}pan=stereo|"
        f"FL=0.92*FL+0.7071*FC+0.50*BL+0.55*SL+0.35*TFL+0.25*TBL{lfe_left}|"
        f"FR=0.92*FR+0.7071*FC+0.50*BR+0.55*SR+0.35*TFR+0.25*TBR{lfe_right},"
        "aresample=48000,aformat=channel_layouts=stereo"
    )
    if limiter:
        filter_text += ",alimiter=limit=0.98"
    return f"{filter_text}{output_label}"


__all__ = [
    "focus_duck_filter",
    "focus_target_duck_scale",
    "focus_bed_duck_mix",
    "focus_bed_duck_filter",
    "priority_duck_filter",
    "priority_sidechain_filters",
    "append_bass_drum_multiband_duck",
    "append_overlap_band_duck",
    "append_priority_masking_duck",
    "pan_has_signal",
    "space_bed_channels",
    "split_direct_space_pan",
    "should_use_space_bed",
    "space_bed_delay_pair_ms",
    "space_bed_filter_chain",
    "sampled_automation_segments",
    "automation_volume_filter",
    "temporal_space_bed_values",
    "temporal_space_bed_volume_filter",
    "temporal_mid_decongest_values",
    "append_temporal_mid_decongest",
    "append_temporal_low_tighten",
    "stereo_temporal_low_tighten_values",
    "stereo_temporal_mid_decongest_values",
    "stereo_temporal_harshness_guard_values",
    "append_stereo_temporal_mastering",
    "stereo_fold_temporal_filter",
    "build_714_filter",
    "render_714",
    "fold_714_to_51_filter",
    "lfe_fold_gain_for_mode",
    "fold_714_to_stereo_filter",
]
