"""Post-analysis mix tuning guards and corrective gain/EQ decisions."""

from __future__ import annotations

import math
from typing import Iterable

from .models import (
    BAND_NAMES,
    BASS_DRUM_GUARD_RANGES,
    CHANNELS_714,
    MULTIBAND_EQ_CANDIDATES,
    MULTIBAND_EQ_SPECS,
    MULTIBAND_RANGES,
    PAN_TERM_RE,
    POST_RENDER_EFFECTIVE_EQ_DB,
    POST_RENDER_EFFECTIVE_GAIN_DB,
    POST_RENDER_EFFECTIVE_PLACEMENT_DELTA,
    SPACE_CHANNELS,
    STEM_ORDER,
    STEREO_FOLD_WEIGHTS,
    TOP_CHANNELS,
    BassDrumCollisionBand,
    BassDrumCollisionGuardResult,
    BassFoldSupportResult,
    DrumDominanceResult,
    FocusDecision,
    FocusDuckTarget,
    MasteringReferenceResult,
    MultibandStats,
    Placement,
    PostRenderCorrectionResult,
    PostRenderCorrectionStep,
    PostRenderFoldAnalysis,
    PriorityDuckTarget,
    StemEqBand,
    StemPresenceItem,
    StemPresenceResult,
    StemStats,
    VocalLeakageItem,
    VocalLeakageResult,
    WindowStats,
)
from .placement import (
    focus_group_stems,
    focus_weight,
    ordered_focus_weights,
    scale_pan_expr,
    sorted_stem_eq_bands,
    stem_energy_shares,
    stem_eq_band_map,
)
from .stems import (
    activity_overlap_score,
    band_fractions,
    band_overlap_score,
    envelope_correlation,
    low_light_score,
    placement_collision_score,
    selected_overlap_bands,
    stats_band_energies,
    stats_band_fraction_map,
    stem_range_energies_numpy,
    wide_score,
)
from .utils import clamp, db_to_amp, fmt, linear_energy, median_float


def tune_bass_drum_collision_guard(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    placements: dict[str, Placement],
    decision: FocusDecision,
    *,
    mode: str,
    max_duck_db: float,
) -> BassDrumCollisionGuardResult:
    if mode == "off":
        return BassDrumCollisionGuardResult(False, 0.0, 0.0, 0.0, 0.0, "disabled")
    if decision.stem != "bass":
        return BassDrumCollisionGuardResult(
            False, 0.0, 0.0, 0.0, 0.0, "bass is not the focus stem"
        )
    if focus_weight(decision, "drums") >= 0.45:
        return BassDrumCollisionGuardResult(
            False, 0.0, 0.0, 0.0, 0.0, "drums are in focus group"
        )

    by_name = {item.name: item for item in stats}
    bass = by_name.get("bass")
    drums = by_name.get("drums")
    if bass is None or drums is None:
        return BassDrumCollisionGuardResult(
            False, 0.0, 0.0, 0.0, 0.0, "missing bass or drums"
        )

    bass_bands = stats_band_fraction_map(bass)
    drums_bands = stats_band_fraction_map(drums)
    low_overlap = min(bass_bands["low"], drums_bands["low"])
    activity = activity_overlap_score(windows.get("bass"), windows.get("drums"))
    collision = placement_collision_score(
        bass, drums, windows.get("bass"), windows.get("drums")
    )
    if collision < 0.32 or activity < 0.55:
        return BassDrumCollisionGuardResult(
            False, collision, low_overlap, activity, 0.0, "below collision threshold"
        )

    try:
        bass_range_energies = stem_range_energies_numpy(bass, BASS_DRUM_GUARD_RANGES)
        drums_range_energies = stem_range_energies_numpy(drums, BASS_DRUM_GUARD_RANGES)
    except Exception as exc:
        return BassDrumCollisionGuardResult(
            False,
            collision,
            low_overlap,
            activity,
            0.0,
            f"band analysis unavailable: {exc}",
        )

    bass_total = max(sum(bass_range_energies.values()), 1e-24)
    bass_fold_power = placement_stereo_fold_power(placements["bass"])
    drums_fold_power = placement_stereo_fold_power(placements["drums"])
    max_duck = max(0.0, abs(max_duck_db))
    band_results: list[BassDrumCollisionBand] = []
    for name, low_hz, high_hz in BASS_DRUM_GUARD_RANGES:
        bass_energy = bass_range_energies.get(name, 0.0)
        drums_energy = drums_range_energies.get(name, 0.0)
        bass_fraction = bass_energy / bass_total
        folded_bass = bass_energy * bass_fold_power
        folded_drums = drums_energy * drums_fold_power
        folded_total = max(folded_bass + folded_drums, 1e-24)
        drums_share = folded_drums / folded_total
        drums_masking = clamp((drums_share - 0.48) / 0.42, 0.0, 1.0)
        bass_presence = clamp(bass_fraction / 0.16, 0.0, 1.0)
        band_score = clamp(
            0.30 * collision
            + 0.20 * activity
            + 0.30 * drums_masking
            + 0.20 * bass_presence,
            0.0,
            1.0,
        )
        if band_score < 0.52 or drums_masking <= 0.0 or bass_presence < 0.20:
            continue
        sensitivity = {"sub": 0.75, "kick": 0.85, "upper_bass": 1.00}[name]
        duck_db = clamp((0.35 + 0.85 * band_score) * sensitivity, 0.0, max_duck)
        if duck_db < 0.25:
            continue
        band_results.append(
            BassDrumCollisionBand(
                name=name,
                low_hz=low_hz,
                high_hz=high_hz,
                overlap_score=band_score,
                drums_share=drums_share,
                duck_db=duck_db,
            )
        )

    if not band_results:
        return BassDrumCollisionGuardResult(
            False, collision, low_overlap, activity, 0.0, "no low-band fold masking"
        )

    duck_db = max(band.duck_db for band in band_results)
    overlap_score = max(band.overlap_score for band in band_results)
    existing = decision.bed_duck_targets.get("drums")
    if existing is None:
        updated = FocusDuckTarget(
            stem="drums",
            bands=("low",),
            overlap_score=overlap_score,
            duck_db=duck_db,
            bass_drum_bands=tuple(band_results),
        )
    else:
        bands = tuple(
            band for band in BAND_NAMES if band in set(existing.bands) | {"low"}
        )
        updated = FocusDuckTarget(
            stem="drums",
            bands=bands,
            overlap_score=max(existing.overlap_score, overlap_score),
            duck_db=max(existing.duck_db, duck_db),
            bass_drum_bands=tuple(band_results),
        )
    decision.bed_duck_targets["drums"] = updated
    return BassDrumCollisionGuardResult(
        True,
        collision,
        low_overlap,
        activity,
        updated.duck_db,
        "drums low bands duck from bass",
        tuple(band_results),
    )


def print_bass_drum_collision_guard(result: BassDrumCollisionGuardResult) -> None:
    if result.reason in {
        "disabled",
        "bass is not the focus stem",
        "drums are in focus group",
        "missing bass or drums",
    }:
        return
    print("Bass/drum collision guard:", flush=True)
    if not result.enabled:
        print(
            f"  inactive ({result.reason}) "
            f"collision={result.collision_score:.2f} "
            f"low_overlap={result.low_overlap_score:.2f} "
            f"active={result.active_overlap_score:.2f}",
            flush=True,
        )
        return
    print(
        f"  active collision={result.collision_score:.2f} "
        f"low_overlap={result.low_overlap_score:.2f} "
        f"active={result.active_overlap_score:.2f} "
        f"drums_low_duck~{result.duck_db:.1f}dB",
        flush=True,
    )
    for band in result.bands:
        print(
            f"  {band.name:<10} {band.low_hz:.0f}-{band.high_hz:.0f}Hz "
            f"drums_share={band.drums_share * 100.0:>5.1f}% "
            f"overlap={band.overlap_score:.2f} "
            f"duck~{band.duck_db:.1f}dB",
            flush=True,
        )


def priority_masking_bands(protector: StemStats, target: StemStats) -> tuple[str, ...]:
    protector_bands = stats_band_fraction_map(protector)
    target_bands = stats_band_fraction_map(target)
    overlaps = {
        band: min(protector_bands[band], target_bands[band]) for band in BAND_NAMES
    }
    selected = selected_overlap_bands(overlaps)
    if protector.name not in {"bass", "drums"} and target.name not in {"bass", "drums"}:
        max_overlap = max(overlaps.values()) if overlaps else 0.0
        tonal_bands = tuple(
            band
            for band in ("mid", "high")
            if overlaps.get(band, 0.0) >= max(0.05, max_overlap * 0.45)
        )
        if tonal_bands:
            return tonal_bands
    return selected


def priority_masking_targets(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    priority: tuple[str, ...],
    *,
    max_duck_db: float,
) -> dict[str, list[PriorityDuckTarget]]:
    if len(priority) < 2 or max_duck_db <= 0.0:
        return {}

    by_name = {item.name: item for item in stats}
    rank = {stem: index for index, stem in enumerate(priority)}
    targets: dict[str, list[PriorityDuckTarget]] = {}
    for target_stem in priority[1:]:
        target = by_name.get(target_stem)
        if target is None:
            continue
        target_window = windows.get(target_stem)
        target_active = target_window.active_ratio if target_window is not None else 0.5
        if target_active < 0.25:
            continue

        for protector_stem in reversed(priority[: rank[target_stem]]):
            protector = by_name.get(protector_stem)
            if protector is None:
                continue
            protector_window = windows.get(protector_stem)
            protector_active = (
                protector_window.active_ratio if protector_window is not None else 0.5
            )
            if protector_active < 0.30:
                continue

            bands = priority_masking_bands(protector, target)
            if not bands:
                continue
            collision = placement_collision_score(
                protector, target, protector_window, target_window
            )
            spectral = band_overlap_score(protector, target)
            activity = activity_overlap_score(protector_window, target_window)
            overlap_score = clamp(
                0.55 * collision + 0.30 * spectral + 0.15 * activity, 0.0, 1.0
            )
            if overlap_score < 0.20:
                continue

            rank_gap = max(1, rank[target_stem] - rank[protector_stem])
            rank_weight = 0.85 + 0.10 * min(rank_gap, 2)
            duck_db = clamp(
                max_duck_db * (0.22 + 0.78 * overlap_score) * rank_weight,
                0.15,
                max_duck_db,
            )
            targets.setdefault(target_stem, []).append(
                PriorityDuckTarget(
                    stem=target_stem,
                    protector=protector_stem,
                    bands=bands,
                    overlap_score=overlap_score,
                    duck_db=duck_db,
                )
            )
            break

    for target_stem in targets:
        targets[target_stem].sort(
            key=lambda item: rank.get(item.protector, len(priority))
        )
    return targets


def db_ratio(numerator: float, denominator: float) -> float:
    return 10.0 * math.log10(max(numerator, 1e-24) / max(denominator, 1e-24))


def tonal_ratios(energies: dict[str, float]) -> tuple[float, float]:
    mid = energies["mid"]
    return db_ratio(energies["low"], mid), db_ratio(energies["high"], mid)


def placement_metrics(placement: Placement) -> tuple[float, float, float]:
    total_sq = 0.0
    space_sq = 0.0
    top_sq = 0.0
    for output_channel, expr in placement.pan.items():
        for raw_coeff, _input_channel in PAN_TERM_RE.findall(expr):
            coeff = float(raw_coeff)
            coeff_sq = coeff * coeff
            total_sq += coeff_sq
            if output_channel in SPACE_CHANNELS:
                space_sq += coeff_sq
            if output_channel in TOP_CHANNELS:
                top_sq += coeff_sq
    if total_sq <= 0.0:
        return 0.0, 0.0, 0.0
    return (
        math.sqrt(total_sq / 2.0),
        math.sqrt(space_sq / total_sq),
        math.sqrt(top_sq / total_sq),
    )


def placement_stereo_fold_power(placement: Placement) -> float:
    folded = {
        "FL": [0.0, 0.0],
        "FR": [0.0, 0.0],
    }
    for output_channel, expr in placement.pan.items():
        fold_left, fold_right = STEREO_FOLD_WEIGHTS.get(output_channel, (0.0, 0.0))
        if fold_left == 0.0 and fold_right == 0.0:
            continue
        for raw_coeff, input_channel in PAN_TERM_RE.findall(expr):
            if input_channel not in folded:
                continue
            coeff = float(raw_coeff)
            folded[input_channel][0] += coeff * fold_left
            folded[input_channel][1] += coeff * fold_right

    power = 0.0
    for left_coeff, right_coeff in folded.values():
        power += left_coeff * left_coeff + right_coeff * right_coeff
    return max(power / len(folded), 1e-9)


def add_pan_terms(expr: str, additions: dict[str, float]) -> str:
    terms: dict[str, float] = {}
    for raw_coeff, input_channel in PAN_TERM_RE.findall(expr):
        terms[input_channel] = terms.get(input_channel, 0.0) + float(raw_coeff)
    for input_channel, coeff in additions.items():
        terms[input_channel] = terms.get(input_channel, 0.0) + coeff

    order: list[str] = list(CHANNELS_714)
    order.extend(sorted(channel for channel in terms if channel not in CHANNELS_714))
    parts: list[str] = []
    for input_channel in order:
        coeff = terms.get(input_channel, 0.0)
        if abs(coeff) < 1e-9:
            continue
        text = f"{fmt(coeff)}*{input_channel}"
        if parts and coeff >= 0.0:
            text = "+" + text
        parts.append(text)
    return "".join(parts) if parts else "0"


def apply_bass_fold_support_to_placement(
    placement: Placement, support_coeff: float
) -> Placement:
    pan = dict(placement.pan)
    pan["FL"] = add_pan_terms(pan.get("FL", "0"), {"FL": support_coeff})
    pan["FR"] = add_pan_terms(pan.get("FR", "0"), {"FR": support_coeff})
    pan["FC"] = add_pan_terms(
        pan.get("FC", "0"), {"FL": support_coeff * 0.32, "FR": support_coeff * 0.32}
    )
    return Placement(
        stem=placement.stem,
        role=f"{placement.role}; bass fold support={support_coeff:.3f}",
        pan=pan,
        lfe_amount=placement.lfe_amount,
        lfe_cutoff_hz=placement.lfe_cutoff_hz,
    )


def tune_bass_fold_support(
    stats: list[StemStats],
    placements: dict[str, Placement],
    windows: dict[str, WindowStats],
    *,
    mode: str,
    target_floor_db: float,
    max_coeff: float,
) -> tuple[dict[str, Placement], BassFoldSupportResult]:
    if mode == "off":
        return placements, BassFoldSupportResult(
            False, 0.0, 0.0, 0.0, target_floor_db, 0.0, 0.0, 0.0, "disabled"
        )

    by_name = {item.name: item for item in stats}
    bass = by_name.get("bass")
    if bass is None or "bass" not in placements:
        return placements, BassFoldSupportResult(
            False, 0.0, 0.0, 0.0, target_floor_db, 0.0, 0.0, 0.0, "missing bass"
        )

    source_shares = stem_energy_shares(stats)
    fold_energies = predicted_fold_full_energies(
        stats, placements, {item.name: 0.0 for item in stats}
    )
    fold_total = sum(fold_energies.values())
    source_share = source_shares.get("bass", 0.0)
    fold_share = (
        fold_energies.get("bass", 0.0) / fold_total if fold_total > 0.0 else 0.0
    )
    fold_delta = db_ratio(fold_share, source_share)
    window = windows.get("bass")
    active = window.active_ratio if window is not None else 0.5
    fold_power_before = placement_stereo_fold_power(placements["bass"])
    if source_share < 0.045 or active < 0.35:
        result = BassFoldSupportResult(
            False,
            source_share,
            fold_share,
            fold_delta,
            target_floor_db,
            0.0,
            fold_power_before,
            fold_power_before,
            "bass not active/prominent enough",
        )
        return placements, result

    deficit = max(0.0, target_floor_db - fold_delta)
    if deficit <= 0.0:
        result = BassFoldSupportResult(
            False,
            source_share,
            fold_share,
            fold_delta,
            target_floor_db,
            0.0,
            fold_power_before,
            fold_power_before,
            "bass fold share ok",
        )
        return placements, result

    support_coeff = clamp(0.030 + 0.026 * deficit, 0.0, max(0.0, abs(max_coeff)))
    if support_coeff < 0.015:
        result = BassFoldSupportResult(
            False,
            source_share,
            fold_share,
            fold_delta,
            target_floor_db,
            0.0,
            fold_power_before,
            fold_power_before,
            "support below threshold",
        )
        return placements, result

    adjusted = dict(placements)
    adjusted["bass"] = apply_bass_fold_support_to_placement(
        placements["bass"], support_coeff
    )
    fold_power_after = placement_stereo_fold_power(adjusted["bass"])
    result = BassFoldSupportResult(
        True,
        source_share,
        fold_share,
        fold_delta,
        target_floor_db,
        support_coeff,
        fold_power_before,
        fold_power_after,
        "front low-anchor support",
    )
    return adjusted, result


def print_bass_fold_support_result(result: BassFoldSupportResult) -> None:
    if result.reason in {"disabled", "missing bass"}:
        return
    print("Bass fold support:", flush=True)
    if not result.enabled:
        print(
            f"  inactive ({result.reason}) "
            f"source={result.source_share * 100.0:.1f}% "
            f"fold={result.fold_share_before * 100.0:.1f}% "
            f"delta={result.fold_delta_db:+.1f}dB "
            f"target={result.target_floor_db:+.1f}dB",
            flush=True,
        )
        return
    print(
        f"  active source={result.source_share * 100.0:.1f}% "
        f"fold={result.fold_share_before * 100.0:.1f}% "
        f"delta={result.fold_delta_db:+.1f}dB "
        f"target={result.target_floor_db:+.1f}dB "
        f"coeff={result.support_coeff:.3f} "
        f"fold_power={result.fold_power_before:.3f}->{result.fold_power_after:.3f}",
        flush=True,
    )


def tune_vocal_leakage_placements(
    stats: list[StemStats],
    placements: dict[str, Placement],
    windows: dict[str, WindowStats],
) -> tuple[dict[str, Placement], VocalLeakageResult]:
    by_name = {item.name: item for item in stats}
    vocals = by_name.get("vocals")
    vocal_window = windows.get("vocals")
    if vocals is None or vocal_window is None:
        return placements, VocalLeakageResult(items=[])

    adjusted = dict(placements)
    items: list[VocalLeakageItem] = []
    for item in stats:
        if item.name in {"vocals", "bass", "drums"} or item.name not in placements:
            continue
        window = windows.get(item.name)
        env_corr = max(0.0, envelope_correlation(window, vocal_window))
        band_overlap = band_overlap_score(item, vocals)
        center_score = 1.0 - wide_score(item)
        _low_frac, mid_frac, _high_frac = band_fractions(item)
        collision = placement_collision_score(item, vocals, window, vocal_window)
        leak_score = clamp(
            0.38 * collision
            + 0.24 * env_corr
            + 0.20 * band_overlap * mid_frac
            + 0.18 * center_score,
            0.0,
            1.0,
        )
        if leak_score < 0.30:
            continue

        space_scale = clamp(1.0 - (0.10 + 0.14 * leak_score), 0.78, 1.0)
        top_scale = clamp(1.0 - (0.14 + 0.20 * leak_score), 0.70, 1.0)
        presence_trim = -clamp(0.10 + 0.42 * ((leak_score - 0.30) / 0.70), 0.10, 0.55)
        scaled = scale_placement_space_channels(
            placements[item.name], space_scale, top_scale
        )
        adjusted[item.name] = Placement(
            stem=scaled.stem,
            role=f"{scaled.role}; vocal leakage guard={leak_score:.2f}",
            pan=scaled.pan,
            lfe_amount=scaled.lfe_amount,
            lfe_cutoff_hz=scaled.lfe_cutoff_hz,
        )
        items.append(
            VocalLeakageItem(
                stem=item.name,
                leak_score=leak_score,
                envelope_correlation=env_corr,
                band_overlap=band_overlap,
                center_score=center_score,
                space_scale=space_scale,
                top_scale=top_scale,
                presence_trim_db=presence_trim,
                reason="vocal-like mid/center content in non-vocal stem",
            )
        )
    return adjusted, VocalLeakageResult(items=items)


def apply_vocal_leakage_eq_guard(
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    result: VocalLeakageResult,
) -> dict[str, tuple[StemEqBand, ...]]:
    if not result.items:
        return stem_eq_bands
    adjusted = dict(stem_eq_bands)
    for item in result.items:
        band_map = stem_eq_band_map(adjusted.get(item.stem, ()))
        presence = band_map.get("presence", StemEqBand("presence", 2600.0, 1.10, 0.0))
        air = band_map.get("air", StemEqBand("air", 6200.0, 1.20, 0.0))
        band_map["presence"] = StemEqBand(
            "presence",
            presence.frequency_hz,
            presence.width_octaves,
            presence.gain_db + item.presence_trim_db,
        )
        band_map["air"] = StemEqBand(
            "air",
            air.frequency_hz,
            air.width_octaves,
            air.gain_db + item.presence_trim_db * 0.58,
        )
        adjusted[item.stem] = sorted_stem_eq_bands(band_map.values())
    return adjusted


def print_vocal_leakage_guard(result: VocalLeakageResult) -> None:
    if not result.items:
        return
    print("Vocal leakage guard:", flush=True)
    print("stem      score   env  band center  space    top    eq", flush=True)
    for item in sorted(
        result.items,
        key=lambda entry: (
            STEM_ORDER.index(entry.stem) if entry.stem in STEM_ORDER else 99
        ),
    ):
        print(
            f"{item.stem:<9} "
            f"{item.leak_score:>5.2f} "
            f"{item.envelope_correlation:>5.2f} "
            f"{item.band_overlap:>5.2f} "
            f"{item.center_score:>6.2f} "
            f"*{item.space_scale:<5.2f} "
            f"*{item.top_scale:<5.2f} "
            f"{item.presence_trim_db:+.2f}dB",
            flush=True,
        )


def spatial_gain_adjustment(placement: Placement) -> float:
    total_power, space_ratio, top_ratio = placement_metrics(placement)
    gain = 0.0
    gain -= 0.45 * max(0.0, space_ratio - 0.40)
    gain -= 0.35 * max(0.0, top_ratio - 0.25)
    gain -= 0.50 * max(0.0, total_power - 0.72)
    return gain


def harman_like_tonal_adjustment(stats: StemStats) -> float:
    low_frac, _mid_frac, high_frac = band_fractions(stats)
    # This is intentionally a small, music-safe tilt. It is not an exact
    # reproduction of any playback target; it just favors low-band anchors and
    # avoids over-prominent bright/wide beds after upmix placement.
    if stats.name == "vocals":
        return clamp(0.10 * low_frac - 0.25 * high_frac, -0.20, 0.10)
    if stats.name == "bass":
        return clamp(0.65 * low_frac - 0.15 * high_frac, 0.0, 0.55)
    if stats.name == "drums":
        return clamp(0.35 * low_frac - 0.15 * high_frac, -0.15, 0.35)
    if stats.name == "guitar":
        return clamp(0.15 * low_frac - 0.55 * high_frac, -0.45, 0.15)
    if stats.name == "piano":
        return clamp(0.10 * low_frac - 0.45 * high_frac, -0.35, 0.10)
    if stats.name == "other":
        return clamp(0.15 * low_frac - 0.45 * high_frac, -0.35, 0.15)
    return clamp(0.20 * low_frac - 0.40 * high_frac, -0.35, 0.20)


def compute_stem_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    *,
    mode: str,
    vocal_gain_db: float,
    limit_db: float,
) -> dict[str, float]:
    gains: dict[str, float] = {}
    for item in stats:
        auto_gain = 0.0
        if mode in {"spatial", "harman"}:
            auto_gain += spatial_gain_adjustment(placements[item.name])
        if mode == "harman":
            auto_gain += harman_like_tonal_adjustment(item)
        auto_gain = clamp(auto_gain, -abs(limit_db), abs(limit_db))
        gains[item.name] = auto_gain + (vocal_gain_db if item.name == "vocals" else 0.0)
    return gains


def apply_focus_gain(
    gains: dict[str, float], decision: FocusDecision
) -> dict[str, float]:
    if decision.stem is None or decision.focus_gain_db <= 0.0:
        return gains
    adjusted = dict(gains)
    for stem, weight in ordered_focus_weights(decision):
        if stem not in adjusted:
            continue
        gain_db = decision.focus_gain_db * weight
        if stem == decision.vocal_anchor and stem != decision.stem:
            gain_db *= 0.65
        adjusted[stem] = adjusted.get(stem, 0.0) + gain_db
    return adjusted


def predicted_fold_band_energies(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
) -> dict[str, float]:
    totals = {band: 0.0 for band in BAND_NAMES}
    for item in stats:
        power = placement_stereo_fold_power(placements[item.name])
        gain_power = db_to_amp(stem_gain_db.get(item.name, 0.0)) ** 2
        energies = stats_band_energies(item)
        for band in BAND_NAMES:
            totals[band] += energies[band] * power * gain_power
    return totals


def predicted_fold_full_energies(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
) -> dict[str, float]:
    energies: dict[str, float] = {}
    for item in stats:
        power = placement_stereo_fold_power(placements[item.name])
        gain_power = db_to_amp(stem_gain_db.get(item.name, 0.0)) ** 2
        energies[item.name] = linear_energy(item.full.mean_db) * power * gain_power
    return energies


def stem_presence_floor_db(stem: str, decision: FocusDecision) -> float:
    if stem == "vocals":
        return -1.0
    if stem in focus_group_stems(decision):
        return -1.6
    if stem == "bass":
        return -2.4
    if stem == "piano":
        return -3.0
    return -2.4


def dynamic_stem_presence_floor_db(
    stats: StemStats,
    decision: FocusDecision,
    *,
    source_share: float,
    active_ratio: float,
    sustain_score: float,
) -> float:
    stem = stats.name
    floor = stem_presence_floor_db(stem, decision)
    wide = wide_score(stats)
    if stem == "piano" and source_share >= 0.012 and active_ratio >= 0.65:
        return max(floor, -1.2)
    if stem == "guitar" and source_share >= 0.020 and active_ratio >= 0.55:
        guitar_bed_score = clamp(0.55 * wide + 0.45 * sustain_score, 0.0, 1.0)
        if guitar_bed_score >= 0.72:
            return max(floor, -1.5)
        if guitar_bed_score >= 0.55:
            return max(floor, -1.8)
        return max(floor, -2.0)
    if stem == "other" and source_share >= 0.010 and active_ratio >= 0.50:
        other_bed_score = clamp(
            0.45 * wide + 0.35 * sustain_score + 0.20 * low_light_score(stats),
            0.0,
            1.0,
        )
        if other_bed_score >= 0.72:
            return max(floor, -1.6)
        if other_bed_score >= 0.52:
            return max(floor, -1.9)
        return max(floor, -2.2)
    return floor


def tune_stem_presence_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    windows: dict[str, WindowStats],
    stem_gain_db: dict[str, float],
    decision: FocusDecision,
    *,
    mode: str,
    limit_db: float,
) -> StemPresenceResult:
    if mode == "off":
        return StemPresenceResult(gains=dict(stem_gain_db), items=[])

    source_shares = stem_energy_shares(stats)
    fold_energies = predicted_fold_full_energies(stats, placements, stem_gain_db)
    fold_total = sum(fold_energies.values())
    fold_shares = {
        name: (energy / fold_total if fold_total > 0.0 else 0.0)
        for name, energy in fold_energies.items()
    }
    adjusted = dict(stem_gain_db)
    items: list[StemPresenceItem] = []
    max_gain = max(0.0, abs(limit_db))

    for item in stats:
        stem = item.name
        source_share = source_shares.get(stem, 0.0)
        fold_share = fold_shares.get(stem, 0.0)
        window = windows.get(stem)
        active = window.active_ratio if window is not None else 0.5
        sustain = window.sustain_score if window is not None else 0.5
        if stem == "drums":
            reason = "drums are allowed to relax in fold-down"
            adjustment = 0.0
            target_floor = -99.0
        elif stem in focus_group_stems(decision):
            reason = "focus group stem is controlled by focus gain/duck"
            adjustment = 0.0
            target_floor = stem_presence_floor_db(stem, decision)
        elif source_share < 0.006 and active < 0.25:
            reason = "very low source presence"
            adjustment = 0.0
            target_floor = stem_presence_floor_db(stem, decision)
        else:
            target_floor = dynamic_stem_presence_floor_db(
                item,
                decision,
                source_share=source_share,
                active_ratio=active,
                sustain_score=sustain,
            )
            fold_delta = db_ratio(fold_share, source_share)
            deficit = max(0.0, target_floor - fold_delta)
            activity_weight = 0.55 + 0.45 * clamp(active, 0.0, 1.0)
            share_weight = clamp(source_share / 0.05, 0.35, 1.0)
            if stem == "piano" and active >= 0.65:
                share_weight = max(share_weight, 0.85)
            elif stem == "guitar" and active >= 0.55:
                share_weight = max(share_weight, 0.70 + 0.10 * wide_score(item))
            elif stem == "other" and active >= 0.50:
                share_weight = max(share_weight, 0.68 + 0.12 * wide_score(item))
            if stem == decision.vocal_anchor:
                share_weight *= 0.75
            adjustment = clamp(
                deficit * 0.32 * activity_weight * share_weight, 0.0, max_gain
            )
            reason = "below fold presence floor" if adjustment > 0.0 else "ok"

        if stem != "drums":
            adjusted[stem] = adjusted.get(stem, 0.0) + adjustment
        items.append(
            StemPresenceItem(
                stem=stem,
                source_share=source_share,
                fold_share_before=fold_share,
                fold_delta_db=db_ratio(fold_share, source_share),
                active_ratio=active,
                target_floor_db=target_floor,
                adjustment_db=adjustment,
                reason=reason,
            )
        )

    return StemPresenceResult(gains=adjusted, items=items)


def tune_drum_dominance_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    windows: dict[str, WindowStats],
    stem_gain_db: dict[str, float],
    focus_decision: FocusDecision,
    *,
    mode: str,
    limit_db: float,
) -> DrumDominanceResult:
    adjusted = dict(stem_gain_db)
    by_name = {item.name: item for item in stats}
    if mode == "off" or "drums" not in by_name:
        return DrumDominanceResult(
            adjusted, 0.0, 0.0, 0.0, 0.0, 0.0, "disabled or no drums"
        )

    source_shares = stem_energy_shares(stats)
    fold_energies = predicted_fold_full_energies(stats, placements, stem_gain_db)
    fold_total = sum(fold_energies.values())
    fold_shares = {
        name: (energy / fold_total if fold_total > 0.0 else 0.0)
        for name, energy in fold_energies.items()
    }
    source_share = source_shares.get("drums", 0.0)
    fold_share = fold_shares.get("drums", 0.0)
    fold_delta = db_ratio(fold_share, source_share)
    drum_window = windows.get("drums")
    active = drum_window.active_ratio if drum_window is not None else 0.5

    excess_share = max(0.0, fold_share - max(0.34, source_share + 0.08))
    excess_delta = max(0.0, fold_delta - 1.2)
    active_weight = 0.65 + 0.35 * clamp(active, 0.0, 1.0)
    trim = 0.0
    trim += 0.85 * excess_share
    trim += 0.065 * excess_delta
    trim *= active_weight
    if focus_weight(focus_decision, "drums") >= 0.45:
        trim *= 0.35
    trim = clamp(trim, 0.0, max(0.0, abs(limit_db)))

    if trim > 0.0:
        adjusted["drums"] = adjusted.get("drums", 0.0) - trim
        reason = (
            "focus-aware fold dominance trim"
            if focus_weight(focus_decision, "drums") >= 0.45
            else "fold dominance trim"
        )
    else:
        reason = "ok"

    return DrumDominanceResult(
        gains=adjusted,
        source_share=source_share,
        fold_share_before=fold_share,
        fold_delta_db=fold_delta,
        active_ratio=active,
        trim_db=trim,
        reason=reason,
    )


def tune_mastering_reference_gains(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
    *,
    guard_stats: StemStats | None,
    style_stats: StemStats | None,
    style_strength: float,
    guard_tolerance_db: float,
    limit_db: float,
) -> MasteringReferenceResult:
    names = [item.name for item in stats]
    guard_ratios = (
        tonal_ratios(stats_band_energies(guard_stats))
        if guard_stats is not None
        else None
    )
    style_ratios = (
        tonal_ratios(stats_band_energies(style_stats))
        if style_stats is not None
        else None
    )
    before_ratios = tonal_ratios(
        predicted_fold_band_energies(stats, placements, stem_gain_db)
    )
    strength = clamp(style_strength, 0.0, 1.0)
    tolerance = max(0.0, guard_tolerance_db)
    adjustment_limits = {
        name: min(abs(limit_db), 0.40) if name == "vocals" else abs(limit_db)
        for name in names
    }
    adjustments = {name: 0.0 for name in names}

    def candidate_gains(candidate_adjustments: dict[str, float]) -> dict[str, float]:
        return {
            name: stem_gain_db.get(name, 0.0) + candidate_adjustments.get(name, 0.0)
            for name in names
        }

    def objective(candidate_adjustments: dict[str, float]) -> float:
        ratios = tonal_ratios(
            predicted_fold_band_energies(
                stats, placements, candidate_gains(candidate_adjustments)
            )
        )
        score = 0.0
        if guard_ratios is not None:
            guard_low_excess = max(0.0, abs(ratios[0] - guard_ratios[0]) - tolerance)
            guard_high_excess = max(0.0, abs(ratios[1] - guard_ratios[1]) - tolerance)
            score += 3.0 * guard_low_excess * guard_low_excess
            score += 2.6 * guard_high_excess * guard_high_excess
        if style_ratios is not None and strength > 0.0:
            low_error = ratios[0] - style_ratios[0]
            high_error = ratios[1] - style_ratios[1]
            score += strength * (low_error * low_error + 0.85 * high_error * high_error)
        regularization = 0.0
        for name, value in candidate_adjustments.items():
            limit = max(adjustment_limits[name], 1e-6)
            weight = 0.45 if name == "vocals" else 0.18
            regularization += weight * (value / limit) ** 2
        return score + regularization

    best_score = objective(adjustments)
    for step in (0.25, 0.10, 0.05):
        improved = True
        while improved:
            improved = False
            for name in names:
                limit = adjustment_limits[name]
                best_value = adjustments[name]
                for direction in (-1.0, 1.0):
                    trial = dict(adjustments)
                    trial[name] = clamp(trial[name] + direction * step, -limit, limit)
                    if abs(trial[name] - adjustments[name]) < 1e-9:
                        continue
                    score = objective(trial)
                    if score + 1e-8 < best_score:
                        best_score = score
                        best_value = trial[name]
                if abs(best_value - adjustments[name]) > 1e-9:
                    adjustments[name] = best_value
                    improved = True

    matched = candidate_gains(adjustments)
    after_ratios = tonal_ratios(
        predicted_fold_band_energies(stats, placements, matched)
    )
    return MasteringReferenceResult(
        gains=matched,
        adjustments=adjustments,
        before_ratios=before_ratios,
        after_ratios=after_ratios,
        guard_ratios=guard_ratios,
        style_ratios=style_ratios,
        guard_tolerance_db=tolerance,
        style_strength=strength,
    )


def post_render_correction_target(
    guard_stats: StemStats | None,
    style_stats: StemStats | None,
) -> tuple[str, StemStats] | None:
    if guard_stats is not None:
        return "guard", guard_stats
    if style_stats is not None:
        return "style", style_stats
    return None


def stem_band_weights(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
    band: str,
    candidates: Iterable[str],
    bias: dict[str, float],
) -> dict[str, float]:
    candidate_list = list(candidates)
    candidate_set = set(candidate_list)
    scores: dict[str, float] = {}
    for item in stats:
        if item.name not in candidate_set or item.name not in placements:
            continue
        band_energy = stats_band_energies(item).get(band, 0.0)
        fold_power = placement_stereo_fold_power(placements[item.name])
        gain_power = db_to_amp(stem_gain_db.get(item.name, 0.0)) ** 2
        scores[item.name] = (
            band_energy * fold_power * gain_power * bias.get(item.name, 1.0)
        )

    total = sum(scores.values())
    if total <= 1e-24:
        available_names = {item.name for item in stats}
        names = [name for name in candidate_list if name in available_names]
        if not names:
            return {}
        weight = 1.0 / len(names)
        return {name: weight for name in names}
    return {name: score / total for name, score in scores.items()}


def multiband_relative_drifts(
    actual: MultibandStats | None,
    target: MultibandStats | None,
) -> tuple[float | None, dict[str, float]]:
    if actual is None or target is None:
        return None, {}
    actual_rows = {row.band: row for row in actual.rows}
    target_rows = {row.band: row for row in target.rows}
    raw_drifts = {
        band: actual_rows[band].full_db - target_rows[band].full_db
        for band in actual_rows.keys() & target_rows.keys()
    }
    if not raw_drifts:
        return None, {}
    offset = median_float(raw_drifts.values())
    return offset, {band: drift - offset for band, drift in raw_drifts.items()}


def stem_multiband_weights(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
    band: str,
    candidates: Iterable[str],
    bias: dict[str, float],
) -> dict[str, float]:
    ranges_by_name = {
        name: (name, low_hz, high_hz) for name, low_hz, high_hz in MULTIBAND_RANGES
    }
    band_range = ranges_by_name.get(band)
    if band_range is None:
        return {}

    candidate_list = list(candidates)
    candidate_set = set(candidate_list)
    scores: dict[str, float] = {}
    for item in stats:
        if item.name not in candidate_set or item.name not in placements:
            continue
        try:
            energies = stem_range_energies_numpy(item, (band_range,))
        except Exception:
            continue
        band_energy = energies.get(band, 0.0)
        fold_power = placement_stereo_fold_power(placements[item.name])
        gain_power = db_to_amp(stem_gain_db.get(item.name, 0.0)) ** 2
        scores[item.name] = (
            band_energy * fold_power * gain_power * bias.get(item.name, 1.0)
        )

    total = sum(scores.values())
    if total <= 1e-24:
        available_names = {item.name for item in stats}
        names = [name for name in candidate_list if name in available_names]
        if not names:
            return {}
        weight = 1.0 / len(names)
        return {name: weight for name in names}
    return {name: score / total for name, score in scores.items()}


def scale_placement_space_channels(
    placement: Placement, space_scale: float, top_scale: float
) -> Placement:
    pan: dict[str, str] = {}
    for channel, expr in placement.pan.items():
        scale = 1.0
        if channel in SPACE_CHANNELS:
            scale *= space_scale
        if channel in TOP_CHANNELS:
            scale *= top_scale
        pan[channel] = scale_pan_expr(expr, scale) if abs(scale - 1.0) >= 1e-6 else expr
    return Placement(
        stem=placement.stem,
        role=placement.role,
        pan=pan,
        lfe_amount=placement.lfe_amount,
        lfe_cutoff_hz=placement.lfe_cutoff_hz,
    )


def tune_post_render_correction(
    stats: list[StemStats],
    placements: dict[str, Placement],
    stem_gain_db: dict[str, float],
    base_gain_db: dict[str, float],
    actual_fold_stats: StemStats,
    stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    base_stem_eq_bands: dict[str, tuple[StemEqBand, ...]],
    *,
    target_name: str,
    target_stats: StemStats,
    actual_multiband_stats: MultibandStats | None,
    target_multiband_stats: MultibandStats | None,
    iteration: int,
    tolerance_db: float,
    gain_limit_db: float,
    placement_scale_limit: float,
    eq_limit_db: float,
) -> PostRenderCorrectionResult:
    actual_ratios = tonal_ratios(stats_band_energies(actual_fold_stats))
    target_ratios = tonal_ratios(stats_band_energies(target_stats))
    low_mid_drift = actual_ratios[0] - target_ratios[0]
    high_mid_drift = actual_ratios[1] - target_ratios[1]
    width_drift = actual_fold_stats.width_db - target_stats.width_db
    rms_drift = actual_fold_stats.full.mean_db - target_stats.full.mean_db
    tolerance = max(0.0, float(tolerance_db))
    gain_limit = max(0.0, abs(float(gain_limit_db)))
    gain_step_limit = min(0.35, gain_limit)
    placement_limit = clamp(float(placement_scale_limit), 0.0, 0.12)
    eq_limit = max(0.0, abs(float(eq_limit_db)))
    eq_step_limit = min(0.22, eq_limit)
    adjustments: dict[str, float] = {}
    placement_scales: dict[str, tuple[float, float]] = {}

    available_names = {item.name for item in stats}
    eq_maps = {
        stem: stem_eq_band_map(stem_eq_bands.get(stem, ())) for stem in available_names
    }
    base_eq_maps = {
        stem: stem_eq_band_map(base_stem_eq_bands.get(stem, ()))
        for stem in available_names
    }
    eq_adjustments: dict[str, dict[str, StemEqBand]] = {}

    def add_gain(stem: str, delta: float) -> None:
        if stem not in stem_gain_db or gain_limit <= 0.0:
            return
        delta = clamp(delta, -gain_step_limit, gain_step_limit)
        current_relative = (
            stem_gain_db.get(stem, 0.0)
            + adjustments.get(stem, 0.0)
            - base_gain_db.get(stem, stem_gain_db.get(stem, 0.0))
        )
        clipped = clamp(
            delta, -gain_limit - current_relative, gain_limit - current_relative
        )
        if abs(clipped) >= 0.005:
            adjustments[stem] = adjustments.get(stem, 0.0) + clipped

    def add_space_scale(
        stem: str, *, space_scale: float = 1.0, top_scale: float = 1.0
    ) -> None:
        if stem not in placements:
            return
        current_space, current_top = placement_scales.get(stem, (1.0, 1.0))
        merged = (
            clamp(current_space * space_scale, 1.0 - placement_limit, 1.0),
            clamp(current_top * top_scale, 1.0 - placement_limit, 1.0),
        )
        if abs(merged[0] - 1.0) >= 0.002 or abs(merged[1] - 1.0) >= 0.002:
            placement_scales[stem] = merged

    def add_eq(
        stem: str,
        band_name: str,
        frequency_hz: float,
        width_octaves: float,
        delta_db: float,
    ) -> None:
        if stem not in available_names or eq_limit <= 0.0:
            return
        delta_db = clamp(delta_db, -eq_step_limit, eq_step_limit)
        current_band = eq_maps.setdefault(stem, {}).get(band_name)
        base_band = base_eq_maps.get(stem, {}).get(band_name)
        current_gain = current_band.gain_db if current_band is not None else 0.0
        base_gain = base_band.gain_db if base_band is not None else 0.0
        clipped_gain = clamp(
            current_gain + delta_db, base_gain - eq_limit, base_gain + eq_limit
        )
        actual_delta = clipped_gain - current_gain
        if abs(actual_delta) < 0.005:
            return
        eq_maps[stem][band_name] = StemEqBand(
            band_name, frequency_hz, width_octaves, clipped_gain
        )
        previous = eq_adjustments.setdefault(stem, {}).get(band_name)
        previous_delta = previous.gain_db if previous is not None else 0.0
        eq_adjustments[stem][band_name] = StemEqBand(
            band_name,
            frequency_hz,
            width_octaves,
            previous_delta + actual_delta,
        )

    low_excess = max(0.0, abs(low_mid_drift) - tolerance)
    if low_excess > 0.0:
        low_step = clamp(low_excess * 0.24, 0.0, gain_step_limit)
        if low_mid_drift < 0.0:
            weights = stem_band_weights(
                stats,
                placements,
                stem_gain_db,
                "low",
                ("bass", "other", "drums"),
                {"bass": 2.1, "other": 0.65, "drums": 0.35},
            )
            for stem, weight in weights.items():
                add_gain(stem, low_step * weight)
        else:
            weights = stem_band_weights(
                stats,
                placements,
                stem_gain_db,
                "low",
                ("bass", "drums", "other"),
                {"bass": 1.5, "drums": 0.75, "other": 0.35},
            )
            for stem, weight in weights.items():
                add_gain(stem, -low_step * weight)

    high_excess = max(0.0, abs(high_mid_drift) - tolerance)
    if high_excess > 0.0:
        if high_mid_drift > 0.0:
            high_step = clamp(high_excess * 0.20, 0.0, gain_step_limit)
            weights = stem_band_weights(
                stats,
                placements,
                stem_gain_db,
                "high",
                ("other", "guitar", "piano", "drums"),
                {"other": 1.15, "guitar": 1.0, "piano": 0.85, "drums": 0.65},
            )
            for stem, weight in weights.items():
                add_gain(stem, -high_step * weight)
            top_reduction = clamp(high_excess * 0.012, 0.0, placement_limit)
            for stem, weight in weights.items():
                if stem in available_names:
                    add_space_scale(
                        stem, top_scale=1.0 - top_reduction * (0.40 + 0.60 * weight)
                    )
        else:
            high_step = clamp(high_excess * 0.16, 0.0, gain_step_limit)
            weights = stem_band_weights(
                stats,
                placements,
                stem_gain_db,
                "high",
                ("other", "guitar", "piano"),
                {"other": 1.1, "guitar": 1.0, "piano": 0.85},
            )
            for stem, weight in weights.items():
                add_gain(stem, high_step * weight)

    width_tolerance = max(1.5, tolerance * 2.0)
    if width_drift > width_tolerance:
        width_excess = width_drift - width_tolerance
        space_reduction = clamp(width_excess * 0.018, 0.0, placement_limit)
        weights = stem_band_weights(
            stats,
            placements,
            stem_gain_db,
            "high",
            ("other", "guitar", "piano", "drums"),
            {"other": 1.2, "guitar": 1.0, "piano": 0.85, "drums": 0.50},
        )
        for stem, weight in weights.items():
            add_space_scale(
                stem,
                space_scale=1.0 - space_reduction * (0.35 + 0.65 * weight),
                top_scale=1.0 - space_reduction * (0.20 + 0.40 * weight),
            )

    multiband_offset, multiband_drifts = multiband_relative_drifts(
        actual_multiband_stats, target_multiband_stats
    )
    if multiband_drifts and eq_limit > 0.0:
        band_tolerance = max(0.32, tolerance * 0.58)
        band_strength = {
            "sub": 0.10,
            "bass": 0.12,
            "lowmid": 0.15,
            "body": 0.14,
            "mid": 0.14,
            "presence": 0.16,
            "air": 0.14,
            "top": 0.10,
        }
        for band_name, _low_hz, _high_hz in MULTIBAND_RANGES:
            drift = multiband_drifts.get(band_name, 0.0)
            excess = max(0.0, abs(drift) - band_tolerance)
            if excess <= 0.0:
                continue
            frequency_hz, width_octaves = MULTIBAND_EQ_SPECS[band_name]
            candidates, bias = MULTIBAND_EQ_CANDIDATES[band_name]
            weights = stem_multiband_weights(
                stats, placements, stem_gain_db, band_name, candidates, bias
            )
            if not weights:
                continue
            band_delta = -math.copysign(
                clamp(excess * band_strength[band_name], 0.0, eq_step_limit),
                drift,
            )
            for stem, weight in weights.items():
                if weight < 0.04:
                    continue
                stem_delta = band_delta * (0.35 + 0.65 * weight)
                if stem == "vocals" and band_name in {"sub", "bass", "air", "top"}:
                    stem_delta *= 0.45
                add_eq(stem, band_name, frequency_hz, width_octaves, stem_delta)

    adjusted_gains = dict(stem_gain_db)
    for stem, delta in adjustments.items():
        adjusted_gains[stem] = adjusted_gains.get(stem, 0.0) + delta

    adjusted_placements = dict(placements)
    for stem, (space_scale, top_scale) in placement_scales.items():
        adjusted_placements[stem] = scale_placement_space_channels(
            placements[stem], space_scale, top_scale
        )

    adjusted_eq_bands = dict(stem_eq_bands)
    for stem, bands in eq_maps.items():
        sorted_bands = sorted_stem_eq_bands(bands.values())
        if sorted_bands:
            adjusted_eq_bands[stem] = sorted_bands
        else:
            adjusted_eq_bands.pop(stem, None)

    return PostRenderCorrectionResult(
        gains=adjusted_gains,
        placements=adjusted_placements,
        stem_eq_bands=adjusted_eq_bands,
        step=PostRenderCorrectionStep(
            iteration=iteration,
            target_name=target_name,
            low_mid_drift_db=low_mid_drift,
            high_mid_drift_db=high_mid_drift,
            width_drift_db=width_drift,
            rms_drift_db=rms_drift,
            gain_adjustments=adjustments,
            placement_scales=placement_scales,
            eq_adjustments={
                stem: sorted_stem_eq_bands(bands.values())
                for stem, bands in eq_adjustments.items()
            },
            multiband_offset_db=multiband_offset,
            multiband_drifts_db=multiband_drifts,
        ),
    )


def post_render_correction_changed(step: PostRenderCorrectionStep) -> bool:
    return (
        any(
            abs(value) >= POST_RENDER_EFFECTIVE_GAIN_DB
            for value in step.gain_adjustments.values()
        )
        or any(
            abs(space_scale - 1.0) >= POST_RENDER_EFFECTIVE_PLACEMENT_DELTA
            or abs(top_scale - 1.0) >= POST_RENDER_EFFECTIVE_PLACEMENT_DELTA
            for space_scale, top_scale in step.placement_scales.values()
        )
        or any(
            abs(band.gain_db) >= POST_RENDER_EFFECTIVE_EQ_DB
            for bands in step.eq_adjustments.values()
            for band in bands
        )
    )


def print_post_render_correction_step(step: PostRenderCorrectionStep) -> None:
    print(
        f"Post-render correction {step.iteration}: "
        f"target={step.target_name} "
        f"low-mid={step.low_mid_drift_db:+.2f}dB "
        f"high-mid={step.high_mid_drift_db:+.2f}dB "
        f"width={step.width_drift_db:+.2f}dB "
        f"rms={step.rms_drift_db:+.2f}dB",
        flush=True,
    )
    if not post_render_correction_changed(step):
        print(
            "  no render-worthy correction needed within effective threshold",
            flush=True,
        )
        return
    if step.gain_adjustments:
        gains = " ".join(
            f"{stem}={delta:+.2f}dB"
            for stem, delta in sorted(step.gain_adjustments.items())
        )
        print(f"  stem gains: {gains}", flush=True)
    if step.placement_scales:
        scales = " ".join(
            f"{stem}=space*{space_scale:.3f}/top*{top_scale:.3f}"
            for stem, (space_scale, top_scale) in sorted(step.placement_scales.items())
        )
        print(f"  placement: {scales}", flush=True)
    if step.eq_adjustments:
        for stem in sorted(step.eq_adjustments):
            bands = " ".join(
                f"{band.name}={band.gain_db:+.2f}dB"
                for band in step.eq_adjustments[stem]
            )
            print(f"  stem EQ: {stem} {bands}", flush=True)
    if step.multiband_drifts_db:
        largest = sorted(
            step.multiband_drifts_db.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:3]
        drifts = " ".join(f"{band}={drift:+.2f}dB" for band, drift in largest)
        offset = (
            step.multiband_offset_db if step.multiband_offset_db is not None else 0.0
        )
        print(f"  8-band relative drift: offset={offset:+.2f}dB {drifts}", flush=True)


def print_stem_gains(gains: dict[str, float]) -> None:
    print("Applied stem gains:", flush=True)
    for stem in sorted(gains):
        print(f"{stem:<9} {gains[stem]:>+5.2f} dB", flush=True)


def print_stem_presence_result(result: StemPresenceResult) -> None:
    if not result.items:
        return
    print("Stem presence guard:", flush=True)
    print("stem      source   fold  delta  active target    adj  status", flush=True)
    for item in sorted(
        result.items,
        key=lambda entry: (
            STEM_ORDER.index(entry.stem) if entry.stem in STEM_ORDER else 99
        ),
    ):
        target = (
            "--" if item.target_floor_db < -50.0 else f"{item.target_floor_db:+.1f}"
        )
        print(
            f"{item.stem:<9} "
            f"{item.source_share * 100.0:>5.1f}% "
            f"{item.fold_share_before * 100.0:>5.1f}% "
            f"{item.fold_delta_db:>+6.1f} "
            f"{item.active_ratio:>6.2f} "
            f"{target:>6} "
            f"{item.adjustment_db:>+5.2f}  "
            f"{item.reason}",
            flush=True,
        )


def print_drum_dominance_result(result: DrumDominanceResult) -> None:
    if not result.reason:
        return
    print("Drum dominance guard:", flush=True)
    print("stem      source   fold  delta  active   trim  status", flush=True)
    print(
        f"{'drums':<9} "
        f"{result.source_share * 100.0:>5.1f}% "
        f"{result.fold_share_before * 100.0:>5.1f}% "
        f"{result.fold_delta_db:>+6.1f} "
        f"{result.active_ratio:>6.2f} "
        f"{-result.trim_db:>+6.2f}  "
        f"{result.reason}",
        flush=True,
    )


def print_mastering_reference_result(
    result: MasteringReferenceResult,
) -> None:
    print("Mastering reference:", flush=True)
    print("tonal        low-mid  high-mid", flush=True)
    if result.guard_ratios is not None:
        print(
            f"guard        {result.guard_ratios[0]:>+7.2f}  {result.guard_ratios[1]:>+8.2f} dB "
            f"(tol +/-{result.guard_tolerance_db:.2f})",
            flush=True,
        )
    if result.style_ratios is not None:
        print(
            f"style        {result.style_ratios[0]:>+7.2f}  {result.style_ratios[1]:>+8.2f} dB "
            f"(strength {result.style_strength:.2f})",
            flush=True,
        )
    print(
        f"before       {result.before_ratios[0]:>+7.2f}  {result.before_ratios[1]:>+8.2f} dB",
        flush=True,
    )
    print(
        f"after        {result.after_ratios[0]:>+7.2f}  {result.after_ratios[1]:>+8.2f} dB",
        flush=True,
    )
    print("Mastering reference stem adjustments:", flush=True)
    for stem in sorted(result.adjustments):
        print(f"{stem:<9} {result.adjustments[stem]:>+5.2f} dB", flush=True)


def print_post_render_fold_analysis(result: PostRenderFoldAnalysis) -> None:
    def ratios(stats: StemStats) -> tuple[float, float]:
        return tonal_ratios(stats_band_energies(stats))

    def row(label: str, stats: StemStats) -> None:
        low_mid, high_mid = ratios(stats)
        print(
            f"{label:<12} {low_mid:>+7.2f}  {high_mid:>+8.2f} "
            f"{stats.full.mean_db:>7.2f} {stats.full.max_db:>7.2f} {stats.width_db:>+8.2f}",
            flush=True,
        )

    actual_ratios = ratios(result.actual)
    print("Post-render fold-down analysis:", flush=True)
    print("source        low-mid  high-mid     rms    peak side-mid", flush=True)
    if result.predicted_ratios is not None:
        print(
            f"{'predicted':<12} {result.predicted_ratios[0]:>+7.2f}  {result.predicted_ratios[1]:>+8.2f} "
            "     --      --       --",
            flush=True,
        )
    row("actual", result.actual)
    if result.guard is not None:
        row("guard", result.guard)
        guard_ratios = ratios(result.guard)
        low_drift = actual_ratios[0] - guard_ratios[0]
        high_drift = actual_ratios[1] - guard_ratios[1]
        rms_drift = result.actual.full.mean_db - result.guard.full.mean_db
        peak_drift = result.actual.full.max_db - result.guard.full.max_db
        width_drift = result.actual.width_db - result.guard.width_db
        low_excess = max(0.0, abs(low_drift) - result.guard_tolerance_db)
        high_excess = max(0.0, abs(high_drift) - result.guard_tolerance_db)
        print(
            f"{'drift':<12} {low_drift:>+7.2f}  {high_drift:>+8.2f} "
            f"{rms_drift:>+7.2f} {peak_drift:>+7.2f} {width_drift:>+8.2f}",
            flush=True,
        )
        if low_excess > 0.0 or high_excess > 0.0:
            print(
                f"Guard status: WARN tonal excess low={low_excess:.2f} dB high={high_excess:.2f} dB "
                f"beyond +/-{result.guard_tolerance_db:.2f} dB.",
                flush=True,
            )
        else:
            print(
                f"Guard status: OK tonal drift within +/-{result.guard_tolerance_db:.2f} dB.",
                flush=True,
            )
    if result.style is not None:
        row("style", result.style)


__all__ = [
    "tune_bass_drum_collision_guard",
    "print_bass_drum_collision_guard",
    "priority_masking_bands",
    "priority_masking_targets",
    "db_ratio",
    "tonal_ratios",
    "placement_metrics",
    "placement_stereo_fold_power",
    "add_pan_terms",
    "apply_bass_fold_support_to_placement",
    "tune_bass_fold_support",
    "print_bass_fold_support_result",
    "tune_vocal_leakage_placements",
    "apply_vocal_leakage_eq_guard",
    "print_vocal_leakage_guard",
    "spatial_gain_adjustment",
    "harman_like_tonal_adjustment",
    "compute_stem_gains",
    "apply_focus_gain",
    "predicted_fold_band_energies",
    "predicted_fold_full_energies",
    "stem_presence_floor_db",
    "dynamic_stem_presence_floor_db",
    "tune_stem_presence_gains",
    "tune_drum_dominance_gains",
    "tune_mastering_reference_gains",
    "post_render_correction_target",
    "stem_band_weights",
    "multiband_relative_drifts",
    "stem_multiband_weights",
    "scale_placement_space_channels",
    "tune_post_render_correction",
    "post_render_correction_changed",
    "print_post_render_correction_step",
    "print_stem_gains",
    "print_stem_presence_result",
    "print_drum_dominance_result",
    "print_mastering_reference_result",
    "print_post_render_fold_analysis",
]
