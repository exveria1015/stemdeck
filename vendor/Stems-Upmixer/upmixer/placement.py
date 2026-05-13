"""Spatial placement, focus selection, ducking decisions, and stem EQ."""

from __future__ import annotations

import re
from typing import Iterable

from .models import (
    BAND_NAMES,
    FRONT_CHANNELS,
    MULTIBAND_RANGES,
    PAN_TERM_RE,
    SPACE_CHANNELS,
    STEM_ORDER,
    TOP_CHANNELS,
    FocusDecision,
    FocusDuckTarget,
    Placement,
    StemEqBand,
    StemStats,
    TemporalDuckGuardItem,
    TemporalDuckGuardResult,
    WindowStats,
)
from .stems import (
    bright_score,
    low_light_score,
    placement_collision_score,
    selected_overlap_bands,
    stats_band_fraction_map,
    vocal_leak_risk,
    wide_score,
)
from .utils import clamp, db_to_amp, fmt, linear_energy, safe_label


def front51_lfe_amounts(lfe_mode: str) -> tuple[float, float]:
    if lfe_mode == "off":
        return 0.0, 0.0
    if lfe_mode == "normal":
        return 0.16, 0.07
    return 0.08, 0.035


def decide_placement_front51(
    stats: StemStats,
    lfe_mode: str,
    window_stats: dict[str, WindowStats],
    all_stats: dict[str, StemStats],
) -> Placement:
    name = stats.name
    wide = wide_score(stats)
    bright = bright_score(stats)
    windows = window_stats.get(name)
    sustain = windows.sustain_score if windows else 0.5
    activity = windows.active_ratio if windows else 0.5
    bass_lfe, drum_lfe = front51_lfe_amounts(lfe_mode)

    if name == "vocals":
        center = 0.20 if stats.width_db <= -10.0 else 0.17
        return Placement(
            stem=name,
            role="front51 vocal anchor",
            pan={
                "FL": "0.50*FL",
                "FR": "0.50*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
            },
        )

    if name == "drums":
        bass = all_stats.get("bass")
        bass_windows = window_stats.get("bass")
        bass_collision = placement_collision_score(stats, bass, windows, bass_windows)
        side = 0.055 + 0.035 * wide
        side += 0.020 * bass_collision
        center = clamp(0.060 - 0.025 * bass_collision, 0.035, 0.060)
        top = 0.025 + 0.025 * bright
        return Placement(
            stem=name,
            role=f"front51 front rhythm, light room col_bass={bass_collision:.2f}",
            pan={
                "FL": "0.70*FL",
                "FR": "0.70*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "BL": "0.025*FL",
                "BR": "0.025*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
            lfe_amount=drum_lfe * (1.0 - 0.25 * bass_collision),
            lfe_cutoff_hz=95 if lfe_mode == "light" else 115,
        )

    if name == "bass":
        return Placement(
            stem=name,
            role="front51 low anchor, additive LFE",
            pan={
                "FL": "0.18*FL",
                "FR": "0.18*FR",
                "FC": "0.25*FL+0.25*FR",
            },
            lfe_amount=bass_lfe,
            lfe_cutoff_hz=85 if lfe_mode == "light" else 100,
        )

    if name == "piano":
        guitar = all_stats.get("guitar")
        other = all_stats.get("other")
        guitar_collision = placement_collision_score(
            stats, guitar, windows, window_stats.get("guitar")
        )
        other_collision = placement_collision_score(
            stats, other, windows, window_stats.get("other")
        )
        collision = max(guitar_collision, other_collision)
        side = clamp(0.035 + 0.035 * wide - 0.020 * collision, 0.025, 0.070)
        top = 0.025 + 0.035 * bright
        return Placement(
            stem=name,
            role=f"front51 front-wide piano col={collision:.2f}",
            pan={
                "FL": "0.48*FL",
                "FR": "0.48*FR",
                "FC": "0.015*FL+0.015*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
        )

    if name == "guitar":
        other = all_stats.get("other")
        other_windows = window_stats.get("other")
        other_collision = placement_collision_score(
            stats, other, windows, other_windows
        )
        piano = all_stats.get("piano")
        piano_collision = placement_collision_score(
            stats, piano, windows, window_stats.get("piano")
        )
        side = clamp(
            0.14 + 0.12 * wide - 0.045 * other_collision + 0.020 * piano_collision,
            0.12,
            0.26,
        )
        rear = clamp(0.025 + 0.055 * wide - 0.050 * other_collision, 0.015, 0.08)
        top = 0.015 + 0.025 * bright
        return Placement(
            stem=name,
            role=f"front51 side-front guitar col_other={other_collision:.2f}",
            pan={
                "FL": "0.52*FL",
                "FR": "0.52*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
            },
        )

    if name == "other":
        guitar = all_stats.get("guitar")
        vocals = all_stats.get("vocals")
        piano = all_stats.get("piano")
        guitar_collision = placement_collision_score(
            stats, guitar, windows, window_stats.get("guitar")
        )
        piano_collision = placement_collision_score(
            stats, piano, windows, window_stats.get("piano")
        )
        vocal_risk = vocal_leak_risk(stats, vocals, windows, window_stats.get("vocals"))
        synth_score = clamp(
            0.34 * wide
            + 0.22 * activity
            + 0.18 * sustain
            + 0.16 * bright
            + 0.10 * low_light_score(stats),
            0.0,
            1.0,
        )
        rear_bias = clamp(
            synth_score
            + 0.30 * guitar_collision
            + 0.12 * piano_collision
            - 0.25 * vocal_risk,
            0.0,
            1.0,
        )
        front = clamp(0.26 - 0.08 * rear_bias, 0.16, 0.26)
        side = clamp(0.12 + 0.20 * rear_bias, 0.12, 0.34)
        rear = clamp(0.055 + 0.18 * rear_bias, 0.055, 0.24)
        top_rear = clamp(0.025 + 0.055 * rear_bias, 0.025, 0.08)
        return Placement(
            stem=name,
            role=f"front51 variable synth bed rear={rear_bias:.2f} col_g={guitar_collision:.2f} vocal_risk={vocal_risk:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    front = 0.46
    side = 0.08 + 0.10 * wide
    top = 0.015 + 0.035 * bright
    return Placement(
        stem=name,
        role="front51 generic front bed",
        pan={
            "FL": f"{fmt(front)}*FL",
            "FR": f"{fmt(front)}*FR",
            "SL": f"{fmt(side)}*FL",
            "SR": f"{fmt(side)}*FR",
            "TFL": f"{fmt(top)}*FL",
            "TFR": f"{fmt(top)}*FR",
        },
    )


def decide_placement(
    stats: StemStats,
    lfe_mode: str,
    mix_profile: str,
    window_stats: dict[str, WindowStats],
    all_stats: dict[str, StemStats],
) -> Placement:
    if mix_profile == "front51":
        return decide_placement_front51(stats, lfe_mode, window_stats, all_stats)

    name = stats.name
    wide = stats.width_db > -7.0
    very_wide = stats.width_db > -3.0
    bright = stats.high_vs_mid_db > -8.0

    if lfe_mode == "off":
        bass_lfe = 0.0
        drum_lfe = 0.0
    elif lfe_mode == "normal":
        bass_lfe = 0.18
        drum_lfe = 0.08
    else:
        bass_lfe = 0.045
        drum_lfe = 0.025

    if name == "vocals":
        center = 0.10 if stats.width_db > -10 else 0.13
        return Placement(
            stem=name,
            role="front vocal anchor",
            pan={
                "FL": "0.46*FL",
                "FR": "0.46*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "SL": "0.03*FL",
                "SR": "0.03*FR",
            },
        )

    if name == "drums":
        windows = window_stats.get(name)
        bass = all_stats.get("bass")
        bass_collision = placement_collision_score(
            stats, bass, windows, window_stats.get("bass")
        )
        side = 0.30 if wide else 0.26
        side = clamp(side + 0.025 * bass_collision, 0.24, 0.34)
        center = clamp(0.03 - 0.012 * bass_collision, 0.015, 0.03)
        top = 0.10 if bright else 0.08
        return Placement(
            stem=name,
            role=f"front rhythm with side/top energy col_bass={bass_collision:.2f}",
            pan={
                "FL": "0.62*FL",
                "FR": "0.62*FR",
                "FC": f"{fmt(center)}*FL+{fmt(center)}*FR",
                "BL": "0.08*FL",
                "BR": "0.08*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
                "TBL": "0.04*FL",
                "TBR": "0.04*FR",
            },
            lfe_amount=drum_lfe * (1.0 - 0.20 * bass_collision),
            lfe_cutoff_hz=95 if lfe_mode == "light" else 115,
        )

    if name == "bass":
        return Placement(
            stem=name,
            role="front low anchor, LFE as additive support",
            pan={
                "FL": "0.13*FL",
                "FR": "0.13*FR",
                "FC": "0.20*FL+0.20*FR",
            },
            lfe_amount=bass_lfe,
            lfe_cutoff_hz=80 if lfe_mode == "light" else 95,
        )

    if name == "guitar":
        windows = window_stats.get(name)
        other_collision = placement_collision_score(
            stats, all_stats.get("other"), windows, window_stats.get("other")
        )
        piano_collision = placement_collision_score(
            stats, all_stats.get("piano"), windows, window_stats.get("piano")
        )
        front = clamp(0.46 + 0.035 * other_collision, 0.46, 0.50)
        side = 0.38 if very_wide else (0.34 if wide else 0.26)
        side = clamp(
            side - 0.045 * other_collision + 0.018 * piano_collision, 0.22, 0.40
        )
        rear = 0.12 if wide else 0.08
        rear = clamp(rear - 0.055 * other_collision, 0.035, 0.12)
        top = 0.07 if bright else 0.04
        top = clamp(top - 0.020 * other_collision, 0.025, 0.07)
        top_rear = clamp(0.03 - 0.018 * other_collision, 0.012, 0.03)
        return Placement(
            stem=name,
            role=f"wide side instrument col_other={other_collision:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top)}*FL",
                "TFR": f"{fmt(top)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    if name == "piano":
        windows = window_stats.get(name)
        guitar_collision = placement_collision_score(
            stats, all_stats.get("guitar"), windows, window_stats.get("guitar")
        )
        other_collision = placement_collision_score(
            stats, all_stats.get("other"), windows, window_stats.get("other")
        )
        collision = max(guitar_collision, other_collision)
        front = clamp(0.34 + 0.035 * collision, 0.34, 0.38)
        rear = clamp(0.06 - 0.035 * collision, 0.025, 0.06)
        side = clamp(0.12 - 0.050 * collision, 0.055, 0.12)
        height = 0.18 if bright else 0.14
        height = clamp(height - 0.045 * collision, 0.08, 0.18)
        top_rear = clamp(0.09 - 0.045 * collision, 0.035, 0.09)
        return Placement(
            stem=name,
            role=f"height/front-wide accent col={collision:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "FC": "0.01*FL+0.01*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(height)}*FL",
                "TFR": f"{fmt(height)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    if name == "other":
        windows = window_stats.get(name)
        guitar_collision = placement_collision_score(
            stats, all_stats.get("guitar"), windows, window_stats.get("guitar")
        )
        piano_collision = placement_collision_score(
            stats, all_stats.get("piano"), windows, window_stats.get("piano")
        )
        vocal_risk = vocal_leak_risk(
            stats, all_stats.get("vocals"), windows, window_stats.get("vocals")
        )
        collision = max(guitar_collision, piano_collision)
        rear = 0.26 if wide else 0.22
        rear = clamp(rear + 0.070 * collision - 0.055 * vocal_risk, 0.16, 0.34)
        side = clamp(0.32 + 0.055 * collision - 0.040 * vocal_risk, 0.24, 0.40)
        front = clamp(0.16 - 0.025 * collision + 0.045 * vocal_risk, 0.12, 0.22)
        top_front = clamp(0.07 - 0.025 * vocal_risk, 0.035, 0.07)
        top_rear = clamp(0.20 + 0.055 * collision - 0.040 * vocal_risk, 0.12, 0.26)
        return Placement(
            stem=name,
            role=f"surround bed col={collision:.2f} vocal_risk={vocal_risk:.2f}",
            pan={
                "FL": f"{fmt(front)}*FL",
                "FR": f"{fmt(front)}*FR",
                "BL": f"{fmt(rear)}*FL",
                "BR": f"{fmt(rear)}*FR",
                "SL": f"{fmt(side)}*FL",
                "SR": f"{fmt(side)}*FR",
                "TFL": f"{fmt(top_front)}*FL",
                "TFR": f"{fmt(top_front)}*FR",
                "TBL": f"{fmt(top_rear)}*FL",
                "TBR": f"{fmt(top_rear)}*FR",
            },
        )

    front = 0.42
    side = 0.24 if wide else 0.12
    top = 0.08 if bright else 0.02
    return Placement(
        stem=name,
        role="generic musical bed",
        pan={
            "FL": f"{fmt(front)}*FL",
            "FR": f"{fmt(front)}*FR",
            "SL": f"{fmt(side)}*FL",
            "SR": f"{fmt(side)}*FR",
            "TFL": f"{fmt(top)}*FL",
            "TFR": f"{fmt(top)}*FR",
        },
    )


def gain_filter(gain_db: float) -> str:
    gain = db_to_amp(gain_db)
    if abs(gain - 1.0) < 1e-6:
        return ""
    return f",volume={fmt(gain)}"


def stem_eq_filter(bands: tuple[StemEqBand, ...]) -> str:
    filters = []
    for band in bands:
        if abs(band.gain_db) < 0.05:
            continue
        filters.append(
            "equalizer="
            f"f={fmt(band.frequency_hz)}:"
            "t=o:"
            f"w={fmt(band.width_octaves)}:"
            f"g={fmt(band.gain_db)}"
        )
    return "," + ",".join(filters) if filters else ""


def stem_eq_band_map(bands: tuple[StemEqBand, ...]) -> dict[str, StemEqBand]:
    return {band.name: band for band in bands}


def sorted_stem_eq_bands(bands: Iterable[StemEqBand]) -> tuple[StemEqBand, ...]:
    band_order = {
        name: index for index, (name, _low_hz, _high_hz) in enumerate(MULTIBAND_RANGES)
    }
    band_order.update(
        {
            "mid": band_order.get("mid", 4),
            "presence": band_order.get("presence", 5),
            "air": band_order.get("air", 6),
        }
    )
    return tuple(
        sorted(
            (band for band in bands if abs(band.gain_db) >= 0.01),
            key=lambda band: (band_order.get(band.name, 99), band.frequency_hz),
        )
    )


def compute_stem_aware_eq(
    stats: list[StemStats],
    *,
    mode: str,
    strength: float,
    limit_db: float,
) -> dict[str, tuple[StemEqBand, ...]]:
    if mode == "off":
        return {}
    strength = clamp(float(strength), 0.0, 1.0)
    limit = max(0.0, abs(float(limit_db)))
    if strength <= 0.0 or limit <= 0.0:
        return {}

    eq: dict[str, tuple[StemEqBand, ...]] = {}
    for item in stats:
        if item.name in {"vocals", "bass"}:
            continue
        wide = wide_score(item)
        bright = bright_score(item)
        space_weight = clamp((wide - 0.25) / 0.75, 0.0, 1.0)
        bands: list[StemEqBand] = []
        if item.name == "drums":
            drum_weight = clamp(0.20 + 0.45 * bright, 0.0, 0.55)
            bands.extend(
                [
                    StemEqBand(
                        "presence",
                        2600.0,
                        1.05,
                        clamp(0.45 * strength * drum_weight, 0.0, limit),
                    ),
                    StemEqBand(
                        "air",
                        6200.0,
                        1.15,
                        clamp(0.55 * strength * drum_weight, 0.0, limit),
                    ),
                ]
            )
        elif space_weight > 0.05:
            bands.extend(
                [
                    StemEqBand(
                        "mid",
                        1150.0,
                        1.00,
                        clamp(0.25 * strength * space_weight, 0.0, limit),
                    ),
                    StemEqBand(
                        "presence",
                        2600.0,
                        1.15,
                        clamp(1.05 * strength * space_weight, 0.0, limit),
                    ),
                    StemEqBand(
                        "air",
                        6200.0,
                        1.20,
                        clamp(
                            (0.70 + 0.20 * bright) * strength * space_weight, 0.0, limit
                        ),
                    ),
                ]
            )
        filtered = tuple(band for band in bands if band.gain_db >= 0.05)
        if filtered:
            eq[item.name] = filtered
    return eq


def print_stem_aware_eq(eq: dict[str, tuple[StemEqBand, ...]]) -> None:
    if not eq:
        return
    print("Stem-aware EQ:", flush=True)
    for stem in sorted(eq):
        text = " ".join(f"{band.name}={band.gain_db:+.2f}dB" for band in eq[stem])
        print(f"{stem:<9} {text}", flush=True)


def gain_suffix(stem: str, gain_db: float) -> str:
    if abs(gain_db) < 0.05:
        return ""
    direction = "plus" if gain_db > 0 else "minus"
    value = str(round(abs(gain_db), 2)).replace(".", "p").rstrip("0").rstrip("p")
    return f"_{stem}_{direction}{value}db"


def stem_energy_shares(stats: list[StemStats]) -> dict[str, float]:
    energies = {item.name: linear_energy(item.full.mean_db) for item in stats}
    total = sum(energies.values())
    if total <= 0.0:
        return {item.name: 0.0 for item in stats}
    return {name: energy / total for name, energy in energies.items()}


def sort_stems_by_share(shares: dict[str, float]) -> list[str]:
    order_index = {name: index for index, name in enumerate(STEM_ORDER)}
    return sorted(
        shares, key=lambda name: (-shares[name], order_index.get(name, len(STEM_ORDER)))
    )


def window_active_threshold(window: WindowStats) -> float:
    return max(window.p90_db - 24.0, window.p50_db - 9.0, -60.0)


def focus_solo_ratio(stem: str, windows: dict[str, WindowStats]) -> float:
    focus_window = windows.get(stem)
    if focus_window is None or not focus_window.envelope_db:
        return 0.0
    compare_windows = {
        name: window
        for name, window in windows.items()
        if name not in {stem, "drums"} and window.envelope_db
    }
    count = len(focus_window.envelope_db)
    for window in compare_windows.values():
        count = min(count, len(window.envelope_db))
    if count <= 0:
        return 0.0
    focus_threshold = window_active_threshold(focus_window)
    other_thresholds = {
        name: window_active_threshold(window)
        for name, window in compare_windows.items()
    }
    dominance_margin_db = 6.0
    active_frames = 0
    solo_frames = 0
    for index in range(count):
        focus_db = focus_window.envelope_db[index]
        if focus_db < focus_threshold:
            continue
        active_frames += 1
        has_competing_stem = any(
            window.envelope_db[index] >= other_thresholds[name]
            and window.envelope_db[index] >= focus_db - dominance_margin_db
            for name, window in compare_windows.items()
        )
        if not has_competing_stem:
            solo_frames += 1
    if active_frames <= 0:
        return 0.0
    return solo_frames / active_frames


def focus_candidate_metrics(
    stem: str,
    shares: dict[str, float],
    windows: dict[str, WindowStats],
) -> tuple[float, float, float]:
    share = shares.get(stem, 0.0)
    solo_ratio = focus_solo_ratio(stem, windows)
    window = windows.get(stem)
    active = window.active_ratio if window is not None else 0.5
    energy_score = clamp((share - 0.04) / 0.28, 0.0, 1.0)
    solo_score = clamp(solo_ratio / 0.32, 0.0, 1.0)
    active_score = clamp((active - 0.15) / 0.55, 0.0, 1.0)
    score = clamp(
        0.50 * energy_score + 0.35 * solo_score + 0.15 * active_score, 0.0, 1.0
    )
    strength = clamp(0.18 + 0.72 * score, 0.18, 0.90)
    return score, strength, solo_ratio


def focus_gain_for_strength(strength: float) -> float:
    return 0.25 * clamp(strength, 0.0, 1.0)


def focus_bed_duck_for_strength(strength: float) -> float:
    return 0.25 + 0.65 * clamp(strength, 0.0, 1.0)


def focus_group_stems(decision: FocusDecision) -> set[str]:
    if decision.focus_weights:
        return {
            stem for stem, weight in decision.focus_weights.items() if weight >= 0.05
        }
    return {decision.stem} if decision.stem is not None else set()


def focus_weight(decision: FocusDecision, stem: str) -> float:
    if decision.focus_weights:
        return clamp(decision.focus_weights.get(stem, 0.0), 0.0, 1.0)
    return 1.0 if stem == decision.stem else 0.0


def ordered_focus_weights(decision: FocusDecision) -> list[tuple[str, float]]:
    weights = {
        stem: focus_weight(decision, stem) for stem in focus_group_stems(decision)
    }
    if not weights:
        return []
    order_index = {name: index for index, name in enumerate(STEM_ORDER)}
    names: list[str] = []
    if decision.stem in weights:
        names.append(decision.stem)
    names.extend(
        name
        for name in sorted(
            weights, key=lambda item: order_index.get(item, len(STEM_ORDER))
        )
        if name not in names
    )
    return [(name, weights[name]) for name in names]


def focus_label(decision: FocusDecision) -> str:
    ordered = ordered_focus_weights(decision)
    if not ordered:
        return "off"
    return "+".join(stem for stem, _weight in ordered)


def sanitized_focus_label(decision: FocusDecision) -> str:
    return (
        "_".join(safe_label(stem) for stem, _weight in ordered_focus_weights(decision))
        or "off"
    )


def auto_focus_weight_map(
    *,
    selected: str | None,
    vocal_anchor: str | None,
    shares: dict[str, float],
    windows: dict[str, WindowStats],
    score: float,
    strength: float,
) -> tuple[dict[str, float], str]:
    if selected is None:
        return {}, ""

    weights = {selected: 1.0}
    note = ""
    if vocal_anchor is not None and vocal_anchor != selected:
        weights[vocal_anchor] = clamp(0.38 + 0.22 * strength, 0.38, 0.60)
        note = "vocal/support weighted focus"
        return weights, note

    drum_share = shares.get("drums", 0.0)
    selected_share = shares.get(selected, 0.0)
    drum_window = windows.get("drums")
    drum_active = drum_window.active_ratio if drum_window is not None else 0.0
    drum_driven = (
        "drums" in shares
        and selected != "drums"
        and drum_share >= max(0.30, selected_share * 0.62)
        and drum_active >= 0.52
    )
    if drum_driven:
        weights[selected] = clamp(0.62 + 0.25 * score, 0.62, 0.87)
        weights["drums"] = clamp(0.54 + 0.62 * (drum_share - 0.30), 0.54, 0.88)
        note = "drum/support weighted focus"
    return weights, note


def ranked_focus_candidates(
    shares: dict[str, float],
    windows: dict[str, WindowStats],
    *,
    exclude_vocals: bool,
) -> list[tuple[float, float, float, str]]:
    excluded = {"drums"}
    if exclude_vocals:
        excluded.add("vocals")
    candidates = []
    for stem in shares:
        if stem in excluded:
            continue
        score, strength, solo_ratio = focus_candidate_metrics(stem, shares, windows)
        candidates.append((score, strength, solo_ratio, stem))
    order_index = {name: index for index, name in enumerate(STEM_ORDER)}
    return sorted(
        candidates,
        key=lambda item: (
            -item[0],
            -shares.get(item[3], 0.0),
            order_index.get(item[3], len(STEM_ORDER)),
        ),
    )


def decide_focus_stem(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    *,
    mode: str,
    manual_stem: str | None,
    vocal_min_share: float,
) -> FocusDecision:
    shares = stem_energy_shares(stats)
    available = set(shares)
    vocal_anchor = (
        "vocals"
        if "vocals" in available and shares.get("vocals", 0.0) >= vocal_min_share
        else None
    )
    support_candidates = ranked_focus_candidates(shares, windows, exclude_vocals=True)
    support_stem = support_candidates[0][3] if support_candidates else None

    if mode == "off":
        return FocusDecision(
            mode=mode,
            stem=None,
            shares=shares,
            reason="disabled",
            support_stem=support_stem,
            vocal_anchor=vocal_anchor,
        )

    if mode == "manual":
        if manual_stem is None:
            raise SystemExit("--focus-stem-mode manual requires --focus-stem.")
        if manual_stem not in available:
            raise SystemExit(
                f"--focus-stem {manual_stem!r} was requested but that stem was not found."
            )
        score, strength, solo_ratio = focus_candidate_metrics(
            manual_stem, shares, windows
        )
        vocal_duck_db = 0.0
        return FocusDecision(
            mode=mode,
            stem=manual_stem,
            shares=shares,
            reason=f"manual selection ({manual_stem})",
            support_stem=support_stem,
            strength=strength,
            solo_ratio=solo_ratio,
            score=score,
            vocal_anchor=vocal_anchor,
            vocal_duck_db=vocal_duck_db,
            focus_gain_db=focus_gain_for_strength(strength),
            bed_duck_db=focus_bed_duck_for_strength(strength),
            focus_weights={manual_stem: 1.0},
        )

    candidates = ranked_focus_candidates(
        shares, windows, exclude_vocals=vocal_anchor is not None
    )
    selected = candidates[0][3] if candidates else None
    if selected is None:
        reason = "no usable stems"
        score = 0.0
        strength = 0.0
        solo_ratio = 0.0
    elif vocal_anchor is not None:
        score, strength, solo_ratio, _stem = candidates[0]
        reason = "vocal anchor present; selected non-vocal support focus by energy/solo score"
    else:
        score, strength, solo_ratio, _stem = candidates[0]
        reason = f"vocals below {vocal_min_share * 100.0:.1f}%; selected non-drum focus by energy/solo score"
    focus_weights, focus_note = auto_focus_weight_map(
        selected=selected,
        vocal_anchor=vocal_anchor,
        shares=shares,
        windows=windows,
        score=score,
        strength=strength,
    )
    if focus_note:
        reason = f"{reason}; {focus_note}"
    vocal_duck_db = 0.0
    return FocusDecision(
        mode=mode,
        stem=selected,
        shares=shares,
        reason=reason,
        support_stem=support_stem,
        strength=strength,
        solo_ratio=solo_ratio,
        score=score,
        vocal_anchor=vocal_anchor,
        vocal_duck_db=vocal_duck_db,
        focus_gain_db=focus_gain_for_strength(strength),
        bed_duck_db=focus_bed_duck_for_strength(strength),
        focus_weights=focus_weights,
    )


def scale_pan_expr(expr: str, scale: float) -> str:
    def repl(match: re.Match[str]) -> str:
        raw_value = match.group(1)
        scaled = float(raw_value) * scale
        prefix = "+" if raw_value.startswith("+") and scaled >= 0.0 else ""
        return f"{prefix}{fmt(scaled)}*{match.group(2)}"

    return PAN_TERM_RE.sub(repl, expr)


def focus_channel_scale(output_channel: str, strength: float) -> float:
    if output_channel == "LFE":
        return 1.0
    if output_channel == "FC":
        return 1.0 + 0.07 * strength
    if output_channel in FRONT_CHANNELS:
        return 1.0 + 0.03 * strength
    if output_channel in TOP_CHANNELS:
        return 1.0 - 0.14 * strength
    if output_channel in SPACE_CHANNELS:
        return 1.0 - 0.08 * strength
    return 1.0


def apply_focus_to_placements(
    placements: dict[str, Placement], decision: FocusDecision
) -> dict[str, Placement]:
    focus_weights = {
        stem: weight
        for stem, weight in ordered_focus_weights(decision)
        if stem in placements and weight >= 0.05
    }
    if not focus_weights:
        return placements
    strength = clamp(decision.strength, 0.0, 1.0)
    focused: dict[str, Placement] = {}
    for name, placement in placements.items():
        weight = focus_weights.get(name, 0.0)
        if weight <= 0.0:
            focused[name] = placement
            continue
        weighted_strength = strength * weight
        pan = {
            channel: scale_pan_expr(
                expr, focus_channel_scale(channel, weighted_strength)
            )
            for channel, expr in placement.pan.items()
        }
        focused[name] = Placement(
            stem=placement.stem,
            role=f"{placement.role}; focus weight={weight:.2f} strength={weighted_strength:.2f}",
            pan=pan,
            lfe_amount=placement.lfe_amount,
            lfe_cutoff_hz=placement.lfe_cutoff_hz,
        )
    return focused


def focus_duck_targets(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    decision: FocusDecision,
) -> dict[str, FocusDuckTarget]:
    if decision.stem is None or decision.bed_duck_db <= 0.0:
        return {}
    by_name = {item.name: item for item in stats}
    focus_stats = by_name.get(decision.stem)
    if focus_stats is None:
        return {}
    focus_bands = stats_band_fraction_map(focus_stats)
    focus_group = focus_group_stems(decision)
    targets: dict[str, FocusDuckTarget] = {}
    for item in stats:
        if item.name in focus_group:
            continue
        if not should_duck_for_focus(item.name, decision):
            continue
        item_bands = stats_band_fraction_map(item)
        overlaps = {
            band: min(focus_bands[band], item_bands[band]) for band in BAND_NAMES
        }
        selected_bands = selected_overlap_bands(overlaps)
        if not selected_bands:
            continue
        spectral_overlap = sum(overlaps.values())
        collision = placement_collision_score(
            focus_stats, item, windows.get(decision.stem), windows.get(item.name)
        )
        overlap_score = clamp(0.62 * spectral_overlap + 0.38 * collision, 0.0, 1.0)
        if overlap_score < 0.18:
            continue
        duck_db = decision.bed_duck_db * (0.35 + 0.65 * overlap_score)
        targets[item.name] = FocusDuckTarget(
            stem=item.name,
            bands=selected_bands,
            overlap_score=overlap_score,
            duck_db=duck_db,
        )
    return targets


def temporal_duck_fragility_score(
    stats: StemStats, window: WindowStats | None
) -> float:
    if stats.name in {"bass", "drums"} or window is None or not window.envelope_db:
        return 0.0
    dynamics_db = max(0.0, window.p90_db - window.p50_db)
    sparse_score = clamp((0.72 - window.active_ratio) / 0.55, 0.0, 1.0)
    burst_score = clamp((dynamics_db - 8.0) / 22.0, 0.0, 1.0)
    sustain_gap = 1.0 - clamp(window.sustain_score, 0.0, 1.0)
    tonal_exposure = clamp(
        0.55 * bright_score(stats) + 0.45 * wide_score(stats), 0.0, 1.0
    )
    return clamp(
        0.34 * sparse_score
        + 0.30 * burst_score
        + 0.24 * sustain_gap
        + 0.12 * tonal_exposure,
        0.0,
        1.0,
    )


def apply_temporal_duck_guard(
    stats: list[StemStats],
    windows: dict[str, WindowStats],
    decision: FocusDecision,
    *,
    mode: str,
    strength: float,
) -> TemporalDuckGuardResult:
    if mode == "off" or not decision.bed_duck_targets:
        return TemporalDuckGuardResult(items=[])

    strength = clamp(float(strength), 0.0, 1.0)
    if strength <= 0.0:
        return TemporalDuckGuardResult(items=[])

    by_name = {item.name: item for item in stats}
    items: list[TemporalDuckGuardItem] = []
    for stem, target in list(decision.bed_duck_targets.items()):
        stats_item = by_name.get(stem)
        window = windows.get(stem)
        if stats_item is None or window is None:
            continue
        fragility = temporal_duck_fragility_score(stats_item, window)
        if fragility < 0.22:
            continue

        protection = clamp(strength * (0.20 + 0.80 * fragility), 0.0, 0.85)
        duck_scale = 1.0 - 0.52 * protection
        adjusted_duck = max(0.12, target.duck_db * duck_scale)
        decision.bed_duck_targets[stem] = FocusDuckTarget(
            stem=target.stem,
            bands=target.bands,
            overlap_score=target.overlap_score,
            duck_db=adjusted_duck,
            bass_drum_bands=target.bass_drum_bands,
            temporal_protection=max(target.temporal_protection, protection),
        )
        items.append(
            TemporalDuckGuardItem(
                stem=stem,
                active_ratio=window.active_ratio,
                dynamics_db=max(0.0, window.p90_db - window.p50_db),
                sustain_score=window.sustain_score,
                fragility_score=fragility,
                protection=protection,
                duck_db_before=target.duck_db,
                duck_db_after=adjusted_duck,
            )
        )

    return TemporalDuckGuardResult(items=items)


def print_temporal_duck_guard_result(result: TemporalDuckGuardResult) -> None:
    if not result.items:
        return
    print("Temporal duck guard:", flush=True)
    print("stem      active    dyn sustain fragile protect   duck", flush=True)
    for item in sorted(
        result.items,
        key=lambda entry: (
            STEM_ORDER.index(entry.stem) if entry.stem in STEM_ORDER else 99
        ),
    ):
        print(
            f"{item.stem:<9} "
            f"{item.active_ratio:>6.2f} "
            f"{item.dynamics_db:>6.1f} "
            f"{item.sustain_score:>7.2f} "
            f"{item.fragility_score:>7.2f} "
            f"{item.protection:>7.2f} "
            f"{item.duck_db_before:>4.1f}->{item.duck_db_after:<4.1f}dB",
            flush=True,
        )


def should_duck_for_focus(stem: str, decision: FocusDecision) -> bool:
    if decision.stem is None or stem in focus_group_stems(decision):
        return False
    if decision.bed_duck_targets:
        return stem in decision.bed_duck_targets
    if stem == decision.vocal_anchor:
        return False
    return stem != "vocals"


__all__ = [
    "front51_lfe_amounts",
    "decide_placement_front51",
    "decide_placement",
    "gain_filter",
    "stem_eq_filter",
    "stem_eq_band_map",
    "sorted_stem_eq_bands",
    "compute_stem_aware_eq",
    "print_stem_aware_eq",
    "gain_suffix",
    "stem_energy_shares",
    "sort_stems_by_share",
    "window_active_threshold",
    "focus_solo_ratio",
    "focus_candidate_metrics",
    "focus_gain_for_strength",
    "focus_bed_duck_for_strength",
    "focus_group_stems",
    "focus_weight",
    "ordered_focus_weights",
    "focus_label",
    "sanitized_focus_label",
    "auto_focus_weight_map",
    "ranked_focus_candidates",
    "decide_focus_stem",
    "scale_pan_expr",
    "focus_channel_scale",
    "apply_focus_to_placements",
    "focus_duck_targets",
    "temporal_duck_fragility_score",
    "apply_temporal_duck_guard",
    "print_temporal_duck_guard_result",
    "should_duck_for_focus",
]
