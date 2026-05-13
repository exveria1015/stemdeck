"""Console reporting helpers for analysis and decisions."""

from __future__ import annotations

from .models import FocusDecision, Placement, PriorityDuckTarget, StemStats, WindowStats
from .placement import focus_label, ordered_focus_weights, sort_stems_by_share
from .stems import band_overlap_score, envelope_correlation, placement_collision_score


def print_analysis(stats: list[StemStats], placements: dict[str, Placement]) -> None:
    print("Stem analysis:", flush=True)
    print("stem       rms    low-mid  high-mid  side-mid  role", flush=True)
    for item in stats:
        placement = placements[item.name]
        print(
            f"{item.name:<9} "
            f"{item.full.mean_db:>6.1f} "
            f"{item.low_vs_mid_db:>8.1f} "
            f"{item.high_vs_mid_db:>9.1f} "
            f"{item.width_db:>8.1f}  "
            f"{placement.role}",
            flush=True,
        )


def print_window_analysis(
    stats: list[StemStats], windows: dict[str, WindowStats]
) -> None:
    if not windows:
        return
    print("Window analysis:", flush=True)
    print("stem      active    p50     p90  sustain", flush=True)
    for item in stats:
        window = windows.get(item.name)
        if window is None:
            continue
        print(
            f"{item.name:<9} "
            f"{window.active_ratio:>6.2f} "
            f"{window.p50_db:>7.1f} "
            f"{window.p90_db:>7.1f} "
            f"{window.sustain_score:>7.2f}",
            flush=True,
        )


def print_collision_analysis(
    stats: list[StemStats], windows: dict[str, WindowStats]
) -> None:
    if not windows:
        return
    by_name = {item.name: item for item in stats}
    pairs = [
        ("guitar", "other"),
        ("piano", "guitar"),
        ("piano", "other"),
        ("vocals", "other"),
        ("drums", "bass"),
    ]
    rows = []
    for left_name, right_name in pairs:
        left = by_name.get(left_name)
        right = by_name.get(right_name)
        if left is None or right is None:
            continue
        collision = placement_collision_score(
            left, right, windows.get(left_name), windows.get(right_name)
        )
        correlation = max(
            0.0, envelope_correlation(windows.get(left_name), windows.get(right_name))
        )
        band = band_overlap_score(left, right)
        rows.append((collision, left_name, right_name, correlation, band))
    if not rows:
        return
    print("Placement collision:", flush=True)
    print("pair            score  env    band", flush=True)
    for collision, left_name, right_name, correlation, band in sorted(
        rows, reverse=True
    ):
        print(
            f"{left_name}+{right_name:<10} {collision:>5.2f}  {correlation:>4.2f}  {band:>5.2f}",
            flush=True,
        )


def print_focus_analysis(decision: FocusDecision) -> None:
    print("Focus analysis:", flush=True)
    print("stem       share", flush=True)
    for stem in sort_stems_by_share(decision.shares):
        print(f"{stem:<9} {decision.shares[stem] * 100.0:>5.1f}%", flush=True)
    if decision.vocal_anchor is not None:
        vocal_share = decision.shares.get(decision.vocal_anchor, 0.0) * 100.0
        print(f"Vocal anchor: {decision.vocal_anchor} {vocal_share:.1f}%", flush=True)
    selected = focus_label(decision)
    primary = decision.stem or "off"
    print(
        f"Selected focus: {selected} "
        f"primary={primary} "
        f"strength={decision.strength:.2f} "
        f"gain={decision.focus_gain_db:+.2f}dB "
        f"solo={decision.solo_ratio * 100.0:.1f}% "
        f"score={decision.score:.2f} "
        f"({decision.reason})",
        flush=True,
    )
    weights = ordered_focus_weights(decision)
    if len(weights) > 1:
        print(
            "Focus weights: "
            + " ".join(f"{stem}={weight:.2f}" for stem, weight in weights),
            flush=True,
        )
    if decision.vocal_duck_db > 0.0:
        print(
            f"Vocal ducking: {selected} ducks by about {decision.vocal_duck_db:.1f} dB when vocal is active",
            flush=True,
        )
    if decision.bed_duck_db > 0.0:
        print(
            f"Focus bed ducking: non-focus beds duck by about {decision.bed_duck_db:.1f} dB on focus peaks",
            flush=True,
        )
        for target in sorted(
            decision.bed_duck_targets.values(), key=lambda item: item.stem
        ):
            protection = (
                f" protect={target.temporal_protection:.2f}"
                if target.temporal_protection > 0.0
                else ""
            )
            print(
                f"  {target.stem:<9} bands={','.join(target.bands):<12} "
                f"overlap={target.overlap_score:.2f} duck~{target.duck_db:.1f} dB{protection}",
                flush=True,
            )
    if decision.support_stem is not None:
        support_share = decision.shares.get(decision.support_stem, 0.0) * 100.0
        print(
            f"Non-vocal support candidate: {decision.support_stem} {support_share:.1f}%",
            flush=True,
        )


def print_priority_masking_analysis(
    priority: tuple[str, ...],
    targets: dict[str, list[PriorityDuckTarget]],
) -> None:
    if not priority:
        return
    print("Priority masking duck:", flush=True)
    print(f"priority: {' > '.join(priority)}", flush=True)
    if not targets:
        print("  no active priority collisions", flush=True)
        return
    for target_stem in priority[1:]:
        for target in targets.get(target_stem, ()):
            print(
                f"  {target.protector}->{target.stem:<9} "
                f"bands={','.join(target.bands):<9} "
                f"overlap={target.overlap_score:.2f} duck~{target.duck_db:.1f} dB",
                flush=True,
            )


__all__ = [
    "print_analysis",
    "print_window_analysis",
    "print_collision_analysis",
    "print_focus_analysis",
    "print_priority_masking_analysis",
]
