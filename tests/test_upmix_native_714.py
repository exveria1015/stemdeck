from __future__ import annotations

import json
import sys
from pathlib import Path

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import filters  # noqa: E402
from upmixer.models import (  # noqa: E402
    AutomationLanes,
    FocusDecision,
    FocusDuckTarget,
    Placement,
    PriorityDuckTarget,
)


def make_lanes() -> AutomationLanes:
    values = (0.0, 1.0)
    return AutomationLanes(
        times=(0.0, 4.0),
        step_sec=4.0,
        vocal_protect=values,
        chorus_space=values,
        breakdown_air=values,
        low_anchor=values,
        low_tighten=values,
        mid_decongest=values,
        harshness_guard=(0.0, 0.0),
        fold_safety=(0.0, 0.0),
        solo_feature={"other": values},
    )


def test_native_714_plan_serializes_pan_gain_eq_and_lfe(tmp_path: Path):
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"RIFF")

    plan = filters._native_714_plan(
        input_order=["vocals"],
        stems={"vocals": stem},
        placements={
            "vocals": Placement(
                stem="vocals",
                role="test",
                pan={"FL": "0.50*FL", "FR": "0.25*FR", "FC": "0.10*FL+0.20*FR"},
                lfe_amount=0.08,
                lfe_cutoff_hz=90,
            )
        },
        output_path=tmp_path / "bed.wav",
        master_gain=0.84,
        stem_gain_db={"vocals": 6.0},
        stem_eq_bands={},
        space_bed_mode="off",
        space_bed_stems=(),
        space_bed_strength=0.0,
        automation_lanes=None,
        focus_decision=FocusDecision(mode="off", stem=None, shares={}, reason="test"),
        priority_duck_targets={},
    )

    rendered_stem = plan["stems"][0]  # type: ignore[index]
    assert rendered_stem["pan"][0] == [0.5, 0.0]
    assert rendered_stem["pan"][1] == [0.0, 0.25]
    assert rendered_stem["pan"][2] == [0.1, 0.2]
    assert rendered_stem["lfe_amount"] == 0.08
    assert rendered_stem["lfe_cutoff_hz"] == 90.0
    assert float(rendered_stem["gain"]) > 1.99
    assert rendered_stem["low_tighten"] == []
    assert rendered_stem["mid_decongest"] == []
    assert rendered_stem["space_bed"] is None
    assert rendered_stem["ducking"] == []


def test_native_714_plan_serializes_automation_and_space_bed(tmp_path: Path):
    stem = tmp_path / "drums.wav"
    stem.write_bytes(b"RIFF")

    plan = filters._native_714_plan(
        input_order=["drums"],
        stems={"drums": stem},
        placements={
            "drums": Placement(
                stem="drums",
                role="test",
                pan={
                    "FL": "1*FL",
                    "FR": "1*FR",
                    "BL": "0.20*FL",
                    "BR": "0.20*FR",
                },
            )
        },
        output_path=tmp_path / "bed.wav",
        master_gain=1.0,
        stem_gain_db={},
        stem_eq_bands={},
        space_bed_mode="auto",
        space_bed_stems=("drums",),
        space_bed_strength=0.5,
        automation_lanes=make_lanes(),
        focus_decision=FocusDecision(mode="off", stem=None, shares={}, reason="test"),
        priority_duck_targets={},
    )

    rendered_stem = plan["stems"][0]  # type: ignore[index]
    assert rendered_stem["low_tighten"]
    assert rendered_stem["mid_decongest"] == []
    assert rendered_stem["space_bed"] is not None
    space_bed = rendered_stem["space_bed"]
    assert space_bed["pan"][4] == [0.2, 0.0]
    assert space_bed["left_delay_ms"] == 12.0
    assert space_bed["right_delay_ms"] == 25.0
    assert space_bed["automation"]


def test_render_714_uses_native_when_plan_is_supported(monkeypatch, tmp_path: Path):
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"RIFF")
    output = tmp_path / "bed.wav"
    fake_binary = tmp_path / "stemdeck-native-audio"
    fake_binary.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_binary.chmod(0o755)
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        assert cmd[0] == str(fake_binary)
        assert cmd[1:3] == ["render714", "--plan"]
        plan = json.loads(Path(cmd[3]).read_text(encoding="utf-8"))
        Path(plan["output"]).write_bytes(b"RIFF")

    monkeypatch.setattr(filters, "native_714_binary", lambda: fake_binary)
    monkeypatch.setattr(filters, "run", fake_run)

    filters.render_714(
        "ffmpeg-disabled",
        {"vocals": stem},
        {"vocals": Placement("vocals", "test", {"FL": "1*FL", "FR": "1*FR"})},
        output,
        1.0,
        {},
        {},
        FocusDecision(mode="off", stem=None, shares={}, reason="test"),
        {},
        None,
        "off",
        (),
        0.0,
        None,
    )

    assert output.read_bytes() == b"RIFF"
    assert len(calls) == 1


def test_render_714_skips_native_when_disabled(monkeypatch, tmp_path: Path):
    stem = tmp_path / "vocals.wav"
    stem.write_bytes(b"RIFF")
    output = tmp_path / "bed.wav"
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs):
        calls.append(cmd)
        assert cmd[0] == "ffmpeg-test"
        output.write_bytes(b"RIFF")

    def fail_native():
        raise AssertionError("native renderer should be skipped")

    monkeypatch.setattr(filters, "native_714_binary", fail_native)
    monkeypatch.setattr(filters, "run", fake_run)

    filters.render_714(
        "ffmpeg-test",
        {"vocals": stem},
        {"vocals": Placement("vocals", "test", {"FL": "1*FL", "FR": "1*FR"})},
        output,
        1.0,
        {},
        {},
        FocusDecision(mode="off", stem=None, shares={}, reason="test"),
        {},
        None,
        "off",
        (),
        0.0,
        None,
        native_enabled=False,
    )

    assert output.read_bytes() == b"RIFF"
    assert calls


def test_native_714_plan_serializes_focus_and_priority_ducking(tmp_path: Path):
    vocals = tmp_path / "vocals.wav"
    other = tmp_path / "other.wav"
    drums = tmp_path / "drums.wav"
    for path in (vocals, other, drums):
        path.write_bytes(b"RIFF")

    plan = filters._native_714_plan(
        input_order=["vocals", "other", "drums"],
        stems={"vocals": vocals, "other": other, "drums": drums},
        placements={
            "vocals": Placement("vocals", "test", {"FL": "1*FL"}),
            "other": Placement("other", "test", {"FR": "1*FR"}),
            "drums": Placement("drums", "test", {"FL": "1*FL"}),
        },
        output_path=tmp_path / "bed.wav",
        master_gain=1.0,
        stem_gain_db={},
        stem_eq_bands={},
        space_bed_mode="off",
        space_bed_stems=(),
        space_bed_strength=0.0,
        automation_lanes=None,
        focus_decision=FocusDecision(
            mode="auto",
            stem="other",
            shares={},
            reason="test",
            strength=0.7,
            vocal_anchor="vocals",
            vocal_duck_db=0.4,
            bed_duck_db=0.5,
            focus_weights={"other": 1.0},
            bed_duck_targets={
                "drums": FocusDuckTarget(
                    stem="drums", bands=("low", "mid"), overlap_score=0.6, duck_db=0.5
                )
            },
        ),
        priority_duck_targets={
            "other": [
                PriorityDuckTarget(
                    stem="other",
                    protector="drums",
                    bands=("low",),
                    overlap_score=0.8,
                    duck_db=0.7,
                )
            ]
        },
    )

    by_path = {Path(stem["path"]).name: stem for stem in plan["stems"]}  # type: ignore[index]
    other_ducking = by_path["other.wav"]["ducking"]
    assert [op["mode"] for op in other_ducking] == ["full", "full"]
    assert other_ducking[0]["compressor"]["link"] == "maximum"
    assert other_ducking[0]["sidechains"][0]["path"] == str(vocals)
    assert other_ducking[1]["sidechain_filter"] == "low180"
    assert other_ducking[1]["sidechains"][0]["path"] == str(drums)

    drums_ducking = by_path["drums.wav"]["ducking"]
    assert drums_ducking[0]["mode"] == "bands3"
    assert {band["band"] for band in drums_ducking[0]["bands"]} == {"low", "mid"}


def test_native_714_accepts_sidechain_ducking_without_ffmpeg_fallback(tmp_path: Path):
    reason = filters.native_714_unsupported_reason(
        input_order=["vocals", "other"],
        placements={
            "vocals": Placement("vocals", "test", {"FL": "1*FL"}),
            "other": Placement("other", "test", {"FR": "1*FR"}),
        },
        focus_decision=FocusDecision(
            mode="auto",
            stem="other",
            shares={},
            reason="test",
            vocal_anchor="vocals",
            vocal_duck_db=0.4,
        ),
        priority_duck_targets={},
        space_bed_mode="off",
        space_bed_stems=(),
        automation_lanes=None,
    )

    assert reason is None


def test_native_714_accepts_automation_and_space_bed():
    reason = filters.native_714_unsupported_reason(
        input_order=["drums"],
        placements={
            "drums": Placement(
                "drums",
                "test",
                {"FL": "1*FL", "FR": "1*FR", "BL": "0.20*FL"},
            )
        },
        focus_decision=FocusDecision(mode="off", stem=None, shares={}, reason="test"),
        priority_duck_targets={},
        space_bed_mode="auto",
        space_bed_stems=("drums",),
        automation_lanes=make_lanes(),
    )

    assert reason is None
