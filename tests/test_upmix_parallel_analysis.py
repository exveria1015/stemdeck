from __future__ import annotations

import sys
from pathlib import Path

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import cli  # noqa: E402
from upmixer.models import (  # noqa: E402
    AutomationLanes,
    StemStats,
    TemporalFeedbackResult,
    TemporalRemasterResult,
    VolumeStats,
)


def make_stem_stats(name: str) -> StemStats:
    zero = VolumeStats(0.0, 0.0)
    return StemStats(
        name=name,
        path=Path(f"/tmp/{name}.wav"),
        full=zero,
        low=zero,
        mid=zero,
        high=zero,
        mono=zero,
        side=zero,
    )


def make_lanes() -> AutomationLanes:
    values = (0.0, 0.5)
    return AutomationLanes(
        times=(0.0, 4.0),
        step_sec=4.0,
        vocal_protect=values,
        chorus_space=values,
        breakdown_air=values,
        low_anchor=values,
        low_tighten=values,
        mid_decongest=values,
        harshness_guard=values,
        fold_safety=values,
        solo_feature={"other": values},
    )


def test_analysis_worker_count_defaults_are_bounded(monkeypatch):
    monkeypatch.delenv("STEMS_UPMIXER_ANALYSIS_WORKERS", raising=False)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 16)

    assert cli.analysis_worker_count("numpy", 6) == 4
    assert cli.analysis_worker_count("auto", 3) == 3
    assert cli.analysis_worker_count("ffmpeg", 6) == 2
    assert cli.analysis_worker_count("numpy", 1) == 1


def test_analysis_worker_count_env_override(monkeypatch):
    monkeypatch.setenv("STEMS_UPMIXER_ANALYSIS_WORKERS", "1")
    assert cli.analysis_worker_count("numpy", 6) == 1

    monkeypatch.setenv("STEMS_UPMIXER_ANALYSIS_WORKERS", "9")
    assert cli.analysis_worker_count("numpy", 6) == 6

    monkeypatch.setenv("STEMS_UPMIXER_ANALYSIS_WORKERS", "not-an-int")
    assert cli.analysis_worker_count("numpy", 6) == 1


def test_map_ordered_parallel_preserves_input_order():
    values = [5, 1, 3, 2]

    result = cli.map_ordered_parallel(values, lambda value: value * 10, max_workers=3)

    assert result == [50, 10, 30, 20]


def test_render_worker_count_defaults_and_env(monkeypatch):
    monkeypatch.delenv("STEMS_UPMIXER_RENDER_WORKERS", raising=False)
    monkeypatch.setattr(cli.os, "cpu_count", lambda: 8)

    assert cli.render_worker_count(3) == 2
    assert cli.render_worker_count(1) == 1

    monkeypatch.setenv("STEMS_UPMIXER_RENDER_WORKERS", "1")
    assert cli.render_worker_count(3) == 1

    monkeypatch.setenv("STEMS_UPMIXER_RENDER_WORKERS", "4")
    assert cli.render_worker_count(3) == 3

    monkeypatch.setenv("STEMS_UPMIXER_RENDER_WORKERS", "invalid")
    assert cli.render_worker_count(3) == 1


def test_run_render_tasks_runs_all_tasks_in_order(monkeypatch):
    monkeypatch.setenv("STEMS_UPMIXER_RENDER_WORKERS", "1")
    calls: list[str] = []

    cli.run_render_tasks(
        [
            ("first", lambda: calls.append("first")),
            ("second", lambda: calls.append("second")),
        ]
    )

    assert calls == ["first", "second"]


def test_prewarm_stem_range_energy_cache_uses_multiband_ranges(monkeypatch):
    calls: list[tuple[str, object]] = []

    def fake_range_energy(item, ranges):
        calls.append((item.name, ranges))
        return {"low": 0.25}

    monkeypatch.setattr(cli, "stem_range_energies_numpy", fake_range_energy)

    cli.prewarm_stem_range_energy_cache(
        [make_stem_stats("vocals"), make_stem_stats("drums")],
        max_workers=1,
    )

    assert calls == [
        ("vocals", cli.MULTIBAND_RANGES),
        ("drums", cli.MULTIBAND_RANGES),
    ]


def test_temporal_feedback_preflight_renders_preview_and_cleans_tmp(
    monkeypatch, tmp_path
):
    bed = tmp_path / "bed.wav"
    bed.write_bytes(b"bed")
    lanes = make_lanes()
    render_calls = []

    def fake_render(ffmpeg, bed_path, output_path, lfe_fold_gain, automation_lanes):
        render_calls.append((ffmpeg, bed_path, output_path, lfe_fold_gain, automation_lanes))
        output_path.write_bytes(b"preview")

    def fake_consistency(**kwargs):
        assert kwargs["render_path"].read_bytes() == b"preview"
        return {"status": "warn", "events": [{"type": "loudness_jump"}]}

    def fake_tune(input_lanes, summary, *, strength):
        assert input_lanes is lanes
        assert summary["temporal_consistency"]["status"] == "warn"
        assert strength == 0.5
        return TemporalFeedbackResult(True, input_lanes, event_count=1, max_damping=0.2)

    monkeypatch.setattr(cli, "render_stereo_fold_wav", fake_render)
    monkeypatch.setattr(cli, "build_temporal_consistency_validation", fake_consistency)
    monkeypatch.setattr(cli, "tune_temporal_feedback", fake_tune)

    result = cli.evaluate_temporal_feedback_preflight(
        ffmpeg="ffmpeg-test",
        output_dir=tmp_path,
        bed_path=bed,
        source_path=None,
        lfe_fold_gain=0.25,
        lanes=lanes,
        temporal_result=TemporalRemasterResult(True, lanes=lanes),
        strength=0.5,
    )

    assert result is not None
    assert result.changed
    assert render_calls[0][0] == "ffmpeg-test"
    assert render_calls[0][1] == bed
    assert render_calls[0][3] == 0.25
    assert render_calls[0][4] is lanes
    assert not (tmp_path / ".feedback_tmp").exists()
