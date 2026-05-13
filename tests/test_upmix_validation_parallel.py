from __future__ import annotations

import sys
from pathlib import Path

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import validation  # noqa: E402


def test_validation_worker_count_defaults_and_env(monkeypatch):
    monkeypatch.delenv("STEMS_UPMIXER_VALIDATION_WORKERS", raising=False)
    monkeypatch.setattr(validation.os, "cpu_count", lambda: 16)

    assert validation.validation_worker_count(3) == 3
    assert validation.validation_worker_count(2) == 2
    assert validation.validation_worker_count(1) == 1

    monkeypatch.setenv("STEMS_UPMIXER_VALIDATION_WORKERS", "1")
    assert validation.validation_worker_count(3) == 1

    monkeypatch.setenv("STEMS_UPMIXER_VALIDATION_WORKERS", "8")
    assert validation.validation_worker_count(3) == 3

    monkeypatch.setenv("STEMS_UPMIXER_VALIDATION_WORKERS", "invalid")
    assert validation.validation_worker_count(3) == 1


def test_map_validation_tasks_preserves_order():
    result = validation.map_validation_tasks(
        [4, 1, 3],
        lambda value: value * 10,
        max_workers=2,
    )

    assert result == [40, 10, 30]


def test_validation_translation_status_warns_on_fold_down_peak_headroom():
    status, warnings = validation.validation_translation_status(
        {
            "sample_peak_dbfs": 0.0,
            "stereo_correlation": 0.9,
            "mono_fold_loss_db": -0.4,
            "phase_risk": 0.0,
            "side_minus_mid_db": -6.0,
        }
    )

    assert status == "warn"
    assert "stereo fold-down reaches 0 dBFS headroom" in warnings


def test_build_validation_summary_keeps_distinct_fold_paths(monkeypatch, tmp_path):
    output_dir = tmp_path
    bed = output_dir / "bed.wav"
    stereo = output_dir / "stereo.flac"
    surround = output_dir / "surround.flac"
    apple = output_dir / "apple.mp4"
    for path in (bed, stereo, surround, apple):
        path.write_bytes(b"audio")

    def fake_render_51_to_stereo_fold_wav(_ffmpeg, _input_path, output_path, _gain):
        output_path.write_bytes(b"fold")

    def fake_validate_audio_output(**kwargs):
        stereo_path = kwargs["stereo_path"]
        return {
            "status": "pass",
            "warnings": [],
            "path": kwargs["path"].name,
            "stereo_path": None if stereo_path is None else stereo_path.name,
        }

    monkeypatch.setenv("STEMS_UPMIXER_VALIDATION_WORKERS", "1")
    monkeypatch.setattr(
        validation, "render_51_to_stereo_fold_wav", fake_render_51_to_stereo_fold_wav
    )
    monkeypatch.setattr(validation, "validate_audio_output", fake_validate_audio_output)
    monkeypatch.setattr(
        validation,
        "build_temporal_consistency_validation",
        lambda **_kwargs: {"status": "pass"},
    )

    summary = validation.build_mastering_validation_summary(
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        output_dir=output_dir,
        source_path=None,
        bed_path=bed,
        stereo_path=stereo,
        surround_path=surround,
        apple_path=apple,
        render_stereo=True,
        render_surround=True,
        render_apple_tv=True,
        stereo_target_i=-14.0,
        stereo_target_tp=-1.0,
        stereo_target_lra=11.0,
        surround_target_i=-14.0,
        surround_target_tp=-1.0,
        surround_target_lra=11.0,
        stereo_lfe_fold_gain=0.0,
        surround_lfe_fold_gain=0.2,
        automation_lanes=None,
        temporal_result=None,
    )

    outputs = summary["outputs"]
    assert outputs["stereo_flac"]["stereo_path"] == "stereo.flac"
    assert (
        outputs["surround_flac"]["stereo_path"]
        == "surround_flac_stereo_fold_validation.wav"
    )
    assert outputs["apple_tv"]["stereo_path"] == "apple_tv_stereo_fold_validation.wav"


def test_validate_audio_output_uses_precomputed_loudness(monkeypatch, tmp_path):
    audio = tmp_path / "surround.flac"
    audio.write_bytes(b"audio")
    folded = tmp_path / "fold.wav"
    folded.write_bytes(b"fold")

    def fail_measure_audio_loudness(*_args, **_kwargs):
        raise AssertionError("loudness should be reused from render")

    monkeypatch.setattr(
        validation, "measure_audio_loudness", fail_measure_audio_loudness
    )
    monkeypatch.setattr(
        validation,
        "media_stream_summary",
        lambda _ffprobe, path: {"path": str(path), "channels": 6},
    )
    monkeypatch.setattr(
        validation,
        "stereo_validation_metrics",
        lambda _path: {
            "stereo_correlation": 0.9,
            "mono_fold_loss_db": -0.4,
            "phase_risk": 0.0,
        },
    )

    result = validation.validate_audio_output(
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        path=audio,
        target_i=-14.0,
        target_tp=-1.0,
        target_lra=11.0,
        stereo_path=folded,
        precomputed_loudness={
            "integrated_lufs": -14.02,
            "true_peak_dbtp": -1.12,
            "lra_lu": 9.4,
            "threshold_lufs": -24.0,
        },
    )

    assert result["loudness_source"] == "render_loudnorm"
    assert result["loudness"]["integrated_lufs"] == -14.02
    assert result["status"] == "pass"
