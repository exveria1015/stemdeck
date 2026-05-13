from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.upmix import UpmixRenderRequest, _upmix_render_args
from app.core.models import Job
from app.core.registry import _jobs
from app.pipeline.upmix import discover_upmix_outputs, run_upmix


@pytest.fixture(autouse=True)
def _isolate_registry():
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client():
    from app.main import app

    return TestClient(app)


def _arg_value(args: list[str], flag: str) -> str:
    return args[args.index(flag) + 1]


def test_discovers_named_upmix_outputs(tmp_path: Path):
    out = tmp_path / "upmix"
    out.mkdir()
    (out / "Song_natural_upmix5.1_lufs_m14_tp_m1.flac").write_bytes(b"FLAC")
    (out / "Song_natural_upmix5.1_lufs_m14_tp_m1.wav").write_bytes(b"RIFF")
    (out / "Song_natural_stereo_lufs_m14_tp_m1.flac").write_bytes(b"FLAC")
    (out / "Song_7.1.4_natural.wav").write_bytes(b"RIFF")
    (out / "Song_natural_upmix5.1_lufs_m14_tp_m1_ddp5.1_apple_tv.mp4").write_bytes(b"MP4")
    (out / "Song_temporal_mastering_intent.json").write_text("{}", encoding="utf-8")

    outputs = discover_upmix_outputs("abcdefabcdef", out)

    assert {item["kind"] for item in outputs} == {"bed", "surround", "stereo", "apple_tv"}
    assert any(item["label"] == "5.1 WAV" for item in outputs)
    assert all(str(item["url"]).startswith("/api/jobs/abcdefabcdef/upmix/") for item in outputs)


def test_run_upmix_skips_apple_tv_by_default(monkeypatch, tmp_path: Path):
    job = Job(id="abcdefabcdef", status="done", title="Song")
    job_dir = tmp_path / job.id
    stems_dir = job_dir / "stems"
    output_dir = job_dir / "upmix"
    stems_dir.mkdir(parents=True)
    output_dir.mkdir()
    source = job_dir / "source.wav"
    source.write_bytes(b"RIFF")
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")
    (output_dir / "song_7.1.4_auto.wav").write_bytes(b"RIFF")
    script = tmp_path / "upmix_cli.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    class FakeStdout:
        def __init__(self):
            self.lines = ["Done.\n", ""]

        def readline(self):
            return self.lines.pop(0)

    class FakeProc:
        stdout = FakeStdout()
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_SCRIPT", script)
    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_DIR", tmp_path)
    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_ARGS", [])
    monkeypatch.setattr("app.pipeline.upmix.subprocess.Popen", fake_popen)

    run_upmix(job, source, job_dir, ["vocals"])

    assert "--skip-apple-tv" in captured["cmd"]
    assert _arg_value(captured["cmd"], "--lfe-mode") == "off"
    assert _arg_value(captured["cmd"], "--mix-profile") == "auto"
    assert _arg_value(captured["cmd"], "--space-bed") == "auto"
    assert _arg_value(captured["cmd"], "--space-bed-strength") == "0.85"
    assert _arg_value(captured["cmd"], "--temporal-remaster") == "auto"
    assert _arg_value(captured["cmd"], "--temporal-mastering") == "auto"
    assert _arg_value(captured["cmd"], "--brickwall-recovery") == "auto"
    assert _arg_value(captured["cmd"], "--stem-de-limiter") == "auto"
    assert _arg_value(captured["cmd"], "--post-render-correct") == "auto"


def test_run_upmix_can_leave_apple_tv_enabled_for_explicit_render(monkeypatch, tmp_path: Path):
    job = Job(id="abcdefabcdef", status="done", title="Song")
    job_dir = tmp_path / job.id
    stems_dir = job_dir / "stems"
    output_dir = job_dir / "upmix"
    stems_dir.mkdir(parents=True)
    output_dir.mkdir()
    source = job_dir / "source.wav"
    source.write_bytes(b"RIFF")
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")
    (output_dir / "song_7.1.4_auto.wav").write_bytes(b"RIFF")
    script = tmp_path / "upmix_cli.py"
    script.write_text("print('ok')\n", encoding="utf-8")
    captured: dict[str, list[str]] = {}

    class FakeStdout:
        def __init__(self):
            self.lines = ["Done.\n", ""]

        def readline(self):
            return self.lines.pop(0)

    class FakeProc:
        stdout = FakeStdout()
        returncode = 0

        def poll(self):
            return self.returncode

        def wait(self):
            return self.returncode

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_SCRIPT", script)
    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_DIR", tmp_path)
    monkeypatch.setattr("app.pipeline.upmix.UPMIXER_ARGS", [])
    monkeypatch.setattr("app.pipeline.upmix.subprocess.Popen", fake_popen)

    run_upmix(job, source, job_dir, ["vocals"], skip_apple_tv=False)

    assert "--skip-apple-tv" not in captured["cmd"]


def test_serves_done_job_upmix_output(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.upmix.JOBS_DIR", tmp_path)
    job = Job(id="abcdefabcdef", status="done")
    _jobs[job.id] = job
    out = tmp_path / job.id / "upmix"
    out.mkdir(parents=True)
    path = out / "song_upmix5.1.flac"
    path.write_bytes(b"FLAC1234")

    r = client.get(f"/api/jobs/{job.id}/upmix/{path.name}")

    assert r.status_code == 200
    assert r.content == b"FLAC1234"
    assert r.headers["content-type"].startswith("audio/flac")


def test_upmix_output_rejects_bad_filename(client):
    job = Job(id="abcdefabcdef", status="done")
    _jobs[job.id] = job

    r = client.get(f"/api/jobs/{job.id}/upmix/../../secret.flac")

    assert r.status_code == 404


def test_tuned_upmix_render_endpoint_runs_with_ui_args(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.upmix.JOBS_DIR", tmp_path)
    job = Job(id="abcdefabcdef", status="done", title="Song")
    _jobs[job.id] = job
    stems_dir = tmp_path / job.id / "stems"
    stems_dir.mkdir(parents=True)
    (stems_dir / "vocals.wav").write_bytes(b"RIFF")
    (stems_dir / "drums.wav").write_bytes(b"RIFF")
    captured = {}

    def fake_run_upmix(
        job,
        source,
        job_dir,
        found_stems,
        *,
        extra_args=None,
        clear_output=False,
        skip_apple_tv=True,
    ):
        captured["job_status"] = job.status
        captured["source"] = source
        captured["job_dir"] = job_dir
        captured["found_stems"] = found_stems
        captured["extra_args"] = extra_args or []
        captured["clear_output"] = clear_output
        captured["skip_apple_tv"] = skip_apple_tv
        return [
            {
                "kind": "surround",
                "label": "5.1 FLAC",
                "filename": "song.flac",
                "url": f"/api/jobs/{job.id}/upmix/song.flac",
                "size_bytes": 4,
            }
        ]

    monkeypatch.setattr("app.api.upmix.run_upmix", fake_run_upmix)

    r = client.post(
        f"/api/jobs/{job.id}/upmix/render",
        json={
            "profile": "front51",
            "lfe_mode": "normal",
            "vocal_gain_db": 1.5,
            "space_bed_strength": 0.7,
            "temporal_mastering_strength": 0.8,
            "stem_aware_eq_strength": 0.6,
            "master_gain": 0.9,
        },
    )

    assert r.status_code == 200
    assert captured["job_status"] == "upmixing"
    assert captured["found_stems"] == ["vocals", "drums"]
    assert captured["clear_output"] is True
    assert captured["skip_apple_tv"] is False
    args = captured["extra_args"]
    assert args[args.index("--mix-profile") + 1] == "front51"
    assert args[args.index("--lfe-mode") + 1] == "normal"
    assert args[args.index("--vocal-gain-db") + 1] == "1.5"
    assert args[args.index("--space-bed") + 1] == "auto"
    assert args[args.index("--master-gain") + 1] == "0.9"
    assert "--skip-apple-tv" in args
    assert job.status == "done"
    assert job.upmix_error is None
    assert job.upmix_outputs[0]["filename"] == "song.flac"


def test_upmix_render_args_resolve_lfe_auto_from_stems():
    payload = UpmixRenderRequest(lfe_mode="auto")

    args = _upmix_render_args(payload, ["bass", "drums"])
    assert args[args.index("--lfe-mode") + 1] == "normal"

    args = _upmix_render_args(payload, ["bass"])
    assert args[args.index("--lfe-mode") + 1] == "light"

    args = _upmix_render_args(payload, ["vocals"])
    assert args[args.index("--lfe-mode") + 1] == "off"


def test_upmix_render_args_default_to_auto_stack():
    args = _upmix_render_args(UpmixRenderRequest(), ["bass", "drums"])

    assert _arg_value(args, "--lfe-mode") == "normal"
    assert _arg_value(args, "--space-bed") == "auto"
    assert _arg_value(args, "--space-bed-strength") == "0.85"
    assert _arg_value(args, "--analysis-backend") == "auto"
    assert _arg_value(args, "--focus-stem-mode") == "auto"
    assert _arg_value(args, "--temporal-remaster") == "auto"
    assert _arg_value(args, "--brickwall-recovery") == "auto"
    assert _arg_value(args, "--stem-de-limiter") == "auto"
    assert _arg_value(args, "--stem-de-limiter-device") == "auto"
    assert _arg_value(args, "--stem-de-limiter-checkpoint").startswith("weights/")
    assert _arg_value(args, "--post-render-correct") == "auto"


def test_upmix_render_args_include_whitelisted_advanced_options():
    payload = UpmixRenderRequest(
        advanced={
            "temporal_mastering": "off",
            "temporal_feedback_preflight": "off",
            "temporal_feedback_max_passes": 2,
            "stem_de_limiter_keep_stems": True,
            "skip_stereo": True,
            "stereo_loudness_i": -16.5,
            "surround_render_backend": "ffmpeg-flac",
            "loudnorm_measure_backend": "ffmpeg",
            "presence_priority": "other,guitar,piano",
        }
    )

    args = _upmix_render_args(payload, ["vocals", "drums"])

    assert args[args.index("--temporal-mastering") + 1] == "off"
    assert args[args.index("--temporal-feedback-preflight") + 1] == "off"
    assert args[args.index("--temporal-feedback-max-passes") + 1] == "2"
    assert "--stem-de-limiter-keep-stems" in args
    assert "--skip-stereo" in args
    assert args[args.index("--stereo-loudness-i") + 1] == "-16.5"
    assert args[args.index("--surround-render-backend") + 1] == "ffmpeg-flac"
    assert args[args.index("--loudnorm-measure-backend") + 1] == "ffmpeg"
    assert args[args.index("--presence-priority") + 1] == "other,guitar,piano"


def test_upmix_render_args_add_default_stem_sgi_checkpoint_when_enabled():
    payload = UpmixRenderRequest(advanced={"stem_de_limiter": "remaster"})

    args = _upmix_render_args(payload, ["vocals", "drums"])

    assert args[args.index("--stem-de-limiter") + 1] == "remaster"
    assert args[args.index("--stem-de-limiter-checkpoint") + 1].startswith("weights/")
    assert args[args.index("--stem-de-limiter-checkpoint") + 1].endswith(
        ("de_limiter_best.safetensors", "de_limiter_best.ckpt")
    )


def test_upmix_render_args_resolves_bare_stem_sgi_checkpoint_to_weights():
    payload = UpmixRenderRequest(
        advanced={
            "stem_de_limiter": "repair",
            "stem_de_limiter_checkpoint": "custom_model.ckpt",
        }
    )

    args = _upmix_render_args(payload, ["vocals", "drums"])

    assert args[args.index("--stem-de-limiter-checkpoint") + 1] == "weights/custom_model.ckpt"


def test_upmix_render_args_does_not_send_stem_sgi_checkpoint_when_off():
    payload = UpmixRenderRequest(advanced={"stem_de_limiter": "off"})

    args = _upmix_render_args(payload, ["vocals", "drums"])

    assert "--stem-de-limiter-checkpoint" not in args


def test_upmix_render_args_can_enable_apple_tv_explicitly():
    payload = UpmixRenderRequest(advanced={"skip_apple_tv": False})

    args = _upmix_render_args(payload, ["vocals", "drums"])

    assert "--skip-apple-tv" not in args


def test_upmix_render_args_reject_unknown_advanced_option():
    payload = UpmixRenderRequest(advanced={"not_a_real_option": "on"})

    with pytest.raises(ValueError, match="unknown upmix option"):
        _upmix_render_args(payload, ["vocals"])


def test_tuned_upmix_render_requires_done_job(client):
    job = Job(id="abcdefabcdef", status="separating")
    _jobs[job.id] = job

    r = client.post(f"/api/jobs/{job.id}/upmix/render", json={})

    assert r.status_code == 409
