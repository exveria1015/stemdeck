from __future__ import annotations

from pathlib import Path

from app.core.models import Job
from app.pipeline import collect as collect_mod
from app.pipeline.collect import collect, make_original_track, make_selected_mix


def test_collect_moves_residual_allocator_stems_and_cleans_output(tmp_path: Path):
    job_dir = tmp_path / "job"
    stems_root = job_dir / "residual_allocator"
    stems_root.mkdir(parents=True)
    (stems_root / "vocals.wav").write_bytes(b"RIFF")

    found = collect(Job(id="abcdefabcdef"), stems_root, job_dir)

    assert found == ["vocals"]
    assert (job_dir / "stems" / "vocals.wav").read_bytes() == b"RIFF"
    assert not stems_root.exists()


def test_make_original_track_prefers_native_wav_mix(monkeypatch, tmp_path: Path):
    job = Job(id="abcdefabcdef", selected_stems=["vocals"])
    job_dir = tmp_path / "job"
    stems_dir = job_dir / "stems"
    stems_dir.mkdir(parents=True)
    for name in ("drums", "bass"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")

    calls: list[tuple[list[Path], Path]] = []

    def fake_native(_job: Job, inputs: list[Path], out: Path) -> bool:
        calls.append((inputs, out))
        out.write_bytes(b"native")
        return True

    def fail_ffmpeg(*_args, **_kwargs) -> bool:
        raise AssertionError("ffmpeg should not run")

    monkeypatch.setattr(collect_mod, "_run_native_wav_mix", fake_native)
    monkeypatch.setattr(collect_mod, "_run_ffmpeg", fail_ffmpeg)

    result = make_original_track(job, job_dir, stems_dir)

    assert result == stems_dir / "original.wav"
    assert result.read_bytes() == b"native"
    assert [path.name for path in calls[0][0]] == ["drums.wav", "bass.wav"]


def test_make_selected_mix_prefers_native_wav_mix(monkeypatch, tmp_path: Path):
    job = Job(id="abcdefabcdef", selected_stems=["vocals", "drums"])
    stems_dir = tmp_path / "stems"
    stems_dir.mkdir()
    for name in ("vocals", "drums", "bass"):
        (stems_dir / f"{name}.wav").write_bytes(b"RIFF")

    def fake_native(_job: Job, inputs: list[Path], out: Path) -> bool:
        out.write_bytes(b"native")
        return True

    def fail_ffmpeg(*_args, **_kwargs) -> bool:
        raise AssertionError("ffmpeg should not run")

    monkeypatch.setattr(collect_mod, "_run_native_wav_mix", fake_native)
    monkeypatch.setattr(collect_mod, "_run_ffmpeg", fail_ffmpeg)

    result = make_selected_mix(job, stems_dir, ["vocals", "drums", "bass"])

    assert result == stems_dir / "mix.wav"
    assert result.read_bytes() == b"native"
