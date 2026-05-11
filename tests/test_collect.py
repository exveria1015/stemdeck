from __future__ import annotations

from pathlib import Path

from app.core.models import Job
from app.pipeline.collect import collect


def test_collect_moves_residual_allocator_stems_and_cleans_output(tmp_path: Path):
    job_dir = tmp_path / "job"
    stems_root = job_dir / "residual_allocator"
    stems_root.mkdir(parents=True)
    (stems_root / "vocals.wav").write_bytes(b"RIFF")

    found = collect(Job(id="abcdefabcdef"), stems_root, job_dir)

    assert found == ["vocals"]
    assert (job_dir / "stems" / "vocals.wav").read_bytes() == b"RIFF"
    assert not stems_root.exists()
