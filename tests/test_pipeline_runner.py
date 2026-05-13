from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.models import Job, JobCancelled
from app.pipeline.runner import _run_common, run_pipeline


@pytest.mark.asyncio
async def test_pipeline_transitions_to_error_on_stage_failure(tmp_path: Path):
    job = Job(id="abcdefabcdef")

    def boom(*args, **kwargs):
        raise RuntimeError("download blew up")

    with patch("app.pipeline.runner._run_blocking", side_effect=boom):
        await run_pipeline(job, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert job.status == "error"
    assert job.error  # generic message returned to client; detail is in server logs


@pytest.mark.asyncio
async def test_pipeline_marks_done_on_success(tmp_path: Path):
    job = Job(id="abcdefabcdee")

    with patch("app.pipeline.runner._run_blocking", return_value=None):
        await run_pipeline(job, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert job.status == "done"
    assert job.progress == 1.0


@pytest.mark.asyncio
async def test_pipeline_handles_jobcancelled(tmp_path: Path):
    job = Job(id="abcdefabcdec")
    job.cancel_requested = True

    def cancel(*args, **kwargs):
        raise JobCancelled()

    with patch("app.pipeline.runner._run_blocking", side_effect=cancel):
        await run_pipeline(job, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert job.status == "cancelled"
    # Partial job dir is removed.
    assert not (tmp_path / job.id).exists()


@pytest.mark.asyncio
async def test_pipeline_handles_wrapped_cancel(tmp_path: Path):
    """yt-dlp wraps hook exceptions in DownloadError; the runner must still
    treat it as a cancel when the flag is set."""
    job = Job(id="abcdefabcdeb")
    job.cancel_requested = True

    def wrapped(*args, **kwargs):
        raise RuntimeError("yt-dlp DownloadError wrapping JobCancelled")

    with patch("app.pipeline.runner._run_blocking", side_effect=wrapped):
        await run_pipeline(job, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", tmp_path)

    assert job.status == "cancelled"


@pytest.mark.asyncio
async def test_pipeline_recovers_from_mkdir_failure(tmp_path: Path):
    """If something pre-lock raises, the job must transition to error
    instead of staying stuck on `queued`."""
    job = Job(id="abcdefabcdea")
    bad_jobs_dir = tmp_path / "blocked"
    # Make jobs_dir a regular file so mkdir(parents=True) under it raises.
    bad_jobs_dir.write_bytes(b"not a directory")

    await run_pipeline(job, "https://www.youtube.com/watch?v=dQw4w9WgXcQ", bad_jobs_dir)

    assert job.status == "error"


def test_common_pipeline_runs_requested_upmix(tmp_path: Path):
    job = Job(id="abcdefabcdef", upmix_requested=True)
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    job_dir = tmp_path / job.id
    stems_root = job_dir / "separated"

    upmix_outputs = [
        {
            "kind": "surround",
            "label": "5.1 FLAC",
            "filename": "song.flac",
            "url": f"/api/jobs/{job.id}/upmix/song.flac",
            "size_bytes": 8,
        }
    ]

    with (
        patch("app.pipeline.runner.analyze", return_value=None),
        patch("app.pipeline.runner.separate", return_value=stems_root),
        patch("app.pipeline.runner.collect", return_value=["vocals", "drums"]),
        patch("app.pipeline.runner.make_original_track", return_value=None),
        patch("app.pipeline.runner.make_selected_mix", return_value=None),
        patch("app.pipeline.runner.cleanup_source", return_value=None) as cleanup,
        patch("app.pipeline.runner.run_upmix", return_value=upmix_outputs) as upmix,
    ):
        _run_common(job, source, job_dir)

    upmix.assert_called_once_with(job, source, job_dir, ["vocals", "drums"])
    cleanup.assert_called_once_with(job_dir)
    assert job.upmix_outputs == upmix_outputs
    assert job.upmix_error is None


def test_common_pipeline_keeps_stems_when_upmix_fails(tmp_path: Path):
    job = Job(id="abcdefabcdef", upmix_requested=True)
    source = tmp_path / "source.wav"
    source.write_bytes(b"RIFF")
    job_dir = tmp_path / job.id
    stems_root = job_dir / "separated"

    with (
        patch("app.pipeline.runner.analyze", return_value=None),
        patch("app.pipeline.runner.separate", return_value=stems_root),
        patch("app.pipeline.runner.collect", return_value=["vocals"]),
        patch("app.pipeline.runner.make_original_track", return_value=None),
        patch("app.pipeline.runner.make_selected_mix", return_value=None),
        patch("app.pipeline.runner.cleanup_source", return_value=None),
        patch("app.pipeline.runner.run_upmix", side_effect=RuntimeError("upmix failed")),
    ):
        _run_common(job, source, job_dir)

    assert job.stems == [{"name": "vocals", "url": f"/api/jobs/{job.id}/stems/vocals.wav"}]
    assert job.upmix_outputs == []
    assert "upmix failed" in str(job.upmix_error)
