from __future__ import annotations

from io import BytesIO
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core.registry import _jobs


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Each test gets a fresh in-memory registry."""
    _jobs.clear()
    yield
    _jobs.clear()


@pytest.fixture
def client():
    # Patch pipeline entrypoints so the test never spawns separators / yt-dlp.
    async def _noop_pipeline(job, url, jobs_dir):
        return None

    async def _noop_local_pipeline(job, source_path, jobs_dir):
        return None

    with (
        patch("app.api.jobs.run_pipeline", _noop_pipeline),
        patch("app.api.jobs.run_local_pipeline", _noop_local_pipeline),
        patch("app.api.jobs._probe_duration", return_value=120.0),
    ):
        from app.main import app

        with TestClient(app) as c:
            yield c


def test_post_rejects_invalid_url(client):
    r = client.post("/api/jobs", json={"url": "https://example.com/foo"})
    assert r.status_code == 422
    assert "unsupported host" in r.json()["detail"]


def test_post_rejects_empty_url(client):
    r = client.post("/api/jobs", json={"url": ""})
    assert r.status_code == 422


def test_post_accepts_youtube_url(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    assert r.status_code == 200
    assert "job_id" in r.json()
    assert len(r.json()["job_id"]) == 12


def test_post_accepts_youtube_upmix_flag(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ", "upmix": True})
    assert r.status_code == 200
    assert _jobs[r.json()["job_id"]].upmix_requested is True


def test_get_unknown_job_returns_404(client):
    r = client.get("/api/jobs/000000000000")
    assert r.status_code == 404


def test_cancel_unknown_job_returns_404(client):
    r = client.post("/api/jobs/000000000000/cancel")
    assert r.status_code == 404


def test_delete_running_job_rejected(client):
    # Submit a job; the patched pipeline is a noop so status stays "queued"
    # for the test's lifetime (no event loop tick advances it).
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    # Simulate a still-running job by leaving it on its default status.
    r = client.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 409


def test_cancel_sets_flag_and_returns_state(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert _jobs[job_id].cancel_requested is True


def test_cancel_after_done_is_idempotent(client):
    r = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ"})
    job_id = r.json()["job_id"]
    _jobs[job_id].status = "done"
    r = client.post(f"/api/jobs/{job_id}/cancel")
    assert r.status_code == 200
    assert _jobs[job_id].cancel_requested is False  # not flipped on terminal jobs


def test_post_accepts_flac_upload(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.jobs.JOBS_DIR", tmp_path)

    r = client.post(
        "/api/jobs",
        files={"file": ("song.flac", BytesIO(b"not really flac"), "audio/flac")},
        data={"stems": "[]"},
    )

    assert r.status_code == 200
    job_id = r.json()["job_id"]
    assert (tmp_path / job_id / "source.flac").is_file()


def test_post_accepts_upload_upmix_flag(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.jobs.JOBS_DIR", tmp_path)

    r = client.post(
        "/api/jobs",
        files={"file": ("song.flac", BytesIO(b"not really flac"), "audio/flac")},
        data={"stems": "[]", "upmix": "1"},
    )

    assert r.status_code == 200
    assert _jobs[r.json()["job_id"]].upmix_requested is True


def test_post_upload_has_no_100mb_cap(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.jobs.JOBS_DIR", tmp_path)
    monkeypatch.setattr("app.api.jobs._check_file_size", lambda file_obj: 101 * 1024 * 1024)

    r = client.post(
        "/api/jobs",
        files={"file": ("large.wav", BytesIO(b"RIFF"), "audio/wav")},
        data={"stems": "[]"},
    )

    assert r.status_code == 200


def test_post_rejects_unsupported_local_audio_extension(client, monkeypatch, tmp_path):
    monkeypatch.setattr("app.api.jobs.JOBS_DIR", tmp_path)

    r = client.post(
        "/api/jobs",
        files={"file": ("movie.mp4", BytesIO(b"data"), "video/mp4")},
        data={"stems": "[]"},
    )

    assert r.status_code == 422
    assert "accepted extensions" in r.json()["detail"]
