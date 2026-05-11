from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.core.config import (
    JOB_ID_RE,
    JOBS_DIR,
    MAX_DURATION_SEC,
    MAX_PENDING_JOBS,
    STEM_NAMES,
    ffprobe_executable,
)
from app.core.models import Job
from app.core.registry import all_jobs as registry_all_jobs
from app.core.registry import get as registry_get
from app.core.registry import get_proc as registry_get_proc
from app.core.registry import persist as registry_persist
from app.core.registry import register as registry_register
from app.core.registry import remove as registry_remove
from app.pipeline import run_local_pipeline, run_pipeline
from app.pipeline.download import InvalidYouTubeURL, validate_youtube_url

router = APIRouter(tags=["jobs"])
logger = logging.getLogger("stemdeck.api")

_ALLOWED_EXTS = frozenset(
    (
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
    )
)
_ALLOWED_EXTS_LABEL = ", ".join(sorted(_ALLOWED_EXTS))
_WS_RE = re.compile(r"\s+")


def _sanitize_title(filename: str) -> str:
    """Strip extension, normalize whitespace, cap at 120 chars."""
    stem = Path(filename).stem
    return _WS_RE.sub(" ", stem).strip()[:120]


def _probe_duration(path: Path) -> float:
    """Run ffprobe to get file duration in seconds."""
    result = subprocess.run(
        [
            ffprobe_executable(),
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as e:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {result.stdout!r}") from e


def _check_file_size(file_obj: object) -> int:
    """Seek to end, return size, rewind. Operates on the SpooledTemporaryFile
    backing a starlette UploadFile — synchronous, suitable for to_thread."""
    file_obj.seek(0, 2)  # type: ignore[union-attr]
    size = file_obj.tell()  # type: ignore[union-attr]
    file_obj.seek(0)  # type: ignore[union-attr]
    return size


def _copy_to_dest(src_file: object, dest: Path) -> None:
    """Copy SpooledTemporaryFile contents to dest. Synchronous, run in thread."""
    with dest.open("wb") as out:
        shutil.copyfileobj(src_file, out)  # type: ignore[arg-type]


def _rmtree_job(job_id: str) -> None:
    job_dir = JOBS_DIR / job_id
    if not job_dir.is_dir():
        return
    try:
        shutil.rmtree(job_dir)
    except Exception:
        logger.warning("failed to remove job dir %s", job_dir, exc_info=True)


def _task_error_cb(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("pipeline task raised unhandled exception", exc_info=exc)


class JobRequest(BaseModel):
    url: str
    # Subset of stems to include in the post-processing "selected mix"
    # audio file. None = all 6 (no extra mix produced; would equal the
    # original). Unknown stem names are dropped silently rather than
    # rejected, so a future model with extra stems doesn't break older
    # clients pinning the old set.
    stems: list[str] | None = None


@router.post("")
async def create_job(request: Request) -> dict[str, str]:
    ct = request.headers.get("content-type", "")
    if "multipart/form-data" in ct:
        return await _create_local_job(request)
    return await _create_youtube_job(request)


async def _create_youtube_job(request: Request) -> dict[str, str]:
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid JSON: {e}") from e
    try:
        payload = JobRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        url = validate_youtube_url(payload.url)
    except InvalidYouTubeURL as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    pending = sum(1 for j in registry_all_jobs().values() if j.status == "queued")
    if pending >= MAX_PENDING_JOBS:
        raise HTTPException(status_code=503, detail="Server busy, please try again later")

    selected = [s for s in payload.stems if s in STEM_NAMES] if payload.stems else list(STEM_NAMES)
    if not selected:
        selected = list(STEM_NAMES)

    job = registry_register(Job(id=uuid.uuid4().hex[:12], selected_stems=selected, source_url=url))
    task = asyncio.create_task(run_pipeline(job, url, JOBS_DIR))
    task.add_done_callback(_task_error_cb)
    return {"job_id": job.id}


async def _create_local_job(request: Request) -> dict[str, str]:
    # Reject busy server BEFORE touching disk so no orphan dir is created.
    pending = sum(1 for j in registry_all_jobs().values() if j.status == "queued")
    if pending >= MAX_PENDING_JOBS:
        raise HTTPException(status_code=503, detail="Server busy, please try again later")

    form = await request.form()
    upload = form.get("file")
    stems_raw = form.get("stems", "[]")

    if upload is None or not hasattr(upload, "filename"):
        raise HTTPException(status_code=422, detail="No file provided")

    filename: str = getattr(upload, "filename", "") or ""
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}': accepted extensions are {_ALLOWED_EXTS_LABEL}",
        )

    # Validate stems list from form field
    try:
        stems_list = json.loads(stems_raw)
        if not isinstance(stems_list, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        stems_list = []
    selected = [s for s in stems_list if s in STEM_NAMES] or list(STEM_NAMES)

    # Check actual file size (SpooledTemporaryFile is already buffered at this
    # point; seek/tell are fast and don't re-read the body).
    file_obj = upload.file  # type: ignore[union-attr]
    file_size = await asyncio.to_thread(_check_file_size, file_obj)
    if file_size == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty")

    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    source_path = job_dir / f"source{ext}"

    job_dir.mkdir(parents=True, exist_ok=True)
    try:
        await asyncio.to_thread(_copy_to_dest, file_obj, source_path)

        # Duration check before registering the job so a violation leaves no
        # registered job and no leftover directory.
        try:
            duration = await asyncio.to_thread(_probe_duration, source_path)
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not read file duration: {e}") from e

        if duration > MAX_DURATION_SEC:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"File is {int(duration // 60)} min — limit is {MAX_DURATION_SEC // 60} min"
                ),
            )
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    title = _sanitize_title(filename)
    local_source_url = f"local:{title}"
    job = registry_register(
        Job(
            id=job_id,
            selected_stems=selected,
            title=title,
            duration_sec=duration,
            source_url=local_source_url,
        )
    )
    task = asyncio.create_task(run_local_pipeline(job, source_path, JOBS_DIR))
    task.add_done_callback(_task_error_cb)
    return {"job_id": job.id}


@router.get("")
def list_jobs() -> list[dict]:
    return [
        job.to_state()
        for job in sorted(registry_all_jobs().values(), key=lambda j: j.created_at)
        if job.status == "done"
    ]


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_state()


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status in ("done", "error", "cancelled"):
        return job.to_state()
    job.cancel_requested = True
    proc = registry_get_proc(job_id)
    if proc is not None and proc.poll() is None:
        proc.terminate()
    return job.to_state()


@router.delete("/{job_id}")
def delete_job(job_id: str) -> dict[str, str]:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status not in ("done", "error", "cancelled"):
        raise HTTPException(status_code=409, detail="job is still running")
    _rmtree_job(job_id)
    registry_remove(job_id)
    registry_persist(JOBS_DIR)
    return {"job_id": job_id, "status": "deleted"}
