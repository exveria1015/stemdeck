from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.core.config import (
    DEMUCS_DEVICE,
    DEMUCS_MODEL,
    RESIDUAL_ALLOCATOR_ARGS,
    RESIDUAL_ALLOCATOR_BASE_CHECKPOINT,
    RESIDUAL_ALLOCATOR_BASE_CONFIG,
    RESIDUAL_ALLOCATOR_CHECKPOINT,
    RESIDUAL_ALLOCATOR_CONFIG,
    RESIDUAL_ALLOCATOR_DEVICE,
    RESIDUAL_ALLOCATOR_DIR,
    RESIDUAL_ALLOCATOR_SCRIPT,
    SEPARATION_BACKEND,
    ffmpeg_executable,
)
from app.core.models import Job, JobCancelled
from app.core.registry import set_proc

logger = logging.getLogger("stemdeck.pipeline")

_PCT_RE = re.compile(r"(\d{1,3})%")
# Terminate the separator if it produces no output for this many seconds.
# GPU processing can be silent for minutes; 30 min covers legitimate pauses
# while still catching genuine hangs (GPU deadlock, OOM stall, etc.).
_SEPARATOR_STALL_TIMEOUT = 1800


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"{label} not found: {path}")


def _base_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
        env.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ModuleNotFoundError:
        pass
    return env


def _demucs_cmd(source: Path, job_dir: Path) -> tuple[list[str], Path, Path | None]:
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "-n",
        DEMUCS_MODEL,
        "-d",
        DEMUCS_DEVICE,
        "-o",
        str(job_dir),
        str(source),
    ]
    return cmd, job_dir / DEMUCS_MODEL / source.stem, None


def _prepare_residual_allocator_source(job: Job, source: Path, job_dir: Path) -> Path:
    """Residual Allocator uses librosa/soundfile for loading. Convert
    container formats from yt-dlp to a predictable WAV first."""
    if source.suffix.lower() == ".wav":
        return source

    from app.pipeline.download import _set

    dest = job_dir / "source.residual_allocator.wav"
    _set(job, stage="Preparing audio for separation...")
    cmd = [
        ffmpeg_executable(),
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ar",
        "44100",
        "-ac",
        "2",
        "-sample_fmt",
        "s16",
        "-y",
        str(dest),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    set_proc(job.id, proc)
    try:
        _, stderr = proc.communicate(timeout=900)
    except subprocess.TimeoutExpired as e:
        proc.kill()
        proc.communicate()
        raise RuntimeError("ffmpeg transcode for Residual Allocator timed out") from e
    finally:
        set_proc(job.id, None)

    if job.cancel_requested:
        raise JobCancelled()
    if proc.returncode != 0:
        detail = (stderr or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg transcode for Residual Allocator failed: {detail}")
    return dest


def _residual_allocator_cmd(job: Job, source: Path, job_dir: Path) -> tuple[list[str], Path, Path]:
    _require_file(RESIDUAL_ALLOCATOR_SCRIPT, "Residual Allocator script")
    _require_file(RESIDUAL_ALLOCATOR_BASE_CONFIG, "Residual Allocator base config")
    _require_file(RESIDUAL_ALLOCATOR_BASE_CHECKPOINT, "Residual Allocator base checkpoint")
    _require_file(RESIDUAL_ALLOCATOR_CONFIG, "Residual Allocator config")
    _require_file(RESIDUAL_ALLOCATOR_CHECKPOINT, "Residual Allocator checkpoint")

    prepared_source = _prepare_residual_allocator_source(job, source, job_dir)
    output_dir = job_dir / "residual_allocator"
    cmd = [
        sys.executable,
        str(RESIDUAL_ALLOCATOR_SCRIPT),
        "--input",
        str(prepared_source),
        "--output",
        str(output_dir),
        "--output-dir-is-stem-dir",
        "--device",
        RESIDUAL_ALLOCATOR_DEVICE,
        "--base-config",
        str(RESIDUAL_ALLOCATOR_BASE_CONFIG),
        "--base-checkpoint",
        str(RESIDUAL_ALLOCATOR_BASE_CHECKPOINT),
        "--allocator-config",
        str(RESIDUAL_ALLOCATOR_CONFIG),
        "--allocator-checkpoint",
        str(RESIDUAL_ALLOCATOR_CHECKPOINT),
        *RESIDUAL_ALLOCATOR_ARGS,
    ]
    return cmd, output_dir, RESIDUAL_ALLOCATOR_DIR


def _separator_cmd(job: Job, source: Path, job_dir: Path) -> tuple[list[str], Path, Path | None]:
    if SEPARATION_BACKEND == "demucs":
        return _demucs_cmd(source, job_dir)
    if SEPARATION_BACKEND == "residual_allocator":
        return _residual_allocator_cmd(job, source, job_dir)
    raise RuntimeError(
        "unsupported STEMDECK_SEPARATOR "
        f"{SEPARATION_BACKEND!r}; expected 'residual_allocator' or 'demucs'"
    )


def separate(job: Job, source: Path, job_dir: Path) -> Path:
    from app.pipeline.download import _set

    _set(
        job,
        status="separating",
        progress=0.0,
        stage=(
            "Separating stems with Residual Allocator..."
            if SEPARATION_BACKEND == "residual_allocator"
            else "Separating stems..."
        ),
    )

    cmd, stems_root, cwd = _separator_cmd(job, source, job_dir)
    env = _base_env()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
    )
    if proc.stdout is None:
        raise RuntimeError("separator subprocess has no output pipe")
    set_proc(job.id, proc)

    # tqdm uses \r to redraw -- read char-by-char and split on \r or \n.
    # Keep the last few non-progress lines so we can surface them if the separator
    # exits non-zero (otherwise the only signal would be a bare exit code).
    buf = ""
    tail: list[str] = []
    last_output: list[float] = [time.monotonic()]
    max_pct = 0

    def _watchdog() -> None:
        while proc.poll() is None:
            time.sleep(30)
            if time.monotonic() - last_output[0] > _SEPARATOR_STALL_TIMEOUT:
                logger.warning(
                    "separator stalled for %ss with no output, terminating job %s",
                    _SEPARATOR_STALL_TIMEOUT,
                    job.id,
                )
                proc.terminate()
                break

    wt = threading.Thread(target=_watchdog, daemon=True)
    wt.start()
    try:
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            last_output[0] = time.monotonic()
            if ch in ("\r", "\n"):
                line = buf.strip()
                buf = ""
                if not line:
                    continue
                m = _PCT_RE.search(line)
                if m:
                    pct = max(max_pct, min(100, int(m.group(1))))
                    max_pct = pct
                    _set(job, progress=pct / 100.0, stage=f"Separating {pct}%")
                else:
                    tail.append(line)
                    if len(tail) > 40:
                        tail.pop(0)
            else:
                buf += ch

        proc.wait()
    finally:
        set_proc(job.id, None)
        wt.join(timeout=5)

    # POST /cancel calls proc.terminate() directly, which causes the read loop
    # above to hit EOF and proc.wait() to return a nonzero status. Translate
    # that into JobCancelled before the generic separator-failed path.
    if job.cancel_requested:
        raise JobCancelled()
    if proc.returncode != 0:
        detail = "\n".join(tail[-15:]) if tail else "(no stderr captured)"
        logger.error("separator exited %s; tail:\n%s", proc.returncode, detail)
        last = tail[-1] if tail else f"exit status {proc.returncode}"
        raise RuntimeError(f"separator failed: {last}")

    if not stems_root.is_dir():
        raise RuntimeError(f"separator output not found at {stems_root}")
    return stems_root
