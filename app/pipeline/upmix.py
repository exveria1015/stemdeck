from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

from app.core.config import (
    STEM_NAMES,
    UPMIXER_ARGS,
    UPMIXER_DIR,
    UPMIXER_SCRIPT,
    ffmpeg_executable,
    ffprobe_executable,
)
from app.core.models import Job, JobCancelled
from app.core.registry import set_proc
from app.pipeline.download import _set

logger = logging.getLogger("stemdeck.upmix")

_OUTPUT_SUFFIXES = frozenset((".wav", ".flac", ".mp4"))
_PCT_RE = re.compile(r"(\d{1,3})%")
_UPMIX_STALL_TIMEOUT = 1800
_INITIAL_AUTO_VALUE_ARGS: tuple[tuple[str, str], ...] = (
    ("--mix-profile", "auto"),
    ("--analysis-backend", "auto"),
    ("--silent-stem-filter", "auto"),
    ("--focus-stem-mode", "auto"),
    ("--priority-duck", "auto"),
    ("--temporal-duck-guard", "auto"),
    ("--temporal-remaster", "auto"),
    ("--temporal-mastering", "auto"),
    ("--temporal-mastering-profile", "auto"),
    ("--temporal-feedback", "auto"),
    ("--temporal-feedback-preflight", "auto"),
    ("--brickwall-recovery", "auto"),
    ("--stem-de-limiter", "auto"),
    ("--stem-de-limiter-device", "auto"),
    ("--space-bed", "auto"),
    ("--space-bed-strength", "0.85"),
    ("--stem-aware-eq", "auto"),
    ("--bass-fold-support", "auto"),
    ("--reference-guard", "auto"),
    ("--stem-presence-guard", "auto"),
    ("--drum-dominance-guard", "auto"),
    ("--bass-drum-guard", "auto"),
    ("--post-render-analysis", "auto"),
    ("--post-render-correct", "auto"),
    ("--post-render-correct-eq", "auto"),
    ("--stereo-envelope-match", "auto"),
)


def upmix_output_url(job_id: str, filename: str) -> str:
    return f"/api/jobs/{job_id}/upmix/{quote(filename)}"


def _auto_lfe_mode(found_stems: list[str]) -> str:
    stems = set(found_stems)
    if {"bass", "drums"}.issubset(stems):
        return "normal"
    if stems.intersection({"bass", "drums"}):
        return "light"
    return "off"


def initial_auto_upmix_args(found_stems: list[str]) -> list[str]:
    args = ["--lfe-mode", _auto_lfe_mode(found_stems)]
    for key, value in _INITIAL_AUTO_VALUE_ARGS:
        args.extend([key, value])
    return args


def _output_kind(path: Path) -> tuple[str, str]:
    name = path.name.lower()
    if path.suffix.lower() == ".wav" and "upmix5.1" in name:
        return "surround", "5.1 WAV"
    if path.suffix.lower() == ".wav" and "_7.1.4_" in name:
        return "bed", "7.1.4 WAV"
    if path.suffix.lower() == ".flac" and "upmix5.1" in name:
        return "surround", "5.1 FLAC"
    if path.suffix.lower() == ".flac" and "stereo" in name:
        return "stereo", "Stereo FLAC"
    if path.suffix.lower() == ".mp4":
        return "apple_tv", "Apple TV MP4"
    return path.suffix.lower().lstrip(".") or "file", path.name


def discover_upmix_outputs(job_id: str, output_dir: Path) -> list[dict[str, object]]:
    if not output_dir.is_dir():
        return []
    outputs: list[dict[str, object]] = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file() or path.suffix.lower() not in _OUTPUT_SUFFIXES:
            continue
        kind, label = _output_kind(path)
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        outputs.append(
            {
                "kind": kind,
                "label": label,
                "filename": path.name,
                "url": upmix_output_url(job_id, path.name),
                "size_bytes": size,
            }
        )
    return outputs


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


def _require_upmixer() -> None:
    if not UPMIXER_SCRIPT.is_file():
        raise RuntimeError(f"Stems-Upmixer CLI not found: {UPMIXER_SCRIPT}")
    if not UPMIXER_DIR.is_dir():
        raise RuntimeError(f"Stems-Upmixer directory not found: {UPMIXER_DIR}")


def _stage_from_line(line: str, current_progress: float) -> tuple[float, str] | None:
    m = _PCT_RE.search(line)
    if m:
        pct = max(current_progress, min(0.95, int(m.group(1)) / 100.0))
        return pct, f"Rendering upmix {int(pct * 100)}%"

    lower = line.lower()
    markers = (
        ("analysis backend", 0.12, "Analyzing stem placement..."),
        ("rendering 7.1.4", 0.35, "Rendering 7.1.4 bed..."),
        ("post-render", 0.48, "Checking stereo fold-down..."),
        ("measuring shared 5.1", 0.58, "Measuring surround loudness..."),
        ("rendering 5.1", 0.68, "Rendering 5.1 FLAC..."),
        ("rendering stereo", 0.78, "Rendering stereo fold-down..."),
        ("rendering apple tv", 0.88, "Rendering Apple TV MP4..."),
        ("done.", 0.98, "Upmix complete"),
    )
    for token, progress, stage in markers:
        if token in lower:
            return max(current_progress, progress), stage
    return None


def run_upmix(
    job: Job,
    source: Path,
    job_dir: Path,
    found_stems: list[str],
    *,
    extra_args: list[str] | None = None,
    clear_output: bool = False,
    skip_apple_tv: bool = True,
) -> list[dict[str, object]]:
    _require_upmixer()
    stems_dir = job_dir / "stems"
    output_dir = job_dir / "upmix"
    stems = [name for name in STEM_NAMES if name in found_stems and (stems_dir / f"{name}.wav").is_file()]
    if not stems:
        raise RuntimeError("no stem WAV files are available for upmixing")
    if clear_output:
        shutil.rmtree(output_dir, ignore_errors=True)

    extra = list(extra_args or [])
    default_args = [
        *(initial_auto_upmix_args(found_stems) if extra_args is None else []),
        *UPMIXER_ARGS,
    ]
    if skip_apple_tv and "--skip-apple-tv" not in default_args and "--skip-apple-tv" not in extra:
        default_args.append("--skip-apple-tv")

    cmd = [
        sys.executable,
        str(UPMIXER_SCRIPT),
        "--stem-dir",
        str(stems_dir),
        "--output-dir",
        str(output_dir),
        "--stems",
        ",".join(stems),
        "--ffmpeg",
        ffmpeg_executable(),
        "--ffprobe",
        ffprobe_executable(),
        *default_args,
        *extra,
    ]
    if source.is_file():
        cmd.extend(["--input", str(source)])
    if job.title:
        cmd.extend(["--title", job.title])

    _set(job, status="upmixing", progress=0.0, stage="Rendering surround upmix...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=_base_env(),
        cwd=str(UPMIXER_DIR),
    )
    if proc.stdout is None:
        raise RuntimeError("Stems-Upmixer subprocess has no output pipe")
    set_proc(job.id, proc)

    tail: list[str] = []
    current_progress = 0.0
    last_output = time.monotonic()
    try:
        while True:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                if time.monotonic() - last_output > _UPMIX_STALL_TIMEOUT:
                    proc.terminate()
                    raise RuntimeError("Stems-Upmixer produced no output for too long")
                time.sleep(0.2)
                continue
            last_output = time.monotonic()
            text = line.strip()
            if not text:
                continue
            tail.append(text)
            if len(tail) > 60:
                tail.pop(0)
            update = _stage_from_line(text, current_progress)
            if update is not None:
                current_progress, stage = update
                _set(job, progress=current_progress, stage=stage)

        proc.wait()
    finally:
        set_proc(job.id, None)

    if job.cancel_requested:
        raise JobCancelled()
    if proc.returncode != 0:
        detail = tail[-1] if tail else f"exit status {proc.returncode}"
        logger.error("Stems-Upmixer exited %s; tail:\n%s", proc.returncode, "\n".join(tail[-15:]))
        raise RuntimeError(f"Stems-Upmixer failed: {detail}")

    outputs = discover_upmix_outputs(job.id, output_dir)
    if not outputs:
        raise RuntimeError(f"Stems-Upmixer completed but produced no media outputs in {output_dir}")
    _set(job, progress=1.0, stage="Upmix complete")
    return outputs
