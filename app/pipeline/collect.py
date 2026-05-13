from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from app.core.config import JOB_TTL_SECONDS, ROOT, STEM_NAMES, ffmpeg_executable
from app.core.models import Job
from app.core.registry import all_jobs as registry_all
from app.core.registry import persist as registry_persist
from app.core.registry import remove as registry_remove
from app.core.registry import set_proc

logger = logging.getLogger("stemdeck.collect")
_NATIVE_AUDIO_BIN: Path | None | bool = None


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("failed to remove %s", path, exc_info=True)


def _run_registered_process(job: Job, cmd: list[str], *, label: str, timeout: int = 300) -> bool:
    """Run a helper command, registering the subprocess with the job
    registry so POST /api/jobs/{id}/cancel can terminate it. Returns
    True on success, False on failure or external termination.

    Without registering the proc, an in-flight audio mix would block
    cancellation for up to its 300s timeout -- the cancel flag is set
    but the runner can't see it until subprocess.run returns. With
    set_proc, the cancel API can call proc.terminate() directly and
    communicate() returns within ~1s with a non-zero returncode."""
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    set_proc(job.id, proc)
    try:
        try:
            _, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            logger.warning("%s timed out for job %s", label, job.id)
            return False
        if proc.returncode != 0:
            tail = (stderr or b"").decode(errors="replace").splitlines()[-3:]
            logger.warning(
                "%s exit %s for job %s: %s",
                label,
                proc.returncode,
                job.id,
                " | ".join(tail) or "(no stderr)",
            )
            return False
        return True
    finally:
        set_proc(job.id, None)


def _run_ffmpeg(job: Job, cmd: list[str]) -> bool:
    return _run_registered_process(job, cmd, label="ffmpeg")


def _native_audio_executable() -> Path | None:
    global _NATIVE_AUDIO_BIN
    if _NATIVE_AUDIO_BIN is False:
        return None
    if isinstance(_NATIVE_AUDIO_BIN, Path):
        return _NATIVE_AUDIO_BIN

    env_path = os.environ.get("STEMDECK_NATIVE_AUDIO_BIN", "").strip()
    exe_name = "stemdeck-native-audio.exe" if sys.platform.startswith("win") else "stemdeck-native-audio"
    candidates = []
    if env_path:
        candidates.append(Path(env_path).expanduser())
    manifest = ROOT / "native_audio" / "Cargo.toml"
    expected = ROOT / "native_audio" / "target" / "release" / exe_name
    candidates.append(expected)
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            _NATIVE_AUDIO_BIN = path.resolve()
            return _NATIVE_AUDIO_BIN

    if env_path or not manifest.is_file() or shutil.which("cargo") is None:
        _NATIVE_AUDIO_BIN = False
        return None

    started = time.perf_counter()
    proc = subprocess.run(
        ["cargo", "build", "--release", "--quiet", "--manifest-path", str(manifest)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        logger.warning(
            "native audio helper build failed: %s",
            " | ".join(detail) or f"exit {proc.returncode}",
        )
        _NATIVE_AUDIO_BIN = False
        return None
    logger.info("native audio helper built in %.2fs", time.perf_counter() - started)
    if expected.is_file() and os.access(expected, os.X_OK):
        _NATIVE_AUDIO_BIN = expected.resolve()
        return _NATIVE_AUDIO_BIN
    _NATIVE_AUDIO_BIN = False
    return None


def _run_native_wav_mix(job: Job, inputs: list[Path], out: Path) -> bool:
    binary = _native_audio_executable()
    if binary is None:
        return False
    cmd = [str(binary), "mix", "--output", str(out)]
    for path in inputs:
        cmd.extend(["--input", str(path)])
    ok = _run_registered_process(job, cmd, label="native audio mix")
    if not ok:
        out.unlink(missing_ok=True)
    return ok


_TERMINAL = frozenset(("done", "error", "cancelled"))


def collect(job: Job, stems_root: Path, job_dir: Path) -> list[str]:
    """Move separator-emitted stems into the job's stems/ dir and clean up
    the separator intermediate dir. Does NOT delete the source download --
    cleanup_source() is called by the runner after any post-processing
    that needs to re-encode the source (e.g. building original.wav)."""
    target_dir = job_dir / "stems"
    target_dir.mkdir(exist_ok=True)
    found: list[str] = []
    for name in STEM_NAMES:
        src = stems_root / f"{name}.wav"
        if src.exists():
            shutil.move(str(src), target_dir / f"{name}.wav")
            found.append(name)
    _cleanup_separator_output(stems_root, job_dir)
    if not found:
        raise RuntimeError("no stems produced by separator")
    return found


def _cleanup_separator_output(stems_root: Path, job_dir: Path) -> None:
    try:
        resolved_stems_root = stems_root.resolve()
        resolved_job_dir = job_dir.resolve()
    except OSError:
        return

    if resolved_stems_root.parent == resolved_job_dir:
        _rmtree(stems_root)
    elif resolved_stems_root.parent.parent == resolved_job_dir:
        _rmtree(stems_root.parent)


def cleanup_source(job_dir: Path) -> None:
    """Delete the source audio file. Called after collect AND after any
    post-processing that re-encodes the source (make_original_track).
    The source can be large, so getting rid of it is the bulk of disk reclaim
    per job; only the stems remain."""
    for f in job_dir.glob("source.*"):
        f.unlink(missing_ok=True)


def make_original_track(job: Job, job_dir: Path, stems_dir: Path) -> Path | None:
    """Build the "Original" backing track at stems/original.wav as the
    sum of the stems the user did NOT select. This way the studio can
    play (original + each selected stem) and reconstruct the full song
    without doubling the selected stems -- which is what would happen
    if "original" were the raw source download (drum hits in original
    + isolated drums.wav = drums at 2x amplitude).

    Skipped when the user kept all 6 stems (no complement to mix) or
    when none of the unselected stem WAVs are on disk."""
    unselected = [s for s in STEM_NAMES if s not in job.selected_stems]
    inputs = [stems_dir / f"{name}.wav" for name in unselected]
    inputs = [p for p in inputs if p.exists()]
    if not inputs:
        return None
    out = stems_dir / "original.wav"
    if _run_native_wav_mix(job, inputs, out):
        return out

    cmd: list[str] = [
        ffmpeg_executable(),
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    for p in inputs:
        cmd += ["-i", str(p)]
    if len(inputs) == 1:
        # Single complement stem -- copy as-is so we still produce a
        # canonical mix.wav-shaped output without invoking amix on a
        # 1-input graph (which is a no-op anyway).
        cmd += ["-c:a", "pcm_s16le", str(out)]
    else:
        filter_inputs = "".join(f"[{i}:a]" for i in range(len(inputs)))
        cmd += [
            "-filter_complex",
            f"{filter_inputs}amix=inputs={len(inputs)}:normalize=0",
            "-c:a",
            "pcm_s16le",
            str(out),
        ]
    return out if _run_ffmpeg(job, cmd) else None


def make_selected_mix(job: Job, stems_dir: Path, found: list[str]) -> Path | None:
    """If the user picked a strict subset of stems at submit time,
    sum those stems with ffmpeg amix into mix.wav. Returns the output
    path on success, or None when there's nothing to mix.

    Returns the existing single stem path (no ffmpeg) if exactly one
    stem was selected -- copying it to mix.wav would be 30 MB of
    duplicate data. The caller uses the returned path's name for the
    download URL, so a single-stem selection points the Download Mix
    button directly at the existing stem file.

    amix normalize=0 keeps stem amplitudes as-is. Separator outputs
    are expected to sum back to (close to) the original signal, so a 2-stem subset
    fits comfortably below 0 dBFS without normalization headroom."""
    selected = [s for s in job.selected_stems if s in found]
    if not selected or set(selected) == set(found):
        return None  # no subset -> "mix" would just be the original
    if len(selected) == 1:
        return stems_dir / f"{selected[0]}.wav"
    inputs = [stems_dir / f"{name}.wav" for name in selected]
    out = stems_dir / "mix.wav"
    if _run_native_wav_mix(job, inputs, out):
        return out

    cmd: list[str] = [
        ffmpeg_executable(),
        "-y",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    for p in inputs:
        cmd += ["-i", str(p)]
    filter_inputs = "".join(f"[{i}:a]" for i in range(len(inputs)))
    cmd += [
        "-filter_complex",
        f"{filter_inputs}amix=inputs={len(inputs)}:normalize=0",
        "-c:a",
        "pcm_s16le",
        str(out),
    ]
    return out if _run_ffmpeg(job, cmd) else None


def sweep_old_jobs(jobs_dir: Path) -> None:
    """Delete job directories older than JOB_TTL_SECONDS and remove them from
    the in-memory registry. Called once per new job submission so stale data
    doesn't accumulate on disk.

    Prefers Job.created_at over directory mtime (which can be touched by
    unrelated filesystem events), and never deletes the directory of an
    active (non-terminal) registered job even if its timestamp looks old.
    Falls back to mtime for orphan directories left over from a previous
    server run, since the registry is in-memory only."""
    cutoff = time.time() - JOB_TTL_SECONDS
    if not jobs_dir.is_dir():
        return
    jobs = registry_all()
    removed = False
    for d in jobs_dir.iterdir():
        if not d.is_dir():
            continue
        job = jobs.get(d.name)
        if job is not None:
            if job.status not in _TERMINAL:
                continue  # never delete an active job's working dir
            if job.created_at >= cutoff:
                continue
        elif d.stat().st_mtime >= cutoff:
            continue
        _rmtree(d)
        registry_remove(d.name)
        removed = True
    if removed:
        registry_persist(jobs_dir)
