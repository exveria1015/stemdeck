from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import JOB_ID_RE, JOBS_DIR, STEM_NAMES, UPMIXER_DIR
from app.core.models import Job, JobCancelled
from app.core.registry import get as registry_get
from app.core.registry import persist as registry_persist
from app.pipeline.download import _set
from app.pipeline.upmix import run_upmix

router = APIRouter(tags=["upmix"])

_ALLOWED_SUFFIXES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp4": "video/mp4",
    ".json": "application/json",
}


class UpmixRenderRequest(BaseModel):
    profile: Literal["auto", "front51"] = "auto"
    lfe_mode: Literal["auto", "off", "light", "normal"] = "auto"
    vocal_gain_db: float = Field(0.0, ge=-6.0, le=6.0)
    space_bed_strength: float = Field(0.85, ge=0.0, le=1.0)
    temporal_mastering_strength: float = Field(0.65, ge=0.0, le=1.0)
    stem_aware_eq_strength: float = Field(0.75, ge=0.0, le=1.0)
    master_gain: float = Field(0.84, ge=0.25, le=1.2)
    auto_modes: list[
        Literal["space", "height", "focus", "vocal", "guard", "master", "lfe"]
    ] = Field(default_factory=list)
    advanced: dict[str, Any] = Field(default_factory=dict)


_STEM_CHOICES = {"vocals", "drums", "bass", "guitar", "piano", "other"}
_STEM_DE_LIMITER_CHECKPOINT_SUFFIXES = {".ckpt", ".safetensors"}
_STEM_DE_LIMITER_CHECKPOINT_CANDIDATES = (
    Path("de_limiter_best.safetensors"),
    Path("de_limiter_best.ckpt"),
    Path("de_limiter_sgi_stem_sections_2") / "de_limiter_best.safetensors",
    Path("de_limiter_sgi_stem_sections_2") / "de_limiter_best.ckpt",
)
_ADVANCED_OPTIONS: dict[str, dict[str, object]] = {
    "analysis_backend": {"arg": "--analysis-backend", "type": "choice", "choices": {"auto", "numpy", "ffmpeg"}},
    "silent_stem_filter": {"arg": "--silent-stem-filter", "type": "choice", "choices": {"auto", "off"}},
    "focus_stem_mode": {"arg": "--focus-stem-mode", "type": "choice", "choices": {"auto", "manual", "off"}},
    "focus_stem": {"arg": "--focus-stem", "type": "choice", "choices": _STEM_CHOICES},
    "focus_vocal_min_share": {"arg": "--focus-vocal-min-share", "type": "float"},
    "presence_priority": {"arg": "--presence-priority", "type": "text"},
    "priority_duck": {"arg": "--priority-duck", "type": "choice", "choices": {"auto", "off"}},
    "priority_duck_max_db": {"arg": "--priority-duck-max-db", "type": "float"},
    "temporal_duck_guard": {"arg": "--temporal-duck-guard", "type": "choice", "choices": {"auto", "off"}},
    "temporal_duck_guard_strength": {"arg": "--temporal-duck-guard-strength", "type": "float"},
    "temporal_remaster": {"arg": "--temporal-remaster", "type": "choice", "choices": {"auto", "off"}},
    "temporal_remaster_strength": {"arg": "--temporal-remaster-strength", "type": "float"},
    "temporal_remaster_step_sec": {"arg": "--temporal-remaster-step-sec", "type": "float"},
    "temporal_mastering": {"arg": "--temporal-mastering", "type": "choice", "choices": {"auto", "off"}},
    "temporal_mastering_profile": {"arg": "--temporal-mastering-profile", "type": "choice", "choices": {"auto", "conservative", "natural", "open"}},
    "temporal_mastering_report": {"arg": "--temporal-mastering-report", "type": "path"},
    "temporal_feedback": {"arg": "--temporal-feedback", "type": "choice", "choices": {"auto", "off"}},
    "temporal_feedback_preflight": {"arg": "--temporal-feedback-preflight", "type": "choice", "choices": {"auto", "off"}},
    "temporal_feedback_max_passes": {"arg": "--temporal-feedback-max-passes", "type": "int"},
    "temporal_feedback_strength": {"arg": "--temporal-feedback-strength", "type": "float"},
    "brickwall_recovery": {"arg": "--brickwall-recovery", "type": "choice", "choices": {"auto", "off", "conservative", "natural", "open"}},
    "brickwall_recovery_strength": {"arg": "--brickwall-recovery-strength", "type": "float"},
    "brickwall_recovery_keep_stems": {"arg": "--brickwall-recovery-keep-stems", "type": "bool"},
    "stem_de_limiter": {"arg": "--stem-de-limiter", "type": "choice", "choices": {"off", "auto", "safe", "remaster", "repair", "sections2"}},
    "stem_de_limiter_checkpoint": {"arg": "--stem-de-limiter-checkpoint", "type": "path"},
    "stem_de_limiter_mix": {"arg": "--stem-de-limiter-mix", "type": "float"},
    "stem_de_limiter_device": {"arg": "--stem-de-limiter-device", "type": "text"},
    "stem_de_limiter_chunk_sec": {"arg": "--stem-de-limiter-chunk-sec", "type": "float"},
    "stem_de_limiter_overlap_sec": {"arg": "--stem-de-limiter-overlap-sec", "type": "float"},
    "stem_de_limiter_peak_limit": {"arg": "--stem-de-limiter-peak-limit", "type": "float"},
    "stem_de_limiter_keep_stems": {"arg": "--stem-de-limiter-keep-stems", "type": "bool"},
    "space_bed_stems": {"arg": "--space-bed-stems", "type": "text"},
    "window_ms": {"arg": "--window-ms", "type": "float"},
    "hop_ms": {"arg": "--hop-ms", "type": "float"},
    "stem_gain_mode": {"arg": "--stem-gain-mode", "type": "choice", "choices": {"off", "spatial", "harman"}},
    "stem_gain_limit_db": {"arg": "--stem-gain-limit-db", "type": "float"},
    "stem_aware_eq": {"arg": "--stem-aware-eq", "type": "choice", "choices": {"auto", "off"}},
    "stem_aware_eq_limit_db": {"arg": "--stem-aware-eq-limit-db", "type": "float"},
    "bass_fold_support": {"arg": "--bass-fold-support", "type": "choice", "choices": {"auto", "off"}},
    "bass_fold_support_target_db": {"arg": "--bass-fold-support-target-db", "type": "float"},
    "bass_fold_support_max_coeff": {"arg": "--bass-fold-support-max-coeff", "type": "float"},
    "reference_match": {"arg": "--reference-match", "type": "choice", "choices": {"off", "original"}},
    "reference_guard": {"arg": "--reference-guard", "type": "choice", "choices": {"auto", "off", "original"}},
    "reference_match_limit_db": {"arg": "--reference-match-limit-db", "type": "float"},
    "reference_guard_tolerance_db": {"arg": "--reference-guard-tolerance-db", "type": "float"},
    "style_reference": {"arg": "--style-reference", "type": "path"},
    "style_reference_strength": {"arg": "--style-reference-strength", "type": "float"},
    "stem_presence_guard": {"arg": "--stem-presence-guard", "type": "choice", "choices": {"auto", "off"}},
    "stem_presence_guard_limit_db": {"arg": "--stem-presence-guard-limit-db", "type": "float"},
    "drum_dominance_guard": {"arg": "--drum-dominance-guard", "type": "choice", "choices": {"auto", "off"}},
    "drum_dominance_guard_limit_db": {"arg": "--drum-dominance-guard-limit-db", "type": "float"},
    "bass_drum_guard": {"arg": "--bass-drum-guard", "type": "choice", "choices": {"auto", "off"}},
    "bass_drum_guard_max_db": {"arg": "--bass-drum-guard-max-db", "type": "float"},
    "post_render_analysis": {"arg": "--post-render-analysis", "type": "choice", "choices": {"auto", "off"}},
    "post_render_correct": {"arg": "--post-render-correct", "type": "choice", "choices": {"auto", "off"}},
    "post_render_correct_iterations": {"arg": "--post-render-correct-iterations", "type": "int"},
    "post_render_correct_tolerance_db": {"arg": "--post-render-correct-tolerance-db", "type": "float"},
    "post_render_correct_limit_db": {"arg": "--post-render-correct-limit-db", "type": "float"},
    "post_render_correct_space_limit": {"arg": "--post-render-correct-space-limit", "type": "float"},
    "post_render_correct_eq": {"arg": "--post-render-correct-eq", "type": "choice", "choices": {"auto", "off"}},
    "post_render_correct_eq_limit_db": {"arg": "--post-render-correct-eq-limit-db", "type": "float"},
    "flac_compression_level": {"arg": "--flac-compression-level", "type": "int"},
    "skip_flac": {"arg": "--skip-flac", "type": "bool"},
    "skip_stereo": {"arg": "--skip-stereo", "type": "bool"},
    "stereo_normalize": {"arg": "--stereo-normalize", "type": "choice", "choices": {"loudnorm", "linear", "off"}},
    "stereo_loudness_i": {"arg": "--stereo-loudness-i", "type": "float"},
    "stereo_loudness_tp": {"arg": "--stereo-loudness-tp", "type": "float"},
    "stereo_loudness_lra": {"arg": "--stereo-loudness-lra", "type": "float"},
    "stereo_lfe_fold_gain": {"arg": "--stereo-lfe-fold-gain", "type": "float"},
    "stereo_envelope_match": {"arg": "--stereo-envelope-match", "type": "choice", "choices": {"auto", "off"}},
    "stereo_envelope_window_sec": {"arg": "--stereo-envelope-window-sec", "type": "float"},
    "stereo_envelope_hop_sec": {"arg": "--stereo-envelope-hop-sec", "type": "float"},
    "stereo_envelope_smooth_sec": {"arg": "--stereo-envelope-smooth-sec", "type": "float"},
    "stereo_envelope_strength": {"arg": "--stereo-envelope-strength", "type": "float"},
    "stereo_envelope_max_boost_db": {"arg": "--stereo-envelope-max-boost-db", "type": "float"},
    "stereo_envelope_max_cut_db": {"arg": "--stereo-envelope-max-cut-db", "type": "float"},
    "de_limiter_checkpoint": {"arg": "--de-limiter-checkpoint", "type": "path"},
    "de_limiter_mix": {"arg": "--de-limiter-mix", "type": "float"},
    "de_limiter_device": {"arg": "--de-limiter-device", "type": "text"},
    "de_limiter_chunk_sec": {"arg": "--de-limiter-chunk-sec", "type": "float"},
    "de_limiter_overlap_sec": {"arg": "--de-limiter-overlap-sec", "type": "float"},
    "de_limiter_peak_limit": {"arg": "--de-limiter-peak-limit", "type": "float"},
    "surround_normalize": {"arg": "--surround-normalize", "type": "choice", "choices": {"loudnorm", "off"}},
    "surround_loudness_i": {"arg": "--surround-loudness-i", "type": "float"},
    "surround_loudness_tp": {"arg": "--surround-loudness-tp", "type": "float"},
    "surround_loudness_lra": {"arg": "--surround-loudness-lra", "type": "float"},
    "surround_render_backend": {"arg": "--surround-render-backend", "type": "choice", "choices": {"native-wav", "ffmpeg-flac"}},
    "loudnorm_measure_backend": {"arg": "--loudnorm-measure-backend", "type": "choice", "choices": {"native", "ffmpeg"}},
    "apple_tv_bitrate": {"arg": "--apple-tv-bitrate", "type": "text"},
    "skip_apple_tv": {"arg": "--skip-apple-tv", "type": "bool"},
    "skip_bed": {"arg": "--skip-bed", "type": "bool"},
}
_DEFAULT_AUTO_ADVANCED: dict[str, str] = {
    "analysis_backend": "auto",
    "silent_stem_filter": "auto",
    "focus_stem_mode": "auto",
    "priority_duck": "auto",
    "temporal_duck_guard": "auto",
    "temporal_remaster": "auto",
    "temporal_mastering": "auto",
    "temporal_mastering_profile": "auto",
    "temporal_feedback": "auto",
    "temporal_feedback_preflight": "auto",
    "brickwall_recovery": "auto",
    "stem_de_limiter": "auto",
    "stem_de_limiter_device": "auto",
    "stem_aware_eq": "auto",
    "bass_fold_support": "auto",
    "reference_guard": "auto",
    "stem_presence_guard": "auto",
    "drum_dominance_guard": "auto",
    "bass_drum_guard": "auto",
    "post_render_analysis": "auto",
    "post_render_correct": "auto",
    "post_render_correct_eq": "auto",
    "stereo_envelope_match": "auto",
}


def _fmt_float(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _default_stem_de_limiter_checkpoint() -> str:
    weights_dir = UPMIXER_DIR / "weights"
    for candidate in _STEM_DE_LIMITER_CHECKPOINT_CANDIDATES:
        if (weights_dir / candidate).is_file():
            return (Path("weights") / candidate).as_posix()
    return (Path("weights") / _STEM_DE_LIMITER_CHECKPOINT_CANDIDATES[0]).as_posix()


def _normalize_stem_de_limiter_checkpoint(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute():
        return text
    if path.suffix.lower() in _STEM_DE_LIMITER_CHECKPOINT_SUFFIXES and (
        not path.parts or path.parts[0] != "weights"
    ):
        path = Path("weights") / path
    return path.as_posix()


def _advanced_value_args(key: str, value: object) -> list[str]:
    spec = _ADVANCED_OPTIONS.get(key)
    if spec is None:
        raise ValueError(f"unknown upmix option: {key}")
    arg = str(spec["arg"])
    kind = str(spec["type"])

    if value is None or value == "":
        return []
    if key == "stem_de_limiter_checkpoint":
        text = _normalize_stem_de_limiter_checkpoint(value)
        return [arg, text] if text else []
    if kind == "bool":
        return [arg] if bool(value) else []
    if kind == "choice":
        text = str(value).strip()
        choices = spec.get("choices")
        if isinstance(choices, set) and text not in choices:
            raise ValueError(f"invalid value for {key}: {text}")
        return [arg, text]
    if kind == "int":
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid integer for {key}") from exc
        return [arg, str(parsed)]
    if kind == "float":
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid number for {key}") from exc
        if not math.isfinite(parsed):
            raise ValueError(f"invalid number for {key}")
        return [arg, _fmt_float(parsed)]
    if kind in {"text", "path"}:
        text = str(value).strip()
        return [arg, text] if text else []
    raise ValueError(f"unsupported upmix option type for {key}")


def _upmix_advanced_args(advanced: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, value in advanced.items():
        args.extend(_advanced_value_args(key, value))
    return args


def _resolve_lfe_mode(payload: UpmixRenderRequest, found_stems: list[str] | None) -> str:
    if payload.lfe_mode != "auto":
        return payload.lfe_mode
    stems = set(found_stems or [])
    if {"bass", "drums"}.issubset(stems):
        return "normal"
    if stems.intersection({"bass", "drums"}):
        return "light"
    return "off"


def _upmix_render_args(
    payload: UpmixRenderRequest,
    found_stems: list[str] | None = None,
) -> list[str]:
    space_bed = "auto" if payload.space_bed_strength > 0.001 else "off"
    advanced = {**_DEFAULT_AUTO_ADVANCED, **payload.advanced}
    skip_apple_tv = advanced.pop("skip_apple_tv", None)
    stem_sgi_mode = str(advanced.get("stem_de_limiter", "off")).strip()
    stem_sgi_checkpoint = _normalize_stem_de_limiter_checkpoint(
        advanced.get("stem_de_limiter_checkpoint")
    )
    if stem_sgi_checkpoint:
        advanced["stem_de_limiter_checkpoint"] = stem_sgi_checkpoint
    elif stem_sgi_mode and stem_sgi_mode not in {"off", "safe"}:
        advanced["stem_de_limiter_checkpoint"] = _default_stem_de_limiter_checkpoint()
    args = [
        "--mix-profile",
        payload.profile,
        "--lfe-mode",
        _resolve_lfe_mode(payload, found_stems),
        "--vocal-gain-db",
        _fmt_float(payload.vocal_gain_db),
        "--space-bed",
        space_bed,
        "--space-bed-strength",
        _fmt_float(payload.space_bed_strength),
        "--temporal-mastering-strength",
        _fmt_float(payload.temporal_mastering_strength),
        "--stem-aware-eq-strength",
        _fmt_float(payload.stem_aware_eq_strength),
        "--master-gain",
        _fmt_float(payload.master_gain),
    ]
    if skip_apple_tv is not False:
        args.append("--skip-apple-tv")
    args.extend(_upmix_advanced_args(advanced))
    return args


def _available_stems(job_dir: Path) -> list[str]:
    stems_dir = job_dir / "stems"
    return [name for name in STEM_NAMES if (stems_dir / f"{name}.wav").is_file()]


def _best_source_path(job_dir: Path) -> Path:
    for name in ("source.wav", "source.m4a", "source.mp3", "source.flac", "source.ogg"):
        path = job_dir / name
        if path.is_file():
            return path
    return job_dir / "source.wav"


def _write_upmix_metadata(job: Job, job_dir: Path, payload: UpmixRenderRequest) -> None:
    path = job_dir / "metadata.json"
    data: dict[str, object] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data.update(loaded)
        except (OSError, json.JSONDecodeError):
            data = {}
    data.update(
        {
            "upmix_requested": job.upmix_requested,
            "upmix_outputs": job.upmix_outputs,
            "upmix_error": job.upmix_error,
            "upmix_tuning": payload.model_dump(),
        }
    )
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _run_tuned_upmix_blocking(job: Job, job_dir: Path, payload: UpmixRenderRequest) -> None:
    previous_status = job.status
    found = _available_stems(job_dir)
    try:
        job.upmix_requested = True
        job.upmix_error = None
        job.upmix_outputs = []
        _set(job, status="upmixing", progress=0.0, stage="Rendering tuned upmix...")
        job.upmix_outputs = run_upmix(
            job,
            _best_source_path(job_dir),
            job_dir,
            found,
            extra_args=_upmix_render_args(payload, found),
            clear_output=True,
            skip_apple_tv=False,
        )
        job.upmix_error = None
        _set(job, status="done", progress=1.0, stage="Done")
    except JobCancelled:
        job.cancel_requested = False
        _set(job, status="done", progress=1.0, stage="Upmix cancelled")
    except Exception as exc:
        job.upmix_outputs = []
        job.upmix_error = str(exc)
        _set(job, status="done" if previous_status == "done" else previous_status, progress=1.0, stage="Upmix failed; stems are ready")
    finally:
        _write_upmix_metadata(job, job_dir, payload)
        registry_persist(JOBS_DIR)


async def _run_tuned_upmix(job: Job, job_dir: Path, payload: UpmixRenderRequest) -> None:
    from app.pipeline.runner import _pipeline_lock

    async with _pipeline_lock:
        await asyncio.to_thread(_run_tuned_upmix_blocking, job, job_dir, payload)


@router.post("/jobs/{job_id}/upmix/render")
def render_upmix_output(
    job_id: str,
    payload: UpmixRenderRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    job = registry_get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "upmixing":
        raise HTTPException(status_code=409, detail="upmix render already running")
    if job.status != "done":
        raise HTTPException(status_code=409, detail="job is not ready")
    job_dir = (JOBS_DIR / job_id).resolve()
    if not job_dir.is_dir() or not job_dir.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="job files not found")
    if not _available_stems(job_dir):
        raise HTTPException(status_code=404, detail="no stems available for upmix")
    try:
        _upmix_advanced_args(payload.advanced)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    job.upmix_requested = True
    job.upmix_error = None
    _set(job, status="upmixing", progress=0.0, stage="Queued tuned upmix")
    background_tasks.add_task(_run_tuned_upmix, job, job_dir, payload)
    return job.to_state()


@router.api_route("/jobs/{job_id}/upmix/{filename}", methods=["GET", "HEAD"])
def get_upmix_output(job_id: str, filename: str) -> FileResponse:
    if not JOB_ID_RE.match(job_id):
        raise HTTPException(status_code=404, detail="job not found")
    name = Path(filename).name
    if name != filename or not name:
        raise HTTPException(status_code=404, detail="upmix output not found")
    media_type = _ALLOWED_SUFFIXES.get(Path(name).suffix.lower())
    if media_type is None:
        raise HTTPException(status_code=404, detail="unsupported upmix output")
    job = registry_get(job_id)
    if job is None or job.status != "done":
        raise HTTPException(status_code=404, detail="job not ready")
    path = (JOBS_DIR / job_id / "upmix" / name).resolve()
    if not path.is_file() or not path.is_relative_to(JOBS_DIR.resolve()):
        raise HTTPException(status_code=404, detail="upmix output not found")
    return FileResponse(path, media_type=media_type, filename=name)
