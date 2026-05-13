import os
import re
import shlex
import sys
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    return Path(raw).expanduser().resolve() if raw else default


def _env_path_from(name: str, default: Path, base: Path) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return (base / default).resolve() if not default.is_absolute() else default.resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _env_args(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    try:
        return shlex.split(raw, posix=os.name != "nt")
    except ValueError:
        return []


def _detect_device(env_name: str) -> str:
    """Pick best available Torch device. Override with an env var
    containing 'cuda', 'mps', or 'cpu'. Apple Silicon silently falls back
    to CPU otherwise.

    torch.cuda.is_available() can be true even when the installed wheel cannot
    execute kernels for the visible GPU architecture. Smoke-test one simple CUDA
    kernel before selecting cuda so unsupported GPUs fall back cleanly.
    """
    forced = os.environ.get(env_name, "").strip().lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch

        if torch.cuda.is_available() and _cuda_kernel_smoke_ok(torch):
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _cuda_kernel_smoke_ok(torch_module: object) -> bool:
    try:
        x = torch_module.zeros((1, 1, 8), device="cuda")  # type: ignore[attr-defined]
        torch_module.nn.functional.pad(  # type: ignore[attr-defined]
            x,
            (0, 1),
            mode="constant",
            value=0,
        )
        torch_module.cuda.synchronize()  # type: ignore[attr-defined]
        return True
    except Exception:
        return False


ROOT = Path(__file__).resolve().parent.parent.parent


def _default_residual_allocator_dir() -> Path:
    vendored = (ROOT / "vendor" / "Residual-Allocator").resolve()
    if vendored.is_dir():
        return vendored
    return (ROOT / "../Residual-Allocator").resolve()


STATIC_DIR = ROOT / "static"
STEM_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "guitar", "piano", "other")
JOB_ID_RE = re.compile(r"^[a-f0-9]{12}$")

# Runtime knobs -- env-backed so Docker / desktop packaging / local dev can
# tune without a code edit. STEMDECK_DATA_DIR is the portable app root for
# mutable runtime data; when unset, dev behavior remains the repo-local jobs/
# folder.
PORTABLE_DATA_DIR_ENABLED = bool(os.environ.get("STEMDECK_DATA_DIR", "").strip())
DATA_DIR = _env_path("STEMDECK_DATA_DIR", ROOT)
JOBS_DIR = _env_path(
    "STEMDECK_JOBS_DIR",
    (DATA_DIR / "jobs") if PORTABLE_DATA_DIR_ENABLED else (ROOT / "jobs"),
)
CACHE_DIR = _env_path("STEMDECK_CACHE_DIR", DATA_DIR / "cache")
DOWNLOADS_DIR = _env_path("STEMDECK_DOWNLOADS_DIR", DATA_DIR / "downloads")
MODELS_DIR = _env_path("STEMDECK_MODELS_DIR", DATA_DIR / "models")
LOGS_DIR = _env_path("STEMDECK_LOGS_DIR", DATA_DIR / "logs")
FFMPEG_DIR = _env_path("STEMDECK_FFMPEG_DIR", DATA_DIR / "ffmpeg")
FFMPEG_BIN = _env_path(
    "STEMDECK_FFMPEG",
    FFMPEG_DIR / ("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg"),
)
FFPROBE_BIN = _env_path(
    "STEMDECK_FFPROBE",
    FFMPEG_DIR / ("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe"),
)
SEPARATION_BACKEND = (
    os.environ.get("STEMDECK_SEPARATOR", "residual_allocator").strip().lower().replace("-", "_")
    or "residual_allocator"
)
DEMUCS_MODEL = os.environ.get("STEMDECK_DEMUCS_MODEL", "htdemucs_6s").strip() or "htdemucs_6s"
DEMUCS_DEVICE = _detect_device("STEMDECK_DEMUCS_DEVICE")
RESIDUAL_ALLOCATOR_DIR = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_DIR",
    _default_residual_allocator_dir(),
    ROOT,
)
RESIDUAL_ALLOCATOR_SCRIPT = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_SCRIPT",
    RESIDUAL_ALLOCATOR_DIR / "infer.py",
    ROOT,
)
RESIDUAL_ALLOCATOR_BASE_CONFIG = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_BASE_CONFIG",
    RESIDUAL_ALLOCATOR_DIR / "configs" / "bs_roformer_sw_fixed_alloc.yaml",
    ROOT,
)
RESIDUAL_ALLOCATOR_BASE_CHECKPOINT = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_BASE_CHECKPOINT",
    RESIDUAL_ALLOCATOR_DIR / "weights" / "BS-Rofo-SW-Fixed.ckpt",
    ROOT,
)
RESIDUAL_ALLOCATOR_CONFIG = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_CONFIG",
    RESIDUAL_ALLOCATOR_DIR / "configs" / "residual_allocator.yaml",
    ROOT,
)
RESIDUAL_ALLOCATOR_CHECKPOINT = _env_path_from(
    "STEMDECK_RESIDUAL_ALLOCATOR_CHECKPOINT",
    RESIDUAL_ALLOCATOR_DIR / "weights" / "residual_allocator.safetensors",
    ROOT,
)
RESIDUAL_ALLOCATOR_DEVICE = _detect_device("STEMDECK_RESIDUAL_ALLOCATOR_DEVICE")
RESIDUAL_ALLOCATOR_ARGS = _env_args("STEMDECK_RESIDUAL_ALLOCATOR_ARGS")
UPMIXER_DIR = _env_path_from(
    "STEMDECK_UPMIXER_DIR",
    ROOT / "vendor" / "Stems-Upmixer",
    ROOT,
)
UPMIXER_SCRIPT = _env_path_from(
    "STEMDECK_UPMIXER_SCRIPT",
    UPMIXER_DIR / "upmix_cli.py",
    ROOT,
)
UPMIXER_ARGS = _env_args("STEMDECK_UPMIXER_ARGS")
MAX_DURATION_SEC = max(60, _env_int("STEMDECK_MAX_DURATION_SEC", 1200))  # 20 min default
JOB_TTL_SECONDS = max(300, _env_int("STEMDECK_JOB_TTL_SECONDS", 24 * 3600))  # 24 h default
MAX_PENDING_JOBS = max(1, min(50, _env_int("STEMDECK_MAX_PENDING_JOBS", 3)))


def ffmpeg_executable() -> str:
    """Return the preferred FFmpeg executable.

    In portable mode, setup places FFmpeg under DATA_DIR/ffmpeg. Prefer that
    binary when present; otherwise fall back to PATH so local dev and Docker
    keep working exactly as before.
    """
    return str(FFMPEG_BIN) if FFMPEG_BIN.is_file() else "ffmpeg"


def ffprobe_executable() -> str:
    """Return the preferred ffprobe executable (same bundled dir as ffmpeg)."""
    return str(FFPROBE_BIN) if FFPROBE_BIN.is_file() else "ffprobe"


def configure_portable_environment() -> None:
    """Keep generated caches inside the portable data folder when requested.

    This is intentionally best-effort. It only sets variables that are still
    unset, so explicit caller/env choices win.
    """
    if FFMPEG_DIR.is_dir():
        path = os.environ.get("PATH", "")
        ffmpeg_path = str(FFMPEG_DIR)
        if ffmpeg_path not in path.split(os.pathsep):
            os.environ["PATH"] = ffmpeg_path + (os.pathsep + path if path else "")

    if PORTABLE_DATA_DIR_ENABLED:
        os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
        os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))


def ensure_runtime_dirs() -> None:
    paths = (
        (JOBS_DIR, CACHE_DIR, DOWNLOADS_DIR, MODELS_DIR, LOGS_DIR)
        if PORTABLE_DATA_DIR_ENABLED
        else (JOBS_DIR,)
    )
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
