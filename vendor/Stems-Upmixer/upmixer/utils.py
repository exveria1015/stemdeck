"""Small utilities, process helpers, and media probing primitives."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from numpy.typing import NDArray

    AudioArray = NDArray[Any]
else:
    AudioArray = Any


from .models import (
    FILTER_COMPLEX_SCRIPT_CHAR_THRESHOLD,
    FILTER_COMPLEX_SCRIPT_NODE_THRESHOLD,
    FILTER_COMPLEX_WARN_CHAR_THRESHOLD,
    FILTER_COMPLEX_WARN_NODE_THRESHOLD,
    STEM_ORDER,
    VOL_RE,
    VolumeStats,
)


def db_to_amp(db: float) -> float:
    if math.isinf(db):
        return 0.0
    return 10.0 ** (db / 20.0)


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def fmt(value: float) -> str:
    if abs(value) < 0.0005:
        return "0"
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def safe_label(text: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    label = label.strip("_")
    return label or "stem"


def safe_filename(text: str) -> str:
    name = re.sub(r"[^\w .()-]+", "_", text.strip(), flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name.strip("._") or "upmix"


def parse_presence_priority(raw: str, available: Iterable[str]) -> tuple[str, ...]:
    available_set = set(available)
    known = set(STEM_ORDER)
    priority: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        stem = part.strip()
        if not stem:
            continue
        if stem not in known:
            raise SystemExit(
                f"Unknown --presence-priority stem {stem!r}. Expected one of: {', '.join(STEM_ORDER)}"
            )
        if stem not in available_set:
            continue
        if stem not in seen:
            priority.append(stem)
            seen.add(stem)
    return tuple(priority)


def parse_stem_subset(
    raw: str, available: Iterable[str], *, option: str
) -> tuple[str, ...]:
    available_set = set(available)
    known = set(STEM_ORDER)
    selected: list[str] = []
    seen: set[str] = set()
    for part in raw.split(","):
        stem = part.strip()
        if not stem:
            continue
        if stem not in known:
            raise SystemExit(
                f"Unknown {option} stem {stem!r}. Expected one of: {', '.join(STEM_ORDER)}"
            )
        if stem not in available_set:
            continue
        if stem not in seen:
            selected.append(stem)
            seen.add(stem)
    return tuple(selected)


def run(
    cmd: list[str], *, check: bool = True, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    kwargs = {
        "text": True,
        "stdout": subprocess.PIPE if capture else None,
        "stderr": subprocess.PIPE if capture else None,
    }
    proc = subprocess.run(cmd, **kwargs)
    if check and proc.returncode != 0:
        if capture:
            if proc.stdout:
                print(proc.stdout, file=sys.stderr)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    return proc


def format_elapsed(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds >= 3600.0:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remain = seconds % 60
        return f"{hours}h{minutes:02d}m{remain:04.1f}s"
    if seconds >= 60.0:
        minutes = int(seconds // 60)
        remain = seconds % 60
        return f"{minutes}m{remain:04.1f}s"
    return f"{seconds:.2f}s"


def print_elapsed(label: str, start_time: float) -> None:
    print(
        f"{label} elapsed: {format_elapsed(time.perf_counter() - start_time)}",
        flush=True,
    )


def filter_complex_metrics(filter_complex: str) -> tuple[int, int]:
    return len(filter_complex), filter_complex.count(";") + 1 if filter_complex else 0


def should_use_filter_complex_script(filter_complex: str) -> bool:
    char_count, node_count = filter_complex_metrics(filter_complex)
    return (
        char_count >= FILTER_COMPLEX_SCRIPT_CHAR_THRESHOLD
        or node_count >= FILTER_COMPLEX_SCRIPT_NODE_THRESHOLD
    )


def append_filter_complex_arg(
    cmd: list[str],
    filter_complex: str,
    temp_paths: list[Path],
    *,
    label: str,
) -> None:
    char_count, node_count = filter_complex_metrics(filter_complex)
    if should_use_filter_complex_script(filter_complex):
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            suffix=".ffgraph",
            prefix=f"upmix_{safe_label(label)}_",
            delete=False,
        ) as handle:
            handle.write(filter_complex)
            handle.write("\n")
            path = Path(handle.name)
        temp_paths.append(path)
        print(
            f"FFmpeg filter graph: using -filter_complex_script for {label} "
            f"(chars={char_count} nodes={node_count})",
            flush=True,
        )
        cmd.extend(["-filter_complex_script", str(path)])
        return
    if (
        char_count >= FILTER_COMPLEX_WARN_CHAR_THRESHOLD
        or node_count >= FILTER_COMPLEX_WARN_NODE_THRESHOLD
    ):
        print(
            f"FFmpeg filter graph: {label} is large "
            f"(chars={char_count} nodes={node_count})",
            flush=True,
        )
    cmd.extend(["-filter_complex", filter_complex])


def parse_volume(stderr: str) -> VolumeStats:
    values: dict[str, float] = {}
    for key, raw in VOL_RE.findall(stderr):
        values[key] = float("-inf") if raw == "-inf" else float(raw)
    if "mean" not in values or "max" not in values:
        raise RuntimeError(
            "ffmpeg volumedetect output did not include mean_volume/max_volume"
        )
    return VolumeStats(mean_db=values["mean"], max_db=values["max"])


def volumedetect(ffmpeg: str, path: Path, audio_filter: str) -> VolumeStats:
    proc = run(
        [
            ffmpeg,
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            audio_filter,
            "-f",
            "null",
            "-",
        ],
        capture=True,
    )
    return parse_volume(proc.stderr or "")


def volumedetect_complex(
    ffmpeg: str, path: Path, filter_complex: str, label: str
) -> VolumeStats:
    temp_paths: list[Path] = []
    cmd = [ffmpeg, "-hide_banner", "-nostats", "-i", str(path)]
    append_filter_complex_arg(
        cmd, filter_complex, temp_paths, label=f"volumedetect_{label}"
    )
    cmd.extend(["-map", f"[{label}]", "-f", "null", "-"])
    try:
        proc = run(cmd, capture=True)
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)
    return parse_volume(proc.stderr or "")


def ffprobe_json(ffprobe: str, path: Path) -> dict:
    proc = run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:format_tags:stream=index,codec_type,codec_name,sample_rate,channels,channel_layout,duration,width,height",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    return json.loads(proc.stdout or "{}")


def media_duration(ffprobe: str, path: Path) -> float:
    try:
        data = ffprobe_json(ffprobe, path)
    except (Exception, SystemExit):
        try:
            import soundfile as sf

            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception as exc:
            raise RuntimeError(f"Could not determine duration for {path}") from exc
    duration = (data.get("format") or {}).get("duration")
    if duration is not None:
        return float(duration)
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "audio" and stream.get("duration"):
            return float(stream["duration"])
    raise RuntimeError(f"Could not determine duration for {path}")


def media_tags(ffprobe: str, path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    try:
        data = ffprobe_json(ffprobe, path)
    except (Exception, SystemExit):
        return {}
    tags = (data.get("format") or {}).get("tags") or {}
    return {str(k).lower(): str(v) for k, v in tags.items()}


def read_audio_float(
    path: Path,
    *,
    dtype: str = "float64",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
) -> tuple[AudioArray, int]:
    import numpy as np

    try:
        import soundfile as sf

        audio_raw, sample_rate = sf.read(path, always_2d=True, dtype=dtype)
        audio = cast(AudioArray, audio_raw)
        return audio, int(sample_rate)
    except Exception as soundfile_exc:
        data = ffprobe_json(ffprobe, path)
        audio_stream = next(
            (
                stream
                for stream in data.get("streams", [])
                if stream.get("codec_type") == "audio"
            ),
            {},
        )
        sample_rate = int(audio_stream.get("sample_rate") or 48000)
        channels = max(1, int(audio_stream.get("channels") or 2))
        proc = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-map",
                "0:a:0",
                "-f",
                "f32le",
                "-acodec",
                "pcm_f32le",
                "-ac",
                str(channels),
                "-ar",
                str(sample_rate),
                "-",
            ],
            capture_output=True,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"audio decode failed for {path}: {detail}"
            ) from soundfile_exc
        raw = np.frombuffer(proc.stdout, dtype=np.float32)
        frame_count = raw.size // channels
        if frame_count <= 0:
            raise RuntimeError(
                f"audio decode produced no samples for {path}"
            ) from soundfile_exc
        audio = raw[: frame_count * channels].reshape(frame_count, channels)
        return cast(AudioArray, audio.astype(dtype, copy=False)), sample_rate


def linear_energy(mean_db: float) -> float:
    amp = db_to_amp(mean_db)
    return amp * amp


def median_float(values: Iterable[float]) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    middle = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return float(sorted_values[middle])
    return float((sorted_values[middle - 1] + sorted_values[middle]) * 0.5)


__all__ = [
    "db_to_amp",
    "clamp",
    "fmt",
    "safe_label",
    "safe_filename",
    "parse_presence_priority",
    "parse_stem_subset",
    "run",
    "format_elapsed",
    "print_elapsed",
    "filter_complex_metrics",
    "should_use_filter_complex_script",
    "append_filter_complex_arg",
    "parse_volume",
    "volumedetect",
    "volumedetect_complex",
    "ffprobe_json",
    "media_duration",
    "media_tags",
    "read_audio_float",
    "linear_energy",
    "median_float",
]
