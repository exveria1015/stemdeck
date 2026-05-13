from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import outputs  # noqa: E402

NATIVE_MANIFEST = UPMIXER_ROOT / "native_loudnorm" / "Cargo.toml"


def _build_native_loudnorm() -> Path:
    if shutil.which("cargo") is None:
        pytest.skip("cargo is required for native loudnorm tests")
    subprocess.run(
        ["cargo", "build", "--quiet", "--manifest-path", str(NATIVE_MANIFEST)],
        check=True,
    )
    binary = UPMIXER_ROOT / "native_loudnorm" / "target" / "debug" / "stems-upmixer-loudnorm"
    if not binary.exists():
        pytest.skip("native loudnorm binary was not built")
    return binary


def _write_reference_bed(path: Path, *, duration: float = 20.0) -> None:
    sample_rate = 48_000
    rng = np.random.default_rng(123)
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    audio = np.zeros((len(time), 12), dtype=np.float32)
    for channel in range(12):
        noise = rng.normal(0.0, 1.0, len(time)).astype(np.float32)
        noise = np.convolve(noise, np.ones(8, dtype=np.float32) / 8.0, mode="same")
        envelope = np.where(time < 6.0, 0.6, np.where(time < 14.0, 1.0, 0.4))
        audio[:, channel] = (0.025 + 0.002 * channel) * envelope * noise
    sf.write(path, audio, sample_rate, subtype="PCM_24")


def test_native_loudnorm_matches_ffmpeg_measurement_envelope(monkeypatch, tmp_path):
    binary = _build_native_loudnorm()
    monkeypatch.setenv("STEMS_UPMIXER_LOUDNORM_BIN", str(binary))
    bed = tmp_path / "bed_714.wav"
    _write_reference_bed(bed)

    ffmpeg = outputs.measure_51_loudnorm(
        "ffmpeg",
        bed,
        0.45,
        target_i=-14.0,
        target_tp=-1.3,
        target_lra=11.0,
        label="test_native_loudnorm_ffmpeg",
    )
    native = outputs.measure_51_loudnorm_native(
        bed,
        0.45,
        target_i=-14.0,
        target_tp=-1.3,
        target_lra=11.0,
    )

    assert native is not None
    assert abs(float(native["input_i"]) - float(ffmpeg["input_i"])) <= 0.75
    assert abs(float(native["input_thresh"]) - float(ffmpeg["input_thresh"])) <= 0.75
    assert abs(float(native["input_tp"]) - float(ffmpeg["input_tp"])) <= 0.25
    assert abs(float(native["input_lra"]) - float(ffmpeg["input_lra"])) <= 0.25
    assert abs(float(native["target_offset"]) - float(ffmpeg["target_offset"])) <= 0.75


def test_native_loudnorm_writes_shortterm_analysis(tmp_path):
    binary = _build_native_loudnorm()
    bed = tmp_path / "bed_714.wav"
    analysis_json = tmp_path / "shortterm.json"
    analysis_svg = tmp_path / "shortterm.svg"
    _write_reference_bed(bed)

    subprocess.run(
        [
            str(binary),
            "--bed",
            str(bed),
            "--analysis-json",
            str(analysis_json),
            "--analysis-svg",
            str(analysis_svg),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    data = json.loads(analysis_json.read_text())
    assert data["schema"] == "stems-upmixer-shortterm-v1"
    assert data["window_sec"] == 3.0
    assert data["hop_sec"] == 0.1
    assert data["input"]["stats"]["total_count"] > 0
    assert data["input"]["stats"]["rel_gated_count"] > 0
    assert data["input"]["stats"]["p95_lufs"] >= data["input"]["stats"]["p10_lufs"]
    assert data["input"]["audition_markers"]
    assert data["input"]["timeline"][0]["center_sec"] == 1.5
    assert analysis_svg.read_text().startswith("<svg")


def test_native_loudnorm_renders_51_wav(monkeypatch, tmp_path):
    binary = _build_native_loudnorm()
    monkeypatch.setenv("STEMS_UPMIXER_LOUDNORM_BIN", str(binary))
    bed = tmp_path / "bed_714.wav"
    rendered = tmp_path / "rendered_51.wav"
    _write_reference_bed(bed, duration=8.0)

    loudness = outputs.render_51_wav_native(
        "ffmpeg",
        bed,
        rendered,
        0.45,
        "loudnorm",
        -14.0,
        -1.3,
        11.0,
    )

    info = sf.info(rendered)
    audio, sample_rate = sf.read(rendered, always_2d=True)
    assert loudness is not None
    assert info.channels == 6
    assert sample_rate == 48_000
    assert audio.shape[0] > 0
    assert np.isfinite(audio).all()
    assert float(np.max(np.abs(audio))) > 0.0
    assert "integrated_lufs" in loudness
    assert "true_peak_dbtp" in loudness


def test_native_loudnorm_reference_job_matches_current_ffmpeg_when_enabled():
    if os.environ.get("RUN_UPMIX_REAL_JOB_TESTS") != "1":
        pytest.skip("set RUN_UPMIX_REAL_JOB_TESTS=1 to run the full reference job check")
    binary = _build_native_loudnorm()
    bed_candidates = list(
        Path("jobs/d301ce723981/upmix_verify_loudnorm_reuse_20260512_123619").glob(
            "*_7.1.4_*.wav"
        )
    )
    if not bed_candidates:
        pytest.skip("reference upmix bed is not available")
    bed = bed_candidates[0]

    proc = subprocess.run(
        [
            str(binary),
            "--bed",
            str(bed),
            "--lfe-fold-gain",
            "0.45",
            "--target-i",
            "-14",
            "--target-tp",
            "-1.3",
            "--target-lra",
            "11",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    native = json.loads(proc.stdout)
    ffmpeg = outputs.measure_51_loudnorm(
        "ffmpeg",
        bed,
        0.45,
        target_i=-14.0,
        target_tp=-1.3,
        target_lra=11.0,
        label="reference_native_loudnorm_ffmpeg",
    )

    assert abs(float(native["input_i"]) - float(ffmpeg["input_i"])) <= 0.05
    assert abs(float(native["input_thresh"]) - float(ffmpeg["input_thresh"])) <= 0.05
    assert abs(float(native["input_tp"]) - float(ffmpeg["input_tp"])) <= 0.10
    assert abs(float(native["input_lra"]) - float(ffmpeg["input_lra"])) <= 0.15
    assert abs(float(native["target_offset"]) - float(ffmpeg["target_offset"])) <= 0.25
