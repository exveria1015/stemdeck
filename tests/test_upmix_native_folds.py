from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import outputs  # noqa: E402


def test_parse_loudnorm_output_loudness_reads_render_output_fields():
    loudness = outputs.parse_loudnorm_output_loudness(
        """
        [Parsed_loudnorm_1 @ 0x0]
        {
            "input_i" : "-19.23",
            "input_tp" : "-3.93",
            "input_lra" : "10.80",
            "input_thresh" : "-29.55",
            "output_i" : "-13.85",
            "output_tp" : "-1.26",
            "output_lra" : "8.70",
            "output_thresh" : "-23.91",
            "normalization_type" : "linear",
            "target_offset" : "-0.15"
        }
        """
    )

    assert loudness == {
        "integrated_lufs": -13.85,
        "true_peak_dbtp": -1.26,
        "lra_lu": 8.7,
        "threshold_lufs": -23.91,
    }


def _write_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(path, audio.astype(np.float32), 48000, subtype="FLOAT")


def _read_wav(path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float64")
    assert sample_rate == 48000
    return audio


def _disable_ffmpeg(monkeypatch) -> None:
    def fail_run(*_args, **_kwargs):
        raise AssertionError("ffmpeg subprocess should not be used for native fold")

    monkeypatch.setattr(outputs, "run", fail_run)


def test_render_stereo_fold_wav_uses_native_path(monkeypatch, tmp_path):
    _disable_ffmpeg(monkeypatch)
    bed = tmp_path / "bed.wav"
    out = tmp_path / "stereo.wav"
    audio = np.linspace(-0.08, 0.08, 5 * 12, dtype=np.float64).reshape(5, 12)
    _write_wav(bed, audio)

    outputs.render_stereo_fold_wav("ffmpeg-disabled", bed, out, 0.25, None)

    folded = _read_wav(out)
    expected_left = (
        0.92 * audio[:, 0]
        + 0.7071 * audio[:, 2]
        + 0.50 * audio[:, 4]
        + 0.55 * audio[:, 6]
        + 0.35 * audio[:, 8]
        + 0.25 * audio[:, 10]
        + 0.25 * audio[:, 3]
    )
    expected_right = (
        0.92 * audio[:, 1]
        + 0.7071 * audio[:, 2]
        + 0.50 * audio[:, 5]
        + 0.55 * audio[:, 7]
        + 0.35 * audio[:, 9]
        + 0.25 * audio[:, 11]
        + 0.25 * audio[:, 3]
    )
    np.testing.assert_allclose(
        folded, np.stack([expected_left, expected_right], axis=1), atol=2e-6
    )


def test_render_51_fold_wav_uses_native_path(monkeypatch, tmp_path):
    _disable_ffmpeg(monkeypatch)
    bed = tmp_path / "bed.wav"
    out = tmp_path / "surround.wav"
    audio = np.linspace(-0.05, 0.05, 4 * 12, dtype=np.float64).reshape(4, 12)
    _write_wav(bed, audio)

    outputs.render_51_fold_wav("ffmpeg-disabled", bed, out, 0.4)

    folded = _read_wav(out)
    expected = np.stack(
        [
            0.92 * audio[:, 0]
            + 0.24 * audio[:, 2]
            + 0.20 * audio[:, 4]
            + 0.18 * audio[:, 8]
            + 0.10 * audio[:, 10],
            0.92 * audio[:, 1]
            + 0.24 * audio[:, 2]
            + 0.20 * audio[:, 5]
            + 0.18 * audio[:, 9]
            + 0.10 * audio[:, 11],
            0.78 * audio[:, 2] + 0.05 * audio[:, 0] + 0.05 * audio[:, 1],
            0.4 * audio[:, 3],
            0.78 * audio[:, 6]
            + 0.40 * audio[:, 4]
            + 0.20 * audio[:, 8]
            + 0.34 * audio[:, 10],
            0.78 * audio[:, 7]
            + 0.40 * audio[:, 5]
            + 0.20 * audio[:, 9]
            + 0.34 * audio[:, 11],
        ],
        axis=1,
    )
    np.testing.assert_allclose(folded, expected, atol=2e-6)


def test_render_51_to_stereo_fold_wav_uses_native_path(monkeypatch, tmp_path):
    _disable_ffmpeg(monkeypatch)
    surround = tmp_path / "surround.wav"
    out = tmp_path / "stereo.wav"
    audio = np.linspace(-0.06, 0.06, 6 * 6, dtype=np.float64).reshape(6, 6)
    _write_wav(surround, audio)

    outputs.render_51_to_stereo_fold_wav("ffmpeg-disabled", surround, out, 0.2)

    folded = _read_wav(out)
    expected = np.stack(
        [
            0.92 * audio[:, 0] + 0.7071 * audio[:, 2] + 0.55 * audio[:, 4]
            + 0.2 * audio[:, 3],
            0.92 * audio[:, 1] + 0.7071 * audio[:, 2] + 0.55 * audio[:, 5]
            + 0.2 * audio[:, 3],
        ],
        axis=1,
    )
    np.testing.assert_allclose(folded, expected, atol=2e-6)


def test_media_duration_uses_soundfile_when_ffprobe_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    _write_wav(path, np.zeros((4800, 2), dtype=np.float32))

    def fail_ffprobe(_ffprobe, _path):
        raise FileNotFoundError("ffprobe")

    from upmixer import utils

    monkeypatch.setattr(utils, "ffprobe_json", fail_ffprobe)

    assert utils.media_duration("ffprobe-disabled", path) == 0.1


def test_media_tags_returns_empty_when_ffprobe_unavailable(monkeypatch, tmp_path):
    path = tmp_path / "audio.wav"
    _write_wav(path, np.zeros((16, 2), dtype=np.float32))
    from upmixer import utils

    def fail_ffprobe(_ffprobe, _path):
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(utils, "ffprobe_json", fail_ffprobe)

    assert utils.media_tags("ffprobe-disabled", path) == {}


def test_window_rms_db_matches_reference_loop():
    audio = np.arange(20, dtype=np.float64).reshape(10, 2) / 100.0
    centers = [0.001, 0.003, 0.007]
    sample_rate = 1000
    window_sec = 0.004
    half = max(1, int(sample_rate * window_sec * 0.5))
    expected = []
    for center in centers:
        center_index = int(center * sample_rate)
        start = max(0, center_index - half)
        end = min(len(audio), center_index + half)
        frame = audio[start:end]
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        expected.append(20.0 * math.log10(max(rms, 1e-12)))

    actual = outputs._window_rms_db(
        audio, sample_rate, centers, window_sec=window_sec
    )

    np.testing.assert_allclose(actual, np.asarray(expected), atol=1e-12)


def test_analyze_multiband_filters_each_band_once(monkeypatch, tmp_path):
    audio = np.column_stack(
        [
            np.linspace(-0.2, 0.2, 32, dtype=np.float64),
            np.linspace(0.15, -0.15, 32, dtype=np.float64),
        ]
    )
    calls = []

    def fake_audio(_path, _lfe_fold_gain):
        return audio, 48000

    def fake_filter(data, sample_rate, low_hz, high_hz):
        calls.append((sample_rate, low_hz, high_hz, data.shape))
        return np.asarray(data) * (1.0 + low_hz / high_hz)

    monkeypatch.setattr(outputs, "multiband_stereo_audio_numpy", fake_audio)
    monkeypatch.setattr(outputs, "multiband_filter_numpy", fake_filter)

    stats = outputs.analyze_multiband_numpy("actual", tmp_path / "bed.wav", 0.4)

    assert len(stats.rows) == len(outputs.MULTIBAND_RANGES)
    assert len(calls) == len(outputs.MULTIBAND_RANGES)
    assert {call[3] for call in calls} == {audio.shape}
