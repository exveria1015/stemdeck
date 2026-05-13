from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import stems  # noqa: E402
from upmixer.models import StemStats, VolumeStats  # noqa: E402


def test_analyze_window_stats_numpy_matches_reference_loop(tmp_path):
    path = tmp_path / "stem.wav"
    audio = np.column_stack(
        [
            np.sin(np.linspace(0.0, 4.0, 96, dtype=np.float64)) * 0.15,
            np.cos(np.linspace(0.0, 3.0, 96, dtype=np.float64)) * 0.11,
        ]
    )
    sample_rate = 1000
    window_ms = 18.0
    hop_ms = 7.0
    sf.write(path, audio.astype(np.float32), sample_rate, subtype="FLOAT")

    result = stems.analyze_window_stats_numpy(
        "guitar", path, window_ms=window_ms, hop_ms=hop_ms
    )

    window = max(1, int(sample_rate * window_ms / 1000.0))
    hop = max(1, int(sample_rate * hop_ms / 1000.0))
    frames = [
        audio[start : start + window]
        for start in range(0, len(audio) - window + 1, hop)
    ]
    envelope = []
    for frame in frames:
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        envelope.append(20.0 * math.log10(max(rms, 1e-12)))
    expected = np.asarray(envelope, dtype=np.float64)
    p10, p50, p90 = [float(np.percentile(expected, value)) for value in (10, 50, 90)]
    active_threshold = max(p90 - 24.0, p50 - 9.0, -60.0)

    np.testing.assert_allclose(result.envelope_db, expected, atol=1e-5)
    assert result.name == "guitar"
    assert result.p10_db == pytest.approx(p10, abs=1e-5)
    assert result.p50_db == pytest.approx(p50, abs=1e-5)
    assert result.p90_db == pytest.approx(p90, abs=1e-5)
    assert result.active_ratio == pytest.approx(
        float(np.mean(expected >= active_threshold)), abs=1e-5
    )


def test_stem_range_energies_cache_reuses_file_result(monkeypatch, tmp_path):
    path = tmp_path / "stem.wav"
    path.write_bytes(b"RIFF")
    stats = StemStats(
        name="bass",
        path=path,
        full=VolumeStats(0.0, 0.0),
        low=VolumeStats(0.0, 0.0),
        mid=VolumeStats(0.0, 0.0),
        high=VolumeStats(0.0, 0.0),
        mono=VolumeStats(0.0, 0.0),
        side=VolumeStats(0.0, 0.0),
    )
    calls = []

    def fake_uncached(input_path, ranges):
        calls.append((input_path, ranges))
        return (("sub", 0.25), ("bass", 0.5))

    monkeypatch.setattr(stems, "_stem_range_energies_uncached", fake_uncached)
    stems._stem_range_energies_cached.cache_clear()
    stems._stem_standard_range_energies_cached.cache_clear()
    try:
        ranges = (("sub", 20.0, 60.0), ("bass", 60.0, 120.0))

        first = stems.stem_range_energies_numpy(stats, ranges)
        second = stems.stem_range_energies_numpy(stats, ranges)
    finally:
        stems._stem_range_energies_cached.cache_clear()
        stems._stem_standard_range_energies_cached.cache_clear()

    assert first == {"sub": 0.25, "bass": 0.5}
    assert second == first
    assert len(calls) == 1
