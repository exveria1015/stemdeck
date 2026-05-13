from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import temporal  # noqa: E402


def test_temporal_window_rms_and_loudness_match_reference_loop():
    data = np.column_stack(
        [
            np.linspace(-0.3, 0.3, 64, dtype=np.float64),
            np.cos(np.linspace(0.0, 3.0, 64, dtype=np.float64)) * 0.2,
        ]
    )
    centers = (0.004, 0.012, 0.027, 0.055)
    sample_rate = 1000
    window_sec = 0.018

    expected_rms = [
        temporal.audio_rms_db(
            temporal.temporal_window(data, sample_rate, center, window_sec)
        )
        for center in centers
    ]
    expected_loudness = [
        temporal.loudness_lufs_estimate(
            temporal.temporal_window(data, sample_rate, center, window_sec)
        )
        for center in centers
    ]

    np.testing.assert_allclose(
        temporal.window_rms_db_values(data, sample_rate, centers, window_sec),
        expected_rms,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        temporal.window_loudness_values(data, sample_rate, centers, window_sec),
        expected_loudness,
        atol=1e-12,
    )


def test_temporal_window_peak_matches_reference_loop():
    data = np.column_stack(
        [
            np.sin(np.linspace(0.0, 5.0, 72, dtype=np.float64)) * 0.18,
            np.cos(np.linspace(0.0, 4.0, 72, dtype=np.float64)) * 0.13,
        ]
    )
    data[20, 1] = -0.61
    centers = (0.004, 0.017, 0.031, 0.068)
    sample_rate = 1000
    window_sec = 0.020

    expected = [
        temporal.audio_peak_db(
            temporal.temporal_window(data, sample_rate, center, window_sec),
            sample_rate,
            oversample=1,
        )
        for center in centers
    ]

    np.testing.assert_allclose(
        temporal.window_peak_db_values(
            data,
            sample_rate,
            centers,
            window_sec,
            oversample=1,
        ),
        expected,
        atol=1e-12,
    )


def test_temporal_window_values_allow_centers_past_audio_end():
    data = np.column_stack(
        [
            np.linspace(-0.1, 0.1, 32, dtype=np.float64),
            np.linspace(0.1, -0.1, 32, dtype=np.float64),
        ]
    )
    centers = (0.010, 0.080)
    sample_rate = 1000
    window_sec = 0.020

    assert temporal.window_rms_db_values(data, sample_rate, centers, window_sec)[1] == -240.0
    assert temporal.window_peak_db_values(data, sample_rate, centers, window_sec)[1] == -120.0
    assert temporal.window_correlation_values(data, sample_rate, centers, window_sec)[1] == 1.0
    assert temporal.window_mono_loss_values(data, sample_rate, centers, window_sec)[1] == 0.0


def test_temporal_correlation_and_mono_loss_match_reference_loop():
    data = np.column_stack(
        [
            np.sin(np.linspace(0.0, 6.0, 80, dtype=np.float64)) * 0.25,
            np.cos(np.linspace(0.0, 5.0, 80, dtype=np.float64)) * 0.18,
        ]
    )
    centers = (0.006, 0.019, 0.041, 0.073)
    sample_rate = 1000
    window_sec = 0.022

    expected_corr = [
        temporal.stereo_correlation(
            temporal.temporal_window(data, sample_rate, center, window_sec)
        )
        for center in centers
    ]
    expected_mono_loss = [
        temporal.mono_fold_loss_db(
            temporal.temporal_window(data, sample_rate, center, window_sec)
        )
        for center in centers
    ]

    np.testing.assert_allclose(
        temporal.window_correlation_values(data, sample_rate, centers, window_sec),
        expected_corr,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        temporal.window_mono_loss_values(data, sample_rate, centers, window_sec),
        expected_mono_loss,
        atol=1e-12,
    )
