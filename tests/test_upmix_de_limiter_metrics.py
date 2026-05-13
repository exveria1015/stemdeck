from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import de_limiter  # noqa: E402
from upmixer.models import StemDeLimiterMetrics  # noqa: E402


def reference_crest_100ms_p95_db(data, sample_rate: int) -> float:
    window = max(1, int(round(sample_rate * 0.100)))
    hop = max(1, int(round(sample_rate * 0.050)))
    values: list[float] = []
    if len(data) < window:
        rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        if rms_db <= -100.0:
            return 0.0
        peak = float(np.max(np.abs(data))) if data.size else 0.0
        return 20.0 * math.log10(max(peak, 1e-12)) - rms_db
    for start in range(0, len(data) - window + 1, hop):
        frame = data[start : start + window]
        rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0
        rms_db = 20.0 * math.log10(max(rms, 1e-12))
        if rms_db <= -100.0:
            continue
        peak = float(np.max(np.abs(frame))) if frame.size else 0.0
        values.append(20.0 * math.log10(max(peak, 1e-12)) - rms_db)
    return de_limiter.percentile_float(values, 95.0) if values else 0.0


def test_crest_100ms_p95_db_matches_reference_loop():
    sample_rate = 1000
    time = np.linspace(0.0, 1.2, 1200, endpoint=False)
    data = np.column_stack(
        [
            0.15 * np.sin(time * 2.0 * math.pi * 4.0),
            0.12 * np.cos(time * 2.0 * math.pi * 7.0),
        ]
    )
    data[320:335, 0] += 0.55
    data[760:770, 1] -= 0.45

    result = de_limiter.crest_100ms_p95_db(data, sample_rate)

    assert result == pytest.approx(
        reference_crest_100ms_p95_db(data, sample_rate), abs=1e-9
    )


def test_stem_de_limiter_metrics_reuses_preloaded_mix(monkeypatch):
    mix = np.zeros((480, 2), dtype=np.float32)
    mix[120:180, 0] = 0.2
    paths = {"vocals": Path("/tmp/vocals.wav")}
    diagnostics_args = {}

    def fake_read_stem_mix_audio(stats):
        assert [item.name for item in stats] == ["vocals"]
        return mix, 48000

    def fake_compute_temporal_mix_diagnostics(stats, **kwargs):
        diagnostics_args.update(kwargs)
        return ()

    monkeypatch.setattr(de_limiter, "read_stem_mix_audio", fake_read_stem_mix_audio)
    monkeypatch.setattr(
        de_limiter,
        "compute_temporal_mix_diagnostics",
        fake_compute_temporal_mix_diagnostics,
    )

    metrics = de_limiter.stem_de_limiter_metrics_from_paths(
        "before",
        paths,
        hop_sec=1.0,
    )

    assert metrics.label == "before"
    assert diagnostics_args["mix_audio"] is mix
    assert diagnostics_args["sample_rate"] == 48000


def make_metrics(
    label: str,
    *,
    true_peak: float = -6.0,
    sample_peak: float = -6.0,
    rms: float = -20.0,
    density90: float = 0.0,
) -> StemDeLimiterMetrics:
    crest = sample_peak - rms
    return StemDeLimiterMetrics(
        label=label,
        true_peak_dbtp=true_peak,
        sample_peak_dbfs=sample_peak,
        rms_dbfs=rms,
        crest_factor_db=crest,
        crest_100ms_p95_db=crest,
        transient_lwmean=0.0,
        transient_p95=0.0,
        punch_lwmean=0.0,
        punch_p95=0.0,
        presence_harshness_lwmean=0.0,
        presence_harshness_p95=0.0,
        sibilance_lwmean=0.0,
        sibilance_p95=0.0,
        phase_risk_lwmean=0.0,
        phase_risk_p95=0.0,
        mono_fold_loss_db=0.0,
        mono_fold_loss_min_db=0.0,
        peak_density_gt_095_pct=0.0,
        peak_density_gt_090_pct=density90,
    )


def test_select_stem_de_limiter_candidates_screens_safe_stems_with_local_metrics(tmp_path):
    vocals = tmp_path / "vocals.wav"
    drums = tmp_path / "drums.wav"
    vocals_out = tmp_path / "vocals_sgi.wav"
    drums_out = tmp_path / "drums_sgi.wav"

    class FakeCache:
        def __init__(self):
            self.local_calls: list[str] = []
            self.full_calls: list[str] = []

        def local(self, label, paths):
            self.local_calls.append(label)
            if label == "drums_source":
                return make_metrics(
                    label,
                    true_peak=-0.2,
                    sample_peak=-0.3,
                    rms=-20.0,
                    density90=0.25,
                )
            return make_metrics(label)

        def full(self, label, paths, *, hop_sec=4.0):
            self.full_calls.append(label)
            if label == "drums_source":
                return make_metrics(
                    label,
                    true_peak=-0.2,
                    sample_peak=-0.3,
                    rms=-20.0,
                    density90=0.25,
                )
            return make_metrics(
                label,
                true_peak=-0.7,
                sample_peak=-0.8,
                rms=-20.6,
                density90=0.05,
            )

    cache = FakeCache()

    selected, decisions = de_limiter.select_stem_de_limiter_candidates(
        {"vocals": vocals, "drums": drums},
        {"vocals": vocals_out, "drums": drums_out},
        selected_mode="remaster",
        mix=0.65,
        hop_sec=4.0,
        metrics_cache=cache,
    )

    assert cache.local_calls == ["vocals_source", "drums_source"]
    assert cache.full_calls == [
        "drums_source",
        "drums_sgi_remaster_mix0.65",
    ]
    assert selected["vocals"] == vocals
    assert selected["drums"] == drums_out
    assert [(item.stem, item.status) for item in decisions] == [
        ("vocals", "skipped"),
        ("drums", "applied"),
    ]
