from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np

UPMIXER_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "Stems-Upmixer"
if str(UPMIXER_ROOT) not in sys.path:
    sys.path.insert(0, str(UPMIXER_ROOT))

from upmixer import outputs  # noqa: E402


def _stats() -> SimpleNamespace:
    return SimpleNamespace(
        chunks=1,
        input_sample_rate=48000,
        model_sample_rate=48000,
        mix=0.65,
        peak_before=0.1,
        peak_after=0.1,
        peak_scale=1.0,
    )


def test_stem_de_limiter_reuses_loaded_model(monkeypatch, tmp_path):
    calls: dict[str, object] = {
        "loads": 0,
        "conditions": [],
        "writes": {},
    }

    class FakeModel:
        def to(self, device):
            calls["model_device"] = device
            return self

        def eval(self):
            calls["model_eval"] = True
            return self

    fake_runtime = SimpleNamespace()

    def resolve_device(device):
        calls["resolved_device"] = device
        return f"resolved:{device}"

    def load_checkpoint(path, *, device):
        calls["loads"] = int(calls["loads"]) + 1
        calls["checkpoint"] = path
        calls["load_device"] = device
        return FakeModel()

    def apply_model(audio, sample_rate, model, **kwargs):
        conditions = calls["conditions"]
        assert isinstance(conditions, list)
        conditions.append(kwargs["condition_name"])
        return audio.copy(), _stats()

    fake_runtime._resolve_device = resolve_device
    fake_runtime.load_de_limiter_checkpoint = load_checkpoint
    fake_runtime.apply_de_limiter_model_to_array = apply_model

    fake_soundfile = types.SimpleNamespace()

    def read(path, *, always_2d, dtype):
        assert always_2d is True
        assert dtype == "float32"
        return np.full((8, 2), 0.1, dtype=np.float32), 48000

    def write(path, data, sample_rate, *, subtype):
        writes = calls["writes"]
        assert isinstance(writes, dict)
        writes[Path(path).name] = (np.asarray(data), sample_rate, subtype)

    fake_soundfile.read = read
    fake_soundfile.write = write

    monkeypatch.setattr(outputs, "load_de_limiter_module", lambda: fake_runtime)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    stems = {
        "vocals": tmp_path / "vocals.wav",
        "drums": tmp_path / "drums.wav",
    }
    result = outputs.apply_stem_de_limiter_to_stems(
        stems,
        tmp_path / "checkpoint.safetensors",
        tmp_path / "out",
        mix=0.65,
        chunk_seconds=8.0,
        overlap_seconds=1.0,
        device="auto",
        peak_limit=0.98,
    )

    assert calls["loads"] == 1
    assert calls["resolved_device"] == "auto"
    assert calls["load_device"] == "resolved:auto"
    assert calls["model_device"] == "resolved:auto"
    assert calls["model_eval"] is True
    assert calls["conditions"] == ["vocals", "drums"]
    assert set(result) == {"vocals", "drums"}
    assert set(calls["writes"]) == {"vocals_sgi_delimited.wav", "drums_sgi_delimited.wav"}
