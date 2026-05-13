from __future__ import annotations

from pathlib import Path

from app.core.models import Job


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return path


def test_residual_allocator_command_uses_configured_checkout(monkeypatch, tmp_path: Path):
    import app.pipeline.separate as separate

    checkout = tmp_path / "Residual-Allocator"
    script = _touch(checkout / "infer.py")
    base_config = _touch(checkout / "configs" / "bs_roformer_sw_fixed_alloc.yaml")
    base_checkpoint = _touch(checkout / "weights" / "BS-Rofo-SW-Fixed.ckpt")
    allocator_config = _touch(checkout / "configs" / "residual_allocator.yaml")
    allocator_checkpoint = _touch(checkout / "weights" / "residual_allocator.safetensors")
    source = _touch(tmp_path / "source.wav")

    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_DIR", checkout)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_SCRIPT", script)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_BASE_CONFIG", base_config)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_BASE_CHECKPOINT", base_checkpoint)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_CONFIG", allocator_config)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_CHECKPOINT", allocator_checkpoint)
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_DEVICE", "cpu")
    monkeypatch.setattr(separate, "RESIDUAL_ALLOCATOR_ARGS", ["--base-only"])

    cmd, output_dir, cwd = separate._residual_allocator_cmd(
        Job(id="abcdefabcdef"),
        source,
        tmp_path,
    )

    assert cmd[1] == str(script)
    assert "--output-dir-is-stem-dir" in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--base-checkpoint") + 1] == str(base_checkpoint)
    assert "--base-only" in cmd
    assert output_dir == tmp_path / "residual_allocator"
    assert cwd == checkout
