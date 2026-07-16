"""Tests for checkpoint save/load and export."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sr_engine.models.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    _build_model_from_checkpoint,
    export_to_torchscript,
)


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestCheckpoint:
    """Tests for ``save_checkpoint`` and ``load_checkpoint``."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """A saved checkpoint should be loadable with identical state dict."""
        path = tmp_path / "model.pt"
        state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        save_checkpoint(path, state_dict=state_dict, step=10)
        loaded = load_checkpoint(path)
        assert torch.allclose(loaded["state_dict"]["weight"], state_dict["weight"])
        assert loaded["step"] == 10

    def test_save_with_metadata(self, tmp_path: Path):
        """Checkpoints should preserve metadata like config and backend info."""
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(
            path,
            state_dict=state_dict,
            optimizer_state={"lr": 1e-4},
            step=5,
            config={"name": "rrdbnet", "scale": 4},
            backend_info={"device": "cuda"},
        )
        loaded = load_checkpoint(path)
        assert loaded["step"] == 5
        assert loaded["config"]["name"] == "rrdbnet"
        assert loaded["backend_info"]["device"] == "cuda"

    def test_load_missing_file_raises(self, tmp_path: Path):
        """Loading a nonexistent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_with_map_location(self, tmp_path: Path):
        """map_location should be respected when loading."""
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        loaded = load_checkpoint(path, map_location="cpu")
        assert loaded["state_dict"]["w"].device.type == "cpu"

    def test_load_ema_state(self, tmp_path: Path):
        """Loading with load_ema=True should return EMA weights."""
        path = tmp_path / "model.pt"
        state = {"w": torch.tensor([1.0])}
        ema_state = {"w": torch.tensor([2.0])}
        save_checkpoint(path, state_dict=state, ema_state_dict=ema_state)
        loaded = load_checkpoint(path, load_ema=True)
        assert torch.allclose(loaded["state_dict"]["w"], torch.tensor([2.0]))

    def test_load_ema_missing_raises(self, tmp_path: Path):
        """Loading EMA from a checkpoint without EMA state should raise."""
        path = tmp_path / "model.pt"
        save_checkpoint(path, state_dict={"w": torch.tensor([1.0])})
        with pytest.raises(ValueError, match="EMA"):
            load_checkpoint(path, load_ema=True)

    def test_weights_only_safe_by_default(self, tmp_path: Path):
        """Loading should succeed with weights_only=True."""
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        loaded = load_checkpoint(path)
        assert loaded is not None

    def test_atomic_write_does_not_corrupt_on_failure(self, tmp_path: Path):
        """Temporary files should not remain after a successful save."""
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        assert path.is_file()
        assert path.suffix == ".pt"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        assert not tmp_path.exists()


class TestBuildModelFromCheckpoint:
    """Tests for ``_build_model_from_checkpoint``."""

    def _make_rrdb_checkpoint(self):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        return {
            "state_dict": model.state_dict(),
            "config": {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3},
        }

    def test_rebuilds_model(self):
        """A checkpoint with config should rebuild the model and match state dict."""
        checkpoint = self._make_rrdb_checkpoint()
        rebuilt = _build_model_from_checkpoint(checkpoint)
        assert isinstance(rebuilt, nn.Module)
        for k, v in checkpoint["state_dict"].items():
            assert torch.allclose(v, rebuilt.state_dict()[k])

    def test_raises_on_missing_config(self):
        """A checkpoint without a config should raise ValueError."""
        with pytest.raises(ValueError, match="no usable 'config'"):
            _build_model_from_checkpoint({"state_dict": {}})

    def test_raises_on_missing_name(self):
        """A checkpoint config without 'name' should raise ValueError."""
        with pytest.raises(ValueError, match="no usable 'config'"):
            _build_model_from_checkpoint({"state_dict": {}, "config": {"scale": 4}})


class TestExportToTorchscript:
    """Tests for ``export_to_torchscript``."""

    def test_export_and_load(self, tmp_path):
        """A torchscript export should be loadable and produce valid output."""
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        state_dict = model.state_dict()
        checkpoint = {
            "state_dict": state_dict,
            "config": {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3},
        }
        tmp_ckpt = tmp_path / "model.pt"
        torch.save(checkpoint, tmp_ckpt)

        out_path = tmp_path / "model.ts"
        export_to_torchscript(tmp_ckpt, out_path)
        assert out_path.exists()

        loaded = torch.jit.load(str(out_path))
        dummy = torch.randn(1, 3, 16, 16)
        result = loaded(dummy)
        assert result.shape[0] == 1
        assert result.shape[1] == 3
        assert result.shape[2] >= 16
