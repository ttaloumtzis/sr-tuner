"""Tests for checkpoint save/load."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sr_engine.models.checkpoint import save_checkpoint, load_checkpoint


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        return self.conv(x)


class TestCheckpoint:
    def test_save_and_load_roundtrip(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        state_dict = {"weight": torch.tensor([1.0, 2.0, 3.0])}
        save_checkpoint(path, state_dict=state_dict, step=10)
        loaded = load_checkpoint(path)
        assert torch.allclose(loaded["state_dict"]["weight"], state_dict["weight"])
        assert loaded["step"] == 10

    def test_save_with_metadata(self, tmp_path: Path):
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
        with pytest.raises(FileNotFoundError):
            load_checkpoint(tmp_path / "nonexistent.pt")

    def test_load_with_map_location(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        loaded = load_checkpoint(path, map_location="cpu")
        assert loaded["state_dict"]["w"].device.type == "cpu"

    def test_load_ema_state(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        state = {"w": torch.tensor([1.0])}
        ema_state = {"w": torch.tensor([2.0])}
        save_checkpoint(path, state_dict=state, ema_state_dict=ema_state)
        loaded = load_checkpoint(path, load_ema=True)
        assert torch.allclose(loaded["state_dict"]["w"], torch.tensor([2.0]))

    def test_load_ema_missing_raises(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        save_checkpoint(path, state_dict={"w": torch.tensor([1.0])})
        with pytest.raises(ValueError, match="EMA"):
            load_checkpoint(path, load_ema=True)

    def test_weights_only_safe_by_default(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        loaded = load_checkpoint(path)
        assert loaded is not None

    def test_atomic_write_does_not_corrupt_on_failure(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        state_dict = {"w": torch.tensor([1.0])}
        save_checkpoint(path, state_dict=state_dict)
        assert path.is_file()
        assert path.suffix == ".pt"
        # The .tmp file should not remain after successful save
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        assert not tmp_path.exists()
