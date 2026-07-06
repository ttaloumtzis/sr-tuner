"""Tests for checkpoint save/load."""

from pathlib import Path

import pytest

from sr_engine.models.checkpoint import save_checkpoint, load_checkpoint


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_checkpoint(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "model.pt"
        with pytest.raises(NotImplementedError):
            save_checkpoint(ckpt_path, state_dict={})

    def test_load_checkpoint(self, tmp_path: Path) -> None:
        ckpt_path = tmp_path / "model.pt"
        with pytest.raises(NotImplementedError):
            load_checkpoint(ckpt_path)
