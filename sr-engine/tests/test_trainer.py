"""Tests for the Trainer class."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from sr_engine.engine.trainer import Trainer


def _make_image(path: Path, w: int = 64, h: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_dataset_dir(tmp_path: Path, num_pairs: int = 5) -> Path:
    d = tmp_path / "dataset"
    for i in range(num_pairs):
        _make_image(d / "HR" / f"frame_{i:04d}.png", w=256)
        _make_image(d / "LR" / f"frame_{i:04d}.png", w=64)
    return d


@pytest.fixture
def model_cfg():
    return {"name": "rrdb_esrgan", "scale": 4}


@pytest.fixture
def train_cfg():
    return {
        "max_epochs": 2,
        "save_per_epoch": 1,
        "batch_size": 2,
        "num_workers": 0,
        "patch_size": 16,
        "seed": 42,
        "checkpoint_dir": "checkpoints",
        "losses": {"perceptual_weight": 0.0},
        "validation": {"enabled": False},
    }


class TestTrainerInit:
    def test_validation_disabled_uses_all_data(self, model_cfg, train_cfg, tmp_path):
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        assert trainer.val_dataset is None
        assert trainer.val_dataloader is None
        assert len(trainer.train_dataset) == 5

    def test_validation_enabled_splits_data(self, model_cfg, train_cfg, tmp_path):
        d = _create_dataset_dir(tmp_path, num_pairs=10)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
        )
        assert trainer.val_dataset is not None
        assert len(trainer.val_dataset) > 0
        assert trainer.val_dataloader is not None

    def test_validation_split_zero_disables_val(self, model_cfg, train_cfg, tmp_path):
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.0,
        )
        assert trainer.val_dataset is None


class TestTrainRunStep:
    def test_run_step_returns_loss_dict(self, model_cfg, train_cfg, tmp_path):
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        lr = torch.randn(2, 3, 16, 16)
        hr = torch.randn(2, 3, 64, 64)
        losses = trainer._run_step(lr, hr)
        assert "pixel" in losses
        assert "total" in losses
        assert losses["pixel"] > 0.0
        assert losses["total"] > 0.0
