"""Tests for mixed precision training (BF16/FP16 AMP)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

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


class TestTrainerAmpInit:
    """Tests for AMP configuration in ``Trainer.__init__``."""

    def test_bf16_on_cpu_disables_amp(self, model_cfg, train_cfg, tmp_path):
        """BF16 on CPU should disable AMP entirely."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "bf16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        assert trainer.amp_enabled is False
        assert trainer.amp_dtype is None
        assert trainer.grad_scaler is None

    def test_float32_disables_amp(self, model_cfg, train_cfg, tmp_path):
        """Default dtype=float32 should disable AMP entirely."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        trainer = Trainer(model_cfg, train_cfg, d, device="cpu", validation_enabled=False)
        assert trainer.amp_enabled is False
        assert trainer.amp_dtype is None
        assert trainer.grad_scaler is None

    def test_missing_dtype_disables_amp(self, model_cfg, train_cfg, tmp_path):
        """Missing dtype key should default to float32 (no AMP)."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {k: v for k, v in train_cfg.items() if k != "dtype"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        assert trainer.amp_enabled is False
        assert trainer.amp_dtype is None

    def test_float16_on_cpu_disables_amp(self, model_cfg, train_cfg, tmp_path):
        """float16 on CPU should disable AMP."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "float16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        assert trainer.amp_enabled is False
        assert trainer.amp_dtype is None
        assert trainer.grad_scaler is None

    def _cuda_trainer(self, train_cfg, tmp_path, mock_torch_cuda, bf16_support=True):
        """Helper: create a Trainer with device='cuda' under mocked CUDA."""
        d = _create_dataset_dir(tmp_path, 3)
        dummy = torch.nn.Conv2d(3, 3, 3, padding=1)
        model_cfg = {"name": "test", "scale": 4}
        with patch("sr_engine.engine.trainer.build_model", return_value=dummy):
            with patch.object(torch.nn.Module, "to", return_value=dummy):
                with mock_torch_cuda(available=True, bf16=bf16_support):
                    return Trainer(model_cfg, train_cfg, d, device="cuda", validation_enabled=False)

    def test_bf16_on_cuda_enables_amp(self, model_cfg, train_cfg, tmp_path, mock_torch_cuda):
        """BF16 on CUDA with supported hardware should enable bfloat16 AMP."""
        tc = {**train_cfg, "dtype": "bf16"}
        trainer = self._cuda_trainer(tc, tmp_path, mock_torch_cuda, bf16_support=True)
        assert trainer.amp_enabled is True
        assert trainer.amp_dtype == torch.bfloat16
        assert trainer.grad_scaler is None

    def test_fp16_on_cuda_creates_scaler(self, model_cfg, train_cfg, tmp_path, mock_torch_cuda):
        """float16 on CUDA should enable fp16 AMP with GradScaler."""
        tc = {**train_cfg, "dtype": "float16"}
        trainer = self._cuda_trainer(tc, tmp_path, mock_torch_cuda, bf16_support=True)
        assert trainer.amp_enabled is True
        assert trainer.amp_dtype == torch.float16
        assert trainer.grad_scaler is not None

    def test_bf16_fallback_on_unsupported_hw(self, model_cfg, train_cfg, tmp_path, mock_torch_cuda):
        """BF16 on CUDA without HW support should fall back to autocast_dtype()."""
        tc = {**train_cfg, "dtype": "bf16"}
        trainer = self._cuda_trainer(tc, tmp_path, mock_torch_cuda, bf16_support=False)
        # Without bf16 support, autocast_dtype() should return float16
        assert trainer.amp_enabled is True
        assert trainer.amp_dtype == torch.float16
        assert trainer.grad_scaler is not None


class TestTrainerRunStepAmp:
    """Tests for ``_run_step`` under AMP."""

    def test_run_step_uses_autocast(self, model_cfg, train_cfg, tmp_path):
        """_run_step should call torch.autocast when AMP is enabled."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "bf16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        # Force AMP enabled for test
        trainer.amp_enabled = True
        trainer.amp_dtype = torch.bfloat16

        with patch("torch.autocast") as mock_autocast:
            mock_ctx = MagicMock()
            mock_autocast.return_value.__enter__.return_value = None
            mock_autocast.return_value.__exit__.return_value = None

            lr = torch.randn(2, 3, 16, 16)
            hr = torch.randn(2, 3, 64, 64)
            trainer._run_step(lr, hr)

        mock_autocast.assert_called_once_with(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=True,
        )

    def test_run_step_without_amp_no_autocast(self, model_cfg, train_cfg, tmp_path):
        """_run_step should NOT call torch.autocast when AMP is disabled."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        trainer = Trainer(model_cfg, train_cfg, d, device="cpu", validation_enabled=False)
        assert trainer.amp_enabled is False

        with patch("torch.autocast") as mock_autocast:
            lr = torch.randn(2, 3, 16, 16)
            hr = torch.randn(2, 3, 64, 64)
            trainer._run_step(lr, hr)

        mock_autocast.assert_called_once_with(
            device_type="cpu",
            dtype=None,
            enabled=False,
        )

    def test_run_step_returns_losses(self, model_cfg, train_cfg, tmp_path):
        """_run_step with AMP should return expected loss keys."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "bf16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        trainer.amp_enabled = True
        trainer.amp_dtype = torch.bfloat16

        lr = torch.randn(2, 3, 16, 16)
        hr = torch.randn(2, 3, 64, 64)
        losses = trainer._run_step(lr, hr)

        assert "pixel" in losses
        assert "total" in losses
        assert "lr" in losses

    def test_fp16_scaler_step_called(self, model_cfg, train_cfg, tmp_path):
        """_run_step with float16 should call GradScaler scale/step/update."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "float16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        trainer.amp_enabled = True
        trainer.amp_dtype = torch.float16
        trainer.grad_scaler = MagicMock()

        lr = torch.randn(2, 3, 16, 16)
        hr = torch.randn(2, 3, 64, 64)
        trainer._run_step(lr, hr)

        trainer.grad_scaler.scale.assert_called_once()
        trainer.grad_scaler.step.assert_called_once_with(trainer.optimizer)
        trainer.grad_scaler.update.assert_called_once()


class TestTrainerCheckpointAmp:
    """Tests that checkpoints record training dtype."""

    def test_save_records_training_dtype(self, model_cfg, train_cfg, tmp_path):
        """Checkpoint should include training_dtype in config."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "checkpoint_dir": str(tmp_path / "ckpts"), "dtype": "bf16"}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)
        trainer.amp_enabled = True
        trainer.amp_dtype = torch.bfloat16

        with patch("sr_engine.engine.trainer.save_checkpoint") as mock_save:
            trainer._save(epoch=1)

        args, kwargs = mock_save.call_args
        saved_config = kwargs.get("config", args[1] if len(args) > 1 else {})
        assert "training_dtype" in saved_config
        assert saved_config["training_dtype"] == "torch.bfloat16"

    def test_save_without_amp_records_none(self, model_cfg, train_cfg, tmp_path):
        """Checkpoint without AMP should record training_dtype as None."""
        # _create_dataset_dir defined at module level above
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "checkpoint_dir": str(tmp_path / "ckpts")}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=False)

        with patch("sr_engine.engine.trainer.save_checkpoint") as mock_save:
            trainer._save(epoch=1)

        args, kwargs = mock_save.call_args
        saved_config = kwargs.get("config", args[1] if len(args) > 1 else {})
        assert "training_dtype" in saved_config
        assert saved_config["training_dtype"] == "None"


class TestTrainerValidateAmp:
    """Tests for validation under AMP."""

    def test_validate_uses_autocast(self, model_cfg, train_cfg, tmp_path):
        """Validation forward pass should use autocast when AMP enabled."""
        d = _create_dataset_dir(tmp_path, 3)
        cfg = {**train_cfg, "dtype": "bf16", "validation": {"enabled": True, "split": 0.5}}
        trainer = Trainer(model_cfg, cfg, d, device="cpu", validation_enabled=True, validation_split=0.5)
        trainer.amp_enabled = True
        trainer.amp_dtype = torch.bfloat16

        with patch("torch.autocast") as mock_autocast:
            with patch.object(trainer, "val_dataloader", [(
                torch.randn(1, 3, 16, 16),
                torch.randn(1, 3, 64, 64),
            )]):
                trainer._validate(epoch=1)

        mock_autocast.assert_called_with(
            device_type="cpu",
            dtype=torch.bfloat16,
            enabled=True,
        )
