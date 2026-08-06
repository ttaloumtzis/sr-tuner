"""Tests for the Trainer class."""

import pickle
import random
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from sr_engine.engine.trainer import Trainer, TrainerCallback, TrainingCancelled, _TransformSubset
from sr_engine.engine.inference import CancellationRequested, _super_resolve_tensor
from sr_engine.models.registry import build_model


def _make_image(path: Path, w: int = 64, h: int = 64) -> None:
    """Write a random RGB image to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _create_dataset_dir(tmp_path: Path, num_pairs: int = 5) -> Path:
    """Create a temporary HR/LR dataset directory (4x scale, square images)."""
    d = tmp_path / "dataset"
    for i in range(num_pairs):
        _make_image(d / "HR" / f"frame_{i:04d}.png", w=256, h=256)
        _make_image(d / "LR" / f"frame_{i:04d}.png", w=64, h=64)
    return d


@pytest.fixture
def model_cfg():
    """Return a minimal RRDB model config."""
    return {"name": "rrdb_esrgan", "scale": 4}


@pytest.fixture
def train_cfg():
    """Return a minimal training config for CPU-based tests."""
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


class TestTrainerWorkerInit:
    """Tests for the DataLoader ``worker_init_fn``."""

    def test_worker_init_fn_is_picklable(self, model_cfg, train_cfg, tmp_path):
        """worker_init_fn must survive pickling — spawned DataLoader workers pickle it."""
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg={**train_cfg, "num_workers": 4, "seed": 42},
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        restored = pickle.loads(pickle.dumps(trainer._worker_init_fn))
        assert callable(restored)

    def test_worker_init_fn_seeds_deterministically(self, model_cfg, train_cfg, tmp_path):
        """Each worker id should get a distinct, seed-derived Python RNG state."""
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg={**train_cfg, "num_workers": 4, "seed": 7},
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )

        random.seed(0)
        trainer._worker_init_fn(0)
        first = random.random()
        random.seed(0)
        trainer._worker_init_fn(0)
        assert random.random() == first
        trainer._worker_init_fn(1)
        assert random.random() != first


class TestTrainerInit:
    """Tests for ``Trainer.__init__`` — dataset splitting."""

    def test_validation_disabled_uses_all_data(self, model_cfg, train_cfg, tmp_path):
        """All dataset pairs should be used for training when validation is off."""
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
        """Data should be split between train and validation sets."""
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
        """A validation split of 0 should disable validation entirely."""
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


class TestTrainerSplitSeed:
    """Tests for the independent train/validation split seed."""

    def _split_indices(self, model_cfg, tmp_path, seed, split_seed):
        d = _create_dataset_dir(tmp_path, num_pairs=10)
        cfg = {
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patch_size": 16,
            "seed": seed,
            "checkpoint_dir": "checkpoints",
            "losses": {"perceptual_weight": 0.0},
            "validation": {"enabled": True, "split": 0.5, "split_seed": split_seed},
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
        )
        assert isinstance(trainer.train_dataset, _TransformSubset)
        assert isinstance(trainer.val_dataset, _TransformSubset)
        return list(trainer.train_dataset.indices), list(trainer.val_dataset.indices)

    def test_split_stable_across_general_seeds(self, model_cfg, tmp_path):
        """Changing the general seed must NOT reshuffle the train/validation split."""
        train_a, val_a = self._split_indices(model_cfg, tmp_path, seed=7, split_seed=1234)
        train_b, val_b = self._split_indices(model_cfg, tmp_path, seed=8, split_seed=1234)
        assert train_a == train_b
        assert val_a == val_b

    def test_split_changes_with_split_seed(self, model_cfg, tmp_path):
        """Changing the split seed must produce a different train/validation split."""
        train_a, val_a = self._split_indices(model_cfg, tmp_path, seed=7, split_seed=1234)
        train_c, val_c = self._split_indices(model_cfg, tmp_path, seed=7, split_seed=999)
        assert train_a != train_c or val_a != val_c


class TestTrainerValidate:
    """Tests for ``Trainer._validate`` — full-set validation."""

    def test_validate_runs_full_pass_on_all_val_images(self, model_cfg, train_cfg, tmp_path):
        """Full-image SR should run once per validation image and average the metrics."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
        )
        assert trainer.val_dataset is not None
        val_count = len(trainer.val_dataset)
        assert val_count >= 1

        captured: list[torch.Tensor] = []

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            captured.append(lr_t)
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            result = trainer._validate(epoch=1)

        assert len(captured) == val_count
        assert result["full_psnr"] is not None
        assert result["full_ssim"] is not None

    def test_validate_saves_frames_only_for_first_image(self, model_cfg, train_cfg, tmp_path):
        """Display frames should be saved only for the first validation image."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        frame_dir = tmp_path / "frames"
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
            validation_frame_dir=frame_dir,
        )
        assert trainer.val_dataset is not None

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            result = trainer._validate(epoch=1)

        epoch_dir = frame_dir / "epoch_001"
        for name in ("lr.png", "sr.png", "gt.png", "diff.png"):
            assert (epoch_dir / name).exists()
        assert result["frames"]["lrPath"] == str((epoch_dir / "lr.png").resolve())
        assert result["full_psnr"] is not None

    def test_validate_first_image_is_deterministic(self, model_cfg, train_cfg, tmp_path):
        """The displayed first image must be the same across validation runs."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        frame_dir = tmp_path / "frames"
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
            validation_frame_dir=frame_dir,
        )

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            trainer._validate(epoch=1)
            first_lr = (frame_dir / "epoch_001" / "lr.png").read_bytes()
            first_gt = (frame_dir / "epoch_001" / "gt.png").read_bytes()
            trainer._validate(epoch=2)
            second_lr = (frame_dir / "epoch_002" / "lr.png").read_bytes()
            second_gt = (frame_dir / "epoch_002" / "gt.png").read_bytes()

        assert first_lr == second_lr
        assert first_gt == second_gt

    def test_validate_reports_val_loss(self, model_cfg, train_cfg, tmp_path):
        """Validation should compute and return a composite ``val_loss``."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
        )

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            result = trainer._validate(epoch=1)

        assert "val_loss" in result
        assert result["val_loss"] > 0

    def test_validate_full_image_pass_capped_by_limit(self, model_cfg, train_cfg, tmp_path):
        """The tiled full-image pass should be capped at ``full_image_limit``."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234, "full_image_limit": 2},
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
        )
        assert trainer.val_dataset is not None

        captured: list[torch.Tensor] = []

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            captured.append(lr_t)
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            result = trainer._validate(epoch=1)

        assert len(captured) == 2
        assert result["full_psnr"] is not None

    def test_validate_emits_progress_events(self, model_cfg, train_cfg, tmp_path):
        """Validation should emit a progress event after each processed image."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        events: list[tuple[int, int, int]] = []

        class Recorder(TrainerCallback):
            def on_validate_progress(self, epoch: int, done: int, total: int) -> None:
                events.append((epoch, done, total))

        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
            callbacks=[Recorder()],
        )
        assert trainer.val_dataset is not None
        val_count = len(trainer.val_dataset)

        def fake_super_resolve(**kwargs):
            lr_t = kwargs["lr_tensor"]
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            trainer._validate(epoch=1)

        batch_size = int(cfg["batch_size"])
        import math
        patch_events = math.ceil(val_count / batch_size)
        full_events = min(val_count, 8)
        total = val_count + full_events
        assert len(events) == patch_events + full_events
        assert events[0][0] == 1
        assert events[0][2] == total
        assert events[-1][1] == total
        assert all(e[0] == 1 for e in events)
        assert all(e[2] == total for e in events)


class TestTrainRunStep:
    """Tests for ``Trainer._run_step``."""

    def test_run_step_returns_loss_dict(self, model_cfg, train_cfg, tmp_path):
        """Calling ``_run_step`` should return a dict with pixel and total loss."""
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


class TestOptimizerConfig:
    """Tests for optimizer creation with config values."""

    def test_optimizer_receives_weight_decay_and_betas(self, model_cfg, tmp_path):
        """The Adam optimizer should receive weight_decay and betas from config."""
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        cfg = {
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patch_size": 16,
            "seed": 42,
            "checkpoint_dir": "checkpoints",
            "losses": {"perceptual_weight": 0.0},
            "validation": {"enabled": False},
            "weight_decay": 0.1,
            "betas": [0.8, 0.888],
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.1
        assert list(trainer.optimizer.param_groups[0]["betas"]) == [0.8, 0.888]

    def test_optimizer_defaults_when_config_missing(self, model_cfg, tmp_path):
        """The Adam optimizer should use default values when config has no weight_decay/betas."""
        d = _create_dataset_dir(tmp_path, num_pairs=5)
        cfg = {
            "max_epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "patch_size": 16,
            "seed": 42,
            "checkpoint_dir": "checkpoints",
            "losses": {"perceptual_weight": 0.0},
            "validation": {"enabled": False},
        }
        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 0.0
        assert list(trainer.optimizer.param_groups[0]["betas"]) == [0.9, 0.99]


class TestGradientCheckpointing:
    """Tests for gradient checkpointing parity."""

    def test_swinir_checkpointing_off_and_on_match(self):
        """SwinIR forward must produce identical outputs with checkpointing off/on."""
        model_cfg = {"name": "swinir", "embed_dim": 48, "depths": [2, 2], "num_heads": [3, 3],
                     "window_size": 4, "scale": 1, "num_in_ch": 3, "num_out_ch": 3}
        model = build_model("swinir", model_cfg).eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out_off = model(x)
        model.gradient_checkpointing = True
        model.train()
        out_on = model(x)
        assert torch.allclose(out_off, out_on, atol=1e-5)

    def test_rrdb_checkpointing_off_and_on_match(self):
        """RRDBNet forward must produce identical outputs with checkpointing off/on."""
        model_cfg = {"name": "rrdb_esrgan", "num_feat": 16, "num_block": 2, "num_grow_ch": 8,
                     "scale": 1, "num_in_ch": 3, "num_out_ch": 3}
        model = build_model("rrdb_esrgan", model_cfg).eval()
        x = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            out_off = model(x)
        model.gradient_checkpointing = True
        model.train()
        out_on = model(x)
        assert torch.allclose(out_off, out_on, atol=1e-5)

    def test_trainer_auto_enables_for_swinir(self, tmp_path):
        """gradient_checkpointing=auto should enable for swinir, disable for rrdb."""
        d = _create_dataset_dir(tmp_path, num_pairs=3)
        swin_cfg = {"name": "swinir", "embed_dim": 48, "depths": [2, 2], "num_heads": [3, 3],
                    "window_size": 4, "scale": 1}
        trainer = Trainer(
            model_cfg=swin_cfg,
            train_cfg={"max_epochs": 1, "batch_size": 1, "num_workers": 0, "patch_size": 16,
                       "seed": 42, "checkpoint_dir": "checkpoints", "losses": {"perceptual_weight": 0.0},
                       "validation": {"enabled": False}},
            dataset_dir=d,
            device="cpu",
            validation_enabled=False,
        )
        assert trainer.model.gradient_checkpointing is True


class TestValidateCancel:
    """Tests for cancellation responsiveness inside validation."""

    def test_super_resolve_aborts_on_cancel(self):
        """_super_resolve_tensor should raise CancellationRequested when cancel_check returns True."""
        model_cfg = {"name": "rrdb_esrgan", "num_feat": 8, "num_block": 1, "num_grow_ch": 4,
                     "scale": 1}
        model = build_model("rrdb_esrgan", model_cfg).eval()
        lr_t = torch.rand(3, 128, 128)
        with pytest.raises(CancellationRequested):
            _super_resolve_tensor(
                model=model, lr_tensor=lr_t, scale=1,
                tile_size=32, tile_overlap=4, device="cpu",
                cancel_check=lambda: True,
            )

    def test_super_resolve_ignores_cancel_when_false(self):
        """cancel_check returning False should not abort the pass."""
        model_cfg = {"name": "rrdb_esrgan", "num_feat": 8, "num_block": 1, "num_grow_ch": 4,
                     "scale": 1}
        model = build_model("rrdb_esrgan", model_cfg).eval()
        lr_t = torch.rand(3, 64, 64)
        out = _super_resolve_tensor(
            model=model, lr_tensor=lr_t, scale=1,
            tile_size=32, tile_overlap=4, device="cpu",
            cancel_check=lambda: False,
        )
        assert out.shape == (3, 64, 64)

    def test_validate_raises_when_cancelled(self, model_cfg, train_cfg, tmp_path):
        """_validate must raise TrainingCancelled promptly when cancel_check returns True."""
        d = _create_dataset_dir(tmp_path, num_pairs=8)
        cfg = {
            **train_cfg,
            "validation": {"enabled": True, "split": 0.5, "split_seed": 1234},
        }
        cancelled = {"flag": False}

        def cancel_check() -> bool:
            return cancelled["flag"]

        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=cfg,
            dataset_dir=d,
            device="cpu",
            validation_enabled=True,
            validation_split=0.5,
            cancel_check=cancel_check,
        )

        def fake_super_resolve(**kwargs):
            cancelled["flag"] = True
            lr_t = kwargs["lr_tensor"]
            scale = kwargs["scale"]
            return torch.rand(3, lr_t.shape[1] * scale, lr_t.shape[2] * scale)

        with patch("sr_engine.engine.trainer._super_resolve_tensor",
                   side_effect=fake_super_resolve):
            with pytest.raises(TrainingCancelled):
                trainer._validate(epoch=1)
