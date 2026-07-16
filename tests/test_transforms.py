"""Tests for data/transforms.py — augmentation transforms."""

import torch
import pytest

from sr_engine.data.transforms import (
    RandomCrop,
    RandomFlip,
    RandomRotate,
    CenterCrop,
    Compose,
)


def _tensor(h=64, w=64, c=3):
    return torch.rand(c, h, w)


class TestRandomCrop:
    """Tests for RandomCrop."""

    def test_lr_crop_shape(self):
        """LR crop should match requested patch size."""
        lr = _tensor(64, 64)
        hr = _tensor(256, 256)
        t = RandomCrop(patch_size=32, scale=4)
        lr_out, hr_out = t(lr, hr)
        assert lr_out.shape[-2:] == (32, 32)

    def test_hr_crop_shape(self):
        """HR crop should be scaled by scale factor."""
        lr = _tensor(64, 64)
        hr = _tensor(256, 256)
        t = RandomCrop(patch_size=32, scale=4)
        lr_out, hr_out = t(lr, hr)
        assert hr_out.shape[-2:] == (128, 128)

    def test_raises_if_too_small(self):
        """Should raise ValueError if LR is smaller than patch_size."""
        lr = _tensor(16, 16)
        hr = _tensor(64, 64)
        t = RandomCrop(patch_size=32, scale=4)
        with pytest.raises(ValueError, match="smaller"):
            t(lr, hr)


class TestRandomFlip:
    """Tests for RandomFlip."""

    def test_no_flip_at_zero_prob(self):
        """No flip should occur when probabilities are 0."""
        lr = _tensor(32, 32)
        hr = _tensor(128, 128)
        t = RandomFlip(p_horizontal=0.0, p_vertical=0.0)
        lr_out, hr_out = t(lr.clone(), hr.clone())
        assert torch.equal(lr_out, lr)
        assert torch.equal(hr_out, hr)


class TestRandomRotate:
    """Tests for RandomRotate."""

    def test_zero_rotation(self):
        """Rotation by 0 should preserve identity."""
        lr = _tensor(32, 32)
        hr = _tensor(128, 128)
        t = RandomRotate(angles=[0])
        lr_out, hr_out = t(lr.clone(), hr.clone())
        assert torch.equal(lr_out, lr)
        assert torch.equal(hr_out, hr)

    def test_invalid_angle_raises(self):
        """Non-multiple-of-90 angles should raise ValueError."""
        with pytest.raises(ValueError, match="multiples of 90"):
            RandomRotate(angles=[45])


class TestCenterCrop:
    """Tests for CenterCrop."""

    def test_crop_shapes(self):
        """LR and HR crops should be properly scaled."""
        lr = _tensor(64, 64)
        hr = _tensor(256, 256)
        t = CenterCrop(patch_size=32, scale=4)
        lr_out, hr_out = t(lr, hr)
        assert lr_out.shape[-2:] == (32, 32)
        assert hr_out.shape[-2:] == (128, 128)

    def test_deterministic(self):
        """Identical inputs should produce identical crops."""
        lr = _tensor(64, 64)
        hr = _tensor(256, 256)
        t = CenterCrop(patch_size=32, scale=4)
        lr1, hr1 = t(lr.clone(), hr.clone())
        lr2, hr2 = t(lr.clone(), hr.clone())
        assert torch.equal(lr1, lr2)
        assert torch.equal(hr1, hr2)


class TestCompose:
    """Tests for Compose."""

    def test_applies_all_transforms(self):
        """All transforms should be applied in sequence."""
        lr = _tensor(64, 64)
        hr = _tensor(256, 256)
        t = Compose([
            CenterCrop(patch_size=32, scale=4),
            RandomFlip(p_horizontal=0.0, p_vertical=0.0),
        ])
        lr_out, hr_out = t(lr, hr)
        assert lr_out.shape[-2:] == (32, 32)

    def test_empty_compose(self):
        """Empty compose should pass through unchanged."""
        lr = _tensor(32, 32)
        hr = _tensor(128, 128)
        t = Compose([])
        lr_out, hr_out = t(lr, hr)
        assert torch.equal(lr_out, lr)
        assert torch.equal(hr_out, hr)
