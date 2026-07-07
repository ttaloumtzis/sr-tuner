"""Tests for data transforms."""

import torch

from sr_engine.data.transforms import CenterCrop, Compose, RandomCrop, RandomFlip, RandomRotate


class TestRandomCrop:
    def test_output_size(self):
        crop = RandomCrop(patch_size=48, scale=4)
        lr = torch.rand(3, 48, 48)
        hr = torch.rand(3, 192, 192)
        lr_out, hr_out = crop(lr, hr)
        assert lr_out.shape == (3, 48, 48)
        assert hr_out.shape == (3, 192, 192)

    def test_crop_aligns_lr_and_hr(self):
        import random
        random.seed(0)
        crop = RandomCrop(patch_size=16, scale=4)
        lr = torch.arange(3 * 64 * 64, dtype=torch.float32).view(3, 64, 64)
        hr = torch.arange(3 * 256 * 256, dtype=torch.float32).view(3, 256, 256)
        lr_out, hr_out = crop(lr, hr)
        assert lr_out.shape == (3, 16, 16)
        assert hr_out.shape == (3, 64, 64)


class TestRandomFlip:
    def test_horizontal_flip_changes_order(self):
        flip = RandomFlip(p_horizontal=1.0, p_vertical=0.0)
        lr = torch.arange(3 * 8 * 8, dtype=torch.float32).view(1, 3, 8, 8)
        hr = torch.arange(3 * 32 * 32, dtype=torch.float32).view(1, 3, 32, 32)
        lr_out, hr_out = flip(lr[0], hr[0])
        expected_lr = torch.flip(lr[0], dims=[-1])
        expected_hr = torch.flip(hr[0], dims=[-1])
        assert torch.allclose(lr_out, expected_lr)
        assert torch.allclose(hr_out, expected_hr)

    def test_vertical_flip(self):
        flip = RandomFlip(p_horizontal=0.0, p_vertical=1.0)
        lr = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)
        hr = torch.arange(3 * 32 * 32, dtype=torch.float32).view(3, 32, 32)
        lr_out, hr_out = flip(lr, hr)
        expected_lr = torch.flip(lr, dims=[-2])
        expected_hr = torch.flip(hr, dims=[-2])
        assert torch.allclose(lr_out, expected_lr)
        assert torch.allclose(hr_out, expected_hr)


class TestRandomRotate:
    def test_90_degree_rotation(self):
        rotate = RandomRotate(angles=[90])
        lr = torch.arange(3 * 8 * 8, dtype=torch.float32).view(3, 8, 8)
        hr = torch.arange(3 * 32 * 32, dtype=torch.float32).view(3, 32, 32)
        lr_out, hr_out = rotate(lr, hr)
        expected_lr = torch.rot90(lr, k=1, dims=[-2, -1])
        expected_hr = torch.rot90(hr, k=1, dims=[-2, -1])
        assert torch.allclose(lr_out, expected_lr)
        assert torch.allclose(hr_out, expected_hr)


class TestCenterCrop:
    def test_center_crop_output_size(self):
        crop = CenterCrop(patch_size=32, scale=4)
        lr = torch.rand(3, 64, 64)
        hr = torch.rand(3, 256, 256)
        lr_out, hr_out = crop(lr, hr)
        assert lr_out.shape == (3, 32, 32)
        assert hr_out.shape == (3, 128, 128)

    def test_center_crop_is_deterministic(self):
        crop = CenterCrop(patch_size=16, scale=2)
        lr = torch.rand(3, 64, 64)
        hr = torch.rand(3, 128, 128)
        a, b = crop(lr, hr)
        c, d = crop(lr, hr)
        assert torch.equal(a, c)
        assert torch.equal(b, d)


class TestCompose:
    def test_chains_all_transforms(self):
        pipeline = Compose([RandomCrop(patch_size=32, scale=4), RandomFlip(p_horizontal=1.0, p_vertical=0.0)])
        lr = torch.rand(3, 64, 64)
        hr = torch.rand(3, 256, 256)
        lr_out, hr_out = pipeline(lr, hr)
        assert lr_out.shape == (3, 32, 32)
        assert hr_out.shape == (3, 128, 128)
