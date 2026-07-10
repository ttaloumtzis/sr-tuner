"""Tests for device/backend.py — device detection and capability flags."""

import torch
import pytest

from sr_engine.device.backend import (
    get_device_name,
    get_device,
    is_rocm,
    autocast_dtype,
    supports_flash_attn,
)


class TestGetDeviceName:
    def test_cpu_when_cuda_unavailable(self, mock_torch_cuda):
        with mock_torch_cuda(available=False):
            assert get_device_name() == "cpu"

    def test_cuda_when_available(self, mock_torch_cuda):
        with mock_torch_cuda(available=True):
            assert get_device_name() == "cuda"


class TestGetDevice:
    def test_cpu_device(self, mock_torch_cuda):
        with mock_torch_cuda(available=False):
            device = get_device()
            assert device.type == "cpu"

    def test_cuda_device(self, mock_torch_cuda):
        with mock_torch_cuda(available=True):
            device = get_device()
            assert device.type == "cuda"


class TestIsRocm:
    def test_false_on_cpu(self, mock_torch_cuda):
        with mock_torch_cuda(available=False):
            assert is_rocm() is False

    def test_false_when_not_hip(self, mock_torch_cuda):
        with mock_torch_cuda(available=True, hip=False):
            assert is_rocm() is False

    def test_true_when_hip(self, mock_torch_cuda):
        with mock_torch_cuda(available=True, hip=True):
            assert is_rocm() is True


class TestAutocastDtype:
    def test_float32_on_cpu(self, mock_torch_cuda):
        with mock_torch_cuda(available=False):
            assert autocast_dtype() == torch.float32

    def test_bfloat16_when_supported(self, mock_torch_cuda):
        with mock_torch_cuda(available=True, bf16=True):
            assert autocast_dtype() == torch.bfloat16

    def test_float16_when_no_bf16(self, mock_torch_cuda):
        with mock_torch_cuda(available=True, bf16=False):
            assert autocast_dtype() == torch.float16


class TestSupportsFlashAttn:
    def test_false_on_cpu(self, mock_torch_cuda):
        with mock_torch_cuda(available=False):
            assert supports_flash_attn() is False
