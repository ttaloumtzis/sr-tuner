"""Tests for the hardware monitoring module (nvidia-smi / rocm-smi parsing)."""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from sr_engine.monitoring.hardware import _get_gpu_stats


def _mock_nvidia_smi(stdout: str) -> MagicMock:
    """Patch shutil.which + subprocess.run to simulate nvidia-smi output."""
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.returncode = 0
    proc.stdout = stdout
    proc.stderr = ""
    return proc


class TestGetGpuStats:
    """Tests for _get_gpu_stats() — mocked to avoid real GPU dependencies."""

    @patch("sr_engine.monitoring.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("sr_engine.monitoring.hardware.subprocess.run")
    def test_dot_decimal_us_locale(self, mock_run, mock_which):
        """US locale: nvidia-smi outputs comma-space-delimited CSV with dot decimals."""
        mock_run.return_value = _mock_nvidia_smi("42.5, 1234, 8192, 68\n")

        util, vram_used, vram_total, temp = _get_gpu_stats()

        assert util == 42.5
        assert vram_used == pytest.approx(1234 / 1024)
        assert vram_total == pytest.approx(8192 / 1024)
        assert temp == 68.0

    @patch("sr_engine.monitoring.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("sr_engine.monitoring.hardware.subprocess.run")
    def test_comma_decimal_european_locale(self, mock_run, mock_which):
        """European locale: nvidia-smi outputs comma as decimal separator.

        The fix in hardware.py uses .replace(",", ".") to normalize before float().
        """
        mock_run.return_value = _mock_nvidia_smi("42,5, 1234, 8192, 68\n")

        util, vram_used, vram_total, temp = _get_gpu_stats()

        assert util == 42.5
        assert vram_used == pytest.approx(1234 / 1024)
        assert vram_total == pytest.approx(8192 / 1024)
        assert temp == 68.0

    @patch("sr_engine.monitoring.hardware.shutil.which", return_value=None)
    def test_no_gpu_fallback(self, mock_which):
        """No GPU tools found — all values should be None."""
        util, vram_used, vram_total, temp = _get_gpu_stats()

        assert util is None
        assert vram_used is None
        assert vram_total is None
        assert temp is None

    @patch("sr_engine.monitoring.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("sr_engine.monitoring.hardware.subprocess.run")
    def test_nvidia_smi_nonzero_exit(self, mock_run, mock_which):
        """nvidia-smi returns non-zero exit code — should fall through."""
        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 1
        proc.stdout = ""
        proc.stderr = "error"
        mock_run.return_value = proc

        util, vram_used, vram_total, temp = _get_gpu_stats()

        assert util is None
        assert vram_used is None
        assert vram_total is None
        assert temp is None

    @patch("sr_engine.monitoring.hardware.shutil.which", return_value="/usr/bin/nvidia-smi")
    @patch("sr_engine.monitoring.hardware.subprocess.run")
    def test_malformed_output(self, mock_run, mock_which):
        """nvidia-smi returns garbled output — should fall through gracefully."""
        proc = MagicMock(spec=subprocess.CompletedProcess)
        proc.returncode = 0
        proc.stdout = "not a csv, at all\n"
        proc.stderr = ""
        mock_run.return_value = proc

        util, vram_used, vram_total, temp = _get_gpu_stats()

        assert util is None
        assert vram_used is None
        assert vram_total is None
        assert temp is None
