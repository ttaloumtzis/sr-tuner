"""Tests for inference."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from sr_engine.engine.inference import (
    _read_image_tensor,
    _frame_to_tensor,
    _tensor_to_bgr_image,
    _super_resolve_tensor,
)


class TestImageTensorConversion:
    def test_read_image_tensor_shape(self):
        # Use a simple generated image via numpy/cv2
        import cv2
        import numpy as np
        import tempfile
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            cv2.imwrite(f.name, img)
            tensor = _read_image_tensor(Path(f.name))
        assert tensor.shape == (3, 64, 64)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_frame_to_tensor_converts_bgr_to_rgb(self):
        import numpy as np
        # BGR frame from cv2
        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        # Make it asymmetric in B,G,R so we can detect channel swap
        bgr[:, :, 0] = 255  # Blue channel full
        bgr[:, :, 1] = 0    # Green channel zero
        bgr[:, :, 2] = 128  # Red channel mid
        tensor = _frame_to_tensor(bgr)
        # Channel order should be RGB now
        assert tensor[0, 0, 0].item() == pytest.approx(128.0 / 255.0, abs=1e-6)  # Red
        assert tensor[1, 0, 0].item() == pytest.approx(0.0, abs=1e-6)            # Green
        assert tensor[2, 0, 0].item() == pytest.approx(1.0, abs=1e-6)            # Blue

    def test_tensor_to_bgr_roundtrip(self):
        import numpy as np
        tensor = torch.rand(3, 64, 64)
        bgr = _tensor_to_bgr_image(tensor)
        assert bgr.shape == (64, 64, 3)
        assert bgr.dtype == np.uint8


class TestSuperResolveTensor:
    def test_no_tiling_small_image(self):
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        result = _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_tiling_large_image(self):
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 32, 32)
        lr = torch.randn(3, 32, 32)
        result = _super_resolve_tensor(model, lr, scale=2, tile_size=16, tile_overlap=4, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_model_called_once_without_tiling(self):
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert model.call_count == 1


class TestTiling:
    def test_tile_and_stitch_roundtrip(self):
        from sr_engine.engine.tiling import tile_image, stitch_tiles
        import numpy as np
        lr = torch.randn(3, 64, 64)
        tiles = tile_image(lr, tile_size=32, overlap=8)
        assert len(tiles) > 1  # Should produce multiple tiles
        stitched = stitch_tiles(tiles, output_size=(64, 64), overlap=8)
        assert stitched.shape == (3, 64, 64)
