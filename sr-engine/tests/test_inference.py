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
    """Tests for image-to-tensor and tensor-to-image conversion functions."""

    def test_read_image_tensor_shape(self):
        """``_read_image_tensor`` should return a (3, H, W) tensor in [0, 1]."""
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
        """``_frame_to_tensor`` should convert BGR to RGB channel order."""
        import numpy as np
        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255
        bgr[:, :, 1] = 0
        bgr[:, :, 2] = 128
        tensor = _frame_to_tensor(bgr)
        assert tensor[0, 0, 0].item() == pytest.approx(128.0 / 255.0, abs=1e-6)
        assert tensor[1, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert tensor[2, 0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_tensor_to_bgr_roundtrip(self):
        """``_tensor_to_bgr_image`` should produce a uint8 BGR image."""
        import numpy as np
        tensor = torch.rand(3, 64, 64)
        bgr = _tensor_to_bgr_image(tensor)
        assert bgr.shape == (64, 64, 3)
        assert bgr.dtype == np.uint8


class TestSuperResolveTensor:
    """Tests for ``_super_resolve_tensor`` — tiled and non-tiled inference."""

    def test_no_tiling_small_image(self):
        """A small image should be processed without tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        result = _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_tiling_large_image(self):
        """A large image should be processed with tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 32, 32)
        lr = torch.randn(3, 32, 32)
        result = _super_resolve_tensor(model, lr, scale=2, tile_size=16, tile_overlap=4, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_model_called_once_without_tiling(self):
        """The model should be called exactly once without tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert model.call_count == 1


class TestTiling:
    """Tests for tile/stitch roundtrip."""

    def test_tile_and_stitch_roundtrip(self):
        """Tiling then stitching should produce the original shape."""
        from sr_engine.engine.tiling import tile_image, stitch_tiles
        import numpy as np
        lr = torch.randn(3, 64, 64)
        tiles = tile_image(lr, tile_size=32, overlap=8)
        assert len(tiles) > 1
        stitched = stitch_tiles(tiles, output_size=(64, 64), overlap=8)
        assert stitched.shape == (3, 64, 64)


class TestLoadModel:
    """Tests for ``_load_model``."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        config = {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3}
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
        }, ckpt)
        return ckpt

    def test_raises_on_missing_config(self, tmp_path):
        """A checkpoint without a config should raise ValueError."""
        from sr_engine.engine.inference import _load_model
        ckpt = tmp_path / "model.pt"
        torch.save({"state_dict": {"w": torch.tensor([1.0])}}, ckpt)
        with pytest.raises(ValueError, match="no usable 'config'"):
            _load_model(ckpt, device="cpu")

    def test_loads_model_from_checkpoint(self, tmp_path):
        """A valid checkpoint should produce an nn.Module and scale."""
        from sr_engine.engine.inference import _load_model
        import torch.nn as nn
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        loaded_model, scale = _load_model(ckpt, device="cpu")
        assert scale == 4
        assert isinstance(loaded_model, nn.Module)
        loaded_model.eval()


class TestInferImage:
    """Tests for the ``infer_image`` entry point."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        config = {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3}
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
        }, ckpt)
        return ckpt

    def test_saves_output(self, tmp_path, sample_image):
        """infer_image should produce an output image at the specified path."""
        from sr_engine.engine.inference import infer_image
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        out_path = tmp_path / "output.png"
        result = infer_image(
            model_checkpoint=ckpt,
            input_path=sample_image,
            output_path=out_path,
            device="cpu",
        )
        assert result == out_path
        assert out_path.exists()
