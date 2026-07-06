"""Tests for inference."""

from pathlib import Path

import pytest

from sr_engine.engine.inference import infer_image, infer_video


class TestInference:
    """Tests for inference functions."""

    def test_infer_image(self, tmp_path: Path) -> None:
        dummy_ckpt = tmp_path / "model.pt"
        dummy_input = tmp_path / "input.png"
        dummy_output = tmp_path / "output.png"
        # Create dummy files so paths exist
        dummy_ckpt.touch()
        dummy_input.touch()
        with pytest.raises(NotImplementedError):
            infer_image(dummy_ckpt, dummy_input, dummy_output)

    def test_infer_video(self, tmp_path: Path) -> None:
        dummy_ckpt = tmp_path / "model.pt"
        dummy_input = tmp_path / "input.mp4"
        dummy_output = tmp_path / "output.mp4"
        dummy_ckpt.touch()
        dummy_input.touch()
        with pytest.raises(NotImplementedError):
            infer_video(dummy_ckpt, dummy_input, dummy_output)
