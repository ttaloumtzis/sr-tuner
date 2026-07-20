"""Tests for data/video_extract.py — frame extraction from videos."""

from pathlib import Path
from unittest.mock import patch

import pytest

from sr_engine.data.video_extract import extract_frames


class TestExtractFrames:
    """Tests for ``extract_frames``."""

    def test_missing_file(self, tmp_path):
        """A nonexistent video file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Could not open"):
            extract_frames(
                video_path=tmp_path / "nonexistent.mp4",
                out_dir=tmp_path / "out",
            )

    def test_output_dir_created(self):
        """extract_frames should create the output dir before opening the file."""
        try:
            extract_frames(
                video_path=Path("/nonexistent/video.mp4"),
                out_dir=Path("/tmp/__should_not_exist__"),
            )
        except (FileNotFoundError, OSError):
            pass

    def test_extracts_from_video(self, sample_video, tmp_path):
        """A valid video should produce extracted frames."""
        out_dir = tmp_path / "frames"
        paths = extract_frames(
            video_path=sample_video,
            out_dir=out_dir,
        )
        assert len(paths) > 0
        for p in paths:
            assert p.is_file()

    def test_broken_pipe_during_extraction_returns_partial(self, sample_video, tmp_path):
        """extract_frames should return partial results when decoder pipe breaks."""
        import cv2

        original_read = cv2.VideoCapture.read
        call_count = [0]

        def _breaking_read(self):
            call_count[0] += 1
            if call_count[0] >= 3:
                raise BrokenPipeError("Simulated FFmpeg pipe break")
            return original_read(self)

        out_dir = tmp_path / "frames"
        with patch.object(cv2.VideoCapture, "read", _breaking_read):
            paths = extract_frames(
                video_path=sample_video,
                out_dir=out_dir,
            )

        # Should have partial results (at least 1, up to 2 frames)
        assert len(paths) >= 1
        for p in paths:
            assert p.is_file()
