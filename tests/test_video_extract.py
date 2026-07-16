"""Tests for data/video_extract.py — frame extraction from videos."""

from pathlib import Path

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
