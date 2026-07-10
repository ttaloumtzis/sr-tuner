"""Tests for data/video_extract.py — video frame extraction."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from sr_engine.data.video_extract import extract_frames


class TestExtractFrames:
    def test_raises_on_missing_video(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Could not open video"):
            extract_frames(
                video_path=tmp_path / "nonexistent.mp4",
                out_dir=tmp_path / "frames",
            )

    def test_extracts_all_frames(self, tmp_path, sample_video):
        out = tmp_path / "frames"
        result = extract_frames(video_path=sample_video, out_dir=out)
        assert len(result) == 10
        assert all(p.exists() for p in result)

    def test_frame_rate_lower_than_video(self, tmp_path, sample_video):
        out = tmp_path / "frames_sub"
        result = extract_frames(
            video_path=sample_video,
            out_dir=out,
            frame_rate=15,
        )
        assert 0 < len(result) <= 10

    def test_frame_rate_higher_than_video(self, tmp_path, sample_video):
        out = tmp_path / "frames_high"
        result = extract_frames(
            video_path=sample_video,
            out_dir=out,
            frame_rate=60,
        )
        assert len(result) == 10

    def test_start_time(self, tmp_path, sample_video):
        out = tmp_path / "frames_start"
        result = extract_frames(
            video_path=sample_video,
            out_dir=out,
            start_time=0.1,
        )
        assert len(result) < 10
        assert len(result) >= 7

    def test_duration(self, tmp_path, sample_video):
        out = tmp_path / "frames_dur"
        result = extract_frames(
            video_path=sample_video,
            out_dir=out,
            duration=0.1,
        )
        assert 1 <= len(result) <= 4

    def test_calls_reporter(self, tmp_path, sample_video):
        out = tmp_path / "frames_rep"
        reporter = MagicMock()
        result = extract_frames(
            video_path=sample_video,
            out_dir=out,
            reporter=reporter,
        )
        reporter.start.assert_called_once()
        reporter.finish.assert_called_once()
