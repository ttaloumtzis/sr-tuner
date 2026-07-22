"""Tests for data/video_extract.py — frame extraction from videos."""

import json
import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sr_engine.data.ffmpeg_extractor import CancelledError
from sr_engine.data.video_extract import extract_frames


def _make_ffprobe_json(**overrides: str | int) -> str:
    data = {
        "streams": [{
            "codec_name": overrides.get("codec_name", "h264"),
            "pix_fmt": overrides.get("pix_fmt", "yuv420p"),
            "bit_depth": int(overrides.get("bit_depth", 8)),
            "r_frame_rate": overrides.get("r_frame_rate", "30/1"),
            "nb_frames": overrides.get("nb_frames", "10"),
            "duration": overrides.get("duration", "0.333"),
            "width": int(overrides.get("width", 64)),
            "height": int(overrides.get("height", 64)),
        }]
    }
    return json.dumps(data)


class TestExtractFrames:
    """Tests for ``extract_frames``."""

    def test_missing_file(self, tmp_path):
        """A nonexistent video file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Could not open"):
            extract_frames(
                video_path=tmp_path / "nonexistent.mp4",
                out_dir=tmp_path / "out",
            )

    def test_missing_ffmpeg(self, tmp_path):
        """If ffmpeg/ffprobe is not in PATH, raise FileNotFoundError."""
        video = tmp_path / "video.mp4"
        video.touch()
        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="ffprobe not found"):
                extract_frames(
                    video_path=video,
                    out_dir=tmp_path / "out",
                )

    def test_output_dir_created(self, tmp_path):
        """extract_frames should create the output dir before running ffmpeg."""
        video = tmp_path / "video.mp4"
        video.touch()
        out_dir = tmp_path / "out"
        ffprobe_result = MagicMock(stdout=_make_ffprobe_json(), returncode=0)
        with (
            patch("shutil.which", return_value="/usr/bin/ffprobe"),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.run",
                  return_value=ffprobe_result),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = 0
            proc_mock.wait.return_value = 0
            proc_mock.returncode = 0
            proc_mock.communicate.return_value = (None, "")
            mock_popen.return_value = proc_mock
            try:
                extract_frames(
                    video_path=video,
                    out_dir=out_dir,
                )
            except (FileNotFoundError, OSError):
                pass
            assert out_dir.exists()

    def test_extracts_from_video(self, sample_video, tmp_path):
        """A valid video should produce extracted frames via FFmpeg."""
        import shutil
        if shutil.which("ffprobe") is None or shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg/ffprobe not available")

        out_dir = tmp_path / "frames"
        paths = extract_frames(
            video_path=sample_video,
            out_dir=out_dir,
        )
        assert len(paths) > 0
        for p in paths:
            assert p.is_file()

    def test_cancelled_during_extraction(self, tmp_path):
        """extract_frames should raise CancelledError when cancel_event is set."""
        video = tmp_path / "video.mp4"
        video.touch()
        ffprobe_result = MagicMock(stdout=_make_ffprobe_json(), returncode=0)
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            patch("shutil.which", return_value="/usr/bin/ffprobe"),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.run",
                  return_value=ffprobe_result),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = None
            proc_mock.wait.return_value = 0
            proc_mock.returncode = 0
            proc_mock.communicate.return_value = (None, "")
            mock_popen.return_value = proc_mock

            with pytest.raises(CancelledError, match="cancelled"):
                extract_frames(
                    video_path=video,
                    out_dir=tmp_path / "out",
                    cancel_event=cancel_event,
                )

    def test_broken_pipe_during_extraction_returns_partial(self, tmp_path):
        """extract_frames should return partial results when ffmpeg exits non-zero."""
        video = tmp_path / "video.mp4"
        video.touch()
        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Create some fake PNGs to simulate partial extraction
        for i in range(3):
            fake_png = out_dir / f"00000{i}.png"
            cv2.imwrite(str(fake_png), np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))

        ffprobe_result = MagicMock(stdout=_make_ffprobe_json(nb_frames="10"), returncode=0)

        with (
            patch("shutil.which", return_value="/usr/bin/ffprobe"),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.run",
                  return_value=ffprobe_result),
            patch("sr_engine.data.ffmpeg_extractor.subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = 1  # non-zero exit
            proc_mock.wait.return_value = 1
            proc_mock.returncode = 1
            proc_mock.communicate.return_value = (None, "")
            mock_popen.return_value = proc_mock

            paths = extract_frames(
                video_path=video,
                out_dir=out_dir,
            )

        # Should have partial results (the 3 fake PNGs)
        assert len(paths) >= 1
        for p in paths:
            assert p.is_file()