"""Tests for data/ffmpeg_extractor.py — FFmpegExtractor class."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from sr_engine.data.ffmpeg_extractor import (
    CancelledError,
    FFmpegExtractor,
    VideoInfo,
)


def _make_ffprobe_json(codec="h264", pix_fmt="yuv420p", bit_depth=8,
                       fps="30/1", nb_frames="300", duration="10.0",
                       width=1920, height=1080) -> str:
    return json.dumps({
        "streams": [{
            "codec_name": codec,
            "pix_fmt": pix_fmt,
            "bit_depth": bit_depth,
            "r_frame_rate": fps,
            "nb_frames": nb_frames,
            "duration": duration,
            "width": width,
            "height": height,
        }]
    })


class TestProbe:
    """Tests for FFmpegExtractor.probe()."""

    def test_probe_success(self, tmp_path):
        """probe should parse ffprobe JSON correctly."""
        video = tmp_path / "test.mkv"
        video.touch()
        ffprobe_out = _make_ffprobe_json(
            codec="av1", pix_fmt="yuv420p", bit_depth=8,
            fps="30000/1001", nb_frames="150", duration="5.0",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ffprobe_out, returncode=0)
            info = FFmpegExtractor().probe(video)

        assert isinstance(info, VideoInfo)
        assert info.codec_name == "av1"
        assert info.pix_fmt == "yuv420p"
        assert info.bit_depth == 8
        assert abs(info.fps - 29.97) < 0.01
        assert info.frame_count == 150
        assert info.duration == 5.0
        assert info.width == 1920
        assert info.height == 1080

    def test_probe_no_nb_frames_fallback_to_duration(self, tmp_path):
        """probe should calculate frame count from duration when N/A."""
        video = tmp_path / "test.mkv"
        video.touch()
        ffprobe_out = _make_ffprobe_json(
            nb_frames="N/A", duration="10.0", fps="30/1",
        )

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=ffprobe_out, returncode=0)
            info = FFmpegExtractor().probe(video)

        assert info.frame_count == 300  # 10s * 30fps

    def test_probe_no_video_stream(self, tmp_path):
        """probe should raise ValueError when no video stream exists."""
        video = tmp_path / "test.mkv"
        video.touch()
        no_stream_json = json.dumps({"streams": []})

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=no_stream_json, returncode=0)
            with pytest.raises(ValueError, match="No video stream"):
                FFmpegExtractor().probe(video)

    def test_probe_ffprobe_fails(self, tmp_path):
        """probe should raise RuntimeError when ffprobe returns non-zero."""
        video = tmp_path / "test.mkv"
        video.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", returncode=1, stderr="error")
            with pytest.raises(RuntimeError, match="ffprobe failed"):
                FFmpegExtractor().probe(video)

    def test_probe_ffprobe_not_found(self, tmp_path):
        """probe should raise FileNotFoundError when ffprobe is missing."""
        video = tmp_path / "test.mkv"
        video.touch()

        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="ffprobe not found"):
                FFmpegExtractor().probe(video)


class TestDecoderSelection:
    """Tests for FFmpegExtractor.select_decoder()."""

    def test_select_decoder_for_known_codec(self):
        """select_decoder should try decoders from the priority list."""
        extractor = FFmpegExtractor(prefer_hardware=False)

        with patch.object(extractor, "_decoder_works") as mock_works:
            mock_works.side_effect = lambda d: d == "libdav1d"
            info = VideoInfo("av1", "yuv420p", 8, 30.0, 300, 10.0, 1920, 1080)
            decoder = extractor.select_decoder(info)

        assert decoder == "libdav1d"

    def test_select_decoder_fallback_to_none(self):
        """select_decoder should return None when no decoder works."""
        extractor = FFmpegExtractor(prefer_hardware=False)

        with patch.object(extractor, "_decoder_works", return_value=False):
            info = VideoInfo("av1", "yuv420p", 8, 30.0, 300, 10.0, 1920, 1080)
            decoder = extractor.select_decoder(info)

        assert decoder is None

    def test_select_decoder_unknown_codec(self):
        """select_decoder should return None for unknown codecs."""
        info = VideoInfo("vp9", "yuv420p", 8, 30.0, 300, 10.0, 1920, 1080)
        decoder = FFmpegExtractor().select_decoder(info)
        assert decoder is None

    def test_select_decoder_prefer_hardware(self):
        """prefer_hardware=True should try hardware decoders first."""
        extractor = FFmpegExtractor(prefer_hardware=True)

        calls: list[str] = []
        def _mock_works(d: str) -> bool:
            calls.append(d)
            return False

        with patch.object(extractor, "_decoder_works", _mock_works):
            info = VideoInfo("av1", "yuv420p", 8, 30.0, 300, 10.0, 1920, 1080)
            extractor.select_decoder(info)

        # Should try all candidates (hw first, then sw)
        for d in ["av1_vaapi", "av1_cuvid", "av1_qsv", "libdav1d", "libaom-av1"]:
            assert d in calls, f"Should have tried {d}"


class TestDecoderWorks:
    """Tests for FFmpegExtractor._decoder_works()."""

    def test_decoder_works_success(self):
        """_decoder_works should return True when decoder is listed."""
        decoders_output = (
            " V..... libdav1d           Dav1d AV1 decoder by VideoLAN\n"
            " V..... libaom-av1         Alliance for Open Media AV1 decoder\n"
            " V..... h264               H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10\n"
        )
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=decoders_output, returncode=0)
            assert FFmpegExtractor()._decoder_works("libdav1d")
            assert FFmpegExtractor()._decoder_works("h264")

    def test_decoder_works_fails(self):
        """_decoder_works should return False for unlisted decoder."""
        decoders_output = " V..... h264               H.264 decoder\n"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout=decoders_output, returncode=0)
            assert not FFmpegExtractor()._decoder_works("nonexistent")

    def test_decoder_works_timeout(self):
        """_decoder_works should return False on timeout."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 10)):
            assert not FFmpegExtractor()._decoder_works("slow_decoder")


class TestExtract:
    """Tests for FFmpegExtractor.extract()."""

    def test_extract_success(self, tmp_path):
        """extract should run ffmpeg and return sorted PNG paths."""
        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Simulate FFmpeg creating PNG files before extraction
        for i in range(5):
            cv2.imwrite(
                str(out_dir / f"{i:06d}.png"),
                np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
            )

        info = VideoInfo("h264", "yuv420p", 8, 30.0, 10, 0.333, 64, 64)

        with (
            patch.object(FFmpegExtractor, "select_decoder", return_value=None),
            patch("subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = 0  # process already finished
            proc_mock.wait.return_value = 0
            proc_mock.returncode = 0
            proc_mock.communicate.return_value = (None, "")
            mock_popen.return_value = proc_mock

            paths = FFmpegExtractor().extract(
                video_path=tmp_path / "video.mp4",
                out_dir=out_dir,
                info=info,
            )

        assert len(paths) == 5
        assert all(p.is_file() for p in paths)

    def test_extract_timeout(self, tmp_path):
        """extract should raise RuntimeError on ffmpeg timeout."""
        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        info = VideoInfo("h264", "yuv420p", 8, 30.0, 10, 0.333, 64, 64)

        with (
            patch.object(FFmpegExtractor, "select_decoder", return_value=None),
            patch("subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = None  # process still running
            proc_mock.wait.side_effect = [subprocess.TimeoutExpired("cmd", 3600), 0]
            proc_mock.communicate.side_effect = [
                subprocess.TimeoutExpired("cmd", 3600),
                (None, ""),
            ]
            mock_popen.return_value = proc_mock

            with pytest.raises(RuntimeError, match="timed out"):
                FFmpegExtractor().extract(
                    video_path=tmp_path / "video.mp4",
                    out_dir=out_dir,
                    info=info,
                )

    def test_extract_cancelled(self, tmp_path):
        """extract should raise CancelledError when cancel_event is set."""
        import threading

        out_dir = tmp_path / "frames"
        out_dir.mkdir(parents=True, exist_ok=True)

        info = VideoInfo("h264", "yuv420p", 8, 30.0, 10, 0.333, 64, 64)
        cancel_event = threading.Event()
        cancel_event.set()

        with (
            patch.object(FFmpegExtractor, "select_decoder", return_value=None),
            patch("subprocess.Popen") as mock_popen,
        ):
            proc_mock = MagicMock()
            proc_mock.poll.return_value = None
            proc_mock.wait.return_value = 0
            proc_mock.returncode = 0
            proc_mock.communicate.return_value = (None, "")
            mock_popen.return_value = proc_mock

            with pytest.raises(CancelledError, match="cancelled"):
                FFmpegExtractor().extract(
                    video_path=tmp_path / "video.mp4",
                    out_dir=out_dir,
                    info=info,
                    cancel_event=cancel_event,
                )

    def test_extract_ffmpeg_not_found(self, tmp_path):
        """extract should raise FileNotFoundError when ffmpeg is missing."""
        out_dir = tmp_path / "frames"
        info = VideoInfo("h264", "yuv420p", 8, 30.0, 10, 0.333, 64, 64)

        with patch("shutil.which", return_value=None):
            with pytest.raises(FileNotFoundError, match="ffmpeg not found"):
                FFmpegExtractor().extract(
                    video_path=tmp_path / "video.mp4",
                    out_dir=out_dir,
                    info=info,
                )