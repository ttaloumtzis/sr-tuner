"""Video frame extraction — FFmpeg CLI-based decoding with codec-aware fallback."""

import logging
import threading
from pathlib import Path
from typing import Optional

from sr_engine.data.ffmpeg_extractor import FFmpegExtractor
from sr_engine.utils.progress import ProgressReporter

log = logging.getLogger(__name__)


def extract_frames(
        video_path: Path,
        out_dir: Path,
        frame_rate: int | None = None,
        start_time: float = 0.0,
        duration: float | None = None,
        reporter: Optional[ProgressReporter] = None,
        cancel_event: Optional[threading.Event] = None,
) -> list[Path]:
    """Extract frames from a video file into a directory as sequential PNG images.

    Uses ``ffmpeg`` CLI for decoding with automatic codec-aware decoder
    selection and software fallback. Supports frame-rate throttling, seeking
    to a start time, and limiting extraction to a duration window.

    Args:
        video_path: Path to the input video file.
        out_dir: Output directory for the extracted PNG frames.
        frame_rate: Target output frame rate. If ``None`` or greater than
            the video's native FPS, all frames are kept.
        start_time: Time in seconds to start extracting from.
        duration: Maximum duration in seconds to extract. ``None`` extracts
            the entire video.
        reporter: Optional progress reporter.
        cancel_event: Optional event for cancelling extraction mid-flight.

    Returns:
        List of paths to the extracted PNG files, sorted alphabetically.

    Raises:
        FileNotFoundError: If ``ffmpeg`` or ``ffprobe`` is not installed.
        RuntimeError: If ffmpeg fails to extract frames.
        CancelledError: If extraction is cancelled via *cancel_event*.
    """
    extractor = FFmpegExtractor(prefer_hardware=False)
    if not video_path.exists():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    info = extractor.probe(video_path)
    return extractor.extract(
        video_path=video_path,
        out_dir=out_dir,
        info=info,
        frame_rate=frame_rate,
        start_time=start_time,
        duration=duration,
        reporter=reporter,
        cancel_event=cancel_event,
    )