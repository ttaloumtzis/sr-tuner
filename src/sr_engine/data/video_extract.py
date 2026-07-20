# video_extract.py
"""Video frame extraction — optimized single-stream decoding."""

import logging
from pathlib import Path
from typing import Optional
import cv2

from sr_engine.utils.progress import ProgressReporter

log = logging.getLogger(__name__)


def extract_frames(
        video_path: Path,
        out_dir: Path,
        frame_rate: int | None = None,
        start_time: float = 0.0,
        duration: float | None = None,
        reporter: Optional[ProgressReporter] = None,
) -> list[Path]:
    """Extract frames from a video file into a directory as sequential PNG images.

    Uses ``cv2.VideoCapture`` for decoding. Supports frame-rate throttling,
    seeking to a start time, and limiting extraction to a duration window.
    Frames that fall outside the desired sampling rate are skipped via
    ``grab()`` (decode bypass) for performance.

    Args:
        video_path: Path to the input video file.
        out_dir: Output directory for the extracted PNG frames.
        frame_rate: Target output frame rate. If ``None`` or greater than
            the video's native FPS, all frames are kept.
        start_time: Time in seconds to start extracting from.
        duration: Maximum duration in seconds to extract. ``None`` extracts
            the entire video.
        reporter: Optional progress reporter.

    Returns:
        List of paths to the extracted PNG files, sorted alphabetically.

    Raises:
        FileNotFoundError: If the video file cannot be opened.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    vidcap = cv2.VideoCapture(str(video_path))
    if not vidcap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")

    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    padding_length = max(6, len(str(total_frames)))

    start_frame = int(start_time * video_fps)
    end_frame = total_frames
    if duration is not None:
        end_frame = min(total_frames, start_frame + int(duration * video_fps))

    if frame_rate is None or frame_rate >= video_fps or frame_rate <= 0:
        frame_step = 1
    else:
        frame_step = max(1, round(video_fps / frame_rate))

    # Fast forward to start frame
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    extracted_paths: list[Path] = []
    saved_count = 0
    current_frame = start_frame

    total_to_process = end_frame - start_frame
    reporter = reporter or ProgressReporter()
    reporter.start(total=total_to_process, desc="Extracting Frames")

    try:
        while current_frame < end_frame:
            if (current_frame - start_frame) % frame_step == 0:
                # We want this frame: grab AND decode it
                success, image = vidcap.read()
                if not success:
                    break

                filename = f"{str(saved_count).zfill(padding_length)}.png"
                frame_path = out_dir / filename
                cv2.imwrite(str(frame_path), image)
                extracted_paths.append(frame_path)
                saved_count += 1
            else:
                # We DON'T want this frame: grab it quickly, bypass heavy pixel decoding
                success = vidcap.grab()
                if not success:
                    break

            current_frame += 1
            reporter.update(1)
    except BrokenPipeError:
        log.warning(
            "Video decoder pipe broken at frame %d/%d. "
            "Returning %d extracted frames.",
            current_frame, total_frames, saved_count,
        )

    reporter.finish()

    vidcap.release()
    extracted_paths.sort()
    return extracted_paths