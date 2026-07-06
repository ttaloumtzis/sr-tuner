# video_extract.py
"""Video frame extraction — optimized single-stream decoding."""

from pathlib import Path
import cv2
from tqdm import tqdm  # Added import


def extract_frames(
        video_path: Path,
        out_dir: Path,
        frame_rate: int | None = None,
        start_time: float = 0.0,
        duration: float | None = None,
) -> list[Path]:
    """Extract frames from *video_path* into *out_dir* as PNG images efficiently."""
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

    # Set up the progress bar tracking the exact frame span we are processing
    total_to_process = end_frame - start_frame

    with tqdm(
            total=total_to_process,
            desc="🎬 Extracting Frames",
            unit="fr"
    ) as pbar:
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
            pbar.update(1)  # Increment the progress bar by 1 frame

    vidcap.release()
    extracted_paths.sort()
    return extracted_paths