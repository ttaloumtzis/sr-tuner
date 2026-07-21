"""FFmpeg-based video frame extraction with codec-aware decoder fallback."""

import json
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from sr_engine.utils.progress import ProgressReporter

log = logging.getLogger(__name__)


class CancelledError(RuntimeError):
    """Raised when frame extraction is cancelled via cancel_event."""


@dataclass
class VideoInfo:
    codec_name: str
    pix_fmt: str
    bit_depth: int
    fps: float
    frame_count: int
    duration: float
    width: int
    height: int


class FFmpegExtractor:
    DECODER_PRIORITY: dict[str, list[str]] = {
        "av1":  ["av1_vaapi", "av1_cuvid", "av1_qsv", "libdav1d", "libaom-av1"],
        "hevc": ["hevc_vaapi", "hevc_cuvid", "hevc_qsv", "hevc"],
        "h264": ["h264_vaapi", "h264_cuvid", "h264_qsv", "h264"],
    }

    def __init__(self, prefer_hardware: bool = False) -> None:
        self._prefer_hardware = prefer_hardware

    # ── probing ──────────────────────────────────────────────────────────────

    def probe(self, video_path: Path) -> VideoInfo:
        self._ensure_binary("ffprobe")
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,bit_depth,r_frame_rate,nb_frames,duration,width,height",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {video_path}: {result.stderr.strip()}"
            )
        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            raise ValueError(f"No video stream found in {video_path}")
        s = streams[0]

        codec_name = s.get("codec_name", "unknown")
        pix_fmt = s.get("pix_fmt", "unknown")
        bit_depth = int(s.get("bit_depth", 8))

        raw_fps = s.get("r_frame_rate", "30/1")
        try:
            num, den = raw_fps.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            fps = 30.0

        raw_nb = s.get("nb_frames")
        raw_duration = s.get("duration", "0")
        if raw_nb and raw_nb != "N/A":
            frame_count = int(raw_nb)
        else:
            duration = float(raw_duration) if raw_duration else 0.0
            frame_count = int(duration * fps) if duration > 0 else 0

        # Fallback: try format-level duration when stream-level duration is missing
        if frame_count == 0:
            fmt_cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "json",
                str(video_path),
            ]
            try:
                fmt_result = subprocess.run(fmt_cmd, capture_output=True, text=True, timeout=15)
                if fmt_result.returncode == 0:
                    fmt_data = json.loads(fmt_result.stdout)
                    fmt_duration = fmt_data.get("format", {}).get("duration")
                    if fmt_duration:
                        duration = float(fmt_duration)
                        frame_count = int(duration * fps) if duration > 0 else 0
            except (json.JSONDecodeError, ValueError, subprocess.TimeoutExpired, OSError):
                pass

        # Use the best available duration for VideoInfo
        stream_dur = float(raw_duration) if raw_duration else 0.0
        if stream_dur <= 0 and frame_count > 0:
            stream_dur = frame_count / fps
        width = int(s.get("width", 0))
        height = int(s.get("height", 0))

        return VideoInfo(
            codec_name=codec_name,
            pix_fmt=pix_fmt,
            bit_depth=bit_depth,
            fps=fps,
            frame_count=frame_count,
            duration=stream_dur,
            width=width,
            height=height,
        )

    # ── decoder selection ────────────────────────────────────────────────────

    def select_decoder(self, info: VideoInfo) -> str | None:
        candidates = self.DECODER_PRIORITY.get(info.codec_name)
        if not candidates:
            return None

        if not self._prefer_hardware:
            sw_candidates = [c for c in candidates if "vaapi" not in c and "cuvid" not in c and "qsv" not in c]
            if sw_candidates:
                candidates = sw_candidates

        for dec in candidates:
            if self._decoder_works(dec):
                return dec

        return None

    def _decoder_works(self, decoder: str) -> bool:
        try:
            result = subprocess.run(
                ["ffmpeg", "-decoders"],
                capture_output=True, text=True, timeout=10,
            )
            return f" {decoder} " in f" {result.stdout} "
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    # ── frame extraction ─────────────────────────────────────────────────────

    def extract(
        self,
        video_path: Path,
        out_dir: Path,
        info: VideoInfo,
        frame_rate: int | None = None,
        start_time: float = 0.0,
        duration: float | None = None,
        reporter: Optional[ProgressReporter] = None,
        cancel_event: Optional[threading.Event] = None,
        pixel_format: str = "rgb24",
    ) -> list[Path]:
        self._ensure_binary("ffmpeg")
        out_dir.mkdir(parents=True, exist_ok=True)

        decoder = self.select_decoder(info)
        padding = max(6, len(str(info.frame_count)))
        output_pattern = str(out_dir / f"%0{padding}d.png")

        if frame_rate is None or frame_rate <= 0 or frame_rate >= info.fps:
            effective_rate = info.fps
        else:
            effective_rate = float(frame_rate)

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-ss", str(start_time),
            "-hwaccel", "none",
        ]
        if decoder:
            cmd.extend(["-c:v", decoder])
        cmd.extend(["-i", str(video_path)])
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        cmd.extend([
            "-vf", f"fps={effective_rate}",
            "-sws_flags", "spline+full_chroma_int+accurate_rnd",
            "-pix_fmt", pixel_format,
            "-compression_level", "3",
            "-an", "-sn", "-dn",
            "-start_number", "0",
            "-y",
            output_pattern,
        ])

        total_frames = info.frame_count
        if duration is not None:
            total_frames = min(total_frames, int(duration * info.fps))

        log.info("Running ffmpeg: %s", " ".join(str(a) for a in cmd))

        reporter = reporter or ProgressReporter()
        reporter.start(total=total_frames if total_frames > 0 else None, desc="Extracting Frames")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Watchdog thread for cancellation
        cancel_watchdog: threading.Thread | None = None
        if cancel_event is not None:

            def _watchdog() -> None:
                cancel_event.wait()
                proc.kill()

            cancel_watchdog = threading.Thread(target=_watchdog, daemon=True)
            cancel_watchdog.start()

        # Polling thread: count extracted PNGs to report progress.
        # This avoids stdout buffering issues with FFmpeg's -progress pipe:1.
        last_count = 0
        stop_polling = threading.Event()

        def _poll_progress() -> None:
            nonlocal last_count
            while not stop_polling.is_set() and proc.poll() is None:
                count = len(list(out_dir.glob("*.png")))
                delta = count - last_count
                if delta > 0:
                    reporter.update(delta)
                    last_count = count
                stop_polling.wait(0.5)

        poll_thread = threading.Thread(target=_poll_progress, daemon=True)
        poll_thread.start()

        stderr_output: str | None = None
        try:
            _, stderr_output = proc.communicate(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise RuntimeError(f"ffmpeg timed out extracting frames from {video_path}")
        finally:
            reporter.finish()
            stop_polling.set()
            poll_thread.join()

        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("Frame extraction was cancelled")

        if proc.returncode != 0:
            msg = (
                f"ffmpeg exited with code {proc.returncode} for {video_path}. "
                f"stderr: {stderr_output or '(empty)'}"
            )
            log.warning(msg)
            if len(list(out_dir.glob("*.png"))) == 0:
                raise RuntimeError(msg)

        extracted: list[Path] = sorted(out_dir.glob("*.png"))

        # Verify the first extracted frame is readable by OpenCV
        if extracted:
            test_img = cv2.imread(str(extracted[0]))
            if test_img is None:
                raise RuntimeError(
                    f"OpenCV cannot decode the first extracted frame from {video_path}. "
                    f"The video likely uses a pixel format or codec incompatible with OpenCV. "
                    f"Try re-encoding: ffmpeg -i {video_path} -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4"
                )

        return extracted

    @staticmethod
    def _ensure_binary(name: str) -> None:
        if shutil.which(name) is None:
            raise FileNotFoundError(
                f"{name} not found in PATH. "
                f"Install ffmpeg (https://ffmpeg.org/download.html)"
            )