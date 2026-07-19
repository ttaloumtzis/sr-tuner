import json
import logging
import shutil
import subprocess
import threading
import time

from sr_engine.api.event_manager import SSEEventManager

log = logging.getLogger(__name__)

_POLL_INTERVAL = 3.0


def _get_gpu_stats() -> tuple[float | None, float | None, float | None, float | None]:
    """Query GPU stats — tries ``nvidia-smi`` first, then ``rocm-smi``.

    Returns:
        ``(util_percent, vram_used_gb, vram_total_gb, temp_c)``.
        All ``None`` if no GPU is detected.
    """

    # 1. nvidia-smi (NVIDIA)
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                line = result.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(", ")]
                if len(parts) >= 4:
                    util = float(parts[0])
                    vram_used = float(parts[1]) / 1024.0
                    vram_total = float(parts[2]) / 1024.0
                    temp = float(parts[3])
                    return util, vram_used, vram_total, temp
        except (FileNotFoundError, ValueError, subprocess.TimeoutExpired, OSError):
            log.debug("nvidia-smi query failed", exc_info=True)

    # 2. rocm-smi (AMD)
    if shutil.which("rocm-smi"):
        try:
            result = subprocess.run(
                ["rocm-smi", "--showtemp", "--showuse", "--showmeminfo", "vram", "--json"],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                card = data.get("card0", {})

                def _val(key: str) -> str:
                    k = next((k for k in card if k.lower() == key.lower()), "")
                    return str(card.get(k, "")).strip()

                def _num(s: str) -> float:
                    cleaned = "".join(c for c in s if c.isdigit() or c in ".-")
                    return float(cleaned) if cleaned else 0.0

                util_str = _val("GPU use (%)")
                util = _num(util_str)

                vram_bytes = _num(_val("VRAM Total Used Memory (B)"))
                vram_total_bytes = _num(_val("VRAM Total Memory (B)"))

                temp_str = _val("Temperature (Sensor edge) (C)")
                temp = _num(temp_str)

                return util, vram_bytes / 1e9, vram_total_bytes / 1e9, temp
        except (json.JSONDecodeError, KeyError, ValueError, subprocess.TimeoutExpired, OSError):
            log.debug("rocm-smi query failed", exc_info=True)

    # 3. No GPU detected
    return None, None, None, None


class HardwareMonitor:
    """Background thread that periodically polls GPU/CPU/RAM and publishes
    ``hardware`` events to the job's SSE event stream."""

    def __init__(self, events: SSEEventManager, job_id: str) -> None:
        self._events = events
        self._job_id = job_id
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="hw-monitor")
        self._thread.start()
        log.info("HardwareMonitor started for job %s", self._job_id)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        log.info("HardwareMonitor stopped for job %s", self._job_id)

    def _run(self) -> None:
        while not self._stop.wait(_POLL_INTERVAL):
            try:
                event = self._poll()
                self._events.publish(self._job_id, event)
            except Exception:
                log.warning("HardwareMonitor poll error", exc_info=True)

    @staticmethod
    def _poll() -> dict:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0)
        mem = psutil.virtual_memory()
        ram_used_gb = round(mem.used / (1024**3), 1)
        ram_total_gb = round(mem.total / (1024**3), 1)

        gpu_util, vram_used_gb, vram_total_gb, temp_c = _get_gpu_stats()

        return {
            "type": "hardware",
            "cpu_percent": cpu_percent,
            "ram_used_gb": ram_used_gb,
            "ram_total_gb": ram_total_gb,
            "gpu_util_percent": gpu_util,
            "vram_used_gb": vram_used_gb,
            "vram_total_gb": vram_total_gb,
            "temp_c": temp_c,
        }
