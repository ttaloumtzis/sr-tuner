import multiprocessing
import shutil
import threading
import time
from pathlib import Path

import structlog

from sr_engine.api.callbacks import SSECallback
from sr_engine.api.event_manager import SSEEventManager
from sr_engine.api.progress import SSEProgressReporter
from sr_engine.api.task_manager import (
    BackgroundTaskManager,
    acquire_training_slot,
    release_training_slot,
    register_subprocess_cancel_event,
    unregister_subprocess_cancel_event,
)
from sr_engine.data.dataset_builder import build_from_preprocessed, build_from_video
from sr_engine.monitoring.hardware import HardwareMonitor
from sr_engine.engine.inference import infer_image, infer_video, load_model
from sr_engine.engine.metrics_stream import MetricsStream
from sr_engine.engine.trainer import Trainer, TrainingCancelled
from sr_engine.models.registry import build_model
from sr_engine.utils.config import DefaultConfigs, load_config, merge_overrides
from sr_engine.utils.logging import get_logger
from sr_engine.workspace import Workspace

log = get_logger(__name__)


class QueueEventBus:
    """Drop-in for SSEEventManager that forwards events via multiprocessing.Queue.

    Used in the training subprocess so the main process can bridge to SSE.
    """

    def __init__(self, queue: multiprocessing.Queue) -> None:
        self._queue = queue

    def publish(self, job_id: str, event: dict) -> None:
        try:
            self._queue.put_nowait((job_id, event))
        except Exception:
            pass


def _run_training_subprocess(
    job_id: str,
    params: dict,
    ws_path: str | None,
    event_queue: multiprocessing.Queue,
    cancel_event,
) -> None:
    """Run training in a subprocess with an isolated CUDA context.

    Sends events back to the main process via *event_queue*.
    Uses *cancel_event* (multiprocessing.Event) for cancellation checks.
    """
    events = QueueEventBus(event_queue)

    def _cancel_check() -> bool:
        return cancel_event.is_set()

    try:
        ws = Workspace(Path(ws_path)) if ws_path else None
        cfg = DefaultConfigs(ws)

        model_name: str = params["model_name"]
        instance: str = params["instance"]
        dataset: str = params["dataset"]

        model_inst = ws.get_model_instance(instance) if ws else None
        inst_cfg_path = model_inst.path / "config.yaml" if model_inst else None
        inst_cfg = load_config(inst_cfg_path) if inst_cfg_path and inst_cfg_path.exists() else {}

        model_cfg = cfg.get_model_config(model_name) or {}
        if inst_cfg:
            model_cfg = merge_overrides(model_cfg, inst_cfg)

        if ws:
            dataset_path = ws.resolve_dataset(Path(dataset))
        else:
            dataset_path = Path(dataset)

        custom_config = params.get("config")
        if custom_config:
            train_cfg = load_config(Path(custom_config))
        else:
            train_cfg = cfg.get_train_config()

        overrides = params.get("overrides", {})
        if overrides:
            train_cfg = merge_overrides(train_cfg, overrides)

        run_dir = ws.get_run_path(instance) if ws else None

        val_cfg = train_cfg.get("validation", {})
        val_enabled = bool(val_cfg.get("enabled", True))
        val_split = float(val_cfg.get("split", 0.1))
        val_dataset_dir = Path(val_cfg["dataset"]) if val_cfg.get("dataset") else None
        if val_dataset_dir and ws:
            val_dataset_dir = ws.resolve_dataset(val_dataset_dir)

        load_weights_from = None
        resume_from = None
        resume_spec = params.get("resume")
        if resume_spec and ws and instance:
            version_path = ws.resolve_version(instance, resume_spec)
            if version_path:
                load_weights_from = version_path
            else:
                path = Path(resume_spec)
                if path.exists():
                    resume_from = path

        sse_reporter = SSEProgressReporter(events, job_id)
        sse_callback = SSECallback(events, job_id)

        metrics_stream: MetricsStream | None = None
        validation_frame_dir: Path | None = None
        write_metrics_file = params.get("write_metrics_file", True)
        if run_dir is not None and write_metrics_file:
            metrics_path = run_dir / "metrics.jsonl"
            metrics_stream = MetricsStream(metrics_path, metadata={
                "job_id": job_id,
                "model": model_name,
                "instance": instance,
                "dataset": dataset,
            })
        if run_dir is not None:
            validation_frame_dir = run_dir / "validation"
            validation_frame_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=dataset_path,
            load_weights_from=load_weights_from,
            resume_from=resume_from,
            checkpoint_dir=run_dir,
            device=train_cfg.get("device", "cuda"),
            validation_enabled=val_enabled,
            validation_split=val_split,
            val_dataset_dir=val_dataset_dir,
            metrics_stream=metrics_stream,
            metrics_frequency=int(train_cfg.get("metrics_frequency", 1)),
            validation_frame_dir=validation_frame_dir,
            progress_reporter=sse_reporter,
            callbacks=[sse_callback],
            cancel_check=_cancel_check,
        )

        trainer.train()

        if ws:
            next_ver = ws.next_model_version(instance)
            ws.save_model_version(
                instance,
                next_ver,
                trainer.get_model().state_dict(),
                {
                    "run": run_dir.name if run_dir else "",
                    "timestamp": time.time(),
                },
            )

        events.publish(job_id, {"type": "phase", "phase": "complete"})
        event_queue.put(None)

    except TrainingCancelled:
        events.publish(job_id, {"type": "phase", "phase": "cancelled"})
        event_queue.put(None)

    except Exception as e:
        log.exception("Training subprocess %s failed", job_id)
        error_type = type(e).__name__
        error_code = "CUDA_OUT_OF_MEMORY" if "out of memory" in str(e).lower() else error_type
        events.publish(job_id, {"type": "error", "code": error_code, "message": str(e)})
        event_queue.put(None)


def _bridge_loop(
    bridge_stop: threading.Event,
    event_queue: multiprocessing.Queue,
    events: SSEEventManager,
    job_id: str,
    tasks: BackgroundTaskManager,
) -> None:
    """Read events from subprocess queue and publish to SSE + task manager."""
    while not bridge_stop.is_set():
        try:
            item = event_queue.get(timeout=1)
        except Exception:
            continue
        if item is None:
            break
        jid, evt = item
        events.publish(jid, evt)


def run_training(
    job_id: str,
    params: dict,
    ws: Workspace | None,
    cfg_loader,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Run training in a subprocess to isolate the CUDA context.

    The main process manages the training slot, hardware monitoring,
    and bridges SSE events from the subprocess.
    """
    if not acquire_training_slot():
        tasks.fail_job(job_id, "Another training is already running")
        return

    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    ctx = multiprocessing.get_context("spawn")
    event_queue = ctx.Queue()
    cancel_event = ctx.Event()
    register_subprocess_cancel_event(job_id, cancel_event)

    hw_monitor = HardwareMonitor(events, job_id)
    bridge_stop = threading.Event()

    proc = ctx.Process(
        target=_run_training_subprocess,
        args=(
            job_id,
            {
                "model_name": params["model_name"],
                "instance": params["instance"],
                "dataset": params["dataset"],
                "config": params.get("config"),
                "resume": params.get("resume"),
                "overrides": params.get("overrides", {}),
                "write_metrics_file": params.get("write_metrics_file", True),
            },
            str(ws.path) if ws else None,
            event_queue,
            cancel_event,
        ),
    )

    try:
        proc.start()
        hw_monitor.start()

        bridge_thread = threading.Thread(
            target=_bridge_loop,
            args=(bridge_stop, event_queue, events, job_id, tasks),
            daemon=True,
        )
        bridge_thread.start()

        proc.join()
        bridge_stop.set()

        if proc.exitcode != 0 and proc.exitcode is not None:
            msg = f"Training subprocess exited with code {proc.exitcode}"
            log.error(msg)
            tasks.fail_job(job_id, msg)
            events.publish(job_id, {"type": "error", "code": "SUBPROCESS_CRASH", "message": msg})

        rec = tasks.get_job(job_id)
        if rec and rec.status == "running":
            tasks.complete_job(job_id)

    except Exception as e:
        log.exception("Training job %s failed", job_id)
        tasks.fail_job(job_id, str(e))
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
    finally:
        hw_monitor.stop()
        events.publish(job_id, None)
        unregister_subprocess_cancel_event(job_id)
        release_training_slot()
        # Clear the queue
        while not event_queue.empty():
            try:
                event_queue.get_nowait()
            except Exception:
                break


def run_inference(
    job_id: str,
    params: dict,
    ws: Workspace | None,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Run inference in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    try:
        model = params.get("model")
        instance = params.get("instance")
        version = params.get("version")
        input_path = Path(params["input"])
        output_path = Path(params["output"])
        tile = params.get("tile", 512)
        overlap = params.get("overlap", 64)
        device = params.get("device", "cuda")

        loaded_model = None
        model_scale = None

        if instance and ws:
            model_inst = ws.get_model_instance(instance)
            inst_cfg = load_config(model_inst.path / "config.yaml")
            v_path = ws.resolve_version(instance, version)
            if not v_path:
                raise FileNotFoundError(f"No version found for instance '{instance}'")
            import torch
            state_dict = torch.load(v_path, weights_only=True, map_location="cpu")
            loaded_model = build_model(inst_cfg["name"], inst_cfg)
            loaded_model.load_state_dict(state_dict)
            loaded_model = loaded_model.to(device).eval()
            model_scale = int(inst_cfg.get("scale", 4))
        elif model:
            loaded_model, model_scale = load_model(Path(model), device)
        else:
            raise ValueError("Provide --model <path> or --instance")

        sse_reporter = SSEProgressReporter(events, job_id)

        suffix = input_path.suffix.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".ts"}

        if suffix in video_exts:
            result = infer_video(
                model_checkpoint=Path(model) if model else None,
                input_path=input_path,
                output_path=output_path,
                tile_size=tile,
                tile_overlap=overlap,
                device=device,
                reporter=sse_reporter,
                model=loaded_model,
                scale=model_scale,
            )
        else:
            result = infer_image(
                model_checkpoint=Path(model) if model else None,
                input_path=input_path,
                output_path=output_path,
                tile_size=tile,
                tile_overlap=overlap,
                device=device,
                model=loaded_model,
                scale=model_scale,
            )

        tasks.complete_job(job_id, {"output": str(result)})

    except Exception as e:
        log.exception("Inference job %s failed", job_id)
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)


def run_dataset_build(
    job_id: str,
    params: dict,
    ws: Workspace | None,
    cfg_loader: DefaultConfigs,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Build a dataset in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    t0 = time.time()
    try:
        input_path = Path(params["input"])
        out = params.get("out")
        out_path = Path(out) if out else None

        if ws and out_path is None and input_path.is_file():
            out_path = ws.path / "datasets" / input_path.stem

        cfg = cfg_loader.get_dataset_config()

        degradations = params.get("degradations")
        if degradations:
            enabled = set(d.strip() for d in degradations.split(","))
            _DEG_MAP = {
                "blur": "blur", "noise": "noise", "jpeg": "jpeg",
                "jpeg2000": "jpeg2000", "color-jitter": "color_jitter",
            }
            deg_cfg = cfg.setdefault("degradation", {})
            for cli_name, cfg_key in _DEG_MAP.items():
                if cfg_key in deg_cfg:
                    deg_cfg[cfg_key]["enabled"] = cli_name in enabled

        config_overrides = params.get("config_overrides")
        if config_overrides:
            cfg = merge_overrides(cfg, config_overrides)

        sse_reporter = SSEProgressReporter(events, job_id)

        if input_path.is_dir():
            result = build_from_preprocessed(input_path, cfg, reporter=sse_reporter)
        else:
            if out_path is None:
                raise ValueError("--out is required for video files")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cancel_event = tasks.get_job(job_id).cancel_event if tasks else None
            result = build_from_video(input_path, out_path, cfg, reporter=sse_reporter, cancel_event=cancel_event)

        events.publish(job_id, {"type": "done", "elapsed_seconds": time.time() - t0, "output": str(result)})
        tasks.complete_job(job_id, {"output": str(result)})

    except Exception as e:
        log.exception("Dataset build job %s failed", job_id)
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)


def run_dataset_merge(
    job_id: str,
    params: dict,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Merge datasets in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    t0 = time.time()
    try:
        from sr_engine.data.dataset_merge import merge_datasets

        sse = SSEProgressReporter(events, job_id)

        datasets_root = Path(params["input"])
        out_dir = Path(params["out"]) if params.get("out") else None
        if out_dir is None:
            raise ValueError("--out is required for dataset merge")

        results = merge_datasets(
            datasets_root=datasets_root,
            out_dir=out_dir,
            scale=params.get("scale"),
            output_name=params.get("name"),
            reporter=sse,
            dataset_dirs=[Path(d) for d in params["input_datasets"]] if params.get("input_datasets") else None,
        )

        results_dict = [
            {"scale": r.scale, "output_path": str(r.output_path), "source_datasets": [str(s) for s in r.source_datasets]}
            for r in results
        ]

        if not params.get("keep_sources", False):
            for r in results:
                for src in r.source_datasets:
                    shutil.rmtree(src)

        events.publish(job_id, {"type": "done", "elapsed_seconds": time.time() - t0, "results": results_dict})
        tasks.complete_job(job_id, {"results": results_dict})

    except Exception as e:
        log.exception("Dataset merge job %s failed", job_id)
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)


def run_dataset_health(
    job_id: str,
    params: dict,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Run dataset health check in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    t0 = time.time()
    try:
        from sr_engine.data.dataset_health import check_dataset_health, save_health_report

        sse = SSEProgressReporter(events, job_id)
        dataset_dir = Path(params["path"])
        report = check_dataset_health(dataset_dir, reporter=sse)
        save_health_report(dataset_dir, report)

        events.publish(job_id, {"type": "done", "elapsed_seconds": time.time() - t0, "report": report})
        tasks.complete_job(job_id, {"report": report})

    except Exception as e:
        log.exception("Dataset health job %s failed", job_id)
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)


def run_dataset_validate(
    job_id: str,
    params: dict,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Validate a dataset in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    t0 = time.time()
    try:
        from sr_engine.data.dataset_validator import validate

        sse = SSEProgressReporter(events, job_id)
        dataset_dir = Path(params["path"])
        report = validate(dataset_dir, reporter=sse)

        events.publish(job_id, {
            "type": "done",
            "elapsed_seconds": time.time() - t0,
            "validation": {"valid": report.ok, "problems": report.problems, "num_pairs": report.num_pairs},
        })
        tasks.complete_job(job_id, {"valid": report.ok, "problems": report.problems})

    except Exception as e:
        log.exception("Dataset validate job %s failed", job_id)
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)


def run_dataset_prune(
    job_id: str,
    dataset_path: str,
    black_frames: list[str],
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Prune black frame pairs in a background thread."""
    tasks.start_job(job_id)
    structlog.contextvars.bind_contextvars(job_id=job_id)
    try:
        from sr_engine.data.dataset_health import prune_black_frames

        sse = SSEProgressReporter(events, job_id)
        dataset_dir = Path(dataset_path)
        prune_black_frames(dataset_dir, black_frames, reporter=sse)

        events.publish(job_id, {"type": "done", "message": f"Pruned {len(black_frames)} frames"})
        tasks.complete_job(job_id, {"pruned": len(black_frames)})

    except Exception as e:
        log.exception("Dataset prune job %s failed", job_id)
        events.publish(job_id, {"type": "error", "code": type(e).__name__, "message": str(e)})
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)
