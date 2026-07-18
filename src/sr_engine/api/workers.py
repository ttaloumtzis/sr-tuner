import logging
import time
from pathlib import Path

from sr_engine.api.callbacks import SSECallback
from sr_engine.api.event_manager import SSEEventManager
from sr_engine.api.progress import SSEProgressReporter
from sr_engine.api.task_manager import (
    BackgroundTaskManager,
    acquire_training_slot,
    release_training_slot,
)
from sr_engine.data.dataset_builder import build_from_preprocessed, build_from_video
from sr_engine.engine.inference import infer_image, infer_video, load_model
from sr_engine.engine.trainer import Trainer, TrainingCancelled
from sr_engine.models.registry import build_model
from sr_engine.utils.config import load_config, merge_overrides
from sr_engine.workspace import Workspace

log = logging.getLogger(__name__)


def run_training(
    job_id: str,
    params: dict,
    ws: Workspace | None,
    cfg_loader,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Run training in a background thread."""
    if not acquire_training_slot():
        tasks.fail_job(job_id, "Another training is already running")
        return

    tasks.start_job(job_id)
    try:
        model_name = params["model_name"]
        instance = params["instance"]
        dataset = params["dataset"]

        model_inst = ws.get_model_instance(instance)
        inst_cfg_path = model_inst.path / "config.yaml"
        inst_cfg = load_config(inst_cfg_path) if inst_cfg_path.exists() else {}

        model_cfg = cfg_loader.get_model_config(model_name) or {}
        if inst_cfg:
            model_cfg = merge_overrides(model_cfg, inst_cfg)

        if ws:
            dataset_path = ws.resolve_dataset(Path(dataset))
        else:
            dataset_path = Path(dataset)

        train_cfg = cfg_loader.get_train_config()
        overrides = params.get("overrides", {})
        if overrides:
            train_cfg = merge_overrides(train_cfg, overrides)

        run_dir = ws.get_run_path(instance) if ws else None

        sse_reporter = SSEProgressReporter(events, job_id)
        sse_callback = SSECallback(events, job_id)

        def _cancel_check() -> bool:
            rec = tasks.get_job(job_id)
            return rec is not None and rec.cancel_event.is_set()

        trainer = Trainer(
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_dir=dataset_path,
            checkpoint_dir=run_dir,
            device=train_cfg.get("device", "cuda"),
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

        tasks.complete_job(job_id)

    except TrainingCancelled:
        tasks.cancel_job(job_id)
    except Exception as e:
        log.exception("Training job %s failed", job_id)
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)
        release_training_slot()


def run_inference(
    job_id: str,
    params: dict,
    ws: Workspace | None,
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Run inference in a background thread."""
    tasks.start_job(job_id)
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
    tasks: BackgroundTaskManager,
    events: SSEEventManager,
) -> None:
    """Build a dataset in a background thread."""
    tasks.start_job(job_id)
    try:
        input_path = Path(params["input"])
        out = params.get("out")
        out_path = Path(out) if out else None

        if ws and out_path is None and input_path.is_file():
            out_path = ws.path / "datasets" / input_path.stem

        sse_reporter = SSEProgressReporter(events, job_id)

        if input_path.is_dir():
            result = build_from_preprocessed(input_path, {}, reporter=sse_reporter)
        else:
            if out_path is None:
                raise ValueError("--out is required for video files")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            result = build_from_video(input_path, out_path, {}, reporter=sse_reporter)

        tasks.complete_job(job_id, {"output": str(result)})

    except Exception as e:
        log.exception("Dataset build job %s failed", job_id)
        tasks.fail_job(job_id, str(e))
    finally:
        events.publish(job_id, None)