"""Tests for inference."""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from sr_engine.engine.inference import (
    _read_image_tensor,
    _frame_to_tensor,
    _tensor_to_bgr_image,
    _super_resolve_tensor,
)


class TestImageTensorConversion:
    """Tests for image-to-tensor and tensor-to-image conversion functions."""

    def test_read_image_tensor_shape(self):
        """``_read_image_tensor`` should return a (3, H, W) tensor in [0, 1]."""
        import cv2
        import numpy as np
        import tempfile
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            cv2.imwrite(f.name, img)
            tensor = _read_image_tensor(Path(f.name))
        assert tensor.shape == (3, 64, 64)
        assert tensor.min() >= 0.0
        assert tensor.max() <= 1.0

    def test_frame_to_tensor_converts_bgr_to_rgb(self):
        """``_frame_to_tensor`` should convert BGR to RGB channel order."""
        import numpy as np
        bgr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        bgr[:, :, 0] = 255
        bgr[:, :, 1] = 0
        bgr[:, :, 2] = 128
        tensor = _frame_to_tensor(bgr)
        assert tensor[0, 0, 0].item() == pytest.approx(128.0 / 255.0, abs=1e-6)
        assert tensor[1, 0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert tensor[2, 0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_tensor_to_bgr_roundtrip(self):
        """``_tensor_to_bgr_image`` should produce a uint8 BGR image."""
        import numpy as np
        tensor = torch.rand(3, 64, 64)
        bgr = _tensor_to_bgr_image(tensor)
        assert bgr.shape == (64, 64, 3)
        assert bgr.dtype == np.uint8


class TestSuperResolveTensor:
    """Tests for ``_super_resolve_tensor`` — tiled and non-tiled inference."""

    def test_no_tiling_small_image(self):
        """A small image should be processed without tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        result = _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_tiling_large_image(self):
        """A large image should be processed with tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 32, 32)
        lr = torch.randn(3, 32, 32)
        result = _super_resolve_tensor(model, lr, scale=2, tile_size=16, tile_overlap=4, device="cpu")
        assert result.shape == (3, 64, 64)

    def test_model_called_once_without_tiling(self):
        """The model should be called exactly once without tiling."""
        model = MagicMock()
        model.return_value = torch.randn(1, 3, 64, 64)
        lr = torch.randn(3, 16, 16)
        _super_resolve_tensor(model, lr, scale=4, tile_size=0, tile_overlap=0, device="cpu")
        assert model.call_count == 1


class TestTiling:
    """Tests for tile/stitch roundtrip."""

    def test_tile_and_stitch_roundtrip(self):
        """Tiling then stitching should produce the original shape."""
        from sr_engine.engine.tiling import tile_image, stitch_tiles
        import numpy as np
        lr = torch.randn(3, 64, 64)
        tiles = tile_image(lr, tile_size=32, overlap=8)
        assert len(tiles) > 1
        stitched = stitch_tiles(tiles, output_size=(64, 64), overlap=8)
        assert stitched.shape == (3, 64, 64)


class TestLoadModel:
    """Tests for ``_load_model``."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        config = {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3}
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
        }, ckpt)
        return ckpt

    def test_raises_on_missing_config(self, tmp_path):
        """A checkpoint without a config should raise ValueError."""
        from sr_engine.engine.inference import _load_model
        ckpt = tmp_path / "model.pt"
        torch.save({"state_dict": {"w": torch.tensor([1.0])}}, ckpt)
        with pytest.raises(ValueError, match="no usable 'config'"):
            _load_model(ckpt, device="cpu")

    def test_loads_model_from_checkpoint(self, tmp_path):
        """A valid checkpoint should produce an nn.Module and scale."""
        from sr_engine.engine.inference import _load_model
        import torch.nn as nn
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        loaded_model, scale = _load_model(ckpt, device="cpu")
        assert scale == 4
        assert isinstance(loaded_model, nn.Module)
        loaded_model.eval()


class TestInferImage:
    """Tests for the ``infer_image`` entry point."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        config = {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3}
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
        }, ckpt)
        return ckpt

    def test_saves_output(self, tmp_path, sample_image):
        """infer_image should produce an output image at the specified path."""
        from sr_engine.engine.inference import infer_image
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        out_path = tmp_path / "output.png"
        result = infer_image(
            model_checkpoint=ckpt,
            input_path=sample_image,
            output_path=out_path,
            device="cpu",
        )
        assert result == out_path
        assert out_path.exists()

    def test_reports_tile_progress(self, tmp_path, sample_image):
        """Tiled inference should drive the reporter through the full lifecycle."""
        from sr_engine.engine.inference import infer_image
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        events = []
        class FakeReporter:
            def start(self, total=None, desc=""):
                events.append(("start", total))
            def update(self, n=1):
                events.append(("update", n))
            def finish(self):
                events.append(("finish",))
        out_path = tmp_path / "output.png"
        infer_image(
            model_checkpoint=ckpt,
            input_path=sample_image,
            output_path=out_path,
            tile_size=8,
            tile_overlap=2,
            device="cpu",
            reporter=FakeReporter(),
        )
        kinds = [e[0] for e in events]
        assert kinds[0] == "start"
        assert kinds[-1] == "finish"
        updates = [e for e in events if e[0] == "update"]
        assert len(updates) >= 1


class TestImageMetadataAndMetrics:
    """Tests for image_size, write_preview, and metrics_suite."""

    def test_image_size(self, tmp_path):
        from sr_engine.engine.inference import image_size
        import cv2
        import numpy as np
        img = np.zeros((50, 80, 3), dtype=np.uint8)
        p = tmp_path / "img.png"
        cv2.imwrite(str(p), img)
        assert image_size(p) == (80, 50)

    def test_write_preview_downscales(self, tmp_path):
        from sr_engine.engine.inference import image_size, write_preview
        import cv2
        import numpy as np
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        src = tmp_path / "big.png"
        cv2.imwrite(str(src), img)
        out = write_preview(src, tmp_path / "preview" / "p.png", max_dim=512)
        assert out.exists()
        assert image_size(out) == (512, 256)

    def test_metrics_suite_identical(self, tmp_path):
        """metrics_suite should report near-perfect scores for identical SR/GT."""
        from sr_engine.engine.inference import metrics_suite
        import cv2
        import numpy as np
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        sr = tmp_path / "sr.png"
        gt = tmp_path / "gt.png"
        cv2.imwrite(str(sr), img)
        cv2.imwrite(str(gt), img)
        m = metrics_suite(sr, gt, device="cpu")
        assert m["psnr"] > 50.0
        assert m["ssim"] > 0.9
        assert m["ms_ssim"] > 0.9

    def test_metrics_suite_aligns_gt(self, tmp_path):
        """GT at a different resolution than SR should still compare."""
        from sr_engine.engine.inference import metrics_suite
        import cv2
        import numpy as np
        sr = tmp_path / "sr.png"
        gt = tmp_path / "gt.png"
        cv2.imwrite(str(sr), np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
        cv2.imwrite(str(gt), np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8))
        m = metrics_suite(sr, gt, device="cpu")
        assert "psnr" in m and "ssim" in m and "ms_ssim" in m


class TestRunInferenceWorker:
    """Tests for the API ``run_inference`` worker."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        config = {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3}
        torch.save({
            "state_dict": model.state_dict(),
            "config": config,
        }, ckpt)
        return ckpt

    def _run(self, tmp_path, params):
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference
        import cv2
        import numpy as np
        cv2.imwrite(
            str(tmp_path / "gt.png"),
            np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
        )
        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        run_inference(job_id, params, None, tasks, FakeEvents())
        return tasks, events

    def test_image_payload(self, tmp_path, sample_image):
        """A successful image run yields a rich result payload + previews."""
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        out = tmp_path / "out.png"
        tasks, events = self._run(tmp_path, {
            "model": str(ckpt),
            "input": str(sample_image),
            "output": str(out),
            "gt": str(tmp_path / "gt.png"),
            "format": "png",
            "tile": 0,
            "overlap": 0,
            "device": "cpu",
        })
        # find the actual job record
        jobs = tasks.list_jobs()
        assert jobs, "no job created"
        rec = jobs[0]
        assert rec.status == "completed", rec.error

        payload = rec.result
        assert payload["success"] is True
        assert payload["output"] == str(out)
        assert payload["input_resolution"] == {"width": 64, "height": 64}
        assert payload["output_resolution"] == {"width": 256, "height": 256}
        assert payload["inference_time_ms"] >= 0
        assert Path(payload["preview_input_path"]).exists()
        assert Path(payload["preview_output_path"]).exists()
        m = payload["metrics"]
        assert "psnr" in m and "ssim" in m and "ms_ssim" in m

        types = [e.get("type") for _, e in events if isinstance(e, dict)]
        assert "done" in types

    def test_error_payload(self, tmp_path):
        """A failing run should fail the job and publish an error event."""
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference
        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        run_inference(job_id, {
            "model": str(tmp_path / "does-not-exist.pt"),
            "input": str(tmp_path / "missing.png"),
            "output": str(tmp_path / "out.png"),
        }, None, tasks, FakeEvents())
        rec = tasks.list_jobs()[0]
        assert rec.status == "failed"
        assert any(e.get("type") == "error" for _, e in events if isinstance(e, dict))


class TestRunInferenceWorkerInstance:
    """``run_inference`` with an instance + version (config.yaml driven).

    Covers both instance config layouts: the API-created one (``architecture``
    key, no ``name``) and the CLI-created one (``name`` key). Regression test
    for the ``KeyError: 'name'`` bug.
    """

    def _make_instance(self, tmp_path, *, with_name: bool):
        from sr_engine.workspace import Workspace
        from sr_engine.models.archs.rrdbnet import RRDBNet

        ws = Workspace(tmp_path / "ws")
        config = {
            "num_feat": 32,
            "num_block": 8,
            "num_grow_ch": 16,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "scale": 4,
        }
        if with_name:
            config["name"] = "rrdb_esrgan"
        else:
            config["architecture"] = "rrdb_esrgan"
        ws.create_model_instance("my-model", config)

        model = RRDBNet(
            num_feat=32, num_block=8, num_grow_ch=16,
            num_in_ch=3, num_out_ch=3, scale=4,
        )
        v_dir = ws.path / "models" / "my-model" / "versions" / "v1"
        v_dir.mkdir(parents=True)
        torch.save(model.state_dict(), v_dir / "model.pt")
        return ws

    def _run_instance(self, tmp_path, ws, sample_image):
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference

        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        out = tmp_path / "out.png"
        run_inference(job_id, {
            "instance": "my-model",
            "version": "v1",
            "input": str(sample_image),
            "output": str(out),
            "format": "png",
            "tile": 0,
            "overlap": 0,
            "device": "cpu",
        }, ws, tasks, FakeEvents())
        return tasks, out

    def test_instance_with_architecture_key(self, tmp_path, sample_image):
        """API-created instances store ``architecture`` (no ``name`` key)."""
        ws = self._make_instance(tmp_path, with_name=False)
        tasks, out = self._run_instance(tmp_path, ws, sample_image)
        rec = tasks.list_jobs()[0]
        assert rec.status == "completed", rec.error
        assert rec.result["success"] is True
        assert rec.result["output_resolution"] == {"width": 256, "height": 256}
        assert out.exists()

    def test_instance_with_name_key(self, tmp_path, sample_image):
        """CLI-created instances store a ``name`` key in config.yaml."""
        ws = self._make_instance(tmp_path, with_name=True)
        tasks, _ = self._run_instance(tmp_path, ws, sample_image)
        rec = tasks.list_jobs()[0]
        assert rec.status == "completed", rec.error
        assert rec.result["success"] is True

    def test_instance_missing_architecture(self, tmp_path, sample_image):
        """A config without architecture/name fails with a clear error."""
        from sr_engine.workspace import Workspace
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference

        ws = Workspace(tmp_path / "ws")
        ws.create_model_instance("my-model", {"num_feat": 32})
        v_dir = ws.path / "models" / "my-model" / "versions" / "v1"
        v_dir.mkdir(parents=True)
        from sr_engine.models.archs.rrdbnet import RRDBNet
        torch.save(
            RRDBNet(num_feat=32, num_block=8, num_grow_ch=16, num_in_ch=3, num_out_ch=3, scale=4).state_dict(),
            v_dir / "model.pt",
        )

        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        run_inference(job_id, {
            "instance": "my-model",
            "version": "v1",
            "input": str(sample_image),
            "output": str(tmp_path / "out.png"),
            "format": "png",
            "tile": 0,
            "overlap": 0,
            "device": "cpu",
        }, ws, tasks, FakeEvents())
        rec = tasks.list_jobs()[0]
        assert rec.status == "failed"
        assert "architecture" in (rec.error or "")


class TestRunInferenceWorkerRunCheckpoint:
    """``run_inference`` with a run checkpoint (``model``) + owning ``instance``.

    Mid-run checkpoints store a config that omits architecture params; the
    architecture must be reconstructed from the owning instance's config.yaml.
    """

    def _make_custom_arch_instance(self, tmp_path):
        from sr_engine.workspace import Workspace
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_model_instance("my-model", {
            "architecture": "rrdb_esrgan",
            "num_feat": 32,
            "num_block": 8,
            "num_grow_ch": 16,
            "num_in_ch": 3,
            "num_out_ch": 3,
            "scale": 4,
        })
        return ws

    def _make_run_checkpoint(self, tmp_path, *, num_feat=32, num_block=8):
        """Save a mid-run checkpoint whose config omits architecture params."""
        from sr_engine.models.archs.rrdbnet import RRDBNet
        from sr_engine.models.checkpoint import save_checkpoint
        model = RRDBNet(
            num_feat=num_feat, num_block=num_block, num_grow_ch=16,
            num_in_ch=3, num_out_ch=3, scale=4,
        )
        ckpt = tmp_path / "runs" / "run_001" / "epoch_001.pt"
        save_checkpoint(
            path=ckpt,
            state_dict=model.state_dict(),
            step=1,
            config={
                "name": "rrdb_esrgan",
                "scale": 4,
                "model_format": "torch",
                "training_dtype": "float32",
            },
        )
        return ckpt

    def _run(self, tmp_path, params, ws):
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference
        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        run_inference(job_id, params, ws, tasks, FakeEvents())
        return tasks, events

    def test_custom_architecture_run_checkpoint_loads(self, tmp_path, sample_image):
        """A run checkpoint (config without arch params) + owning instance works."""
        ws = self._make_custom_arch_instance(tmp_path)
        ckpt = self._make_run_checkpoint(tmp_path)
        out = tmp_path / "out.png"
        tasks, _ = self._run(tmp_path, {
            "model": str(ckpt),
            "instance": "my-model",
            "input": str(sample_image),
            "output": str(out),
            "format": "png",
            "tile": 0,
            "overlap": 0,
            "device": "cpu",
        }, ws)
        rec = tasks.list_jobs()[0]
        assert rec.status == "completed", rec.error
        assert rec.result["success"] is True
        assert rec.result["output"] == str(out)
        assert out.exists()

    def test_weight_mismatch_fails_with_clear_error(self, tmp_path, sample_image):
        """Weights from a different architecture than the instance fail clearly."""
        ws = self._make_custom_arch_instance(tmp_path)
        ckpt = self._make_run_checkpoint(tmp_path, num_feat=64, num_block=23)
        out = tmp_path / "out.png"
        tasks, events = self._run(tmp_path, {
            "model": str(ckpt),
            "instance": "my-model",
            "input": str(sample_image),
            "output": str(out),
            "format": "png",
            "tile": 0,
            "overlap": 0,
            "device": "cpu",
        }, ws)
        rec = tasks.list_jobs()[0]
        assert rec.status == "failed"
        assert "my-model" in (rec.error or "")
        assert any(e.get("type") == "error" for _, e in events if isinstance(e, dict))


class TestRunInferenceUniquePreviews:
    """Preview filenames derive from the output stem — unique per run."""

    def _make_rrdb_checkpoint(self, tmp_path):
        from sr_engine.models.archs.rrdbnet import RRDBNet
        model = RRDBNet(num_in_ch=3, num_out_ch=3, scale=4)
        ckpt = tmp_path / "model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config": {"name": "rrdb_esrgan", "scale": 4, "num_in_ch": 3, "num_out_ch": 3},
        }, ckpt)
        return ckpt

    def _make_image(self, path):
        import cv2
        import numpy as np
        cv2.imwrite(str(path), np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))

    def _run(self, tmp_path, params):
        from sr_engine.api.task_manager import BackgroundTaskManager
        from sr_engine.api.workers import run_inference
        tasks = BackgroundTaskManager()
        job_id = tasks.create_job("infer")
        events = []
        class FakeEvents:
            def publish(self, jid, event):
                events.append((jid, event))
        run_inference(job_id, params, None, tasks, FakeEvents())
        rec = tasks.list_jobs()[0]
        assert rec.status == "completed", rec.error
        return rec.result

    def test_consecutive_runs_produce_distinct_previews(self, tmp_path):
        """Two runs into the same output dir yield distinct preview paths."""
        ckpt = self._make_rrdb_checkpoint(tmp_path)
        img_a = tmp_path / "a.png"
        img_b = tmp_path / "b.png"
        self._make_image(img_a)
        self._make_image(img_b)

        results = []
        for i, img in enumerate((img_a, img_b)):
            results.append(self._run(tmp_path, {
                "model": str(ckpt),
                "input": str(img),
                "output": str(tmp_path / "outdir" / f"result_{i}.png"),
                "format": "png",
                "tile": 0,
                "overlap": 0,
                "device": "cpu",
            }))

        assert results[0]["preview_input_path"] != results[1]["preview_input_path"]
        assert results[0]["preview_output_path"] != results[1]["preview_output_path"]
        assert "result_0" in results[0]["preview_input_path"]
        assert "result_1" in results[1]["preview_input_path"]
        assert "result_0" in results[0]["preview_output_path"]
        assert "result_1" in results[1]["preview_output_path"]
        for r in results:
            assert Path(r["preview_input_path"]).exists()
            assert Path(r["preview_output_path"]).exists()
