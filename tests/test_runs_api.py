"""Tests for the Runs API and per-run disk status ground truth."""

import json

import pytest

from sr_engine.api.routes import runs as runs_module
from sr_engine.workspace import Workspace


@pytest.fixture
def ws_with_run(tmp_path):
    """Workspace with a model instance and one run directory."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    ws.create_model_instance("m1", {"name": "rrdb_esrgan", "scale": 4})
    run_dir = ws.get_run_path("m1")
    return ws, run_dir


# ── Workspace status helpers ─────────────────────────────────────────

class TestRunStatus:
    def test_write_and_read(self, ws_with_run):
        ws, run_dir = ws_with_run
        ws.write_run_status(run_dir, "running", job_id="train_1")
        st = ws.read_run_status(run_dir)
        assert st["status"] == "running"
        assert st["job_id"] == "train_1"
        assert "updated_at" in st

    def test_write_rejects_unknown_status(self, ws_with_run):
        ws, run_dir = ws_with_run
        with pytest.raises(ValueError):
            ws.write_run_status(run_dir, "nonsense")

    def test_read_missing_returns_none(self, ws_with_run):
        ws, run_dir = ws_with_run
        assert ws.read_run_status(run_dir) is None

    def test_read_corrupt_returns_none(self, ws_with_run):
        ws, run_dir = ws_with_run
        (run_dir / "run_status.json").write_text("{not json", encoding="utf-8")
        assert ws.read_run_status(run_dir) is None

    def test_infer_running_with_live_job(self, ws_with_run):
        ws, run_dir = ws_with_run
        ws.write_run_status(run_dir, "running", job_id="train_1")
        assert ws.infer_run_status(run_dir, {"train_1"}) == "running"

    def test_infer_stale_running_becomes_interrupted(self, ws_with_run):
        ws, run_dir = ws_with_run
        ws.write_run_status(run_dir, "running", job_id="train_1")
        assert ws.infer_run_status(run_dir, set()) == "interrupted"
        assert ws.infer_run_status(run_dir, None) == "running"

    def test_infer_terminal_status_trusted(self, ws_with_run):
        ws, run_dir = ws_with_run
        ws.write_run_status(run_dir, "failed", error="boom")
        assert ws.infer_run_status(run_dir, set()) == "failed"

    def test_infer_legacy_with_done_marker(self, ws_with_run):
        ws, run_dir = ws_with_run
        (run_dir / "metrics.jsonl").write_text(
            '{"type": "step", "epoch": 1}\n{"type": "done", "elapsed_seconds": 1}\n',
            encoding="utf-8",
        )
        assert ws.infer_run_status(run_dir, set()) == "finished"

    def test_infer_legacy_without_done_marker(self, ws_with_run):
        ws, run_dir = ws_with_run
        (run_dir / "metrics.jsonl").write_text(
            '{"type": "step", "epoch": 1}\n', encoding="utf-8",
        )
        assert ws.infer_run_status(run_dir, set()) == "interrupted"


# ── Run summary / deletion ───────────────────────────────────────────

class TestRunSummary:
    def test_summary_counts_checkpoints(self, ws_with_run):
        ws, run_dir = ws_with_run
        (run_dir / "epoch_002.pt").write_bytes(b"\x00" * 100)
        (run_dir / "epoch_010.pt").write_bytes(b"\x00" * 200)
        ws.write_run_status(run_dir, "finished")
        s = ws.run_summary(run_dir, set())
        assert s["run_id"] == run_dir.name
        assert s["status"] == "finished"
        assert s["checkpoint_count"] == 2
        assert s["last_epoch"] == 10
        assert s["total_size_mb"] == pytest.approx(300 / (1024 * 1024), abs=0.01)
        assert s["has_metrics"] is False

    def test_summary_reads_run_config(self, ws_with_run):
        ws, run_dir = ws_with_run
        (run_dir / "run_config.json").write_text(
            json.dumps({"model": "swinir", "created_at": "2026-01-01T00:00:00Z"}),
            encoding="utf-8",
        )
        s = ws.run_summary(run_dir, set())
        assert s["config"]["model"] == "swinir"
        assert s["created_at"] == "2026-01-01T00:00:00Z"

    def test_delete_run(self, ws_with_run):
        ws, run_dir = ws_with_run
        assert run_dir.is_dir()
        ws.delete_run("m1", run_dir.name)
        assert not run_dir.is_dir()

    def test_delete_rejects_traversal(self, ws_with_run):
        ws, _ = ws_with_run
        with pytest.raises(ValueError):
            ws.delete_run("m1", "../../etc")
        with pytest.raises(FileNotFoundError):
            ws.delete_run("m1", "run_missing")


# ── API ──────────────────────────────────────────────────────────────

class TestRunsApi:
    @pytest.fixture(autouse=True)
    def _init_ws(self, ws_with_run):
        ws, run_dir = ws_with_run
        self.ws = ws
        self.run_dir = run_dir
        yield

    @pytest.mark.anyio
    async def test_list_runs(self):
        models = await runs_module.list_runs(self.ws)
        assert len(models) == 1
        assert models[0].name == "m1"
        assert models[0].architecture == "rrdb_esrgan"
        assert models[0].scale == 4
        assert [r.run_id for r in models[0].runs] == [self.run_dir.name]

    @pytest.mark.anyio
    async def test_list_checkpoints_with_sidecar(self):
        self.ws.write_run_status(self.run_dir, "finished")
        (self.run_dir / "epoch_005.pt").write_bytes(b"\x00" * 128)
        (self.run_dir / "epoch_005_metrics.json").write_text(
            json.dumps({"epoch": 5, "psnr": 27.1, "ssim": 0.9}),
            encoding="utf-8",
        )
        entries = await runs_module.run_checkpoints("m1", self.run_dir.name, self.ws)
        assert len(entries) == 1
        assert entries[0].epoch == 5
        assert entries[0].metrics.psnr == 27.1
        assert entries[0].file_size_mb == pytest.approx(128 / (1024 * 1024), abs=0.01)

    @pytest.mark.anyio
    async def test_list_checkpoints_jsonl_fallback(self):
        (self.run_dir / "epoch_003.pt").write_bytes(b"\x00" * 64)
        (self.run_dir / "metrics.jsonl").write_text(
            '{"type": "validate", "epoch": 3, "psnr": 25.5, "ssim": 0.85, "val_loss": 0.1}\n'
            + '{"type": "validate", "epoch": 3, "psnr": 26.0, "ssim": 0.86}\n',
            encoding="utf-8",
        )
        entries = await runs_module.run_checkpoints("m1", self.run_dir.name, self.ws)
        assert entries[0].metrics.psnr == 26.0
        assert entries[0].metrics.ssim == 0.86

    @pytest.mark.anyio
    async def test_checkpoints_404(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await runs_module.run_checkpoints("nope", "x", self.ws)
        assert exc.value.status_code == 404
        with pytest.raises(HTTPException) as exc:
            await runs_module.run_checkpoints("m1", "run_ghost", self.ws)
        assert exc.value.status_code == 404

    @pytest.mark.anyio
    async def test_delete_run(self):
        deleted = await runs_module.delete_run("m1", self.run_dir.name, self.ws)
        assert deleted == {"deleted": self.run_dir.name}
        assert not self.run_dir.is_dir()

    @pytest.mark.anyio
    async def test_delete_run_invalid_id(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await runs_module.delete_run("m1", "../etc", self.ws)
        assert exc.value.status_code == 400

    @pytest.mark.anyio
    async def test_delete_active_run_409(self, monkeypatch):
        from fastapi import HTTPException
        monkeypatch.setattr(
            runs_module,
            "_active_jobs",
            lambda: [type("R", (), {"params": {"run_dir": str(self.run_dir)}})()],
        )
        with pytest.raises(HTTPException) as exc:
            await runs_module.delete_run("m1", self.run_dir.name, self.ws)
        assert exc.value.status_code == 409
        assert self.run_dir.is_dir()
