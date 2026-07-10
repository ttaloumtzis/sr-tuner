"""Tests for gui_bridge/jobs.py — JobManager, CLI arg builders, signal handling."""

import json
import signal
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sr_engine.gui_bridge.jobs import (
    cli_args_for_train,
    cli_args_for_infer,
    cli_args_for_dataset_build,
    install_cancel_handler,
    was_cancelled,
    JobManager,
)


class TestCliArgsForTrain:
    def test_minimal(self):
        args = cli_args_for_train({})
        assert args == ["train", "run"]

    def test_all_fields(self):
        params = {
            "model_name": "rrdb_esrgan",
            "dataset": "/data/hr",
            "config": "/cfg.yaml",
            "resume": "/ckpt.pt",
            "device": "cpu",
            "batch_size": 4,
            "learning_rate": 0.0001,
            "max_epochs": 50,
            "project": "my_proj",
            "machine": True,
            "experiment_id": "exp_001",
        }
        args = cli_args_for_train(params)
        assert "--model" in args
        assert args[args.index("--model") + 1] == "rrdb_esrgan"
        assert "--dataset" in args
        assert "--config" in args
        assert "--resume" in args
        assert "--device" in args
        assert "--batch-size" in args
        assert args[args.index("--batch-size") + 1] == "4"
        assert "--learning-rate" in args
        assert "--max-epochs" in args
        assert args[args.index("--max-epochs") + 1] == "50"
        assert "--project" in args
        assert "--machine" in args
        assert "--experiment-id" in args

    def test_partial_fields(self):
        args = cli_args_for_train({"model_name": "swinir", "batch_size": 2})
        assert args == ["train", "run", "--model", "swinir", "--batch-size", "2"]


class TestCliArgsForInfer:
    def test_minimal(self):
        args = cli_args_for_infer({})
        assert args == ["infer", "run"]

    def test_all_fields(self):
        params = {
            "model": "/ckpt.pt",
            "input_path": "/input.png",
            "output": "/output.png",
            "tile": 256,
            "overlap": 32,
            "device": "cuda",
        }
        args = cli_args_for_infer(params)
        assert "--model" in args
        assert "--input" in args
        assert "--output" in args
        assert "--tile" in args
        assert args[args.index("--tile") + 1] == "256"
        assert "--overlap" in args
        assert "--device" in args


class TestCliArgsForDatasetBuild:
    def test_minimal(self):
        args = cli_args_for_dataset_build({})
        assert args == ["dataset", "build"]

    def test_all_fields(self):
        params = {"input": "/video.mp4", "out": "/dataset", "config": "/cfg.yaml"}
        args = cli_args_for_dataset_build(params)
        assert "--input" in args
        assert "--out" in args
        assert "--config" in args


class TestInstallCancelHandler:
    def test_installs_sigterm_handler(self):
        original = signal.getsignal(signal.SIGTERM)
        try:
            install_cancel_handler()
            assert was_cancelled() is False
        finally:
            signal.signal(signal.SIGTERM, original)

    def test_was_cancelled_after_sigterm(self):
        original = signal.getsignal(signal.SIGTERM)
        try:
            install_cancel_handler()
            assert was_cancelled() is False
            import os
            os.kill(os.getpid(), signal.SIGTERM)
            assert was_cancelled() is True
        finally:
            signal.signal(signal.SIGTERM, original)


class TestJobManager:
    @pytest.fixture
    def broadcast(self):
        return MagicMock()

    @pytest.fixture
    def manager(self, tmp_path, broadcast):
        m = JobManager(workspace=tmp_path / "ws", broadcast_fn=broadcast)
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        return m

    def test_jobs_dir_created(self, manager):
        assert manager.jobs_dir.is_dir()

    def test_list_jobs_empty(self, manager):
        assert manager.list_jobs() == []

    def test_get_job_not_found(self, manager):
        assert manager.get_job("nonexistent") is None

    def _setup_mock_proc(self, mock_subprocess_popen, returncode=0):
        mock_proc = mock_subprocess_popen(returncode=returncode)
        mock_proc.returncode = returncode
        mock_proc.wait.return_value = None
        return mock_proc

    def test_start_job_creates_manifest_on_completion(self, manager, broadcast, mock_subprocess_popen):
        mock_proc = self._setup_mock_proc(mock_subprocess_popen, returncode=0)
        manager._job_listener_port = 9999

        with patch("subprocess.Popen", return_value=mock_proc):
            job_id, result = manager.start_job("train", {"model_name": "rrdb"})

        assert result["status"] == "accepted"
        assert job_id in result["job_id"]

        import time
        time.sleep(0.1)

        manifest = manager.get_job(job_id)
        assert manifest is not None
        assert manifest["status"] == "completed"
        assert manifest["exit_code"] == 0

    def test_list_jobs_returns_completed(self, manager, broadcast, mock_subprocess_popen):
        mock_proc = self._setup_mock_proc(mock_subprocess_popen, returncode=0)
        manager._job_listener_port = 9999

        with patch("subprocess.Popen", return_value=mock_proc):
            job_id, _ = manager.start_job("train", {"model_name": "rrdb"})

        import time
        time.sleep(0.1)

        jobs = manager.list_jobs()
        ids = [j["job_id"] for j in jobs]
        assert job_id in ids

    def test_cancel_job_not_found(self, manager):
        result = manager.cancel_job("nonexistent")
        assert result["status"] == "not_found"

    def test_job_cli_args_for_infer(self, manager, broadcast, mock_subprocess_popen):
        mock_proc = self._setup_mock_proc(mock_subprocess_popen, returncode=0)
        manager._job_listener_port = 9999

        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock_proc
            params = {"model": "/ckpt.pt", "input_path": "/in.png", "output": "/out.png"}
            manager.start_job("infer", params)
            call_args = mock_popen.call_args[0][0]
            assert "--model" in call_args
            assert "/ckpt.pt" in call_args
            assert "--input" in call_args
            assert "/in.png" in call_args

    def test_job_failed_status(self, manager, broadcast, mock_subprocess_popen):
        mock_proc = self._setup_mock_proc(mock_subprocess_popen, returncode=1)
        manager._job_listener_port = 9999

        with patch("subprocess.Popen", return_value=mock_proc):
            job_id, _ = manager.start_job("train", {})

        import time
        time.sleep(0.1)

        manifest = manager.get_job(job_id)
        assert manifest is not None
        assert manifest["status"] == "failed"
        assert manifest["exit_code"] == 1

    def test_hello_accepts_valid_token(self, manager):
        manager._pending_hello["job_1"] = {"token": "abc", "spawned_at": 0.0}
        assert manager._on_hello("job_1", "abc") is True
        assert "job_1" in manager._active_jobs
        assert "job_1" not in manager._pending_hello

    def test_hello_rejects_wrong_token(self, manager):
        manager._pending_hello["job_1"] = {"token": "abc", "spawned_at": 0.0}
        assert manager._on_hello("job_1", "wrong") is False

    def test_hello_rejects_unknown_job(self, manager):
        assert manager._on_hello("unknown", "abc") is False
