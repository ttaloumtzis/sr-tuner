"""Tests for train run CLI command — default vs custom configs."""

import json
import os
from pathlib import Path

from sr_engine.workspace import Workspace
from sr_engine.utils.config import save_config

from tests.conftest import _create_dataset_dir


TRAIN_BASE = [
    "--model", "rrdb_esrgan", "--device", "cpu",
    "--max-epochs", "2", "--num-workers", "0",
    "--patch-size", "16", "--batch-size", "2",
]
TRAIN_BASE_NO_VAL = TRAIN_BASE + ["--no-validation-enabled"]


class TestTrainRunDefaultConfig:
    def test_default_config(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker(["train", "run", "--dataset", str(dataset)] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_cli_overrides_win(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--batch-size", "99",
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_unknown_model_fails(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--model", "nonexistent", "--device", "cpu",
        ])
        assert r.exit_code != 0
        assert "not found" in r.output.lower()


class TestTrainRunCustomConfig:
    def test_custom_config_file(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        cfg_path = tmp_path / "custom.yaml"
        save_config({"max_epochs": 3, "batch_size": 4}, cfg_path)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--config", str(cfg_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_dump_config(self, cli_invoker, tmp_path):
        r = cli_invoker([
            "train", "run", "--dump-config",
            "--dataset", str(tmp_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert cfg["max_epochs"] == 2  # from CLI override


class TestTrainRunMachineMode:
    def test_machine_mode_creates_jsonl(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        ckpt_dir = tmp_path / "my_checkpoints"
        cfg_path = tmp_path / "machine_cfg.yaml"
        save_config({"checkpoint_dir": str(ckpt_dir)}, cfg_path)

        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--config", str(cfg_path),
            "--machine", "--experiment-id", "test_001",
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

        metrics_dir = ckpt_dir / "metrics"
        jsonl_files = list(metrics_dir.glob("*.jsonl"))
        assert len(jsonl_files) >= 1
        content = jsonl_files[0].read_text()
        lines = [l for l in content.split("\n") if l and not l.startswith("#")]
        assert len(lines) >= 1
        messages = [json.loads(l) for l in lines]
        assert any(m["type"] == "done" for m in messages)


class TestTrainRunWorkspaceAware:
    def test_workspace_resolves_dataset(self, cli_invoker, tmp_path):
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj1")
        dataset = _create_dataset_dir(tmp_path / "ws" / "datasets" / "my_set", 3)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj1",
                "--dataset", str(dataset),
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code == 0, r.output
        finally:
            os.chdir(str(old_cwd))

    def test_workspace_resolves_checkpoint_dir(self, cli_invoker, tmp_path):
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj_cp")
        dataset = _create_dataset_dir(tmp_path / "tmp_data", 3)
        cfg_path = tmp_path / "ws_cfg.yaml"
        save_config({
            "checkpoint_dir": str(tmp_path / "ws" / "projects" / "proj_cp" / "checkpoints"),
        }, cfg_path)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj_cp",
                "--dataset", str(dataset),
                "--config", str(cfg_path),
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code == 0, r.output
            assert (tmp_path / "ws" / "projects" / "proj_cp" / "checkpoints").exists()
        finally:
            os.chdir(str(old_cwd))


class TestTrainRunValidationConfig:
    def test_validation_enabled(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 10)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--validation-split", "0.5",
        ] + TRAIN_BASE + ["--validation-enabled"])
        assert r.exit_code == 0, r.output

    def test_validation_disabled(self, cli_invoker, tmp_path):
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
