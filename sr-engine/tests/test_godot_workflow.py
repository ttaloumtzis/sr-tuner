"""Integration tests mirroring Godot subprocess calls."""

import json
import os
from pathlib import Path

from sr_engine.workspace import Workspace
from sr_engine.utils.config import save_config

from tests.conftest import _create_dataset_dir


GODOT_TRAIN = [
    "--model", "rrdb_esrgan", "--device", "cpu",
    "--max-epochs", "2", "--num-workers", "0",
    "--patch-size", "16", "--batch-size", "2",
    "--no-validation-enabled",
]


class TestGodotHappyPath:
    def test_full_workflow(self, cli_invoker, tmp_path):
        ws_path = tmp_path / "workspace"

        r = cli_invoker(["workspace", "init", "--path", str(ws_path)])
        assert r.exit_code == 0, r.output
        assert (ws_path / ".sr_workspace").exists()

        old_cwd = os.getcwd()
        try:
            os.chdir(str(ws_path))

            r = cli_invoker(["project", "create", "my_test"])
            assert r.exit_code == 0, r.output
            assert (ws_path / "projects" / "my_test").is_dir()

            r = cli_invoker(["workspace", "check"])
            assert r.exit_code == 0, r.output

            dataset = _create_dataset_dir(ws_path / "tmp_data", 3)
            ckpt_dir = ws_path / "projects" / "my_test" / "checkpoints"
            cfg_path = ws_path / "train_cfg.yaml"
            save_config({"checkpoint_dir": str(ckpt_dir)}, cfg_path)

            r = cli_invoker([
                "train", "run",
                "--project", "my_test",
                "--dataset", str(dataset),
                "--config", str(cfg_path),
                "--machine",
                "--experiment-id", "run_001",
            ] + GODOT_TRAIN)
            assert r.exit_code == 0, r.output

            metrics_dir = ws_path / "projects" / "my_test" / "metrics"
            jsonl_files = list(metrics_dir.glob("*.jsonl"))
            assert len(jsonl_files) == 1, f"No .jsonl found in {metrics_dir}"
            content = jsonl_files[0].read_text()
            lines = [l for l in content.split("\n") if l and not l.startswith("#")]
            assert len(lines) >= 1
            messages = [json.loads(l) for l in lines]
            assert messages[0]["type"] in ("phase", "step")
            assert any(m["type"] == "done" for m in messages)

        finally:
            os.chdir(str(old_cwd))


class TestGodotNoWorkspace:
    def test_project_without_workspace_fails(self, cli_invoker, tmp_path):
        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path))
            r = cli_invoker([
                "train", "run",
                "--project", "nonexistent",
                "--dataset", str(tmp_path / "data"),
                "--model", "rrdb_esrgan",
                "--device", "cpu",
            ])
            assert r.exit_code != 0
            assert "workspace" in r.output.lower()
        finally:
            os.chdir(str(old_cwd))


class TestGodotReplayMetrics:
    def test_replay_old_experiment(self, cli_invoker, tmp_path):
        ws_path = tmp_path / "ws_replay"
        ws = Workspace(ws_path)
        ws.init()
        ws.create_project("replay_proj")
        dataset = _create_dataset_dir(tmp_path / "data", 3)
        ckpt_dir = ws_path / "projects" / "replay_proj" / "checkpoints"
        cfg_path = tmp_path / "replay_cfg.yaml"
        save_config({"checkpoint_dir": str(ckpt_dir)}, cfg_path)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(ws_path))
            r = cli_invoker([
                "train", "run",
                "--project", "replay_proj",
                "--dataset", str(dataset),
                "--config", str(cfg_path),
                "--machine",
                "--experiment-id", "replay_001",
            ] + GODOT_TRAIN)
            assert r.exit_code == 0, r.output
        finally:
            os.chdir(str(old_cwd))

        metrics_dir = ws_path / "projects" / "replay_proj" / "metrics"
        jsonl_path = metrics_dir / "replay_001.jsonl"
        assert jsonl_path.is_file()

        content = jsonl_path.read_text()
        lines = [l for l in content.split("\n") if l and not l.startswith("#")]
        messages = [json.loads(l) for l in lines]
        types = {m["type"] for m in messages}
        assert "step" in types
        assert "done" in types
