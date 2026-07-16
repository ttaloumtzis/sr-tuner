"""Tests for train run CLI command — default vs custom configs."""

import json
import os

from sr_engine.workspace import Workspace
from sr_engine.utils.config import save_config


def test_train_run_help(cli_invoker):
    """``train run --help`` should succeed."""
    r = cli_invoker(["train", "run", "--help"])
    assert r.exit_code == 0


TRAIN_BASE = [
    "--model", "rrdb_esrgan", "--device", "cpu",
    "--max-epochs", "2", "--num-workers", "0",
    "--patch-size", "16", "--batch-size", "2",
]
TRAIN_BASE_NO_VAL = TRAIN_BASE + ["--no-validation-enabled"]


class TestTrainRunDefaultConfig:
    """Tests for training with default config."""

    def test_default_config(self, cli_invoker, tmp_path):
        """Training with default config should complete successfully."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker(["train", "run", "--dataset", str(dataset)] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_cli_overrides_win(self, cli_invoker, tmp_path):
        """CLI overrides should take precedence over defaults."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--batch-size", "99",
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_unknown_model_fails(self, cli_invoker, tmp_path):
        """An unknown model name should produce an error."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--model", "nonexistent", "--device", "cpu",
        ])
        assert r.exit_code != 0
        assert "not found" in r.output.lower()


class TestTrainRunCustomConfig:
    """Tests for training with a custom config file."""

    def test_custom_config_file(self, cli_invoker, tmp_path):
        """A custom config file should be loaded successfully."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        cfg_path = tmp_path / "custom.yaml"
        save_config({"max_epochs": 3, "batch_size": 4}, cfg_path)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--config", str(cfg_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_dump_config(self, cli_invoker, tmp_path):
        """``--dump-config`` should print the merged config and exit."""
        r = cli_invoker([
            "train", "run", "--dump-config",
            "--dataset", str(tmp_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert cfg["max_epochs"] == 2


class TestTrainRunMachineMode:
    """Tests for machine-readable metrics mode."""

    def test_machine_mode_creates_jsonl(self, cli_invoker, tmp_path):
        """``--machine`` should produce a JSONL metrics file."""
        from conftest import _create_dataset_dir
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
    """Tests for workspace-aware training."""

    def test_workspace_resolves_dataset(self, cli_invoker, tmp_path):
        """Workspace-aware dataset resolution should work."""
        from conftest import _create_dataset_dir
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
        """Workspace-aware checkpoint directory resolution."""
        from conftest import _create_dataset_dir
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


class TestTrainRunWithInstance:
    """Tests for training with --instance flag."""

    def test_train_run_with_instance_creates_run_dir(self, cli_invoker, tmp_path):
        """Training with --instance should create a run directory."""
        from conftest import _create_dataset_dir
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "rrdb_esrgan", "scale": 4})
        dataset = _create_dataset_dir(tmp_path / "ws" / "datasets" / "my_set", 3)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj1", "--instance", "v1",
                "--dataset", str(dataset),
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code == 0, r.output
        finally:
            os.chdir(str(old_cwd))

        runs_dir = tmp_path / "ws" / "projects" / "proj1" / "models" / "v1" / "runs"
        run_dirs = list(runs_dir.glob("run_*"))
        assert len(run_dirs) >= 1

    def test_train_run_with_instance_saves_train_config(self, cli_invoker, tmp_path):
        """Training with --instance should save train_config.yaml in run dir."""
        from conftest import _create_dataset_dir
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "rrdb_esrgan", "scale": 4})
        dataset = _create_dataset_dir(tmp_path / "ws" / "datasets" / "my_set", 3)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj1", "--instance", "v1",
                "--dataset", str(dataset),
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code == 0, r.output
        finally:
            os.chdir(str(old_cwd))

        runs_dir = tmp_path / "ws" / "projects" / "proj1" / "models" / "v1" / "runs"
        run_dirs = sorted(runs_dir.glob("run_*"))
        assert len(run_dirs) >= 1
        tc = run_dirs[0] / "train_config.yaml"
        assert tc.is_file()
        import yaml
        cfg = yaml.safe_load(tc.read_text())
        assert cfg["max_epochs"] == 2

    def test_train_run_instance_without_project_raises(self, cli_invoker, tmp_path):
        """--instance without --project should raise an error."""
        r = cli_invoker(["train", "run", "--instance", "v1", "--dataset", str(tmp_path)])
        assert r.exit_code != 0
        assert "--instance requires --project" in r.output

    def test_train_run_instance_without_create_raises(self, cli_invoker, tmp_path):
        """--instance without creating it first should raise an error."""
        from conftest import _create_dataset_dir
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj1")
        dataset = _create_dataset_dir(tmp_path / "ws" / "datasets" / "my_set", 3)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj1", "--instance", "nonexistent",
                "--dataset", str(dataset),
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code != 0
            assert "not found" in r.output.lower()
        finally:
            os.chdir(str(old_cwd))

    def test_train_run_with_instance_machine_mode(self, cli_invoker, tmp_path):
        """--machine with --instance should write metrics to run dir."""
        from conftest import _create_dataset_dir
        ws = Workspace(tmp_path / "ws")
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "rrdb_esrgan", "scale": 4})
        dataset = _create_dataset_dir(tmp_path / "ws" / "datasets" / "my_set", 3)

        old_cwd = os.getcwd()
        try:
            os.chdir(str(tmp_path / "ws"))
            r = cli_invoker([
                "train", "run", "--project", "proj1", "--instance", "v1",
                "--dataset", str(dataset),
                "--machine", "--experiment-id", "test_inst_001",
            ] + TRAIN_BASE_NO_VAL)
            assert r.exit_code == 0, r.output
        finally:
            os.chdir(str(old_cwd))

        runs_dir = tmp_path / "ws" / "projects" / "proj1" / "models" / "v1" / "runs"
        run_dirs = sorted(runs_dir.glob("run_*"))
        assert len(run_dirs) >= 1
        jsonl_files = list(run_dirs[0].glob("*.jsonl"))
        assert len(jsonl_files) >= 1


class TestTrainRunValidationConfig:
    """Tests for validation split configuration."""

    def test_validation_enabled(self, cli_invoker, tmp_path):
        """Training with validation enabled should succeed."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 10)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--validation-split", "0.5",
        ] + TRAIN_BASE + ["--validation-enabled"])
        assert r.exit_code == 0, r.output

    def test_validation_disabled(self, cli_invoker, tmp_path):
        """Training with validation disabled should succeed."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output


class TestTrainRunMixedPrecision:
    """Tests for --bf16/--no-bf16 mixed precision flags."""

    def test_bf16_flag_in_help(self, cli_invoker):
        """``--bf16`` should appear in help output."""
        r = cli_invoker(["train", "run", "--help"])
        assert r.exit_code == 0
        assert "--bf16" in r.output
        assert "--no-bf16" in r.output

    def test_bf16_flag_shows_in_dump_config(self, cli_invoker, tmp_path):
        """``--bf16 --dump-config`` should show dtype: bf16."""
        r = cli_invoker([
            "train", "run", "--bf16", "--dump-config",
            "--dataset", str(tmp_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert cfg.get("dtype") == "bf16"

    def test_no_bf16_flag_shows_float32_in_dump_config(self, cli_invoker, tmp_path):
        """Without ``--bf16``, dump-config should show float32 (default)."""
        r = cli_invoker([
            "train", "run", "--dump-config",
            "--dataset", str(tmp_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert cfg.get("dtype") == "float32"

    def test_no_bf16_overrides_dtype_to_float32(self, cli_invoker, tmp_path):
        """``--no-bf16 --dump-config`` should always show dtype: float32."""
        r = cli_invoker([
            "train", "run", "--no-bf16", "--dump-config",
            "--dataset", str(tmp_path),
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
        import yaml
        cfg = yaml.safe_load(r.output)
        assert cfg.get("dtype") == "float32"

    def test_bf16_training_succeeds(self, cli_invoker, tmp_path):
        """Training with ``--bf16`` should complete successfully."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--bf16",
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output

    def test_no_bf16_training_succeeds(self, cli_invoker, tmp_path):
        """Training with ``--no-bf16`` should complete successfully."""
        from conftest import _create_dataset_dir
        dataset = _create_dataset_dir(tmp_path, 3)
        r = cli_invoker([
            "train", "run", "--dataset", str(dataset),
            "--no-bf16",
        ] + TRAIN_BASE_NO_VAL)
        assert r.exit_code == 0, r.output
