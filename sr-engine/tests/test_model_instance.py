"""Tests for workspace.py — ModelInstance API."""

from pathlib import Path

import pytest

from sr_engine.workspace import Workspace, ModelInstance


class TestCreateModelInstance:
    """Tests for ``Workspace.create_model_instance``."""

    def test_create_model_instance_creates_dirs(self, tmp_path):
        """create_model_instance() should create the directory structure."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        inst = ws.create_model_instance("proj1", "v1", {"name": "swinir", "scale": 4})
        assert isinstance(inst, ModelInstance)
        assert inst.name == "v1"
        assert inst.project == "proj1"
        assert (inst.path / "config.yaml").is_file()
        assert (inst.path / "checkpoints").is_dir()
        assert (inst.path / "runs").is_dir()

    def test_create_model_instance_writes_config(self, tmp_path):
        """config.yaml should contain the frozen arch config."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        arch = {"name": "swinir", "scale": 4, "num_in_ch": 3}
        ws.create_model_instance("proj1", "v1", arch)
        import yaml
        loaded = yaml.safe_load(
            (tmp_path / "projects" / "proj1" / "models" / "v1" / "config.yaml").read_text()
        )
        assert loaded == arch

    def test_create_model_instance_duplicate_raises(self, tmp_path):
        """Creating a duplicate instance should raise FileExistsError."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        with pytest.raises(FileExistsError):
            ws.create_model_instance("proj1", "v1", {"name": "swinir"})

    def test_create_model_instance_missing_project_raises(self, tmp_path):
        """Creating an instance in a nonexistent project should raise FileNotFoundError."""
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.create_model_instance("nope", "v1", {"name": "swinir"})


class TestGetModelInstance:
    """Tests for ``Workspace.get_model_instance``."""

    def test_get_model_instance_found(self, tmp_path):
        """get_model_instance() should return the matching instance."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        inst = ws.get_model_instance("proj1", "v1")
        assert inst.name == "v1"
        assert inst.project == "proj1"

    def test_get_model_instance_not_found(self, tmp_path):
        """get_model_instance() should raise FileNotFoundError."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        with pytest.raises(FileNotFoundError):
            ws.get_model_instance("proj1", "nonexistent")


class TestListModelInstances:
    """Tests for ``Workspace.list_model_instances``."""

    def test_list_model_instances(self, tmp_path):
        """list_model_instances() should return instances sorted by name."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "z_inst", {"name": "a"})
        ws.create_model_instance("proj1", "a_inst", {"name": "b"})
        instances = ws.list_model_instances("proj1")
        assert [i.name for i in instances] == ["a_inst", "z_inst"]

    def test_list_model_instances_empty(self, tmp_path):
        """list_model_instances() should return empty list when none exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        assert ws.list_model_instances("proj1") == []

    def test_list_model_instances_no_models_dir(self, tmp_path):
        """list_model_instances() should return empty list if models/ dir missing."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        import shutil
        shutil.rmtree(tmp_path / "projects" / "proj1" / "models")
        assert ws.list_model_instances("proj1") == []


class TestInstanceCheckpoints:
    """Tests for ``Workspace.get_instance_checkpoints``."""

    def test_get_instance_checkpoints(self, tmp_path):
        """get_instance_checkpoints() should return .pt files sorted by mtime."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        ckpt_dir = tmp_path / "projects" / "proj1" / "models" / "v1" / "checkpoints"

        (ckpt_dir / "epoch_001.pt").write_text("a")
        import time
        time.sleep(0.02)
        (ckpt_dir / "epoch_002.pt").write_text("b")

        ckpts = ws.get_instance_checkpoints("proj1", "v1")
        assert len(ckpts) == 2
        assert ckpts[0].name == "epoch_002.pt"
        assert ckpts[1].name == "epoch_001.pt"

    def test_get_instance_checkpoints_empty(self, tmp_path):
        """get_instance_checkpoints() should return empty list when no checkpoints."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        assert ws.get_instance_checkpoints("proj1", "v1") == []


class TestListRuns:
    """Tests for ``Workspace.list_runs``."""

    def test_list_runs(self, tmp_path):
        """list_runs() should return run dirs sorted by mtime descending."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        runs_dir = tmp_path / "projects" / "proj1" / "models" / "v1" / "runs"

        (runs_dir / "run_001").mkdir()
        import time
        time.sleep(0.02)
        (runs_dir / "run_002").mkdir()

        runs = ws.list_runs("proj1", "v1")
        assert len(runs) == 2
        assert runs[0].name == "run_002"
        assert runs[1].name == "run_001"

    def test_list_runs_empty(self, tmp_path):
        """list_runs() should return empty list when no runs exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        assert ws.list_runs("proj1", "v1") == []


class TestGetRunPath:
    """Tests for ``Workspace.get_run_path``."""

    def test_get_run_path_creates_dir(self, tmp_path):
        """get_run_path() should create a timestamped directory."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        ws.create_model_instance("proj1", "v1", {"name": "swinir"})
        run_dir = ws.get_run_path("proj1", "v1")
        assert run_dir.is_dir()
        assert run_dir.name.startswith("run_")
