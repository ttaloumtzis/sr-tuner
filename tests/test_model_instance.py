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
        inst = ws.create_model_instance("v1", {"name": "swinir", "scale": 4})
        assert isinstance(inst, ModelInstance)
        assert inst.name == "v1"
        assert (inst.path / "config.yaml").is_file()
        assert (inst.path / "checkpoints").is_dir()
        assert (inst.path / "runs").is_dir()

    def test_create_model_instance_writes_config(self, tmp_path):
        """config.yaml should contain the frozen arch config."""
        ws = Workspace(tmp_path)
        ws.init()
        arch = {"name": "swinir", "scale": 4, "num_in_ch": 3}
        ws.create_model_instance("v1", arch)
        import yaml
        loaded = yaml.safe_load(
            (tmp_path / "models" / "v1" / "config.yaml").read_text()
        )
        assert loaded == arch

    def test_create_model_instance_duplicate_raises(self, tmp_path):
        """Creating a duplicate instance should raise FileExistsError."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        with pytest.raises(FileExistsError):
            ws.create_model_instance("v1", {"name": "swinir"})


class TestGetModelInstance:
    """Tests for ``Workspace.get_model_instance``."""

    def test_get_model_instance_found(self, tmp_path):
        """get_model_instance() should return the matching instance."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        inst = ws.get_model_instance("v1")
        assert inst.name == "v1"

    def test_get_model_instance_not_found(self, tmp_path):
        """get_model_instance() should raise FileNotFoundError."""
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.get_model_instance("nonexistent")


class TestListModelInstances:
    """Tests for ``Workspace.list_model_instances``."""

    def test_list_model_instances(self, tmp_path):
        """list_model_instances() should return instances sorted by name."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("z_inst", {"name": "a"})
        ws.create_model_instance("a_inst", {"name": "b"})
        instances = ws.list_model_instances()
        assert [i.name for i in instances] == ["a_inst", "z_inst"]

    def test_list_model_instances_empty(self, tmp_path):
        """list_model_instances() should return empty list when none exist."""
        ws = Workspace(tmp_path)
        ws.init()
        assert ws.list_model_instances() == []

    def test_list_model_instances_no_models_dir(self, tmp_path):
        """list_model_instances() should return empty list if models/ dir missing."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        import shutil
        shutil.rmtree(tmp_path / "models")
        assert ws.list_model_instances() == []


class TestInstanceCheckpoints:
    """Tests for ``Workspace.get_instance_checkpoints``."""

    def test_get_instance_checkpoints(self, tmp_path):
        """get_instance_checkpoints() should return .pt files sorted by mtime."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        ckpt_dir = tmp_path / "models" / "v1" / "checkpoints"

        (ckpt_dir / "epoch_001.pt").write_text("a")
        import time
        time.sleep(0.02)
        (ckpt_dir / "epoch_002.pt").write_text("b")

        ckpts = ws.get_instance_checkpoints("v1")
        assert len(ckpts) == 2
        assert ckpts[0].name == "epoch_002.pt"
        assert ckpts[1].name == "epoch_001.pt"

    def test_get_instance_checkpoints_empty(self, tmp_path):
        """get_instance_checkpoints() should return empty list when no checkpoints."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        assert ws.get_instance_checkpoints("v1") == []


class TestListRuns:
    """Tests for ``Workspace.list_runs``."""

    def test_list_runs(self, tmp_path):
        """list_runs() should return run dirs sorted by mtime descending."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        runs_dir = tmp_path / "models" / "v1" / "runs"

        (runs_dir / "run_001").mkdir()
        import time
        time.sleep(0.02)
        (runs_dir / "run_002").mkdir()

        runs = ws.list_runs("v1")
        assert len(runs) == 2
        assert runs[0].name == "run_002"
        assert runs[1].name == "run_001"

    def test_list_runs_empty(self, tmp_path):
        """list_runs() should return empty list when no runs exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        assert ws.list_runs("v1") == []


class TestGetRunPath:
    """Tests for ``Workspace.get_run_path``."""

    def test_get_run_path_creates_dir(self, tmp_path):
        """get_run_path() should create a timestamped directory."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("v1", {"name": "swinir"})
        run_dir = ws.get_run_path("v1")
        assert run_dir.is_dir()
        assert run_dir.name.startswith("run_")


class TestModelVersions:
    """Tests for ``Workspace.list_model_versions``."""

    def test_list_model_versions_basic(self, tmp_path):
        """list_model_versions() returns versions sorted ascending."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v2").mkdir()
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "model.pt").write_text("dummy")
        versions = ws.list_model_versions("m1")
        assert [v["tag"] for v in versions] == ["v1", "v2"]
        assert versions[0]["has_weights"] is True
        assert versions[1]["has_weights"] is False

    def test_list_model_versions_corrupt_metadata(self, tmp_path):
        """Corrupt version.json is treated as empty metadata."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "version.json").write_text("{not json", encoding="utf-8")
        versions = ws.list_model_versions("m1")
        assert len(versions) == 1
        assert versions[0]["metadata"] == {}

    def test_list_model_versions_non_dict_metadata(self, tmp_path):
        """Non-dict JSON in version.json is treated as empty metadata."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "version.json").write_text('"just a string"', encoding="utf-8")
        versions = ws.list_model_versions("m1")
        assert len(versions) == 1
        assert versions[0]["metadata"] == {}

    def test_list_model_versions_empty(self, tmp_path):
        """list_model_versions() returns empty list when no versions exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        assert ws.list_model_versions("m1") == []


class TestLatestModelVersion:
    """Tests for ``Workspace.latest_model_version``."""

    def test_prefers_complete_version(self, tmp_path):
        """latest_model_version prefers a version with model.pt over a higher tag without."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "model.pt").write_text("dummy")
        (v_dir / "v2").mkdir()  # no model.pt
        assert ws.latest_model_version("m1") == "v1"

    def test_falls_back_to_highest_tag(self, tmp_path):
        """When no version has model.pt, returns the highest tag."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v2").mkdir()
        assert ws.latest_model_version("m1") == "v2"

    def test_no_versions(self, tmp_path):
        """Returns None when no versions exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        assert ws.latest_model_version("m1") is None

    def test_resolve_version_prefers_complete(self, tmp_path):
        """resolve_version('latest') returns a path only when the latest complete version has weights."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "model.pt").write_text("dummy")
        (v_dir / "v2").mkdir()
        path = ws.resolve_version("m1", "latest")
        assert path is not None
        assert path.name == "model.pt"
        assert path.parent.name == "v1"


class TestDeleteModelVersion:
    """Tests for ``Workspace.delete_model_version``."""

    def test_delete_version_success(self, tmp_path):
        """delete_model_version removes the version directory."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        v_dir = tmp_path / "models" / "m1" / "versions"
        (v_dir / "v1").mkdir()
        (v_dir / "v1" / "model.pt").write_text("dummy")
        assert (v_dir / "v1").is_dir()
        ws.delete_model_version("m1", "v1")
        assert not (v_dir / "v1").is_dir()

    def test_delete_version_missing_instance(self, tmp_path):
        """Raises FileNotFoundError when instance does not exist."""
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.delete_model_version("nonexistent", "v1")

    def test_delete_version_missing_version(self, tmp_path):
        """Raises FileNotFoundError when version does not exist."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        with pytest.raises(FileNotFoundError):
            ws.delete_model_version("m1", "v99")

    def test_delete_version_invalid_tag(self, tmp_path):
        """Raises ValueError for invalid tag format."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        with pytest.raises(ValueError):
            ws.delete_model_version("m1", "latest")
        with pytest.raises(ValueError):
            ws.delete_model_version("m1", "../../etc")

    def test_delete_version_containment_guard(self, tmp_path):
        """Prevents path traversal via version tag."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        # Create a directory outside the versions dir that should not be reachable
        outside = tmp_path / "outside"
        outside.mkdir()
        # '../outside' fails the v\d+ regex so ValueError is raised first
        with pytest.raises(ValueError):
            ws.delete_model_version("m1", "../outside")
