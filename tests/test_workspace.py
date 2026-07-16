"""Tests for workspace.py — Workspace discovery, init, models, check, dataset resolution."""

import os
from pathlib import Path

import pytest

from sr_engine.workspace import Workspace, MARKER


class TestWorkspaceInit:
    """Tests for ``Workspace.init``."""

    def test_init_creates_directories(self, tmp_path):
        """init() should create models, datasets, experiments, configs dirs and the marker."""
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        assert (tmp_path / "my_ws" / "models").is_dir()
        assert (tmp_path / "my_ws" / "datasets").is_dir()
        assert (tmp_path / "my_ws" / "experiments").is_dir()
        assert (tmp_path / "my_ws" / "configs").is_dir()
        assert (tmp_path / "my_ws" / MARKER).is_file()

    def test_init_is_idempotent(self, tmp_path):
        """Calling init() twice should not raise."""
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        ws.init()
        assert (tmp_path / "my_ws" / "models").is_dir()


class TestWorkspaceDiscover:
    """Tests for ``Workspace.discover``."""

    def test_discover_from_root(self, tmp_path):
        """Discover should find a workspace when CWD is the workspace root."""
        ws = Workspace(tmp_path)
        ws.init()
        old_cwd = Path.cwd()
        try:
            os.chdir(str(tmp_path))
            found = Workspace.discover()
            assert found is not None
            assert found.path == tmp_path
        finally:
            os.chdir(str(old_cwd))

    def test_discover_from_subdirectory(self, tmp_path):
        """Discover should walk up from a subdirectory to find the workspace."""
        ws = Workspace(tmp_path)
        ws.init()
        sub = tmp_path / "sub" / "dir"
        sub.mkdir(parents=True)
        old_cwd = Path.cwd()
        try:
            os.chdir(str(sub))
            found = Workspace.discover()
            assert found is not None
            assert found.path == tmp_path
        finally:
            os.chdir(str(old_cwd))

    def test_discover_returns_none_when_no_workspace(self, tmp_path):
        """Discover should return None when no workspace marker exists."""
        old_cwd = Path.cwd()
        try:
            os.chdir(str(tmp_path))
            found = Workspace.discover()
            assert found is None
        finally:
            os.chdir(str(old_cwd))


class TestWorkspaceCheck:
    """Tests for ``Workspace.check``."""

    def test_check_healthy(self, tmp_path):
        """A healthy workspace should return status 'ok'."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_model_instance("m1", {"name": "swinir"})
        report = ws.check()
        assert report["status"] == "ok"
        assert len(report["issues"]) == 0
        assert "m1" in report["models"]

    def test_check_missing_models_dir(self, tmp_path):
        """A workspace with a missing models dir should report an error."""
        ws = Workspace(tmp_path)
        ws.init()
        import shutil
        shutil.rmtree(tmp_path / "models")
        report = ws.check()
        assert report["status"] == "error"
        assert len(report["issues"]) > 0

    def test_check_detects_old_projects_dir(self, tmp_path):
        """An old projects/ dir should be flagged as an issue."""
        ws = Workspace(tmp_path)
        ws.init()
        (tmp_path / "projects").mkdir(exist_ok=True)
        report = ws.check()
        assert any("projects" in i.lower() for i in report["issues"])


class TestWorkspaceResolveDataset:
    """Tests for ``Workspace.resolve_dataset``."""

    def test_resolves_absolute_path(self, tmp_path):
        """An absolute path should be returned as-is."""
        ws = Workspace(tmp_path)
        ws.init()
        d = tmp_path / "some" / "dataset"
        d.mkdir(parents=True)
        result = ws.resolve_dataset(d)
        assert result == d

    def test_resolves_existing_relative(self, tmp_path):
        """A CWD-relative path should be resolved."""
        ws = Workspace(tmp_path)
        ws.init()
        old_cwd = Path.cwd()
        try:
            os.chdir(str(tmp_path))
            d = Path("local_dataset")
            d.mkdir()
            result = ws.resolve_dataset(d)
            assert result == d.resolve()
        finally:
            os.chdir(str(old_cwd))

    def test_resolves_workspace_dataset(self, tmp_path):
        """A dataset name should be resolved inside the workspace datasets dir."""
        ws = Workspace(tmp_path)
        ws.init()
        d = tmp_path / "datasets" / "my_set"
        d.mkdir(parents=True)
        result = ws.resolve_dataset(Path("my_set"))
        assert result == d

    def test_resolve_not_found(self, tmp_path):
        """An unresolvable dataset should raise FileNotFoundError."""
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.resolve_dataset(Path("nonexistent"))
