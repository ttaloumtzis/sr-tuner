"""Tests for workspace.py — Workspace discovery, init, projects, check, dataset resolution."""

import os
from pathlib import Path

import pytest

from sr_engine.workspace import Workspace, MARKER


class TestWorkspaceInit:
    """Tests for ``Workspace.init``."""

    def test_init_creates_directories(self, tmp_path):
        """init() should create datasets, projects, configs dirs and the marker."""
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        assert (tmp_path / "my_ws" / "datasets").is_dir()
        assert (tmp_path / "my_ws" / "projects").is_dir()
        assert (tmp_path / "my_ws" / "configs").is_dir()
        assert (tmp_path / "my_ws" / MARKER).is_file()

    def test_init_is_idempotent(self, tmp_path):
        """Calling init() twice should not raise."""
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        ws.init()
        assert (tmp_path / "my_ws" / "datasets").is_dir()


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


class TestWorkspaceProjects:
    """Tests for project CRUD operations."""

    def test_create_project(self, tmp_path):
        """create_project() should create the project directory structure."""
        ws = Workspace(tmp_path)
        ws.init()
        proj = ws.create_project("my_proj")
        assert proj.name == "my_proj"
        assert proj.path == tmp_path / "projects" / "my_proj"
        assert (proj.path / "configs").is_dir()
        assert (proj.path / "checkpoints").is_dir()
        assert (proj.path / "metrics").is_dir()

    def test_create_duplicate_project_raises(self, tmp_path):
        """Creating a project with an existing name should raise FileExistsError."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("my_proj")
        with pytest.raises(FileExistsError):
            ws.create_project("my_proj")

    def test_list_projects(self, tmp_path):
        """list_projects() should return projects sorted by name."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj_b")
        ws.create_project("proj_a")
        projects = ws.list_projects()
        assert [p.name for p in projects] == ["proj_a", "proj_b"]

    def test_list_projects_empty(self, tmp_path):
        """list_projects() should return an empty list when no projects exist."""
        ws = Workspace(tmp_path)
        ws.init()
        assert ws.list_projects() == []

    def test_get_project_found(self, tmp_path):
        """get_project() should return the matching project."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("my_proj")
        proj = ws.get_project("my_proj")
        assert proj.name == "my_proj"

    def test_get_project_not_found(self, tmp_path):
        """get_project() should raise FileNotFoundError for unknown projects."""
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.get_project("nonexistent")


class TestWorkspaceCheck:
    """Tests for ``Workspace.check``."""

    def test_check_healthy(self, tmp_path):
        """A healthy workspace should return status 'ok'."""
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        report = ws.check()
        assert report["status"] == "ok"
        assert len(report["issues"]) == 0
        assert "proj1" in report["projects"]

    def test_check_missing_project_dir(self, tmp_path):
        """A workspace with a missing projects dir should report an error."""
        ws = Workspace(tmp_path)
        ws.init()
        (tmp_path / "projects").rmdir()
        report = ws.check()
        assert report["status"] == "error"
        assert len(report["issues"]) > 0


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
