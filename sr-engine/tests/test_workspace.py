import os
from pathlib import Path

import pytest

from sr_engine.workspace import Workspace, MARKER


class TestWorkspaceInit:
    def test_init_creates_directories(self, tmp_path):
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        assert (tmp_path / "my_ws" / "datasets").is_dir()
        assert (tmp_path / "my_ws" / "projects").is_dir()
        assert (tmp_path / "my_ws" / "configs").is_dir()
        assert (tmp_path / "my_ws" / MARKER).is_file()

    def test_init_is_idempotent(self, tmp_path):
        ws = Workspace(tmp_path / "my_ws")
        ws.init()
        ws.init()
        assert (tmp_path / "my_ws" / "datasets").is_dir()


class TestWorkspaceDiscover:
    def test_discover_from_root(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        found = Workspace.discover()
        # May or may not match depending on test CWD
        # Instead: explicitly test that discover finds a marker in a given path
        # by temporarily changing CWD
        old_cwd = Path.cwd()
        try:
            os.chdir(str(tmp_path))
            found = Workspace.discover()
            assert found is not None
            assert found.path == tmp_path
        finally:
            os.chdir(str(old_cwd))

    def test_discover_from_subdirectory(self, tmp_path):
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
        old_cwd = Path.cwd()
        try:
            os.chdir(str(tmp_path))
            found = Workspace.discover()
            assert found is None
        finally:
            os.chdir(str(old_cwd))


class TestWorkspaceProjects:
    def test_create_project(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        proj = ws.create_project("my_proj")
        assert proj.name == "my_proj"
        assert proj.path == tmp_path / "projects" / "my_proj"
        assert (proj.path / "configs").is_dir()
        assert (proj.path / "checkpoints").is_dir()
        assert (proj.path / "metrics").is_dir()

    def test_create_duplicate_project_raises(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("my_proj")
        with pytest.raises(FileExistsError):
            ws.create_project("my_proj")

    def test_list_projects(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj_b")
        ws.create_project("proj_a")
        projects = ws.list_projects()
        assert [p.name for p in projects] == ["proj_a", "proj_b"]

    def test_list_projects_empty(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        assert ws.list_projects() == []

    def test_get_project_found(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("my_proj")
        proj = ws.get_project("my_proj")
        assert proj.name == "my_proj"

    def test_get_project_not_found(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.get_project("nonexistent")


class TestWorkspaceCheck:
    def test_check_healthy(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        ws.create_project("proj1")
        report = ws.check()
        assert report["status"] == "ok"
        assert len(report["issues"]) == 0
        assert "proj1" in report["projects"]

    def test_check_missing_project_dir(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        (tmp_path / "projects").rmdir()
        report = ws.check()
        assert report["status"] == "error"
        assert len(report["issues"]) > 0


class TestWorkspaceResolveDataset:
    def test_resolves_absolute_path(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        d = tmp_path / "some" / "dataset"
        d.mkdir(parents=True)
        result = ws.resolve_dataset(d)
        assert result == d

    def test_resolves_existing_relative(self, tmp_path):
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
        ws = Workspace(tmp_path)
        ws.init()
        d = tmp_path / "datasets" / "my_set"
        d.mkdir(parents=True)
        result = ws.resolve_dataset(Path("my_set"))
        assert result == d

    def test_resolve_not_found(self, tmp_path):
        ws = Workspace(tmp_path)
        ws.init()
        with pytest.raises(FileNotFoundError):
            ws.resolve_dataset(Path("nonexistent"))
