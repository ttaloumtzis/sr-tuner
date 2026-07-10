"""Workspace discovery, initialisation, project CRUD, and dataset resolution."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


MARKER = ".sr_workspace"
"""Filename used to mark a workspace root directory."""


@dataclass
class Project:
    """A named project directory within a workspace."""

    name: str
    path: Path


class Workspace:
    """Manages the workspace directory tree — projects, datasets, configs."""

    def __init__(self, path: Path):
        """Initialise with the resolved workspace root path.

        Args:
            path: Workspace root directory (will be resolved to absolute).
        """
        self.path = path.resolve()

    @classmethod
    def discover(cls) -> "Workspace | None":
        """Walk up from CWD to find a workspace marker.

        Returns:
            Workspace if a marker file is found, else None.
        """
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            marker = parent / MARKER
            if marker.is_file():
                return cls(path=parent)
        return None

    def init(self, reset_configs: bool = False) -> None:
        """Create the workspace directory structure and copy built-in configs.

        Creates ``datasets/``, ``projects/``, ``configs/`` and the marker file.

        Args:
            reset_configs: If True, overwrite existing config files.
        """
        dirs = [
            self.path / "datasets",
            self.path / "projects",
            self.path / "configs",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        marker = self.path / MARKER
        if not marker.exists():
            marker.write_text(
                json.dumps({"version": 1, "created": str(Path.cwd())}, indent=2) + "\n"
            )

        self._copy_builtin_configs(reset_configs)

    def _copy_builtin_configs(self, reset: bool = False) -> None:
        """Copy default YAML configs from the package into the workspace configs dir.

        Args:
            reset: If True, overwrite any existing config files.
        """
        from sr_engine.utils.config import DefaultConfigs

        src = DefaultConfigs.builtin_config_path()
        dst_root = self.path / "configs"

        for sub_dir in ("train", "datasets", "models"):
            src_sub = src / sub_dir
            if not src_sub.is_dir():
                continue
            (dst_root / sub_dir).mkdir(parents=True, exist_ok=True)
            for fpath in src_sub.iterdir():
                if fpath.suffix not in (".yaml", ".yml"):
                    continue
                dst = dst_root / sub_dir / fpath.name
                if reset or not dst.exists():
                    shutil.copy2(fpath, dst)

    def check(self) -> dict:
        """Validate workspace structure and return a health report.

        Returns:
            Dict with keys ``status``, ``issues``, ``path``, ``projects``, ``datasets``.
        """
        issues = []
        structure_ok = True

        for name in ("datasets", "projects", "configs"):
            d = self.path / name
            if not d.is_dir():
                issues.append(f"Missing {name}/ directory")
                structure_ok = False

        if not (self.path / MARKER).is_file():
            issues.append("Missing .sr_workspace marker")

        try:
            projects = self.list_projects()
        except Exception as e:
            projects = []
            issues.append(f"Cannot list projects: {e}")

        dataset_names = []
        datasets_dir = self.path / "datasets"
        if datasets_dir.is_dir():
            dataset_names = sorted(
                d.name for d in datasets_dir.iterdir() if d.is_dir()
            )

        status = "ok" if not issues else ("warn" if structure_ok else "error")

        return {
            "status": status,
            "issues": issues,
            "path": str(self.path),
            "projects": [p.name for p in projects],
            "datasets": dataset_names,
        }

    def info(self) -> dict:
        """Return a summary of the workspace contents.

        Returns:
            Dict with keys ``path``, ``projects``, ``datasets``.
        """
        projects = self.list_projects()
        datasets_dir = self.path / "datasets"
        dataset_names = sorted(
            d.name for d in datasets_dir.iterdir() if d.is_dir()
        ) if datasets_dir.is_dir() else []

        return {
            "path": str(self.path),
            "projects": [p.name for p in projects],
            "datasets": dataset_names,
        }

    def create_project(self, name: str) -> Project:
        """Create a new project directory under the workspace.

        Args:
            name: Project name (used as directory name).

        Returns:
            The newly created Project.

        Raises:
            FileExistsError: If the project directory already exists.
        """
        project_path = self.path / "projects" / name
        if project_path.exists():
            raise FileExistsError(f"Project '{name}' already exists at {project_path}")
        for sub in ("configs", "checkpoints", "metrics"):
            (project_path / sub).mkdir(parents=True, exist_ok=True)
        return Project(name=name, path=project_path)

    def list_projects(self) -> list[Project]:
        """Return all projects sorted by name.

        Returns:
            List of Project dataclass instances.
        """
        projects_dir = self.path / "projects"
        if not projects_dir.is_dir():
            return []
        return sorted(
            (
                Project(name=d.name, path=d)
                for d in projects_dir.iterdir()
                if d.is_dir()
            ),
            key=lambda p: p.name,
        )

    def get_project(self, name: str) -> Project:
        """Look up a project by name.

        Args:
            name: Project name.

        Returns:
            The matching Project.

        Raises:
            FileNotFoundError: If the project does not exist.
        """
        project_path = self.path / "projects" / name
        if not project_path.is_dir():
            raise FileNotFoundError(
                f"Project '{name}' not found in workspace. "
                f"Available: {[p.name for p in self.list_projects()]}"
            )
        return Project(name=name, path=project_path)

    def resolve_dataset(self, name_or_path: Path) -> Path:
        """Resolve a dataset reference to an absolute path.

        Checks, in order: absolute path, CWD-relative path, workspace datasets dir.

        Args:
            name_or_path: Dataset name or path.

        Returns:
            Resolved absolute Path.

        Raises:
            FileNotFoundError: If the dataset cannot be found.
        """
        if name_or_path.is_absolute():
            return name_or_path
        resolved_cwd = name_or_path.resolve()
        if resolved_cwd.exists():
            return resolved_cwd
        resolved_ws = (self.path / "datasets" / name_or_path).resolve()
        if resolved_ws.exists():
            return resolved_ws
        raise FileNotFoundError(
            f"Dataset '{name_or_path}' not found "
            f"(checked CWD: {resolved_cwd}, workspace: {resolved_ws})"
        )
