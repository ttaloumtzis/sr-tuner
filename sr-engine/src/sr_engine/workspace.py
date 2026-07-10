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


@dataclass
class ModelInstance:
    """A named model instance within a project."""

    name: str
    path: Path
    project: str


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

    # ── Model instance API ──────────────────────────────────────────

    def create_model_instance(self, project: str, name: str, arch_config: dict) -> ModelInstance:
        """Create a named model instance inside a project.

        Creates ``projects/<project>/models/<name>/`` with ``config.yaml``,
        ``checkpoints/``, and ``runs/`` subdirectories.

        Args:
            project: Project name.
            name: Instance name.
            arch_config: Frozen model-architecture config dict.

        Returns:
            The new ModelInstance.

        Raises:
            FileNotFoundError: If the project does not exist.
            FileExistsError: If the instance already exists.
        """
        project_path = self.path / "projects" / project
        if not project_path.is_dir():
            raise FileNotFoundError(f"Project '{project}' not found in workspace")
        inst_path = project_path / "models" / name
        if inst_path.exists():
            raise FileExistsError(
                f"Model instance '{name}' already exists in project '{project}'"
            )
        (inst_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (inst_path / "runs").mkdir(parents=True, exist_ok=True)

        import yaml
        (inst_path / "config.yaml").write_text(
            yaml.safe_dump(arch_config, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return ModelInstance(name=name, path=inst_path, project=project)

    def get_model_instance(self, project: str, name: str) -> ModelInstance:
        """Look up a model instance by project and name.

        Args:
            project: Project name.
            name: Instance name.

        Returns:
            The matching ModelInstance.

        Raises:
            FileNotFoundError: If the project or instance does not exist.
        """
        inst_path = self.path / "projects" / project / "models" / name
        if not inst_path.is_dir():
            raise FileNotFoundError(
                f"Model instance '{name}' not found in project '{project}'. "
                f"Create it with: sre model create-instance {project} {name} --model <arch>"
            )
        return ModelInstance(name=name, path=inst_path, project=project)

    def list_model_instances(self, project: str) -> list[ModelInstance]:
        """Return all model instances in a project, sorted by name.

        Args:
            project: Project name.

        Returns:
            List of ModelInstance dataclass instances.
        """
        models_dir = self.path / "projects" / project / "models"
        if not models_dir.is_dir():
            return []
        return sorted(
            (
                ModelInstance(name=d.name, path=d, project=project)
                for d in models_dir.iterdir()
                if d.is_dir()
            ),
            key=lambda x: x.name,
        )

    def get_instance_checkpoints(self, project: str, instance: str) -> list[Path]:
        """List ``.pt`` checkpoint files for an instance, sorted by mtime descending.

        Args:
            project: Project name.
            instance: Instance name.

        Returns:
            List of checkpoint Paths (latest first).
        """
        ckpt_dir = self.path / "projects" / project / "models" / instance / "checkpoints"
        if not ckpt_dir.is_dir():
            return []
        return sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    def list_runs(self, project: str, instance: str) -> list[Path]:
        """List ``run_*`` directories for an instance, sorted by mtime descending.

        Args:
            project: Project name.
            instance: Instance name.

        Returns:
            List of run directory Paths (latest first).
        """
        runs_dir = self.path / "projects" / project / "models" / instance / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted(
            (d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def get_run_path(self, project: str, instance: str) -> Path:
        """Return a new timestamp-based run directory (creates it).

        The directory is named ``run_<YYYYMMDD_HHMMSS>``.

        Args:
            project: Project name.
            instance: Instance name.

        Returns:
            Path to the newly created run directory.
        """
        from datetime import datetime
        runs_dir = self.path / "projects" / project / "models" / instance / "runs"
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
