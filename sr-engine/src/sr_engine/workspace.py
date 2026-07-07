import json
from dataclasses import dataclass
from pathlib import Path


MARKER = ".sr_workspace"


@dataclass
class Project:
    name: str
    path: Path


class Workspace:
    def __init__(self, path: Path):
        self.path = path.resolve()

    @classmethod
    def discover(cls) -> "Workspace | None":
        cwd = Path.cwd().resolve()
        for parent in [cwd] + list(cwd.parents):
            marker = parent / MARKER
            if marker.is_file():
                return cls(path=parent)
        return None

    def init(self) -> None:
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

    def check(self) -> dict:
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
        project_path = self.path / "projects" / name
        if project_path.exists():
            raise FileExistsError(f"Project '{name}' already exists at {project_path}")
        for sub in ("configs", "checkpoints", "metrics"):
            (project_path / sub).mkdir(parents=True, exist_ok=True)
        return Project(name=name, path=project_path)

    def list_projects(self) -> list[Project]:
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
        project_path = self.path / "projects" / name
        if not project_path.is_dir():
            raise FileNotFoundError(
                f"Project '{name}' not found in workspace. "
                f"Available: {[p.name for p in self.list_projects()]}"
            )
        return Project(name=name, path=project_path)

    def resolve_dataset(self, name_or_path: Path) -> Path:
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
