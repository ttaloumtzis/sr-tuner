"""Workspace discovery, initialisation, model instances, versioning, and dataset resolution."""

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


MARKER = ".sr_workspace"
"""Filename used to mark a workspace root directory."""


@dataclass
class ModelInstance:
    """A named model instance in the workspace."""

    name: str
    path: Path


class Workspace:
    """Manages the workspace directory tree — models, datasets, configs."""

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

        Creates ``datasets/``, ``models/``, ``experiments/``, ``configs/``
        and the marker file.

        Args:
            reset_configs: If True, overwrite existing config files.
        """
        dirs = [
            self.path / "datasets",
            self.path / "models",
            self.path / "experiments",
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
            Dict with keys ``status``, ``issues``, ``path``, ``models``, ``datasets``.
        """
        issues = []
        structure_ok = True

        for name in ("datasets", "models", "experiments", "configs"):
            d = self.path / name
            if not d.is_dir():
                issues.append(f"Missing {name}/ directory")
                structure_ok = False

        if (self.path / "projects").is_dir():
            issues.append("Found old projects/ directory — structure was flattened. "
                          "Manually remove it after migrating instances.")

        if not (self.path / MARKER).is_file():
            issues.append("Missing .sr_workspace marker")

        try:
            instances = self.list_model_instances()
        except Exception as e:
            instances = []
            issues.append(f"Cannot list model instances: {e}")

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
            "models": [i.name for i in instances],
            "datasets": dataset_names,
        }

    def info(self) -> dict:
        """Return a summary of the workspace contents.

        Returns:
            Dict with keys ``path``, ``models``, ``datasets``.
        """
        instances = self.list_model_instances()
        datasets_dir = self.path / "datasets"
        dataset_names = sorted(
            d.name for d in datasets_dir.iterdir() if d.is_dir()
        ) if datasets_dir.is_dir() else []

        return {
            "path": str(self.path),
            "models": [i.name for i in instances],
            "datasets": dataset_names,
        }

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

    def create_model_instance(self, name: str, arch_config: dict) -> ModelInstance:
        """Create a named model instance in the workspace.

        Creates ``models/<name>/`` with ``config.yaml``,
        ``checkpoints/``, and ``runs/`` subdirectories.

        Args:
            name: Instance name.
            arch_config: Frozen model-architecture config dict.

        Returns:
            The new ModelInstance.

        Raises:
            FileExistsError: If the instance already exists.
        """
        inst_path = self.path / "models" / name
        if inst_path.exists():
            raise FileExistsError(
                f"Model instance '{name}' already exists"
            )
        (inst_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (inst_path / "versions").mkdir(parents=True, exist_ok=True)
        (inst_path / "runs").mkdir(parents=True, exist_ok=True)

        import yaml
        (inst_path / "config.yaml").write_text(
            yaml.safe_dump(arch_config, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )
        return ModelInstance(name=name, path=inst_path)

    def get_model_instance(self, name: str) -> ModelInstance:
        """Look up a model instance by name.

        Args:
            name: Instance name.

        Returns:
            The matching ModelInstance.

        Raises:
            FileNotFoundError: If the instance does not exist.
        """
        inst_path = self.path / "models" / name
        if not inst_path.is_dir():
            raise FileNotFoundError(
                f"Model instance '{name}' not found in workspace. "
                f"Create it with: sre model create-instance {name} --model <arch>"
            )
        return ModelInstance(name=name, path=inst_path)

    def list_model_instances(self) -> list[ModelInstance]:
        """Return all model instances in the workspace, sorted by name.

        Returns:
            List of ModelInstance dataclass instances.
        """
        models_dir = self.path / "models"
        if not models_dir.is_dir():
            return []
        return sorted(
            (
                ModelInstance(name=d.name, path=d)
                for d in models_dir.iterdir()
                if d.is_dir()
            ),
            key=lambda x: x.name,
        )

    # ── Model version API ─────────────────────────────────────────────

    def latest_model_version(self, instance_name: str) -> str | None:
        """Return the highest existing version tag (e.g. ``'v3'``) or ``None``."""
        versions_dir = self.path / "models" / instance_name / "versions"
        if not versions_dir.is_dir():
            return None
        versions = []
        for d in versions_dir.iterdir():
            m = re.fullmatch(r"v(\d+)", d.name)
            if d.is_dir() and m:
                versions.append((int(m.group(1)), d.name))
        if not versions:
            return None
        return max(versions, key=lambda x: x[0])[1]

    def next_model_version(self, instance_name: str) -> str:
        """Return the next available version tag (e.g. ``'v4'``).

        The slot is claimed atomically via ``mkdir`` to prevent races
        between concurrent processes.
        """
        versions_dir = self.path / "models" / instance_name / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        n = 1
        while True:
            candidate = f"v{n}"
            try:
                (versions_dir / candidate).mkdir(parents=False, exist_ok=False)
                return candidate
            except FileExistsError:
                n += 1

    def save_model_version(
        self, instance_name: str, version: str,
        state_dict: dict, metadata: dict | None = None,
    ) -> None:
        """Save a model version (bare state_dict on CPU + metadata JSON).

        Args:
            instance_name: Model instance name.
            version: Version tag (e.g. ``'v1'``).
            state_dict: Model state dict (will be moved to CPU).
            metadata: Optional dict saved as ``version.json``.
        """
        if torch is None:
            raise RuntimeError(
                "PyTorch is not available. Install it with: envs/build.sh --backend cpu"
            )
        v_path = self.path / "models" / instance_name / "versions" / version
        v_path.mkdir(parents=True, exist_ok=True)
        cpu_sd = {k: v.contiguous().cpu() for k, v in state_dict.items()}
        torch.save(cpu_sd, v_path / "model.pt")
        if metadata:
            (v_path / "version.json").write_text(
                json.dumps(metadata, indent=2, default=str) + "\n",
                encoding="utf-8",
            )

    def get_model_version_path(self, instance_name: str, version: str) -> Path:
        """Return the path to a model version's ``model.pt`` file."""
        return self.path / "models" / instance_name / "versions" / version / "model.pt"

    def resolve_version(self, instance_name: str, spec: str | None) -> Path | None:
        """Resolve a version specification to a ``model.pt`` path.

        Args:
            instance_name: Model instance name.
            spec: ``None``, ``'latest'``, or a tag like ``'v1'``.

        Returns:
            Path to the version's ``model.pt``, or ``None`` if no version exists.
        """
        if spec and spec != "latest":
            tag = spec
        else:
            tag = self.latest_model_version(instance_name)
            if tag is None:
                return None
        path = self.get_model_version_path(instance_name, tag)
        return path if path.is_file() else None

    def get_instance_checkpoints(self, instance: str) -> list[Path]:
        """List ``.pt`` checkpoint files for an instance, sorted by mtime descending.

        Args:
            instance: Instance name.

        Returns:
            List of checkpoint Paths (latest first).
        """
        ckpt_dir = self.path / "models" / instance / "checkpoints"
        if not ckpt_dir.is_dir():
            return []
        return sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)

    def list_runs(self, instance: str) -> list[Path]:
        """List ``run_*`` directories for an instance, sorted by mtime descending.

        Args:
            instance: Instance name.

        Returns:
            List of run directory Paths (latest first).
        """
        runs_dir = self.path / "models" / instance / "runs"
        if not runs_dir.is_dir():
            return []
        return sorted(
            (d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    def get_run_path(self, instance: str) -> Path:
        """Return a new timestamp-based run directory (creates it).

        The directory is named ``run_<YYYYMMDD_HHMMSS>``.

        Args:
            instance: Instance name.

        Returns:
            Path to the newly created run directory.
        """
        from datetime import datetime
        runs_dir = self.path / "models" / instance / "runs"
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = runs_dir / f"run_{run_ts}"
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir
