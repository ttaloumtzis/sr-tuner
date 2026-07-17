"""Dotted-key expansion, config validation, and temp YAML I/O."""

import copy
from pathlib import Path


def expand_dotted_config(
    flat: dict, strip_prefix: str | None = None
) -> dict:
    """Convert dotted keys to nested dicts, optionally stripping a prefix.

    Args:
        flat: Flat dict like ``{"train.batch_size": 16, "train.losses.perceptual_weight": 0.1}``.
        strip_prefix: If set, strip this prefix from every key before expanding.

    Returns:
        Nested dict like ``{"batch_size": 16, "losses": {"perceptual_weight": 0.1}}``.

    Raises:
        ValueError: On key conflicts (e.g. ``a.b`` and ``a`` both present).
    """
    result: dict = {}
    for key, value in flat.items():
        if strip_prefix:
            if not key.startswith(strip_prefix + ".") and key != strip_prefix:
                continue
            key = key.removeprefix(strip_prefix + ".")
            key = key.removeprefix(strip_prefix)

        parts = key.split(".")
        current = result
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                if part in current and not isinstance(current[part], (int, float, str, bool, list, type(None))):
                    raise ValueError(
                        f"Key conflict: '{key}' collides with existing nested structure"
                    )
                current[part] = copy.deepcopy(value)
            else:
                if part not in current:
                    current[part] = {}
                elif not isinstance(current[part], dict):
                    raise ValueError(
                        f"Key conflict: '{'.'.join(parts[:i+1])}' is not a dict "
                        f"but '{key}' needs it to be one"
                    )
                current = current[part]
    return result


def validate_config_values(config: dict, schema: list[dict]) -> list[str]:
    """Validate config values against a schema.

    Args:
        config: Flat config dict (dotted keys) to validate.
        schema: List of param dicts from the config schema.

    Returns:
        List of error messages (empty if valid).
    """
    errors: list[str] = []
    schema_by_key = {p["key"]: p for p in schema}

    for key, value in config.items():
        if key not in schema_by_key:
            continue
        param = schema_by_key[key]
        param_type = param["type"]

        if param_type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                errors.append(f"{key}: expected int, got {type(value).__name__}")
            else:
                if "min" in param and value < param["min"]:
                    errors.append(f"{key}: value {value} is below minimum {param['min']}")
                if "max" in param and value > param["max"]:
                    errors.append(f"{key}: value {value} is above maximum {param['max']}")
                if "choices" in param and value not in param["choices"]:
                    errors.append(f"{key}: {value} is not a valid choice ({param['choices']})")

        elif param_type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                errors.append(f"{key}: expected float, got {type(value).__name__}")
            else:
                fval = float(value)
                if "min" in param and fval < param["min"]:
                    errors.append(f"{key}: value {fval} is below minimum {param['min']}")
                if "max" in param and fval > param["max"]:
                    errors.append(f"{key}: value {fval} is above maximum {param['max']}")

        elif param_type == "bool":
            if not isinstance(value, bool):
                errors.append(f"{key}: expected bool, got {type(value).__name__}")

        elif param_type == "choice":
            if value not in param.get("choices", []):
                errors.append(f"{key}: '{value}' is not a valid choice ({param.get('choices', [])})")

        elif param_type == "multi_choice":
            if isinstance(value, list):
                valid = set(param.get("choices", []))
                for v in value:
                    if v not in valid:
                        errors.append(f"{key}: '{v}' is not a valid choice ({param.get('choices', [])})")
            elif isinstance(value, str):
                if value not in param.get("choices", []):
                    errors.append(f"{key}: '{value}' is not a valid choice ({param.get('choices', [])})")

    return errors


def write_temp_config(
    jobs_dir: Path, job_id: str, config: dict, suffix: str = ""
) -> Path:
    """Write a config dict to a temp YAML file.

    Args:
        jobs_dir: Directory to write into.
        job_id: Job identifier used in the filename.
        suffix: Optional suffix before ``.yaml``.
        config: Config dict to serialise.

    Returns:
        Path to the written file.
    """
    import yaml
    jobs_dir.mkdir(parents=True, exist_ok=True)
    path = jobs_dir / f"{job_id}{suffix}.yaml"
    path.write_text(yaml.safe_dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return path