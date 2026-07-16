"""Model registry — name-to-class mapping with config-driven instantiation."""

from typing import Any
import torch.nn as nn

_registry: dict[str, type[nn.Module]] = {}


def register(name: str):
    """Decorator that registers an ``nn.Module`` subclass under *name*.

    Args:
        name: Registry key (used in config YAML to select the model).

    Returns:
        The decorated class unchanged.
    """
    def wrapper(cls: type[nn.Module]) -> type[nn.Module]:
        _registry[name] = cls
        return cls
    return wrapper


def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate a model by looking up *name* in the registry.

    Args:
        name: Registered model name.
        config: Dict of keyword arguments passed to the model constructor.

    Returns:
        An ``nn.Module`` instance.

    Raises:
        ValueError: If *name* is not registered.
    """
    if name not in _registry:
        raise ValueError(f"Model '{name}' not found in registry. "
                         f"Available models: {list(_registry.keys())}")

    model_class = _registry[name]
    return model_class(**config)
