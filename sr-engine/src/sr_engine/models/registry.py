"""Model registry — name-to-class mapping with config-driven instantiation."""

from typing import Any
import torch.nn as nn

_registry: dict[str, type[nn.Module]] = {}


def register(name: str):
    """Decorator that registers an ``nn.Module`` subclass under *name*."""
    def wrapper(cls: type[nn.Module]) -> type[nn.Module]:
        _registry[name] = cls
        return cls
    return wrapper


def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate a model by looking up *name* in the registry and passing *config*.

    The config dict is unpacked as keyword arguments to the model constructor.
    """
    raise NotImplementedError("TODO: implement model instantiation from registry")
