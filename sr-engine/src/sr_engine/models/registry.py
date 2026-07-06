"""Model registry — name-to-class mapping with config-driven instantiation."""

from typing import Any
import torch.nn as nn

_registry: dict[str, type[nn.Module]] = {}

def register(name: str):
    def wrapper(cls: type[nn.Module]) -> type[nn.Module]:
        _registry[name] = cls
        print(f"Registered model: {name}") # Debugging aid
        return cls
    return wrapper

def build_model(name: str, config: dict[str, Any]) -> nn.Module:
    """Instantiate a model by looking up *name* in the registry."""
    if name not in _registry:
        raise ValueError(f"Model '{name}' not found in registry. "
                         f"Available models: {list(_registry.keys())}")

    # Instantiate the class, unpacking the dictionary as keyword arguments
    model_class = _registry[name]
    return model_class(**config)