# src/sr_engine/models/__init__.py
from . import archs  # This imports the archs package
from .archs import swinir, rrdbnet  # Explicitly import the modules to trigger @register
from .registry import build_model