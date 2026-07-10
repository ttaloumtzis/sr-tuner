"""Model architectures and registry."""

from . import archs
from .archs import swinir, rrdbnet
from .registry import build_model
