"""Logging utilities."""

import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with the given *name* configured for sr-engine.

    Output goes to stdout with a ``[name] level: message`` format.
    """
    raise NotImplementedError("TODO: implement logger setup")
