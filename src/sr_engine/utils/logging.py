"""Logging utilities."""

import logging
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger with the given *name* configured for sr-engine.

    Output goes to stdout with a ``[name] level: message`` format.

    Safe to call multiple times with the same *name* (e.g. from different
    modules) — it won't attach duplicate handlers, which would otherwise
    cause every log line to be printed multiple times.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter(f"[{name}] %(levelname)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Avoid double-printing through the root logger's own handlers, since
    # this logger already has its own dedicated stdout handler.
    logger.propagate = False

    return logger