"""Logging configuration — structlog-based, dual-output (stdout + file)."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

import structlog


def _build_shared_processors() -> list:
    """Processors common to both structlog and the ProcessorFormatter."""
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def _build_formatter(renderer: structlog.types.Processor) -> structlog.stdlib.ProcessorFormatter:
    """Build a ProcessorFormatter with the given renderer.

    The formatter uses ``foreign_pre_chain`` to process events from
    third-party (non-structlog) loggers, and ``remove_processors_meta``
    to strip internal structlog metadata before rendering.
    """
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_build_shared_processors(),
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )


def configure_logging(
    log_format: Optional[str] = None,
    log_level: Optional[str] = None,
) -> None:
    """Phase 1: Configure structlog processors + stdout handler.

    Called once at app startup.  The file handler is added lazily by
    ``set_log_file()`` once the workspace path is known.
    """
    fmt = (log_format or os.getenv("SR_LOG_FORMAT", "json")).lower()
    lvl = (log_level or os.getenv("SR_LOG_LEVEL", "info")).upper()

    shared = _build_shared_processors()

    if fmt == "dev":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer()

    # structlog chain — ends with wrap_for_formatter so the event dict
    # is passed to stdlib's ProcessorFormatter for final rendering.
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    # Preserve file handlers (added lazily by set_log_file())
    file_handlers = [h for h in root.handlers
                     if isinstance(h, RotatingFileHandler)]
    root.handlers.clear()
    for h in file_handlers:
        root.addHandler(h)

    formatter = _build_formatter(renderer)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    root.addHandler(stdout_handler)


def set_log_file(workspace_path: Path) -> None:
    """Phase 2: Add a rotating JSON file handler under ``<workspace>/logs/``.

    Called by ``init_workspace()`` once the workspace path is known.
    The file is always JSON regardless of the ``SR_LOG_FORMAT`` setting.
    """
    log_dir = Path(os.getenv("SR_LOG_DIR", workspace_path / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "sr-engine.log"

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        _build_formatter(structlog.processors.JSONRenderer()),
    )
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named stdlib logger.

    No handlers, no propagation override — all formatting is handled
    centrally by ``configure_logging()``.  Use this everywhere instead
    of ``logging.getLogger()`` for consistency.
    """
    return logging.getLogger(name)