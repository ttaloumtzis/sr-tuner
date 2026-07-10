"""Progress reporting abstraction — decouples engine functions from terminal I/O.

Provides a no-op base class for library callers who do not need progress
feedback, and a ``tqdm``-based implementation for the CLI layer. GUI
consumers subclass ``ProgressReporter`` and pass the same engine functions
without any terminal coupling.
"""

from __future__ import annotations

from typing import Any, Optional


class ProgressReporter:
    """No-op progress reporter — safe default for programmatic callers.

    All methods are no-ops. Subclass and override to surface progress
    to a terminal, GUI, log stream, or WebSocket.
    """

    def start(self, total: Optional[int] = None, desc: str = "") -> None:
        """Begin tracking progress with an optional *total* and *desc*."""

    def update(self, n: int = 1) -> None:
        """Advance progress by *n* units."""

    def finish(self) -> None:
        """Signal that the tracked operation is complete."""

    def set_description(self, desc: str) -> None:
        """Update the description text without resetting progress."""

    def set_postfix(self, **kwargs: Any) -> None:
        """Display supplementary data alongside the progress indicator."""


class TqdmReporter(ProgressReporter):
    """Progress reporter that renders a ``tqdm`` progress bar."""

    def __init__(self, **tqdm_kwargs: Any) -> None:
        self._tqdm_kwargs = tqdm_kwargs
        self._bar = None

    def start(self, total: Optional[int] = None, desc: str = "") -> None:
        from tqdm import tqdm

        self._bar = tqdm(
            total=total,
            desc=desc,
            **self._tqdm_kwargs,
        )

    def update(self, n: int = 1) -> None:
        if self._bar is not None:
            self._bar.update(n)

    def finish(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def set_description(self, desc: str) -> None:
        if self._bar is not None:
            self._bar.set_description(desc)

    def set_postfix(self, **kwargs: Any) -> None:
        if self._bar is not None:
            self._bar.set_postfix(**kwargs)
