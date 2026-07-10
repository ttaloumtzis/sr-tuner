"""JSONL streaming of training metrics to a file."""

import json
from pathlib import Path


class MetricsStream:
    """Append-only JSONL stream for training metrics."""

    def __init__(self, path: Path, metadata: dict | None = None) -> None:
        """Open a JSONL file for appending, optionally writing a metadata header.

        Args:
            path: Path to the ``.jsonl`` file.
            metadata: Optional dict written as a comment line at the top.
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        if metadata:
            self.file.write(f"# {json.dumps(metadata, default=str)}\n")
        self.file.flush()

    def write(self, msg: dict) -> None:
        """Append a JSON line to the stream.

        Args:
            msg: Dict to serialise as a JSON line.
        """
        self.file.write(json.dumps(msg, default=str) + "\n")
        self.file.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        self.file.close()
