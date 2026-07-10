import json
from pathlib import Path


class MetricsStream:
    def __init__(self, path: Path, metadata: dict | None = None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        if metadata:
            self.file.write(f"# {json.dumps(metadata, default=str)}\n")
        self.file.flush()

    def write(self, msg: dict) -> None:
        self.file.write(json.dumps(msg, default=str) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()
