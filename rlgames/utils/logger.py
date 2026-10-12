import json
import os


class MetricsWriter:
    """Appends one JSON object per line to `<save_dir>/metrics.jsonl`."""

    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "metrics.jsonl")
        self.file = open(self.path, "a")

    def write(self, record: dict) -> None:
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
