"""The list of indexed demo repos (`<home>/repos.json`): name -> checkout path, namespace and index stats."""

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class RepoRecord:
    name: str
    root: str
    source: str
    namespace: str
    repo_id: str
    branch_id: str
    files: int = 0
    chunks: int = 0
    voyage_tokens: int = 0
    est_cost_usd: float = 0.0
    state: str = "absent"
    coverage: float = 0.0
    indexed_at: str = ""


class Registry:
    def __init__(self, home: Path) -> None:
        self._path = home / "repos.json"

    def load(self) -> dict[str, RepoRecord]:
        try:
            raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        out: dict[str, RepoRecord] = {}
        for name, rec in raw.items():
            try:
                out[name] = RepoRecord(**rec)
            except TypeError:
                continue
        return out

    def get(self, name: str) -> RepoRecord | None:
        return self.load().get(name)

    def put(self, rec: RepoRecord) -> None:
        data = self.load()
        data[rec.name] = rec
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=".repos-")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in data.items()}, f, indent=2, sort_keys=True)
        os.replace(tmp, self._path)
