"""Append-only audit trail: one JSON line per question and per clarification.

Each record says what was asked, which tables were offered to the model, what
SQL it wrote and how many tries it took, what the guardrails rewrote or
blocked, what actually ran, how long it took, and how it ended. Result *data*
is never logged, only column names and row counts, since results can contain
personal data. `request_id` ties a clarification and its follow-up together.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class QueryLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, event: str, response: Any, **extra: Any) -> None:
        record: Dict[str, Any] = {"event": event, **response.to_dict(include_rows=False), **extra}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def records(self, request_id: Optional[str] = None) -> List[dict]:
        return [r for r in self._iter() if request_id is None or r["request_id"] == request_id]

    def _iter(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
