"""Per-session history on disk, so it survives restarts.

One JSON-lines file per session (cache/history/<session_id>.jsonl), each line
a full response as the API returns it, with result rows capped. Unlike the
audit log (metadata only, for operators), this is the user's own record of
their questions and results, and is only served back to the same session.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Optional

_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class HistoryStore:
    def __init__(self, directory: Path, max_rows: int = 200):
        self.directory = Path(directory)
        self.max_rows = max_rows

    def append(self, session_id: str, response: Any) -> None:
        data = response.to_dict(include_rows=True)
        if data.get("rows"):
            data["rows"] = data["rows"][: self.max_rows]
        verification = data.get("verification") or {}
        multiquery = verification.get("multiquery") or {}
        if multiquery.get("alternative_rows"):
            multiquery["alternative_rows"] = multiquery["alternative_rows"][: self.max_rows]
        path = self._path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def load(self, session_id: str, limit: Optional[int] = None) -> List[dict]:
        """Oldest first, like the in-memory history."""
        path = self._path(session_id)
        if not path.exists():
            return []
        items = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return items[-limit:] if limit else items

    def find(self, session_id: str, request_id: str) -> Optional[dict]:
        """The latest answer for a request (a clarification and its answer share an id)."""
        matches = [i for i in self.load(session_id) if i.get("request_id") == request_id
                   and i.get("status") != "needs_clarification"]
        return matches[-1] if matches else None

    def _path(self, session_id: str) -> Path:
        if not _SAFE_ID.match(session_id):  # never let an id escape the directory
            raise ValueError(f"invalid session id: {session_id!r}")
        return self.directory / f"{session_id}.jsonl"
