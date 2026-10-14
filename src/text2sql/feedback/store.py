"""Phase 4 Step 3: store what users say about answers.

Append-only JSON lines: a new item when feedback arrives, and a status record
when it is promoted or rejected; an item's current state is its latest record.

Feedback never changes the system directly. "Correct" answers become
*candidate* few-shot examples and corrected wrong answers become *candidate*
test cases; a reviewer promotes them (see feedback/review.py), after the SQL
is re-checked. One mistaken or malicious click can't teach the model bad SQL.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PENDING, PROMOTED, REJECTED = "pending", "promoted", "rejected"


@dataclass
class FeedbackItem:
    feedback_id: str
    request_id: str
    session_id: Optional[str]
    question: str
    sql: Optional[str]  # what the system ran
    verdict: str  # correct | incorrect
    corrected_sql: Optional[str] = None  # the user's fix, for incorrect answers
    comment: str = ""
    confidence_band: Optional[str] = None  # what the system thought, for calibration
    status: str = PENDING
    note: str = ""  # why it was promoted/rejected, or what it's waiting for
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def kind(self) -> str:
        """What it can become: an example, a test case, or nothing yet."""
        if self.verdict == "correct":
            return "example"
        return "test_case" if self.corrected_sql else "needs_correction"


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def add(self, request_id: str, question: str, sql: Optional[str], verdict: str,
            session_id: Optional[str] = None, corrected_sql: Optional[str] = None,
            comment: str = "", confidence_band: Optional[str] = None) -> FeedbackItem:
        if verdict not in ("correct", "incorrect"):
            raise ValueError("verdict must be 'correct' or 'incorrect'")
        item = FeedbackItem(
            feedback_id=uuid.uuid4().hex[:10], request_id=request_id, session_id=session_id,
            question=question, sql=sql, verdict=verdict,
            corrected_sql=(corrected_sql or "").strip() or None, comment=comment.strip(),
            confidence_band=confidence_band,
        )
        if item.kind == "needs_correction":
            item.note = "marked incorrect without a corrected query; needs one to become a test case"
        self._append({"type": "item", **asdict(item)})
        return item

    def set_status(self, feedback_id: str, status: str, note: str = "") -> FeedbackItem:
        item = self.get(feedback_id)
        if item is None:
            raise KeyError(feedback_id)
        self._append({"type": "status", "feedback_id": feedback_id, "status": status,
                      "note": note})
        item.status, item.note = status, note
        return item

    def get(self, feedback_id: str) -> Optional[FeedbackItem]:
        return self._items().get(feedback_id)

    def items(self, status: Optional[str] = None,
              session_id: Optional[str] = None) -> List[FeedbackItem]:
        return [i for i in self._items().values()
                if (status is None or i.status == status)
                and (session_id is None or i.session_id == session_id)]

    def _items(self) -> Dict[str, FeedbackItem]:
        items: Dict[str, FeedbackItem] = {}
        if not self.path.exists():
            return items
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            kind = record.pop("type")
            if kind == "item":
                items[record["feedback_id"]] = FeedbackItem(**record)
            elif kind == "status" and record["feedback_id"] in items:
                items[record["feedback_id"]].status = record["status"]
                items[record["feedback_id"]].note = record["note"]
        return items

    def _append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")
