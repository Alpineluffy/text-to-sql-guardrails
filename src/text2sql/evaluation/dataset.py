"""The golden dataset: knowledge/<db>/eval.yaml.

Each case says how the system *should* behave:

  expect: answer         (default) results must match the verified `sql`
  expect: clarify        ambiguous; the system should ask. `accept` optionally
                         lists the right SQL for each reading
  expect: cannot_answer  the database can't answer it; the system must say so
                         instead of inventing SQL

The SQL is written in the dialect named by semantic.yaml (`sql_dialect`) and
translated with sqlglot when the database is something else.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import yaml

from ..prompt.knowledge import translate_sql

EXPECTATIONS = ("answer", "clarify", "cannot_answer")


@dataclass
class GoldenCase:
    question: str
    expect: str = "answer"
    sql: Optional[str] = None
    accept: List[str] = field(default_factory=list)
    choice: Optional[str] = None  # option to pick for an `answer` case that gets a clarification
    tags: List[str] = field(default_factory=list)


def load_golden(knowledge_dir: Path, dialect: Optional[str] = None
                ) -> Tuple[Optional[date], List[GoldenCase]]:
    knowledge_dir = Path(knowledge_dir)
    data = yaml.safe_load((knowledge_dir / "eval.yaml").read_text())
    semantic = yaml.safe_load((knowledge_dir / "semantic.yaml").read_text()) or {}
    source = semantic.get("sql_dialect", "sqlite")
    today = data.get("today")
    if isinstance(today, str):
        today = date.fromisoformat(today)

    def tr(sql: Optional[str]) -> Optional[str]:
        return translate_sql(sql.strip(), source, dialect) if sql else None

    cases = []
    for c in data["cases"]:
        expect = c.get("expect", "answer")
        if expect not in EXPECTATIONS:
            raise ValueError(f"{c['question']!r}: unknown expect {expect!r}")
        if expect == "answer" and not c.get("sql"):
            raise ValueError(f"{c['question']!r}: an answer case needs sql")
        cases.append(GoldenCase(
            question=c["question"], expect=expect, sql=tr(c.get("sql")),
            accept=[tr(s) for s in c.get("accept", [])], choice=c.get("choice"),
            tags=list(c.get("tags", [])),
        ))
    return today, cases
