"""Review feedback and promote it into the knowledge files.

    python -m text2sql.feedback list                # pending items
    python -m text2sql.feedback promote <id>        # re-check, then add
    python -m text2sql.feedback reject <id> --reason "..."

Promotion re-runs the SQL through validation, the guardrails and the
sandbox, and requires a non-empty result, then appends:

  correct answer            -> knowledge/<db>/examples.yaml  (few-shot example)
  incorrect + corrected SQL -> knowledge/<db>/eval.yaml      (test case, gold = the fix)

Questions already in the target file are not added twice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List

import yaml

from .store import PROMOTED, REJECTED, FeedbackItem, FeedbackStore


class PromotionError(ValueError):
    pass


def check_sql(sql: str, validator: Any, executor: Any) -> List[str]:
    """Problems that keep `sql` out of the knowledge files (empty list = fine)."""
    report = validator.validate(sql)
    if not report.ok:
        return report.errors
    try:
        result = executor.run(sql)
    except Exception as e:
        return [f"doesn't run safely: {e}"]
    return [] if result.rows else ["returns no rows"]


def promote(item: FeedbackItem, store: FeedbackStore, knowledge_dir: Path,
            validator: Any, executor: Any) -> Path:
    if item.status != "pending":
        raise PromotionError(f"item {item.feedback_id} is already {item.status}")
    if item.kind == "needs_correction":
        raise PromotionError("marked incorrect without a corrected query; nothing to promote")
    sql = item.sql if item.kind == "example" else item.corrected_sql
    if not sql:
        raise PromotionError("no SQL to promote")
    problems = check_sql(sql, validator, executor)
    if problems:
        raise PromotionError("the SQL failed re-checking: " + "; ".join(problems))

    if item.kind == "example":
        target = Path(knowledge_dir) / "examples.yaml"
        existing = yaml.safe_load(target.read_text()) or [] if target.exists() else []
        questions = {e["question"].strip().lower() for e in existing}
        block = _yaml_entry(item.question, sql, indent="")
    else:
        target = Path(knowledge_dir) / "eval.yaml"
        data = yaml.safe_load(target.read_text()) if target.exists() else {"cases": []}
        questions = {c["question"].strip().lower() for c in data.get("cases", [])}
        block = _yaml_entry(item.question, sql, indent="  ", tags=["from-feedback"])
    if item.question.strip().lower() in questions:
        raise PromotionError(f"'{item.question}' is already in {target.name}")

    text = target.read_text() if target.exists() else ""
    target.write_text(text.rstrip("\n") + "\n\n" + block)
    yaml.safe_load(target.read_text())  # never leave a broken file behind
    store.set_status(item.feedback_id, PROMOTED, f"added to {target.name}")
    return target


def reject(item: FeedbackItem, store: FeedbackStore, reason: str) -> None:
    store.set_status(item.feedback_id, REJECTED, reason or "rejected by reviewer")


def _yaml_entry(question: str, sql: str, indent: str, tags: List[str] = ()) -> str:
    q = yaml.safe_dump(question, default_style='"', width=10_000).strip()
    body = "\n".join(f"{indent}    {line}" for line in sql.strip().splitlines())
    lines = [f"{indent}- question: {q}", f"{indent}  sql: |", body]
    if tags:
        lines.append(f"{indent}  tags: [{', '.join(tags)}]")
    return "\n".join(lines) + "\n"
