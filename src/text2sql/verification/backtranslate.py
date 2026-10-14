"""Phase 3 Step 1: does the SQL answer the question that was asked?

Two calls, on purpose:

1. Describe: the model gets only the SQL and the schema, never the question,
   and says what question the SQL answers, broken into aspects (measure,
   filters, grouping, time range, ordering/limit). Showing it the question
   would anchor it into rationalising the SQL instead of reading it.
2. Compare: a second call lines that description up against the original
   question aspect by aspect. A different date range, a missing filter, or
   COUNT where SUM was meant is a mismatch with a specific explanation, which
   is far more useful than a single similarity number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, List, Optional

from ..prompt.dates import date_context

ASPECTS = ["measure", "filters", "grouping", "time_range", "ordering_limit", "other"]

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string",
                     "description": "The business question this SQL answers, in plain English."},
        "measures": {"type": "array", "items": {"type": "string"},
                     "description": "What is computed, e.g. 'sum of order total_amount'."},
        "filters": {"type": "array", "items": {"type": "string"},
                    "description": "Every row filter, with exact values."},
        "grouping": {"type": "array", "items": {"type": "string"}},
        "time_range": {"type": "string",
                       "description": "The date range covered, or 'all time'."},
        "ordering_limit": {"type": "string",
                           "description": "Sort order and row limit, or 'none'."},
    },
    "required": ["question", "measures", "filters", "grouping", "time_range", "ordering_limit"],
    "additionalProperties": False,
}

COMPARE_SCHEMA = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "aspect": {"type": "string", "enum": ASPECTS},
                    "verdict": {"type": "string", "enum": ["match", "mismatch", "not_applicable"]},
                    "explanation": {"type": "string"},
                },
                "required": ["aspect", "verdict", "explanation"],
                "additionalProperties": False,
            },
        },
        "alignment": {"type": "number",
                      "description": "0 to 1: how fully the SQL answers the question as asked."},
        "summary": {"type": "string"},
    },
    "required": ["aspects", "alignment", "summary"],
    "additionalProperties": False,
}

DESCRIBE_SYSTEM = """\
You read SQL queries and state exactly what they compute. You are not told \
why the query was written, so describe only what the SQL actually does: \
every aggregation, every filter with its exact values, joins that restrict or \
multiply rows, grouping, the date range (with exact dates), ordering and \
limits. If there is no date filter, the time range is "all time". Do not \
guess intent or fix mistakes; a reader will compare your description with \
what was asked."""

COMPARE_SYSTEM = """\
You check whether a SQL query answers a user's question. You get the user's \
question and an independent description of what the SQL computes, written by \
someone who never saw the question. Compare them aspect by aspect:

- measure: is the computed value what was asked (count vs sum vs average, \
which column, gross vs net)?
- filters: are all filters the question implies present, with the right \
values, and no filters that change the answer?
- grouping: is the breakdown what was asked?
- time_range: does the date range match, using the date context to resolve \
phrases like "last quarter"?
- ordering_limit: for "top N" or rankings, are the order and N right?

Be strict about anything that changes the answer; a wrong date range, a \
missing filter or the wrong metric is a mismatch. Rewording, extra helpful \
columns and harmless ordering are matches. Mark aspects the question doesn't \
touch as not_applicable. Alignment is 1.0 only when every applicable aspect \
matches."""


@dataclass
class AspectVerdict:
    aspect: str
    verdict: str
    explanation: str


@dataclass
class BackTranslation:
    back_translated_question: str
    description: dict  # the aspect-by-aspect description of the SQL
    aspects: List[AspectVerdict]
    alignment: float
    summary: str
    aligned: bool  # alignment above threshold and no aspect mismatched
    mismatches: List[str] = field(default_factory=list)


class BackTranslator:
    def __init__(self, llm: Any, threshold: float = 0.75):
        """`llm.complete(system, user, schema) -> (dict, model)`, e.g. ClaudeJSON."""
        self.llm = llm
        self.threshold = threshold

    def check(
        self,
        question: str,
        sql: str,
        schema_text: str,
        today: Optional[date] = None,
        glossary_text: str = "",
    ) -> BackTranslation:
        description, _ = self.llm.complete(
            DESCRIBE_SYSTEM,
            f"## Schema\n{schema_text}\n\n## SQL\n{sql}",
            DESCRIBE_SCHEMA,
        )
        compare_user = "\n\n".join(part for part in [
            f"## The user's question\n{question}",
            "## Independent description of the SQL\n" + _format_description(description),
            "## Date context\n" + "\n".join(date_context(today or date.today())),
            f"## Business glossary\n{glossary_text}" if glossary_text else "",
        ] if part)
        verdict, _ = self.llm.complete(COMPARE_SYSTEM, compare_user, COMPARE_SCHEMA)

        aspects = [AspectVerdict(a["aspect"], a["verdict"], a["explanation"])
                   for a in verdict.get("aspects", [])]
        mismatches = [f"{a.aspect.replace('_', ' ')}: {a.explanation}"
                      for a in aspects if a.verdict == "mismatch"]
        alignment = min(1.0, max(0.0, float(verdict.get("alignment", 0.0))))
        return BackTranslation(
            back_translated_question=description.get("question", ""),
            description=description,
            aspects=aspects,
            alignment=alignment,
            summary=verdict.get("summary", ""),
            aligned=alignment >= self.threshold and not mismatches,
            mismatches=mismatches,
        )


def _format_description(d: dict) -> str:
    lines = [f"Question it answers: {d.get('question', '')}"]
    for key in ("measures", "filters", "grouping"):
        values = d.get(key) or []
        lines.append(f"{key.capitalize()}: {'; '.join(values) if values else 'none'}")
    lines.append(f"Time range: {d.get('time_range', '')}")
    lines.append(f"Ordering/limit: {d.get('ordering_limit', '')}")
    return "\n".join(lines)
