"""Phase 3 Step 3: answer complex questions twice, independently, and compare.

Agreement between two genuinely different approaches is strong evidence;
two calls with the same prompt tend to repeat the same mistake. So the second
attempt gets:
  - no few-shot examples (the first attempt's pattern source), and
  - an explicit instruction to use a different strategy, chosen from the
    *shape* of the first query (subqueries -> CTEs, joins -> pre-aggregate),
    without ever showing the first SQL.

Results are compared, not SQL text: row order, column names and tiny
rounding differences are ignored.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}


@dataclass
class Comparison:
    agree: Optional[bool]  # None: couldn't be compared fairly
    explanation: str


@dataclass
class MultiQueryCheck:
    ran: bool
    reason: str  # why it ran ("joins 3 tables") or why not
    alternative_sql: Optional[str] = None
    alternative_explanation: str = ""
    alternative_columns: List[str] = field(default_factory=list)
    alternative_rows: List[tuple] = field(default_factory=list)  # first rows only
    agreed: Optional[bool] = None
    explanation: str = ""


def complexity(sql: str, dialect: str = "sqlite") -> Optional[str]:
    """Why a query counts as complex enough to double-check, or None."""
    try:
        tree = sqlglot.parse_one(sql, read=SQLGLOT_DIALECTS.get(dialect, dialect))
    except ParseError:
        return None
    tables = {t.name.lower() for t in tree.find_all(exp.Table)}
    joins = list(tree.find_all(exp.Join))
    reasons = []
    if len(tables) >= 3:
        reasons.append(f"reads {len(tables)} tables")
    elif len(joins) >= 2:
        reasons.append(f"has {len(joins)} joins")
    if len(list(tree.find_all(exp.Select))) > 1:
        reasons.append("uses subqueries or CTEs")
    if tree.find(exp.Window):
        reasons.append("uses window functions")
    if tree.find(exp.Having):
        reasons.append("filters groups with HAVING")
    return ", ".join(reasons) or None


def strategy_for(sql: str, dialect: str = "sqlite") -> str:
    """A different way to solve it than the first query used."""
    try:
        tree = sqlglot.parse_one(sql, read=SQLGLOT_DIALECTS.get(dialect, dialect))
    except ParseError:
        return "Use a different formulation than the most obvious one."
    if tree.find(exp.Subquery) or tree.find(exp.Exists) or tree.find(exp.In):
        return ("Avoid subqueries in WHERE/FROM: build each step as a named CTE and "
                "combine them with explicit JOINs.")
    if tree.find(exp.Join):
        return ("Aggregate each table in its own CTE before joining, so that joins "
                "can't multiply rows, then join the aggregated results.")
    return ("Use a different formulation, e.g. conditional aggregation "
            "(SUM(CASE WHEN ...)) or a CTE, rather than the most obvious query.")


def compare_results(
    a_columns: Sequence[str], a_rows: Sequence[tuple], a_truncated: bool,
    b_columns: Sequence[str], b_rows: Sequence[tuple], b_truncated: bool,
) -> Comparison:
    if a_truncated or b_truncated:
        return Comparison(None, "a result was cut off at the row limit, so they can't be "
                                "compared in full")
    a, b = _normalise(a_rows), _normalise(b_rows)
    if len(a) != len(b):
        return Comparison(False, f"the approaches return different numbers of rows "
                                 f"({len(a)} vs {len(b)})")
    if Counter(a) == Counter(b) or _rows_close(a, b):
        return Comparison(True, f"both approaches return the same {len(a)} row(s)")
    # Same data with columns in another order, or one extra column (e.g. a label)?
    if _columns_match(a, b) or _columns_match(b, a):
        return Comparison(True, "both approaches return the same values "
                                "(columns differ in order or naming)")
    if len(a) == 1 and len(a[0]) == len(b[0]) == 1:
        x, y = a[0][0], b[0][0]
        if isinstance(x, float) and isinstance(y, float) and max(abs(x), abs(y)) > 0:
            gap = abs(x - y) / max(abs(x), abs(y))
            return Comparison(False, f"the approaches disagree: {x:,.2f} vs {y:,.2f} "
                                     f"({gap:.1%} apart)")
        return Comparison(False, f"the approaches disagree: {x} vs {y}")
    differing = sum((Counter(a) - Counter(b)).values())
    return Comparison(False, f"{differing} of {len(a)} rows differ between the approaches")


class MultiQueryChecker:
    def __init__(self, builder: Any, generator: Any, executor: Any, mode: str = "auto",
                 preview_rows: int = 10):
        """mode: "auto" (complex queries only), "always" or "never"."""
        if mode not in ("auto", "always", "never"):
            raise ValueError(f"unknown multi-query mode: {mode}")
        self.builder = builder
        self.generator = generator
        self.executor = executor
        self.mode = mode
        self.preview_rows = preview_rows

    def check(
        self,
        question: str,
        sql: str,
        columns: Sequence[str],
        rows: Sequence[tuple],
        truncated: bool,
        tables: Optional[Iterable[str]] = None,
        preferences: Optional[Dict[str, str]] = None,
        today: Optional[date] = None,
    ) -> MultiQueryCheck:
        dialect = self.builder.schema.dialect
        why = complexity(sql, dialect)
        if self.mode == "never":
            return MultiQueryCheck(False, "multi-query checking is off")
        if self.mode == "auto" and why is None:
            return MultiQueryCheck(False, "simple query; not double-checked")
        why = why or "requested for every query"

        prompt = self.builder.build(question, tables=tables, today=today,
                                    preferences=preferences, max_examples=0)
        prompt = prompt.with_section(
            "Approach",
            "Answer the question independently. " + strategy_for(sql, dialect)
            + " Correctness matters more than style; don't change what is being asked.")
        try:
            alternative = self.generator.generate(prompt)
        except Exception as e:
            return MultiQueryCheck(False, f"second approach failed to generate: {e}")
        if alternative.needs_clarification:
            return MultiQueryCheck(False, "the second approach found the question ambiguous")
        if not alternative.sql:
            return MultiQueryCheck(False, "no second approach: "
                                   + (alternative.explanation or "the model declined"))
        validation = getattr(alternative, "validation", None)
        if validation is not None and not validation.ok:
            return MultiQueryCheck(False, "the second approach didn't produce valid SQL",
                                   alternative_sql=alternative.sql)
        try:
            result = self.executor.run(alternative.sql)
        except Exception as e:
            return MultiQueryCheck(False, f"the second approach failed to run: {e}",
                                   alternative_sql=alternative.sql)

        comparison = compare_results(columns, rows, truncated,
                                     result.columns, result.rows, result.truncated)
        guardrail_sql = getattr(getattr(result, "guardrails", None), "sql", None)
        return MultiQueryCheck(
            ran=True, reason=why,
            alternative_sql=guardrail_sql or alternative.sql,
            alternative_explanation=alternative.explanation,
            alternative_columns=list(result.columns),
            alternative_rows=list(result.rows[: self.preview_rows]),
            agreed=comparison.agree, explanation=comparison.explanation,
        )


def _value(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float, Decimal)):
        return round(float(v), 2)  # ROUND(x, 2) in one approach and not the other
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    return str(v).strip()


def _normalise(rows: Sequence[tuple]) -> List[Tuple]:
    return [tuple(_value(v) for v in row) for row in rows]


def _close(x: Any, y: Any) -> bool:
    """Equal, or numbers within rounding distance: 2.565 rounds to 2.56 on SQLite
    (binary float) and 2.57 on PostgreSQL (exact decimal); both are right."""
    if isinstance(x, float) and isinstance(y, float):
        return abs(x - y) <= 0.0100001 or abs(x - y) <= 1e-6 * max(abs(x), abs(y))
    return x == y


def _rows_close(a: List[Tuple], b: List[Tuple]) -> bool:
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        return False

    def key(row):  # text first, then numbers coarsely, so near-equal rows line up
        return tuple((1, round(v, 1)) if isinstance(v, float) else (0, str(v)) for v in row)

    return all(_close(x, y) for ra, rb in zip(sorted(a, key=key), sorted(b, key=key))
               for x, y in zip(ra, rb))


def _columns_match(small: List[Tuple], large: List[Tuple]) -> bool:
    """Every column of `small` appears (as a whole column of values) in `large`."""
    if not small or not large or len(small[0]) > len(large[0]):
        return False
    large_columns = [Counter(col) for col in zip(*large)]
    used = set()
    for col in zip(*small):
        target = Counter(col)
        match = next((i for i, c in enumerate(large_columns)
                      if i not in used and c == target), None)
        if match is None:
            return False
        used.add(match)
    return True
