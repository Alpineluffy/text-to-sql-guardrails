"""Phase 3 Step 4 (schema coverage): does the SQL use what the question requires?

Expectations come from the question itself, so each one is specific:

  value       "orders from India"      -> filters country with 'India'
  glossary    "net revenue"            -> applies the definition's conditions
                                          ('CANCELLED' and 'RETURNED' excluded)
  time        "last quarter", "2025"   -> filters on a date column
  ranking     "top 5 products"         -> ORDER BY ... LIMIT 5
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ..prompt.knowledge import Knowledge
from ..prompt.render import find_value_mentions
from ..schema.models import DatabaseSchema

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|"
           "november|december")
TIME_PHRASE = re.compile(
    r"\b(last|this|previous|past|current)\s+(\d+\s+)?(day|week|month|quarter|year)s?\b"
    r"|\b(19|20)\d{2}\b|\bq[1-4]\b|\b(" + _MONTHS + r")\b"
    r"|\b(yesterday|today|ytd|since)\b|year[- ]to[- ]date",
    re.IGNORECASE,
)
RANKING = re.compile(
    r"\b(?:top|first|bottom)\s+(\d+)\b|\b(\d+)\s+(?:most|best|highest|largest|biggest|"
    r"lowest|least|worst|smallest)\b|^\s*(?:which|what)\s+(\d+)\b", re.IGNORECASE)
# "Which country has the most customers?", "our most expensive product": one winner.
SINGLE_WINNER = re.compile(
    r"^\s*(?:which|what)\b[^?]*?\b(?:has|had|is|was|brings in|brought in|generates|generated|"
    r"sells|sold)\s+(?:the|our)\s+(?:most|highest|largest|biggest|lowest|least|fewest|best)\b"
    r"|^\s*(?:which|what)\s+(?:is|was)\s+(?:our|the)\s+(?:most|least)\s+\w+",
    re.IGNORECASE)
_STRING = re.compile(r"'((?:[^']|'')*)'")
_QUALIFIED = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


@dataclass
class Expectation:
    kind: str  # value | glossary | time | ranking
    description: str
    satisfied: bool


@dataclass
class CoverageReport:
    expectations: List[Expectation] = field(default_factory=list)

    @property
    def score(self) -> Optional[float]:
        if not self.expectations:
            return None
        return round(sum(e.satisfied for e in self.expectations) / len(self.expectations), 3)

    @property
    def missing(self) -> List[Expectation]:
        return [e for e in self.expectations if not e.satisfied]


class CoverageChecker:
    def __init__(self, schema: DatabaseSchema, knowledge: Knowledge):
        self.schema = schema
        self.knowledge = knowledge
        self.dialect = SQLGLOT_DIALECTS.get(schema.dialect, schema.dialect)
        self._date_columns = {c.name.lower() for t in schema.tables for c in t.columns
                              if c.type_family in ("date", "datetime")}

    def check(self, question: str, sql: str,
              preferences: Optional[Dict[str, str]] = None) -> CoverageReport:
        report = CoverageReport()
        try:
            tree = sqlglot.parse_one(sql, read=self.dialect)
        except ParseError:
            return report
        literals = {lit.this for lit in tree.find_all(exp.Literal) if lit.is_string}
        column_names = {c.name.lower() for c in tree.find_all(exp.Column)}

        seen = set()
        for m in find_value_mentions(question, self.schema, [t.name for t in self.schema.tables]):
            key = (m.column.lower(), str(m.value))
            if key in seen:
                continue
            seen.add(key)
            report.expectations.append(Expectation(
                "value", f"filters {m.table}.{m.column} on '{m.value}' "
                         f"(the question mentions \"{m.phrase}\")",
                str(m.value) in literals and m.column.lower() in column_names))

        for match in self.knowledge.matching_terms(question, preferences):
            if match.is_ambiguous:
                continue  # clarification's job, not coverage's
            d = match.definitions[0]
            needed = [v.replace("''", "'") for v in _STRING.findall(d.sql)]
            if needed:
                ok = all(v in literals for v in needed)
                detail = ", ".join(f"'{v}'" for v in needed)
                report.expectations.append(Expectation(
                    "glossary", f"applies the definition of {d.name} (uses {detail})", ok))
            else:
                cols = {c.lower() for _, c in _QUALIFIED.findall(d.sql)}
                report.expectations.append(Expectation(
                    "glossary", f"applies the definition of {d.name}",
                    bool(cols & column_names)))

        if TIME_PHRASE.search(question):
            phrase = TIME_PHRASE.search(question).group(0)
            report.expectations.append(Expectation(
                "time", f"filters on a date (the question says \"{phrase}\")",
                self._filters_on_date(tree)))

        ranking = RANKING.search(question)
        n = None
        if ranking:
            n = int(next(g for g in ranking.groups() if g))
        elif SINGLE_WINNER.search(question):
            n = 1
        if n is not None:
            limit = tree.args.get("limit")
            value = limit.expression if limit is not None else None
            limited = isinstance(value, exp.Literal) and value.is_int and int(value.name) == n
            report.expectations.append(Expectation(
                "ranking", f"returns the top {n} in order (ORDER BY ... LIMIT {n})",
                limited and tree.args.get("order") is not None))
        return report

    def _filters_on_date(self, tree: exp.Expression) -> bool:
        for clause in list(tree.find_all(exp.Where)) + list(tree.find_all(exp.Having)):
            if any(c.name.lower() in self._date_columns for c in clause.find_all(exp.Column)):
                return True
        # Grouped by a date part (e.g. strftime('%Y', ...)) also scopes by time.
        for group in tree.find_all(exp.Group):
            if any(c.name.lower() in self._date_columns for c in group.find_all(exp.Column)):
                return True
        return False
