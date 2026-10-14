"""Measure hallucination detection by injecting known bugs into correct SQL.

A model that answers most questions correctly makes too few mistakes to
measure detection on. So each verified query is broken on purpose, in the
ways generated SQL typically goes wrong, and we count how many broken
queries the deterministic checks (result sanity + coverage, no API calls)
flag with a specific problem:

  value_case      'CANCELLED' -> 'cancelled'             (wrong spelling/case)
  wrong_value     'India' -> 'Indi'                       (value that doesn't exist)
  drop_date       remove the date filter                  (wrong time range)
  drop_status     remove the status filter                (glossary ignored)
  fanout_join     join order_items into an orders sum     (double counting)
  drop_limit      top-N without its LIMIT                 (ranking ignored)
  wrong_agg       SUM -> AVG, COUNT -> COUNT(DISTINCT ..) (wrong measure)

A mutant only counts if it runs and its result differs from the correct one,
so every counted mutant is a real wrong answer. The same checks run on the
unmodified queries to count false alarms.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from ..verification.coverage import CoverageChecker
from ..verification.multiquery import compare_results
from ..verification.sanity import ResultSanityChecker
from .dataset import GoldenCase

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}


@dataclass
class MutantResult:
    question: str
    kind: str
    sql: str
    detected: bool
    problems: List[str]


@dataclass
class MutationReport:
    mutants: List[MutantResult] = field(default_factory=list)
    gold_checked: int = 0
    false_alarms: List[str] = field(default_factory=list)  # correct queries that were flagged
    skipped: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def detection_rate(self) -> float:
        return _rate([m.detected for m in self.mutants])

    @property
    def false_alarm_rate(self) -> float:
        return len(self.false_alarms) / self.gold_checked if self.gold_checked else 0.0

    def by_kind(self) -> Dict[str, Dict[str, float]]:
        groups: Dict[str, List[bool]] = defaultdict(list)
        for m in self.mutants:
            groups[m.kind].append(m.detected)
        return {k: {"n": len(v), "detected": sum(v), "rate": _rate(v)}
                for k, v in sorted(groups.items())}

    def to_dict(self) -> dict:
        return {"detection_rate": self.detection_rate, "false_alarm_rate": self.false_alarm_rate,
                "mutants": len(self.mutants), "gold_checked": self.gold_checked,
                "by_kind": self.by_kind(), "false_alarms": self.false_alarms,
                "missed": [asdict(m) for m in self.mutants if not m.detected]}


class MutationSuite:
    def __init__(self, schema, knowledge, executor):
        """`executor` runs SQL safely (a GuardedExecutor or SandboxedExecutor)."""
        self.schema = schema
        self.executor = executor
        self.dialect = SQLGLOT_DIALECTS.get(schema.dialect, schema.dialect)
        self.sanity = ResultSanityChecker(schema, executor)
        self.coverage = CoverageChecker(schema, knowledge)
        self._categorical = {c.name.lower() for t in schema.tables for c in t.columns
                             if c.profile and (c.profile.is_categorical or c.profile.value_index)}
        self._dates = {c.name.lower() for t in schema.tables for c in t.columns
                       if c.type_family in ("date", "datetime")}
        self.mutators: Dict[str, Callable[[exp.Expression], Optional[exp.Expression]]] = {
            "value_case": self._value_case, "wrong_value": self._wrong_value,
            "drop_date": self._drop_date, "drop_status": self._drop_status,
            "fanout_join": self._fanout_join, "drop_limit": self._drop_limit,
            "wrong_agg": self._wrong_agg,
        }

    def run(self, cases: List[GoldenCase]) -> MutationReport:
        report = MutationReport()
        for case in cases:
            if case.expect != "answer" or not case.sql:
                continue
            gold = self.executor.run(case.sql)
            report.gold_checked += 1
            if self._problems(case.question, case.sql, gold):
                report.false_alarms.append(case.question)
            for kind, mutate in self.mutators.items():
                tree = self._parse(case.sql)
                mutant = mutate(tree) if tree is not None else None
                if mutant is None:
                    continue
                sql = mutant.sql(dialect=self.dialect)
                try:
                    result = self.executor.run(sql)
                except Exception:
                    report.skipped[f"{kind}: doesn't run"] += 1
                    continue
                same = compare_results(gold.columns, gold.rows, gold.truncated,
                                       result.columns, result.rows, result.truncated)
                if same.agree is not False:
                    report.skipped[f"{kind}: same result"] += 1  # not actually a wrong answer
                    continue
                problems = self._problems(case.question, sql, result)
                report.mutants.append(MutantResult(case.question, kind, sql, bool(problems),
                                                   problems))
        return report

    def _problems(self, question: str, sql: str, result) -> List[str]:
        final_sql = getattr(getattr(result, "guardrails", None), "sql", None) or sql
        sanity = self.sanity.check(final_sql, result.columns, result.rows, result.truncated)
        coverage = self.coverage.check(question, sql)
        return ([f.message for f in sanity.flags]
                + [f"not covered: {e.description}" for e in coverage.missing])

    def _parse(self, sql: str) -> Optional[exp.Expression]:
        try:
            return sqlglot.parse_one(sql, read=self.dialect)
        except ParseError:
            return None

    # --- mutators: each returns a changed tree, or None if it doesn't apply -------

    def _string_filter(self, tree):
        for node in tree.find_all(exp.EQ, exp.NEQ, exp.In):
            col = node.this if isinstance(node.this, exp.Column) else None
            values = (node.expressions if isinstance(node, exp.In)
                      else [node.expression])
            if col is not None and col.name.lower() in self._categorical:
                for lit in values:
                    if isinstance(lit, exp.Literal) and lit.is_string and lit.this:
                        return lit
        return None

    def _value_case(self, tree):
        lit = self._string_filter(tree)
        if lit is None:
            return None
        changed = lit.this.lower() if lit.this != lit.this.lower() else lit.this.upper()
        if changed == lit.this:
            return None
        lit.replace(exp.Literal.string(changed))
        return tree

    def _wrong_value(self, tree):
        lit = self._string_filter(tree)
        if lit is None or len(lit.this) < 3:
            return None
        lit.replace(exp.Literal.string(lit.this[:-1]))
        return tree

    def _drop_conjuncts(self, tree, predicate):
        """Remove the AND-ed WHERE conditions matching `predicate`."""
        where = tree.find(exp.Where)
        if where is None:
            return None
        conds = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
        keep = [c for c in conds if not predicate(c)]
        if len(keep) == len(conds):
            return None
        if keep:
            where.set("this", exp.and_(*[c.copy() for c in keep]))
        else:
            where.pop()
        return tree

    def _drop_date(self, tree):
        return self._drop_conjuncts(tree, lambda c: any(
            col.name.lower() in self._dates for col in c.find_all(exp.Column)))

    def _drop_status(self, tree):
        return self._drop_conjuncts(tree, lambda c: any(
            col.name.lower() == "status" for col in c.find_all(exp.Column)))

    def _fanout_join(self, tree):
        if not isinstance(tree, exp.Select) or tree.find(exp.Subquery) or tree.args.get("with"):
            return None
        orders = next((t for t in tree.find_all(exp.Table) if t.name.lower() == "orders"), None)
        if orders is None or any(t.name.lower() == "order_items" for t in tree.find_all(exp.Table)):
            return None
        sums_orders = any(
            isinstance(agg, (exp.Sum, exp.Avg)) and isinstance(agg.this, exp.Column)
            and agg.this.name.lower() == "total_amount"
            for e in tree.expressions for agg in e.find_all(exp.Sum, exp.Avg))
        if not sums_orders:
            return None
        alias = orders.alias_or_name
        return tree.join("order_items", on=f"fan_oi.order_id = {alias}.id",
                         join_alias="fan_oi", dialect=self.dialect)

    def _drop_limit(self, tree):
        if tree.args.get("limit") is None or tree.args.get("order") is None:
            return None
        tree.set("limit", None)
        return tree

    def _wrong_agg(self, tree):
        for agg in tree.find_all(exp.Sum):
            agg.replace(exp.Avg(this=agg.this.copy()))
            return tree
        for agg in tree.find_all(exp.Count):
            if isinstance(agg.this, exp.Star) and isinstance(tree, exp.Select) \
                    and tree.find(exp.Join) is not None:
                key = next(iter(tree.find_all(exp.Column)), None)
                if key is not None:
                    agg.replace(exp.Count(this=exp.Distinct(expressions=[key.copy()])))
                    return tree
        return None


def _rate(flags: List[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0
