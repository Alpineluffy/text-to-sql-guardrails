"""Phase 3 Step 2: is the result plausible given what we know about the data?

Deterministic checks against the Step 1 profile (real values, ranges, row
counts, null rates). Each flag says what is wrong and why it probably
happened:

  filter_values     WHERE status = 'delivered' but the column holds 'DELIVERED'
  date_filters      filters 2027 but the data ends 2026-09-30
  count_magnitude   COUNT larger than any table read: a join multiplied rows
  join_fanout       SUM/AVG of a column whose rows the joins repeat (measured by
                    re-running the query's joins: COUNT(*) vs COUNT(DISTINCT key))
  sum_magnitude     SUM larger than the column's total over the whole table
  value_range       an AVG/MIN/MAX outside the column's real range
  null_heavy        a mostly-NULL column that is rarely NULL in the table
  duplicate_rows    identical rows from a join without aggregation
  percentage_range  a percentage below 0 or above 100
  empty_result      no rows, with nothing else to explain it
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from ..schema.models import ColumnInfo, DatabaseSchema

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}
_PERCENT_NAME = re.compile(r"(pct|percent|percentage|share|ratio|rate)", re.IGNORECASE)
_DATE_FAMILIES = {"date", "datetime"}
_NUMERIC_FAMILIES = {"integer", "numeric"}
_FLIP = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE, exp.EQ: exp.EQ}


@dataclass
class SanityFlag:
    check: str
    severity: str  # error | warning
    message: str


@dataclass
class SanityReport:
    flags: List[SanityFlag] = field(default_factory=list)
    checks_run: List[str] = field(default_factory=list)

    @property
    def errors(self) -> List[SanityFlag]:
        return [f for f in self.flags if f.severity == "error"]

    @property
    def pass_rate(self) -> Optional[float]:
        if not self.checks_run:
            return None
        failed = {f.check for f in self.flags}
        return round((len(self.checks_run) - len(failed & set(self.checks_run)))
                     / len(self.checks_run), 3)


class ResultSanityChecker:
    def __init__(self, schema: DatabaseSchema, executor: Any = None):
        """`executor.run(sql)` is used to fetch whole-table column totals for
        the sum check; without it that check is skipped."""
        self.schema = schema
        self.executor = executor
        self.dialect = SQLGLOT_DIALECTS.get(schema.dialect, schema.dialect)
        self._columns: Dict[Tuple[str, str], ColumnInfo] = {
            (t.name.lower(), c.name.lower()): c for t in schema.tables for c in t.columns
        }
        self._row_counts = {t.name.lower(): t.row_count for t in schema.tables
                            if t.row_count is not None}
        self._mapping = {t.name.lower(): {c.name.lower(): "TEXT" for c in t.columns}
                         for t in schema.tables}
        self._totals: Dict[Tuple[str, str], Optional[float]] = {}

    def check(self, sql: str, columns: Sequence[str], rows: Sequence[tuple],
              truncated: bool = False) -> SanityReport:
        report = SanityReport()
        try:
            tree = qualify(sqlglot.parse_one(sql, read=self.dialect), schema=self._mapping,
                           dialect=self.dialect, validate_qualify_columns=False)
        except (ParseError, OptimizeError):
            return report
        aliases = _alias_map(tree)
        ctx = _Context(tree, aliases, list(columns), list(rows), truncated)

        self._check_filter_values(ctx, report)
        self._check_date_filters(ctx, report)
        if isinstance(tree, exp.Select):  # projections line up with result columns
            self._check_projections(ctx, report)
            self._check_duplicates(ctx, report)
        self._check_nulls(ctx, report)
        self._check_percentages(ctx, report)
        report.checks_run.append("empty_result")
        if not rows and not report.flags:
            report.flags.append(SanityFlag(
                "empty_result", "warning",
                "the query returned no rows; check that the filters match real values"))
        return report

    # --- before looking at results: are the filters even possible? ---------

    def _check_filter_values(self, ctx: "_Context", report: SanityReport) -> None:
        comparisons = []
        for node in ctx.tree.find_all(exp.EQ, exp.NEQ):
            col, lit = _column_and_literal(node)
            if col is not None:
                comparisons.append((col, lit, isinstance(node, exp.EQ)))
        for node in ctx.tree.find_all(exp.In):
            if isinstance(node.this, exp.Column):
                for value in node.expressions:
                    if isinstance(value, exp.Literal) and value.is_string:
                        comparisons.append((node.this, value, True))
        if not comparisons:
            return
        report.checks_run.append("filter_values")
        for col, lit, positive in comparisons:
            info = self._info(col, ctx)
            if info is None or info.profile is None or not lit.is_string:
                continue
            p, value = info.profile, lit.this
            effect = "matches nothing" if positive else "excludes nothing"
            if p.is_categorical and p.sample_values:
                known = [str(v) for v in p.sample_values]
                if value not in known:
                    close = next((k for k in known if k.lower() == value.lower()), None)
                    hint = f"; did you mean '{close}'?" if close else f" (values: {', '.join(known[:8])})"
                    report.flags.append(SanityFlag(
                        "filter_values", "error",
                        f"{info.name} has no value '{value}', so this filter {effect}{hint}"))
            elif p.value_index and not p.value_index_truncated and value not in p.value_index:
                close = next((k for k in p.value_index if k.lower() == value.lower()), None)
                hint = f"; did you mean '{close}'?" if close else ""
                report.flags.append(SanityFlag(
                    "filter_values", "warning",
                    f"'{value}' was not found in {info.name}, so this filter {effect}{hint}"))

    def _check_date_filters(self, ctx: "_Context", report: SanityReport) -> None:
        found = False
        for node in ctx.tree.find_all(exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.Between):
            if isinstance(node, exp.Between):
                info = self._date_info(node.this, ctx)
                low, high = _as_date(node.args.get("low")), _as_date(node.args.get("high"))
                if info is None or low is None or high is None:
                    continue
                found = True
                lo, hi = _profile_dates(info)
                if lo and hi and (high < lo or low > hi):
                    report.flags.append(SanityFlag(
                        "date_filters", "error",
                        f"{info.name} BETWEEN {low} AND {high} is outside the data "
                        f"({lo} to {hi}), so it matches nothing"))
                continue
            left, right = node.this, node.expression
            op = type(node)
            if isinstance(left, exp.Literal) and isinstance(right, exp.Column):
                left, right, op = right, left, _FLIP[op]
            info = self._date_info(left, ctx)
            value = _as_date(right)
            if info is None or value is None:
                continue
            found = True
            lo, hi = _profile_dates(info)
            if lo is None or hi is None:
                continue
            if op in (exp.GT, exp.GTE) and value > hi:
                report.flags.append(SanityFlag(
                    "date_filters", "error",
                    f"filters {info.name} from {value} on, but the data ends {hi}"))
            elif op in (exp.LT, exp.LTE) and value < lo:
                report.flags.append(SanityFlag(
                    "date_filters", "error",
                    f"filters {info.name} before {value}, but the data starts {lo}"))
            elif op is exp.EQ and not lo <= value <= hi:
                report.flags.append(SanityFlag(
                    "date_filters", "warning",
                    f"{info.name} = {value} is outside the data ({lo} to {hi})"))
        if found:
            report.checks_run.append("date_filters")

    # --- the result values -----------------------------------------------------

    def _check_projections(self, ctx: "_Context", report: SanityReport) -> None:
        tables_read = sorted(set(ctx.aliases.values()))
        largest = max((self._row_counts.get(t, 0) for t in tables_read), default=0)
        for i, projection in enumerate(ctx.tree.expressions):
            if i >= len(ctx.columns):
                break
            values = [v for v in (_number(r[i]) for r in ctx.rows) if v is not None]
            name = ctx.columns[i]
            node = _unwrap(projection)

            if isinstance(node, exp.Count) and largest:
                report.checks_run.append("count_magnitude")
                inner = node.this
                if isinstance(inner, exp.Distinct) and inner.expressions:
                    info = self._info(inner.expressions[0], ctx)
                    limit = self._row_counts.get(info.table, largest) if info else largest
                else:
                    limit = largest
                if values and max(values) > limit:
                    report.flags.append(SanityFlag(
                        "count_magnitude", "error",
                        f"{name} = {max(values):,.0f} is more than the {limit:,} rows in the "
                        f"largest table read ({', '.join(tables_read)}): a join is "
                        f"multiplying rows"))

            elif isinstance(node, (exp.Sum, exp.Avg)) and ctx.tree.args.get("joins"):
                self._check_fanout(ctx, node, name, report)

            elif isinstance(node, exp.Sum):
                info = self._info(node.this, ctx)
                if info is None or ctx.truncated or not values:
                    continue
                p = info.profile
                if p is None or p.min_value is None or _number(p.min_value) is None \
                        or _number(p.min_value) < 0:
                    continue  # with negative values a partial sum can exceed the total
                total = self._column_total(info)
                if total is None:
                    continue
                report.checks_run.append("sum_magnitude")
                if sum(values) > total * (1 + 1e-6) + 0.01:
                    report.flags.append(SanityFlag(
                        "sum_magnitude", "error",
                        f"{name} adds up to {sum(values):,.2f}, more than the total of "
                        f"{info.name} over the entire table ({total:,.2f}): rows were counted "
                        f"more than once, most likely by a join"))

            elif isinstance(node, (exp.Avg, exp.Min, exp.Max, exp.Column)):
                target = node if isinstance(node, exp.Column) else node.this
                info = self._info(target, ctx)
                if info is None or info.profile is None or info.type_family not in _NUMERIC_FAMILIES:
                    continue
                lo, hi = _number(info.profile.min_value), _number(info.profile.max_value)
                if lo is None or hi is None or not values:
                    continue
                report.checks_run.append("value_range")
                tolerance = max(abs(hi - lo) * 1e-6, 1e-6)
                outside = [v for v in values if v < lo - tolerance or v > hi + tolerance]
                if outside:
                    severity = "warning" if info.profile.is_sampled else "error"
                    report.flags.append(SanityFlag(
                        "value_range", severity,
                        f"{name} has values ({outside[0]:,.2f}) outside the real range of "
                        f"{info.name} ({lo:,.2f} to {hi:,.2f})"))

    def _check_fanout(self, ctx: "_Context", node: exp.Expression, name: str,
                      report: SanityReport) -> None:
        """Do the query's joins repeat rows of the table being summed/averaged?"""
        column = node.this
        info = self._info(column, ctx)
        if info is None or self.executor is None:
            return
        table = next(t for t in self.schema.tables if t.name.lower() == info.table)
        if len(table.primary_key) != 1:
            return
        key = exp.column(table.primary_key[0], table=column.table, quoted=True)
        probe = ctx.tree.copy()
        probe.set("expressions", [exp.Count(this=exp.Star()),
                                  exp.Count(this=exp.Distinct(expressions=[key]))])
        for arg in ("group", "having", "order", "limit", "offset", "distinct"):
            probe.set(arg, None)
        try:
            joined, distinct = self.executor.run(probe.sql(dialect=self.dialect)).rows[0]
        except Exception:
            return
        report.checks_run.append("join_fanout")
        if joined and distinct and joined > distinct:
            others = sorted({t for a, t in ctx.aliases.items() if t != info.table})
            what = "summed" if isinstance(node, exp.Sum) else "averaged"
            report.flags.append(SanityFlag(
                "join_fanout", "error",
                f"the query's joins (with {', '.join(others)}) repeat rows of {info.table} "
                f"({joined:,} joined rows for {distinct:,} {info.table}), so {info.name} is "
                f"{what} with some rows counted more than once; aggregate before joining"))

    def _check_duplicates(self, ctx: "_Context", report: SanityReport) -> None:
        tree = ctx.tree
        if not tree.args.get("joins") or tree.args.get("group") or tree.args.get("distinct"):
            return
        if any(e.find(exp.AggFunc) for e in tree.expressions) or not ctx.rows:
            return
        report.checks_run.append("duplicate_rows")
        dupes = sum(n - 1 for n in Counter(ctx.rows).values() if n > 1)
        if dupes:
            report.flags.append(SanityFlag(
                "duplicate_rows", "warning",
                f"{dupes} row(s) are exact duplicates: a join may be repeating rows "
                f"(check the join keys, or use DISTINCT if that's intended)"))

    def _check_nulls(self, ctx: "_Context", report: SanityReport) -> None:
        if len(ctx.rows) < 5:
            return
        report.checks_run.append("null_heavy")
        outer_join = any(j.args.get("side") for j in ctx.tree.find_all(exp.Join))
        projections = ctx.tree.expressions if isinstance(ctx.tree, exp.Select) else []
        for i, name in enumerate(ctx.columns):
            nulls = sum(1 for r in ctx.rows if r[i] is None) / len(ctx.rows)
            if nulls <= 0.5:
                continue
            source = (self._info(_unwrap(projections[i]), ctx)
                      if i < len(projections) and isinstance(_unwrap(projections[i]), exp.Column)
                      else None)
            usual = source.profile.null_fraction if source and source.profile else None
            if usual is not None and usual < 0.2:
                report.flags.append(SanityFlag(
                    "null_heavy", "warning",
                    f"{nulls:.0%} of {name} is NULL, but only {usual:.0%} of {source.name} is: "
                    f"likely an outer join that found no match, or a wrong join key"))
            elif outer_join and usual is None:
                report.flags.append(SanityFlag(
                    "null_heavy", "warning",
                    f"{nulls:.0%} of {name} is NULL after an outer join: check the join condition"))

    def _check_percentages(self, ctx: "_Context", report: SanityReport) -> None:
        for i, name in enumerate(ctx.columns):
            if not _PERCENT_NAME.search(name):
                continue
            values = [v for v in (_number(r[i]) for r in ctx.rows) if v is not None]
            if not values:
                continue
            report.checks_run.append("percentage_range")
            if min(values) < 0 or max(values) > 100:
                report.flags.append(SanityFlag(
                    "percentage_range", "error",
                    f"{name} ranges from {min(values):,.2f} to {max(values):,.2f}; "
                    f"a percentage should be between 0 and 100"))

    # --- helpers -------------------------------------------------------------

    def _info(self, node: Any, ctx: "_Context") -> Optional["_ColumnRef"]:
        if not isinstance(node, exp.Column):
            return None
        table = ctx.aliases.get(node.table)
        if table is None:
            return None
        info = self._columns.get((table, node.name.lower()))
        return _ColumnRef(table, info) if info else None

    def _date_info(self, node: Any, ctx: "_Context") -> Optional["_ColumnRef"]:
        ref = self._info(node, ctx)
        return ref if ref and ref.type_family in _DATE_FAMILIES else None

    def _column_total(self, ref: "_ColumnRef") -> Optional[float]:
        key = (ref.table, ref.column.name.lower())
        if key not in self._totals:
            total = None
            if self.executor is not None:
                sql = exp.select(exp.func("SUM", exp.column(ref.column.name, quoted=True))) \
                    .from_(exp.to_table(ref.table)).sql(dialect=self.dialect)
                try:
                    total = _number(self.executor.run(sql).rows[0][0])
                except Exception:
                    total = None
            self._totals[key] = total
        return self._totals[key]


@dataclass
class _ColumnRef:
    table: str
    column: ColumnInfo

    @property
    def name(self) -> str:
        return f"{self.table}.{self.column.name}"

    @property
    def profile(self):
        return self.column.profile

    @property
    def type_family(self) -> str:
        return self.column.type_family


@dataclass
class _Context:
    tree: exp.Expression
    aliases: Dict[str, str]
    columns: List[str]
    rows: List[tuple]
    truncated: bool


def _alias_map(tree: exp.Expression) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    try:
        scopes = traverse_scope(tree)
    except OptimizeError:
        return aliases
    for scope in scopes:
        for alias, source in scope.sources.items():
            if isinstance(source, exp.Table):
                aliases[alias] = source.name.lower()
    return aliases


def _unwrap(node: exp.Expression) -> exp.Expression:
    while isinstance(node, (exp.Alias, exp.Paren, exp.Cast, exp.Round)):
        node = node.this
    return node


def _column_and_literal(node: exp.Expression):
    left, right = node.this, node.expression
    if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
        return left, right
    if isinstance(right, exp.Column) and isinstance(left, exp.Literal):
        return right, left
    return None, None


def _as_date(node: Any) -> Optional[date]:
    if isinstance(node, exp.Literal) and node.is_string:
        return _parse_date(node.this)
    return None


def _parse_date(value: Any) -> Optional[date]:
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _profile_dates(ref: _ColumnRef):
    p = ref.profile
    if p is None:
        return None, None
    return _parse_date(p.min_value), _parse_date(p.max_value)


def _number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    return None
