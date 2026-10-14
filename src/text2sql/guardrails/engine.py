"""Phase 2 Step 2: the safety layer every query passes before it runs.

Rules, cheapest first (EXPLAIN is last because it's the only one that touches
the database):

  rule                 kind       on failure
  -------------------  ---------  ------------------------------------------
  parse                hard       refuse
  single_statement     hard       refuse
  read_only            hard       refuse   (allow-list: only SELECT/UNION/...)
  blocked_function     hard       refuse   (file access, sleep, locks, ...)
  subquery_depth       retryable  model may rewrite it
  references           retryable  model may rewrite it
  table_access         retryable  model may rewrite it
  sensitive_columns    retryable  model may rewrite it
  cartesian_join       retryable  model may rewrite it
  row_limit            rewrite    LIMIT added or lowered, query still runs
  scan_estimate        retryable  model may add filters

"Hard" violations mean the query is unsafe whatever the intent, so it is never
sent back for repair. Every blocked or rewritten query is written to the audit
log with its reasons.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from ..schema.models import DatabaseSchema
from .config import GuardrailConfig
from .explain import ScanEstimate, estimate_postgres, estimate_sqlite

logger = logging.getLogger(__name__)
logging.getLogger("sqlglot").setLevel(logging.ERROR)  # "falling back to Command" noise

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}

_QUERY_ROOTS = tuple(c for c in (getattr(exp, n, None) for n in (
    "Select", "Union", "Intersect", "Except", "Subquery")) if c)
# Anything that writes, changes settings, touches files or runs commands.
_FORBIDDEN = tuple(c for c in (getattr(exp, n, None) for n in (
    "Insert", "Update", "Delete", "Merge", "Create", "Drop", "Alter", "AlterColumn",
    "TruncateTable", "Command", "Pragma", "Attach", "Detach", "Grant", "Revoke", "Copy",
    "Set", "Use", "Transaction", "Commit", "Rollback", "LoadData", "Cache", "Uncache",
    "Describe", "Kill", "Analyze", "Comment",
)) if c)


@dataclass
class Violation:
    rule: str
    message: str
    retryable: bool


@dataclass
class GuardrailDecision:
    original_sql: str
    sql: str  # what will run: the original, or rewritten (e.g. LIMIT added)
    violations: List[Violation] = field(default_factory=list)
    rewrites: List[str] = field(default_factory=list)
    scan: Optional[ScanEstimate] = None
    tables: List[str] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    applied_limit: Optional[int] = None  # set when the row-limit rule added/lowered LIMIT

    @property
    def allowed(self) -> bool:
        return not self.violations

    @property
    def retryable(self) -> bool:
        """True when every violation is something the model can fix by rewriting."""
        return bool(self.violations) and all(v.retryable for v in self.violations)

    def reasons(self) -> List[str]:
        return [f"[{v.rule}] {v.message}" for v in self.violations]


class GuardrailError(RuntimeError):
    def __init__(self, decision: GuardrailDecision):
        self.decision = decision
        super().__init__("Blocked by guardrails: " + "; ".join(decision.reasons()))


class AuditLog:
    """Append-only JSON-lines log of blocked and rewritten queries."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def write(self, event: str, decision: GuardrailDecision) -> None:
        self.write_event(
            event,
            sql=decision.original_sql,
            final_sql=decision.sql,
            violations=[vars(v) for v in decision.violations],
            rewrites=decision.rewrites,
            scan_rows=decision.scan.rows if decision.scan else None,
        )

    def write_event(self, event: str, **fields: Any) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line]


class Guardrails:
    def __init__(
        self,
        schema: DatabaseSchema,
        config: GuardrailConfig = GuardrailConfig(),
        explain: Optional[Callable[[str], Any]] = None,
        audit: Optional[AuditLog] = None,
    ):
        """`explain(sql)` returns the raw plan: SQLite EXPLAIN QUERY PLAN rows,
        or PostgreSQL EXPLAIN (FORMAT JSON). Without it the scan rule is skipped."""
        self.schema = schema
        self.config = config
        self.explain = explain
        self.audit = audit
        self.dialect = SQLGLOT_DIALECTS.get(schema.dialect, schema.dialect)
        self._mapping = {t.name.lower(): {c.name.lower(): "TEXT" for c in t.columns}
                         for t in schema.tables}
        self._pii = {(t.name.lower(), c.name.lower())
                     for t in schema.tables for c in t.columns if c.is_pii}
        self._row_counts = {t.name.lower(): t.row_count for t in schema.tables
                            if t.row_count is not None}
        self._blocked_functions = {f.lower() for f in config.blocked_functions}
        self._cache: Dict[str, GuardrailDecision] = {}

    def check(self, sql: str) -> GuardrailDecision:
        cached = self._cache.get(sql)
        if cached is not None:
            return cached
        decision = self._check(sql)
        if len(self._cache) > 256:
            self._cache.clear()
        self._cache[sql] = decision

        if not decision.allowed:
            logger.warning("blocked query: %s | %s", " ".join(sql.split()), decision.reasons())
            if self.audit:
                self.audit.write("blocked", decision)
        elif decision.rewrites and self.audit:
            self.audit.write("rewritten", decision)
        return decision

    # --- the rules ---------------------------------------------------------

    def _check(self, sql: str) -> GuardrailDecision:
        d = GuardrailDecision(original_sql=sql, sql=sql)
        hard = lambda rule, msg: d.violations.append(Violation(rule, msg, False))  # noqa: E731
        soft = lambda rule, msg: d.violations.append(Violation(rule, msg, True))  # noqa: E731

        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError as e:
            hard("parse", f"could not parse the SQL: {str(e).splitlines()[0]}")
            return d
        if len(statements) != 1:
            hard("single_statement", f"exactly one statement is allowed, found {len(statements)}")
            return d
        tree = statements[0]

        # read_only: allow-list the statement type, then look at every node,
        # because a write can hide inside a CTE of an otherwise-SELECT query.
        if not isinstance(tree, _QUERY_ROOTS):
            hard("read_only", f"only SELECT queries are allowed, not {_kind(tree)}")
        for node in tree.walk():
            if isinstance(node, _FORBIDDEN) and node is not tree:
                hard("read_only", f"{_kind(node)} is not allowed inside a query")
            if isinstance(node, exp.Select):
                if node.args.get("into"):
                    hard("read_only", "SELECT ... INTO creates a table")
                if node.args.get("locks"):
                    hard("read_only", "locking clauses (FOR UPDATE/SHARE) are not allowed")

        for func in tree.find_all(exp.Func):
            name = (func.name if isinstance(func, exp.Anonymous) else func.sql_name()).lower()
            if name in self._blocked_functions:
                hard("blocked_function", f"function {name}() is not allowed")

        if d.violations:
            d.violations = _dedupe(d.violations)
            return d  # unsafe whatever else is true; no point analysing further

        if self.config.max_subquery_depth is not None:
            depth = max((_select_depth(s) for s in tree.find_all(exp.Select)), default=0)
            if depth > self.config.max_subquery_depth:
                soft("subquery_depth", f"subqueries are nested {depth} levels deep "
                                       f"(max {self.config.max_subquery_depth}); flatten them, "
                                       f"e.g. with CTEs joined at the top level")

        qualified = self._qualify(tree)
        if qualified is None:
            soft("references", "could not resolve every table and column against the schema")
        else:
            d.tables, d.columns = _references(qualified)
            self._check_access(d, soft)
            if self.config.block_cartesian_joins:
                for msg in _cartesian_joins(qualified):
                    soft("cartesian_join", msg)

        if self.config.max_rows is not None:
            tree = self._apply_row_limit(tree, d)

        if not d.violations and self.explain and self.config.max_scan_rows is not None:
            self._check_scan(tree, d, soft)

        d.violations = _dedupe(d.violations)
        return d

    def _qualify(self, tree: exp.Expression) -> Optional[exp.Expression]:
        try:
            return qualify(tree.copy(), schema=self._mapping, dialect=self.dialect,
                           validate_qualify_columns=True)
        except (OptimizeError, ParseError, KeyError):
            return None

    def _check_access(self, d: GuardrailDecision, soft) -> None:
        blocked = {t.lower() for t in self.config.blocked_tables}
        allowed = ({t.lower() for t in self.config.allowed_tables}
                   if self.config.allowed_tables is not None else None)
        for table in d.tables:
            if table in blocked or (allowed is not None and table not in allowed):
                soft("table_access", f"table {table} may not be queried")
        if self.config.block_sensitive_columns:
            sensitive = [c for c in d.columns if tuple(c.split(".", 1)) in self._pii]
            if sensitive:
                soft("sensitive_columns",
                     f"reads personal data ({', '.join(sensitive)}); leave these columns "
                     f"out (use ids or aggregates instead)")

    def _apply_row_limit(self, tree: exp.Expression, d: GuardrailDecision) -> exp.Expression:
        max_rows = self.config.max_rows
        limit = tree.args.get("limit")
        if limit is None and _single_row(tree):
            return tree  # e.g. SELECT COUNT(*) ...: a LIMIT would only add noise
        if limit is None:
            tree = tree.limit(max_rows)
            d.rewrites.append(f"added LIMIT {max_rows}")
        else:
            value = limit.expression
            current = int(value.name) if isinstance(value, exp.Literal) and value.is_int else None
            if current is None or current > max_rows:
                limit.set("expression", exp.Literal.number(max_rows))
                was = current if current is not None else value.sql(dialect=self.dialect)
                d.rewrites.append(f"lowered LIMIT {was} to {max_rows}")
        if d.rewrites:
            d.sql = tree.sql(dialect=self.dialect)
            d.applied_limit = max_rows
        return tree

    def _derived_rows(self, tree: exp.Expression) -> Dict[str, int]:
        """Rough output size of each CTE / aliased subquery, for the scan estimate:
        1 for a single-row aggregate, else the largest table it reads."""
        largest = max(self._row_counts.values(), default=0)
        sizes = {}
        for node in list(tree.find_all(exp.CTE)) + list(tree.find_all(exp.Subquery)):
            name = node.alias_or_name
            if not name:
                continue
            body = node.this
            if _single_row(body):
                sizes[name] = 1
            else:
                reads = [self._row_counts.get(t.name.lower(), largest)
                         for t in body.find_all(exp.Table)]
                sizes[name] = max(reads, default=largest)
        return sizes

    def _check_scan(self, tree: exp.Expression, d: GuardrailDecision, soft) -> None:
        try:
            raw = self.explain(d.sql)
        except Exception as e:  # the query itself will report this error when it runs
            logger.info("EXPLAIN failed: %s", e)
            return
        if self.dialect == "sqlite":
            aliases = {}
            for t in tree.find_all(exp.Table):
                aliases[t.name.lower()] = t.name.lower()
                aliases[t.alias_or_name.lower()] = t.name.lower()
            plan = [(int(r[0]), int(r[1]), str(r[-1])) for r in raw]
            d.scan = estimate_sqlite(plan, aliases, self._row_counts, self._derived_rows(tree))
        elif self.dialect == "postgres":
            d.scan = estimate_postgres(raw, self._row_counts)
        else:
            return
        if d.scan.rows > self.config.max_scan_rows:
            soft("scan_estimate", f"the query would read about {d.scan.rows:,} rows "
                                  f"(max {self.config.max_scan_rows:,}); add filters or "
                                  f"join on indexed keys")


# --- helpers ---------------------------------------------------------------------

def _kind(node: exp.Expression) -> str:
    if isinstance(node, exp.Command):
        return (node.name or "command").upper()
    return type(node).__name__.upper()


def _single_row(tree: exp.Expression) -> bool:
    """An aggregate SELECT without GROUP BY returns exactly one row."""
    if not isinstance(tree, exp.Select) or tree.args.get("group"):
        return False
    return bool(tree.expressions) and all(
        e.find(exp.AggFunc) is not None for e in tree.expressions
    )


def _select_depth(select: exp.Select) -> int:
    depth, node = 0, select.parent
    while node is not None:
        if isinstance(node, exp.Select):
            depth += 1
        node = node.parent
    return depth


def _references(qualified: exp.Expression):
    tables: Set[str] = set()
    columns: Set[str] = set()
    for scope in traverse_scope(qualified):
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                tables.add(source.name.lower())
        for col in scope.columns:
            source = scope.sources.get(col.table)
            if isinstance(source, exp.Table):
                columns.add(f"{source.name.lower()}.{col.name.lower()}")
    return sorted(tables), sorted(columns)


def _cartesian_joins(qualified: exp.Expression) -> List[str]:
    problems = []
    for select in qualified.find_all(exp.Select):
        joins = select.args.get("joins") or []
        if not joins:
            continue
        where = select.args.get("where")
        for join in joins:
            if join.args.get("on") or join.args.get("using"):
                continue
            if isinstance(join.this, (exp.Unnest, exp.Lateral)):
                continue
            alias = join.this.alias_or_name
            if where is not None and _links(where, alias):
                continue  # old-style join: FROM a, b WHERE a.x = b.y
            problems.append(f"join with {alias} has no join condition (cartesian product); "
                            f"add an ON clause using the listed relationships")
    return problems


_COMPARISONS = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)


def _links(where: exp.Expression, alias: str) -> bool:
    """Does WHERE compare a column of `alias` with a column of another table?

    Range joins (a.s > b.avg_s) count as well as equality; a huge range join
    is still caught by the scan estimate.
    """
    for cmp in where.find_all(*_COMPARISONS):
        left, right = cmp.this, cmp.expression
        if isinstance(left, exp.Column) and isinstance(right, exp.Column):
            if alias in (left.table, right.table) and left.table != right.table:
                return True
    for between in where.find_all(exp.Between):
        tables = {c.table for c in between.find_all(exp.Column)}
        if alias in tables and len(tables) > 1:
            return True
    return False


def _dedupe(violations: List[Violation]) -> List[Violation]:
    seen, out = set(), []
    for v in violations:
        if (v.rule, v.message) not in seen:
            seen.add((v.rule, v.message))
            out.append(v)
    return out


class GuardedExecutor:
    """Runs a query only if the guardrails allow it, and runs the checked version.

    Wraps ReadOnlyExecutor, so even a query that slips past the guardrails
    still runs on a read-only connection (Step 3 adds a SELECT-only user).
    """

    def __init__(self, executor, guardrails: Guardrails):
        self.executor = executor
        self.guardrails = guardrails

    def run(self, sql: str):
        decision = self.guardrails.check(sql)
        if not decision.allowed:
            raise GuardrailError(decision)
        result = self.executor.run(decision.sql)
        result.guardrails = decision
        # With LIMIT N added by us, N rows back means there may well be more.
        if decision.applied_limit is not None and len(result.rows) >= decision.applied_limit:
            result.truncated = True
        return result
