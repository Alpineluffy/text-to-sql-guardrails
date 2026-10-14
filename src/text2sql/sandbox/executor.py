"""Phase 2 Step 3: run queries where they can't do damage, even if the
guardrails miss something.

Layers, outermost first:

1. Text check: one statement, starting with SELECT or WITH.
2. Read-only transaction, always rolled back, with a statement timeout.
3. The database refuses anything else:
   - PostgreSQL: connect as a SELECT-only role (see sandbox/postgres.py), with
     column-level grants that leave out personal data.
   - SQLite has no users, so the connection is opened read-only with
     `query_only` on, and an authorizer callback (SQLite's own permission hook,
     consulted for every table, column and function a statement touches)
     allows only reads, and denies personal-data columns, blocked tables and
     blocked functions.

When layer 3 refuses a query, the guardrails let through something they
shouldn't have, so it's logged as a security event, not an ordinary error.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

_LEADING_COMMENTS = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/|\s)*", re.DOTALL)
_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")

# SQLite authorizer: the only actions a read-only query needs.
_SQLITE_ALLOWED = {
    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE, sqlite3.SQLITE_TRANSACTION,
}

# SQLAlchemy reads this one to learn the isolation level; it changes nothing.
_SQLITE_ALLOWED_PRAGMAS = {"read_uncommitted"}

# Error kinds that mean an earlier layer let something through.
SECURITY_KINDS = {"denied", "read_only"}


class UnsafeQueryError(ValueError):
    pass


class SandboxError(RuntimeError):
    """A query the sandbox stopped or the database rejected.

    kind: timeout | denied | read_only | query_error
    """

    def __init__(self, kind: str, message: str, sql: str):
        self.kind = kind
        self.sql = sql
        super().__init__(message)

    @property
    def is_security_event(self) -> bool:
        return self.kind in SECURITY_KINDS


@dataclass(frozen=True)
class SandboxConfig:
    statement_timeout_ms: int = 5_000
    lock_timeout_ms: int = 1_000
    max_rows: int = 1_000  # rows fetched; anything beyond marks the result truncated


@dataclass
class QueryResult:
    columns: List[str]
    rows: List[Tuple[Any, ...]]
    truncated: bool
    elapsed_ms: Optional[float] = None
    guardrails: Any = None  # GuardrailDecision when run through GuardedExecutor

    def fingerprint(self) -> Tuple:
        """Order-insensitive summary, for checking whether two queries agree."""
        def norm(v):
            return round(v, 6) if isinstance(v, float) else v
        return tuple(sorted(repr(tuple(norm(v) for v in row)) for row in self.rows)), self.truncated


def check_read_only(sql: str) -> str:
    body = _LEADING_COMMENTS.sub("", sql).strip().rstrip(";").strip()
    if not re.match(r"(?i)^(select|with)\b", body):
        raise UnsafeQueryError("only SELECT queries are allowed")
    if ";" in _STRING_LITERAL.sub("''", body):
        raise UnsafeQueryError("only one statement is allowed")
    return body


class SandboxedExecutor:
    def __init__(
        self,
        engine: Engine,
        config: SandboxConfig = SandboxConfig(),
        denied_columns: Iterable[Tuple[str, str]] = (),
        denied_tables: Iterable[str] = (),
        denied_functions: Iterable[str] = (),
        audit: Any = None,
        max_rows: Optional[int] = None,
    ):
        """`denied_*` are enforced by SQLite's authorizer; on PostgreSQL the
        role's grants do the same job (sandbox/postgres.py)."""
        if max_rows is not None:
            config = SandboxConfig(config.statement_timeout_ms, config.lock_timeout_ms, max_rows)
        self.engine = engine
        self.config = config
        self.max_rows = config.max_rows
        self.dialect = engine.dialect.name
        self.audit = audit
        self._denied_columns: Set[Tuple[str, str]] = {(t.lower(), c.lower())
                                                      for t, c in denied_columns}
        self._denied_tables = {t.lower() for t in denied_tables}
        self._denied_functions = {f.lower() for f in denied_functions}
        # Per thread: a web server runs several queries at once on one executor.
        self._local = threading.local()
        if self.dialect == "sqlite":
            uri = Path(engine.url.database).resolve().as_uri() + "?mode=ro"
            self._engine = sa.create_engine(
                "sqlite://", creator=lambda: self._sqlite_connect(uri)
            )
        else:
            self._engine = engine

    @property
    def _deadline(self) -> Optional[float]:
        return getattr(self._local, "deadline", None)

    @_deadline.setter
    def _deadline(self, value: Optional[float]) -> None:
        self._local.deadline = value

    @classmethod
    def for_schema(cls, engine: Engine, schema, guardrail_config=None, **kwargs):
        """Deny at the database level what the guardrails deny at the SQL level."""
        from ..guardrails import GuardrailConfig

        g = guardrail_config or GuardrailConfig()
        pii = [(t.name, c.name) for t in schema.tables for c in t.columns if c.is_pii]
        return cls(
            engine,
            denied_columns=pii if g.block_sensitive_columns else (),
            denied_tables=g.blocked_tables,
            denied_functions=g.blocked_functions,
            **kwargs,
        )

    # --- public API --------------------------------------------------------

    def run(self, sql: str) -> QueryResult:
        body = check_read_only(sql)
        start = time.perf_counter()
        with self._engine.connect() as conn:
            try:
                self._begin(conn)
                self._deadline = time.monotonic() + self.config.statement_timeout_ms / 1000
                result = conn.exec_driver_sql(body)
                rows = result.fetchmany(self.max_rows + 1)
                columns = list(result.keys())
            except sa.exc.DBAPIError as e:
                raise self._classify(e, sql) from e
            finally:
                self._deadline = None
                conn.rollback()
        return QueryResult(
            columns=columns,
            rows=[tuple(r) for r in rows[: self.max_rows]],
            truncated=len(rows) > self.max_rows,
            elapsed_ms=round((time.perf_counter() - start) * 1000, 2),
        )

    def explain(self, sql: str) -> Any:
        """The database's plan for `sql` (never runs the query itself)."""
        body = check_read_only(sql)
        with self._engine.connect() as conn:
            try:
                self._begin(conn)
                self._deadline = time.monotonic() + self.config.statement_timeout_ms / 1000
                if self.dialect == "sqlite":
                    return [tuple(r) for r in conn.exec_driver_sql("EXPLAIN QUERY PLAN " + body)]
                if self.dialect == "postgresql":
                    plan = conn.exec_driver_sql("EXPLAIN (FORMAT JSON) " + body).scalar()
                    return json.loads(plan) if isinstance(plan, str) else plan
                raise NotImplementedError(f"no EXPLAIN support for {self.dialect}")
            except sa.exc.DBAPIError as e:
                raise self._classify(e, sql) from e
            finally:
                self._deadline = None
                conn.rollback()

    # --- per-dialect setup -------------------------------------------------

    def _begin(self, conn) -> None:
        """Make the transaction read-only and time-limited (non-SQLite)."""
        if self.dialect == "postgresql":
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            conn.exec_driver_sql(f"SET LOCAL statement_timeout = {int(self.config.statement_timeout_ms)}")
            conn.exec_driver_sql(f"SET LOCAL lock_timeout = {int(self.config.lock_timeout_ms)}")
        elif self.dialect == "mysql":
            conn.exec_driver_sql(
                f"SET SESSION MAX_EXECUTION_TIME = {int(self.config.statement_timeout_ms)}"
            )
        # SQLite: read-only connection + authorizer + progress-handler timeout,
        # all installed once per connection in _sqlite_connect.

    def _sqlite_connect(self, uri: str) -> sqlite3.Connection:
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.execute("PRAGMA query_only = ON")  # before the authorizer forbids PRAGMAs
        conn.set_authorizer(self._sqlite_authorizer)
        # Called every 1,000 VM steps; returning non-zero aborts the statement.
        conn.set_progress_handler(
            lambda: 1 if self._deadline is not None and time.monotonic() > self._deadline else 0,
            1_000,
        )
        return conn

    def _sqlite_authorizer(self, action, arg1, arg2, db_name, source):
        if action == sqlite3.SQLITE_PRAGMA and (arg1 or "").lower() in _SQLITE_ALLOWED_PRAGMAS:
            return sqlite3.SQLITE_OK
        if action not in _SQLITE_ALLOWED:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_READ:
            table, column = (arg1 or "").lower(), (arg2 or "").lower()
            if table in self._denied_tables or (table, column) in self._denied_columns:
                return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in self._denied_functions:
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    # --- errors ------------------------------------------------------------

    def _classify(self, error: sa.exc.DBAPIError, sql: str) -> SandboxError:
        message = str(getattr(error, "orig", error)).strip().splitlines()[0]
        lowered = message.lower()
        if "interrupted" in lowered or "statement timeout" in lowered or "max_execution_time" in lowered:
            kind = "timeout"
            message = f"query took longer than {self.config.statement_timeout_ms / 1000:g}s and was stopped"
        elif "not authorized" in lowered or "prohibited" in lowered or "permission denied" in lowered:
            kind = "denied"
        elif "readonly" in lowered or "read-only" in lowered or "read only" in lowered:
            kind = "read_only"
        else:
            kind = "query_error"
        err = SandboxError(kind, message, sql)
        if err.is_security_event:
            logger.error("database refused a query the guardrails allowed (%s): %s | %s",
                         kind, message, " ".join(sql.split()))
            if self.audit is not None:
                self.audit.write_event("sandbox_refused", sql=sql, kind=kind, message=message)
        return err
