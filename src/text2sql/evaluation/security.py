"""Guardrail effectiveness: are dangerous queries blocked, and do safe ones run?

Run against a *copy* of the database. The copy is fingerprinted (tables,
views, row counts) before and after, so "zero unsafe queries executed" is
measured, not assumed. Three numbers:

  refused          attacks stopped by the full stack (guardrails + sandbox)
  db_layer_alone   attacks the database layer refuses with the guardrails
                   switched off: the second line of defence on its own
  false_blocks     legitimate queries (incl. every golden answer) wrongly blocked
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List

import sqlalchemy as sa

from ..guardrails import GuardedExecutor, GuardrailConfig, Guardrails
from ..sandbox import SandboxedExecutor
from .attacks import LEGITIMATE, PG_ATTACKS, SQL_ATTACKS


@dataclass
class SecurityReport:
    attacks: int = 0
    refused: int = 0
    executed: List[str] = field(default_factory=list)  # must stay empty
    db_layer_checked: int = 0
    db_layer_refused: int = 0
    pg_attacks: int = 0
    pg_refused: int = 0  # Postgres-only attacks, checked by the guardrails
    legitimate: int = 0
    false_blocks: List[str] = field(default_factory=list)
    database_unchanged: bool = True
    db_layer_allowed: List[str] = field(default_factory=list)  # attacks the DB layer alone lets run
    fingerprint: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def fingerprint(engine) -> Dict[str, int]:
    """Row count of every table and view, plus the set of objects."""
    inspector = sa.inspect(engine)
    names = sorted(inspector.get_table_names()) + sorted(inspector.get_view_names())
    prints = {"objects": len(names)}
    with engine.connect() as conn:
        for name in names:
            prints[name] = conn.execute(
                sa.text(f'SELECT COUNT(*) FROM "{name}"')).scalar()
    return prints


def run_security(engine, schema, golden_sql: List[str] = (),
                 config: GuardrailConfig = GuardrailConfig()) -> SecurityReport:
    """`engine` must point at a disposable copy of the database."""
    import logging

    # Every attack is expected to be refused; don't log each one as an incident.
    for name in ("text2sql.guardrails", "text2sql.sandbox"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    report = SecurityReport()
    before = fingerprint(engine)
    sandbox = SandboxedExecutor.for_schema(engine, schema, config)
    guarded = GuardedExecutor(sandbox, Guardrails(schema, config, explain=sandbox.explain))

    for sql in SQL_ATTACKS:
        report.attacks += 1
        try:
            guarded.run(sql)
            report.executed.append(sql)
        except Exception:
            report.refused += 1

    # Second line of defence on its own: straight to the sandbox's connection,
    # no guardrails and no text check.
    for sql in SQL_ATTACKS:
        if not sql.strip():
            continue
        report.db_layer_checked += 1
        with sandbox._engine.connect() as conn:
            try:
                conn.exec_driver_sql(sql)
                report.db_layer_allowed.append(sql)
            except Exception:
                report.db_layer_refused += 1
            finally:
                conn.rollback()

    pg_schema = _as_dialect(schema, "postgresql")
    pg_guard = Guardrails(pg_schema, config)
    for sql in PG_ATTACKS:
        report.pg_attacks += 1
        if not pg_guard.check(sql).allowed:
            report.pg_refused += 1

    for sql in list(LEGITIMATE) + list(golden_sql):
        report.legitimate += 1
        try:
            guarded.run(sql)
        except Exception as e:
            report.false_blocks.append(f"{' '.join(sql.split())[:80]}  ({e})")

    after = fingerprint(engine)
    report.fingerprint = after
    report.database_unchanged = before == after
    return report


def _as_dialect(schema, dialect: str):
    import copy

    clone = copy.deepcopy(schema)
    clone.dialect = dialect
    return clone
