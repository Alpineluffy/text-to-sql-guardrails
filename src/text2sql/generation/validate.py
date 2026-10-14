"""Phase 2 Step 1: check generated SQL before it goes anywhere near the database.

Three checks on the parsed SQL (sqlglot, in the database's dialect):

1. Syntax: it parses, and is exactly one statement.
2. Grounding: every table and column exists in the extracted schema. Invented
   names are the most common hallucination, and the cheapest to catch.
3. Claims: the tables/columns the model *says* it used match what the SQL
   actually uses. A mismatch means its explanation can't be trusted either.

Failures of 1 and 2 go back to the model to fix (ValidatingGenerator); 3 is
reported as a warning. Safety policy (SELECT-only, limits, cost) is the
guardrail layer's job, not this module's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError, ParseError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import traverse_scope

from ..guardrails import Guardrails
from ..prompt.builder import Prompt
from ..schema.models import DatabaseSchema
from .llm import SQLGenerator, SQLResponse

SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}
_SQL_TYPES = {
    "integer": "BIGINT", "numeric": "DECIMAL", "boolean": "BOOLEAN", "date": "DATE",
    "datetime": "TIMESTAMP", "time": "TIME", "text": "TEXT",
}


@dataclass
class ValidationReport:
    sql: str
    syntax_error: Optional[str] = None
    unknown_tables: List[str] = field(default_factory=list)
    unknown_columns: List[str] = field(default_factory=list)  # as written, e.g. "o.customer_name"
    ambiguous_columns: List[str] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)  # real tables the SQL reads
    columns: List[str] = field(default_factory=list)  # as table.column
    warnings: List[str] = field(default_factory=list)  # e.g. claim mismatches

    @property
    def errors(self) -> List[str]:
        """Problems the model must fix before the SQL can be used."""
        out = []
        if self.syntax_error:
            out.append(f"Syntax error: {self.syntax_error}")
        out += [f"Unknown table: {t}" for t in self.unknown_tables]
        out += [f"Unknown column: {c}" for c in self.unknown_columns]
        out += [f"Ambiguous column (qualify it with a table): {c}"
                for c in self.ambiguous_columns]
        return out

    @property
    def ok(self) -> bool:
        return not self.errors


class SQLValidator:
    def __init__(self, schema: DatabaseSchema):
        self.dialect = SQLGLOT_DIALECTS.get(schema.dialect, schema.dialect)
        self.columns: Dict[str, Set[str]] = {
            t.name.lower(): {c.name.lower() for c in t.columns} for t in schema.tables
        }
        self._mapping = {
            t.name.lower(): {c.name.lower(): _SQL_TYPES.get(c.type_family, "TEXT")
                             for c in t.columns}
            for t in schema.tables
        }

    def validate(
        self,
        sql: str,
        claimed_tables: Optional[List[str]] = None,
        claimed_columns: Optional[List[str]] = None,
    ) -> ValidationReport:
        report = ValidationReport(sql=sql)
        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except ParseError as e:
            report.syntax_error = _first_line(str(e))
            return report
        if len(statements) != 1:
            report.syntax_error = f"expected one statement, found {len(statements)}"
            return report
        tree = statements[0]

        # Pass 1, before qualifying: unknown tables and explicitly qualified
        # unknown columns (qualify() would stop at the first one).
        for scope in _scopes(tree):
            for alias, source in scope.sources.items():
                if isinstance(source, exp.Table) and source.name.lower() not in self.columns:
                    report.unknown_tables.append(source.name)
            for col in scope.columns:
                source = scope.sources.get(col.table)
                if isinstance(source, exp.Table) and source.name.lower() in self.columns:
                    if col.name.lower() not in self.columns[source.name.lower()]:
                        report.unknown_columns.append(f"{col.table}.{col.name}")
        if report.unknown_tables or report.unknown_columns:
            report.unknown_tables = _dedupe(report.unknown_tables)
            report.unknown_columns = _dedupe(report.unknown_columns)
            return report

        # Pass 2: resolve unqualified columns against the schema.
        try:
            qualified = qualify(
                tree, schema=self._mapping, dialect=self.dialect,
                validate_qualify_columns=False,
            )
        except (OptimizeError, ParseError) as e:
            report.syntax_error = _first_line(str(e))
            return report

        tables, columns = set(), set()
        for scope in _scopes(qualified):
            for source in scope.sources.values():
                if isinstance(source, exp.Table):
                    tables.add(source.name.lower())
            for col in scope.columns:
                source = scope.sources.get(col.table)
                if isinstance(source, exp.Table):
                    columns.add(f"{source.name.lower()}.{col.name.lower()}")
                elif not col.table:
                    owners = [t for t in tables_in(scope) if col.name.lower() in self.columns[t]]
                    target = report.ambiguous_columns if len(owners) > 1 else report.unknown_columns
                    target.append(col.name)
        report.unknown_columns = _dedupe(report.unknown_columns)
        report.ambiguous_columns = _dedupe(report.ambiguous_columns)
        report.tables, report.columns = sorted(tables), sorted(columns)

        if report.ok:
            report.warnings = self._claim_warnings(report, claimed_tables, claimed_columns)
        return report

    @staticmethod
    def _claim_warnings(
        report: ValidationReport,
        claimed_tables: Optional[List[str]],
        claimed_columns: Optional[List[str]],
    ) -> List[str]:
        warnings = []
        if claimed_tables is not None:
            claimed = {t.lower() for t in claimed_tables}
            actual = set(report.tables)
            if claimed - actual:
                warnings.append(f"Model says it used tables the SQL doesn't read: "
                                f"{', '.join(sorted(claimed - actual))}")
            if actual - claimed:
                warnings.append(f"SQL reads tables the model didn't mention: "
                                f"{', '.join(sorted(actual - claimed))}")
        if claimed_columns is not None:
            claimed = {c.lower() for c in claimed_columns if "." in c}
            actual = set(report.columns)
            missing = actual - claimed
            if missing:
                warnings.append(f"SQL uses columns the model didn't mention: "
                                f"{', '.join(sorted(missing))}")
            extra = claimed - actual
            if extra:
                warnings.append(f"Model says it used columns the SQL doesn't: "
                                f"{', '.join(sorted(extra))}")
        return warnings


class ValidatingGenerator:
    """Wraps SQLGenerator: validates every query and asks the model to fix errors.

    Drop-in for SQLGenerator (same generate(prompt) method), so the
    clarification session gets validated SQL without knowing about it. With
    `guardrails`, fixable guardrail violations (e.g. a cartesian join) are sent
    back too; hard ones (e.g. DELETE) end the attempt, since there's nothing
    legitimate to repair.
    """

    def __init__(
        self,
        generator: SQLGenerator,
        validator: SQLValidator,
        max_repairs: int = 2,
        guardrails: Optional["Guardrails"] = None,
    ):
        self.generator = generator
        self.validator = validator
        self.max_repairs = max_repairs
        self.guardrails = guardrails

    def generate(self, prompt: Prompt) -> SQLResponse:
        attempt_prompt = prompt
        for attempt in range(1, self.max_repairs + 2):
            response = self.generator.generate(attempt_prompt)
            problems, repairable = self._check(response)
            response.attempts = attempt
            if not problems or not repairable or attempt > self.max_repairs:
                return response
            attempt_prompt = prompt.with_feedback(_feedback(response, problems))
        return response  # unreachable; keeps type checkers happy

    def _check(self, response: SQLResponse) -> Tuple[List[Tuple[str, List[str]]], bool]:
        """Check the main SQL and every interpretation.

        Returns ((label, errors) pairs, whether all of them are worth a repair).
        """
        problems: List[Tuple[str, List[str]]] = []
        repairable = True
        targets = []
        if response.sql:
            response.validation = self.validator.validate(
                response.sql, response.tables_used, response.columns_used
            )
            targets.append(("SQL", response.sql, response))
        for interp in response.interpretations:
            interp.validation = self.validator.validate(interp.sql)
            targets.append((f"Interpretation '{interp.label}'", interp.sql, interp))

        for label, sql, holder in targets:
            decision = self.guardrails.check(sql) if self.guardrails is not None else None
            holder.guardrails = decision
            if decision is not None and any(not v.retryable for v in decision.violations):
                # Unsafe whatever else is wrong: report it as such, never "repair" it.
                problems.append((label, decision.reasons()))
                repairable = False
            elif not holder.validation.ok:
                problems.append((label, holder.validation.errors))
            elif decision is not None and not decision.allowed:
                problems.append((label, decision.reasons()))
                repairable = repairable and decision.retryable
        return problems, repairable


def tables_in(scope) -> List[str]:
    return [s.name.lower() for s in scope.sources.values() if isinstance(s, exp.Table)]


def _scopes(tree: exp.Expression):
    try:
        return traverse_scope(tree)
    except OptimizeError:
        return []


def _feedback(response: SQLResponse, problems: List[Tuple[str, List[str]]]) -> str:
    lines = []
    if response.sql:
        lines += ["Your previous SQL:", response.sql, ""]
    for label, errors in problems:
        lines.append(f"{label} was rejected:")
        lines += [f"- {e}" for e in errors]
    lines.append("")
    lines.append("Fix these problems. Use only tables and columns listed in the schema.")
    return "\n".join(lines)


def _dedupe(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _first_line(message: str) -> str:
    return message.strip().splitlines()[0] if message.strip() else message
