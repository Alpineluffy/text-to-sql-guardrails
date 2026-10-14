"""Render the extracted schema as annotated CREATE TABLE text.

DDL is compact and close to what models saw in training. Descriptions, real
values and ranges go in trailing comments, e.g.

    status VARCHAR(20) NOT NULL,  -- Fulfilment status. Values: 'DELIVERED', ...
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Set

from ..schema.models import ColumnInfo, DatabaseSchema, ForeignKey, TableInfo
from .knowledge import Knowledge, TableNotes

MAX_VALUES_PER_COLUMN = 20


def sql_literal(value: Any, dialect: str) -> str:
    if isinstance(value, bool):
        if dialect == "sqlite":
            return "1" if value else "0"
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def relationships(schema: DatabaseSchema, tables: Set[str]) -> List[ForeignKey]:
    return [
        fk for fk in schema.foreign_keys
        if fk.source_table in tables and fk.target_table in tables
    ]


def render_schema(
    schema: DatabaseSchema,
    knowledge: Knowledge,
    tables: Iterable[str],
    max_values: int = MAX_VALUES_PER_COLUMN,
) -> str:
    wanted = set(tables)
    fks = relationships(schema, wanted)
    blocks = [
        _render_table(t, knowledge.tables.get(t.name, TableNotes()), fks, schema.dialect, max_values)
        for t in schema.tables
        if t.name in wanted
    ]
    return "\n\n".join(blocks)


def render_relationships(schema: DatabaseSchema, tables: Iterable[str]) -> str:
    lines = []
    for fk in relationships(schema, set(tables)):
        join = " AND ".join(
            f"{fk.source_table}.{s} = {fk.target_table}.{t}"
            for s, t in zip(fk.source_columns, fk.target_columns)
        )
        note = "" if fk.origin == "declared" else "  (inferred from data; not a declared constraint)"
        lines.append(f"{join}{note}")
    return "\n".join(lines)


def _render_table(
    table: TableInfo, notes: TableNotes, fks: List[ForeignKey], dialect: str, max_values: int
) -> str:
    header = []
    description = notes.description or table.comment
    if description:
        size = f" ({table.row_count:,} rows)" if table.row_count is not None else ""
        header.append(f"-- {description}{size}")
    if notes.synonyms:
        header.append(f"-- Also called: {', '.join(notes.synonyms)}")

    references = {
        fk.source_columns[0]: fk
        for fk in fks
        if fk.source_table == table.name and len(fk.source_columns) == 1
    }
    single_pk = table.primary_key if len(table.primary_key) == 1 else []

    defs = []
    for col in table.columns:
        parts = [col.name, col.type]
        if col.name in single_pk:
            parts.append("PRIMARY KEY")
        elif not col.nullable:
            parts.append("NOT NULL")
        fk = references.get(col.name)
        if fk:
            parts.append(f"REFERENCES {fk.target_table}({fk.target_columns[0]})")
        defs.append((" ".join(parts), _column_comment(col, notes, dialect, max_values)))
    if len(table.primary_key) > 1:
        defs.append((f"PRIMARY KEY ({', '.join(table.primary_key)})", None))

    keyword = "VIEW" if table.kind == "view" else "TABLE"
    lines = [*header, f"CREATE {keyword} {table.name} ("]
    for i, (definition, comment) in enumerate(defs):
        line = f"  {definition}{',' if i < len(defs) - 1 else ''}"
        lines.append(f"{line}  -- {comment}" if comment else line)
    lines.append(");")
    return "\n".join(lines)


def _column_comment(
    col: ColumnInfo, notes: TableNotes, dialect: str, max_values: int
) -> Optional[str]:
    parts = []
    description = notes.columns.get(col.name) or col.comment
    if description:
        parts.append(description.rstrip("."))
    if col.is_pii:
        parts.append("SENSITIVE personal data")
    profile = col.profile
    if profile and not col.is_pii:
        if profile.is_categorical and profile.sample_values:
            shown = [sql_literal(v, dialect) for v in profile.sample_values[:max_values]]
            extra = len(profile.sample_values) - len(shown)
            parts.append("Values: " + ", ".join(shown) + (f" (+{extra} more)" if extra > 0 else ""))
        elif profile.min_value is not None:
            parts.append(f"Range: {_short(profile.min_value)} to {_short(profile.max_value)}")
        if profile.null_fraction is not None and profile.null_fraction >= 0.05:
            parts.append(f"{profile.null_fraction:.0%} null")
    return ". ".join(parts) if parts else None


def _short(value: Any) -> str:
    # Trim timestamps to the date; that's all the model needs for range hints.
    text = str(value)
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}T", text) else text


@dataclass(frozen=True)
class ValueMention:
    table: str
    column: str
    value: Any
    phrase: str  # as written in the question

    @property
    def case_differs(self) -> bool:
        return self.phrase != str(self.value)


def find_value_mentions(
    question: str, schema: DatabaseSchema, tables: Iterable[str]
) -> List[ValueMention]:
    """Real column values that appear in the question, e.g. 'india' -> country 'India'."""
    wanted = set(tables)
    found: List[ValueMention] = []
    for table in schema.tables:
        if table.name not in wanted:
            continue
        for col in table.columns:
            p = col.profile
            if col.is_pii or not p:
                continue
            for value in [*p.sample_values, *p.value_index]:
                if not isinstance(value, str) or len(value) < 3:
                    continue
                m = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", question, re.IGNORECASE)
                if m:
                    found.append(ValueMention(table.name, col.name, value, m.group(0)))
    # Drop values contained in a longer match ("Laptop" inside "Laptop Pro 14").
    return [
        v for v in found
        if not any(
            o is not v and len(str(o.value)) > len(str(v.value))
            and str(v.value).lower() in str(o.value).lower()
            for o in found
        )
    ]
