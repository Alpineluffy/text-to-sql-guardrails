"""Infer foreign keys the database never declared.

Many real databases skip FK constraints, which leaves the model guessing join
keys. A column like `product_id` is linked to `products.id` when:

1. its name (minus the `_id` suffix) matches a table name, allowing for plurals
   and prefixes (`parent_customer_id` -> `customers`),
2. the target has a single-column primary key of a compatible type, and
3. the data agrees: at least `fk_min_match_ratio` of sampled non-null values
   exist in the target key.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .config import ExtractionConfig
from .models import ColumnInfo, DatabaseSchema, ForeignKey, TableInfo

ID_SUFFIX = re.compile(r"^(?P<base>.+?)_?id$", re.IGNORECASE)

_NUMERIC_FAMILIES = {"integer", "numeric"}


def looks_like_identifier(column_name: str) -> bool:
    return bool(ID_SUFFIX.match(column_name))


def candidate_table_names(base: str) -> List[str]:
    """Table names a reference like `base_id` could point to, best match first."""
    tokens = [t for t in base.lower().split("_") if t]
    names: List[str] = []
    # Longest suffix first: parent_customer, then customer.
    for i in range(len(tokens)):
        stem = "_".join(tokens[i:])
        names += [stem, stem + "s", stem + "es"]
        if stem.endswith("y"):
            names.append(stem[:-1] + "ies")
    return names


def _compatible(a: ColumnInfo, b: ColumnInfo) -> bool:
    if a.type_family == b.type_family:
        return True
    return a.type_family in _NUMERIC_FAMILIES and b.type_family in _NUMERIC_FAMILIES


def _target_key(table: TableInfo) -> Optional[ColumnInfo]:
    if len(table.primary_key) == 1:
        return table.column(table.primary_key[0])
    for col in table.columns:
        if col.name.lower() == "id":
            return col
    return None


def _match_ratio(
    engine: Engine, source: sa.Table, column: str, target: sa.Table, key: str, rows: int
) -> Optional[float]:
    """Share of sampled non-null source values that exist in the target key."""
    sample = (
        sa.select(source.c[column].label("v"))
        .where(source.c[column].is_not(None))
        .limit(rows)
        .subquery()
    )
    total_q = sa.select(sa.func.count()).select_from(sample)
    orphan_q = (
        sa.select(sa.func.count())
        .select_from(sample)
        # Select only the key: `EXISTS (SELECT * ...)` would need read access to
        # every column, which a restricted role may not have (e.g. PII).
        .where(~sa.select(target.c[key]).where(target.c[key] == sample.c.v).exists())
    )
    with engine.connect() as conn:
        total = conn.execute(total_q).scalar_one()
        if total == 0:
            return None
        orphans = conn.execute(orphan_q).scalar_one()
    return round((total - orphans) / total, 4)


def infer_foreign_keys(
    engine: Engine,
    schema: DatabaseSchema,
    sa_tables: Dict[str, sa.Table],
    config: ExtractionConfig,
) -> List[ForeignKey]:
    tables_by_lower = {t.name.lower(): t for t in schema.tables if t.kind == "table"}
    declared = {(fk.source_table, c) for fk in schema.foreign_keys for c in fk.source_columns}

    inferred: List[ForeignKey] = []
    for table in schema.tables:
        if table.kind != "table":
            continue
        for col in table.columns:
            if (table.name, col.name) in declared:
                continue
            if col.is_primary_key and len(table.primary_key) == 1:
                continue
            m = ID_SUFFIX.match(col.name)
            if not m or len(m.group("base").strip("_")) < 2:
                continue

            target = next(
                (tables_by_lower[n] for n in candidate_table_names(m.group("base"))
                 if n in tables_by_lower),
                None,
            )
            if target is None:
                continue
            key = _target_key(target)
            if key is None or not _compatible(col, key):
                continue

            ratio = _match_ratio(
                engine, sa_tables[table.name], col.name,
                sa_tables[target.name], key.name, config.fk_check_rows,
            )
            # No data to check against: keep the name-based guess, unverified.
            if ratio is not None and ratio < config.fk_min_match_ratio:
                continue
            inferred.append(
                ForeignKey(
                    source_table=table.name,
                    source_columns=[col.name],
                    target_table=target.name,
                    target_columns=[key.name],
                    origin="inferred",
                    match_ratio=ratio,
                )
            )
    return inferred
