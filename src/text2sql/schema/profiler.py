"""Profile column data: real values, ranges and null rates.

Per table this runs one aggregate query (counts, distinct counts, min/max)
plus one GROUP BY per text/boolean column. With `sample_rows` set, queries
read only the first N rows; that keeps big tables cheap but can miss rare
values, so the profile records whether it was sampled.
"""
from __future__ import annotations

import warnings
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Iterable, List, Optional, Set, Tuple

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .config import ExtractionConfig
from .fk_inference import looks_like_identifier
from .models import ColumnInfo, ColumnProfile, TableInfo

_ORDERABLE = {"integer", "numeric", "date", "datetime", "time"}
_COUNTABLE = _ORDERABLE | {"text", "boolean"}
_VALUE_FAMILIES = {"text", "boolean"}


def to_jsonable(value: Any) -> Any:
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def case_style(values: Iterable[Any]) -> Optional[str]:
    words = [v for v in values if isinstance(v, str) and any(ch.isalpha() for ch in v)]
    if not words:
        return None
    if all(v.isupper() for v in words):
        return "UPPER"
    if all(v.islower() for v in words):
        return "lower"
    if all(v.istitle() for v in words):
        return "Title"
    return "mixed"


class TableProfiler:
    def __init__(self, engine: Engine, config: ExtractionConfig):
        self.engine = engine
        self.config = config

    def profile(
        self, sa_table: sa.Table, table: TableInfo, key_columns: Set[Tuple[str, str]]
    ) -> None:
        cfg = self.config
        # SQLite stores NUMERIC as float; SQLAlchemy warns on every Decimal read.
        with warnings.catch_warnings(), self.engine.connect() as conn:
            warnings.filterwarnings("ignore", message=r".*support Decimal objects natively")

            table.row_count = conn.execute(
                sa.select(sa.func.count()).select_from(sa_table)
            ).scalar_one()
            sampled = cfg.sample_rows is not None and table.row_count > cfg.sample_rows
            # Personal-data columns are never read, not even counted, so a
            # restricted database role without access to them can profile too.
            readable = [sa_table.c[c.name] for c in table.columns if not c.is_pii]
            source = (sa.select(*readable).limit(cfg.sample_rows).subquery()
                      if sampled and readable else sa_table)
            rows_profiled = min(table.row_count, cfg.sample_rows) if sampled else table.row_count

            stats = self._aggregate_stats(conn, source, table)
            for col in table.columns:
                is_key = (table.name, col.name) in key_columns or looks_like_identifier(col.name)
                col.profile = self._profile_column(
                    conn, source, col, stats, rows_profiled, sampled, is_key
                )

    def _aggregate_stats(self, conn, source, table: TableInfo) -> dict:
        """One query for every column's non-null count, distinct count and range."""
        exprs: List[sa.ColumnElement] = []
        labels: List[Tuple[str, str]] = []
        for col in table.columns:
            if col.is_pii:
                continue
            c = source.c[col.name]
            exprs.append(sa.func.count(c))
            labels.append((col.name, "non_null"))
            if col.type_family in _COUNTABLE:
                exprs.append(sa.func.count(sa.distinct(c)))
                labels.append((col.name, "distinct"))
            if col.type_family in _ORDERABLE and not col.is_pii:
                exprs += [sa.func.min(c), sa.func.max(c)]
                labels += [(col.name, "min"), (col.name, "max")]
        if not exprs:
            return {}
        row = conn.execute(sa.select(*exprs)).one()
        stats: dict = {}
        for (name, stat), value in zip(labels, row):
            stats.setdefault(name, {})[stat] = value
        return stats

    def _profile_column(
        self, conn, source, col: ColumnInfo, stats: dict,
        rows_profiled: int, sampled: bool, is_key: bool,
    ) -> ColumnProfile:
        cfg = self.config
        if col.is_pii:
            return ColumnProfile(rows_profiled=rows_profiled, is_sampled=sampled,
                                 null_fraction=None, skipped_reason="pii")
        s = stats[col.name]
        non_null = s["non_null"]
        profile = ColumnProfile(
            rows_profiled=rows_profiled,
            is_sampled=sampled,
            null_fraction=round(1 - non_null / rows_profiled, 4) if rows_profiled else 0.0,
            distinct_count=s.get("distinct"),
        )
        if is_key:
            profile.skipped_reason = "key column"
            return profile

        if "min" in s:
            profile.min_value = to_jsonable(s["min"])
            profile.max_value = to_jsonable(s["max"])

        if col.type_family not in _VALUE_FAMILIES or non_null == 0:
            return profile

        # Most frequent values first; one row past the limit tells us if we truncated.
        c = source.c[col.name]
        freq = (
            sa.select(c)
            .where(c.is_not(None))
            .group_by(c)
            .order_by(sa.func.count().desc(), c)
            .limit(cfg.max_value_index + 1)
        )
        values = [to_jsonable(v) for v in conn.execute(freq).scalars()]
        truncated = len(values) > cfg.max_value_index
        values = values[: cfg.max_value_index]

        distinct = len(values) if not truncated else (profile.distinct_count or len(values))
        if (
            distinct <= cfg.categorical_max_distinct
            and distinct / non_null <= cfg.categorical_max_ratio
        ):
            profile.is_categorical = True
            profile.sample_values = values
            profile.case_style = case_style(values)
        elif col.type_family == "text":
            profile.value_index = values
            profile.value_index_truncated = truncated
        return profile
