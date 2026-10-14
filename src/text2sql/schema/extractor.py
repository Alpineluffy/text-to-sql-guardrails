"""Introspect a database into a DatabaseSchema.

Extraction runs in two stages so the cache can skip the expensive one:

1. extract_structure(): metadata only (tables, columns, keys, comments).
   Cheap, and its hash tells us whether a cached profile is still valid.
2. enrich(): infers missing foreign keys and profiles the data.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from .config import ExtractionConfig
from .fk_inference import infer_foreign_keys
from .models import ColumnInfo, DatabaseSchema, ForeignKey, TableInfo
from .pii import is_pii
from .profiler import TableProfiler


def type_family(sa_type: sa.types.TypeEngine) -> str:
    # Order matters: Text/Enum subclass String, Float subclasses Numeric.
    if isinstance(sa_type, sa.Boolean):
        return "boolean"
    if isinstance(sa_type, sa.Integer):
        return "integer"
    if isinstance(sa_type, sa.Numeric):
        return "numeric"
    if isinstance(sa_type, sa.DateTime):
        return "datetime"
    if isinstance(sa_type, sa.Date):
        return "date"
    if isinstance(sa_type, sa.Time):
        return "time"
    if isinstance(sa_type, sa.String):
        return "text"
    if isinstance(sa_type, sa.LargeBinary):
        return "binary"
    if isinstance(sa_type, sa.JSON):
        return "json"
    return "other"


class SchemaExtractor:
    def __init__(self, engine: Engine, config: Optional[ExtractionConfig] = None):
        self.engine = engine
        self.config = config or ExtractionConfig()
        self._metadata: Optional[sa.MetaData] = None

    def extract(self) -> DatabaseSchema:
        return self.enrich(self.extract_structure())

    def extract_structure(self) -> DatabaseSchema:
        cfg = self.config
        metadata = sa.MetaData()
        metadata.reflect(bind=self.engine, schema=cfg.namespace, views=cfg.include_views)
        self._metadata = metadata
        view_names = set(sa.inspect(self.engine).get_view_names(schema=cfg.namespace))

        tables: List[TableInfo] = []
        foreign_keys: List[ForeignKey] = []
        for table in sorted(metadata.tables.values(), key=lambda t: t.name):
            if table.name in cfg.exclude_tables:
                continue
            pk = [c.name for c in table.primary_key.columns]
            columns = [
                ColumnInfo(
                    name=col.name,
                    type=self._type_name(col.type),
                    type_family=type_family(col.type),
                    nullable=bool(col.nullable),
                    is_primary_key=col.name in pk,
                    is_pii=is_pii(col.name, cfg.pii_patterns),
                    comment=col.comment,
                )
                for col in table.columns
            ]
            tables.append(
                TableInfo(
                    name=table.name,
                    kind="view" if table.name in view_names else "table",
                    columns=columns,
                    primary_key=pk,
                    comment=table.comment,
                )
            )
            for constraint in table.foreign_key_constraints:
                if constraint.referred_table.name in cfg.exclude_tables:
                    continue
                foreign_keys.append(
                    ForeignKey(
                        source_table=table.name,
                        source_columns=[c.name for c in constraint.columns],
                        target_table=constraint.referred_table.name,
                        target_columns=[e.column.name for e in constraint.elements],
                    )
                )

        schema = DatabaseSchema(
            dialect=self.engine.dialect.name,
            database=self.engine.url.database,
            namespace=cfg.namespace,
            tables=tables,
            foreign_keys=foreign_keys,
            structure_hash="",
        )
        schema.structure_hash = self.structure_hash(schema)
        return schema

    def enrich(self, schema: DatabaseSchema) -> DatabaseSchema:
        if self._metadata is None:
            raise RuntimeError("call extract_structure() before enrich()")
        sa_tables = {t.name: t for t in self._metadata.tables.values()}

        if self.config.infer_foreign_keys:
            schema.foreign_keys.extend(
                infer_foreign_keys(self.engine, schema, sa_tables, self.config)
            )

        key_columns = {(t.name, c) for t in schema.tables for c in t.primary_key}
        for fk in schema.foreign_keys:
            key_columns.update((fk.source_table, c) for c in fk.source_columns)
            key_columns.update((fk.target_table, c) for c in fk.target_columns)

        profiler = TableProfiler(self.engine, self.config)
        for table in schema.tables:
            if table.kind == "view" and not self.config.profile_views:
                continue
            profiler.profile(sa_tables[table.name], table, key_columns)

        schema.extracted_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return schema

    def structure_hash(self, schema: DatabaseSchema) -> str:
        """Fingerprint of the structure plus config; changes invalidate the cache."""
        payload = {
            "tables": [
                [t.name, t.kind, t.primary_key, [[c.name, c.type, c.nullable] for c in t.columns]]
                for t in schema.tables
            ],
            "foreign_keys": [asdict(fk) for fk in schema.foreign_keys],
            "config": asdict(self.config),
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def _type_name(self, sa_type: sa.types.TypeEngine) -> str:
        try:
            return sa_type.compile(dialect=self.engine.dialect)
        except Exception:
            return type(sa_type).__name__.upper()
