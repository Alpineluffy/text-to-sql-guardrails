"""Data model for an extracted database schema.

Everything here is plain data that round-trips through JSON, so the extracted
schema can be cached on disk and handed to the prompt builder (Step 2) without
touching the database again.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ColumnProfile:
    """What the data in a column actually looks like."""

    rows_profiled: int
    is_sampled: bool  # True when only the first `sample_rows` rows were read
    null_fraction: Optional[float]  # None when the column wasn't read (PII)
    distinct_count: Optional[int] = None
    # Categorical columns: every distinct value, most frequent first. These go
    # into the prompt so the model writes 'DELIVERED' rather than 'delivered'.
    is_categorical: bool = False
    sample_values: List[Any] = field(default_factory=list)
    case_style: Optional[str] = None  # UPPER, lower, Title or mixed
    # High-cardinality text columns: distinct values kept for exact keyword
    # lookup during schema filtering (Step 3), never sent to the prompt whole.
    value_index: List[str] = field(default_factory=list)
    value_index_truncated: bool = False
    min_value: Any = None
    max_value: Any = None
    skipped_reason: Optional[str] = None  # why values were not collected


@dataclass
class ColumnInfo:
    name: str
    type: str  # type as the database reports it, e.g. "VARCHAR(50)"
    type_family: str  # text, integer, numeric, boolean, date, datetime, ...
    nullable: bool
    is_primary_key: bool = False
    is_pii: bool = False
    comment: Optional[str] = None
    profile: Optional[ColumnProfile] = None


@dataclass
class ForeignKey:
    source_table: str
    source_columns: List[str]
    target_table: str
    target_columns: List[str]
    origin: str = "declared"  # "declared" by the database or "inferred" by us
    # For inferred keys: share of sampled source values found in the target.
    match_ratio: Optional[float] = None

    def describe(self) -> str:
        src = ", ".join(f"{self.source_table}.{c}" for c in self.source_columns)
        dst = ", ".join(f"{self.target_table}.{c}" for c in self.target_columns)
        return f"{src} -> {dst}"


@dataclass
class TableInfo:
    name: str
    kind: str  # "table" or "view"
    columns: List[ColumnInfo]
    primary_key: List[str] = field(default_factory=list)
    comment: Optional[str] = None
    row_count: Optional[int] = None

    def column(self, name: str) -> ColumnInfo:
        for col in self.columns:
            if col.name == name:
                return col
        raise KeyError(f"{self.name}.{name}")


@dataclass
class DatabaseSchema:
    dialect: str
    database: Optional[str]
    namespace: Optional[str]  # DB schema such as "public"; None = default
    tables: List[TableInfo]
    foreign_keys: List[ForeignKey]
    structure_hash: str
    extracted_at: Optional[str] = None  # ISO timestamp, set once profiled

    def table(self, name: str) -> TableInfo:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(name)

    def foreign_keys_for(self, table_name: str) -> List[ForeignKey]:
        return [
            fk
            for fk in self.foreign_keys
            if table_name in (fk.source_table, fk.target_table)
        ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatabaseSchema:
        tables = []
        for t in data["tables"]:
            columns = []
            for c in t["columns"]:
                profile = c.get("profile")
                columns.append(
                    ColumnInfo(**{**c, "profile": ColumnProfile(**profile) if profile else None})
                )
            tables.append(TableInfo(**{**t, "columns": columns}))
        return cls(
            **{
                **data,
                "tables": tables,
                "foreign_keys": [ForeignKey(**fk) for fk in data["foreign_keys"]],
            }
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> DatabaseSchema:
        return cls.from_dict(json.loads(path.read_text()))
