from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .pii import DEFAULT_PII_PATTERNS


@dataclass(frozen=True)
class ExtractionConfig:
    namespace: Optional[str] = None  # DB schema to read, e.g. "public"
    include_views: bool = True
    profile_views: bool = False  # views can be expensive to scan
    exclude_tables: Tuple[str, ...] = ()

    # Profiling reads at most this many rows per table (None = whole table).
    sample_rows: Optional[int] = 100_000
    # A text/boolean column is categorical when it has at most this many
    # distinct values AND distinct/non-null stays under the ratio, so a
    # 30-row table of unique product names is not mistaken for categories.
    categorical_max_distinct: int = 50
    categorical_max_ratio: float = 0.5
    max_value_index: int = 1_000

    infer_foreign_keys: bool = True
    fk_min_match_ratio: float = 0.9
    fk_check_rows: int = 10_000

    pii_patterns: Tuple[str, ...] = DEFAULT_PII_PATTERNS
