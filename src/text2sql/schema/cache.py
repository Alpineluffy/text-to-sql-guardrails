"""Reuse a saved schema until the database structure (or config) changes.

The structure hash only covers metadata, so data-only changes such as a new
status value are not detected. Use `max_age_hours` or `refresh=True` to
re-profile on a schedule.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

from sqlalchemy.engine import Engine

from .config import ExtractionConfig
from .extractor import SchemaExtractor
from .models import DatabaseSchema


def _is_fresh(cached: DatabaseSchema, max_age_hours: Optional[float]) -> bool:
    if max_age_hours is None:
        return True
    if not cached.extracted_at:
        return False
    age = datetime.now(timezone.utc) - datetime.fromisoformat(cached.extracted_at)
    return age <= timedelta(hours=max_age_hours)


def load_or_extract(
    engine: Engine,
    cache_path: Path,
    config: Optional[ExtractionConfig] = None,
    refresh: bool = False,
    max_age_hours: Optional[float] = None,
) -> Tuple[DatabaseSchema, bool]:
    """Return (schema, from_cache)."""
    extractor = SchemaExtractor(engine, config)
    structure = extractor.extract_structure()

    if not refresh and cache_path.exists():
        try:
            cached = DatabaseSchema.load(cache_path)
        except (ValueError, KeyError, TypeError):
            cached = None  # unreadable or from an older format: rebuild
        if (
            cached is not None
            and cached.structure_hash == structure.structure_hash
            and _is_fresh(cached, max_age_hours)
        ):
            return cached, True

    schema = extractor.enrich(structure)
    schema.save(cache_path)
    return schema, False
