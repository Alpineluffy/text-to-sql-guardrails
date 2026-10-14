"""Extract a database schema to JSON.

    python -m text2sql.schema --db-url sqlite:///data/ecommerce.db
"""
from __future__ import annotations

import argparse
from pathlib import Path

import sqlalchemy as sa

from .cache import load_or_extract
from .config import ExtractionConfig
from .summary import format_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", default="sqlite:///data/ecommerce.db",
                        help="SQLAlchemy URL (default: the sample database)")
    parser.add_argument("--out", type=Path,
                        help="output JSON (default: cache/schema/<database>.json)")
    parser.add_argument("--namespace", help="DB schema to read, e.g. public")
    parser.add_argument("--sample-rows", type=int, default=100_000,
                        help="max rows profiled per table (0 = whole table)")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache")
    parser.add_argument("--max-age-hours", type=float,
                        help="re-profile when the cache is older than this")
    args = parser.parse_args()

    engine = sa.create_engine(args.db_url)
    name = Path(engine.url.database or "database").stem
    out = args.out or Path("cache/schema") / f"{name}.json"
    config = ExtractionConfig(namespace=args.namespace, sample_rows=args.sample_rows or None)

    schema, from_cache = load_or_extract(
        engine, out, config, refresh=args.refresh, max_age_hours=args.max_age_hours
    )
    print(format_summary(schema))
    print(f"\n{'Loaded from cache' if from_cache else 'Extracted and saved'}: {out}")


if __name__ == "__main__":
    main()
