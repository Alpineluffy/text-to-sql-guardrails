"""Run the API.

    export ANTHROPIC_API_KEY=...
    python -m text2sql.api                    # http://127.0.0.1:8000  (docs at /docs)
    python -m text2sql.api --demo             # no API key: replays the example library
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import sqlalchemy as sa
import uvicorn

from ..execution import Text2SQLApp
from ..feedback import FeedbackStore
from .app import create_app

HISTORY_DIR = Path(os.environ.get("TEXT2SQL_HISTORY_DIR", "cache/history"))


def build(db_url: str, knowledge: Path, demo: bool, use_filter: bool, multi_query: str):
    engine = sa.create_engine(db_url)
    if demo:
        from .demo import ReplayGenerator

        replay = ReplayGenerator(knowledge)
        text2sql = Text2SQLApp(engine, knowledge_dir=knowledge, use_filter=use_filter,
                               generator=replay, back_translate=False, multi_query="auto",
                               history_dir=HISTORY_DIR)
        replay.validator = text2sql.generator.validator
        replay.dialect = text2sql.schema.dialect
    else:
        text2sql = Text2SQLApp(engine, knowledge_dir=knowledge, use_filter=use_filter,
                               multi_query=multi_query, history_dir=HISTORY_DIR)
    return create_app(text2sql, FeedbackStore(Path("cache/feedback/feedback.jsonl")), demo=demo)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db-url", default=os.environ.get("TEXT2SQL_DB_URL",
                                                           "sqlite:///data/ecommerce.db"))
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--demo", action="store_true",
                        help="answer from the example library; no API key needed")
    parser.add_argument("--no-filter", action="store_true",
                        help="skip the embedding model (faster start)")
    parser.add_argument("--multi-query", default="auto", choices=["auto", "always", "never"])
    args = parser.parse_args()
    logging.getLogger("text2sql.guardrails").setLevel(logging.ERROR)
    logging.getLogger("text2sql.sandbox").setLevel(logging.CRITICAL)

    app = build(args.db_url, args.knowledge, args.demo, not args.no_filter, args.multi_query)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
