"""Review user feedback; see feedback/review.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sqlalchemy as sa

from .review import PromotionError, promote, reject
from .store import FeedbackStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Review feedback and promote it.")
    parser.add_argument("--store", type=Path, default=Path("cache/feedback/feedback.jsonl"))
    parser.add_argument("--db-url", default="sqlite:///data/ecommerce.db")
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list")
    listing.add_argument("--status", default="pending",
                         choices=["pending", "promoted", "rejected", "all"])
    p = sub.add_parser("promote")
    p.add_argument("feedback_id")
    p.add_argument("--reload-url", default="http://127.0.0.1:8000",
                   help="tell the running API to pick up the change ('' to skip)")
    r = sub.add_parser("reject")
    r.add_argument("feedback_id")
    r.add_argument("--reason", default="")
    args = parser.parse_args()

    store = FeedbackStore(args.store)
    if args.command == "list":
        items = store.items(None if args.status == "all" else args.status)
        if not items:
            print(f"No {args.status} feedback.")
        for i in items:
            print(f"{i.feedback_id}  {i.status:<9} {i.kind:<17} [{i.confidence_band or '-'}] "
                  f"{i.question}")
            if i.comment:
                print(f"            comment: {i.comment}")
            if i.note:
                print(f"            {i.note}")
        return

    item = store.get(args.feedback_id)
    if item is None:
        sys.exit(f"No feedback with id {args.feedback_id}")
    if args.command == "reject":
        reject(item, store, args.reason)
        print(f"Rejected {item.feedback_id}.")
        return

    from ..generation import SQLValidator
    from ..guardrails import GuardedExecutor, GuardrailConfig, Guardrails
    from ..sandbox import SandboxedExecutor
    from ..schema import load_or_extract

    engine = sa.create_engine(args.db_url)
    schema, _ = load_or_extract(engine, Path("cache/schema") / f"{Path(engine.url.database).stem}.json")
    config = GuardrailConfig()
    sandbox = SandboxedExecutor.for_schema(engine, schema, config)
    executor = GuardedExecutor(sandbox, Guardrails(schema, config, explain=sandbox.explain))
    try:
        target = promote(item, store, args.knowledge, SQLValidator(schema), executor)
    except PromotionError as e:
        sys.exit(f"Not promoted: {e}")
    print(f"Promoted {item.feedback_id} to {target}.")
    if args.reload_url:
        import httpx

        try:
            httpx.post(f"{args.reload_url}/v1/admin/reload", timeout=60).raise_for_status()
            print("The running API reloaded its knowledge; the change is live.")
        except httpx.HTTPError:
            print("No running API reached; the change applies on its next start.")


if __name__ == "__main__":
    main()
