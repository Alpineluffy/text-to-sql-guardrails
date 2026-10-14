"""Ask questions in plain English; get SQL, answers, or a clarifying question.

    export ANTHROPIC_API_KEY=...
    python -m text2sql.ask                                   # interactive session
    python -m text2sql.ask "What was revenue from India last quarter?"
    python -m text2sql.ask "Top 10 products by units sold" --plan --csv top.csv

Choices made when asked to clarify are remembered for the rest of the session.
Every question is logged to cache/logs/queries.jsonl (no result data).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import sqlalchemy as sa

from .execution import ClarificationResponse, QueryResponse, Text2SQLService, create_service

MAX_PRINT_ROWS = 10

FAILURE_HEADINGS = {
    "invalid_sql": "The model couldn't produce valid SQL",
    "blocked": "Blocked by guardrails",
    "timeout": "Query timed out",
    "denied": "The database refused this query",
    "query_error": "Query failed",
    "generation_failed": "Couldn't generate SQL",
}


def format_table(columns: Sequence[str], rows: Sequence[tuple], limit: int = MAX_PRINT_ROWS,
                 truncated: bool = False) -> str:
    shown = [[("" if v is None else str(v)) for v in r] for r in rows[:limit]]
    widths = [max([len(c)] + [len(r[i]) for r in shown]) for i, c in enumerate(columns)]
    line = lambda cells: "  ".join(c.ljust(w) for c, w in zip(cells, widths))  # noqa: E731
    out = [line(columns), line(["-" * w for w in widths])] + [line(r) for r in shown]
    hidden = len(rows) - len(shown)
    if hidden > 0:
        out.append(f"... {hidden} more row(s) not shown")
    if truncated:
        out.append("... and more rows exist beyond the row limit")
    return "\n".join(out)


BAND_LABELS = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW"}


def print_confidence(r: QueryResponse) -> None:
    if r.confidence_band is None:
        if r.confidence_reasons:
            print(f"\nConfidence: unknown ({r.confidence_reasons[0]})")
        return
    print(f"\nConfidence: {BAND_LABELS[r.confidence_band]} ({r.confidence:.2f})")
    report = r.verification
    if report is not None:
        for problem in report.confidence.problems:
            print(f"  ! {problem}")
        for support in report.confidence.support:
            print(f"  + {support}")
    mq = report.multiquery if report is not None else None
    if mq is not None and mq.ran and mq.agreed is False:
        print(f"\nSecond approach (disagrees):\n{mq.alternative_sql}\n")
        if mq.alternative_explanation:
            print(mq.alternative_explanation)
        print(format_table(mq.alternative_columns, mq.alternative_rows))


def print_response(r: QueryResponse, show_plan: bool = False) -> None:
    if r.status == "cannot_answer":
        print(f"\nCan't answer this from the database: {r.explanation}")
        return
    if r.sql:
        print(f"\nSQL:\n{r.sql}\n")
    if r.explanation:
        print(r.explanation)
    for a in r.assumptions:
        print(f"  * {a}")
    for w in r.warnings:
        print(f"  ! {w}")
    for note in r.rewrites:
        print(f"  (guardrails: {note})")
    if r.attempts > 1:
        print(f"  (fixed after {r.attempts - 1} rejected attempt(s))")

    if not r.ok:
        print(f"\n{FAILURE_HEADINGS.get(r.status, 'Failed')}: {r.error}")
        return
    print("\n" + format_table(r.columns, r.rows, truncated=r.truncated))
    print_confidence(r)
    timing = f"{r.row_count} row(s)"
    if r.execution_ms is not None:
        timing += f" in {r.execution_ms:.0f} ms (database)"
    if r.total_ms is not None:
        timing += f", {r.total_ms / 1000:.1f} s total"
    print(f"\n{timing}  [request {r.request_id}]")
    if show_plan and r.plan:
        estimate = (f" (estimated rows read: {r.estimated_scan_rows:,})"
                    if r.estimated_scan_rows is not None else "")
        print(f"\nQuery plan{estimate}:")
        for step in r.plan:
            print(f"  {step}")


def ask_user_to_choose(c: ClarificationResponse) -> Optional[int]:
    print(f"\n{c.prompt}")
    for n, option in enumerate(c.options, 1):
        print(f"\n  [{n}] {option.label}: {option.description}")
        if option.columns:
            preview = format_table(option.columns, option.preview_rows, limit=3)
            print("      " + preview.replace("\n", "\n      "))
        elif option.error:
            print(f"      (preview failed: {option.error})")
    while True:
        choice = input(f"\nChoose 1-{len(c.options)} (or Enter to skip): ").strip()
        if not choice:
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(c.options):
            return int(choice) - 1


def handle(service: Text2SQLService, question: str, show_plan: bool,
           csv_path: Optional[Path]) -> None:
    response = service.ask(question)
    if isinstance(response, ClarificationResponse):
        index = ask_user_to_choose(response)
        if index is None:
            return
        response = service.choose(response.request_id, index)
    print_response(response, show_plan)
    if csv_path and response.ok:
        response.dataframe().to_csv(csv_path, index=False)
        print(f"Saved {response.row_count} row(s) to {csv_path}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="?", help="omit for an interactive session")
    parser.add_argument("--db-url", default="sqlite:///data/ecommerce.db",
                        help="on PostgreSQL, use the SELECT-only reader role's URL")
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--no-filter", action="store_true", help="send the full schema")
    parser.add_argument("--guardrails", type=Path, default=Path("config/guardrails.yaml"))
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--plan", action="store_true", help="show the query plan")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the Phase 3 checks (saves 2-3 model calls per question)")
    parser.add_argument("--multi-query", default="auto", choices=["auto", "always", "never"],
                        help="answer twice independently: complex queries only (auto)")
    parser.add_argument("--csv", type=Path, help="save the result rows to this CSV file")
    args = parser.parse_args(argv)
    # Blocked queries go to the audit log and the answer, not the console.
    logging.getLogger("text2sql.guardrails").setLevel(logging.ERROR)
    logging.getLogger("text2sql.sandbox").setLevel(logging.CRITICAL)

    try:
        service = create_service(
            sa.create_engine(args.db_url), knowledge_dir=args.knowledge,
            guardrails_path=args.guardrails, use_filter=not args.no_filter,
            model=args.model, effort=args.effort,
            verify=not args.no_verify, multi_query=args.multi_query,
        )
    except ValueError as e:
        sys.exit(str(e))

    questions = [args.question] if args.question else None
    while True:
        if questions is not None:
            if not questions:
                break
            question = questions.pop()
        else:
            try:
                question = input("\nQuestion (Enter to quit): ").strip()
            except EOFError:
                break
            if not question:
                break
        try:
            handle(service, question, args.plan, args.csv)
        except TypeError as e:
            # The Anthropic SDK raises TypeError when it finds no credentials at all.
            if "authentication method" not in str(e):
                raise
            sys.exit("\nNo Anthropic credentials found. Run: export ANTHROPIC_API_KEY=<your key>")


if __name__ == "__main__":
    main()
