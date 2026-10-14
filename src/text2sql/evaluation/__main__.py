"""Run the evaluation suite.

    python -m text2sql.evaluation offline     # no API key: golden set, mutations, security
    python -m text2sql.evaluation live        # calls Claude (asks first): accuracy, behaviour
    python -m text2sql.evaluation report      # rebuild docs/EVAL.md and the README block

Results are saved to docs/eval/*.json and the README's results block is
updated, so the published numbers are always the last measured ones.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

import sqlalchemy as sa

from .report import RESULTS_DIR, write_reports


def _setup(db_url: str, knowledge: Path):
    from ..guardrails import GuardedExecutor, GuardrailConfig, Guardrails
    from ..prompt import Knowledge
    from ..sandbox import SandboxedExecutor
    from ..schema import load_or_extract
    from .dataset import load_golden

    engine = sa.create_engine(db_url)
    name = Path(engine.url.database or "database").stem
    schema, _ = load_or_extract(engine, Path("cache/schema") / f"{name}.json")
    dialect = schema.dialect
    knowledge_obj = Knowledge.load(knowledge, dialect=dialect)
    today, cases = load_golden(knowledge, dialect=dialect)
    sandbox = SandboxedExecutor.for_schema(engine, schema)
    guarded = GuardedExecutor(sandbox, Guardrails(schema, GuardrailConfig(),
                                                  explain=sandbox.explain))
    return engine, schema, knowledge_obj, today, cases, sandbox, guarded


def offline(args) -> dict:
    from .mutation import MutationSuite
    from .security import run_security

    engine, schema, knowledge, today, cases, sandbox, guarded = _setup(args.db_url, args.knowledge)
    gold_sql = [c.sql for c in cases if c.sql] + [s for c in cases for s in c.accept]

    print(f"Golden set: {len(cases)} questions; checking every verified query runs…")
    empty = [s for s in gold_sql if not sandbox.run(s).rows]
    if empty:
        sys.exit(f"{len(empty)} verified queries return no rows:\n" + "\n".join(empty))
    databases = [schema.dialect]
    if args.pg_url:  # the same answers on PostgreSQL (translated by sqlglot)
        from ..verification.multiquery import compare_results
        from .dataset import load_golden

        pg = sa.create_engine(args.pg_url)
        from ..sandbox import SandboxedExecutor

        pg_run = SandboxedExecutor(pg)
        _, pg_cases = load_golden(args.knowledge, dialect="postgresql")
        for a, b in zip(cases, pg_cases):
            for x, y in zip([a.sql] + a.accept, [b.sql] + b.accept):
                if not x:
                    continue
                r1, r2 = sandbox.run(x), pg_run.run(y)
                if not compare_results(r1.columns, r1.rows, False, r2.columns, r2.rows, False).agree:
                    sys.exit(f"SQLite and PostgreSQL disagree on: {a.question}")
        databases.append("postgresql")

    print("Mutation suite…")
    mutation = MutationSuite(schema, knowledge, guarded).run(cases).to_dict()

    if engine.dialect.name == "sqlite":
        print("Security suite (on a copy of the database)…")
        tmp = Path(tempfile.mkdtemp()) / "copy.db"
        shutil.copy(engine.url.database, tmp)
        target = sa.create_engine(f"sqlite:///{tmp}")
    else:  # run as the SELECT-only role: it can't change anything
        print("Security suite (as the read-only role)…")
        target = engine
    security = run_security(target, schema, gold_sql).to_dict()

    return {
        "measured_at": date.today().isoformat(),
        "dataset": {"total": len(cases),
                    "by_expect": dict(Counter(c.expect for c in cases)),
                    "databases": databases},
        "mutation": mutation,
        "security": security,
    }


def live(args) -> dict:
    from ..execution import Text2SQLApp
    from .live import run_live
    from .security import fingerprint

    engine, schema, _, today, cases, sandbox, _ = _setup(args.db_url, args.knowledge)
    cases = cases[: args.limit] if args.limit else cases
    calls = len(cases) * 3.5 + 8 * 3.5
    print(f"{len(cases)} questions + 8 prompt-injection questions: roughly {calls:.0f} calls "
          f"to {args.model}.")
    if not args.yes and input("Run it? [y/N] ").strip().lower() != "y":
        sys.exit("Cancelled.")
    app = Text2SQLApp(engine, knowledge_dir=args.knowledge, today=today, model=args.model,
                      use_filter=not args.no_filter, multi_query=args.multi_query,
                      log_dir=Path("cache/logs/eval"))
    report = run_live(app.new_service(), cases, sandbox, schema.dialect, progress=True,
                      fingerprint=lambda: fingerprint(engine))
    data = report.to_dict()
    data.update(measured_at=date.today().isoformat(), model=args.model)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=["offline", "live", "report"])
    parser.add_argument("--db-url", default="sqlite:///data/ecommerce.db")
    parser.add_argument("--pg-url", help="also check the golden answers on this PostgreSQL")
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--multi-query", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--yes", action="store_true", help="don't ask before calling the API")
    parser.add_argument("--no-readme", action="store_true", help="don't touch README.md")
    args = parser.parse_args()
    import logging

    for name in ("text2sql.guardrails", "text2sql.sandbox"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    if args.mode in ("offline", "live"):
        data = offline(args) if args.mode == "offline" else live(args)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{args.mode}.json").write_text(json.dumps(data, indent=2, default=str))
        print(f"Saved {RESULTS_DIR / (args.mode + '.json')}")
        if args.mode == "offline":
            m, s = data["mutation"], data["security"]
            print(f"\n  golden set     {data['dataset']['total']} questions, verified on "
                  f"{' + '.join(data['dataset']['databases'])}"
                  f"\n  mutations      {m['detection_rate']:.0%} of {m['mutants']} injected bugs "
                  f"detected, {len(m['false_alarms'])} false alarms on {m['gold_checked']} correct queries"
                  f"\n  security       {s['refused'] + s['pg_refused']}/{s['attacks'] + s['pg_attacks']} "
                  f"attacks refused, {len(s['executed'])} executed, database unchanged: "
                  f"{s['database_unchanged']}, {len(s['false_blocks'])}/{s['legitimate']} legitimate blocked")
    if not args.no_readme:
        write_reports()
        print("Updated docs/EVAL.md and the README results block.")


if __name__ == "__main__":
    main()
