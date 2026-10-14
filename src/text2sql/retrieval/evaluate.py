"""Measure schema-linking recall on a labelled question set.

    python -m text2sql.retrieval.evaluate            # default settings, per question
    python -m text2sql.retrieval.evaluate --sweep    # compare thresholds

Recall is what matters: a question is "covered" when every table it needs was
selected. Fewer tables per prompt is the secondary goal.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List

import yaml

from ..prompt.knowledge import Knowledge
from ..schema.models import DatabaseSchema
from .embeddings import CachedEmbedder, HashingEmbedder, SentenceTransformerEmbedder
from .linking import FilterConfig, SchemaFilter

MODELS = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", ""),
    "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: "),
}


@dataclass
class Case:
    question: str
    tables: List[str]


@dataclass
class Report:
    covered: int
    total: int
    table_recall: float
    avg_selected: float
    misses: List[str]

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0


def load_cases(path: Path) -> List[Case]:
    return [Case(c["question"], c["tables"]) for c in yaml.safe_load(path.read_text())]


def evaluate(schema_filter: SchemaFilter, cases: List[Case], verbose: bool = False) -> Report:
    covered, found, needed, selected_total, misses = 0, 0, 0, 0, []
    for case in cases:
        selection = schema_filter.select(case.question)
        missing = sorted(set(case.tables) - set(selection.tables))
        found += len(case.tables) - len(missing)
        needed += len(case.tables)
        selected_total += len(selection.tables)
        if missing:
            misses.append(f"{case.question}  -> missing {', '.join(missing)}")
        else:
            covered += 1
        if verbose:
            mark = "MISS" if missing else "ok  "
            print(f"{mark} [{len(selection.tables)}] {case.question}")
            if missing:
                print(f"       needed {case.tables}, missing {missing}")
                print(selection.explain())
    return Report(covered, len(cases), found / needed, selected_total / len(cases), misses)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--schema", type=Path, default=Path("cache/schema/ecommerce.json"))
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--model", choices=[*MODELS, "hashing"], default="minilm")
    parser.add_argument("--max-tables", type=int, default=FilterConfig.max_tables)
    parser.add_argument("--min-score", type=float, default=FilterConfig.min_score)
    parser.add_argument("--relative", type=float, default=FilterConfig.relative_to_best)
    parser.add_argument("--sweep", action="store_true", help="try a grid of settings")
    args = parser.parse_args()

    if not args.schema.exists():
        sys.exit(f"{args.schema} not found; run `python -m text2sql.schema` first")
    schema = DatabaseSchema.load(args.schema)
    knowledge = Knowledge.load(args.knowledge)
    cases = load_cases(args.knowledge / "linking_eval.yaml")

    if args.model == "hashing":
        embedder = HashingEmbedder()
    else:
        name, prefix = MODELS[args.model]
        embedder = CachedEmbedder(SentenceTransformerEmbedder(name, prefix), Path("cache/embeddings"))

    base = FilterConfig(
        max_tables=args.max_tables, min_score=args.min_score, relative_to_best=args.relative
    )
    schema_filter = SchemaFilter(schema, knowledge, embedder, base)
    n_tables = len(schema.tables)

    if args.sweep:
        print(f"{embedder.name}, {len(cases)} questions, {n_tables} tables in the database\n")
        print("max_tables  min_score  relative  covered  table_recall  avg_tables")
        for max_tables in (2, 3):
            for min_score in (0.2, 0.25):
                for relative in (0.0, 0.5, 0.6, 0.7, 0.8):
                    schema_filter.config = replace(
                        base, max_tables=max_tables, min_score=min_score, relative_to_best=relative
                    )
                    r = evaluate(schema_filter, cases)
                    print(f"{max_tables:>10}  {min_score:>9}  {relative:>8}  {r.covered:>2}/{r.total:<4} "
                          f"{r.table_recall:>13.1%}  {r.avg_selected:>10.2f}")
        schema_filter.config = base
        no_semantic = replace(base, max_tables=0)
        schema_filter.config = no_semantic
        r = evaluate(schema_filter, cases)
        print(f"\nexact matches only (no embeddings): {r.covered}/{r.total} covered, "
              f"{r.avg_selected:.2f} tables")
        return

    r = evaluate(schema_filter, cases, verbose=True)
    print(f"\n{embedder.name}  max_tables={base.max_tables} min_score={base.min_score} "
          f"relative_to_best={base.relative_to_best}")
    print(f"questions fully covered: {r.covered}/{r.total} ({r.coverage:.0%})")
    print(f"table recall: {r.table_recall:.1%}")
    print(f"avg tables per prompt: {r.avg_selected:.2f} of {n_tables}")


if __name__ == "__main__":
    main()
