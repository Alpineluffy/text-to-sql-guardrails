"""Print the prompt that would be sent to the LLM for a question.

    python -m text2sql.prompt "What was net revenue from India last quarter?"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..schema.models import DatabaseSchema
from .builder import PromptBuilder
from .knowledge import Knowledge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--schema", type=Path, default=Path("cache/schema/ecommerce.json"),
                        help="schema JSON from `python -m text2sql.schema`")
    parser.add_argument("--knowledge", type=Path, default=Path("knowledge/ecommerce"))
    parser.add_argument("--tables", help="comma-separated tables to include (default: all)")
    parser.add_argument("--filter", action="store_true",
                        help="pick tables and examples with embeddings (Step 3)")
    parser.add_argument("--system", action="store_true", help="also print the system prompt")
    args = parser.parse_args()

    if not args.schema.exists():
        sys.exit(f"{args.schema} not found; run `python -m text2sql.schema` first")
    schema = DatabaseSchema.load(args.schema)
    knowledge = Knowledge.load(args.knowledge)
    problems = knowledge.validate(schema)
    if problems:
        sys.exit("Knowledge files don't match the schema:\n  " + "\n  ".join(problems))

    tables = args.tables.split(",") if args.tables else None
    selector = selection = None
    if args.filter:
        from ..retrieval import EmbeddingExampleSelector, SchemaFilter, default_embedder

        embedder = default_embedder()
        selection = SchemaFilter(schema, knowledge, embedder).select(args.question)
        tables = selection.tables
        selector = EmbeddingExampleSelector(knowledge.examples, embedder)
    prompt = PromptBuilder(schema, knowledge, selector=selector).build(args.question, tables=tables)

    if args.system:
        print("=" * 30, "SYSTEM", "=" * 30)
        print(prompt.system)
        print("=" * 30, "USER", "=" * 32)
    print(prompt.user)
    print("\n" + "-" * 68)
    print(f"~{prompt.estimated_tokens} tokens | tables: {len(prompt.tables)} | "
          f"glossary: {prompt.glossary_terms or '-'} | ambiguous: {prompt.ambiguous_terms or '-'} | "
          f"examples: {len(prompt.examples)}")
    for w in prompt.warnings:
        print(f"warning: {w}")
    if selection:
        print("\nTable selection:")
        print(selection.explain())


if __name__ == "__main__":
    main()
