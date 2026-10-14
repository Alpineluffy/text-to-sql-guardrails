"""Step 3: pick the tables a question needs (schema linking).

Two kinds of evidence, combined:

- Exact matches always include a table: its name or a synonym is mentioned,
  a column name is mentioned, a real value is mentioned ("India"), or a
  glossary term in the question is defined on it.
- Semantic similarity between the question and each table's and column's
  document adds the best-scoring tables, up to `max_tables` and only above
  `min_score`.

Bridge tables needed to join the selection are then added. Missing a table
guarantees a wrong query while an extra one only costs tokens, so when nothing
qualifies the filter falls back to the full schema rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from ..prompt.knowledge import Knowledge, find_phrase
from ..prompt.render import find_value_mentions
from ..schema.models import DatabaseSchema
from .documents import build_documents, humanize
from .embeddings import Embedder, similarities
from .joins import connect

# Column names too generic to count as evidence on their own.
GENERIC_COLUMNS = {"id", "name", "type", "date", "created at", "updated at", "description"}
_QUALIFIED_NAME = re.compile(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*\b")


@dataclass(frozen=True)
class FilterConfig:
    max_tables: int = 2  # cap on tables added by similarity (exact matches always count)
    # Minimum cosine similarity for a similarity-only match. Model-specific:
    # tuned for all-MiniLM-L6-v2 on knowledge/ecommerce/linking_eval.yaml.
    min_score: float = 0.2
    # ...and at least this fraction of the best table's score, so a clear
    # winner (0.46) doesn't drag in weak matches (0.25). 0 disables it.
    relative_to_best: float = 0.5
    full_schema_below: int = 0  # skip filtering when the DB has fewer tables than this
    add_bridge_tables: bool = True


@dataclass
class Selection:
    tables: List[str]
    reasons: Dict[str, List[str]]  # why each selected table is in
    scores: Dict[str, float]  # best similarity for every table
    best_match: Dict[str, str]  # which document gave that score
    used_full_schema: bool = False
    unreachable: List[str] = field(default_factory=list)

    def explain(self) -> str:
        lines = []
        for table in self.tables:
            lines.append(f"  + {table:<18} {'; '.join(self.reasons[table])}")
        for table in sorted(self.scores, key=self.scores.get, reverse=True):
            if table not in self.tables:
                lines.append(f"  - {table:<18} similarity {self.scores[table]:.2f}")
        if self.unreachable:
            lines.append(f"  ! no join path to: {', '.join(self.unreachable)}")
        return "\n".join(lines)


class SchemaFilter:
    def __init__(
        self,
        schema: DatabaseSchema,
        knowledge: Knowledge,
        embedder: Embedder,
        config: FilterConfig = FilterConfig(),
    ):
        self.schema = schema
        self.knowledge = knowledge
        self.embedder = embedder
        self.config = config
        self.documents = build_documents(schema, knowledge)
        self._doc_vectors = embedder.embed([d.text for d in self.documents])

    def select(self, question: str) -> Selection:
        all_tables = [t.name for t in self.schema.tables]
        scores, best_match = self._similarity(question)

        if len(all_tables) < self.config.full_schema_below:
            return Selection(
                all_tables, {t: ["small database: full schema"] for t in all_tables},
                scores, best_match, used_full_schema=True,
            )

        reasons: Dict[str, List[str]] = {}
        for table, why in self._exact_matches(question):
            if why not in reasons.setdefault(table, []):
                reasons[table].append(why)

        ranked = sorted(all_tables, key=lambda t: scores[t], reverse=True)
        floor = max(self.config.min_score, self.config.relative_to_best * scores[ranked[0]])
        added = 0
        for table in ranked:
            if added >= self.config.max_tables or scores[table] < floor:
                break
            if table not in reasons:
                added += 1
            reasons.setdefault(table, []).append(
                f"similarity {scores[table]:.2f} ({best_match[table]})"
            )

        if not reasons:
            return Selection(
                all_tables, {t: ["nothing matched: full schema"] for t in all_tables},
                scores, best_match, used_full_schema=True,
            )

        selected = sorted(reasons, key=lambda t: scores[t], reverse=True)
        unreachable: List[str] = []
        if self.config.add_bridge_tables:
            bridges, unreachable = connect(self.schema, selected)
            for table, a, b in bridges:
                reasons.setdefault(table, []).append(f"bridge: joins {a} to {b}")
                if table not in selected:
                    selected.append(table)

        # Keep schema order so prompts are stable across questions.
        ordered = [t for t in all_tables if t in reasons]
        return Selection(ordered, reasons, scores, best_match, unreachable=unreachable)

    def _similarity(self, question: str):
        query = self.embedder.embed([question], is_query=True)[0]
        sims = similarities(self._doc_vectors, query)
        scores: Dict[str, float] = {t.name: 0.0 for t in self.schema.tables}
        best_match: Dict[str, str] = {t.name: "-" for t in self.schema.tables}
        for doc, sim in zip(self.documents, sims):
            if sim > scores[doc.table]:
                scores[doc.table] = float(sim)
                best_match[doc.table] = f"column {doc.column}" if doc.column else "table"
        return scores, best_match

    def _exact_matches(self, question: str):
        # (table, why, span) for every table/synonym/column name in the question.
        hits: List[Tuple[str, str, Tuple[int, int]]] = []
        for table in self.schema.tables:
            notes = self.knowledge.tables.get(table.name)
            name = humanize(table.name)
            phrases = {name, name[:-1] if name.endswith("s") else name}
            phrases.update(notes.synonyms if notes else [])
            for phrase in sorted(phrases):
                span = find_phrase(question, phrase)
                if span:
                    hits.append((table.name, f"mentions '{phrase}'", span))
            if table.kind == "view":
                continue  # views repeat their base tables' columns; don't double count
            for col in table.columns:
                col_name = humanize(col.name)
                if (
                    col.name in table.primary_key
                    or col_name in GENERIC_COLUMNS
                    or col_name.endswith(" id")
                ):
                    continue
                span = find_phrase(question, col_name)
                if span:
                    hits.append((table.name, f"mentions column '{col.name}'", span))

        # "line items" names order_items; the "item" inside it is not products.
        for table, why, (start, end) in hits:
            shadowed = any(
                s <= start and end <= e and (e - s) > (end - start)
                for _, _, (s, e) in hits
            )
            if not shadowed:
                yield table, why

        all_names = [t.name for t in self.schema.tables]
        for m in find_value_mentions(question, self.schema, all_names):
            yield m.table, f"value {m.value!r} in {m.column}"

        known = set(all_names)
        for match in self.knowledge.matching_terms(question):
            for d in match.definitions:
                for table in sorted(set(_QUALIFIED_NAME.findall(d.sql)) & known):
                    yield table, f"glossary '{match.term.term}'"
