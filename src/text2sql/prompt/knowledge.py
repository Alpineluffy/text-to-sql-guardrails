"""Hand-maintained knowledge about a database: descriptions, glossary, examples.

Lives in YAML (knowledge/<db>/semantic.yaml and examples.yaml) so analysts can
edit it without touching code. validate() checks it against the extracted
schema, so a renamed column fails loudly instead of silently misleading the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from ..schema.models import DatabaseSchema

_QUALIFIED_NAME = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_QUOTED = re.compile(r"'[^']*'")


@dataclass
class TableNotes:
    description: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    columns: Dict[str, str] = field(default_factory=dict)


@dataclass
class Definition:
    name: str
    sql: str
    note: Optional[str] = None


@dataclass
class GlossaryTerm:
    term: str
    definitions: List[Definition]
    synonyms: List[str] = field(default_factory=list)
    description: Optional[str] = None

    @property
    def is_ambiguous(self) -> bool:
        return len(self.definitions) > 1

    def matches(self, text: str) -> bool:
        return any(mentions(text, phrase) for phrase in [self.term, *self.synonyms])

    def resolve(self, text: str) -> List[Definition]:
        """Definitions the text names explicitly ("net revenue"), else all of them."""
        named = [d for d in self.definitions if mentions(text, d.name)]
        return named or self.definitions


@dataclass
class TermMatch:
    term: GlossaryTerm
    definitions: List[Definition]  # narrowed to what the question asked for
    from_preference: bool = False  # narrowed by an earlier user choice

    @property
    def is_ambiguous(self) -> bool:
        return len(self.definitions) > 1


@dataclass
class Example:
    question: str
    sql: str


SQLGLOT_DIALECTS = {"sqlite": "sqlite", "postgresql": "postgres", "mysql": "mysql"}


def translate_sql(sql: str, source: str, target: Optional[str]) -> str:
    """Rewrite SQL written for `source` in `target`'s dialect (no-op if the same)."""
    src = SQLGLOT_DIALECTS.get(source, source)
    dst = SQLGLOT_DIALECTS.get(target or source, target or source)
    if src == dst:
        return sql
    import sqlglot

    return sqlglot.transpile(sql, read=src, write=dst, pretty=True)[0]


@dataclass
class Knowledge:
    tables: Dict[str, TableNotes] = field(default_factory=dict)
    glossary: List[GlossaryTerm] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)
    sql_dialect: str = "sqlite"  # what the SQL in these files is written in

    @classmethod
    def load(cls, directory: Path, dialect: Optional[str] = None) -> Knowledge:
        """`dialect`: the database's dialect; example SQL is translated to it."""
        directory = Path(directory)
        semantic = _read_yaml(directory / "semantic.yaml") or {}
        examples = _read_yaml(directory / "examples.yaml") or []
        source = semantic.get("sql_dialect", "sqlite")
        return cls(
            tables={
                name: TableNotes(
                    description=notes.get("description"),
                    synonyms=notes.get("synonyms", []),
                    columns=notes.get("columns", {}),
                )
                for name, notes in (semantic.get("tables") or {}).items()
            },
            glossary=[
                GlossaryTerm(
                    term=t["term"],
                    synonyms=t.get("synonyms", []),
                    description=t.get("description"),
                    definitions=[Definition(**d) for d in t["definitions"]],
                )
                for t in semantic.get("glossary") or []
            ],
            examples=[Example(question=e["question"],
                              sql=translate_sql(e["sql"].strip(), source, dialect))
                      for e in examples],
            sql_dialect=dialect or source,
        )

    def matching_terms(
        self, question: str, preferences: Optional[Dict[str, str]] = None
    ) -> List[TermMatch]:
        """Glossary terms in the question.

        `preferences` maps a term to the definition the user chose earlier in
        the session; it applies unless the question names a definition itself.
        """
        preferences = preferences or {}
        matches = []
        for t in self.glossary:
            if not (t.matches(question) or any(mentions(question, d.name) for d in t.definitions)):
                continue
            definitions = t.resolve(question)
            preferred = [d for d in t.definitions if d.name == preferences.get(t.term)]
            if len(definitions) > 1 and preferred:
                matches.append(TermMatch(t, preferred, from_preference=True))
            else:
                matches.append(TermMatch(t, definitions))
        return matches

    def term(self, name: str) -> GlossaryTerm:
        for t in self.glossary:
            if t.term == name:
                return t
        raise KeyError(name)

    def validate(self, schema: DatabaseSchema) -> List[str]:
        """Problems where this knowledge refers to tables/columns that don't exist."""
        columns = {t.name: {c.name for c in t.columns} for t in schema.tables}
        problems = []
        for table, notes in self.tables.items():
            if table not in columns:
                problems.append(f"semantic.yaml: unknown table '{table}'")
                continue
            for col in notes.columns:
                if col not in columns[table]:
                    problems.append(f"semantic.yaml: unknown column '{table}.{col}'")
        for term in self.glossary:
            for d in term.definitions:
                for table, col in _QUALIFIED_NAME.findall(_QUOTED.sub("''", d.sql)):
                    if table not in columns or col not in columns[table]:
                        problems.append(
                            f"glossary '{term.term}' ({d.name}): unknown column '{table}.{col}'"
                        )
        return problems


def find_phrase(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    """Span of a whole-phrase, case-insensitive match that tolerates a plural 's'."""
    words = phrase.lower().split()
    if not words:
        return None
    pattern = r"\b" + r"\s+".join(map(re.escape, words)) + r"s?\b"
    m = re.search(pattern, text.lower())
    return m.span() if m else None


def mentions(text: str, phrase: str) -> bool:
    return find_phrase(text, phrase) is not None


def _read_yaml(path: Path):
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())
