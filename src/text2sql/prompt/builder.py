"""Assemble the prompt for one user question.

The system prompt holds everything that is the same for every question on a
database (role, rules, dialect notes, response format), so it can be prompt
cached. The user message holds what changes per question: the relevant schema,
matched values and glossary terms, date context and examples.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Dict, Iterable, List, Optional

from ..schema.models import DatabaseSchema
from .dates import date_context
from .examples import ExampleSelector, LexicalExampleSelector
from .knowledge import Example, Knowledge, TermMatch, mentions
from .render import (
    ValueMention, find_value_mentions, render_relationships, render_schema, sql_literal,
)

DIALECT_NOTES: Dict[str, List[str]] = {
    "sqlite": [
        "Dates are stored as ISO text ('YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'): compare them "
        "as strings, and use strftime()/date() for parts and arithmetic, e.g. "
        "strftime('%Y-%m', order_date), date('2026-10-06', '-90 days').",
        "Booleans are stored as 1/0.",
        "There is no ILIKE; LIKE is already case-insensitive for ASCII text.",
    ],
    "postgresql": [
        "Use date_trunc('month', col) and INTERVAL arithmetic for dates.",
        "Use ILIKE for case-insensitive matching.",
    ],
    "mysql": [
        "Use DATE_FORMAT() and DATE_SUB()/INTERVAL for dates.",
        "Quote identifiers with backticks when needed.",
    ],
}

SYSTEM_TEMPLATE = """\
You are an expert data analyst. You translate questions into {dialect} SQL for a \
read-only analytics database.

Rules:
1. Write exactly one SELECT statement (a WITH ... SELECT is fine). Never modify data.
2. Use only the tables and columns in the schema provided. Never invent names. If the \
question can't be answered from that schema, return "sql": null and explain why.
3. Join tables only through the relationships listed.
4. Filter on values exactly as listed in the schema, with the same spelling and case. \
If the question mentions a value that isn't listed, say so in assumptions.
5. When the question uses a term from the business glossary, apply its definition. \
If the term has several definitions and the question doesn't say which one, don't \
guess: set needs_clarification to true, set sql to null, and give one \
interpretation (with its SQL) per definition, labelled with the definition's name. \
Do the same for any other ambiguity that would change the answer.
6. Resolve relative dates ("last quarter", "this year") with the date context provided.
7. Don't select columns marked SENSITIVE unless the question explicitly needs them.
8. Use explicit JOINs, readable aliases and ORDER BY for rankings.

{dialect} notes:
{dialect_notes}

Respond with only a JSON object with these fields:
- sql: the query, or null
- explanation: what the query returns, in plain English
- tables_used: table names referenced
- columns_used: every column referenced, as table.column
- assumptions: interpretations you made that the user should know about
- confidence: 0 to 1, how sure you are that the query answers the question
- needs_clarification: true only when the question is ambiguous
- clarification: null, or {{"question": ..., "interpretations": [{{"label": ..., \
"description": ..., "sql": ...}}]}}"""


@dataclass
class Prompt:
    system: str
    user: str
    tables: List[str]
    glossary_terms: List[str]
    ambiguous_terms: List[str]
    term_matches: List[TermMatch]
    value_mentions: List[ValueMention]
    examples: List[Example]
    estimated_tokens: int
    warnings: List[str] = field(default_factory=list)

    def messages(self) -> List[Dict[str, str]]:
        return [{"role": "user", "content": self.user}]

    def with_section(self, title: str, body: str) -> "Prompt":
        return replace(self, user=f"{self.user}\n\n## {title}\n{body}")

    def with_feedback(self, feedback: str) -> "Prompt":
        """Same prompt plus why the last answer was rejected, for a repair attempt.

        Kept single-turn on purpose: the system prompt stays cached and no
        earlier model output has to be replayed.
        """
        return self.with_section("Your previous answer was rejected", feedback)


def estimate_tokens(text: str) -> int:
    return len(text) // 4  # rough rule for English and SQL


class PromptBuilder:
    def __init__(
        self,
        schema: DatabaseSchema,
        knowledge: Knowledge,
        selector: Optional[ExampleSelector] = None,
        max_examples: int = 3,
        token_budget: int = 8_000,
    ):
        self.schema = schema
        self.knowledge = knowledge
        self.selector = selector or LexicalExampleSelector(knowledge.examples)
        self.max_examples = max_examples
        self.token_budget = token_budget
        self.system = SYSTEM_TEMPLATE.format(
            dialect=self._dialect_label(),
            dialect_notes="\n".join(f"- {n}" for n in DIALECT_NOTES.get(schema.dialect, [])),
        )

    def build(
        self,
        question: str,
        tables: Optional[Iterable[str]] = None,
        today: Optional[date] = None,
        preferences: Optional[Dict[str, str]] = None,
        max_examples: Optional[int] = None,
    ) -> Prompt:
        """`tables` comes from schema filtering (Step 3); default is every table.
        `preferences` holds glossary choices the user made earlier (Step 4).
        `max_examples` overrides the builder's default (0 = no examples)."""
        known = [t.name for t in self.schema.tables]
        selected = [t for t in known if tables is None or t in set(tables)]
        warnings = []
        if tables is not None:
            unknown = sorted(set(tables) - set(known))
            if unknown:
                warnings.append(f"ignored unknown tables: {', '.join(unknown)}")

        terms = self.knowledge.matching_terms(question, preferences)
        mentions = find_value_mentions(question, self.schema, selected)
        examples = self._select_examples(question, terms)
        if max_examples is not None:
            examples = examples[:max_examples]
        today = today or date.today()

        max_values = 20
        while True:
            user = self._user_message(question, selected, terms, mentions, examples, today, max_values)
            tokens = estimate_tokens(self.system) + estimate_tokens(user)
            if tokens <= self.token_budget:
                break
            # Shed the least essential context first: examples, then long value lists.
            if examples:
                examples = examples[:-1]
                warnings.append("dropped an example to fit the token budget")
            elif max_values > 5:
                max_values = 5
                warnings.append("shortened value lists to fit the token budget")
            else:
                warnings.append(
                    f"prompt is ~{tokens} tokens, over the {self.token_budget} budget; "
                    "filter the schema to fewer tables"
                )
                break

        return Prompt(
            system=self.system,
            user=user,
            tables=selected,
            glossary_terms=[m.term.term for m in terms],
            ambiguous_terms=[m.term.term for m in terms if m.is_ambiguous],
            term_matches=terms,
            value_mentions=mentions,
            examples=examples,
            estimated_tokens=tokens,
            warnings=warnings,
        )

    def _select_examples(self, question: str, terms: List[TermMatch]) -> List[Example]:
        # An example that commits to one meaning of an unresolved term ("net
        # revenue") would nudge the model to pick that meaning instead of asking.
        biased = [d.name for m in terms if m.is_ambiguous for d in m.definitions]
        candidates = self.selector.select(question, len(self.knowledge.examples))
        unbiased = [e for e in candidates if not any(mentions(e.question, b) for b in biased)]
        return unbiased[: self.max_examples]

    def _user_message(
        self, question: str, tables: List[str], terms: List[TermMatch],
        mentions: List[ValueMention], examples: List[Example], today: date, max_values: int,
    ) -> str:
        sections = [("Schema", render_schema(self.schema, self.knowledge, tables, max_values))]

        joins = render_relationships(self.schema, tables)
        if joins:
            sections.append(("Relationships (the only valid joins)", joins))

        if mentions:
            lines = []
            for m in mentions:
                literal = sql_literal(m.value, self.schema.dialect)
                note = f'  (the question says "{m.phrase}")' if m.case_differs else ""
                lines.append(f"{m.table}.{m.column} = {literal}{note}")
            sections.append(("Values mentioned in the question", "\n".join(lines)))

        if terms:
            sections.append(("Business glossary", "\n".join(_render_term(t) for t in terms)))

        sections.append(("Date context", "\n".join(date_context(today))))

        if examples:
            sections.append((
                "Examples of correct queries for this database",
                "\n\n".join(f"Question: {e.question}\nSQL:\n{e.sql}" for e in examples),
            ))

        sections.append(("Question", question))
        return "\n\n".join(f"## {title}\n{body}" for title, body in sections)

    def _dialect_label(self) -> str:
        return {"sqlite": "SQLite", "postgresql": "PostgreSQL", "mysql": "MySQL"}.get(
            self.schema.dialect, self.schema.dialect
        )


def _render_term(match: TermMatch) -> str:
    term = match.term
    aka = f" (also: {', '.join(term.synonyms)})" if term.synonyms else ""
    lines = [f"- {term.term}{aka}" + (f": {term.description}" if term.description else "")]
    if match.is_ambiguous:
        lines.append("  AMBIGUOUS: several definitions are in use. Ask which one unless the question says.")
    elif match.from_preference:
        lines.append("  The user chose this definition earlier in the session; use it.")
    for d in match.definitions:
        note = f" ({d.note})" if d.note else ""
        lines.append(f"  - {d.name}: {d.sql}{note}")
    return "\n".join(lines)
