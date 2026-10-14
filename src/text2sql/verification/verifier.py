"""Run every Phase 3 check on an answered question and score the result.

Cost per question: result sanity checks and coverage are free (local).
Back-translation is two extra model calls; the multi-query check is one more
call plus one query, and by default only runs for complex queries.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, List, Optional

from ..prompt.builder import _render_term
from ..prompt.render import render_schema
from .backtranslate import BackTranslation, BackTranslator
from .confidence import Confidence, ConfidenceConfig, score_confidence
from .coverage import CoverageChecker, CoverageReport
from .multiquery import MultiQueryCheck, MultiQueryChecker
from .sanity import ResultSanityChecker, SanityReport

logger = logging.getLogger(__name__)


@dataclass
class VerificationReport:
    confidence: Confidence
    sanity: Optional[SanityReport] = None
    coverage: Optional[CoverageReport] = None
    backtranslation: Optional[BackTranslation] = None
    multiquery: Optional[MultiQueryCheck] = None
    notes: List[str] = field(default_factory=list)  # checks that couldn't run, and why

    def to_dict(self, include_rows: bool = True) -> dict:
        data = asdict(self)
        if data.get("multiquery") and not include_rows:
            data["multiquery"]["alternative_rows"] = None
        return data


class Verifier:
    def __init__(
        self,
        session: Any,  # ClarificationSession: builder, generator, executor, preferences
        llm: Any = None,  # for back-translation, e.g. ClaudeJSON(effort="medium")
        multi_query: str = "auto",
        config: ConfidenceConfig = ConfidenceConfig(),
    ):
        self.session = session
        builder = session.builder
        self.schema, self.knowledge = builder.schema, builder.knowledge
        self.sanity = ResultSanityChecker(self.schema, session.executor)
        self.coverage = CoverageChecker(self.schema, self.knowledge)
        self.backtranslator = BackTranslator(llm) if llm is not None else None
        self.multiquery = MultiQueryChecker(builder, session.generator, session.executor,
                                            mode=multi_query)
        self.config = config

    def verify(
        self,
        question: str,
        response: Any,  # QueryResponse with status ok
        tables: Optional[Iterable[str]] = None,
        intent_note: Optional[str] = None,  # e.g. the option chosen in a clarification
    ) -> VerificationReport:
        notes: List[str] = []
        asked = f"{question} (meaning: {intent_note})" if intent_note else question
        generated = response.generated_sql or response.sql

        sanity = self.sanity.check(response.sql, response.columns, response.rows,
                                   response.truncated)
        coverage = self.coverage.check(question, generated, self.session.preferences)

        backtranslation = None
        if self.backtranslator is not None:
            try:
                backtranslation = self.backtranslator.check(
                    asked, generated, self._schema_text(response.tables),
                    today=self.session.today, glossary_text=self._glossary_text(question),
                )
            except Exception as e:
                notes.append(f"back-translation unavailable: {e}")
                logger.warning("back-translation failed: %s", e)
        else:
            notes.append("back-translation off (no model configured)")

        multiquery = self.multiquery.check(
            asked, response.sql, response.columns, response.rows, response.truncated,
            tables=tables, preferences=self.session.preferences, today=self.session.today,
        )
        if not multiquery.ran:
            notes.append(f"second query: {multiquery.reason}")

        confidence = score_confidence(
            attempts=response.attempts or 1, backtranslation=backtranslation, sanity=sanity,
            multiquery=multiquery, coverage=coverage,
            model_confidence=response.model_confidence, config=self.config,
        )
        return VerificationReport(confidence, sanity, coverage, backtranslation, multiquery, notes)

    def _schema_text(self, tables: Iterable[str]) -> str:
        names = list(tables) or [t.name for t in self.schema.tables]
        return render_schema(self.schema, self.knowledge, names)

    def _glossary_text(self, question: str) -> str:
        matches = self.knowledge.matching_terms(question, self.session.preferences)
        return "\n".join(_render_term(m) for m in matches)
