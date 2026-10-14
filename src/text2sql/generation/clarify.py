"""Step 4: ask the user instead of guessing when a question is ambiguous.

    session = ClarificationSession(builder, generator, executor)
    result = session.ask("What was revenue from India last quarter?")
    if isinstance(result, ClarificationRequest):
        result = session.choose(result, 1)  # user picked "net revenue"
    # Later "revenue" questions in this session use net revenue without asking.

When to ask:
- the glossary defines the term several ways, or the model flags ambiguity;
- AND the interpretations actually give different answers. When an executor
  is available each interpretation is run, and if they all return the same
  rows the question is answered directly with the assumption stated.

If the glossary says a term is ambiguous but the model answered anyway, one
query per definition is generated so the user still gets to choose.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Union

from ..prompt.builder import PromptBuilder
from .executor import QueryResult, ReadOnlyExecutor
from .llm import GenerationError, Interpretation, SQLGenerator, SQLResponse

PREVIEW_ROWS = 5


@dataclass
class Answer:
    question: str
    sql: Optional[str]
    explanation: str
    assumptions: List[str]
    confidence: Optional[float]  # the model's own estimate; None if unknown
    response: Optional[SQLResponse] = None
    result: Optional[QueryResult] = None
    error: Optional[str] = None  # set when the SQL was invalid or running it failed
    # invalid | blocked | timeout | denied | read_only | query_error
    error_kind: Optional[str] = None
    warnings: List[str] = field(default_factory=list)  # e.g. model claims vs actual SQL


@dataclass
class Option:
    label: str
    description: str
    sql: str
    confidence: Optional[float] = None
    preview: Optional[QueryResult] = None
    error: Optional[str] = None  # set when the preview query failed
    error_kind: Optional[str] = None
    term: Optional[str] = None  # glossary term this option resolves, if any


@dataclass
class ClarificationRequest:
    question: str  # the user's original question
    prompt: str  # what to ask the user
    options: List[Option]
    reason: str  # "glossary" or "model"
    ambiguous_terms: List[str] = field(default_factory=list)


Result = Union[Answer, ClarificationRequest]


class ClarificationSession:
    def __init__(
        self,
        builder: PromptBuilder,
        generator: SQLGenerator,
        executor: Optional[ReadOnlyExecutor] = None,
        today: Optional[date] = None,
    ):
        self.builder = builder
        self.generator = generator
        self.executor = executor
        self.today = today
        # glossary term -> definition name the user chose; reused for the session.
        self.preferences: Dict[str, str] = {}

    def ask(self, question: str, tables: Optional[Iterable[str]] = None) -> Result:
        prompt = self.builder.build(
            question, tables=tables, today=self.today, preferences=self.preferences
        )
        response = self.generator.generate(prompt)
        ambiguous = prompt.ambiguous_terms

        if response.needs_clarification and response.interpretations:
            options = [self._option(i, ambiguous, response.confidence)
                       for i in response.interpretations]
            reason = "glossary" if ambiguous else "model"
            ask_text = response.clarification_question
        elif ambiguous:
            # The glossary says this term has several meanings but the model
            # picked one silently. Generate each meaning so the user decides.
            options = self._per_definition_options(question, tables, ambiguous[0])
            reason = "glossary"
            ask_text = None
        else:
            return self._answer(question, response, self._preference_notes(prompt))

        options = [self._preview(o) for o in options]
        if len(options) < 2:
            if options:
                return self._answer_from_option(question, options[0], [])
            return self._answer(question, response, [])

        agreed = self._all_agree(options)
        if agreed:
            note = (f"'{', '.join(o.label for o in options)}' all give the same result here, "
                    f"so the question was answered without asking")
            return self._answer_from_option(question, options[0], [note])

        return ClarificationRequest(
            question=question,
            prompt=ask_text or self._default_prompt(ambiguous, options),
            options=options,
            reason=reason,
            ambiguous_terms=ambiguous,
        )

    def choose(self, request: ClarificationRequest, index: int) -> Answer:
        """Resolve a clarification with the user's pick (0-based) and remember it."""
        option = request.options[index]
        if option.term:
            self.preferences[option.term] = option.label
        return self._answer_from_option(
            request.question, option, [f"You chose: {option.label} ({option.description})"]
        )

    # --- helpers -----------------------------------------------------------

    def _option(
        self, interp: Interpretation, ambiguous: List[str], confidence: float
    ) -> Option:
        """Link a model interpretation to a glossary definition when the labels match."""
        for term_name in ambiguous:
            term = self.builder.knowledge.term(term_name)
            for d in term.definitions:
                if d.name.lower() == interp.label.strip().lower():
                    return Option(d.name, interp.description, interp.sql, confidence=confidence,
                                  error=_invalid(interp.validation), term=term.term)
        return Option(interp.label, interp.description, interp.sql, confidence=confidence,
                      error=_invalid(interp.validation))

    def _per_definition_options(
        self, question: str, tables: Optional[Iterable[str]], term_name: str
    ) -> List[Option]:
        term = self.builder.knowledge.term(term_name)
        options = []
        for d in term.definitions:
            prompt = self.builder.build(
                question, tables=tables, today=self.today,
                preferences={**self.preferences, term.term: d.name},
            )
            response = self.generator.generate(prompt)
            if response.sql:
                options.append(Option(d.name, response.explanation or d.sql, response.sql,
                                      confidence=response.confidence,
                                      error=_invalid(response.validation), term=term.term))
        return options

    def _preview(self, option: Option) -> Option:
        if option.error:
            option.error_kind = option.error_kind or "invalid"
            return option  # never run SQL that failed validation
        if self.executor is None:
            return option
        try:
            option.preview = self.executor.run(option.sql)
        except Exception as e:  # a broken interpretation shouldn't sink the others
            option.error, option.error_kind = str(e), error_kind(e)
        return option

    @staticmethod
    def _all_agree(options: List[Option]) -> bool:
        if any(o.preview is None for o in options):
            return False  # can't compare without results, so ask
        first = options[0].preview.fingerprint()
        return all(o.preview.fingerprint() == first for o in options[1:])

    @staticmethod
    def _default_prompt(terms: List[str], options: List[Option]) -> str:
        labels = " or ".join(f'"{o.label}"' for o in options)
        subject = f'"{terms[0]}"' if terms else "this question"
        return f"Which meaning of {subject} do you want: {labels}?"

    @staticmethod
    def _preference_notes(prompt) -> List[str]:
        return [
            f"Using {m.definitions[0].name} for '{m.term.term}' (your earlier choice)"
            for m in prompt.term_matches if m.from_preference
        ]

    def _answer(self, question: str, response: SQLResponse, notes: List[str]) -> Answer:
        answer = Answer(question, response.sql, response.explanation,
                        notes + list(response.assumptions), response.confidence,
                        response=response)
        decision = getattr(response, "guardrails", None)
        if decision is not None and any(not v.retryable for v in decision.violations):
            from ..guardrails import GuardrailError

            answer.error, answer.error_kind = str(GuardrailError(decision)), "blocked"
            return answer  # never run unsafe SQL
        report = response.validation
        if report is not None:
            answer.warnings = list(report.warnings)
            if not report.ok:
                answer.error = (f"SQL still invalid after {response.attempts} attempt(s): "
                                + "; ".join(report.errors))
                answer.error_kind = "invalid"
                return answer  # never run SQL that failed validation
        if response.sql and self.executor is not None:
            try:
                answer.result = self.executor.run(response.sql)
            except Exception as e:
                answer.error, answer.error_kind = str(e), error_kind(e)
        return answer

    def _answer_from_option(self, question: str, option: Option, notes: List[str]) -> Answer:
        return Answer(question, option.sql, option.description, notes,
                      confidence=option.confidence, result=option.preview, error=option.error,
                      error_kind=option.error_kind)


def error_kind(error: Exception) -> str:
    """Short machine-readable reason a query didn't produce results."""
    from ..guardrails import GuardrailError
    from ..sandbox import SandboxError, UnsafeQueryError

    if isinstance(error, (GuardrailError, UnsafeQueryError)):
        return "blocked"
    if isinstance(error, SandboxError):
        return error.kind
    return "query_error"


def _invalid(report) -> Optional[str]:
    if report is None or report.ok:
        return None
    return "invalid SQL: " + "; ".join(report.errors)


__all__ = [
    "Answer", "ClarificationRequest", "ClarificationSession", "GenerationError", "Option",
]
