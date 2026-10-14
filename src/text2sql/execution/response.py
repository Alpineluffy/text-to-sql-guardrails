"""Phase 2 Step 4: what a caller gets back for a question.

QueryResponse packages everything about one answered question: the rows (and
a pandas DataFrame of them), how long it took, how many rows came back and
whether more exist, the query plan the guardrails already fetched, and what
was rewritten or warned about on the way. `status` says in one word how it
ended, so callers can branch without parsing error messages.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ..generation.clarify import Answer, ClarificationRequest
from ..schema.profiler import to_jsonable

# How a question can end.
OK = "ok"
NEEDS_CLARIFICATION = "needs_clarification"
CANNOT_ANSWER = "cannot_answer"  # the model said the schema can't answer it
GENERATION_FAILED = "generation_failed"  # API error, refusal, malformed reply
STATUSES_FROM_ERROR = {  # Answer.error_kind -> status
    "invalid": "invalid_sql",
    "blocked": "blocked",
    "timeout": "timeout",
    "denied": "denied",
    "read_only": "denied",
    "query_error": "query_error",
}


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class QueryResponse:
    request_id: str
    question: str
    status: str
    sql: Optional[str] = None  # what actually ran (after guardrail rewrites)
    generated_sql: Optional[str] = None  # what the model wrote
    explanation: str = ""
    assumptions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    # Phase 3: combined confidence from every check, with reasons (problems first).
    confidence: Optional[float] = None
    confidence_band: Optional[str] = None  # high | medium | low
    confidence_reasons: List[str] = field(default_factory=list)
    model_confidence: Optional[float] = None  # the model's own estimate (one weak signal)
    columns: List[str] = field(default_factory=list)
    rows: List[tuple] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False  # more rows exist than were returned
    row_limit: Optional[int] = None  # the cap that applied, if any
    execution_ms: Optional[float] = None  # time in the database
    total_ms: Optional[float] = None  # question in to response out
    plan: List[str] = field(default_factory=list)
    estimated_scan_rows: Optional[int] = None
    rewrites: List[str] = field(default_factory=list)
    attempts: int = 0  # model generations, counting repairs
    model: Optional[str] = None
    tables: List[str] = field(default_factory=list)  # tables the SQL read
    error: Optional[str] = None
    source: str = "model"  # "user" when the SQL was written or edited by the user
    verification: Any = None  # VerificationReport (Phase 3)
    created_at: str = field(default_factory=_now)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def dataframe(self):
        """The rows as a pandas DataFrame (pandas is imported only when asked)."""
        import pandas as pd

        return pd.DataFrame.from_records(self.rows, columns=self.columns)

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        verification = self.verification
        self.verification = None  # serialised separately to control row output
        try:
            data = asdict(self)
        finally:
            self.verification = verification
        data["rows"] = ([[to_jsonable(v) for v in row] for row in self.rows]
                        if include_rows else None)
        data["verification"] = (verification.to_dict(include_rows)
                                if verification is not None else None)
        return data

    def apply_verification(self, report: Any) -> None:
        self.verification = report
        self.confidence = report.confidence.score
        self.confidence_band = report.confidence.band
        self.confidence_reasons = list(report.confidence.reasons)

    @classmethod
    def from_answer(
        cls, request_id: str, answer: Answer, max_rows: Optional[int] = None
    ) -> QueryResponse:
        r = cls(request_id=request_id, question=answer.question, status=OK,
                generated_sql=answer.sql, sql=answer.sql, explanation=answer.explanation,
                assumptions=list(answer.assumptions), warnings=list(answer.warnings),
                model_confidence=answer.confidence, error=answer.error)
        generated = answer.response
        if generated is not None:
            r.attempts, r.model = generated.attempts, generated.model
            if generated.validation is not None:
                r.tables = list(generated.validation.tables)

        if answer.error:
            r.status = STATUSES_FROM_ERROR.get(answer.error_kind or "", "query_error")
        elif not answer.sql:
            r.status = CANNOT_ANSWER

        result = answer.result
        if result is not None:
            r.columns, r.rows = list(result.columns), list(result.rows)
            r.row_count, r.truncated = len(result.rows), result.truncated
            r.execution_ms = result.elapsed_ms
            decision = result.guardrails
            if decision is not None:
                r.sql = decision.sql
                r.rewrites = list(decision.rewrites)
                r.row_limit = decision.applied_limit
                r.tables = r.tables or list(decision.tables)
                if decision.scan is not None:
                    r.plan, r.estimated_scan_rows = list(decision.scan.plan), decision.scan.rows
            r.row_limit = r.row_limit or max_rows
        return r


@dataclass
class OptionView:
    label: str
    description: str
    sql: str
    columns: List[str]
    preview_rows: List[tuple]
    error: Optional[str]


@dataclass
class ClarificationResponse:
    request_id: str
    question: str
    prompt: str
    options: List[OptionView]
    reason: str  # "glossary" or "model"
    status: str = NEEDS_CLARIFICATION
    total_ms: Optional[float] = None
    created_at: str = field(default_factory=_now)

    @classmethod
    def from_request(
        cls, request_id: str, request: ClarificationRequest, preview_rows: int = 5
    ) -> ClarificationResponse:
        options = [
            OptionView(
                label=o.label, description=o.description, sql=o.sql,
                columns=list(o.preview.columns) if o.preview else [],
                preview_rows=list(o.preview.rows[:preview_rows]) if o.preview else [],
                error=o.error,
            )
            for o in request.options
        ]
        return cls(request_id, request.question, request.prompt, options, request.reason)

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        for option in data["options"]:
            option["preview_rows"] = ([[to_jsonable(v) for v in row]
                                       for row in option["preview_rows"]]
                                      if include_rows else None)
        return data


def failed(request_id: str, question: str, message: str) -> QueryResponse:
    return QueryResponse(request_id=request_id, question=question,
                         status=GENERATION_FAILED, error=message)
