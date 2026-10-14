"""The whole pipeline behind one call: question in, packaged response out.

    service = create_service(sa.create_engine("sqlite:///data/ecommerce.db"))
    response = service.ask("What was net revenue by country last quarter?")
    if response.status == "needs_clarification":
        response = service.choose(response.request_id, 0)
    df = response.dataframe()

Layers, in order: schema filtering (Step 3) -> prompt (Step 2) -> Claude with
structured output -> validation and repair (Phase 2 Step 1) -> clarification
(Step 4) -> guardrails (Phase 2 Step 2) -> sandbox (Phase 2 Step 3) -> this
packaging and logging (Phase 2 Step 4).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional, Union

from sqlalchemy.engine import Engine

from ..generation.clarify import ClarificationRequest, ClarificationSession
from ..generation.llm import GenerationError
from .log import QueryLog
from .response import (
    ClarificationResponse, QueryResponse, failed, new_request_id,
)

logger = logging.getLogger(__name__)

Response = Union[QueryResponse, ClarificationResponse]


class Text2SQLService:
    """One user session: its clarification choices, pending questions and history."""

    def __init__(
        self,
        session: ClarificationSession,
        schema_filter: Any = None,
        log: Optional[QueryLog] = None,
        max_rows: Optional[int] = None,
        verifier: Any = None,
        session_id: Optional[str] = None,
        history_size: int = 50,
        sql_verifier: Any = None,
        history_store: Any = None,
    ):
        self.session = session
        self.schema_filter = schema_filter
        self.log = log
        self.max_rows = max_rows
        self.verifier = verifier
        self.sql_verifier = sql_verifier  # checks user-written SQL (no second query)
        self.session_id = session_id or new_request_id()
        self.history: Deque[Any] = deque(maxlen=history_size)
        self.history_store = history_store  # HistoryStore: survives restarts
        self._pending: Dict[str, ClarificationRequest] = {}
        self._tables: Dict[str, Optional[List[str]]] = {}

    def ask(self, question: str) -> Response:
        request_id = new_request_id()
        start = time.perf_counter()
        tables: Optional[List[str]] = None
        if self.schema_filter is not None:
            tables = self.schema_filter.select(question).tables

        try:
            result = self.session.ask(question, tables=tables)
        except Exception as e:
            if not _is_generation_failure(e):
                raise
            response: Response = failed(request_id, question, f"{type(e).__name__}: {e}")
            return self._finish("failed", response, start, offered_tables=tables)

        if isinstance(result, ClarificationRequest):
            self._pending[request_id] = result
            self._tables[request_id] = tables
            response = ClarificationResponse.from_request(request_id, result)
            return self._finish("clarification_requested", response, start, offered_tables=tables)

        response = QueryResponse.from_answer(request_id, result, self.max_rows)
        self._verify(question, response, tables)
        return self._finish("answered", response, start, offered_tables=tables)

    def choose(self, request_id: str, option: int) -> QueryResponse:
        """Answer a pending clarification with the chosen option (0-based)."""
        request = self._pending.get(request_id)
        if request is None:
            raise KeyError(f"no clarification pending for request {request_id}")
        if not 0 <= option < len(request.options):
            raise IndexError(f"option must be between 0 and {len(request.options) - 1}")
        del self._pending[request_id]
        start = time.perf_counter()
        answer = self.session.choose(request, option)
        response = QueryResponse.from_answer(request_id, answer, self.max_rows)
        chosen = request.options[option]
        self._verify(request.question, response, self._tables.pop(request_id, None),
                     intent_note=f"{chosen.label}: {chosen.description}")
        return self._finish("clarified", response, start, choice=chosen.label)

    def run_sql(self, sql: str, question: Optional[str] = None) -> QueryResponse:
        """Run SQL a user wrote or edited, through the same safety layers.

        Validation, guardrails and the sandbox apply exactly as for generated
        SQL; the result checks and (with a question) back-translation still
        run, so an edit that breaks the answer is flagged too.
        """
        from ..generation.clarify import Answer, error_kind

        request_id = new_request_id()
        start = time.perf_counter()
        asked = question or "(SQL written by the user)"
        validator = getattr(self.session.generator, "validator", None)
        guardrails = getattr(self.session.executor, "guardrails", None)
        report = validator.validate(sql) if validator is not None else None
        answer = Answer(asked, sql, "SQL written or edited by the user", [], None)
        unsafe = _hard_violation(guardrails, sql)
        if unsafe is not None:  # dangerous SQL is "blocked", whatever else is wrong with it
            answer.error, answer.error_kind = unsafe, "blocked"
        elif report is not None and not report.ok:
            answer.error, answer.error_kind = "; ".join(report.errors), "invalid"
        else:
            try:
                answer.result = self.session.executor.run(sql)
            except Exception as e:
                answer.error, answer.error_kind = str(e), error_kind(e)
        response = QueryResponse.from_answer(request_id, answer, self.max_rows)
        response.source = "user"
        if report is not None:
            response.tables = response.tables or list(report.tables)
        verifier = self.sql_verifier or self.verifier
        if verifier is not None and response.ok:
            try:
                response.apply_verification(verifier.verify(question or "", response))
            except Exception as e:
                logger.error("verification failed: %s", e)
        return self._finish("user_sql", response, start)

    def _verify(self, question: str, response: QueryResponse,
                tables: Optional[List[str]], intent_note: Optional[str] = None) -> None:
        if self.verifier is None or not response.ok:
            return
        try:
            response.apply_verification(
                self.verifier.verify(question, response, tables, intent_note=intent_note))
        except Exception as e:  # verification must never cost the user their answer
            logger.error("verification failed: %s", e)
            response.confidence_reasons = [f"Verification failed to run: {e}"]

    def _finish(self, event: str, response: Any, start: float, **extra: Any) -> Any:
        response.total_ms = round((time.perf_counter() - start) * 1000, 2)
        self.history.append(response)
        if self.history_store is not None:
            try:
                self.history_store.append(self.session_id, response)
            except (OSError, ValueError) as e:
                logger.error("could not save history: %s", e)
        if self.log is not None:
            try:
                self.log.write(event, response, session_id=self.session_id, **extra)
            except OSError as e:  # never lose an answer because the log disk is full
                logger.error("could not write query log: %s", e)
        return response


def _hard_violation(guardrails: Any, sql: str) -> Optional[str]:
    """The guardrail refusal message if `sql` is unsafe whatever the intent, else None."""
    if guardrails is None:
        return None
    from ..guardrails import GuardrailError

    decision = guardrails.check(sql)
    if any(not v.retryable for v in decision.violations):
        return str(GuardrailError(decision))
    return None


def _is_generation_failure(error: Exception) -> bool:
    if isinstance(error, GenerationError):
        return True
    try:
        import anthropic
    except ImportError:
        return False
    return isinstance(error, (anthropic.APIStatusError, anthropic.APIConnectionError))


class Text2SQLApp:
    """Everything expensive, built once: schema, knowledge, embedding model,
    guardrails, sandbox, Claude clients. Sessions on top are cheap."""

    def __init__(
        self,
        engine: Engine,
        knowledge_dir: Path = Path("knowledge/ecommerce"),
        guardrails_path: Path = Path("config/guardrails.yaml"),
        use_filter: bool = True,
        generator: Any = None,
        model: str = "claude-opus-5-5",
        effort: str = "high",
        log_dir: Path = Path("cache/logs"),
        schema_cache_dir: Path = Path("cache/schema"),
        verify: bool = True,
        verifier_llm: Any = None,
        multi_query: str = "auto",
        today: Any = None,
        max_sessions: int = 1_000,
        back_translate: bool = True,
        history_dir: Optional[Path] = None,
    ):
        """`generator` and `verifier_llm` override Claude (tests, demo mode).
        `verify` turns on the Phase 3 checks; back-translation uses the same
        model at medium effort."""
        from ..generation import SQLGenerator, SQLValidator, ValidatingGenerator
        from ..guardrails import AuditLog, GuardedExecutor, GuardrailConfig, Guardrails
        from ..sandbox import SandboxedExecutor
        from ..schema import load_or_extract

        self.engine = engine
        self.knowledge_dir = Path(knowledge_dir)
        self.today = today
        self.verify = verify
        self.multi_query = multi_query
        self.max_sessions = max_sessions

        name = Path(engine.url.database or "database").stem
        self.schema, _ = load_or_extract(engine, Path(schema_cache_dir) / f"{name}.json")
        self.config = (GuardrailConfig.from_yaml(guardrails_path)
                       if Path(guardrails_path).exists() else GuardrailConfig())
        audit = AuditLog(Path(log_dir) / "guardrails.jsonl")
        sandbox = SandboxedExecutor.for_schema(engine, self.schema, self.config, audit=audit)
        self.guardrails = Guardrails(self.schema, self.config, explain=sandbox.explain,
                                     audit=audit)
        self.executor = GuardedExecutor(sandbox, self.guardrails)
        self.generator = ValidatingGenerator(
            generator or SQLGenerator(model=model, effort=effort),
            SQLValidator(self.schema), guardrails=self.guardrails,
        )
        self.verifier_llm = None
        if verify and back_translate:
            from ..generation.llm import ClaudeJSON

            self.verifier_llm = verifier_llm or ClaudeJSON(model=model, effort="medium",
                                                           max_tokens=8_000)
        self.log = QueryLog(Path(log_dir) / "queries.jsonl")
        from .history import HistoryStore

        self.history_store = HistoryStore(history_dir) if history_dir else None
        self._embedder = None
        if use_filter:
            from ..retrieval import default_embedder

            self._embedder = default_embedder()
        self.reload_knowledge()
        self._sessions: "OrderedDict[str, Text2SQLService]" = OrderedDict()

    def reload_knowledge(self) -> None:
        """Re-read descriptions, glossary and examples (e.g. after promoting feedback).

        Live sessions are rebuilt on the new knowledge but keep their history,
        clarification choices and pending questions."""
        from ..prompt import Knowledge, PromptBuilder

        knowledge = Knowledge.load(self.knowledge_dir, dialect=self.schema.dialect)
        problems = knowledge.validate(self.schema)
        if problems:
            raise ValueError("knowledge files don't match the schema:\n  " + "\n  ".join(problems))
        self.knowledge = knowledge
        self.schema_filter = selector = None
        if self._embedder is not None:
            from ..retrieval import EmbeddingExampleSelector, SchemaFilter

            self.schema_filter = SchemaFilter(self.schema, knowledge, self._embedder)
            selector = EmbeddingExampleSelector(knowledge.examples, self._embedder)
        self.builder = PromptBuilder(self.schema, knowledge, selector=selector)
        for session_id, old in list(getattr(self, "_sessions", {}).items()):
            fresh = self.new_service(session_id)
            fresh.history = old.history
            fresh.session.preferences = old.session.preferences
            fresh._pending, fresh._tables = old._pending, old._tables
            self._sessions[session_id] = fresh

    def new_service(self, session_id: Optional[str] = None) -> Text2SQLService:
        session = ClarificationSession(self.builder, self.generator, self.executor,
                                       today=self.today)
        verifier = sql_verifier = None
        if self.verify:
            from ..verification import Verifier

            verifier = Verifier(session, llm=self.verifier_llm, multi_query=self.multi_query)
            sql_verifier = Verifier(session, llm=self.verifier_llm, multi_query="never")
        return Text2SQLService(session, self.schema_filter, self.log,
                               max_rows=self.config.max_rows, verifier=verifier,
                               session_id=session_id, sql_verifier=sql_verifier,
                               history_store=self.history_store)

    def session(self, session_id: str) -> Text2SQLService:
        """The service for a session, created on first use (oldest evicted past the cap)."""
        service = self._sessions.get(session_id)
        if service is None:
            service = self.new_service(session_id)
            self._sessions[session_id] = service
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        else:
            self._sessions.move_to_end(session_id)
        return service


def create_service(engine: Engine, **kwargs: Any) -> Text2SQLService:
    """One ready-to-use session over every layer (see Text2SQLApp for options)."""
    return Text2SQLApp(engine, **kwargs).new_service()
