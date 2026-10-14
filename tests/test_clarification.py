import json
import sqlite3
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from text2sql.generation import (
    Answer, ClarificationRequest, ClarificationSession, GenerationError, ReadOnlyExecutor,
    SQLGenerator, UnsafeQueryError, check_read_only,
)
from text2sql.prompt import Knowledge, PromptBuilder, RESPONSE_SCHEMA
from text2sql.sample_data import build_sample_db
from text2sql.schema import SchemaExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"

GROSS = "SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders WHERE status <> 'CANCELLED'"
NET = ("SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders "
       "WHERE status NOT IN ('CANCELLED', 'RETURNED')")


def reply(sql=None, interpretations=None, confidence=0.9, assumptions=()):
    data = {
        "sql": sql,
        "explanation": "test explanation",
        "tables_used": ["orders"],
        "columns_used": ["orders.total_amount"],
        "assumptions": list(assumptions),
        "confidence": confidence,
        "needs_clarification": bool(interpretations),
        "clarification": None,
    }
    if interpretations:
        data["clarification"] = {
            "question": "Which revenue do you mean?",
            "interpretations": [
                {"label": label, "description": f"{label} desc", "sql": q}
                for label, q in interpretations
            ],
        }
    return data


class FakeClient:
    """Stands in for anthropic.Anthropic; `handler(user_prompt)` returns the reply dict."""

    def __init__(self, handler, stop_reason="end_turn", raw_text=None):
        self.handler = handler
        self.stop_reason = stop_reason
        self.raw_text = raw_text
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **params):
        self.calls.append(params)
        user = params["messages"][0]["content"]
        text = self.raw_text if self.raw_text is not None else json.dumps(self.handler(user))
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=SimpleNamespace(category="cyber") if self.stop_reason == "refusal" else None,
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text=text)],
            model="claude-opus-5-5",
        )


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    # A directory with a space, like the real project path.
    path = build_sample_db(tmp_path_factory.mktemp("db") / "with space" / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture(scope="module")
def builder(engine):
    schema = SchemaExtractor(engine).extract()
    return PromptBuilder(schema, Knowledge.load(KNOWLEDGE_DIR))


def make_session(builder, engine, handler, **client_kwargs):
    client = FakeClient(handler, **client_kwargs)
    session = ClarificationSession(
        builder, SQLGenerator(client=client), ReadOnlyExecutor(engine), today=date(2026, 10, 6)
    )
    return session, client


# --- the clarification flow -------------------------------------------------------

def test_unambiguous_question_is_answered(builder, engine):
    session, client = make_session(
        builder, engine,
        lambda _: reply("SELECT COUNT(*) AS n FROM orders WHERE channel = 'mobile'"),
    )
    answer = session.ask("How many orders came through mobile?")
    assert isinstance(answer, Answer)
    assert answer.result.columns == ["n"] and answer.result.rows[0][0] > 0
    assert answer.confidence == 0.9
    assert len(client.calls) == 1


def test_ambiguous_term_returns_options_with_previews(builder, engine):
    session, _ = make_session(
        builder, engine,
        lambda _: reply(interpretations=[("gross revenue", GROSS), ("net revenue", NET)]),
    )
    request = session.ask("What was our revenue?")
    assert isinstance(request, ClarificationRequest)
    assert request.reason == "glossary" and request.ambiguous_terms == ["revenue"]
    assert request.prompt == "Which revenue do you mean?"
    gross, net = request.options
    assert (gross.label, gross.term) == ("gross revenue", "revenue")
    assert gross.preview.rows[0][0] > net.preview.rows[0][0]  # returns make net smaller


def test_choice_is_remembered_for_the_session(builder, engine):
    def handler(user):
        if "AMBIGUOUS" in user:
            return reply(interpretations=[("gross revenue", GROSS), ("net revenue", NET)])
        return reply(NET + " AND channel = 'web'")

    session, client = make_session(builder, engine, handler)
    answer = session.choose(session.ask("What was our revenue?"), 1)
    assert answer.sql == NET
    assert answer.assumptions == ["You chose: net revenue (net revenue desc)"]
    assert session.preferences == {"revenue": "net revenue"}

    # Same term again, via a synonym: no question this time.
    follow_up = session.ask("And web sales?")
    assert isinstance(follow_up, Answer)
    last_prompt = client.calls[-1]["messages"][0]["content"]
    assert "net revenue:" in last_prompt and "gross revenue:" not in last_prompt
    assert "chose this definition earlier" in last_prompt
    assert follow_up.assumptions[0] == "Using net revenue for 'revenue' (your earlier choice)"


def test_question_naming_a_definition_overrides_the_preference(builder, engine):
    session, client = make_session(builder, engine, lambda _: reply(GROSS))
    session.preferences["revenue"] = "net revenue"
    session.ask("What was gross revenue?")
    prompt = client.calls[-1]["messages"][0]["content"]
    assert "gross revenue:" in prompt and "net revenue:" not in prompt


def test_identical_results_answer_without_asking(builder, engine):
    same_gross = GROSS + " AND status = 'DELIVERED'"
    same_net = NET + " AND status = 'DELIVERED'"
    session, _ = make_session(
        builder, engine,
        lambda _: reply(interpretations=[("gross revenue", same_gross), ("net revenue", same_net)]),
    )
    answer = session.ask("What was revenue from delivered orders?")
    assert isinstance(answer, Answer)
    assert answer.sql == same_gross
    assert "all give the same result" in answer.assumptions[0]


def test_silent_guess_on_ambiguous_term_still_asks(builder, engine):
    def handler(user):
        if "AMBIGUOUS" in user:
            return reply(GROSS)  # the model guessed instead of asking
        return reply(NET if "net revenue:" in user else GROSS)

    session, client = make_session(builder, engine, handler)
    request = session.ask("What was our revenue?")
    assert isinstance(request, ClarificationRequest)
    assert [o.label for o in request.options] == ["gross revenue", "net revenue"]
    assert [o.sql for o in request.options] == [GROSS, NET]
    assert len(client.calls) == 3  # original + one per definition


def test_model_flagged_ambiguity_outside_glossary(builder, engine):
    by_revenue = "SELECT customer_id FROM orders GROUP BY customer_id ORDER BY SUM(total_amount) DESC LIMIT 5"
    by_orders = "SELECT customer_id FROM orders GROUP BY customer_id ORDER BY COUNT(*) DESC LIMIT 5"
    session, _ = make_session(
        builder, engine,
        lambda _: reply(interpretations=[("by spend", by_revenue), ("by order count", by_orders)]),
    )
    request = session.ask("Who are our best customers?")
    assert request.reason == "model"
    assert all(o.term is None for o in request.options)
    session.choose(request, 0)
    assert session.preferences == {}  # nothing in the glossary to remember


def test_broken_interpretation_is_reported_not_fatal(builder, engine):
    session, _ = make_session(
        builder, engine,
        lambda _: reply(interpretations=[("gross revenue", GROSS),
                                         ("net revenue", "SELECT nope FROM orders")]),
    )
    request = session.ask("What was our revenue?")
    assert isinstance(request, ClarificationRequest)
    assert request.options[0].preview is not None
    assert "no such column" in request.options[1].error


def test_failed_query_is_reported_on_the_answer(builder, engine):
    session, _ = make_session(builder, engine, lambda _: reply("SELECT nope FROM orders"))
    answer = session.ask("How many orders?")
    assert answer.result is None and "no such column" in answer.error


# --- the API call -------------------------------------------------------------

def test_request_uses_structured_output_caching_and_fallbacks(builder, engine):
    session, client = make_session(builder, engine, lambda _: reply("SELECT 1"))
    session.ask("How many orders?")
    params = client.calls[0]
    assert params["model"] == "claude-opus-5-5"
    assert params["output_config"]["effort"] == "high"
    assert params["output_config"]["format"] == {"type": "json_schema", "schema": RESPONSE_SCHEMA}
    assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert params["betas"] == ["server-side-fallback-2026-07-01"]
    assert params["fallbacks"] == "default"


@pytest.mark.parametrize("stop_reason,raw,message", [
    ("refusal", None, "declined"),
    ("max_tokens", None, "max_tokens"),
    ("end_turn", "not json", "not valid JSON"),
])
def test_bad_responses_raise(builder, engine, stop_reason, raw, message):
    session, _ = make_session(builder, engine, lambda _: reply("SELECT 1"),
                              stop_reason=stop_reason, raw_text=raw)
    with pytest.raises(GenerationError, match=message):
        session.ask("How many orders?")


def test_confidence_is_clamped(builder, engine):
    session, _ = make_session(builder, engine, lambda _: reply("SELECT 1", confidence=7))
    assert session.ask("How many orders?").confidence == 1.0


def test_response_schema_meets_structured_output_rules():
    def walk(node):
        if isinstance(node, dict):
            assert "minimum" not in node and "maximum" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(RESPONSE_SCHEMA)


# --- interim read-only executor ---------------------------------------------------

@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "DROP TABLE orders",
    "SELECT 1; DROP TABLE orders",
    "PRAGMA writable_schema = 1",
    "-- comment\nUPDATE orders SET status = 'X'",
])
def test_non_select_rejected(sql):
    with pytest.raises(UnsafeQueryError):
        check_read_only(sql)


def test_semicolon_inside_string_is_allowed():
    assert check_read_only("SELECT * FROM t WHERE x LIKE '%;%';") == "SELECT * FROM t WHERE x LIKE '%;%'"


def test_database_refuses_writes_hidden_in_a_cte(engine):
    from text2sql.sandbox import SandboxError

    executor = ReadOnlyExecutor(engine)
    with pytest.raises(SandboxError) as info:
        executor.run("WITH x AS (SELECT 1) DELETE FROM orders")
    assert info.value.is_security_event
    assert executor.run("SELECT COUNT(*) FROM orders").rows == [(1000,)]


def test_row_cap_sets_truncated(engine):
    result = ReadOnlyExecutor(engine, max_rows=10).run("SELECT id FROM orders")
    assert len(result.rows) == 10 and result.truncated
