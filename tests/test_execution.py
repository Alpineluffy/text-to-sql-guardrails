"""The execution layer, end to end: real prompt building, validation,
guardrails, sandbox and database; only Claude is replaced by a script."""
import json
from pathlib import Path

import pytest
import sqlalchemy as sa

from text2sql.execution import ClarificationResponse, QueryLog, QueryResponse, create_service
from text2sql.generation import Answer, GenerationError, SQLResponse
from text2sql.sample_data import build_sample_db

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"

GROSS = "SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders WHERE status <> 'CANCELLED'"
NET = ("SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders "
       "WHERE status NOT IN ('CANCELLED', 'RETURNED')")


def reply(sql=None, interpretations=None):
    data = {"sql": sql, "explanation": "test", "tables_used": [], "columns_used": [],
            "assumptions": [], "confidence": 0.8,
            "needs_clarification": bool(interpretations), "clarification": None}
    if interpretations:
        data["clarification"] = {"question": "Which revenue?", "interpretations": [
            {"label": l, "description": f"{l} desc", "sql": q} for l, q in interpretations]}
    return SQLResponse.from_json(data, model="claude-opus-5-5")


class Scripted:
    """`script(user_prompt)` returns the reply, or raises."""

    def __init__(self, script):
        self.script = script

    def generate(self, prompt):
        return self.script(prompt.user)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture
def make(engine, tmp_path):
    def build(script):
        return create_service(
            engine, knowledge_dir=KNOWLEDGE_DIR, use_filter=False, generator=Scripted(script),
            log_dir=tmp_path / "logs", schema_cache_dir=tmp_path / "schema", verify=False,
        )
    return build


def log_of(service):
    return service.log.records()


# --- successful answers -----------------------------------------------------------

def test_answer_is_fully_packaged(make):
    service = make(lambda _: reply(
        "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY n DESC"))
    r = service.ask("Orders by status?")
    assert isinstance(r, QueryResponse) and r.ok
    assert r.columns == ["status", "n"] and r.row_count == 5 and not r.truncated
    assert r.rows[0][0] == "DELIVERED"
    assert r.sql.endswith("LIMIT 1000") and r.rewrites == ["added LIMIT 1000"]
    assert r.generated_sql.endswith("ORDER BY n DESC")
    assert r.execution_ms >= 0 and r.total_ms >= r.execution_ms
    assert r.plan and r.estimated_scan_rows == 1000
    assert r.tables == ["orders"] and r.attempts == 1 and r.model == "claude-opus-5-5"

    df = r.dataframe()
    assert list(df.columns) == ["status", "n"] and len(df) == 5 and df["n"].sum() == 1000


def test_hitting_the_row_limit_marks_the_result_truncated(make):
    r = make(lambda _: reply("SELECT order_id FROM order_items")).ask("All line items")
    assert r.row_count == 1000 and r.row_limit == 1000
    assert r.truncated  # 2,565 rows exist; LIMIT 1000 was added by the guardrails


def test_small_results_are_not_truncated(make):
    r = make(lambda _: reply("SELECT id FROM orders WHERE id <= 5")).ask("First orders")
    assert r.row_count == 5 and not r.truncated


def test_response_serialises_to_json(make):
    r = make(lambda _: reply(
        "SELECT order_date, total_amount FROM orders ORDER BY id LIMIT 3")).ask("Recent")
    data = json.loads(json.dumps(r.to_dict()))
    assert data["row_count"] == 3 and len(data["rows"]) == 3
    assert data["rows"][0][0].count("-") == 2  # dates become ISO strings


# --- every way a question can end ---------------------------------------------------

@pytest.mark.parametrize("sql,status", [
    ("DELETE FROM orders", "blocked"),
    ("SELECT nope FROM orders", "invalid_sql"),
    ("SELECT email FROM customers", "blocked"),  # still reads PII after 2 repairs
    (None, "cannot_answer"),
])
def test_statuses(make, sql, status):
    r = make(lambda _: reply(sql)).ask("question")
    assert r.status == status and not r.ok
    assert (r.error is None) == (status == "cannot_answer")
    assert r.rows == []


@pytest.mark.parametrize("kind,status", [
    ("timeout", "timeout"), ("denied", "denied"), ("read_only", "denied"),
    ("query_error", "query_error"),
])
def test_sandbox_error_kinds_map_to_statuses(kind, status):
    answer = Answer("q", "SELECT 1", "", [], None, error="boom", error_kind=kind)
    assert QueryResponse.from_answer("id", answer).status == status


def test_generation_failure_is_a_response_not_a_crash(make):
    def broken(_):
        raise GenerationError("the model declined this request (category: cyber)")

    service = make(broken)
    r = service.ask("question")
    assert r.status == "generation_failed" and "declined" in r.error
    assert log_of(service)[0]["event"] == "failed"


# --- clarification ------------------------------------------------------------------

def test_clarification_round_trip_shares_one_request_id(make):
    service = make(lambda _: reply(interpretations=[("gross revenue", GROSS),
                                                    ("net revenue", NET)]))
    c = service.ask("What was our revenue?")
    assert isinstance(c, ClarificationResponse) and c.status == "needs_clarification"
    assert [o.label for o in c.options] == ["gross revenue", "net revenue"]
    assert c.options[0].columns == ["revenue"] and c.options[0].preview_rows

    r = service.choose(c.request_id, 1)
    assert r.ok and r.request_id == c.request_id and r.sql == NET
    events = [(rec["event"], rec.get("choice")) for rec in log_of(service)]
    assert events == [("clarification_requested", None), ("clarified", "net revenue")]

    with pytest.raises(KeyError):
        service.choose(c.request_id, 0)  # already answered


# --- the audit log -------------------------------------------------------------------

def test_every_question_is_logged_without_result_data(make):
    service = make(lambda _: reply("SELECT status, COUNT(*) AS n FROM orders GROUP BY status"))
    first = service.ask("Orders by status?")
    service.ask("Same again?")
    records = log_of(service)
    assert [r["event"] for r in records] == ["answered", "answered"]
    rec = records[0]
    assert rec["request_id"] == first.request_id and rec["question"] == "Orders by status?"
    assert rec["status"] == "ok" and rec["row_count"] == 5 and rec["columns"] == ["status", "n"]
    assert rec["rows"] is None  # results can hold personal data; never logged
    assert rec["sql"].endswith("LIMIT 1000") and rec["total_ms"] is not None
    assert "DELIVERED" not in json.dumps(rec)


def test_log_filters_by_request(tmp_path):
    log = QueryLog(tmp_path / "q.jsonl")
    for rid in ("a", "b", "a"):
        log.write("answered", QueryResponse(request_id=rid, question="q", status="ok"))
    assert len(log.records("a")) == 2 and len(log.records()) == 3
