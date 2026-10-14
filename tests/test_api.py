"""Phase 4: the HTTP API and the feedback flywheel."""
import shutil
from pathlib import Path

import pytest
import sqlalchemy as sa
import yaml
from fastapi.testclient import TestClient

from text2sql.api.app import create_app
from text2sql.api.demo import ReplayGenerator
from text2sql.execution import Text2SQLApp
from text2sql.feedback import FeedbackStore, PromotionError, promote, reject
from text2sql.generation import SQLResponse
from text2sql.sample_data import build_sample_db

ROOT = Path(__file__).resolve().parents[1]

GROSS = "SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders WHERE status <> 'CANCELLED'"
NET = ("SELECT ROUND(SUM(total_amount), 2) AS revenue FROM orders "
       "WHERE status NOT IN ('CANCELLED', 'RETURNED')")


def reply(sql=None, interpretations=None):
    data = {"sql": sql, "explanation": "test", "tables_used": [], "columns_used": [],
            "assumptions": [], "confidence": 0.8,
            "needs_clarification": bool(interpretations), "clarification": None}
    if interpretations:
        data["clarification"] = {"question": "Which revenue?", "interpretations": [
            {"label": l, "description": l, "sql": q} for l, q in interpretations]}
    return SQLResponse.from_json(data)


class Scripted:
    def generate(self, prompt):
        question = prompt.user.split("## Question\n")[1].split("\n")[0].lower()
        if "revenue" in question and "AMBIGUOUS" in prompt.user:
            return reply(interpretations=[("gross revenue", GROSS), ("net revenue", NET)])
        if "cancelled" in question:
            return reply("SELECT COUNT(*) AS n FROM orders WHERE status = 'CANCELLED'")
        return reply("SELECT status, COUNT(*) AS n FROM orders GROUP BY status")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture
def knowledge_dir(tmp_path):
    # A copy, so promotion tests never touch the real knowledge files.
    target = tmp_path / "knowledge"
    shutil.copytree(ROOT / "knowledge" / "ecommerce", target)
    return target


@pytest.fixture
def setup(engine, knowledge_dir, tmp_path):
    text2sql = Text2SQLApp(engine, knowledge_dir=knowledge_dir, use_filter=False,
                           generator=Scripted(), back_translate=False, multi_query="never",
                           log_dir=tmp_path / "logs", schema_cache_dir=tmp_path / "schema")
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    return TestClient(create_app(text2sql, store)), text2sql, store


def ask(client, question, session="s1"):
    r = client.post("/v1/query", json={"question": question, "session_id": session})
    assert r.status_code == 200
    return r.json()


# --- Step 1: endpoints ------------------------------------------------------------

def test_health(setup):
    client, _, _ = setup
    assert client.get("/v1/health").json() == {
        "status": "ok", "demo": False, "dialect": "sqlite", "database": "ecommerce.db",
        "tables": 6}


def test_query_returns_sql_results_confidence_and_warnings(setup):
    client, _, _ = setup
    data = ask(client, "Orders by status?")
    assert data["status"] == "ok" and data["session_id"] == "s1"
    assert data["sql"].endswith("LIMIT 1000")
    assert data["rewrites"] == ["added LIMIT 1000"]  # guardrail warning
    assert data["columns"] == ["status", "n"] and len(data["rows"]) == 5
    assert data["confidence_band"] == "medium"  # no back-translation configured
    assert "Not independently verified" in " ".join(data["confidence_reasons"])
    assert data["verification"]["confidence"]["signals"]["sanity"] == 1.0


def test_a_session_id_is_issued_when_missing(setup):
    client, _, _ = setup
    data = client.post("/v1/query", json={"question": "Orders by status?"}).json()
    assert len(data["session_id"]) == 16


@pytest.mark.parametrize("body,code", [
    ({"question": ""}, 422),
    ({"question": "x" * 1001}, 422),
    ({"question": "ok", "session_id": "bad id!"}, 400),
])
def test_bad_requests(setup, body, code):
    client, _, _ = setup
    assert client.post("/v1/query", json=body).status_code == code


def test_clarification_and_choice(setup):
    client, _, _ = setup
    c = ask(client, "What was revenue?")
    assert c["status"] == "needs_clarification"
    assert [o["label"] for o in c["options"]] == ["gross revenue", "net revenue"]
    r = client.post(f"/v1/query/{c['request_id']}/choose",
                    json={"session_id": "s1", "option": 1}).json()
    assert r["status"] == "ok" and r["sql"] == NET and r["request_id"] == c["request_id"]
    again = client.post(f"/v1/query/{c['request_id']}/choose",
                        json={"session_id": "s1", "option": 0})
    assert again.status_code == 404


def test_choose_rejects_bad_option_and_keeps_question_pending(setup):
    client, _, _ = setup
    c = ask(client, "What was revenue?")
    url = f"/v1/query/{c['request_id']}/choose"
    assert client.post(url, json={"session_id": "s1", "option": 5}).status_code == 400
    assert client.post(url, json={"session_id": "s1", "option": 0}).json()["status"] == "ok"


@pytest.mark.parametrize("sql,status", [
    ("SELECT COUNT(*) AS n FROM orders WHERE status = 'PENDING'", "ok"),
    ("DELETE FROM orders", "blocked"),
    ("SELECT nope FROM orders", "invalid_sql"),
    ("SELECT email FROM customers", "blocked"),
])
def test_user_sql_goes_through_the_same_safety_layers(setup, sql, status):
    client, _, _ = setup
    data = client.post("/v1/sql", json={"sql": sql, "session_id": "s1"}).json()
    assert data["status"] == status and data["source"] == "user"


def test_edited_sql_is_still_checked(setup):
    client, _, _ = setup
    data = client.post("/v1/sql", json={
        "sql": "SELECT COUNT(*) AS n FROM orders WHERE status = 'cancelled'",
        "question": "How many orders were cancelled?", "session_id": "s1"}).json()
    assert data["status"] == "ok" and data["rows"] == [[0]]
    assert data["confidence_band"] == "low"
    assert any("did you mean 'CANCELLED'" in r for r in data["confidence_reasons"])


def test_schema_shows_values_but_never_personal_data(setup):
    client, _, _ = setup
    data = client.get("/v1/schema").json()
    tables = {t["name"]: t for t in data["tables"]}
    cols = {c["name"]: c for c in tables["customers"]["columns"]}
    assert cols["email"]["sensitive"] and "values" not in cols["email"]
    assert "India" in cols["country"]["values"]
    statuses = {c["name"]: c for c in tables["orders"]["columns"]}["status"]["values"]
    assert "DELIVERED" in statuses
    assert {"join": "order_items.product_id -> products.id", "origin": "inferred"} \
        in data["relationships"]
    assert any(g["term"] == "revenue" for g in data["glossary"])


def test_history_is_per_session_and_newest_first(setup):
    client, _, _ = setup
    ask(client, "Orders by status?", "alice")
    ask(client, "How many orders were cancelled?", "alice")
    ask(client, "Orders by status?", "bob")
    alice = client.get("/v1/history", params={"session_id": "alice"}).json()["items"]
    assert [i["question"] for i in alice] == ["How many orders were cancelled?",
                                              "Orders by status?"]
    assert alice[0]["rows"] == [[92]]  # results included
    bob = client.get("/v1/history", params={"session_id": "bob"}).json()["items"]
    assert len(bob) == 1


def test_clarification_choices_dont_leak_between_sessions(setup):
    client, _, _ = setup
    c = ask(client, "What was revenue?", "alice")
    client.post(f"/v1/query/{c['request_id']}/choose", json={"session_id": "alice", "option": 1})
    assert ask(client, "What was revenue?", "alice")["status"] == "ok"  # remembered
    assert ask(client, "What was revenue?", "bob")["status"] == "needs_clarification"


# --- Step 3: feedback -------------------------------------------------------------

def test_feedback_is_stored_with_the_systems_own_question_and_sql(setup):
    client, _, store = setup
    answer = ask(client, "How many orders were cancelled?")
    r = client.post("/v1/feedback", json={"session_id": "s1", "request_id": answer["request_id"],
                                          "verdict": "correct"}).json()
    assert r["kind"] == "example" and r["status"] == "pending"
    item = store.get(r["feedback_id"])
    assert item.question == "How many orders were cancelled?" and item.sql == answer["sql"]
    assert item.confidence_band == "medium"


def test_feedback_must_reference_an_answer_in_the_session(setup):
    client, _, _ = setup
    answer = ask(client, "Orders by status?", "alice")
    r = client.post("/v1/feedback", json={"session_id": "mallory",
                                          "request_id": answer["request_id"], "verdict": "correct"})
    assert r.status_code == 404


def test_incorrect_feedback_kinds(setup):
    client, _, _ = setup
    answer = ask(client, "Orders by status?")
    base = {"session_id": "s1", "request_id": answer["request_id"], "verdict": "incorrect"}
    plain = client.post("/v1/feedback", json=base).json()
    assert plain["kind"] == "needs_correction"
    fixed = client.post("/v1/feedback", json={**base, "corrected_sql": "SELECT 1"}).json()
    assert fixed["kind"] == "test_case"
    listed = client.get("/v1/feedback", params={"session_id": "s1"}).json()["items"]
    assert len(listed) == 2


# --- the review gate ----------------------------------------------------------------

@pytest.fixture
def checkers(setup):
    _, text2sql, _ = setup
    return text2sql.generator.validator, text2sql.executor


def test_promote_correct_answer_to_example(setup, knowledge_dir, checkers):
    client, text2sql, store = setup
    item = store.add("r1", "How many orders are still pending?",
                     "SELECT COUNT(*) FROM orders WHERE status = 'PENDING'", "correct")
    promote(item, store, knowledge_dir, *checkers)
    examples = yaml.safe_load((knowledge_dir / "examples.yaml").read_text())
    assert examples[-1] == {"question": "How many orders are still pending?",
                            "sql": "SELECT COUNT(*) FROM orders WHERE status = 'PENDING'\n"}
    assert store.get(item.feedback_id).status == "promoted"
    text2sql.reload_knowledge()  # the next prompt can use it
    assert any(e.question == "How many orders are still pending?"
               for e in text2sql.knowledge.examples)


def test_promote_correction_to_test_case(setup, knowledge_dir, checkers):
    _, _, store = setup
    before = len(yaml.safe_load((knowledge_dir / "eval.yaml").read_text())["cases"])
    item = store.add("r2", "How many orders are on hold?",
                     "SELECT COUNT(*) FROM orders WHERE status = 'hold'", "incorrect",
                     corrected_sql="SELECT COUNT(*) FROM orders WHERE status = 'PENDING'")
    promote(item, store, knowledge_dir, *checkers)
    cases = yaml.safe_load((knowledge_dir / "eval.yaml").read_text())["cases"]
    assert cases[-1]["question"] == "How many orders are on hold?"
    assert cases[-1]["tags"] == ["from-feedback"]
    assert len(cases) == before + 1


@pytest.mark.parametrize("kwargs,error", [
    (dict(sql="SELECT COUNT(*) FROM orders WHERE status = 'PENDING'SELECT COUNT(*)",
          verdict="correct"), "failed re-checking: Syntax error"),
    (dict(sql="DELETE FROM orders", verdict="correct"), "failed re-checking"),
    (dict(sql="SELECT id FROM orders WHERE 1 = 0", verdict="correct"), "returns no rows"),
    (dict(sql="SELECT 1", verdict="incorrect"), "without a corrected query"),
])
def test_promotion_gate_refuses_bad_items(setup, knowledge_dir, checkers, kwargs, error):
    _, _, store = setup
    before = (knowledge_dir / "examples.yaml").read_text()
    item = store.add("r3", "Some new question?", **kwargs)
    with pytest.raises(PromotionError, match=error):
        promote(item, store, knowledge_dir, *checkers)
    assert (knowledge_dir / "examples.yaml").read_text() == before
    assert store.get(item.feedback_id).status == "pending"


def test_duplicates_and_reviewed_items_are_refused(setup, knowledge_dir, checkers):
    _, _, store = setup
    dup = store.add("r4", "How many customers are there in each country?",
                    "SELECT country, COUNT(*) FROM customers GROUP BY country", "correct")
    with pytest.raises(PromotionError, match="already in examples.yaml"):
        promote(dup, store, knowledge_dir, *checkers)
    reject(dup, store, "duplicate")
    with pytest.raises(PromotionError, match="already rejected"):
        promote(store.get(dup.feedback_id), store, knowledge_dir, *checkers)


# --- demo mode ---------------------------------------------------------------------

def test_demo_replays_known_questions_only(knowledge_dir):
    replay = ReplayGenerator(knowledge_dir)

    class P:
        def __init__(self, q):
            self.user = f"## Question\n{q}"

    known = replay.generate(P("How many orders were cancelled?"))
    assert known.sql and "Demo mode" in known.explanation and known.model == "demo-replay"
    assert replay.generate(P("Who won the football yesterday?")).sql is None


# --- persistent history and reload ---------------------------------------------------

def _app(engine, knowledge_dir, tmp_path):
    text2sql = Text2SQLApp(engine, knowledge_dir=knowledge_dir, use_filter=False,
                           generator=Scripted(), back_translate=False, multi_query="never",
                           log_dir=tmp_path / "logs", schema_cache_dir=tmp_path / "schema",
                           history_dir=tmp_path / "history")
    return TestClient(create_app(text2sql, FeedbackStore(tmp_path / "feedback.jsonl"))), text2sql


def test_history_and_feedback_survive_a_restart(engine, knowledge_dir, tmp_path):
    client, _ = _app(engine, knowledge_dir, tmp_path)
    answer = ask(client, "How many orders were cancelled?", "alice")

    restarted, _ = _app(engine, knowledge_dir, tmp_path)  # same history directory
    items = restarted.get("/v1/history", params={"session_id": "alice"}).json()["items"]
    assert [i["question"] for i in items] == ["How many orders were cancelled?"]
    assert items[0]["rows"] == [[92]]
    r = restarted.post("/v1/feedback", json={"session_id": "alice",
                                              "request_id": answer["request_id"],
                                              "verdict": "correct"})
    assert r.status_code == 200 and r.json()["question"] == "How many orders were cancelled?"


def test_reload_picks_up_promoted_examples_and_keeps_sessions(engine, knowledge_dir, tmp_path):
    client, text2sql = _app(engine, knowledge_dir, tmp_path)
    c = ask(client, "What was revenue?", "alice")
    client.post(f"/v1/query/{c['request_id']}/choose", json={"session_id": "alice", "option": 1})
    before = len(text2sql.knowledge.examples)

    store = FeedbackStore(tmp_path / "f.jsonl")
    item = store.add("r", "How many orders are pending?",
                     "SELECT COUNT(*) FROM orders WHERE status = 'PENDING'", "correct")
    promote(item, store, knowledge_dir, text2sql.generator.validator, text2sql.executor)
    r = client.post("/v1/admin/reload").json()
    assert r == {"status": "reloaded", "examples": before + 1}
    # alice's earlier choice of net revenue survived the reload
    assert ask(client, "What was revenue?", "alice")["status"] == "ok"
    assert len(client.get("/v1/history", params={"session_id": "alice"}).json()["items"]) == 3


def test_history_store_rejects_unsafe_session_ids(tmp_path):
    from text2sql.execution.history import HistoryStore

    with pytest.raises(ValueError):
        HistoryStore(tmp_path).load("../../etc/passwd")


def test_unsafe_sql_is_reported_as_blocked_even_when_also_invalid(setup):
    client, _, _ = setup
    # The CTE body is a DELETE, so validation alone would call it "unknown table d".
    sql = "WITH d AS (DELETE FROM orders RETURNING id) SELECT COUNT(*) FROM d"
    data = client.post("/v1/sql", json={"sql": sql, "session_id": "s1"}).json()
    assert data["status"] == "blocked" and "DELETE is not allowed" in data["error"]
