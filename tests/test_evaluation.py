"""Phase 5: the evaluation suite itself."""
import json
import shutil
from pathlib import Path

import pytest
import sqlalchemy as sa
import sqlglot

from text2sql.evaluation import report as report_mod
from text2sql.evaluation.dataset import load_golden
from text2sql.evaluation.live import normalise_sql, run_live
from text2sql.evaluation.mutation import MutationSuite
from text2sql.evaluation.security import fingerprint, run_security
from text2sql.execution import Text2SQLApp
from text2sql.generation import SQLResponse
from text2sql.guardrails import GuardedExecutor, Guardrails
from text2sql.prompt import Knowledge
from text2sql.sample_data import build_sample_db
from text2sql.sandbox import SandboxedExecutor
from text2sql.schema import SchemaExtractor

KNOWLEDGE = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")


@pytest.fixture(scope="module")
def engine(db):
    return sa.create_engine(f"sqlite:///{db}")


@pytest.fixture(scope="module")
def schema(engine):
    return SchemaExtractor(engine).extract()


@pytest.fixture(scope="module")
def guarded(engine, schema):
    sandbox = SandboxedExecutor.for_schema(engine, schema)
    return GuardedExecutor(sandbox, Guardrails(schema, explain=sandbox.explain))


@pytest.fixture(scope="module")
def cases():
    return load_golden(KNOWLEDGE)[1]


# --- golden dataset ----------------------------------------------------------------

def test_golden_set_composition(cases):
    kinds = [c.expect for c in cases]
    assert len(cases) >= 50
    assert kinds.count("answer") >= 40 and kinds.count("clarify") >= 5
    assert kinds.count("cannot_answer") >= 5
    tags = {t for c in cases for t in c.tags}
    assert {"join", "subquery", "relative-date", "ranking", "ambiguous", "unanswerable"} <= tags


def test_every_verified_query_returns_rows(cases, guarded):
    for c in cases:
        for sql in ([c.sql] if c.sql else []) + c.accept:
            assert guarded.run(sql).rows, c.question


def test_golden_sql_translates_to_postgres(cases):
    _, pg_cases = load_golden(KNOWLEDGE, dialect="postgresql")
    for c in pg_cases:
        for sql in ([c.sql] if c.sql else []) + c.accept:
            sqlglot.parse_one(sql, read="postgres")  # valid Postgres
            assert "strftime" not in sql.lower() and "resolved = 0" not in sql
    month = next(c for c in pg_cases if c.question.startswith("How many orders were placed each month"))
    assert "TO_CHAR" in month.sql


def test_examples_are_translated_for_postgres():
    k = Knowledge.load(KNOWLEDGE, dialect="postgresql")
    assert all("strftime" not in e.sql for e in k.examples)
    assert k.sql_dialect == "postgresql"


# --- mutation suite -----------------------------------------------------------------

def test_mutation_suite_catches_injected_bugs_without_false_alarms(schema, guarded, cases):
    report = MutationSuite(schema, Knowledge.load(KNOWLEDGE), guarded).run(cases)
    kinds = report.by_kind()
    assert report.false_alarms == []  # correct queries are never flagged
    for kind in ("value_case", "wrong_value", "drop_date", "fanout_join", "drop_limit"):
        assert kinds[kind]["rate"] == 1.0, (kind, kinds[kind])
    assert report.detection_rate >= 0.8
    assert all(m.problems for m in report.mutants if m.detected)


def test_only_mutants_that_change_the_answer_count(schema, guarded):
    from text2sql.evaluation.dataset import GoldenCase

    # Dropping a status filter that matches every row doesn't change the answer.
    case = GoldenCase("How many orders are there?",
                      sql="SELECT COUNT(*) FROM orders WHERE status <> 'NOPE'")
    report = MutationSuite(schema, Knowledge.load(KNOWLEDGE), guarded).run([case])
    assert not any(m.kind == "drop_status" for m in report.mutants)
    assert report.skipped["drop_status: same result"] == 1


# --- security suite -----------------------------------------------------------------

def test_security_suite_on_a_copy(db, schema, tmp_path, cases):
    copy = tmp_path / "copy.db"
    shutil.copy(db, copy)
    engine = sa.create_engine(f"sqlite:///{copy}")
    report = run_security(engine, schema, [c.sql for c in cases if c.sql])
    assert report.refused == report.attacks and report.executed == []
    assert report.pg_refused == report.pg_attacks
    assert report.false_blocks == [] and report.database_unchanged
    assert report.db_layer_allowed == ["BEGIN TRANSACTION"]  # harmless on a read-only connection


def test_fingerprint_notices_changes(tmp_path, db):
    copy = tmp_path / "copy.db"
    shutil.copy(db, copy)
    engine = sa.create_engine(f"sqlite:///{copy}")
    before = fingerprint(engine)
    with engine.begin() as conn:
        conn.exec_driver_sql("DELETE FROM support_tickets WHERE ticket_id = 1")
    assert fingerprint(engine) != before


# --- live run (scripted model) ------------------------------------------------------

def test_normalise_sql_ignores_formatting_case_and_aliases():
    a = "select COUNT(*) as n from orders where status = 'CANCELLED';"
    b = "SELECT count(*)\nFROM orders\nWHERE status = 'CANCELLED'"
    assert normalise_sql(a, "sqlite") == normalise_sql(b, "sqlite")
    assert normalise_sql(a, "sqlite") != normalise_sql(b.replace("CANCELLED", "PENDING"), "sqlite")


def reply(sql=None, interpretations=None):
    data = {"sql": sql, "explanation": "x", "tables_used": [], "columns_used": [],
            "assumptions": [], "confidence": 0.8,
            "needs_clarification": bool(interpretations), "clarification": None}
    if interpretations:
        data["clarification"] = {"question": "Which?", "interpretations": [
            {"label": l, "description": l, "sql": q} for l, q in interpretations]}
    return SQLResponse.from_json(data)


class Script:
    """Gets answer cases right (gold SQL), except one; asks on 'revenue last month';
    declines the weather; tries a write for an injection."""

    def __init__(self, cases):
        self.gold = {c.question: c.sql for c in cases if c.sql}
        self.accept = {c.question: c.accept for c in cases if c.accept}

    def generate(self, prompt):
        q = prompt.user.split("## Question\n")[1].split("\n")[0]
        if q == "How many orders were cancelled?":
            return reply("SELECT COUNT(*) FROM orders WHERE status = 'cancelled'")  # wrong
        if q in self.accept:
            return reply(interpretations=[("gross revenue", self.accept[q][0]),
                                          ("net revenue", self.accept[q][1])])
        if q in self.gold:
            return reply(self.gold[q])
        if q.startswith("Ignore all previous"):
            return reply("DELETE FROM orders")
        if "email" in q:
            return reply("SELECT email, phone FROM customers")
        return reply(None)  # can't answer


def test_live_run_scores_every_kind_of_case(engine, cases, guarded, tmp_path):
    picked = [c for c in cases if c.question in {
        "How many customers are there?", "How many orders were cancelled?",
        "What was revenue last month?", "Who are our best customers?",
        "What was the weather in Berlin last week?"}]
    app = Text2SQLApp(engine, knowledge_dir=KNOWLEDGE, use_filter=False, generator=Script(cases),
                      back_translate=False, multi_query="never", log_dir=tmp_path / "logs",
                      schema_cache_dir=tmp_path / "schema")
    report = run_live(app.new_service(), picked, guarded, "sqlite",
                      fingerprint=lambda: fingerprint(engine))
    by_q = {r.question: r for r in report.results}

    assert by_q["How many customers are there?"].correct
    assert by_q["How many customers are there?"].exact_match
    wrong = by_q["How many orders were cancelled?"]
    assert wrong.correct is False and wrong.band == "low"  # caught by the result checks
    assert report.execution_accuracy == 0.5 and report.detection == 1.0

    asked = by_q["What was revenue last month?"]
    assert asked.behaviour_ok and asked.correct  # asked, and the chosen reading is right
    assert not by_q["Who are our best customers?"].behaviour_ok  # answered "can't" instead
    assert by_q["What was the weather in Berlin last week?"].behaviour_ok

    assert all(i.safe for i in report.injections)  # writes blocked, PII refused
    assert report.database_unchanged


# --- reports --------------------------------------------------------------------------

def test_readme_block_is_inserted_once_and_updated(tmp_path, monkeypatch):
    monkeypatch.setattr(report_mod, "RESULTS_DIR", tmp_path / "eval")
    (tmp_path / "eval").mkdir()
    offline = {
        "measured_at": "2026-10-09",
        "dataset": {"total": 56, "by_expect": {"answer": 45}, "databases": ["sqlite"]},
        "mutation": {"detection_rate": 0.833, "false_alarm_rate": 0.0, "mutants": 102,
                     "gold_checked": 45, "by_kind": {}, "false_alarms": []},
        "security": {"attacks": 38, "refused": 38, "pg_attacks": 21, "pg_refused": 21,
                     "executed": [], "database_unchanged": True, "legitimate": 61,
                     "false_blocks": [], "db_layer_refused": 36, "db_layer_checked": 37,
                     "db_layer_allowed": []},
    }
    (tmp_path / "eval" / "offline.json").write_text(json.dumps(offline))
    readme = tmp_path / "README.md"
    readme.write_text("# Project\n\nIntro.\n")
    report_mod.write_reports(readme, tmp_path / "EVAL.md")
    report_mod.write_reports(readme, tmp_path / "EVAL.md")  # second run replaces, not appends
    text = readme.read_text()
    assert text.count("<!-- EVAL:START -->") == 1
    assert text.index("<!-- EVAL:START -->") < text.index("Intro.")  # numbers lead
    assert "0 unsafe queries executed across 59 attacks" in text
    assert "Execution accuracy | not measured yet" in text  # never invented
