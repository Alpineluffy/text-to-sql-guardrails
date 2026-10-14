"""Guardrail tests, centred on an adversarial suite: every attack must be
blocked, and every legitimate analytics query must still run."""
import copy
from datetime import date
from pathlib import Path

import pytest
import sqlalchemy as sa

from text2sql.generation import (
    ClarificationSession, ReadOnlyExecutor, SQLResponse, SQLValidator, ValidatingGenerator,
)
from text2sql.guardrails import (
    AuditLog, GuardedExecutor, GuardrailConfig, GuardrailError, Guardrails, estimate_postgres,
    estimate_sqlite,
)
from text2sql.evaluation.attacks import LEGITIMATE, PG_ATTACKS, SQL_ATTACKS
from text2sql.prompt import Knowledge, PromptBuilder
from text2sql.sample_data import build_sample_db
from text2sql.schema import SchemaExtractor

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture(scope="module")
def schema(engine):
    return SchemaExtractor(engine).extract()


@pytest.fixture(scope="module")
def executor(engine):
    return ReadOnlyExecutor(engine)


@pytest.fixture(scope="module")
def guard(schema, executor):
    return Guardrails(schema, explain=executor.explain)


@pytest.fixture(scope="module")
def pg_guard(schema):
    pg = copy.deepcopy(schema)
    pg.dialect = "postgresql"
    return Guardrails(pg)


# --- adversarial suite: must be refused outright (never sent back for "repair") ---

HARD_BLOCKED = SQL_ATTACKS

PG_HARD_BLOCKED = PG_ATTACKS


@pytest.mark.parametrize("sql", HARD_BLOCKED)
def test_attack_is_refused(guard, sql):
    decision = guard.check(sql)
    assert not decision.allowed, f"allowed: {sql!r}"
    assert not any(v.retryable for v in decision.violations), decision.reasons()


@pytest.mark.parametrize("sql", PG_HARD_BLOCKED)
def test_postgres_attack_is_refused(pg_guard, sql):
    decision = pg_guard.check(sql)
    assert not decision.allowed, f"allowed: {sql!r}"
    assert not any(v.retryable for v in decision.violations), decision.reasons()


def test_attacks_never_reach_the_database(guard, executor, engine):
    guarded = GuardedExecutor(executor, guard)
    for sql in HARD_BLOCKED:
        with pytest.raises(GuardrailError):
            guarded.run(sql)
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM orders").scalar() == 1000


# --- policy rules: blocked, but the model may fix them ----------------------------

@pytest.mark.parametrize("sql,rule", [
    ("SELECT * FROM customers", "sensitive_columns"),
    ("SELECT email FROM customers WHERE country = 'India'", "sensitive_columns"),
    ("SELECT COUNT(*) FROM customers WHERE email LIKE '%@gmail.com'", "sensitive_columns"),
    ("SELECT o.id, c.country FROM orders o, customers c", "cartesian_join"),
    ("SELECT o.id FROM orders o CROSS JOIN products p", "cartesian_join"),
    ("SELECT id FROM orders WHERE customer_id IN (SELECT id FROM customers WHERE id IN "
     "(SELECT customer_id FROM orders WHERE id IN (SELECT order_id FROM order_items WHERE "
     "product_id IN (SELECT id FROM products))))", "subquery_depth"),
])
def test_policy_violations_are_retryable(guard, sql, rule):
    decision = guard.check(sql)
    assert [v.rule for v in decision.violations] == [rule], decision.reasons()
    assert decision.retryable


def test_blocked_and_allowed_tables(schema):
    blocked = Guardrails(schema, GuardrailConfig(blocked_tables=("support_tickets",)))
    assert blocked.check("SELECT priority FROM support_tickets").violations[0].rule == "table_access"
    only = Guardrails(schema, GuardrailConfig(allowed_tables=("orders",)))
    assert only.check("SELECT status FROM orders").allowed
    assert not only.check("SELECT category FROM products").allowed


def test_scan_estimate_blocks_runaway_queries(schema, executor):
    tight = Guardrails(schema, GuardrailConfig(max_scan_rows=50_000, block_cartesian_joins=False),
                       explain=executor.explain)
    decision = tight.check("SELECT o.id, c.country FROM orders o, customers c")
    assert decision.violations[0].rule == "scan_estimate"
    assert decision.scan.rows == 201_000  # 1,000 orders x 200 customers, plus the outer scan
    assert tight.check("SELECT o.id FROM orders o JOIN customers c ON c.id = o.customer_id").allowed


# --- legitimate queries must pass ---------------------------------------------------

LEGIT = LEGITIMATE


@pytest.mark.parametrize("sql", LEGIT)
def test_legitimate_query_passes(guard, executor, sql):
    decision = guard.check(sql)
    assert decision.allowed, decision.reasons()
    GuardedExecutor(executor, guard).run(sql)  # and it actually runs


def test_every_verified_example_passes(guard, executor):
    for example in Knowledge.load(ROOT / "knowledge" / "ecommerce").examples:
        decision = guard.check(example.sql)
        assert decision.allowed, (example.question, decision.reasons())
        GuardedExecutor(executor, guard).run(example.sql)


# --- row limit ------------------------------------------------------------------------

@pytest.mark.parametrize("sql,expected_sql,note", [
    ("SELECT id FROM orders", "SELECT id FROM orders LIMIT 1000", "added LIMIT 1000"),
    ("SELECT id FROM orders LIMIT 50000", "SELECT id FROM orders LIMIT 1000",
     "lowered LIMIT 50000 to 1000"),
    ("SELECT id FROM orders ORDER BY id DESC", "SELECT id FROM orders ORDER BY id DESC LIMIT 1000",
     "added LIMIT 1000"),
    ("SELECT id FROM orders UNION SELECT id FROM customers",
     "SELECT id FROM orders UNION SELECT id FROM customers LIMIT 1000", "added LIMIT 1000"),
])
def test_row_limit_rewrites(guard, sql, expected_sql, note):
    decision = guard.check(sql)
    assert decision.allowed and decision.sql == expected_sql and decision.rewrites == [note]


@pytest.mark.parametrize("sql", [
    "SELECT id FROM orders LIMIT 10",
    "SELECT COUNT(*) FROM orders",  # one row: a LIMIT would only add noise
    "SELECT SUM(total_amount), AVG(total_amount) FROM orders WHERE status = 'DELIVERED'",
])
def test_row_limit_leaves_alone(guard, sql):
    decision = guard.check(sql)
    assert decision.allowed and decision.sql == sql and decision.rewrites == []


def test_guarded_executor_runs_the_rewritten_query(guard, executor):
    result = GuardedExecutor(executor, guard).run("SELECT order_id FROM order_items")
    assert len(result.rows) == 1000  # 2,565 rows exist; LIMIT 1000 was added
    assert result.guardrails.rewrites == ["added LIMIT 1000"]


# --- configuration ------------------------------------------------------------------

def test_yaml_config_and_switching_rules_off(schema):
    config = GuardrailConfig.from_yaml(ROOT / "config" / "guardrails.yaml")
    assert config == GuardrailConfig()  # the shipped file matches the defaults

    relaxed = Guardrails(schema, GuardrailConfig(
        max_rows=None, max_subquery_depth=None, block_sensitive_columns=False,
        block_cartesian_joins=False,
    ))
    decision = relaxed.check("SELECT * FROM customers, orders")
    assert decision.allowed and decision.rewrites == []
    assert not relaxed.check("DELETE FROM orders").allowed  # read-only can't be switched off


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown guardrail settings: max_row"):
        GuardrailConfig.from_dict({"max_row": 10})


# --- audit log ----------------------------------------------------------------------

def test_blocked_and_rewritten_queries_are_logged(schema, tmp_path):
    audit = AuditLog(tmp_path / "guardrails.jsonl")
    g = Guardrails(schema, audit=audit)
    g.check("DELETE FROM orders")
    g.check("SELECT id FROM orders")
    g.check("SELECT COUNT(*) FROM orders")  # clean: not logged
    g.check("DELETE FROM orders")  # repeat: served from cache, logged once
    records = audit.read()
    assert [r["event"] for r in records] == ["blocked", "rewritten"]
    assert records[0]["violations"] == [{
        "rule": "read_only", "message": "only SELECT queries are allowed, not DELETE",
        "retryable": False,
    }]
    assert records[1]["final_sql"] == "SELECT id FROM orders LIMIT 1000"


# --- scan estimators --------------------------------------------------------------------

def test_sqlite_estimate_models_loops():
    counts = {"orders": 1000, "customers": 200}
    aliases = {"o": "orders", "c": "customers", "orders": "orders", "customers": "customers"}
    indexed_join = [(1, 0, "SCAN o"), (2, 0, "SEARCH c USING INTEGER PRIMARY KEY (rowid=?)")]
    assert estimate_sqlite(indexed_join, aliases, counts).rows == 2000
    correlated = [(1, 0, "SCAN c"), (2, 0, "CORRELATED SCALAR SUBQUERY 1"), (3, 2, "SCAN o")]
    assert estimate_sqlite(correlated, aliases, counts).rows == 200 + 200 * 1000
    unknown = estimate_sqlite([(1, 0, "SCAN mystery")], aliases, counts)
    assert unknown.rows == 1000 and "assumed" in unknown.notes[0]


def test_postgres_estimate_multiplies_nested_loops():
    plan = [{"Plan": {"Node Type": "Nested Loop", "Plan Rows": 1000, "Plans": [
        {"Node Type": "Seq Scan", "Relation Name": "orders", "Plan Rows": 1000},
        {"Node Type": "Index Scan", "Relation Name": "customers", "Plan Rows": 1},
    ]}}]
    estimate = estimate_postgres(plan, {"orders": 1000, "customers": 200})
    assert estimate.rows == 1000 + 1000 * 1
    assert estimate.plan[0] == "Nested Loop"


# --- with the generator: repair what's fixable, refuse what isn't --------------------

def _response(sql):
    return SQLResponse.from_json({
        "sql": sql, "explanation": "x", "tables_used": [], "columns_used": [],
        "assumptions": [], "confidence": 0.5, "needs_clarification": False,
        "clarification": None,
    })


class Scripted:
    def __init__(self, *sqls):
        self.sqls, self.prompts = list(sqls), []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return _response(self.sqls.pop(0))


def _session(schema, guard, executor, scripted):
    builder = PromptBuilder(schema, Knowledge.load(ROOT / "knowledge" / "ecommerce"))
    generator = ValidatingGenerator(scripted, SQLValidator(schema), guardrails=guard)
    return ClarificationSession(builder, generator, GuardedExecutor(executor, guard),
                                today=date(2026, 10, 6))


def test_fixable_violation_goes_back_to_the_model(schema, guard, executor):
    scripted = Scripted("SELECT email FROM customers WHERE country = 'India'",
                        "SELECT COUNT(*) FROM customers WHERE country = 'India'")
    answer = _session(schema, guard, executor, scripted).ask("How many customers in India?")
    assert answer.error is None and answer.result.rows[0][0] > 0
    assert "[sensitive_columns] reads personal data (customers.email)" in scripted.prompts[1].user


def test_unsafe_query_is_refused_without_retry(schema, guard, executor):
    scripted = Scripted("DELETE FROM orders")
    answer = _session(schema, guard, executor, scripted).ask("Delete all orders")
    assert len(scripted.prompts) == 1  # no repair attempt for a hard violation
    assert answer.error == ("Blocked by guardrails: [read_only] only SELECT queries "
                            "are allowed, not DELETE")
