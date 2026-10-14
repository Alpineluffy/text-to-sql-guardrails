"""Sandbox tests. The point of the sandbox is to hold even when every earlier
layer fails, so most tests go around the guardrails and the text check and
talk to the database layer directly."""
import os
import time

import pytest
import sqlalchemy as sa

from text2sql.guardrails import AuditLog, GuardrailConfig
from text2sql.sample_data import build_sample_db, load_into
from text2sql.sandbox import SandboxConfig, SandboxedExecutor, SandboxError
from text2sql.sandbox.postgres import create_reader_role, role_setup_sql, verify_reader
from text2sql.schema import ExtractionConfig, SchemaExtractor

RUNAWAY = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture(scope="module")
def schema(engine):
    return SchemaExtractor(engine).extract()


@pytest.fixture
def audit(tmp_path):
    return AuditLog(tmp_path / "audit.jsonl")


@pytest.fixture
def sandbox(engine, schema, audit):
    return SandboxedExecutor.for_schema(engine, schema, audit=audit)


# --- SQLite: the database itself refuses, even with every other layer skipped ---

@pytest.mark.parametrize("sql", [
    "DELETE FROM orders",
    "UPDATE orders SET status = 'X'",
    "INSERT INTO orders (id) VALUES (999999)",
    "DROP TABLE orders",
    "CREATE TABLE t (x INT)",
    "CREATE TEMP TABLE t (x INT)",
    "PRAGMA writable_schema = 1",
    "PRAGMA query_only = 0",
    "ATTACH DATABASE ':memory:' AS other",
    "WITH x AS (SELECT 1) DELETE FROM orders",
])
def test_database_layer_refuses_writes(sandbox, engine, sql):
    # Straight to the sandbox's connection: no guardrails, no text check.
    with sandbox._engine.connect() as conn:
        with pytest.raises(sa.exc.DBAPIError):
            conn.exec_driver_sql(sql)
        conn.rollback()
    with engine.connect() as conn:
        assert conn.exec_driver_sql("SELECT COUNT(*) FROM orders").scalar() == 1000


def test_personal_data_refused_by_the_database(sandbox, audit):
    for sql in ["SELECT email FROM customers",
                "SELECT x FROM (SELECT phone AS x FROM customers)",
                "SELECT * FROM customers"]:
        with pytest.raises(SandboxError) as info:
            sandbox.run(sql)
        assert info.value.kind == "denied" and info.value.is_security_event
    events = audit.read()
    assert len(events) == 3 and {e["event"] for e in events} == {"sandbox_refused"}
    assert events[0]["message"] == "access to customers.email is prohibited"


def test_non_sensitive_columns_still_readable(sandbox):
    assert sandbox.run("SELECT COUNT(*) FROM customers").rows == [(200,)]
    assert sandbox.run("SELECT country FROM customers WHERE id = 1").rows[0][0]
    assert sandbox.run("SELECT country, COUNT(*) FROM order_summary GROUP BY country").rows


def test_blocked_functions_refused_by_the_database(sandbox):
    with pytest.raises(SandboxError) as info:
        sandbox.run("SELECT randomblob(10)")
    assert info.value.kind == "denied"


def test_blocked_tables_refused_by_the_database(engine, schema):
    sandbox = SandboxedExecutor.for_schema(
        engine, schema, GuardrailConfig(blocked_tables=("support_tickets",))
    )
    with pytest.raises(SandboxError, match="prohibited"):
        sandbox.run("SELECT priority FROM support_tickets")
    assert sandbox.run("SELECT COUNT(*) FROM orders").rows == [(1000,)]


def test_runaway_query_is_stopped_and_connection_recovers(engine, schema):
    sandbox = SandboxedExecutor.for_schema(engine, schema,
                                           config=SandboxConfig(statement_timeout_ms=200))
    start = time.monotonic()
    with pytest.raises(SandboxError) as info:
        sandbox.run(RUNAWAY)
    assert info.value.kind == "timeout" and not info.value.is_security_event
    assert str(info.value) == "query took longer than 0.2s and was stopped"
    assert time.monotonic() - start < 2
    assert sandbox.run("SELECT COUNT(*) FROM orders").rows == [(1000,)]  # usable afterwards


def test_ordinary_errors_are_not_security_events(sandbox, audit):
    with pytest.raises(SandboxError) as info:
        sandbox.run("SELECT nope FROM orders")
    assert info.value.kind == "query_error" and not info.value.is_security_event
    assert audit.read() == []


def test_results_are_capped_and_timed(engine, schema):
    sandbox = SandboxedExecutor.for_schema(engine, schema, config=SandboxConfig(max_rows=10))
    result = sandbox.run("SELECT id FROM orders")
    assert len(result.rows) == 10 and result.truncated
    assert result.elapsed_ms is not None and result.elapsed_ms >= 0


def test_explain_works_inside_the_sandbox(sandbox):
    plan = sandbox.explain("SELECT COUNT(*) FROM orders WHERE status = 'DELIVERED'")
    assert any("SCAN" in row[-1] for row in plan)


# --- PostgreSQL role SQL (no database needed) -------------------------------------

def test_role_sql_grants_least_privilege(schema):
    stmts = role_setup_sql(schema, "shop", password="pw'quote",
                           guardrails=GuardrailConfig(blocked_tables=("support_tickets",)))
    text = "\n".join(stmts)
    assert "NOSUPERUSER" in text and "PASSWORD 'pw''quote'" in text
    assert 'ALTER ROLE "text2sql_reader" SET default_transaction_read_only = on' in stmts
    assert 'ALTER ROLE "text2sql_reader" SET statement_timeout = 5000' in stmts
    assert 'GRANT SELECT ON "public"."orders" TO "text2sql_reader"' in stmts
    # customers: column-level grant without the personal-data columns
    assert ('GRANT SELECT ("id", "country", "segment", "created_at") ON "public"."customers" '
            'TO "text2sql_reader"') in stmts
    assert "support_tickets" not in text  # blocked: never granted
    assert not any("INSERT" in s or "UPDATE" in s or "DELETE" in s for s in stmts)


# --- PostgreSQL, end to end (needs a server: see docker-compose.yml) --------------

PG_ADMIN_URL = os.environ.get("TEXT2SQL_PG_ADMIN_URL")
postgres = pytest.mark.skipif(not PG_ADMIN_URL, reason="set TEXT2SQL_PG_ADMIN_URL")


@pytest.fixture(scope="module")
def pg_reader():
    admin = sa.create_engine(PG_ADMIN_URL)
    load_into(admin)
    schema = SchemaExtractor(admin, ExtractionConfig(infer_foreign_keys=False)).extract_structure()
    create_reader_role(admin, schema, role="text2sql_test_reader", password="test-reader-pw")
    url = sa.engine.make_url(PG_ADMIN_URL).set(username="text2sql_test_reader",
                                               password="test-reader-pw")
    return sa.create_engine(url), schema


@postgres
def test_pg_reader_role_passes_every_check(pg_reader):
    engine, schema = pg_reader
    failed = [c for c in verify_reader(engine, schema) if not c.passed]
    assert failed == []


@postgres
def test_pg_sandbox_reads_but_refuses_everything_else(pg_reader):
    engine, schema = pg_reader
    sandbox = SandboxedExecutor(engine, SandboxConfig(statement_timeout_ms=300))
    assert sandbox.run("SELECT COUNT(*) FROM orders").rows == [(1000,)]
    assert sandbox.run("SELECT country FROM customers LIMIT 1").rows

    for sql, kind in [
        ("SELECT email FROM customers", "denied"),
        ("SELECT * FROM customers", "denied"),
        ("WITH d AS (DELETE FROM orders RETURNING id) SELECT COUNT(*) FROM d", "read_only"),
        ("SELECT pg_read_file('/etc/passwd')", "denied"),
        ("SELECT pg_sleep(5)", "timeout"),
    ]:
        with pytest.raises(SandboxError) as info:
            sandbox.run(sql)
        assert info.value.kind == kind, (sql, info.value.kind, str(info.value))
    assert sandbox.run("SELECT COUNT(*) FROM orders").rows == [(1000,)]


@postgres
def test_pg_schema_can_be_extracted_as_the_reader(pg_reader):
    # The reader can't see personal data, so extraction must never touch it
    # (regression: FK inference once used EXISTS (SELECT * ...)).
    engine, _ = pg_reader
    schema = SchemaExtractor(engine).extract()
    assert schema.table("orders").row_count == 1000
    assert {fk.describe() for fk in schema.foreign_keys if fk.origin == "inferred"} == {
        "order_items.product_id -> products.id", "support_tickets.customer_id -> customers.id",
    }
    assert schema.table("customers").column("email").profile.skipped_reason == "pii"
