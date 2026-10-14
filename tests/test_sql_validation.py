import copy
from datetime import date
from pathlib import Path

import pytest
import sqlalchemy as sa

from text2sql.generation import (
    Answer, ClarificationRequest, ClarificationSession, SQLResponse, SQLValidator,
    ValidatingGenerator,
)
from text2sql.prompt import Knowledge, PromptBuilder
from text2sql.sample_data import build_sample_db
from text2sql.schema import SchemaExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture(scope="module")
def schema(engine):
    return SchemaExtractor(engine).extract()


@pytest.fixture(scope="module")
def validator(schema):
    return SQLValidator(schema)


@pytest.fixture(scope="module")
def knowledge():
    return Knowledge.load(KNOWLEDGE_DIR)


# --- no false alarms ----------------------------------------------------------

def test_every_verified_example_passes(validator, knowledge):
    for example in knowledge.examples:
        report = validator.validate(example.sql)
        assert report.ok, f"{example.question}: {report.errors}"
        assert report.tables and report.columns


def test_case_and_quoting_differences_are_fine(validator):
    assert validator.validate('SELECT COUNT(*) FROM Orders WHERE STATUS = \'PENDING\'').ok
    assert validator.validate('SELECT "total_amount" FROM "orders"').ok


# --- what it catches -----------------------------------------------------------

@pytest.mark.parametrize("sql,expected", [
    ("SELEC * FROM orders", "Syntax error"),
    ("SELECT * FROM orders WHERE", "Syntax error"),
    ("SELECT 1; SELECT 2", "Syntax error: expected one statement, found 2"),
    ("SELECT id FROM invoices", "Unknown table: invoices"),
    ("SELECT o.customer_name FROM orders o", "Unknown column: o.customer_name"),
    ("SELECT customer_name FROM orders", "Unknown column: customer_name"),
    ("SELECT id FROM orders JOIN customers ON customers.id = orders.customer_id",
     "Ambiguous column (qualify it with a table): id"),
])
def test_errors(validator, sql, expected):
    report = validator.validate(sql)
    assert not report.ok
    assert any(e.startswith(expected) for e in report.errors), report.errors


def test_reports_every_invented_name_at_once(validator):
    report = validator.validate(
        "SELECT o.revenue, c.region FROM orders o JOIN customers c ON c.id = o.customer_id"
    )
    assert report.unknown_columns == ["o.revenue", "c.region"]


def test_columns_resolve_through_ctes_and_subqueries(validator):
    report = validator.validate(
        "WITH spend AS (SELECT customer_id, SUM(total_amount) AS s FROM orders GROUP BY customer_id) "
        "SELECT c.country, AVG(spend.s) FROM spend JOIN customers c ON c.id = spend.customer_id "
        "GROUP BY c.country"
    )
    assert report.ok
    assert report.tables == ["customers", "orders"]
    assert report.columns == [
        "customers.country", "customers.id", "orders.customer_id", "orders.total_amount",
    ]


def test_other_dialects(schema):
    pg = copy.deepcopy(schema)
    pg.dialect = "postgresql"
    report = SQLValidator(pg).validate(
        "SELECT date_trunc('month', order_date) AS m, COUNT(*) FROM orders "
        "WHERE status ILIKE 'deliv%' GROUP BY 1"
    )
    assert report.ok, report.errors


# --- the model's claims --------------------------------------------------------

def test_claims_matching_the_sql_give_no_warnings(validator):
    report = validator.validate(
        "SELECT COUNT(*) FROM orders WHERE status = 'PENDING'",
        claimed_tables=["orders"], claimed_columns=["orders.status"],
    )
    assert report.warnings == []


def test_claim_mismatches_are_warnings(validator):
    report = validator.validate(
        "SELECT c.country, SUM(o.total_amount) FROM orders o "
        "JOIN customers c ON c.id = o.customer_id GROUP BY c.country",
        claimed_tables=["orders", "products"],
        claimed_columns=["orders.total_amount", "customers.country"],
    )
    assert report.ok
    assert report.warnings == [
        "Model says it used tables the SQL doesn't read: products",
        "SQL reads tables the model didn't mention: customers",
        "SQL uses columns the model didn't mention: customers.id, orders.customer_id",
    ]


# --- repair loop ----------------------------------------------------------------

def response(sql=None, interpretations=None, tables=("orders",), columns=()):
    data = {
        "sql": sql, "explanation": "x", "tables_used": list(tables),
        "columns_used": list(columns), "assumptions": [], "confidence": 0.8,
        "needs_clarification": bool(interpretations), "clarification": None,
    }
    if interpretations:
        data["clarification"] = {"question": "Which?", "interpretations": [
            {"label": l, "description": l, "sql": q} for l, q in interpretations]}
    return SQLResponse.from_json(data)


class ScriptedGenerator:
    """Returns the scripted responses in order and records each prompt it saw."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


class ExplodingExecutor:
    def run(self, sql):
        raise AssertionError(f"invalid SQL must never be executed: {sql}")


@pytest.fixture(scope="module")
def builder(schema, knowledge):
    return PromptBuilder(schema, knowledge)


def test_invalid_sql_is_sent_back_and_fixed(builder, validator):
    scripted = ScriptedGenerator(
        response("SELECT o.customer_name FROM orders o"),
        response("SELECT COUNT(*) FROM orders", columns=()),
    )
    result = ValidatingGenerator(scripted, validator).generate(
        builder.build("How many orders?", today=date(2026, 10, 6))
    )
    assert result.sql == "SELECT COUNT(*) FROM orders"
    assert result.attempts == 2 and result.validation.ok
    retry_prompt = scripted.prompts[1].user
    assert "## Your previous answer was rejected" in retry_prompt
    assert "- Unknown column: o.customer_name" in retry_prompt
    assert retry_prompt.startswith(scripted.prompts[0].user)  # same context, plus feedback


def test_gives_up_after_max_repairs_and_never_executes(builder, validator):
    bad = [response("SELECT nope FROM orders") for _ in range(3)]
    session = ClarificationSession(
        builder, ValidatingGenerator(ScriptedGenerator(*bad), validator, max_repairs=2),
        ExplodingExecutor(), today=date(2026, 10, 6),
    )
    answer = session.ask("How many orders?")
    assert isinstance(answer, Answer)
    assert answer.error.startswith("SQL still invalid after 3 attempt(s): Unknown column: nope")
    assert answer.result is None


def test_bad_interpretation_triggers_repair(builder, validator):
    scripted = ScriptedGenerator(
        response(interpretations=[("gross revenue", "SELECT SUM(total_amount) FROM orders"),
                                  ("net revenue", "SELECT SUM(amount) FROM orders")]),
        response(interpretations=[("gross revenue", "SELECT SUM(total_amount) FROM orders"),
                                  ("net revenue", "SELECT SUM(total_amount) FROM orders "
                                                  "WHERE status <> 'RETURNED'")]),
    )
    result = ValidatingGenerator(scripted, validator).generate(
        builder.build("What was revenue?", today=date(2026, 10, 6))
    )
    assert result.attempts == 2
    assert "Interpretation 'net revenue' was rejected:" in scripted.prompts[1].user
    assert all(i.validation.ok for i in result.interpretations)


def test_still_invalid_interpretation_is_not_previewed(builder, validator):
    bad = response(interpretations=[("gross revenue", "SELECT SUM(total_amount) FROM orders"),
                                    ("net revenue", "SELECT SUM(amount) FROM orders")])
    session = ClarificationSession(
        builder, ValidatingGenerator(ScriptedGenerator(bad), validator, max_repairs=0),
        None, today=date(2026, 10, 6),
    )
    request = session.ask("What was revenue?")
    assert isinstance(request, ClarificationRequest)
    assert request.options[0].error is None
    assert request.options[1].error == "invalid SQL: Unknown column: amount"
    assert request.options[1].preview is None


def test_answer_carries_claim_warnings(builder, validator, engine):
    from text2sql.generation import ReadOnlyExecutor

    scripted = ScriptedGenerator(response(
        "SELECT COUNT(*) FROM orders o JOIN customers c ON c.id = o.customer_id "
        "WHERE c.country = 'India'", tables=["orders"],
        columns=["customers.country", "customers.id", "orders.customer_id"],
    ))
    session = ClarificationSession(builder, ValidatingGenerator(scripted, validator),
                                   ReadOnlyExecutor(engine), today=date(2026, 10, 6))
    answer = session.ask("How many orders from India?")
    assert answer.result.rows[0][0] > 0
    assert answer.warnings == ["SQL reads tables the model didn't mention: customers"]
