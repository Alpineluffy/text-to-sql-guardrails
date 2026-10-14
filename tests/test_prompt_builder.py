from datetime import date
from pathlib import Path

import pytest
import sqlalchemy as sa

from text2sql.prompt import Knowledge, LexicalExampleSelector, PromptBuilder, find_value_mentions
from text2sql.prompt.dates import date_context
from text2sql.prompt.knowledge import Definition, GlossaryTerm, TableNotes
from text2sql.sample_data import build_sample_db
from text2sql.schema import SchemaExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"
TODAY = date(2026, 10, 6)


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return sa.create_engine(f"sqlite:///{path}")


@pytest.fixture(scope="module")
def schema(engine):
    return SchemaExtractor(engine).extract()


@pytest.fixture(scope="module")
def knowledge():
    return Knowledge.load(KNOWLEDGE_DIR)


@pytest.fixture(scope="module")
def builder(schema, knowledge):
    return PromptBuilder(schema, knowledge)


# --- knowledge files ---------------------------------------------------------

def test_knowledge_matches_schema(schema, knowledge):
    assert knowledge.validate(schema) == []


def test_validation_catches_stale_references(schema):
    stale = Knowledge(
        tables={"orders": TableNotes(columns={"order_total": "renamed column"}),
                "invoices": TableNotes(description="no such table")},
        glossary=[GlossaryTerm("discount", [Definition(
            "discount", "SUM(orders.discount) WHERE orders.status = 'x.y'")])],
    )
    problems = stale.validate(schema)
    assert len(problems) == 3  # 'x.y' is inside a string literal, so not checked
    assert "semantic.yaml: unknown column 'orders.order_total'" in problems
    assert "semantic.yaml: unknown table 'invoices'" in problems
    assert any("orders.discount" in p for p in problems)


@pytest.mark.parametrize("index", range(15))
def test_every_example_query_runs(engine, knowledge, index):
    assert len(knowledge.examples) == 15
    example = knowledge.examples[index]
    with engine.connect() as conn:
        rows = conn.execute(sa.text(example.sql)).fetchall()
    assert rows, f"no rows for: {example.question}"


# --- schema rendering ---------------------------------------------------------

def test_schema_shows_real_values_and_descriptions(builder):
    user = builder.build("How many orders?", today=TODAY).user
    assert "status VARCHAR(20) NOT NULL,  -- Fulfilment status of the order. Values: 'DELIVERED'" in user
    assert "Values: 'web', 'mobile', 'store'" in user
    assert "resolved BOOLEAN NOT NULL  -- Whether the ticket has been closed as resolved. Values: 1, 0" in user
    assert "Range: 2024-01-06 to 2026-09-30" in user
    assert "customer_id INTEGER NOT NULL REFERENCES customers(id)" in user


def test_pii_is_marked_and_never_shows_values(builder):
    user = builder.build("Show customers", today=TODAY).user
    assert "email VARCHAR(150) NOT NULL,  -- SENSITIVE personal data" in user
    assert "@example.com" not in user


def test_relationships_flag_inferred_keys(builder):
    user = builder.build("Units per product", today=TODAY).user
    assert "orders.customer_id = customers.id\n" in user
    assert "order_items.product_id = products.id  (inferred from data" in user


def test_table_subset_limits_schema_and_joins(builder):
    prompt = builder.build("Revenue by country", tables=["orders", "customers", "nope"], today=TODAY)
    assert prompt.tables == ["customers", "orders"]
    assert "CREATE TABLE products" not in prompt.user
    schema_section = prompt.user.split("## Relationships")[0]
    assert "order_items" not in schema_section and "support_tickets" not in schema_section
    assert "orders.customer_id = customers.id" in prompt.user
    assert prompt.warnings == ["ignored unknown tables: nope"]


# --- question-specific context ------------------------------------------------

def test_value_mentions_fix_case(schema):
    mentions = find_value_mentions("orders from india via MOBILE", schema, ["customers", "orders"])
    found = {(m.column, m.value, m.phrase) for m in mentions}
    assert found == {("country", "India", "india"), ("channel", "mobile", "MOBILE")}


def test_value_mentions_use_value_index(schema):
    mentions = find_value_mentions("units of laptop pro 14 sold", schema, ["products"])
    assert [(m.column, m.value) for m in mentions] == [("name", "Laptop Pro 14")]


def test_ambiguous_term_is_flagged(builder):
    prompt = builder.build("What was revenue from India last quarter?", today=TODAY)
    assert prompt.ambiguous_terms == ["revenue"]
    assert "AMBIGUOUS" in prompt.user
    assert "gross revenue:" in prompt.user and "net revenue:" in prompt.user


def test_named_definition_resolves_ambiguity(builder):
    prompt = builder.build("What was net revenue from India last quarter?", today=TODAY)
    assert prompt.glossary_terms == ["revenue"]
    assert prompt.ambiguous_terms == []
    assert "net revenue:" in prompt.user and "gross revenue:" not in prompt.user


def test_examples_dont_bias_an_ambiguous_term(builder):
    prompt = builder.build("What was revenue by customer country?", today=TODAY)
    assert prompt.examples
    assert not any("revenue" in e.question.lower() for e in prompt.examples)

    resolved = builder.build("What was net revenue by segment?", today=TODAY)
    assert resolved.examples[0].question == "What is the net revenue for each customer segment?"


def test_glossary_synonyms(builder):
    assert builder.build("What's our AOV on mobile?", today=TODAY).glossary_terms == [
        "average order value"
    ]


def test_example_selection(knowledge):
    selector = LexicalExampleSelector(knowledge.examples)
    top = selector.select("top 10 products by units sold", 2)
    assert top[0].question == "What are the top 5 products by units sold?"
    assert selector.select("zzz qqq", 3) == []


def test_date_context():
    assert "Last quarter (Q3 2026): 2026-07-01 to 2026-09-30" in date_context(TODAY)
    lines = date_context(date(2026, 2, 15))
    assert "Last quarter (Q4 2025): 2025-10-01 to 2025-12-31" in lines
    assert "Last month: 2026-01-01 to 2026-01-31" in lines


def test_token_budget_sheds_examples_first(schema, knowledge):
    full = PromptBuilder(schema, knowledge).build("orders by channel", today=TODAY)
    tight = PromptBuilder(schema, knowledge, token_budget=full.estimated_tokens - 50).build(
        "orders by channel", today=TODAY
    )
    assert len(tight.examples) < len(full.examples)
    assert tight.estimated_tokens <= full.estimated_tokens - 50
    assert "dropped an example" in tight.warnings[0]


def test_system_prompt_is_stable_and_dialect_aware(builder):
    a = builder.build("orders by channel", today=TODAY)
    b = builder.build("revenue by country", today=TODAY)
    assert a.system == b.system  # identical across questions, so it can be cached
    assert "SQLite" in a.system and "strftime" in a.system
    assert a.user.endswith("## Question\norders by channel")
