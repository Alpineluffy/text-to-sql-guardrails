import sqlite3

import pytest
import sqlalchemy as sa

from text2sql.sample_data import build_sample_db
from text2sql.schema import DatabaseSchema, ExtractionConfig, SchemaExtractor, load_or_extract
from text2sql.schema.fk_inference import candidate_table_names
from text2sql.schema.pii import is_pii


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    return build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")


@pytest.fixture(scope="module")
def schema(db_path):
    return SchemaExtractor(sa.create_engine(f"sqlite:///{db_path}")).extract()


def fk_pairs(schema, origin):
    return {
        (fk.source_table, fk.source_columns[0], fk.target_table, fk.target_columns[0])
        for fk in schema.foreign_keys
        if fk.origin == origin
    }


def test_tables_views_and_keys(schema):
    names = {t.name: t.kind for t in schema.tables}
    assert names == {
        "customers": "table", "products": "table", "orders": "table",
        "order_items": "table", "support_tickets": "table", "order_summary": "view",
    }
    assert schema.table("support_tickets").primary_key == ["ticket_id"]
    assert schema.table("orders").row_count == 1000
    assert schema.table("orders").column("total_amount").type_family == "numeric"
    assert schema.table("orders").column("order_date").type_family == "date"


def test_declared_foreign_keys(schema):
    assert fk_pairs(schema, "declared") == {
        ("orders", "customer_id", "customers", "id"),
        ("order_items", "order_id", "orders", "id"),
    }


def test_inferred_foreign_keys(schema):
    inferred = fk_pairs(schema, "inferred")
    assert inferred == {
        ("order_items", "product_id", "products", "id"),
        ("support_tickets", "customer_id", "customers", "id"),
    }
    # supplier_id looks like a key but there is no suppliers table.
    assert all(fk.match_ratio == 1.0 for fk in schema.foreign_keys if fk.origin == "inferred")


def test_inference_rejects_keys_the_data_contradicts(tmp_path):
    path = tmp_path / "bad.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE customers (id INTEGER PRIMARY KEY);
        CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER);
        INSERT INTO customers VALUES (1), (2);
        INSERT INTO orders VALUES (1, 1), (2, 77), (3, 88), (4, 99);
    """)
    conn.close()
    schema = SchemaExtractor(sa.create_engine(f"sqlite:///{path}")).extract()
    assert fk_pairs(schema, "inferred") == set()


def test_categorical_values_keep_real_spelling(schema):
    status = schema.table("orders").column("status").profile
    assert status.is_categorical
    assert set(status.sample_values) == {"DELIVERED", "SHIPPED", "PENDING", "CANCELLED", "RETURNED"}
    assert status.sample_values[0] == "DELIVERED"  # most frequent first
    assert status.case_style == "UPPER"

    channel = schema.table("orders").column("channel").profile
    assert channel.case_style == "lower"
    assert "India" in schema.table("customers").column("country").profile.sample_values
    assert set(schema.table("support_tickets").column("resolved").profile.sample_values) == {True, False}


def test_high_cardinality_text_goes_to_value_index(schema):
    name = schema.table("products").column("name").profile
    assert not name.is_categorical
    assert "Laptop Pro 14" in name.value_index
    assert schema.table("support_tickets").column("subject").profile.value_index


def test_pii_values_never_collected(schema):
    for col in ("full_name", "email", "phone"):
        column = schema.table("customers").column(col)
        assert column.is_pii
        assert column.profile.skipped_reason == "pii"
        assert not column.profile.sample_values and not column.profile.value_index
        assert column.profile.min_value is None


def test_key_columns_are_not_profiled_as_values(schema):
    ref = schema.table("orders").column("external_ref_id").profile
    assert ref.skipped_reason == "key column"
    assert not ref.value_index


def test_ranges(schema):
    orders = schema.table("orders")
    assert orders.column("order_date").profile.min_value >= "2024-01-01"
    assert orders.column("order_date").profile.max_value <= "2026-09-30"
    assert orders.column("total_amount").profile.min_value > 0
    assert 0 < orders.column("external_ref_id").profile.null_fraction < 1


def test_sampling_limits_rows_read(db_path):
    engine = sa.create_engine(f"sqlite:///{db_path}")
    schema = SchemaExtractor(engine, ExtractionConfig(sample_rows=100)).extract()
    profile = schema.table("orders").column("status").profile
    assert profile.is_sampled and profile.rows_profiled == 100
    assert schema.table("orders").row_count == 1000


def test_json_round_trip(schema, tmp_path):
    path = tmp_path / "schema.json"
    schema.save(path)
    assert DatabaseSchema.load(path) == schema


def test_cache_reused_until_structure_changes(tmp_path):
    path = build_sample_db(tmp_path / "ecommerce.db")
    engine = sa.create_engine(f"sqlite:///{path}")
    cache = tmp_path / "schema.json"

    _, from_cache = load_or_extract(engine, cache)
    assert not from_cache
    _, from_cache = load_or_extract(engine, cache)
    assert from_cache

    with engine.begin() as conn:
        conn.execute(sa.text("ALTER TABLE orders ADD COLUMN discount NUMERIC(10, 2)"))
    schema, from_cache = load_or_extract(engine, cache)
    assert not from_cache
    assert schema.table("orders").column("discount")


def test_candidate_table_names():
    assert candidate_table_names("customer")[:2] == ["customer", "customers"]
    assert "categories" in candidate_table_names("category")
    assert "customers" in candidate_table_names("parent_customer")


@pytest.mark.parametrize("name,expected", [
    ("email", True), ("contact_email", True), ("password_hash", True), ("full_name", True),
    ("date_of_birth", True), ("ip_address", True), ("phone", True),
    ("name", False), ("country", False), ("status", False), ("channel", False),
])
def test_pii_detection(name, expected):
    assert is_pii(name) is expected
