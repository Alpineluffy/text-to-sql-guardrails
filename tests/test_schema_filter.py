import os
from pathlib import Path

import numpy as np
import pytest
import sqlalchemy as sa

from text2sql.prompt import Knowledge
from text2sql.retrieval import (
    CachedEmbedder, EmbeddingExampleSelector, FilterConfig, HashingEmbedder, SchemaFilter,
)
from text2sql.retrieval.joins import connect
from text2sql.sample_data import build_sample_db
from text2sql.schema import SchemaExtractor

KNOWLEDGE_DIR = Path(__file__).resolve().parents[1] / "knowledge" / "ecommerce"
EXACT_ONLY = FilterConfig(max_tables=0)  # isolate the exact-match signals


@pytest.fixture(scope="module")
def schema(tmp_path_factory):
    path = build_sample_db(tmp_path_factory.mktemp("db") / "ecommerce.db")
    return SchemaExtractor(sa.create_engine(f"sqlite:///{path}")).extract()


@pytest.fixture(scope="module")
def knowledge():
    return Knowledge.load(KNOWLEDGE_DIR)


def make_filter(schema, knowledge, config=FilterConfig()):
    return SchemaFilter(schema, knowledge, HashingEmbedder(), config)


# --- exact-match evidence -----------------------------------------------------

def test_synonyms_and_values_select_tables(schema, knowledge):
    sel = make_filter(schema, knowledge, EXACT_ONLY).select("How many purchases came from Germany?")
    assert sel.tables == ["customers", "orders"]
    assert "value 'Germany' in country" in sel.reasons["customers"]
    assert "mentions 'purchase'" in sel.reasons["orders"]


def test_glossary_term_selects_its_tables(schema, knowledge):
    sel = make_filter(schema, knowledge, EXACT_ONLY).select("What's our AOV?")
    assert sel.tables == ["orders"]
    assert sel.reasons["orders"] == ["glossary 'average order value'"]


def test_longer_phrase_shadows_shorter_synonym(schema, knowledge):
    # "line items" means order_items; the "item" inside it must not pull in products.
    sel = make_filter(schema, knowledge, EXACT_ONLY).select("How many line items per order?")
    assert "order_items" in sel.tables
    assert "products" not in sel.tables


def test_column_mentions_skip_views(schema, knowledge):
    sel = make_filter(schema, knowledge, EXACT_ONLY).select("Count by country")
    assert sel.tables == ["customers"]  # not the order_summary view


def test_bridge_tables_connect_the_selection(schema, knowledge):
    sel = make_filter(schema, knowledge, EXACT_ONLY).select(
        "Gross revenue by product category for customers in Japan"
    )
    assert set(sel.tables) == {"customers", "orders", "order_items", "products"}
    assert sel.reasons["order_items"][0].startswith("bridge: joins ")
    assert not sel.unreachable


# --- similarity and fallbacks -------------------------------------------------

def test_similarity_adds_tables_with_no_exact_match(schema, knowledge):
    sel = make_filter(schema, knowledge, FilterConfig(min_score=0.1)).select(
        "breakdown of fulfilment"
    )
    assert "orders" in sel.tables
    assert sel.reasons["orders"][0].startswith("similarity")


def test_max_tables_caps_similarity_matches(schema, knowledge):
    loose = FilterConfig(max_tables=1, min_score=0.0, relative_to_best=0.0, add_bridge_tables=False)
    sel = make_filter(schema, knowledge, loose).select("breakdown of fulfilment")
    assert len(sel.tables) == 1


def test_nothing_matched_falls_back_to_full_schema(schema, knowledge):
    sel = make_filter(schema, knowledge).select("zzz qqq")
    assert sel.used_full_schema
    assert len(sel.tables) == len(schema.tables)


def test_small_databases_skip_filtering(schema, knowledge):
    sel = make_filter(schema, knowledge, FilterConfig(full_schema_below=10)).select("AOV")
    assert sel.used_full_schema and len(sel.tables) == 6


def test_explain_lists_selected_and_rejected(schema, knowledge):
    text = make_filter(schema, knowledge, EXACT_ONLY).select("What's our AOV?").explain()
    assert "+ orders" in text and "- support_tickets" in text


# --- join paths ---------------------------------------------------------------

def test_connect_finds_bridges_and_ignores_views(schema):
    bridges, unreachable = connect(schema, ["order_summary", "customers", "products"])
    assert {b[0] for b in bridges} == {"orders", "order_items"}
    assert unreachable == []


def test_connect_adds_nothing_for_adjacent_tables(schema):
    assert connect(schema, ["orders", "customers"]) == ([], [])


# --- embeddings ---------------------------------------------------------------

class CountingEmbedder(HashingEmbedder):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def embed(self, texts, is_query=False):
        self.calls += len(texts)
        return super().embed(texts, is_query)


def test_cached_embedder_reuses_document_vectors(tmp_path):
    first = CountingEmbedder()
    vectors = CachedEmbedder(first, tmp_path).embed(["orders table", "customers table"])
    assert first.calls == 2

    second = CountingEmbedder()
    again = CachedEmbedder(second, tmp_path).embed(["customers table", "orders table"])
    assert second.calls == 0
    assert np.allclose(again, vectors[::-1])

    CachedEmbedder(second, tmp_path).embed(["question"], is_query=True)
    assert second.calls == 1  # queries are never cached


def test_embedding_example_selector(knowledge):
    selector = EmbeddingExampleSelector(knowledge.examples, HashingEmbedder())
    top = selector.select("top products by units sold", 1)
    assert top[0].question == "What are the top 5 products by units sold?"


# --- the real model on the labelled set (slow: ~30s to load) ------------------

@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("RUN_SLOW"), reason="set RUN_SLOW=1")
def test_real_model_covers_linking_eval(schema, knowledge):
    from text2sql.retrieval import SentenceTransformerEmbedder
    from text2sql.retrieval.evaluate import evaluate, load_cases

    schema_filter = SchemaFilter(schema, knowledge, SentenceTransformerEmbedder())
    report = evaluate(schema_filter, load_cases(KNOWLEDGE_DIR / "linking_eval.yaml"))
    assert report.covered == report.total, report.misses
    assert report.avg_selected < len(schema.tables)
