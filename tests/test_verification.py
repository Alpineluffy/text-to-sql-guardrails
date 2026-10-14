"""Phase 3: hallucination detection and confidence scoring."""
import json
from datetime import date
from pathlib import Path

import pytest
import sqlalchemy as sa

from text2sql.execution import ClarificationResponse, create_service
from text2sql.generation import SQLResponse
from text2sql.prompt import Knowledge, PromptBuilder
from text2sql.sample_data import build_sample_db
from text2sql.sandbox import SandboxedExecutor
from text2sql.schema import SchemaExtractor
from text2sql.verification import (
    BackTranslator, CoverageChecker, MultiQueryChecker, ResultSanityChecker, compare_results,
    complexity, score_confidence,
)
from text2sql.verification.backtranslate import COMPARE_SCHEMA, DESCRIBE_SCHEMA
from text2sql.verification.coverage import CoverageReport, Expectation
from text2sql.verification.evaluate import load_eval, run_eval
from text2sql.verification.multiquery import MultiQueryCheck
from text2sql.verification.sanity import SanityFlag, SanityReport

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_DIR = ROOT / "knowledge" / "ecommerce"
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
def executor(engine):
    return SandboxedExecutor(engine)


# --- Step 1: back-translation ------------------------------------------------------

class FakeLLM:
    def __init__(self, describe=None, compare=None):
        self.calls = []
        self.describe = describe or {
            "question": "How many orders were cancelled?", "measures": ["count of orders"],
            "filters": ["status = 'CANCELLED'"], "grouping": [], "time_range": "all time",
            "ordering_limit": "none"}
        self.compare = compare or {
            "aspects": [{"aspect": "measure", "verdict": "match", "explanation": "count"}],
            "alignment": 0.95, "summary": "answers it"}

    def complete(self, system, user, schema):
        self.calls.append((system, user, schema))
        return (self.describe if schema is DESCRIBE_SCHEMA else self.compare), "claude-opus-5-5"


def test_back_translation_never_shows_the_question_to_the_describer():
    llm = FakeLLM()
    result = BackTranslator(llm).check(
        "How many orders got cancelled?", "SELECT COUNT(*) FROM orders WHERE status = 'CANCELLED'",
        "CREATE TABLE orders (...)", today=TODAY)
    describe_call, compare_call = llm.calls
    assert describe_call[2] is DESCRIBE_SCHEMA and compare_call[2] is COMPARE_SCHEMA
    assert "got cancelled" not in describe_call[1]  # blind: SQL and schema only
    assert "How many orders got cancelled?" in compare_call[1]
    assert "Last quarter (Q3 2026)" in compare_call[1]  # dates resolved for the judge
    assert result.aligned and result.alignment == 0.95 and result.mismatches == []


def test_back_translation_reports_specific_mismatches():
    llm = FakeLLM(compare={
        "aspects": [{"aspect": "time_range", "verdict": "mismatch",
                     "explanation": "asked for last quarter, SQL covers 2025"},
                    {"aspect": "measure", "verdict": "match", "explanation": "sum"}],
        "alignment": 0.9, "summary": "wrong period"})
    result = BackTranslator(llm).check("q", "SELECT 1", "", today=TODAY)
    assert not result.aligned  # a mismatch fails even with a high alignment number
    assert result.mismatches == ["time range: asked for last quarter, SQL covers 2025"]


# --- Step 2: result sanity checks --------------------------------------------------

def sanity(schema, executor, sql):
    r = executor.run(sql)
    return ResultSanityChecker(schema, executor).check(sql, r.columns, r.rows, r.truncated)


@pytest.mark.parametrize("sql,check,fragment", [
    ("SELECT COUNT(*) AS n FROM orders WHERE status = 'delivered'", "filter_values",
     "did you mean 'DELIVERED'?"),
    ("SELECT SUM(total_amount) AS s FROM orders WHERE status <> 'cancelled'", "filter_values",
     "excludes nothing"),
    ("SELECT COUNT(*) AS n FROM products WHERE name = 'Laptop Pro 15'", "filter_values",
     "'Laptop Pro 15' was not found in products.name"),
    ("SELECT COUNT(*) AS n FROM orders WHERE order_date >= '2027-01-01'", "date_filters",
     "but the data ends 2026-09-30"),
    ("SELECT COUNT(*) AS n FROM orders WHERE order_date BETWEEN '2020-01-01' AND '2020-12-31'",
     "date_filters", "outside the data"),
    ("SELECT ROUND(SUM(o.total_amount), 2) AS revenue FROM orders o "
     "JOIN order_items oi ON oi.order_id = o.id", "join_fanout",
     "(2,565 joined rows for 1,000 orders)"),
    ("SELECT SUM(total_amount) * 3 AS s FROM orders", None, None),
    ("SELECT COUNT(*) AS n FROM orders o, customers c", "count_magnitude",
     "a join is multiplying rows"),
    ("SELECT 100.0 * COUNT(*) AS cancel_pct FROM orders", "percentage_range",
     "between 0 and 100"),
    ("SELECT c.country FROM customers c JOIN orders o ON o.customer_id = c.id",
     "duplicate_rows", "exact duplicates"),
    ("SELECT AVG(total_amount) * 1000 AS avg_total FROM orders", None, None),
])
def test_sanity_flags(schema, executor, sql, check, fragment):
    report = sanity(schema, executor, sql)
    if check is None:  # arithmetic on an aggregate isn't compared with column ranges
        assert report.flags == []
        return
    assert [f.check for f in report.flags] == [check], report.flags
    assert fragment in report.flags[0].message


def test_null_heavy_columns_after_a_bad_outer_join(schema, executor):
    # Joining on the wrong key: almost nothing matches, so country is mostly NULL.
    report = sanity(schema, executor,
                    "SELECT o.id, c.country FROM orders o "
                    "LEFT JOIN customers c ON c.id = o.id + 500 LIMIT 50")
    assert any(f.check == "null_heavy" and "customers.country" in f.message for f in report.flags)


def test_correct_examples_raise_no_sanity_flags(schema, executor, knowledge):
    for example in knowledge.examples:
        report = sanity(schema, executor, example.sql)
        assert report.flags == [], (example.question, report.flags)
        assert report.pass_rate == 1.0


# --- Step 3: multi-query validation ------------------------------------------------

def test_complexity():
    assert complexity("SELECT COUNT(*) FROM orders") is None
    assert "reads 3 tables" in complexity(
        "SELECT 1 FROM a JOIN b ON a.x = b.x JOIN c ON c.y = b.y")
    assert "subqueries" in complexity("SELECT COUNT(*) FROM (SELECT 1 FROM orders) t")


@pytest.mark.parametrize("a_cols,a,b_cols,b,agree", [
    (["n"], [(5,)], ["count"], [(5,)], True),
    (["x", "n"], [("a", 1), ("b", 2)], ["x", "n"], [("b", 2), ("a", 1)], True),  # order
    (["v"], [(1828861.92,)], ["v"], [(1828861.9200000002,)], True),  # rounding
    (["country"], [("India",)], ["country", "n"], [("India", 60)], True),  # extra column
    (["v"], [(2.56,)], ["v"], [(2.57,)], True),  # half-rounding differs by database
    (["c", "v"], [("a", 2.56), ("b", 7.0)], ["c", "v"], [("b", 7.0), ("a", 2.57)], True),
    (["v"], [(2.56,)], ["v"], [(2.6,)], False),
    (["v"], [(1828861.92,)], ["v"], [(1737088.95,)], False),
    (["x"], [("a",), ("b",)], ["x"], [("a",)], False),
])
def test_compare_results(a_cols, a, b_cols, b, agree):
    assert compare_results(a_cols, a, False, b_cols, b, False).agree is agree


def test_truncated_results_are_not_compared():
    assert compare_results(["x"], [(1,)], True, ["x"], [(1,)], False).agree is None


def test_disagreement_is_explained():
    c = compare_results(["v"], [(100.0,)], False, ["v"], [(90.0,)], False)
    assert c.explanation == "the approaches disagree: 100.00 vs 90.00 (10.0% apart)"


def _response(sql):
    return SQLResponse.from_json({
        "sql": sql, "explanation": "alt", "tables_used": [], "columns_used": [],
        "assumptions": [], "confidence": 0.7, "needs_clarification": False,
        "clarification": None})


class Recorder:
    def __init__(self, sql):
        self.sql, self.prompts = sql, []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return _response(self.sql)


PRIMARY = ("SELECT p.category, SUM(oi.quantity) AS units FROM order_items oi "
           "JOIN orders o ON o.id = oi.order_id JOIN products p ON p.id = oi.product_id "
           "WHERE o.status <> 'CANCELLED' GROUP BY p.category")
SAME_DIFFERENT_WAY = (
    "WITH q AS (SELECT oi.product_id, SUM(oi.quantity) AS u FROM order_items oi "
    "WHERE oi.order_id IN (SELECT id FROM orders WHERE status <> 'CANCELLED') "
    "GROUP BY oi.product_id) "
    "SELECT p.category, SUM(q.u) FROM q JOIN products p ON p.id = q.product_id GROUP BY p.category")


def test_second_approach_is_independent_and_agrees(schema, knowledge, executor):
    builder = PromptBuilder(schema, knowledge)
    primary = executor.run(PRIMARY)
    recorder = Recorder(SAME_DIFFERENT_WAY)
    check = MultiQueryChecker(builder, recorder, executor).check(
        "Units sold per category", PRIMARY, primary.columns, primary.rows, primary.truncated,
        today=TODAY)
    assert check.ran and check.agreed and "reads 3 tables" in check.reason
    prompt = recorder.prompts[0].user
    assert "## Examples" not in prompt  # no shared examples with the first attempt
    assert "## Approach" in prompt and "Aggregate each table in its own CTE" in prompt
    assert PRIMARY not in prompt  # the first SQL is never shown


def test_second_approach_disagreement(schema, knowledge, executor):
    builder = PromptBuilder(schema, knowledge)
    primary = executor.run(PRIMARY)
    wrong = PRIMARY.replace("WHERE o.status <> 'CANCELLED' ", "")  # forgot the filter
    check = MultiQueryChecker(builder, Recorder(wrong), executor).check(
        "Units sold per category", PRIMARY, primary.columns, primary.rows, False, today=TODAY)
    assert check.ran and check.agreed is False and "rows differ" in check.explanation
    assert check.alternative_rows


def test_simple_queries_are_not_double_checked(schema, knowledge, executor):
    builder = PromptBuilder(schema, knowledge)
    recorder = Recorder("SELECT 1")
    check = MultiQueryChecker(builder, recorder, executor).check(
        "How many orders?", "SELECT COUNT(*) FROM orders", ["n"], [(1000,)], False)
    assert not check.ran and recorder.prompts == []


# --- Step 4: coverage and confidence -----------------------------------------------

@pytest.fixture(scope="module")
def coverage(schema, knowledge):
    return CoverageChecker(schema, knowledge)


def test_examples_cover_their_questions(coverage, knowledge):
    for example in knowledge.examples:
        report = coverage.check(example.question, example.sql)
        assert report.missing == [], (example.question, report.missing)


@pytest.mark.parametrize("question,sql,missing", [
    ("How many orders from India?", "SELECT COUNT(*) FROM orders", "filters customers.country"),
    ("What was net revenue?", "SELECT SUM(total_amount) FROM orders WHERE status <> 'CANCELLED'",
     "applies the definition of net revenue"),
    ("How many orders were placed last quarter?", "SELECT COUNT(*) FROM orders",
     'filters on a date (the question says "last quarter")'),
    ("Top 5 products by price", "SELECT name FROM products ORDER BY unit_price DESC",
     "returns the top 5"),
])
def test_coverage_catches_what_the_sql_leaves_out(coverage, question, sql, missing):
    report = coverage.check(question, sql)
    assert any(e.description.startswith(missing) for e in report.missing), report.missing


def _sanity(errors=0, clean_checks=3):
    flags = [SanityFlag("count_magnitude", "error", "too many")] * errors
    return SanityReport(flags=flags, checks_run=["count_magnitude"] + ["x"] * (clean_checks - 1))


def _bt(aligned=True, alignment=0.95):
    from text2sql.verification.backtranslate import BackTranslation
    return BackTranslation("q?", {}, [], alignment, "", aligned,
                           [] if aligned else ["time range: wrong year"])


def test_everything_passing_is_high():
    c = score_confidence(backtranslation=_bt(), sanity=_sanity(),
                         multiquery=MultiQueryCheck(True, "complex", agreed=True),
                         coverage=CoverageReport([Expectation("value", "x", True)]),
                         model_confidence=0.9)
    assert c.band == "high" and c.problems == []
    assert any("second query agrees" in s for s in c.support)


@pytest.mark.parametrize("kwargs,band,cap", [
    (dict(sanity=_sanity(errors=1)), "low", "sanity-check error"),
    (dict(multiquery=MultiQueryCheck(True, "x", agreed=False, explanation="90 vs 100")),
     "low", "approaches disagree"),
    (dict(backtranslation=_bt(aligned=False, alignment=0.9)), "medium",
     "back-translation mismatch"),
    (dict(coverage=CoverageReport([Expectation("time", "filters on a date", False)])),
     "medium", "missing coverage"),
])
def test_serious_findings_override_the_average(kwargs, band, cap):
    base = dict(backtranslation=_bt(), sanity=_sanity(), coverage=CoverageReport(),
                model_confidence=1.0)
    c = score_confidence(**{**base, **kwargs})
    assert c.band == band and cap in c.caps and c.problems


def test_unverified_answers_cannot_be_high():
    c = score_confidence(sanity=_sanity(), model_confidence=1.0)
    assert c.band == "medium" and "not independently verified" in c.caps


# --- end to end through the service ------------------------------------------------

class GoldGenerator:
    """Answers each eval question with its gold SQL, except the ones in `wrong`."""

    def __init__(self, gold, wrong):
        self.gold, self.wrong = gold, wrong

    def generate(self, prompt):
        question = prompt.user.split("## Question\n")[1].split("\n")[0]
        sql = self.wrong.get(question) or self.gold.get(question)
        if sql is None:  # the multi-query second approach: reuse the gold SQL
            sql = next(s for q, s in self.gold.items() if q in prompt.user)
        return _response(sql)


class AgreeingLLM(FakeLLM):
    """Back-translation that says the SQL matches the question."""


def _service(engine, tmp_path, generator):
    return create_service(engine, knowledge_dir=KNOWLEDGE_DIR, use_filter=False,
                          generator=generator, verifier_llm=AgreeingLLM(), today=TODAY,
                          log_dir=tmp_path / "logs", schema_cache_dir=tmp_path / "schema")


def test_every_eval_answer_runs(executor):
    today, cases = load_eval(KNOWLEDGE_DIR / "eval.yaml")
    assert today == TODAY and len(cases) == 45
    for case in cases:
        assert executor.run(case.sql).rows, case.question


def test_service_attaches_confidence_and_logs_it(engine, tmp_path):
    gold = {"How many orders were cancelled?":
            "SELECT COUNT(*) FROM orders WHERE status = 'CANCELLED'"}
    service = _service(engine, tmp_path, GoldGenerator(gold, {}))
    r = service.ask("How many orders were cancelled?")
    assert r.ok and r.confidence_band == "high" and r.rows == [(92,)]
    assert r.verification.coverage.expectations[0].satisfied  # 'CANCELLED' used
    record = service.log.records()[0]
    assert record["confidence_band"] == "high"
    assert json.dumps(record)  # whole verification report is JSON-safe


def test_eval_harness_scores_right_and_wrong_answers(engine, tmp_path, executor):
    _, cases = load_eval(KNOWLEDGE_DIR / "eval.yaml")
    gold = {c.question: c.sql for c in cases}
    wrong = {
        # wrong case: matches nothing
        "How many orders were cancelled?":
            "SELECT COUNT(*) FROM orders WHERE status = 'cancelled'",
        # join fan-out double counts revenue
        "What is the gross revenue from customers in Germany?":
            "SELECT ROUND(SUM(o.total_amount), 2) FROM orders o "
            "JOIN customers c ON c.id = o.customer_id JOIN order_items oi ON oi.order_id = o.id "
            "WHERE c.country = 'Germany' AND o.status <> 'CANCELLED'",
        # forgot the date filter
        "What was net revenue last quarter?":
            "SELECT ROUND(SUM(total_amount), 2) FROM orders "
            "WHERE status NOT IN ('CANCELLED', 'RETURNED')",
    }
    service = _service(engine, tmp_path, GoldGenerator(gold, wrong))
    report = run_eval(service, cases, executor)

    assert len(report.results) == 45
    assert report.accuracy == pytest.approx(42 / 45)
    misses = {r.question: r for r in report.results if not r.correct}
    assert set(misses) == set(wrong)
    assert report.detection == 1.0  # every wrong answer was kept out of "high"
    assert misses["How many orders were cancelled?"].band == "low"
    assert misses["What is the gross revenue from customers in Germany?"].band == "low"
    assert misses["What was net revenue last quarter?"].band == "medium"  # coverage: no date
    assert report.by_band()["high"]["accuracy"] == 1.0
    assert report.false_alarms == 0.0
