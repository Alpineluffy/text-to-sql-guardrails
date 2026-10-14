"""The Streamlit UI, driven headlessly against a real API server (demo mode)."""
import os
import socket
import threading
import time
from pathlib import Path

import pytest
import sqlalchemy as sa
import uvicorn

from text2sql.api.app import create_app
from text2sql.api.demo import ReplayGenerator
from text2sql.execution import Text2SQLApp
from text2sql.feedback import FeedbackStore
from text2sql.sample_data import build_sample_db

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge" / "ecommerce"
APP_FILE = str(ROOT / "ui" / "streamlit_app.py")

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    engine = sa.create_engine(f"sqlite:///{build_sample_db(tmp / 'ecommerce.db')}")
    replay = ReplayGenerator(KNOWLEDGE)
    text2sql = Text2SQLApp(engine, knowledge_dir=KNOWLEDGE, use_filter=False, generator=replay,
                           back_translate=False, multi_query="auto", log_dir=tmp / "logs",
                           schema_cache_dir=tmp / "schema", history_dir=tmp / "history")
    replay.validator = text2sql.generator.validator
    store = FeedbackStore(tmp / "feedback.jsonl")
    app = create_app(text2sql, store, demo=True)

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    os.environ["TEXT2SQL_API_URL"] = f"http://127.0.0.1:{port}"
    yield store
    server.should_exit = True
    thread.join(timeout=5)


def start():
    at = AppTest.from_file(APP_FILE, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    return at


def ask(at, question):
    at.text_input[0].input(question)
    at.button[0].click()  # the form's "Ask" button
    at.run()
    assert not at.exception, at.exception
    return at


def markdown(at):
    return "\n".join(m.value for m in at.markdown)


def test_first_load_shows_demo_banner_and_empty_history(api):
    at = start()
    assert "Demo mode" in "\n".join(i.value for i in at.info)
    assert at.title[0].value == "Ask your database"


def test_answer_shows_sql_results_and_confidence(api):
    at = ask(start(), "Which country has the most customers?")
    assert "Confidence: :orange[MEDIUM]" in markdown(at)
    assert any("ORDER BY COUNT(*) DESC" in c.value for c in at.code)
    assert at.dataframe[0].value.iloc[0, 0] == "India"


def test_clarification_choose_and_answer(api):
    at = ask(start(), "What is our total revenue?")
    assert at.warning[0].value == "Which definition do you mean?"
    radio = at.radio[0]
    assert radio.options == ["gross revenue: uses the gross revenue definition",
                             "net revenue: uses the net revenue definition"]
    radio.set_value(1).run()
    next(b for b in at.button if b.label == "Use this meaning").click().run()
    assert not at.exception, at.exception
    assert not at.radio  # the clarification is gone, the answer is shown
    assert at.dataframe[0].value.iloc[0, 0] == pytest.approx(1737088.95)  # net revenue


def test_feedback_is_sent(api):
    store = api
    at = ask(start(), "How many orders were cancelled?")
    before = len(store.items())
    next(b for b in at.button if b.label == "👍 Correct").click().run()
    assert not at.exception, at.exception
    assert "feedback was recorded" in "\n".join(s.value for s in at.success)
    assert len(store.items()) == before + 1


def test_api_down_shows_a_clear_error(api, monkeypatch):
    import streamlit as st

    monkeypatch.setenv("TEXT2SQL_API_URL", "http://127.0.0.1:9")  # nothing listens here
    st.cache_resource.clear()  # the HTTP client is cached per process
    try:
        at = AppTest.from_file(APP_FILE, default_timeout=30)
        at.run()
        assert "Can't reach the API at http://127.0.0.1:9" in at.error[0].value
        assert "python -m text2sql.api" in at.error[0].value
    finally:
        st.cache_resource.clear()


def test_multi_query_discrepancy_can_be_resolved(api):
    at = ask(start(), "What is the gross revenue from customers in Germany?")
    assert "Confidence: :red[LOW]" in markdown(at)
    assert "An independent second query disagrees" in markdown(at)
    inflated = at.dataframe[0].value.iloc[0, 0]
    next(b for b in at.button if b.label == "Use the second approach").click().run()
    assert not at.exception, at.exception
    resolved = at.dataframe[0].value.iloc[0, 0]
    assert resolved == pytest.approx(156580.81) and resolved < inflated
    assert "Confidence: :red[LOW]" not in markdown(at)  # the alternative passes the checks
