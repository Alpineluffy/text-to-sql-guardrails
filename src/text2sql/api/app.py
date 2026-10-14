"""Phase 4 Step 1: the HTTP API.

  POST /v1/query                      question -> SQL, results, confidence, warnings
  POST /v1/query/{request_id}/choose  answer a clarifying question
  POST /v1/sql                        run SQL a user edited (same safety layers)
  GET  /v1/schema                     tables, columns, relationships, glossary
  GET  /v1/history                    this session's past queries and results
  POST /v1/feedback                   mark a result correct / incorrect
  GET  /v1/feedback                   this session's feedback
  GET  /v1/health

Sessions: the client sends a session_id (any short id it makes up; one is
issued if missing). Clarification choices, pending questions and history are
per session. No authentication: run it on localhost or behind your own auth.
"""
from __future__ import annotations

import re
import threading
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..execution import ClarificationResponse, QueryResponse, Text2SQLApp
from ..feedback import FeedbackStore
from ..schema.profiler import to_jsonable

SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_HISTORY_ROWS = 200  # per history entry, to keep responses small


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1_000)
    session_id: Optional[str] = None


class ChooseRequest(BaseModel):
    session_id: str
    option: int = Field(..., ge=0)


class SQLRequest(BaseModel):
    sql: str = Field(..., min_length=1, max_length=20_000)
    session_id: Optional[str] = None
    question: Optional[str] = Field(None, max_length=1_000)


class FeedbackRequest(BaseModel):
    session_id: str
    request_id: str
    verdict: Literal["correct", "incorrect"]
    corrected_sql: Optional[str] = Field(None, max_length=20_000)
    comment: str = Field("", max_length=2_000)


def create_app(text2sql: Text2SQLApp, feedback: FeedbackStore, demo: bool = False) -> FastAPI:
    app = FastAPI(title="Text-to-SQL", version="1.0",
                  description="Natural-language questions to verified SQL answers.")
    # The Streamlit UI calls the API server-side; CORS only matters for a browser
    # frontend, so allow local origins only.
    app.add_middleware(CORSMiddleware, allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    session_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
    registry_lock = threading.Lock()

    def session_for(session_id: Optional[str]):
        session_id = session_id or uuid.uuid4().hex[:16]
        if not SESSION_ID.match(session_id):
            raise HTTPException(400, "session_id must be 1-64 letters, digits, '-' or '_'")
        with registry_lock:
            service = text2sql.session(session_id)
            lock = session_locks[session_id]
        return service, lock

    def payload(response: Any, session_id: str) -> dict:
        data = response.to_dict()
        data["session_id"] = session_id
        data["demo"] = demo
        return data

    @app.get("/v1/health")
    def health() -> dict:
        return {"status": "ok", "demo": demo, "dialect": text2sql.schema.dialect,
                "database": Path(text2sql.engine.url.database or "").name,
                "tables": len(text2sql.schema.tables)}

    @app.post("/v1/query")
    def query(req: QueryRequest) -> dict:
        service, lock = session_for(req.session_id)
        with lock:  # one question at a time per session: history and choices stay consistent
            response = service.ask(req.question.strip())
        return payload(response, service.session_id)

    @app.post("/v1/query/{request_id}/choose")
    def choose(request_id: str, req: ChooseRequest) -> dict:
        service, lock = session_for(req.session_id)
        with lock:
            try:
                response = service.choose(request_id, req.option)
            except KeyError:
                raise HTTPException(404, f"no clarification pending for request {request_id}")
            except IndexError as e:
                raise HTTPException(400, str(e))
        return payload(response, service.session_id)

    @app.post("/v1/sql")
    def run_sql(req: SQLRequest) -> dict:
        service, lock = session_for(req.session_id)
        with lock:
            response = service.run_sql(req.sql, question=req.question)
        return payload(response, service.session_id)

    @app.get("/v1/schema")
    def schema() -> dict:
        return describe_schema(text2sql)

    @app.get("/v1/history")
    def history(session_id: str, limit: int = Query(20, ge=1, le=50)) -> dict:
        service, lock = session_for(session_id)
        with lock:
            if text2sql.history_store is not None:  # survives restarts
                items = text2sql.history_store.load(service.session_id, limit)
            else:
                items = [r.to_dict() for r in list(service.history)[-limit:]]
        for data in items:
            if data.get("rows"):
                data["rows"] = data["rows"][:MAX_HISTORY_ROWS]
        return {"session_id": service.session_id, "items": items[::-1]}  # newest first

    @app.post("/v1/feedback")
    def give_feedback(req: FeedbackRequest) -> dict:
        service, lock = session_for(req.session_id)
        with lock:
            # Feedback must point at an answer this session actually received,
            # so the stored question and SQL are the system's, not the caller's.
            found = next((r.to_dict(include_rows=False) for r in reversed(service.history)
                          if isinstance(r, QueryResponse) and r.request_id == req.request_id),
                         None)
            if found is None and text2sql.history_store is not None:
                found = text2sql.history_store.find(service.session_id, req.request_id)
        if found is None:
            raise HTTPException(404, f"request {req.request_id} not found in this session")
        item = feedback.add(
            request_id=found["request_id"], question=found["question"], sql=found.get("sql"),
            verdict=req.verdict, session_id=service.session_id,
            corrected_sql=req.corrected_sql, comment=req.comment,
            confidence_band=found.get("confidence_band"),
        )
        return {**asdict(item), "kind": item.kind}

    @app.post("/v1/admin/reload")
    def reload(request: Request) -> dict:
        """Pick up promoted examples/test cases without a restart. Local callers only."""
        host = request.client.host if request.client else ""
        if host not in ("127.0.0.1", "::1", "localhost", "testclient"):
            raise HTTPException(403, "reload is only allowed from this machine")
        with registry_lock:
            text2sql.reload_knowledge()
        return {"status": "reloaded", "examples": len(text2sql.knowledge.examples)}

    @app.get("/v1/feedback")
    def list_feedback(session_id: str) -> dict:
        items = feedback.items(session_id=session_id)
        return {"items": [{**asdict(i), "kind": i.kind} for i in items]}

    return app


def describe_schema(text2sql: Text2SQLApp) -> dict:
    """The schema as users should see it: real values for categories, never PII."""
    schema, knowledge = text2sql.schema, text2sql.knowledge
    tables = []
    for t in schema.tables:
        notes = knowledge.tables.get(t.name)
        columns = []
        for c in t.columns:
            p = c.profile
            column: Dict[str, Any] = {
                "name": c.name, "type": c.type, "nullable": c.nullable,
                "primary_key": c.is_primary_key, "sensitive": c.is_pii,
                "description": (notes.columns.get(c.name) if notes else None) or c.comment,
            }
            if p is not None and not c.is_pii:
                if p.is_categorical:
                    column["values"] = [to_jsonable(v) for v in p.sample_values]
                if p.min_value is not None:
                    column["range"] = [to_jsonable(p.min_value), to_jsonable(p.max_value)]
            columns.append(column)
        tables.append({
            "name": t.name, "kind": t.kind, "row_count": t.row_count,
            "description": (notes.description if notes else None) or t.comment,
            "synonyms": notes.synonyms if notes else [], "columns": columns,
        })
    relationships = [{"join": fk.describe(), "origin": fk.origin} for fk in schema.foreign_keys]
    glossary = [{"term": g.term, "synonyms": g.synonyms,
                 "definitions": [{"name": d.name, "sql": d.sql} for d in g.definitions]}
                for g in knowledge.glossary]
    return {"dialect": schema.dialect, "tables": tables, "relationships": relationships,
            "glossary": glossary}
