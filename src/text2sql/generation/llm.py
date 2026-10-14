"""Call Claude with a built prompt and parse the structured reply.

- Structured outputs (`output_config.format`) guarantee the reply is JSON
  matching RESPONSE_SCHEMA, so there is no fragile text parsing.
- The system prompt is identical for every question on a database, so it is
  marked for prompt caching.
- Server-side fallback is on: if the model declines a request, the API
  retries it on a fallback model within the same call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ..prompt.builder import Prompt
from ..prompt.response import RESPONSE_SCHEMA

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class GenerationError(RuntimeError):
    pass


@dataclass
class Interpretation:
    label: str
    description: str
    sql: str
    validation: Any = None  # ValidationReport, set by ValidatingGenerator
    guardrails: Any = None  # GuardrailDecision, set by ValidatingGenerator


@dataclass
class SQLResponse:
    sql: Optional[str]
    explanation: str
    tables_used: List[str]
    columns_used: List[str]
    assumptions: List[str]
    confidence: float
    needs_clarification: bool
    clarification_question: Optional[str] = None
    interpretations: List[Interpretation] = field(default_factory=list)
    model: Optional[str] = None  # which model actually answered (fallbacks may change it)
    validation: Any = None  # ValidationReport for `sql`, set by ValidatingGenerator
    guardrails: Any = None  # GuardrailDecision for `sql`, set by ValidatingGenerator
    attempts: int = 1  # generations needed, counting repairs

    @classmethod
    def from_json(cls, data: dict, model: Optional[str] = None) -> SQLResponse:
        clarification = data.get("clarification") or {}
        return cls(
            sql=data.get("sql") or None,
            explanation=data.get("explanation", ""),
            tables_used=list(data.get("tables_used", [])),
            columns_used=list(data.get("columns_used", [])),
            assumptions=list(data.get("assumptions", [])),
            # The schema can't enforce a numeric range, so clamp here.
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.0)))),
            needs_clarification=bool(data.get("needs_clarification")),
            clarification_question=clarification.get("question"),
            interpretations=[
                Interpretation(i["label"], i["description"], i["sql"])
                for i in clarification.get("interpretations", [])
            ],
            model=model,
        )


class ClaudeJSON:
    """One structured-output call to Claude: system + user text in, parsed JSON out.

    Shared by SQL generation and the Phase 3 verification calls, so every call
    gets the same caching, fallback and error handling.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        effort: str = "high",
        max_tokens: int = 16_000,
        use_fallbacks: bool = True,
    ):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks

    def complete(self, system: str, user: str, schema: dict) -> Tuple[dict, Optional[str]]:
        """Returns (parsed JSON, the model that actually answered)."""
        params = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            # Same system prompt for every call of a kind, so it's cached.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        if self.use_fallbacks:
            params.update(betas=[FALLBACK_BETA], fallbacks="default")
        response = self.client.beta.messages.create(**params)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise GenerationError(f"the model declined this request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise GenerationError("the response hit max_tokens before finishing")

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise GenerationError(f"no text in response (stop_reason={response.stop_reason})")
        try:
            return json.loads(text), getattr(response, "model", None)
        except json.JSONDecodeError as e:
            raise GenerationError(f"response was not valid JSON: {e}") from e


class SQLGenerator(ClaudeJSON):
    """SQL generation: effort defaults to high, since correctness is reasoning-heavy."""

    def generate(self, prompt: Prompt) -> SQLResponse:
        data, model = self.complete(prompt.system, prompt.user, RESPONSE_SCHEMA)
        return SQLResponse.from_json(data, model=model)
