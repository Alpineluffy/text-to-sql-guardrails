"""The structure the LLM must answer with.

Passed to the API as a structured-output JSON schema, which guarantees the
reply parses. Structured outputs require `additionalProperties: false` on
every object and don't support numeric bounds, so the 0-1 range for
confidence is stated in its description and checked after parsing.
"""
from __future__ import annotations

_NULLABLE_STRING = {"anyOf": [{"type": "string"}, {"type": "null"}]}

INTERPRETATION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "description": "Short name; use the glossary definition name when there is one.",
        },
        "description": {"type": "string"},
        "sql": {"type": "string"},
    },
    "required": ["label", "description", "sql"],
    "additionalProperties": False,
}

CLARIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string", "description": "What to ask the user."},
        "interpretations": {"type": "array", "items": INTERPRETATION_SCHEMA},
    },
    "required": ["question", "interpretations"],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sql": {
            **_NULLABLE_STRING,
            "description": "One SELECT statement, or null if the question can't be "
                           "answered or needs clarification.",
        },
        "explanation": {
            "type": "string",
            "description": "Plain-English description of what the query returns.",
        },
        "tables_used": {"type": "array", "items": {"type": "string"}},
        "columns_used": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every column referenced, as table.column.",
        },
        "assumptions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Interpretations made that the user should know about.",
        },
        "confidence": {"type": "number", "description": "Between 0 and 1."},
        "needs_clarification": {"type": "boolean"},
        "clarification": {"anyOf": [CLARIFICATION_SCHEMA, {"type": "null"}]},
    },
    "required": [
        "sql", "explanation", "tables_used", "columns_used",
        "assumptions", "confidence", "needs_clarification", "clarification",
    ],
    "additionalProperties": False,
}
