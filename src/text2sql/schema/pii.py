"""Column-name rules for sensitive data.

Values from matching columns are never collected, so they can't leak into an
LLM prompt. Name matching is a first line of defence only; extend the patterns
(or pass your own through ExtractionConfig) for your database.
"""
from __future__ import annotations

import re
from typing import Iterable

DEFAULT_PII_PATTERNS = (
    r"e_?mail",
    r"phone",
    r"^mobile$|mobile_?(no|num|number)",
    r"pass(word)?$|passwd|pwd",
    r"(^|_)ssn($|_)|social_security",
    r"(first|last|full|middle|given|family)_?name",
    r"birth|(^|_)dob($|_)",
    r"address|street|zip_?code|postal",
    r"card_?(no|num|number)|cvv|iban",
    r"account_?(no|num|number)",
    r"passport|national_?id|tax_?id",
    r"token|secret|api_?key|hash",
    r"ip_?addr",
)


def is_pii(column_name: str, patterns: Iterable[str] = DEFAULT_PII_PATTERNS) -> bool:
    name = column_name.lower()
    return any(re.search(p, name) for p in patterns)
