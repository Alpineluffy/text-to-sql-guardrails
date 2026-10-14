"""Guardrail settings. Every rule can be tuned or switched off (None/False).

Load from YAML with GuardrailConfig.from_yaml("config/guardrails.yaml").
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import yaml

# Functions that can read files, sleep, lock, change settings or exhaust
# resources, across SQLite, PostgreSQL and MySQL. Matched case-insensitively.
DEFAULT_BLOCKED_FUNCTIONS: Tuple[str, ...] = (
    # SQLite
    "load_extension", "readfile", "writefile", "edit", "fts3_tokenizer",
    "randomblob", "zeroblob",
    # PostgreSQL
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file",
    "pg_ls_dir", "pg_stat_file", "lo_import", "lo_export", "lo_get", "lo_put",
    "dblink", "dblink_exec", "dblink_connect", "pg_terminate_backend", "pg_cancel_backend",
    "pg_reload_conf", "pg_rotate_logfile", "set_config", "nextval", "setval",
    "pg_advisory_lock", "pg_advisory_xact_lock", "pg_try_advisory_lock",
    "query_to_xml", "query_to_json", "pg_execute_server_program",
    # MySQL
    "sleep", "benchmark", "load_file", "get_lock", "release_lock", "sys_exec", "sys_eval",
)


@dataclass(frozen=True)
class GuardrailConfig:
    # Row limit: add LIMIT when missing, lower it when larger. None = off.
    max_rows: Optional[int] = 1_000
    # Deepest allowed nesting of SELECTs inside SELECTs (CTE bodies count). None = off.
    max_subquery_depth: Optional[int] = 3
    # Block queries whose plan is estimated to read more rows than this. None = off.
    max_scan_rows: Optional[int] = 1_000_000
    blocked_functions: Tuple[str, ...] = DEFAULT_BLOCKED_FUNCTIONS
    # Table access: blocked tables are never readable; if allowed_tables is set,
    # only those are.
    blocked_tables: Tuple[str, ...] = ()
    allowed_tables: Optional[Tuple[str, ...]] = None
    # Reject queries that read columns flagged as personal data (Step 1 PII rules).
    block_sensitive_columns: bool = True
    # Reject joins with no join condition (a cartesian product).
    block_cartesian_joins: bool = True

    @classmethod
    def from_dict(cls, data: dict) -> GuardrailConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown guardrail settings: {', '.join(unknown)}")
        values = {}
        for key, value in data.items():
            if key in ("blocked_functions", "blocked_tables") and value is not None:
                value = tuple(value)
            if key == "allowed_tables" and value is not None:
                value = tuple(value)
            values[key] = value
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: Path) -> GuardrailConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()) or {})
