"""Human-readable report of an extracted schema, for checking what was found."""
from __future__ import annotations

from typing import List

from .models import DatabaseSchema


def _short(values: list, limit: int = 8) -> str:
    shown = ", ".join(repr(v) for v in values[:limit])
    return shown + (f", ... (+{len(values) - limit})" if len(values) > limit else "")


def format_summary(schema: DatabaseSchema) -> str:
    lines: List[str] = [
        f"Database: {schema.database or '-'} ({schema.dialect})"
        f"   hash={schema.structure_hash}   extracted={schema.extracted_at}",
        "",
        "Tables",
    ]
    for t in schema.tables:
        rows = "?" if t.row_count is None else f"{t.row_count:,}"
        pk = ", ".join(t.primary_key) or "-"
        lines.append(f"  {t.name:<20} {t.kind:<5} rows={rows:<8} pk={pk}")

    lines += ["", "Relationships"]
    for fk in schema.foreign_keys:
        note = "declared" if fk.origin == "declared" else f"inferred, match={fk.match_ratio}"
        lines.append(f"  {fk.describe():<50} [{note}]")

    categorical, indexed, ranges, pii = [], [], [], []
    for t in schema.tables:
        for c in t.columns:
            ref = f"{t.name}.{c.name}"
            p = c.profile
            if c.is_pii:
                pii.append(ref)
            if not p:
                continue
            if p.is_categorical:
                categorical.append(f"  {ref:<28} [{p.case_style}] {_short(p.sample_values)}")
            if p.value_index:
                more = "+" if p.value_index_truncated else ""
                indexed.append(f"  {ref:<28} {len(p.value_index)}{more} values")
            if p.min_value is not None:
                ranges.append(f"  {ref:<28} {p.min_value} .. {p.max_value}")

    lines += ["", "Categorical values (sent to the prompt)"] + (categorical or ["  -"])
    lines += ["", "Value index (for keyword lookup)"] + (indexed or ["  -"])
    lines += ["", "Ranges"] + (ranges or ["  -"])
    lines += ["", "PII columns (values never collected)", "  " + (", ".join(pii) or "-")]
    return "\n".join(lines)
