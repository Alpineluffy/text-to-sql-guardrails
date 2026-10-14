"""Estimate how many rows a query will read, from the database's own plan.

SQLite's EXPLAIN QUERY PLAN has no row counts, only steps like
"SCAN o" / "SEARCH c USING INTEGER PRIMARY KEY". We combine it with the row
counts from the Step 1 profile:

- SCAN of a table reads all its rows; SEARCH via an index reads ~1 per lookup.
- Steps under the same parent are nested loops, so rows multiply:
  SCAN orders (1,000) then SCAN customers (200) = 1,000 + 200,000.
- A correlated subquery runs once per row of the loop it sits in.

PostgreSQL's EXPLAIN (FORMAT JSON) has estimates already; sequential scans
are charged the table's full row count and nested-loop inner sides are
multiplied by the outer side's rows.

These are estimates for catching runaway queries, not precise costs; the
statement timeout in the sandbox is the hard backstop.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PlanRow = Tuple[int, int, str]  # (id, parent, detail)

_SCAN = re.compile(r"^SCAN (\S+)")
_SEARCH = re.compile(r"^SEARCH (\S+)")
_PG_SCANS = {"Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan",
             "Tid Scan", "Sample Scan"}


@dataclass
class ScanEstimate:
    rows: int
    plan: List[str]  # the plan as text, kept for the response and audit log
    notes: List[str] = field(default_factory=list)


def estimate_sqlite(
    plan: List[PlanRow],
    aliases: Dict[str, str],
    row_counts: Dict[str, int],
    derived_rows: Optional[Dict[str, int]] = None,
) -> ScanEstimate:
    """`aliases` maps names used in the query (aliases, table names) to tables;
    `derived_rows` gives estimated output sizes for CTEs/subqueries by name."""
    derived_rows = {k.lower(): v for k, v in (derived_rows or {}).items()}
    children: Dict[int, List[PlanRow]] = {}
    for row in plan:
        children.setdefault(row[1], []).append(row)
    largest = max(row_counts.values(), default=0)
    notes: List[str] = []

    def table_rows(name: str) -> int:
        if name.lower() in derived_rows:
            return derived_rows[name.lower()]
        table = aliases.get(name.lower(), name.lower())
        if table in row_counts:
            return row_counts[table]
        # A CTE, subquery or a name inside a view: assume the worst known table.
        notes.append(f"no row count for '{name}'; assumed {largest:,}")
        return largest

    def cost(parent: int) -> Tuple[int, int]:
        """(rows read by this level, rows it produces per execution)."""
        total, loop_rows = 0, 1
        for _id, _parent, detail in children.get(parent, []):
            scan, search = _SCAN.match(detail), _SEARCH.match(detail)
            if scan:
                n = table_rows(scan.group(1))
                loop_rows *= max(n, 1)
                total += loop_rows
            elif search:
                if "AUTOMATIC" in detail:  # SQLite builds a temporary index first
                    total += table_rows(search.group(1))
                total += loop_rows  # ~1 row per lookup, once per outer row
            elif detail.startswith("CORRELATED"):
                inner, _ = cost(_id)
                total += inner * loop_rows
            else:  # co-routines, materialised CTEs, compound selects, ...
                inner, _ = cost(_id)
                total += inner
        return total, loop_rows

    rows, _ = cost(0)
    return ScanEstimate(rows, [r[2] for r in plan], list(dict.fromkeys(notes)))


def estimate_postgres(plan_json: list, row_counts: Dict[str, int]) -> ScanEstimate:
    lines: List[str] = []

    def walk(node: dict, loops: float, depth: int) -> float:
        node_type = node.get("Node Type", "?")
        relation = node.get("Relation Name")
        lines.append("  " * depth + node_type + (f" on {relation}" if relation else ""))
        total = 0.0
        if node_type in _PG_SCANS:
            if node_type == "Seq Scan" and relation in row_counts:
                total += row_counts[relation] * loops
            else:
                total += float(node.get("Plan Rows", 0)) * loops
        kids = node.get("Plans", [])
        if node_type == "Nested Loop" and len(kids) == 2:
            total += walk(kids[0], loops, depth + 1)
            total += walk(kids[1], loops * float(kids[0].get("Plan Rows", 1)), depth + 1)
        else:
            for kid in kids:
                total += walk(kid, loops, depth + 1)
        return total

    root = plan_json[0]["Plan"] if plan_json and "Plan" in plan_json[0] else {}
    return ScanEstimate(int(walk(root, 1.0, 0)), lines)
