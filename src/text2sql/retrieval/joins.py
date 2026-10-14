"""Add the bridge tables needed to join the selected tables together.

"Revenue by category for Japanese customers" matches `customers` and
`products`, but joining them needs `orders` and `order_items`, which may score
low on their own. Without them the model either fails or invents a join.
"""
from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from ..schema.models import DatabaseSchema


def fk_graph(schema: DatabaseSchema) -> Dict[str, Set[str]]:
    graph: Dict[str, Set[str]] = {t.name: set() for t in schema.tables}
    for fk in schema.foreign_keys:
        if fk.source_table != fk.target_table:
            graph[fk.source_table].add(fk.target_table)
            graph[fk.target_table].add(fk.source_table)
    return graph


def _path_to_group(
    graph: Dict[str, Set[str]], start: str, group: Set[str]
) -> Optional[List[str]]:
    """Shortest path from `start` to any table in `group` (BFS, alphabetical ties)."""
    parent: Dict[str, Optional[str]] = {start: None}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node in group:
            path = [node]
            while parent[path[-1]] is not None:
                path.append(parent[path[-1]])
            return path
        for nxt in sorted(graph[node]):
            if nxt not in parent:
                parent[nxt] = node
                queue.append(nxt)
    return None


def connect(
    schema: DatabaseSchema, tables: List[str]
) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Join the tables into one connected group.

    Returns (bridges, unreachable): each bridge is (added_table, from, to),
    and unreachable lists selected tables no FK path can reach. Tables with
    no relationships at all (typically views) stand alone and are skipped.
    """
    graph = fk_graph(schema)
    joinable = [t for t in tables if graph.get(t)]
    if not joinable:
        return [], []
    group = {joinable[0]}
    bridges: List[Tuple[str, str, str]] = []
    unreachable: List[str] = []
    for table in joinable[1:]:
        if table in group:
            continue
        path = _path_to_group(graph, table, group)
        if path is None:
            unreachable.append(table)
            continue
        # path runs from a table already in the group back to `table`.
        for node in path[1:-1]:
            if node not in group and node not in tables:
                bridges.append((node, table, path[0]))
        group.update(path)
    return bridges, unreachable
