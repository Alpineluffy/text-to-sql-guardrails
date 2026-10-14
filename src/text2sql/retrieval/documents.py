"""Turn tables and columns into short texts worth embedding.

Raw names like `total_amount` embed poorly, so each document spells names out
and adds descriptions, synonyms and real values:

    orders: One row per customer order. Also called purchase, sale.
    Columns: order date, status, channel, total amount
    orders.status (text): Fulfilment status of the order. Values: DELIVERED, SHIPPED...
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..prompt.knowledge import Knowledge, TableNotes
from ..schema.models import ColumnInfo, DatabaseSchema, TableInfo

MAX_VALUES_IN_DOC = 12


def humanize(name: str) -> str:
    return name.replace("_", " ").strip()


@dataclass(frozen=True)
class Document:
    table: str
    column: str  # "" for the table-level document
    text: str


def build_documents(schema: DatabaseSchema, knowledge: Knowledge) -> List[Document]:
    docs: List[Document] = []
    for table in schema.tables:
        notes = knowledge.tables.get(table.name, TableNotes())
        docs.append(Document(table.name, "", _table_text(table, notes)))
        for col in table.columns:
            if col.name in table.primary_key:
                continue  # "id" says nothing about what a question needs
            docs.append(Document(table.name, col.name, _column_text(table, col, notes)))
    return docs


def _table_text(table: TableInfo, notes: TableNotes) -> str:
    parts = [f"{humanize(table.name)}: {notes.description or table.comment or ''}".strip()]
    if notes.synonyms:
        parts.append("Also called " + ", ".join(notes.synonyms) + ".")
    columns = [humanize(c.name) for c in table.columns if c.name not in table.primary_key]
    parts.append("Columns: " + ", ".join(columns))
    return " ".join(parts)


def _column_text(table: TableInfo, col: ColumnInfo, notes: TableNotes) -> str:
    text = f"{humanize(table.name)} {humanize(col.name)} ({col.type_family})"
    description = notes.columns.get(col.name) or col.comment
    if description:
        text += f": {description}"
    p = col.profile
    if p and p.is_categorical and not col.is_pii:
        values = [str(v) for v in p.sample_values[:MAX_VALUES_IN_DOC]]
        text += ". Values: " + ", ".join(values)
    return text
