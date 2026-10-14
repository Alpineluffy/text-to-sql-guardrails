"""Build a small e-commerce SQLite database for development and tests.

It contains the traps the schema extractor is meant to handle:
- order_items.product_id and support_tickets.customer_id have no FK constraint
- products.supplier_id looks like a key but there is no suppliers table
- orders.status is UPPERCASE, orders.channel is lowercase
- customers holds PII (full_name, email, phone)

    python -m text2sql.sample_data [path]
"""
from __future__ import annotations

import random
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

DDL = """
CREATE TABLE customers (
    id          INTEGER PRIMARY KEY,
    full_name   VARCHAR(100) NOT NULL,
    email       VARCHAR(150) NOT NULL UNIQUE,
    phone       VARCHAR(30),
    country     VARCHAR(60) NOT NULL,
    segment     VARCHAR(30) NOT NULL,
    created_at  DATETIME NOT NULL
);
CREATE TABLE products (
    id          INTEGER PRIMARY KEY,
    name        VARCHAR(100) NOT NULL,
    category    VARCHAR(40) NOT NULL,
    unit_price  NUMERIC(10, 2) NOT NULL,
    supplier_id INTEGER
);
CREATE TABLE orders (
    id              INTEGER PRIMARY KEY,
    customer_id     INTEGER NOT NULL REFERENCES customers(id),
    order_date      DATE NOT NULL,
    status          VARCHAR(20) NOT NULL,
    channel         VARCHAR(20) NOT NULL,
    total_amount    NUMERIC(12, 2) NOT NULL,
    external_ref_id VARCHAR(40)
);
CREATE TABLE order_items (
    id          INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL,
    quantity    INTEGER NOT NULL,
    unit_price  NUMERIC(10, 2) NOT NULL
);
CREATE TABLE support_tickets (
    ticket_id   INTEGER PRIMARY KEY,
    customer_id INTEGER,
    opened_at   DATETIME NOT NULL,
    priority    VARCHAR(10) NOT NULL,
    subject     VARCHAR(200) NOT NULL,
    resolved    BOOLEAN NOT NULL
);
CREATE VIEW order_summary AS
    SELECT o.id AS order_id, c.country, o.order_date, o.status, o.total_amount
    FROM orders o JOIN customers c ON c.id = o.customer_id;
"""

FIRST = ["Aarav", "Priya", "Liam", "Emma", "Noah", "Olivia", "Lukas", "Mia", "Hiro",
         "Yuki", "Lucas", "Sofia", "Arjun", "Ananya", "James", "Charlotte"]
LAST = ["Sharma", "Patel", "Smith", "Johnson", "Müller", "Schmidt", "Tanaka", "Sato",
        "Silva", "Santos", "Brown", "Wilson", "Reddy", "Iyer"]
COUNTRIES = [("India", 30), ("United States", 25), ("Germany", 10), ("United Kingdom", 10),
             ("Brazil", 8), ("Japan", 7), ("Canada", 6), ("Australia", 4)]
SEGMENTS = [("Consumer", 60), ("Corporate", 25), ("Small Business", 15)]
PRODUCTS = {
    "Electronics": ["Laptop Pro 14", "Laptop Air 13", "4K Monitor 27", "Wireless Mouse",
                    "Mechanical Keyboard", "USB-C Hub", "Noise Cancelling Headphones",
                    "Webcam HD", "Portable SSD 1TB"],
    "Furniture": ["Ergonomic Chair", "Standing Desk", "Bookshelf Oak", "Filing Cabinet",
                  "Monitor Arm", "Desk Lamp"],
    "Office Supplies": ["A4 Paper Ream", "Gel Pens (12)", "Sticky Notes", "Stapler",
                        "Whiteboard Markers", "Notebook Pack", "Desk Organizer"],
    "Accessories": ["Laptop Sleeve", "Phone Stand", "Cable Organizer", "Mouse Pad XL",
                    "Laptop Backpack", "Screen Cleaner", "HDMI Cable 2m", "Power Bank"],
}
PRICE_RANGE = {"Electronics": (25, 1800), "Furniture": (30, 650),
               "Office Supplies": (3, 40), "Accessories": (8, 90)}
STATUSES = [("DELIVERED", 60), ("SHIPPED", 15), ("PENDING", 10), ("CANCELLED", 10),
            ("RETURNED", 5)]
CHANNELS = [("web", 55), ("mobile", 35), ("store", 10)]
PRIORITIES = [("low", 35), ("medium", 40), ("high", 20), ("urgent", 5)]
ISSUES = ["Damaged on arrival", "Late delivery", "Wrong item received",
          "Refund request", "Warranty claim", "Missing parts"]


def _pick(rng: random.Random, weighted):
    values, weights = zip(*weighted)
    return rng.choices(values, weights=weights)[0]


def build_sample_db(path: Path, seed: int = 42) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    rng = random.Random(seed)
    conn = sqlite3.connect(path)
    conn.executescript(DDL)

    customers = []
    for cid in range(1, 201):
        first, last = rng.choice(FIRST), rng.choice(LAST)
        created = datetime(2023, 6, 1) + timedelta(minutes=rng.randint(0, 60 * 24 * 900))
        customers.append((
            cid, f"{first} {last}", f"{first}.{last}{cid}@example.com".lower(),
            f"+1-555-{rng.randint(1000000, 9999999)}" if rng.random() > 0.2 else None,
            _pick(rng, COUNTRIES), _pick(rng, SEGMENTS), created.strftime("%Y-%m-%d %H:%M:%S"),
        ))
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?,?)", customers)

    products, pid = [], 0
    for category, names in PRODUCTS.items():
        low, high = PRICE_RANGE[category]
        for name in names:
            pid += 1
            products.append((pid, name, category, round(rng.uniform(low, high), 2),
                             rng.randint(1, 5)))
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", products)

    item_id = 0
    start = date(2024, 1, 1)
    span = (date(2026, 9, 30) - start).days
    for oid in range(1, 1001):
        items = []
        for _ in range(rng.randint(1, 4)):
            item_id += 1
            product = rng.choice(products)
            items.append((item_id, oid, product[0], rng.randint(1, 5), product[3]))
        total = round(sum(q * p for _, _, _, q, p in items), 2)
        conn.execute(
            "INSERT INTO orders VALUES (?,?,?,?,?,?,?)",
            (oid, rng.randint(1, 200), (start + timedelta(days=rng.randint(0, span))).isoformat(),
             _pick(rng, STATUSES), _pick(rng, CHANNELS), total,
             f"EXT-{rng.getrandbits(32):08X}" if rng.random() > 0.3 else None),
        )
        conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", items)

    tickets = []
    for tid in range(1, 151):
        opened = datetime(2024, 1, 1) + timedelta(minutes=rng.randint(0, 60 * 24 * 1000))
        tickets.append((
            tid, rng.randint(1, 200) if rng.random() > 0.05 else None,
            opened.strftime("%Y-%m-%d %H:%M:%S"), _pick(rng, PRIORITIES),
            f"{rng.choice(ISSUES)} - {rng.choice(products)[1]}", int(rng.random() > 0.3),
        ))
    conn.executemany("INSERT INTO support_tickets VALUES (?,?,?,?,?,?)", tickets)

    conn.commit()
    conn.close()
    return path


TABLE_ORDER = ["customers", "products", "orders", "order_items", "support_tickets"]


def load_into(engine, seed: int = 42) -> None:
    """Load the same sample data into another database (e.g. PostgreSQL).

    Replaces the sample tables if they exist. The rows are identical to
    build_sample_db(), so tests and examples behave the same on both.
    """
    import tempfile

    import sqlalchemy as sa

    with tempfile.TemporaryDirectory() as tmp:
        source = sqlite3.connect(build_sample_db(Path(tmp) / "sample.db", seed))
        data = {t: source.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in TABLE_ORDER}
        source.close()

    ddl = DDL.replace("DATETIME", "TIMESTAMP")
    with engine.begin() as conn:
        conn.exec_driver_sql("DROP VIEW IF EXISTS order_summary")
        for table in reversed(TABLE_ORDER):
            conn.exec_driver_sql(f"DROP TABLE IF EXISTS {table} CASCADE")
        for stmt in ddl.split(";"):
            if stmt.strip():
                conn.exec_driver_sql(stmt)
        meta = sa.MetaData()
        meta.reflect(conn, only=TABLE_ORDER)
        for name in TABLE_ORDER:
            table = meta.tables[name]
            cols = [c.name for c in table.columns]
            bools = {c.name for c in table.columns if isinstance(c.type, sa.Boolean)}
            rows = [{c: (bool(v) if c in bools and v is not None else v)
                     for c, v in zip(cols, row)} for row in data[name]]
            conn.execute(table.insert(), rows)


if __name__ == "__main__":
    out = build_sample_db(Path(sys.argv[1] if len(sys.argv) > 1 else "data/ecommerce.db"))
    print(f"Sample database written to {out}")
