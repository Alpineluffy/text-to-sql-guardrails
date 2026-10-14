"""Create and verify the SELECT-only PostgreSQL role generated queries run as.

    # once, as an admin (prints nothing secret; the password comes from the env)
    export TEXT2SQL_READER_PASSWORD=...
    python -m text2sql.sandbox.postgres setup --admin-url postgresql://postgres:...@host/db

    # any time, as the reader: checks the role really is locked down
    python -m text2sql.sandbox.postgres verify --db-url postgresql://text2sql_reader:...@host/db

The role is the database's own line of defence: even a query that gets past
every guardrail can only read the columns granted here, can't write, and is
stopped by the statement timeout.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from ..guardrails import GuardrailConfig
from ..schema.models import DatabaseSchema
from .executor import SandboxConfig

DEFAULT_ROLE = "text2sql_reader"


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def role_setup_sql(
    schema: DatabaseSchema,
    database: str,
    role: str = DEFAULT_ROLE,
    password: Optional[str] = None,
    guardrails: GuardrailConfig = GuardrailConfig(),
    sandbox: SandboxConfig = SandboxConfig(),
) -> List[str]:
    """Statements that (re)create the reader role with least privilege.

    Safe to re-run: existing grants are revoked first, then re-granted.
    """
    r = quote_ident(role)
    ns = quote_ident(schema.namespace or "public")
    stmts = [
        f"DO $$ BEGIN CREATE ROLE {r}; EXCEPTION WHEN duplicate_object THEN NULL; END $$",
        f"ALTER ROLE {r} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION "
        f"NOBYPASSRLS NOINHERIT CONNECTION LIMIT 10"
        + (f" PASSWORD {quote_literal(password)}" if password else ""),
        # Defaults that apply to every session, even if the app forgets to set them.
        f"ALTER ROLE {r} SET default_transaction_read_only = on",
        f"ALTER ROLE {r} SET statement_timeout = {int(sandbox.statement_timeout_ms)}",
        f"ALTER ROLE {r} SET lock_timeout = {int(sandbox.lock_timeout_ms)}",
        f"ALTER ROLE {r} SET idle_in_transaction_session_timeout = 30000",
        # Start from nothing, then grant only what's needed.
        f"REVOKE ALL ON ALL TABLES IN SCHEMA {ns} FROM {r}",
        f"REVOKE CREATE ON SCHEMA {ns} FROM {r}",
        f"GRANT CONNECT ON DATABASE {quote_ident(database)} TO {r}",
        f"GRANT USAGE ON SCHEMA {ns} TO {r}",
    ]
    blocked = {t.lower() for t in guardrails.blocked_tables}
    allowed = ({t.lower() for t in guardrails.allowed_tables}
               if guardrails.allowed_tables is not None else None)
    for table in schema.tables:
        name = table.name.lower()
        if name in blocked or (allowed is not None and name not in allowed):
            continue
        target = f"{ns}.{quote_ident(table.name)}"
        hidden = [c for c in table.columns if c.is_pii and guardrails.block_sensitive_columns]
        if not hidden:
            stmts.append(f"GRANT SELECT ON {target} TO {r}")
            continue
        visible = [quote_ident(c.name) for c in table.columns if c not in hidden]
        if visible:  # column-level grant: personal data stays unreadable
            stmts.append(f"GRANT SELECT ({', '.join(visible)}) ON {target} TO {r}")
    return stmts


def create_reader_role(admin_engine: Engine, schema: DatabaseSchema, **kwargs) -> List[str]:
    stmts = role_setup_sql(schema, admin_engine.url.database, **kwargs)
    with admin_engine.begin() as conn:
        for stmt in stmts:
            conn.exec_driver_sql(stmt)
    return stmts


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


def verify_reader(engine: Engine, schema: DatabaseSchema) -> List[Check]:
    """Connect as the reader and prove it can read what it should and nothing more."""
    checks: List[Check] = []

    def scalar(sql: str):
        with engine.connect() as conn:
            try:
                return conn.exec_driver_sql(sql).scalar()
            finally:
                conn.rollback()

    def refused(sql: str, read_write: bool = False) -> Optional[str]:
        """The error message if the database refuses `sql`, else None."""
        with engine.connect() as conn:
            try:
                if read_write:  # prove the *permissions* stop it, not just read-only mode
                    conn.exec_driver_sql("SET TRANSACTION READ WRITE")
                conn.exec_driver_sql(sql)
                return None
            except sa.exc.DBAPIError as e:
                return str(e.orig).strip().splitlines()[0]
            finally:
                conn.rollback()

    flags = scalar("SELECT rolsuper OR rolcreaterole OR rolcreatedb OR rolbypassrls "
                   "FROM pg_roles WHERE rolname = current_user")
    checks.append(Check("not a superuser, can't create roles or databases", flags is False))
    checks.append(Check("read-only by default",
                        scalar("SHOW default_transaction_read_only") == "on"))
    timeout = scalar("SHOW statement_timeout")
    checks.append(Check("statement timeout set", timeout not in (None, "0"), f"= {timeout}"))

    err = refused("CREATE TABLE text2sql_probe (x int)", read_write=True)
    checks.append(Check("can't create tables", err is not None and "permission denied" in err,
                        err or "CREATE TABLE succeeded"))

    real_tables = [t for t in schema.tables if t.kind == "table"]
    if real_tables:
        t = quote_ident(real_tables[0].name)
        err = refused(f"DELETE FROM {t} WHERE false", read_write=True)
        checks.append(Check("can't write data", err is not None and "permission denied" in err,
                            err or "DELETE succeeded"))

    for table in schema.tables:
        for col in table.columns:
            if col.is_pii:
                err = refused(f"SELECT {quote_ident(col.name)} FROM {quote_ident(table.name)} LIMIT 0")
                checks.append(Check(f"can't read {table.name}.{col.name}", err is not None,
                                    err or "column is readable"))
        visible = next((c for c in table.columns if not c.is_pii), None)
        if visible is not None:
            err = refused(f"SELECT {quote_ident(visible.name)} FROM {quote_ident(table.name)} LIMIT 1")
            checks.append(Check(f"can read {table.name}", err is None, err or ""))

    for builtin in ("pg_read_server_files", "pg_write_server_files", "pg_execute_server_program"):
        member = scalar(f"SELECT pg_has_role(current_user, '{builtin}', 'MEMBER')")
        checks.append(Check(f"not a member of {builtin}", member is False))
    can_read_files = scalar("SELECT has_function_privilege('pg_read_file(text)', 'EXECUTE')")
    checks.append(Check("can't call pg_read_file()", can_read_files is False))
    return checks


def main() -> None:
    from ..schema import ExtractionConfig, SchemaExtractor

    parser = argparse.ArgumentParser(description="Set up or verify the SELECT-only role.")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("setup", help="create/refresh the reader role (run as admin)")
    setup.add_argument("--admin-url", required=True)
    setup.add_argument("--role", default=DEFAULT_ROLE)
    setup.add_argument("--guardrails", type=Path, default=Path("config/guardrails.yaml"))
    setup.add_argument("--load-sample", action="store_true",
                       help="first load the sample e-commerce data into this database")
    setup.add_argument("--print-only", action="store_true", help="show the SQL, change nothing")
    verify = sub.add_parser("verify", help="check the role, connecting as the role")
    verify.add_argument("--db-url", required=True)
    args = parser.parse_args()

    if args.command == "setup":
        admin = sa.create_engine(args.admin_url)
        if args.load_sample and not args.print_only:
            from ..sample_data import load_into

            load_into(admin)
            print("Loaded the sample data.")
        password = os.environ.get("TEXT2SQL_READER_PASSWORD")
        if not password and not args.print_only:
            sys.exit("Set TEXT2SQL_READER_PASSWORD to the password for the reader role.")
        config = (GuardrailConfig.from_yaml(args.guardrails)
                  if args.guardrails.exists() else GuardrailConfig())
        schema = SchemaExtractor(admin, ExtractionConfig(infer_foreign_keys=False)).extract_structure()
        kwargs = dict(role=args.role, password=password, guardrails=config)
        if args.print_only:
            for stmt in role_setup_sql(schema, admin.url.database, **{**kwargs, "password": None}):
                print(stmt + ";")
            return
        create_reader_role(admin, schema, **kwargs)
        print(f"Role {args.role} is ready. Check it with:\n"
              f"  python -m text2sql.sandbox.postgres verify --db-url <reader url>")
    else:
        engine = sa.create_engine(args.db_url)
        schema = SchemaExtractor(engine, ExtractionConfig(infer_foreign_keys=False)).extract_structure()
        checks = verify_reader(engine, schema)
        for c in checks:
            print(f"  {'PASS' if c.passed else 'FAIL'}  {c.name}" + (f"  ({c.detail})" if c.detail else ""))
        failed = [c for c in checks if not c.passed]
        print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
