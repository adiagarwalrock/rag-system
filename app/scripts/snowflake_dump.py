"""
Dumps every table from a Snowflake database to a SQLite database and a
Markdown schema document.

Usage:
    uv run python -m app.scripts.snowflake_dump [--out-dir PATH]

Reads from .env: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_TOKEN,
SNOWFLAKE_DATABASE, SNOWFLAKE_SCHEMA, SNOWFLAKE_WAREHOUSE, SNOWFLAKE_ROLE.

Override the database with SNOWFLAKE_DATABASE env var:
    SNOWFLAKE_DATABASE=VECTERA_DEMO uv run python -m app.scripts.snowflake_dump --out-dir sf_dump/VECTERA_DEMO

Output layout per --out-dir:
    dump.sqlite         — all tables loaded into SQLite
    schema.md           — Markdown description of every table and its columns
    _manifest.json      — row counts + dump timestamp
"""

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import snowflake.connector

from app.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _connect() -> snowflake.connector.SnowflakeConnection:
    required = {
        "SNOWFLAKE_ACCOUNT": settings.SNOWFLAKE_ACCOUNT,
        "SNOWFLAKE_USER": settings.SNOWFLAKE_USER,
        "SNOWFLAKE_TOKEN": settings.SNOWFLAKE_TOKEN,
        "SNOWFLAKE_DATABASE": settings.SNOWFLAKE_DATABASE,
        "SNOWFLAKE_SCHEMA": settings.SNOWFLAKE_SCHEMA,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log.error("Missing required .env vars: %s", ", ".join(missing))
        sys.exit(1)

    kwargs: dict = {
        "account": settings.SNOWFLAKE_ACCOUNT,
        "user": settings.SNOWFLAKE_USER,
        "authenticator": "programmatic_access_token",
        "token": settings.SNOWFLAKE_TOKEN,
        "database": settings.SNOWFLAKE_DATABASE,
        "schema": settings.SNOWFLAKE_SCHEMA,
    }
    if settings.SNOWFLAKE_WAREHOUSE:
        kwargs["warehouse"] = settings.SNOWFLAKE_WAREHOUSE
    if settings.SNOWFLAKE_ROLE:
        kwargs["role"] = settings.SNOWFLAKE_ROLE

    log.info(
        "Connecting  account=%s  database=%s  schema=%s",
        settings.SNOWFLAKE_ACCOUNT,
        settings.SNOWFLAKE_DATABASE,
        settings.SNOWFLAKE_SCHEMA,
    )
    return snowflake.connector.connect(**kwargs)


def _all_tables(cur) -> list[str]:
    cur.execute("SHOW TABLES")
    return [row[1] for row in cur.fetchall()]


def _fetch_table(cur, table: str) -> tuple[list[str], list[tuple]]:
    cur.execute(f'SELECT * FROM "{table}"')  # noqa: S608
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return columns, rows


def _col_ddl(columns: list[str]) -> str:
    return ", ".join(f'"{c}" TEXT' for c in columns)


def _write_sqlite(
    db: sqlite3.Connection,
    table: str,
    columns: list[str],
    rows: list[tuple],
) -> None:
    db.execute(f'CREATE TABLE IF NOT EXISTS "{table}" ({_col_ddl(columns)})')
    placeholders = ", ".join("?" * len(columns))
    db.executemany(f'INSERT INTO "{table}" VALUES ({placeholders})', rows)  # noqa: S608
    db.commit()


def _write_schema_md(
    database: str,
    schema: str,
    table_schemas: dict[str, tuple[list[str], int]],
    out_dir: Path,
) -> None:
    lines = [
        f"# {database}.{schema} — Schema",
        "",
        f"_Dumped at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}_",
        "",
        "## Tables",
        "",
    ]
    for table, (columns, row_count) in sorted(table_schemas.items()):
        lines += [
            f"### `{table}`",
            "",
            f"**Rows:** {row_count}",
            "",
            "| Column | Type |",
            "| --- | --- |",
        ]
        for col in columns:
            lines.append(f"| `{col}` | TEXT |")
        lines.append("")

    (out_dir / "schema.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump all Snowflake tables to CSV + SQLite + Markdown schema"
    )
    parser.add_argument(
        "--out-dir",
        default="data/snowflake_dump",
        help="Output directory (default: data/snowflake_dump)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sqlite_path = out_dir / "dump.sqlite"
    db = sqlite3.connect(sqlite_path)

    conn = _connect()
    manifest: dict[str, int] = {}
    table_schemas: dict[str, tuple[list[str], int]] = {}

    try:
        cur = conn.cursor()
        tables = _all_tables(cur)

        log.info("Writing to %s", out_dir.resolve())
        for table in tables:
            columns, rows = _fetch_table(cur, table)
            _write_sqlite(db, table, columns, rows)
            manifest[table] = len(rows)
            table_schemas[table] = (columns, len(rows))
            log.info("  %-35s %d rows", table, len(rows))
    finally:
        conn.close()
        db.close()

    _write_schema_md(
        settings.SNOWFLAKE_DATABASE or "UNKNOWN",
        settings.SNOWFLAKE_SCHEMA or "PUBLIC",
        table_schemas,
        out_dir,
    )

    manifest_data = {
        "dumped_at": datetime.now(timezone.utc).isoformat(),
        "account": settings.SNOWFLAKE_ACCOUNT,
        "database": settings.SNOWFLAKE_DATABASE,
        "schema": settings.SNOWFLAKE_SCHEMA,
        "tables": manifest,
        "total_rows": sum(manifest.values()),
    }
    (out_dir / "_manifest.json").write_text(
        json.dumps(manifest_data, indent=2), encoding="utf-8"
    )

    log.info(
        "Done. %d tables, %d total rows -> %s  (SQLite: %s  Schema: %s)",
        len(manifest),
        manifest_data["total_rows"],
        out_dir,
        sqlite_path.name,
        "schema.md",
    )


if __name__ == "__main__":
    main()
