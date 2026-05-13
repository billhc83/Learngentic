"""
One-time migration: local SQLite → Turso.
Run from the project root: python migrate_to_turso.py
"""
import math
import sqlite3
import sys
from pathlib import Path

import requests as _requests

sys.path.insert(0, str(Path(__file__).parent / "src"))

from learngentic.store.db import (
    _TursoConnection, _load_config,
    _SCHEMA_STATEMENTS, _MIGRATIONS,
)

LOCAL_DB = Path.home() / ".learngentic" / "learngentic.db"
TABLES = ["sessions", "change_events", "interventions", "global_patterns"]

# pattern_embedding can exceed Turso's per-request payload limit.
# Leave it NULL — it will be populated by new Hermes sessions.
_SKIP_COLS = {"pattern_embedding"}


def _safe_arg(value) -> dict:
    """Like _to_arg but sanitises edge-case floats."""
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return {"type": "null"}
        return {"type": "float", "value": value}  # JSON number, not string
    return {"type": "text", "value": str(value)}


def _exec_with_fk_off(turso: _TursoConnection, sql: str, params: tuple) -> None:
    """
    Send PRAGMA foreign_keys=OFF + the INSERT in a single pipeline request.
    Each Turso HTTP call is a fresh connection, so the PRAGMA must travel
    in the same request as the statement it is meant to affect.
    """
    body = {
        "requests": [
            {"type": "execute", "stmt": {"sql": "PRAGMA foreign_keys=OFF"}},
            {
                "type": "execute",
                "stmt": {
                    "sql": sql.strip(),
                    "args": [_safe_arg(p) for p in params],
                },
            },
            {"type": "close"},
        ]
    }
    resp = _requests.post(turso._url, json=body, headers=turso._headers, timeout=30)
    if not resp.ok:
        raise Exception(f"HTTP {resp.status_code}: {resp.text[:800]}")
    results = resp.json().get("results", [])
    # results[0] is PRAGMA (always ok), results[1] is the INSERT
    if len(results) > 1 and results[1].get("type") == "error":
        raise Exception(results[1]["error"]["message"])


def _migrate_table(local: sqlite3.Connection, turso: _TursoConnection, table: str) -> int:
    local.row_factory = sqlite3.Row
    rows = local.execute(f"SELECT * FROM {table}").fetchall()
    if not rows:
        return 0

    cols = [c for c in rows[0].keys() if c not in _SKIP_COLS]
    placeholders = ", ".join(["?"] * len(cols))
    col_list = ", ".join(cols)
    sql = f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})"

    ok = 0
    for row in rows:
        try:
            _exec_with_fk_off(turso, sql, tuple(row[c] for c in cols))
            ok += 1
        except Exception as e:
            msg = str(e)
            vals = {c: repr(row[c])[:60] for c in cols if row[c] is not None}
            print(f"  skip {table} row: {msg}")
            print(f"    values: {vals}")
    return ok


def main():
    if not LOCAL_DB.exists():
        print(f"No local DB found at {LOCAL_DB}")
        print("Nothing to migrate.")
        return

    print(f"Source: {LOCAL_DB}")

    url, token = _load_config()
    print(f"Target: {url}\n")

    turso = _TursoConnection(url, token)

    print("Initialising Turso schema...")
    for stmt in _SCHEMA_STATEMENTS:
        try:
            turso.execute(stmt)
        except Exception:
            pass
    for stmt in _MIGRATIONS:
        try:
            turso.execute(stmt)
        except Exception:
            pass

    local = sqlite3.connect(str(LOCAL_DB))

    for table in TABLES:
        try:
            n = _migrate_table(local, turso, table)
            print(f"  {table}: {n} rows")
        except Exception as e:
            print(f"  {table}: skipped ({e})")

    local.close()
    print("\nDone. Your data is now in Turso.")


if __name__ == "__main__":
    main()
