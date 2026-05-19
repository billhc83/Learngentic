"""
Persistence layer backed by Turso (cloud SQLite via HTTP pipeline API).

Connection flow: _TursoConnection wraps the Turso /v2/pipeline endpoint
with a sqlite3-compatible interface so the rest of the codebase is unchanged.
Schema is initialised once per process; all writes are upserts.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Generator

import requests as _requests

_CONFIG_PATH = Path.home() / ".learngentic" / "config.json"

# Legacy stub kept for imports in vector_index.py and cli.py
DEFAULT_DB_PATH = Path.home() / ".learngentic" / "learngentic.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_STATEMENTS = [
    "PRAGMA foreign_keys=ON",
    """CREATE TABLE IF NOT EXISTS sessions (
        session_id              TEXT PRIMARY KEY,
        project_id              TEXT,
        cwd                     TEXT,
        git_branch              TEXT,
        started_at              TEXT,
        turn_count              INTEGER,
        initial_prompt          TEXT,
        outcome_quality         REAL,
        execution_efficiency    REAL,
        prompt_quality          REAL,
        durability              REAL,
        robustness_delta        REAL,
        change_type             TEXT,
        code_region_type        TEXT,
        failure_mode_tags       TEXT,
        had_intervention        INTEGER DEFAULT 0,
        intervention_id         TEXT,
        pattern_embedding       TEXT,
        ai_title                TEXT,
        agent_type              TEXT DEFAULT 'claude_code',
        hermes_assessment       REAL,
        assessment_source       TEXT DEFAULT 'pending',
        hermes_notes            TEXT,
        tool_calls              TEXT,
        checklist_compliance    REAL
    )""",
    """CREATE TABLE IF NOT EXISTS change_events (
        event_id                TEXT PRIMARY KEY,
        session_id              TEXT REFERENCES sessions(session_id),
        file_path               TEXT,
        repo_relative_path      TEXT,
        tool_name               TEXT,
        turn_index              INTEGER,
        event_timestamp         TEXT,
        days_until_next_touch   REAL,
        was_reverted            INTEGER,
        churn_count_30d         INTEGER,
        blast_radius_7d         INTEGER
    )""",
    """CREATE TABLE IF NOT EXISTS interventions (
        intervention_id         TEXT PRIMARY KEY,
        session_id              TEXT REFERENCES sessions(session_id),
        timestamp               TEXT,
        questions_surfaced      TEXT,
        user_engaged            INTEGER,
        additions_made          TEXT,
        outcome_delta           REAL,
        efficiency_delta        REAL
    )""",
    """CREATE TABLE IF NOT EXISTS global_patterns (
        change_type             TEXT,
        code_region_type        TEXT,
        avg_outcome_quality     REAL,
        avg_efficiency          REAL,
        avg_prompt_quality      REAL,
        avg_durability          REAL,
        common_failure_modes    TEXT,
        sample_count            INTEGER,
        last_updated            TEXT,
        PRIMARY KEY (change_type, code_region_type)
    )""",
    """CREATE TABLE IF NOT EXISTS local_model_runs (
        run_id          TEXT PRIMARY KEY,
        task_type       TEXT,
        task_description TEXT,
        system_prompt   TEXT,
        user_input      TEXT,
        model_output    TEXT,
        model_used      TEXT,
        attempt_number  INTEGER,
        previous_run_id TEXT,
        session_id      TEXT,
        verdict         TEXT,
        local_capable   INTEGER,
        created_at      TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_lmr_task_type ON local_model_runs(task_type)",
    "CREATE INDEX IF NOT EXISTS idx_lmr_session   ON local_model_runs(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_session ON change_events(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_ce_file    ON change_events(repo_relative_path)",
    "CREATE INDEX IF NOT EXISTS idx_s_project  ON sessions(project_id)",
    "CREATE INDEX IF NOT EXISTS idx_s_started  ON sessions(started_at)",
]

_MIGRATIONS = [
    "ALTER TABLE sessions ADD COLUMN ai_title TEXT",
    "ALTER TABLE sessions ADD COLUMN agent_type TEXT DEFAULT 'claude_code'",
    "ALTER TABLE sessions ADD COLUMN hermes_assessment REAL",
    "ALTER TABLE sessions ADD COLUMN assessment_source TEXT DEFAULT 'pending'",
    "ALTER TABLE sessions ADD COLUMN hermes_notes TEXT",
    "ALTER TABLE sessions ADD COLUMN tool_calls TEXT",
    "ALTER TABLE sessions ADD COLUMN checklist_compliance REAL",
]

_schema_ready = False  # initialise once per process

# ---------------------------------------------------------------------------
# Turso HTTP client
# ---------------------------------------------------------------------------

def _load_config() -> tuple[str, str]:
    try:
        cfg = json.loads(_CONFIG_PATH.read_text())
        url = cfg.get("turso_url", "")
        token = cfg.get("turso_auth_token", "")
        if not url or not token:
            raise KeyError
        return url, token
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise RuntimeError(
            f"Turso credentials missing from {_CONFIG_PATH}. "
            "Add turso_url and turso_auth_token."
        ) from exc


def _to_arg(value) -> dict:
    import math as _math
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "integer", "value": str(int(value))}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        if _math.isnan(value) or _math.isinf(value):
            return {"type": "null"}
        return {"type": "float", "value": value}  # JSON number, not string
    return {"type": "text", "value": str(value)}


def _from_val(cell: dict):
    t = cell.get("type")
    v = cell.get("value")
    if t == "null" or v is None:
        return None
    if t == "integer":
        return int(v)
    if t == "float":
        return float(v)
    return v  # text / blob


class _Cursor:
    def __init__(self, cols: list[str], rows: list[list]):
        self._cols = cols
        self._rows = rows
        self._pos = 0

    @property
    def description(self):
        # sqlite3-compatible: sequence of 7-tuples where [0] is column name
        return [(c, None, None, None, None, None, None) for c in self._cols]

    def fetchone(self) -> dict | None:
        if self._pos >= len(self._rows):
            return None
        row = dict(zip(self._cols, self._rows[self._pos]))
        self._pos += 1
        return row

    def fetchall(self) -> list[dict]:
        result = [dict(zip(self._cols, r)) for r in self._rows[self._pos:]]
        self._pos = len(self._rows)
        return result


class _TursoConnection:
    """sqlite3-compatible connection backed by the Turso /v2/pipeline HTTP endpoint."""

    def __init__(self, url: str, token: str):
        http_base = url.replace("libsql://", "https://")
        self._url = f"{http_base}/v2/pipeline"
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    # Ignored — _Cursor always returns dicts
    @property
    def row_factory(self): return None
    @row_factory.setter
    def row_factory(self, _): pass

    def execute(self, sql: str, params=()) -> _Cursor:
        body = {
            "requests": [
                {"type": "execute", "stmt": {"sql": sql.strip(), "args": [_to_arg(p) for p in params]}},
                {"type": "close"},
            ]
        }
        resp = _requests.post(self._url, json=body, headers=self._headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()["results"][0]
        if result.get("type") == "error":
            raise Exception(result["error"]["message"])
        r = result["response"]["result"]
        cols = [c["name"] for c in r.get("cols", [])]
        rows = [[_from_val(cell) for cell in row] for row in r.get("rows", [])]
        return _Cursor(cols, rows)

    def batch(self, statements: list[tuple[str, tuple]]) -> None:
        """Send multiple statements in a single HTTP round-trip."""
        reqs = [
            {"type": "execute", "stmt": {"sql": sql.strip(), "args": [_to_arg(p) for p in params]}}
            for sql, params in statements
        ]
        reqs.append({"type": "close"})
        resp = _requests.post(self._url, json={"requests": reqs}, headers=self._headers, timeout=30)
        resp.raise_for_status()

    def commit(self): pass   # Turso auto-commits each statement
    def close(self): pass


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_conn(_db_path=None) -> Generator[_TursoConnection, None, None]:
    global _schema_ready
    url, token = _load_config()
    conn = _TursoConnection(url, token)
    try:
        if not _schema_ready:
            # Batch all schema + migration statements into two round-trips
            conn.batch([(s, ()) for s in _SCHEMA_STATEMENTS])
            for stmt in _MIGRATIONS:
                try:
                    conn.execute(stmt)
                except Exception:
                    pass  # column already exists
            _schema_ready = True
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Session writes
# ---------------------------------------------------------------------------

def upsert_session(conn: _TursoConnection, session) -> None:
    conn.execute("""
        INSERT INTO sessions (
            session_id, project_id, cwd, git_branch, started_at,
            turn_count, initial_prompt, ai_title, agent_type
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            turn_count     = excluded.turn_count,
            initial_prompt = excluded.initial_prompt,
            ai_title       = excluded.ai_title
    """, (
        session.session_id,
        _project_id(session.cwd),
        session.cwd,
        session.git_branch,
        _dt(session.started_at),
        session.turn_count,
        session.initial_prompt,
        getattr(session, "ai_title", "") or "",
        getattr(session, "agent_type", "claude_code"),
    ))


def upsert_change_event(conn: _TursoConnection, event, repo_relative: str) -> None:
    conn.execute("""
        INSERT INTO change_events (
            event_id, session_id, file_path, repo_relative_path,
            tool_name, turn_index, event_timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO NOTHING
    """, (
        event.event_id,
        event.session_id,
        event.file_path,
        repo_relative,
        event.tool_name,
        event.turn_index,
        _dt(event.session_timestamp),
    ))


def update_change_signals(
    conn: _TursoConnection,
    event_id: str,
    days_until_next_touch: float | None,
    was_reverted: bool,
    churn_count_30d: int,
    blast_radius_7d: int,
) -> None:
    conn.execute("""
        UPDATE change_events SET
            days_until_next_touch = ?,
            was_reverted          = ?,
            churn_count_30d       = ?,
            blast_radius_7d       = ?
        WHERE event_id = ?
    """, (days_until_next_touch, int(was_reverted), churn_count_30d, blast_radius_7d, event_id))


def update_session_scores(
    conn: _TursoConnection,
    session_id: str,
    *,
    outcome_quality: float | None = None,
    execution_efficiency: float | None = None,
    prompt_quality: float | None = None,
    durability: float | None = None,
    robustness_delta: float | None = None,
    change_type: str | None = None,
    code_region_type: str | None = None,
    failure_mode_tags: list[str] | None = None,
    hermes_assessment: float | None = None,
    assessment_source: str | None = None,
) -> None:
    fields: list[str] = []
    values: list = []

    def _add(col, val):
        if val is not None:
            fields.append(f"{col} = ?")
            values.append(val)

    _add("outcome_quality", outcome_quality)
    _add("execution_efficiency", execution_efficiency)
    _add("prompt_quality", prompt_quality)
    _add("durability", durability)
    _add("robustness_delta", robustness_delta)
    _add("change_type", change_type)
    _add("code_region_type", code_region_type)
    _add("hermes_assessment", hermes_assessment)
    _add("assessment_source", assessment_source)
    if failure_mode_tags is not None:
        fields.append("failure_mode_tags = ?")
        values.append(json.dumps(failure_mode_tags))

    if not fields:
        return
    values.append(session_id)
    conn.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE session_id = ?", values)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_session(conn: _TursoConnection, session_id: str) -> dict | None:
    return conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()


def get_all_sessions(conn: _TursoConnection) -> list[dict]:
    return conn.execute("SELECT * FROM sessions ORDER BY started_at DESC").fetchall()


def get_change_events_for_session(conn: _TursoConnection, session_id: str) -> list[dict]:
    return conn.execute("SELECT * FROM change_events WHERE session_id = ?", (session_id,)).fetchall()


def insert_git_change_event(
    conn: _TursoConnection,
    event_id: str,
    session_id: str,
    repo_relative_path: str,
    event_timestamp: str,
) -> None:
    conn.execute("""
        INSERT INTO change_events (
            event_id, session_id, repo_relative_path, event_timestamp, tool_name, turn_index
        ) VALUES (?, ?, ?, ?, 'git', 0)
        ON CONFLICT(event_id) DO NOTHING
    """, (event_id, session_id, repo_relative_path, event_timestamp))


def get_sessions_without_git_signals(conn: _TursoConnection) -> list[dict]:
    return conn.execute("""
        SELECT DISTINCT s.* FROM sessions s
        JOIN change_events ce ON ce.session_id = s.session_id
        WHERE ce.days_until_next_touch IS NULL
          AND ce.was_reverted IS NULL
    """).fetchall()


def get_file_stats(conn: _TursoConnection, repo_relative_path: str) -> dict:
    row = conn.execute("""
        SELECT COUNT(*) AS total_changes, AVG(days_until_next_touch) AS avg_days_stable,
               SUM(was_reverted) AS revert_count, AVG(churn_count_30d) AS avg_churn_30d
        FROM change_events WHERE repo_relative_path = ?
    """, (repo_relative_path,)).fetchone()
    return row or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dt(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _project_id(cwd: str) -> str:
    import hashlib
    return hashlib.sha1(cwd.encode()).hexdigest()[:12]
