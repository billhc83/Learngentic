"""
CLI smoke tests using Click's test runner.
No LLM calls, no real session files — just verifies command wiring and error paths.
"""
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from learngentic.cli import cli
from learngentic.store.db import get_conn, upsert_session
from learngentic.pipeline.session_parser import SessionRecord
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_db(tmp_path: Path) -> Path:
    """Create a minimal DB with one session."""
    db = tmp_path / "learngentic.db"
    session = SessionRecord(
        session_id="test-session-001",
        project_dir=str(tmp_path),
        cwd=str(tmp_path),
        git_branch="main",
        started_at=datetime(2025, 3, 1, tzinfo=timezone.utc),
        turn_count=3,
        initial_prompt="Fix the login bug",
        ai_title="Login fix",
    )
    with get_conn(db) as conn:
        upsert_session(conn, session)
    return db


# ---------------------------------------------------------------------------
# Commands that require a DB — missing DB path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cmd", ["score", "report", "trend", "index", "sync"])
def test_command_missing_db_exits_with_message(cmd, tmp_path):
    runner = CliRunner()
    missing = str(tmp_path / "does_not_exist.db")
    result = runner.invoke(cli, [cmd, "--db", missing])
    assert result.exit_code != 0 or "ingest" in result.output.lower() or "No database" in result.output


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

def test_report_with_db(tmp_path):
    db = _make_db(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--db", str(db)])
    assert result.exit_code == 0
    assert "Total sessions" in result.output
    assert "1" in result.output


def test_report_limit_option(tmp_path):
    db = _make_db(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--db", str(db), "--limit", "5"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# trend command
# ---------------------------------------------------------------------------

def test_trend_no_rated_sessions(tmp_path):
    db = _make_db(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["trend", "--db", str(db)])
    assert result.exit_code == 0
    assert "No rated sessions" in result.output or "0 rated" in result.output


def test_trend_with_rated_session(tmp_path):
    db = _make_db(tmp_path)
    # Manually rate the session
    conn = sqlite3.connect(db)
    conn.execute("UPDATE sessions SET outcome_quality = 0.8 WHERE session_id = 'test-session-001'")
    conn.commit()
    conn.close()

    runner = CliRunner()
    result = runner.invoke(cli, ["trend", "--db", str(db)])
    assert result.exit_code == 0
    assert "1 rated session" in result.output


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------

def test_index_with_db(tmp_path):
    db = _make_db(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["index", "--db", str(db)])
    assert result.exit_code == 0
    assert "Index built" in result.output


# ---------------------------------------------------------------------------
# sync command
# ---------------------------------------------------------------------------

def test_sync_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["sync", "--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output.lower() or "sync" in result.output.lower()


def test_sync_missing_db(tmp_path):
    runner = CliRunner()
    missing = str(tmp_path / "nope.db")
    result = runner.invoke(cli, ["sync", "--db", missing])
    # sync runs ingest first, which creates a DB — but with no projects dir
    # it should either succeed (0 sessions found) or fail gracefully
    assert "Error" not in result.output or result.exit_code == 0


# ---------------------------------------------------------------------------
# rate command
# ---------------------------------------------------------------------------

def test_rate_no_pending(tmp_path):
    db = _make_db(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["rate", "--db", str(db)])
    assert result.exit_code == 0
    # Either no queue entries, or queue entries don't match this isolated DB
    assert "rated 0" in result.output.lower() or "no sessions" in result.output.lower()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "score" in result.output
    assert "rate" in result.output
    assert "sync" in result.output
