"""
Integration tests for the SQLite persistence layer.
Uses an in-memory database — no files written to disk.
"""
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learngentic.store.db import (
    get_conn,
    upsert_session,
    upsert_change_event,
    update_change_signals,
    update_session_scores,
    get_session,
    get_all_sessions,
    get_change_events_for_session,
    get_file_stats,
)
from learngentic.pipeline.session_parser import SessionRecord, ChangeEvent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """A temporary SQLite DB that is cleaned up after each test."""
    return tmp_path / "test.db"


def _make_session(session_id="abc123", cwd="/repo", turn_count=4) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        project_dir="/tmp/project",
        cwd=cwd,
        git_branch="main",
        started_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        turn_count=turn_count,
        initial_prompt="Add a login button",
        ai_title="Login button feature",
    )


def _make_event(session_id="abc123", event_id="evt001", file_path="/repo/src/auth.py") -> ChangeEvent:
    return ChangeEvent(
        session_id=session_id,
        event_id=event_id,
        session_timestamp=datetime(2025, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
        repo_path="/repo",
        git_branch="main",
        file_path=file_path,
        tool_name="Edit",
        user_intent="Add a login button",
        turn_index=2,
    )


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------

def test_upsert_and_retrieve_session(tmp_db):
    session = _make_session()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        row = get_session(conn, "abc123")

    assert row is not None
    assert row["session_id"] == "abc123"
    assert row["initial_prompt"] == "Add a login button"
    assert row["turn_count"] == 4
    assert row["cwd"] == "/repo"


def test_upsert_session_is_idempotent(tmp_db):
    session = _make_session(turn_count=4)
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_session(conn, session)  # second call should not error
        rows = get_all_sessions(conn)

    assert len(rows) == 1


def test_upsert_session_updates_turn_count(tmp_db):
    session_v1 = _make_session(turn_count=4)
    session_v2 = _make_session(turn_count=9)  # same session_id, updated turns
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session_v1)
        upsert_session(conn, session_v2)
        row = get_session(conn, "abc123")

    assert row["turn_count"] == 9


def test_get_all_sessions_empty(tmp_db):
    with get_conn(tmp_db) as conn:
        rows = get_all_sessions(conn)
    assert rows == []


def test_get_all_sessions_multiple(tmp_db):
    s1 = _make_session(session_id="s1")
    s2 = _make_session(session_id="s2")
    with get_conn(tmp_db) as conn:
        upsert_session(conn, s1)
        upsert_session(conn, s2)
        rows = get_all_sessions(conn)
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Score updates
# ---------------------------------------------------------------------------

def test_update_session_scores(tmp_db):
    session = _make_session()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        update_session_scores(
            conn, "abc123",
            outcome_quality=0.8,
            execution_efficiency=0.5,
            durability=0.72,
            change_type="feature_addition",
            code_region_type="auth",
        )
        row = get_session(conn, "abc123")

    assert row["outcome_quality"] == pytest.approx(0.8)
    assert row["execution_efficiency"] == pytest.approx(0.5)
    assert row["durability"] == pytest.approx(0.72)
    assert row["change_type"] == "feature_addition"
    assert row["code_region_type"] == "auth"


def test_update_session_scores_partial(tmp_db):
    session = _make_session()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        update_session_scores(conn, "abc123", outcome_quality=0.7)
        update_session_scores(conn, "abc123", execution_efficiency=0.5)
        row = get_session(conn, "abc123")

    # Both values should be present after two separate partial updates
    assert row["outcome_quality"] == pytest.approx(0.7)
    assert row["execution_efficiency"] == pytest.approx(0.5)


def test_update_scores_noop_when_no_fields(tmp_db):
    session = _make_session()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        # Passing no non-None values should not raise
        update_session_scores(conn, "abc123")
        row = get_session(conn, "abc123")

    assert row["outcome_quality"] is None


def test_update_failure_mode_tags(tmp_db):
    import json
    session = _make_session()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        update_session_scores(conn, "abc123", failure_mode_tags=["missing_acceptance_criteria", "context_gap"])
        row = get_session(conn, "abc123")

    tags = json.loads(row["failure_mode_tags"])
    assert "missing_acceptance_criteria" in tags
    assert "context_gap" in tags


# ---------------------------------------------------------------------------
# Change event CRUD
# ---------------------------------------------------------------------------

def test_upsert_and_retrieve_change_event(tmp_db):
    session = _make_session()
    event = _make_event()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_change_event(conn, event, "src/auth.py")
        rows = get_change_events_for_session(conn, "abc123")

    assert len(rows) == 1
    assert rows[0]["event_id"] == "evt001"
    assert rows[0]["repo_relative_path"] == "src/auth.py"
    assert rows[0]["tool_name"] == "Edit"


def test_upsert_change_event_on_conflict_noop(tmp_db):
    session = _make_session()
    event = _make_event()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_change_event(conn, event, "src/auth.py")
        upsert_change_event(conn, event, "src/auth.py")  # duplicate — should not error
        rows = get_change_events_for_session(conn, "abc123")

    assert len(rows) == 1


def test_update_change_signals(tmp_db):
    session = _make_session()
    event = _make_event()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_change_event(conn, event, "src/auth.py")
        update_change_signals(conn, "evt001", 14.0, False, 1, 2)
        rows = get_change_events_for_session(conn, "abc123")

    row = rows[0]
    assert row["days_until_next_touch"] == pytest.approx(14.0)
    assert row["was_reverted"] == 0
    assert row["churn_count_30d"] == 1
    assert row["blast_radius_7d"] == 2


def test_update_change_signals_reverted(tmp_db):
    session = _make_session()
    event = _make_event()
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_change_event(conn, event, "src/auth.py")
        update_change_signals(conn, "evt001", 1.0, True, 5, 0)
        rows = get_change_events_for_session(conn, "abc123")

    assert rows[0]["was_reverted"] == 1


# ---------------------------------------------------------------------------
# get_file_stats
# ---------------------------------------------------------------------------

def test_get_file_stats(tmp_db):
    session = _make_session()
    e1 = _make_event(event_id="e1", file_path="/repo/src/auth.py")
    e2 = _make_event(event_id="e2", file_path="/repo/src/auth.py")
    with get_conn(tmp_db) as conn:
        upsert_session(conn, session)
        upsert_change_event(conn, e1, "src/auth.py")
        upsert_change_event(conn, e2, "src/auth.py")
        update_change_signals(conn, "e1", 5.0, False, 2, 1)
        update_change_signals(conn, "e2", 15.0, True, 4, 0)
        stats = get_file_stats(conn, "src/auth.py")

    assert stats["total_changes"] == 2
    assert stats["revert_count"] == 1
    assert stats["avg_churn_30d"] == pytest.approx(3.0)


def test_get_file_stats_missing_file(tmp_db):
    with get_conn(tmp_db) as conn:
        stats = get_file_stats(conn, "nonexistent.py")
    # Returns a dict — total_changes should be 0
    assert stats.get("total_changes") == 0
