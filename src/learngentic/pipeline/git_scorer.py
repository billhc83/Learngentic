"""
Incremental git signal scorer.

Finds change_events rows with no git signals, groups them by repo,
parses only the git history needed, computes signals, and writes back.
Touches only new data — safe to run at any scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path


@dataclass
class RawSignals:
    days_until_next_touch: float | None
    was_reverted: bool
    churn_count_30d: int
    blast_radius_7d: int


def _normalise_path(path: str) -> str:
    """Convert WSL-style /mnt/c/... paths to Windows C:\\... paths."""
    if path.startswith("/mnt/") and len(path) > 6:
        return path[5].upper() + ":" + path[6:].replace("/", "\\")
    return path


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def compute_signals(
    file_path: str,
    event_ts: datetime,
    session_files: set[str],
    commit_file_sets: list[tuple],   # list of (CommitEvent, set[str])
) -> RawSignals:
    """Compute all four git signals for one change_event."""
    subsequent = [
        (c, fset) for c, fset in commit_file_sets
        if c.timestamp > event_ts and file_path in fset
    ]

    # days_until_next_touch
    if subsequent:
        earliest = min(subsequent, key=lambda x: x[0].timestamp)
        days: float | None = (earliest[0].timestamp - event_ts).total_seconds() / 86400
    else:
        days = None

    # was_reverted
    was_reverted = any(c.is_revert for c, _ in subsequent)

    # churn_count_30d
    window_30 = event_ts + timedelta(days=30)
    churn = sum(1 for c, _ in subsequent if c.timestamp <= window_30)

    # blast_radius_7d
    window_7 = event_ts + timedelta(days=7)
    blast: set[str] = set()
    for c, fset in commit_file_sets:
        if event_ts < c.timestamp <= window_7:
            if fset & (session_files - {file_path}):
                blast |= fset
    blast_radius = len(blast - {file_path})

    return RawSignals(
        days_until_next_touch=days,
        was_reverted=was_reverted,
        churn_count_30d=churn,
        blast_radius_7d=blast_radius,
    )


def run_incremental() -> dict:
    """
    Score all unscored change_events rows, roll up durability to sessions,
    and flip assessment_source where appropriate.

    Returns a summary dict with counts.
    """
    from learngentic.pipeline.git_parser import find_repo_root, parse_repo
    from learngentic.scoring.bayesian import score_from_signals, aggregate_session_durability
    from learngentic.store.db import get_conn, update_change_signals, update_session_scores

    # ── Step 1: find repos with unscored events ────────────────────────────
    with get_conn() as conn:
        repo_rows = conn.execute("""
            SELECT s.cwd,
                   MIN(ce.event_timestamp) AS oldest_unscored
            FROM change_events ce
            JOIN sessions s ON s.session_id = ce.session_id
            WHERE ce.was_reverted IS NULL
              AND ce.repo_relative_path IS NOT NULL
            GROUP BY s.cwd
        """).fetchall()

    if not repo_rows:
        return {"repos_processed": 0, "events_scored": 0, "sessions_updated": 0}

    events_scored = 0
    sessions_touched: set[str] = set()

    for repo_row in repo_rows:
        cwd = repo_row["cwd"] or ""
        oldest_unscored = repo_row["oldest_unscored"] or ""

        if not cwd:
            continue

        root = find_repo_root(_normalise_path(cwd))
        if not root:
            continue

        # ── Step 2: parse git history for this repo ────────────────────────
        commits = parse_repo(root)
        if not commits:
            continue

        # Trim to commits at or after oldest unscored event (with 1-day buffer)
        if oldest_unscored:
            cutoff = _parse_ts(oldest_unscored) - timedelta(days=1)
            commits = [c for c in commits if c.timestamp >= cutoff]

        commit_file_sets = [(c, set(c.files_changed)) for c in commits]

        # ── Step 3: load unscored events for this repo ─────────────────────
        with get_conn() as conn:
            unscored = conn.execute("""
                SELECT ce.event_id, ce.session_id,
                       ce.repo_relative_path, ce.event_timestamp
                FROM change_events ce
                JOIN sessions s ON s.session_id = ce.session_id
                WHERE s.cwd = ?
                  AND ce.was_reverted IS NULL
                  AND ce.repo_relative_path IS NOT NULL
            """, (cwd,)).fetchall()

        if not unscored:
            continue

        # Build session_files map for blast_radius
        session_files: dict[str, set[str]] = {}
        for e in unscored:
            sid = e["session_id"]
            fp = e["repo_relative_path"]
            if fp:
                session_files.setdefault(sid, set()).add(fp)

        # ── Step 4: compute signals and batch-write ────────────────────────
        signal_updates: list[tuple] = []
        for event in unscored:
            fp = event["repo_relative_path"]
            ts_str = event["event_timestamp"]
            if not fp or not ts_str:
                continue

            event_ts = _parse_ts(ts_str)
            sess_files = session_files.get(event["session_id"], set())
            signals = compute_signals(fp, event_ts, sess_files, commit_file_sets)

            signal_updates.append((
                event["event_id"],
                signals.days_until_next_touch,
                signals.was_reverted,
                signals.churn_count_30d,
                signals.blast_radius_7d,
            ))
            sessions_touched.add(event["session_id"])

        with get_conn() as conn:
            for event_id, days, reverted, churn, blast in signal_updates:
                update_change_signals(conn, event_id, days, reverted, churn, blast)

        events_scored += len(signal_updates)

    # ── Step 5: roll up durability for affected sessions ───────────────────
    sessions_updated = 0
    for session_id in sessions_touched:
        with get_conn() as conn:
            session = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            events = conn.execute(
                "SELECT * FROM change_events WHERE session_id = ?", (session_id,)
            ).fetchall()

        if not session:
            continue

        change_scores = [
            score_from_signals(
                days_until_next_touch=ce["days_until_next_touch"],
                was_reverted=bool(ce["was_reverted"]),
                churn_count_30d=ce["churn_count_30d"] or 0,
                blast_radius_7d=ce["blast_radius_7d"] or 0,
            )
            for ce in events
            if ce["was_reverted"] is not None
        ]

        if not change_scores:
            continue

        durability = aggregate_session_durability(change_scores)
        update_kwargs: dict = {"durability": durability}

        if session.get("assessment_source") == "hermes_self":
            hermes_val = session.get("hermes_assessment")
            if hermes_val is not None and abs(durability - hermes_val) > 0.2:
                update_kwargs["_divergence"] = (hermes_val, durability)
            # Only flip once signals have had 48h to develop
            started_str = session.get("started_at") or ""
            if started_str:
                age = datetime.now(timezone.utc) - _parse_ts(started_str)
                if age.total_seconds() >= 48 * 3600:
                    update_kwargs["assessment_source"] = "git_grounded"

        divergence = update_kwargs.pop("_divergence", None)

        with get_conn() as conn:
            update_session_scores(conn, session_id, **update_kwargs)

        sessions_updated += 1
        if divergence:
            print(
                f"  git_grounded divergence ({session_id[:8]}): "
                f"hermes_self={divergence[0]:.2f} vs git={divergence[1]:.2f} "
                f"(delta={divergence[1]-divergence[0]:+.2f})"
            )

    return {
        "repos_processed": len(repo_rows),
        "events_scored": events_scored,
        "sessions_updated": sessions_updated,
    }
