"""
Joins SessionRecord change events to subsequent CommitEvents on the same files,
computing raw outcome signals for each change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from learngentic.pipeline.session_parser import ChangeEvent, SessionRecord
from learngentic.pipeline.git_parser import CommitEvent


@dataclass
class RawSignals:
    days_until_next_touch: float | None    # None = never touched again
    was_reverted: bool
    churn_count_30d: int                   # commits touching this file in 30 days
    blast_radius_7d: int                   # commits touching co-changed files in 7 days


@dataclass
class JoinedChange:
    session: SessionRecord
    event: ChangeEvent
    signals: RawSignals


def _repo_relative(repo_root: str, abs_path: str) -> str:
    """Normalise an absolute file path to repo-relative POSIX string."""
    try:
        return Path(abs_path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError:
        return Path(abs_path).as_posix()


def join(
    sessions: list[SessionRecord],
    commits_by_repo: dict[str, list[CommitEvent]],
) -> list[JoinedChange]:
    """
    For each ChangeEvent in each session, find subsequent commits that
    touched the same file and compute raw signals.

    commits_by_repo: repo_path (str) → sorted list of CommitEvents (newest first)
    """
    results: list[JoinedChange] = []

    for session in sessions:
        repo_path = session.cwd
        commits = commits_by_repo.get(repo_path, [])
        if not commits:
            # Try finding by repo root in case cwd is a subdirectory
            for rp, c in commits_by_repo.items():
                try:
                    Path(repo_path).relative_to(Path(rp))
                    commits = c
                    repo_path = rp
                    break
                except ValueError:
                    pass

        # Pre-build a set of all files changed per commit for fast lookup
        commit_file_sets: list[tuple[CommitEvent, set[str]]] = [
            (c, set(c.files_changed)) for c in commits
        ]

        # Co-changed files in this session (for blast radius)
        session_files = {
            _repo_relative(repo_path, e.file_path)
            for e in session.change_events
        }

        for event in session.change_events:
            rel_path = _repo_relative(repo_path, event.file_path)
            event_ts = event.session_timestamp

            # All commits after this event that touch this file
            subsequent = [
                c for c, fset in commit_file_sets
                if c.timestamp > event_ts and rel_path in fset
            ]

            # --- days_until_next_touch ---
            if subsequent:
                earliest = min(subsequent, key=lambda c: c.timestamp)
                delta = earliest.timestamp - event_ts
                days_until_next_touch: float | None = delta.total_seconds() / 86400
            else:
                days_until_next_touch = None

            # --- was_reverted ---
            was_reverted = any(c.is_revert for c in subsequent)

            # --- churn_count_30d ---
            window_30d = event_ts + timedelta(days=30)
            churn_count_30d = sum(
                1 for c in subsequent if c.timestamp <= window_30d
            )

            # --- blast_radius_7d ---
            window_7d = event_ts + timedelta(days=7)
            blast_files: set[str] = set()
            for c, fset in commit_file_sets:
                if c.timestamp > event_ts and c.timestamp <= window_7d:
                    # Only count commits that also touched a co-changed file
                    if fset & (session_files - {rel_path}):
                        blast_files |= fset
            blast_radius_7d = len(blast_files - {rel_path})

            results.append(JoinedChange(
                session=session,
                event=event,
                signals=RawSignals(
                    days_until_next_touch=days_until_next_touch,
                    was_reverted=was_reverted,
                    churn_count_30d=churn_count_30d,
                    blast_radius_7d=blast_radius_7d,
                ),
            ))

    return results
