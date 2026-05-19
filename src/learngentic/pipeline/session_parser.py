"""Dataclasses representing parsed Claude Code session data."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ChangeEvent:
    session_id: str
    event_id: str
    session_timestamp: datetime
    repo_path: str
    git_branch: str
    file_path: str
    tool_name: str
    user_intent: str
    turn_index: int


@dataclass
class SessionRecord:
    session_id: str
    project_dir: str
    cwd: str
    git_branch: str
    started_at: datetime
    turn_count: int
    initial_prompt: str
    ai_title: str | None = None
    change_events: list[ChangeEvent] = field(default_factory=list)
