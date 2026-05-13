"""
Parses a git repository's history into CommitEvent records.
Uses gitpython. Falls back gracefully if the path is not a repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import git as gitpython
    HAS_GITPYTHON = True
except ImportError:
    HAS_GITPYTHON = False


_REVERT_RE = re.compile(r'\brevert\b', re.IGNORECASE)


@dataclass
class CommitEvent:
    repo_path: str
    commit_hash: str
    timestamp: datetime
    message: str
    files_changed: list[str] = field(default_factory=list)   # repo-relative paths
    is_revert: bool = False
    author: str = ""


def _is_revert(message: str) -> bool:
    return bool(_REVERT_RE.search(message))


def parse_repo(repo_path: str | Path, max_commits: int = 5000) -> list[CommitEvent]:
    if not HAS_GITPYTHON:
        raise ImportError("gitpython is required: pip install gitpython")

    repo_path = Path(repo_path)
    try:
        repo = gitpython.Repo(repo_path, search_parent_directories=True)
    except gitpython.InvalidGitRepositoryError:
        return []
    except gitpython.NoSuchPathError:
        return []

    actual_root = Path(repo.working_tree_dir)
    events: list[CommitEvent] = []

    for commit in repo.iter_commits(max_count=max_commits):
        ts = datetime.fromtimestamp(commit.committed_date, tz=timezone.utc)

        # Collect changed files (diff against first parent, or tree for root commits)
        try:
            if commit.parents:
                diffs = commit.parents[0].diff(commit)
            else:
                diffs = commit.diff(gitpython.NULL_TREE)
            changed = []
            for d in diffs:
                if d.b_path:
                    changed.append(d.b_path)
                if d.a_path and d.a_path != d.b_path:
                    changed.append(d.a_path)
        except Exception:
            changed = []

        events.append(CommitEvent(
            repo_path=str(actual_root),
            commit_hash=commit.hexsha,
            timestamp=ts,
            message=commit.message.strip(),
            files_changed=changed,
            is_revert=_is_revert(commit.message),
            author=str(commit.author),
        ))

    return events


def find_repo_root(path: str | Path) -> Path | None:
    """Walk up from path until a .git directory is found."""
    p = Path(path).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / ".git").exists():
            return candidate
    return None
