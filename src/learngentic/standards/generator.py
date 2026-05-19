"""
Standards generator for Learngentic.

Given a (change_type, code_region_type) pair, queries global_patterns and
returns a TaskStandard containing benchmarks, a completion checklist, data
confidence level, and signal-tier metadata.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from learngentic.store.db import get_conn

# ---------------------------------------------------------------------------
# Failure mode → checklist item mapping
# ---------------------------------------------------------------------------

_FAILURE_MODE_CHECKLIST: dict[str, str] = {
    "missing_acceptance_criteria": (
        "Define acceptance criteria before starting — what does 'done' look like?"
    ),
    "underspecified_scope": (
        "Clarify scope: which files/functions to change and which to leave unchanged."
    ),
    "missing_constraints": (
        "Identify constraints to preserve: error handling, performance, existing API contracts."
    ),
    "context_gap": (
        "Provide codebase context that isn't obvious from file names alone."
    ),
    "over_specified": (
        "Focus on 'what' not 'how' — over-specifying the approach can block better solutions."
    ),
}

_GENERIC_CHECKLIST: list[str] = [
    "Define acceptance criteria: what does a successful outcome look like?",
    "Clarify scope: which files/functions are in scope and which should not be touched.",
    "Identify constraints: existing interfaces, performance requirements, error handling.",
    "Confirm no in-progress changes exist for the same files (check git status).",
    "Verify the result satisfies all acceptance criteria before declaring done.",
]

_FINAL_CHECKLIST_ITEM = "Verify the result satisfies all acceptance criteria before declaring done."


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaskStandard:
    change_type: str
    code_region_type: str
    confidence: str          # "high" | "medium" | "low" | "none"
    first_instance: bool
    benchmarks: dict
    completion_checklist: list[str]
    signal_tiers: dict
    sample_count: int
    raw_pattern: Optional[dict] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type,
            "code_region_type": self.code_region_type,
            "confidence": self.confidence,
            "first_instance": self.first_instance,
            "benchmarks": self.benchmarks,
            "completion_checklist": self.completion_checklist,
            "signal_tiers": self.signal_tiers,
            "sample_count": self.sample_count,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_standard(change_type: str, code_region_type: str) -> TaskStandard:
    """
    Build a TaskStandard for the given (change_type, code_region_type) pair.

    Returns a cold-start standard when no global_patterns row exists.
    """
    with get_conn() as conn:
        pattern = conn.execute("""
            SELECT avg_outcome_quality, avg_efficiency, avg_prompt_quality,
                   avg_durability, common_failure_modes, sample_count
            FROM global_patterns
            WHERE change_type = ? AND code_region_type = ?
        """, (change_type, code_region_type)).fetchone()
        failure_warning = _recent_failure_warning(change_type, code_region_type, conn)

    if not pattern or not pattern.get("sample_count"):
        return _cold_start_standard(change_type, code_region_type, failure_warning)

    n = pattern["sample_count"] or 0
    avg_eff = pattern["avg_efficiency"]
    avg_dur = pattern["avg_durability"]
    avg_oq  = pattern["avg_outcome_quality"]
    avg_pq  = pattern["avg_prompt_quality"]

    # Confidence tier
    if n >= 10:
        confidence = "high"
    elif n >= 3:
        confidence = "medium"
    else:
        confidence = "low"

    # Turn-count range derived from historical efficiency
    expected_turns = _invert_efficiency(avg_eff) if avg_eff else 20
    benchmarks = {
        "expected_turns_low":   max(1, int(expected_turns * 0.6)),
        "expected_turns_high":  max(2, int(expected_turns * 1.5)),
        "efficiency_threshold": round((avg_eff or 0.3) * 0.75, 3),
        "durability_floor":     round((avg_dur or 0.5) * 0.70, 3),
        "avg_efficiency":       round(avg_eff, 3) if avg_eff is not None else None,
        "avg_durability":       round(avg_dur, 3) if avg_dur is not None else None,
        "avg_outcome_quality":  round(avg_oq,  3) if avg_oq  is not None else None,
        "avg_prompt_quality":   round(avg_pq,  3) if avg_pq  is not None else None,
    }

    # Checklist: invert common failure modes (count ≥ 2) into pre-flight checks
    checklist: list[str] = []
    if pattern["common_failure_modes"]:
        try:
            modes: dict = json.loads(pattern["common_failure_modes"])
            for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
                if count >= 2 and mode in _FAILURE_MODE_CHECKLIST:
                    checklist.append(_FAILURE_MODE_CHECKLIST[mode])
        except (json.JSONDecodeError, AttributeError):
            pass

    if not checklist:
        checklist = _GENERIC_CHECKLIST[:-1]  # final item appended unconditionally below

    if _FINAL_CHECKLIST_ITEM not in checklist:
        checklist.append(_FINAL_CHECKLIST_ITEM)

    if failure_warning:
        checklist = [failure_warning] + checklist

    tier2_available = (avg_oq is not None) or (avg_pq is not None)
    signal_tiers = {
        "tier1_available": (avg_eff is not None) or (avg_dur is not None),
        "tier2_available": tier2_available,
        "tier1_signals":   "execution_efficiency and durability (turn counts + git history)",
        "tier2_signals":   "outcome_quality and prompt_quality (human rating or LLM grading)",
        "note": (
            "Standards grounded in tier 1 only — weight accordingly."
            if not tier2_available
            else "Standards include both tier 1 and tier 2 signals."
        ),
    }

    return TaskStandard(
        change_type=change_type,
        code_region_type=code_region_type,
        confidence=confidence,
        first_instance=False,
        benchmarks=benchmarks,
        completion_checklist=checklist,
        signal_tiers=signal_tiers,
        sample_count=n,
        raw_pattern=dict(pattern),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recent_failure_warning(change_type: str, code_region_type: str, conn) -> str | None:
    """
    Query recent low-scoring sessions for this task type and return a warning string
    to prepend to the checklist, or None if the recent history looks healthy.
    """
    rows = conn.execute("""
        SELECT failure_mode_tags FROM sessions
        WHERE change_type = ? AND code_region_type = ?
          AND failure_mode_tags IS NOT NULL
          AND hermes_assessment < 0.5
        ORDER BY started_at DESC
        LIMIT 5
    """, (change_type, code_region_type)).fetchall()

    if not rows:
        return None

    mode_counts: dict[str, int] = {}
    for row in rows:
        try:
            tags = json.loads(row["failure_mode_tags"])
            if isinstance(tags, list):
                for tag in tags:
                    mode_counts[tag] = mode_counts.get(tag, 0) + 1
            elif isinstance(tags, dict):
                for tag, count in tags.items():
                    mode_counts[tag] = mode_counts.get(tag, 0) + int(count)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    if not mode_counts:
        return None

    top_mode = max(mode_counts, key=mode_counts.__getitem__)
    readable = top_mode.replace("_", " ")
    return (
        f"[PATTERN WARNING] {len(rows)} recent similar sessions scored poorly. "
        f"Most common failure: '{readable}'. "
        "Resolve this explicitly before writing any code."
    )


def _invert_efficiency(eff: float) -> int:
    """Convert an efficiency score back to approximate turn count.

    Inverts:  efficiency = 1 / (1 + log2(n))
    →  n = 2^(1/eff - 1)
    """
    if eff <= 0:
        return 200
    if eff >= 1.0:
        return 1
    import math
    return min(200, max(1, int(round(2 ** (1.0 / eff - 1.0)))))


def _cold_start_standard(
    change_type: str,
    code_region_type: str,
    failure_warning: str | None = None,
) -> TaskStandard:
    checklist = ([failure_warning] + list(_GENERIC_CHECKLIST)) if failure_warning else list(_GENERIC_CHECKLIST)
    return TaskStandard(
        change_type=change_type,
        code_region_type=code_region_type,
        confidence="none",
        first_instance=True,
        benchmarks={
            "expected_turns_low":   3,
            "expected_turns_high":  30,
            "efficiency_threshold": 0.2,
            "durability_floor":     0.4,
            "avg_efficiency":       None,
            "avg_durability":       None,
            "avg_outcome_quality":  None,
            "avg_prompt_quality":   None,
        },
        completion_checklist=checklist,
        signal_tiers={
            "tier1_available": False,
            "tier2_available": False,
            "tier1_signals":   "execution_efficiency and durability (turn counts + git history)",
            "tier2_signals":   "outcome_quality and prompt_quality (human rating or LLM grading)",
            "note":            "No historical data for this task type — this is the first recorded instance.",
        },
        sample_count=0,
    )
