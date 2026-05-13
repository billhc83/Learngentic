"""
Tests for generate_standard and _report_outcome.

Uses lightweight fake DB objects — no Turso credentials required.
"""
import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from learngentic.standards.generator import (
    _cold_start_standard,
    _invert_efficiency,
    generate_standard,
)


# ---------------------------------------------------------------------------
# Fake DB infrastructure
# ---------------------------------------------------------------------------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return list(self._rows)


class _FakeConn:
    """Returns rows keyed by a simple SQL fragment match."""

    def __init__(self, session_row=None, pattern_row=None):
        self._session_row = session_row
        self._pattern_row = pattern_row
        self.executed: list[str] = []

    def execute(self, sql, params=()):
        self.executed.append(sql.strip())
        sql_up = sql.upper()
        if "GLOBAL_PATTERNS" in sql_up:
            rows = [self._pattern_row] if self._pattern_row else []
        elif "SELECT * FROM SESSIONS" in sql_up or "SELECT * FROM sessions" in sql:
            rows = [self._session_row] if self._session_row else []
        else:
            rows = []
        return _FakeCursor(rows)

    def commit(self): pass
    def close(self): pass


def _make_get_conn(session_row=None, pattern_row=None):
    @contextmanager
    def _ctx(*args, **kwargs):
        yield _FakeConn(session_row=session_row, pattern_row=pattern_row)
    return _ctx


# ---------------------------------------------------------------------------
# generate_standard
# ---------------------------------------------------------------------------

def test_generate_standard_cold_start_no_data():
    with patch("learngentic.standards.generator.get_conn", _make_get_conn()):
        std = generate_standard("feature_addition", "api")
    assert std.confidence == "none"
    assert std.first_instance is True
    assert std.sample_count == 0
    assert len(std.completion_checklist) == 5
    assert std.benchmarks["expected_turns_low"] == 3
    assert std.benchmarks["expected_turns_high"] == 30


def test_generate_standard_high_confidence():
    pattern = {
        "avg_outcome_quality": 0.8,
        "avg_efficiency": 0.35,
        "avg_prompt_quality": 0.7,
        "avg_durability": 0.6,
        "common_failure_modes": json.dumps(
            {"missing_acceptance_criteria": 3, "underspecified_scope": 2}
        ),
        "sample_count": 12,
    }
    with patch("learngentic.standards.generator.get_conn", _make_get_conn(pattern_row=pattern)):
        std = generate_standard("feature_addition", "api")
    assert std.confidence == "high"
    assert std.first_instance is False
    assert std.sample_count == 12
    assert std.benchmarks["avg_efficiency"] == pytest.approx(0.35, abs=0.001)
    assert any("acceptance criteria" in item.lower() for item in std.completion_checklist)
    assert std.completion_checklist[-1].startswith("Verify the result")


def test_generate_standard_medium_confidence():
    pattern = {
        "avg_outcome_quality": None,
        "avg_efficiency": 0.25,
        "avg_prompt_quality": None,
        "avg_durability": 0.5,
        "common_failure_modes": None,
        "sample_count": 5,
    }
    with patch("learngentic.standards.generator.get_conn", _make_get_conn(pattern_row=pattern)):
        std = generate_standard("bug_fix", "data_layer")
    assert std.confidence == "medium"
    assert not std.signal_tiers["tier2_available"]
    assert std.signal_tiers["tier1_available"]


def test_generate_standard_low_confidence():
    pattern = {
        "avg_outcome_quality": 0.6,
        "avg_efficiency": 0.3,
        "avg_prompt_quality": None,
        "avg_durability": None,
        "common_failure_modes": None,
        "sample_count": 2,
    }
    with patch("learngentic.standards.generator.get_conn", _make_get_conn(pattern_row=pattern)):
        std = generate_standard("refactor", "ui")
    assert std.confidence == "low"


def test_generate_standard_to_dict_round_trip():
    with patch("learngentic.standards.generator.get_conn", _make_get_conn()):
        std = generate_standard("other", "unknown")
    d = std.to_dict()
    assert "benchmarks" in d
    assert "completion_checklist" in d
    assert "signal_tiers" in d
    assert "raw_pattern" not in d  # excluded from serialisation


# ---------------------------------------------------------------------------
# _invert_efficiency
# ---------------------------------------------------------------------------

def test_invert_efficiency_boundary_zero():
    assert _invert_efficiency(0.0) == 200


def test_invert_efficiency_boundary_one():
    assert _invert_efficiency(1.0) == 1


def test_invert_efficiency_half():
    # 2^(1/0.5 - 1) = 2^1 = 2
    assert _invert_efficiency(0.5) == 2


def test_invert_efficiency_typical_range():
    for eff in [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
        result = _invert_efficiency(eff)
        assert 1 <= result <= 200, f"_invert_efficiency({eff}) = {result} out of range"


# ---------------------------------------------------------------------------
# _report_outcome score logic
# ---------------------------------------------------------------------------

def _make_session(change_type="feature_addition", code_region="api"):
    return {
        "session_id": "sess-001",
        "change_type": change_type,
        "code_region_type": code_region,
        "initial_prompt": "Build feature X",
        "assessment_source": "pending",
        "hermes_assessment": None,
        "turn_count": None,
        "hermes_notes": None,
    }


def _mock_standard(eff_threshold=0.2, turns_high=30):
    std = MagicMock()
    std.benchmarks = {
        "efficiency_threshold": eff_threshold,
        "expected_turns_high": turns_high,
    }
    std.confidence = "medium"
    std.completion_checklist = [
        "Clarify scope.",
        "Verify the result satisfies all acceptance criteria before declaring done.",
    ]
    return std


def _call_report_outcome(session, sent_back, already_existed, is_rewrite, turn_count):
    from learngentic.mcp.server import _report_outcome

    with patch("learngentic.mcp.server.get_conn", _make_get_conn(session_row=session)), \
         patch("learngentic.standards.generator.generate_standard",
               return_value=_mock_standard()), \
         patch("learngentic.mcp.server._fire_prompt_grader_async"):
        return _report_outcome(
            "sess-001", sent_back, already_existed, is_rewrite, turn_count, ""
        )


def test_report_outcome_sent_back_is_fail():
    result = _call_report_outcome(
        _make_session(), sent_back=True, already_existed=False,
        is_rewrite=False, turn_count=10,
    )
    assert result["verdict"] == "fail"
    # 0.7 - 0.40 = 0.30
    assert result["hermes_assessment"] == pytest.approx(0.3, abs=0.01)


def test_report_outcome_clean_is_pass():
    result = _call_report_outcome(
        _make_session(), sent_back=False, already_existed=False,
        is_rewrite=False, turn_count=5,
    )
    assert result["verdict"] == "pass"
    assert result["hermes_assessment"] >= 0.7


def test_report_outcome_all_penalties_clamped():
    result = _call_report_outcome(
        _make_session(), sent_back=True, already_existed=True,
        is_rewrite=True, turn_count=50,
    )
    # 0.7 - 0.40 - 0.20 - 0.15 = -0.05 → clamped to 0.0
    assert result["hermes_assessment"] == pytest.approx(0.0, abs=0.001)
    assert result["verdict"] == "fail"
    assert len(result["checklist_items_failed"]) == 3


def test_report_outcome_missing_session_returns_error():
    from learngentic.mcp.server import _report_outcome

    with patch("learngentic.mcp.server.get_conn", _make_get_conn(session_row=None)):
        result = _report_outcome("nonexistent", False, False, False, 1, "")
    assert "error" in result


def test_report_outcome_already_existed_penalty():
    result = _call_report_outcome(
        _make_session(), sent_back=False, already_existed=True,
        is_rewrite=False, turn_count=5,
    )
    # 0.7 - 0.20 + small efficiency bonus (turn_count=5 is efficient) → ~0.56
    assert result["hermes_assessment"] > 0.5
    assert result["verdict"] == "pass"


def test_report_outcome_turn_count_warning():
    result = _call_report_outcome(
        _make_session(), sent_back=False, already_existed=False,
        is_rewrite=False, turn_count=50,  # exceeds turns_high=30
    )
    assert any("Turn count" in s for s in result["next_steps"])
