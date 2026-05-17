"""
Unit tests for pure scoring functions — no I/O, no LLM calls, no DB.
All values are deterministic and can be verified by hand.
"""
import math
import pytest

from learngentic.scoring.efficiency import efficiency_score, performance_score
from learngentic.scoring.signals import (
    stability_lr,
    revert_lr,
    churn_lr,
    blast_radius_lr,
    all_likelihood_ratios,
)
from learngentic.scoring.bayesian import (
    update,
    score_from_signals,
    aggregate_session_durability,
    PRIOR,
)


# ---------------------------------------------------------------------------
# efficiency_score
# ---------------------------------------------------------------------------

def test_efficiency_one_turn():
    assert efficiency_score(1) == pytest.approx(1.0)


def test_efficiency_two_turns():
    # 1 / (1 + log2(2)) = 1/2
    assert efficiency_score(2) == pytest.approx(0.5)


def test_efficiency_four_turns():
    # 1 / (1 + log2(4)) = 1/3
    assert efficiency_score(4) == pytest.approx(1.0 / 3.0)


def test_efficiency_zero_or_negative():
    assert efficiency_score(0) == pytest.approx(1.0)
    assert efficiency_score(-5) == pytest.approx(1.0)


def test_efficiency_large_turn_count():
    # Score is always positive
    assert efficiency_score(1000) > 0.0


def test_performance_score():
    # outcome × efficiency
    assert performance_score(0.8, 2) == pytest.approx(0.8 * 0.5)


# ---------------------------------------------------------------------------
# stability_lr
# ---------------------------------------------------------------------------

def test_stability_lr_none():
    # Never touched again — strong positive
    assert stability_lr(None) == pytest.approx(1.6)


def test_stability_lr_long_stability():
    assert stability_lr(31) == pytest.approx(1.4)
    assert stability_lr(30.01) == pytest.approx(1.4)


def test_stability_lr_medium_stability():
    assert stability_lr(8) == pytest.approx(1.1)
    assert stability_lr(7.01) == pytest.approx(1.1)


def test_stability_lr_short_stability():
    assert stability_lr(3) == pytest.approx(0.9)
    assert stability_lr(2.01) == pytest.approx(0.9)


def test_stability_lr_very_short():
    # Touched within 48 hours — strong negative
    assert stability_lr(2.0) == pytest.approx(0.5)
    assert stability_lr(0) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# revert_lr
# ---------------------------------------------------------------------------

def test_revert_lr_true():
    assert revert_lr(True) == pytest.approx(0.1)


def test_revert_lr_false():
    assert revert_lr(False) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# churn_lr
# ---------------------------------------------------------------------------

def test_churn_lr_zero():
    assert churn_lr(0) == pytest.approx(1.3)


def test_churn_lr_low():
    assert churn_lr(1) == pytest.approx(1.0)
    assert churn_lr(2) == pytest.approx(1.0)


def test_churn_lr_medium():
    assert churn_lr(3) == pytest.approx(0.7)
    assert churn_lr(4) == pytest.approx(0.7)


def test_churn_lr_high():
    assert churn_lr(5) == pytest.approx(0.5)
    assert churn_lr(100) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# blast_radius_lr
# ---------------------------------------------------------------------------

def test_blast_radius_lr_zero():
    assert blast_radius_lr(0) == pytest.approx(1.1)


def test_blast_radius_lr_low():
    assert blast_radius_lr(1) == pytest.approx(1.0)
    assert blast_radius_lr(3) == pytest.approx(1.0)


def test_blast_radius_lr_medium():
    assert blast_radius_lr(4) == pytest.approx(0.8)
    assert blast_radius_lr(7) == pytest.approx(0.8)


def test_blast_radius_lr_high():
    assert blast_radius_lr(8) == pytest.approx(0.7)
    assert blast_radius_lr(50) == pytest.approx(0.7)


def test_all_likelihood_ratios_length():
    lrs = all_likelihood_ratios(None, False, 0, 0)
    assert len(lrs) == 4


# ---------------------------------------------------------------------------
# Bayesian update
# ---------------------------------------------------------------------------

def test_update_neutral_prior_no_signals():
    # No signals — prior unchanged
    result = update(PRIOR, [])
    assert result == pytest.approx(PRIOR)


def test_update_revert_drives_score_down():
    # Revert LR = 0.1 dramatically lowers confidence
    result = update(0.65, [0.1])
    assert result < 0.2


def test_update_all_positive_signals():
    # odds(0.5)=1.0 × 1.6 × 1.4 × 1.3 × 1.1 = 3.20 → prob ≈ 0.76
    result = update(0.5, [1.6, 1.4, 1.3, 1.1])
    assert result > 0.7


def test_update_clamped_at_max():
    # Even with extreme positive signals, stays ≤ 0.95
    result = update(0.65, [100.0, 100.0, 100.0])
    assert result == pytest.approx(0.95)


def test_update_clamped_at_min():
    # Even with extreme negative signals, stays ≥ 0.05
    result = update(0.65, [0.001, 0.001, 0.001])
    assert result == pytest.approx(0.05)


def test_score_from_signals_reverted():
    result = score_from_signals(
        days_until_next_touch=1,
        was_reverted=True,
        churn_count_30d=5,
        blast_radius_7d=10,
    )
    assert result <= 0.15  # very bad outcome


def test_score_from_signals_stable():
    # LRs: 1.6 × 1.0 × 1.3 × 1.1 = 2.288; odds(0.65)=1.857 → 0.809
    result = score_from_signals(
        days_until_next_touch=None,  # never touched again
        was_reverted=False,
        churn_count_30d=0,
        blast_radius_7d=0,
    )
    assert result > 0.79


# ---------------------------------------------------------------------------
# aggregate_session_durability
# ---------------------------------------------------------------------------

def test_aggregate_empty_returns_prior():
    assert aggregate_session_durability([]) == pytest.approx(PRIOR)


def test_aggregate_single_passthrough():
    assert aggregate_session_durability([0.7]) == pytest.approx(0.7)


def test_aggregate_harmonic_penalizes_bad_file():
    # One great file (0.9) and one terrible file (0.1)
    # Harmonic mean is lower than arithmetic mean
    scores = [0.9, 0.1]
    harmonic = 2 / (1 / 0.9 + 1 / 0.1)
    arithmetic = sum(scores) / len(scores)
    result = aggregate_session_durability(scores)
    assert result < arithmetic
    assert result == pytest.approx(max(0.05, min(0.95, harmonic)))


def test_aggregate_all_good_files():
    scores = [0.85, 0.9, 0.88]
    result = aggregate_session_durability(scores)
    # Should be close to the individual scores
    assert result > 0.8


def test_aggregate_clamped():
    # Even if all files are perfect (score near 1), stays ≤ 0.95
    result = aggregate_session_durability([0.94, 0.94, 0.94])
    assert result <= 0.95
