"""
Bayesian posterior updater for change quality.

Starts from a prior and multiplies by likelihood ratios sequentially.
Uses odds-ratio form so the math stays numerically stable.
Output is clamped to [0.05, 0.95] — never fully certain.
"""
from __future__ import annotations

PRIOR = 0.65  # assume reasonable quality before any signal


def _to_odds(p: float) -> float:
    return p / (1.0 - p)


def _to_prob(odds: float) -> float:
    return odds / (1.0 + odds)


def update(prior: float, likelihood_ratios: list[float]) -> float:
    """
    Apply a sequence of likelihood ratios to a prior probability.
    Returns posterior probability clamped to [0.05, 0.95].
    """
    odds = _to_odds(max(1e-6, min(1 - 1e-6, prior)))
    for lr in likelihood_ratios:
        odds *= lr
    return max(0.05, min(0.95, _to_prob(odds)))


def score_from_signals(
    days_until_next_touch: float | None,
    was_reverted: bool,
    churn_count_30d: int,
    blast_radius_7d: int,
    prior: float = PRIOR,
) -> float:
    from learngentic.scoring.signals import all_likelihood_ratios
    lrs = all_likelihood_ratios(
        days_until_next_touch, was_reverted, churn_count_30d, blast_radius_7d
    )
    return update(prior, lrs)


def aggregate_session_durability(change_scores: list[float]) -> float:
    """
    Combine per-file durability scores into a single session-level score.
    Uses the harmonic mean — penalises sessions where even one file is bad.
    Falls back to arithmetic mean for robustness with empty lists.
    """
    if not change_scores:
        return PRIOR
    if len(change_scores) == 1:
        return change_scores[0]
    n = len(change_scores)
    harmonic = n / sum(1.0 / max(s, 1e-6) for s in change_scores)
    return max(0.05, min(0.95, harmonic))
