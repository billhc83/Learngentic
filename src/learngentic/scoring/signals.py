"""
Maps raw git signals to Bayesian likelihood ratios.
Each function returns a multiplier applied to the current quality prior.
Values > 1.0 increase confidence; values < 1.0 decrease it.
"""
from __future__ import annotations


def stability_lr(days_until_next_touch: float | None) -> float:
    """Likelihood ratio based on how long the change stood untouched."""
    if days_until_next_touch is None:
        return 1.6   # never touched again — strong positive
    if days_until_next_touch > 30:
        return 1.4
    if days_until_next_touch > 7:
        return 1.1
    if days_until_next_touch > 2:
        return 0.9
    return 0.5       # touched within 48 hours — strong negative


def revert_lr(was_reverted: bool) -> float:
    """Revert is the strongest negative signal in version control."""
    return 0.1 if was_reverted else 1.0


def churn_lr(churn_count_30d: int) -> float:
    """How many times the same file was committed in 30 days after this change."""
    if churn_count_30d == 0:
        return 1.3
    if churn_count_30d <= 2:
        return 1.0
    if churn_count_30d <= 4:
        return 0.7
    return 0.5


def blast_radius_lr(blast_radius_7d: int) -> float:
    """How many co-changed files needed follow-up commits within 7 days."""
    if blast_radius_7d == 0:
        return 1.1
    if blast_radius_7d <= 3:
        return 1.0
    if blast_radius_7d <= 7:
        return 0.8
    return 0.7


def all_likelihood_ratios(
    days_until_next_touch: float | None,
    was_reverted: bool,
    churn_count_30d: int,
    blast_radius_7d: int,
) -> list[float]:
    return [
        stability_lr(days_until_next_touch),
        revert_lr(was_reverted),
        churn_lr(churn_count_30d),
        blast_radius_lr(blast_radius_7d),
    ]
