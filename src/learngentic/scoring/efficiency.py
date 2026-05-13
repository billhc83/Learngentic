"""
Execution efficiency score derived from turn count.

1 turn  → 1.00 (perfect, first try)
2 turns → 0.50
4 turns → 0.33
8 turns → 0.25
16 turns→ 0.20

The log₂ decay means each doubling of turns halves the denominator increment,
producing diminishing penalty for very long sessions (which often reflect
exploratory tasks rather than failure).
"""
from __future__ import annotations

import math


def efficiency_score(turn_count: int) -> float:
    """Map turn count to a 0-1 efficiency score."""
    if turn_count <= 0:
        return 1.0
    return 1.0 / (1.0 + math.log2(max(1, turn_count)))


def performance_score(outcome_quality: float, turn_count: int) -> float:
    """
    Combined convergence metric: outcome × efficiency.
    Approaches 1.0 when result is good AND achieved in few turns.
    """
    return outcome_quality * efficiency_score(turn_count)
