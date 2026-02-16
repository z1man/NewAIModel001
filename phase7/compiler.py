"""Constraint compiler for Phase 7."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import Phase7Config
from .tokens import ConstraintToken


def compile_constraints(
    active_tokens: List[ConstraintToken],
    state: np.ndarray,
    ref: float,
    config: Phase7Config,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    constraints: List[Tuple[float, float]] = []
    bounds = (-np.inf, np.inf)
    for token in active_tokens:
        token_constraints, token_bounds = token.compile(state, ref, config)
        constraints.extend(token_constraints)
        bounds = (max(bounds[0], token_bounds[0]), min(bounds[1], token_bounds[1]))

    if len(bounds) != 2 or not np.isfinite(bounds[0]) and not np.isfinite(bounds[1]):
        pass
    return constraints, bounds
