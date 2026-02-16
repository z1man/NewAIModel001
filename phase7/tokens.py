"""Token definitions for Phase 7.

Feature tokens can still be used for learning, but constraint-tokens compile
into linear constraints of the form a*u <= b and/or explicit bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Phase7Config


@dataclass
class TokenStats:
    mean: np.ndarray
    std: np.ndarray


@dataclass
class ConstraintToken:
    name: str

    def compile(self, state: np.ndarray, ref: float, config: Phase7Config) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        raise NotImplementedError


@dataclass
class ActuatorLimitToken(ConstraintToken):
    def compile(self, state: np.ndarray, ref: float, config: Phase7Config) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        u_min = float(config.u_min[0])
        u_max = float(config.u_max[0])
        return [], (u_min, u_max)


@dataclass
class AccLimitToken(ConstraintToken):
    """Acceleration limit token.

    a_nom(s,u) = alpha(s) + beta * u
    beta = dt / m
    alpha(s) = -kv_nom * v
    Constraint: |a_nom| <= a_max
    """

    def compile(self, state: np.ndarray, ref: float, config: Phase7Config) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
        v = float(state[1])
        alpha = -config.kv_nom * v
        beta = config.dt / config.mass
        a_max = config.a_max
        if beta == 0:
            return [], (-np.inf, np.inf)
        lower = (-a_max - alpha) / beta
        upper = (a_max - alpha) / beta
        if lower > upper:
            lower, upper = upper, lower
        return [], (lower, upper)


def reference_signal(config: Phase7Config, t: float) -> float:
    return float(config.ref_amp * np.sin(2 * np.pi * config.ref_freq * t))


def build_tokens(
    config: Phase7Config,
    t_idx: int,
    ref: float,
    state: np.ndarray,
    error_prev: float,
    ref_prev: float,
) -> np.ndarray:
    t_norm = t_idx / max(1, config.steps - 1)
    x, v = state
    error = ref - x
    ref_delta = ref - ref_prev
    tokens = np.array(
        [
            t_norm,
            ref,
            x,
            v,
            error,
            error_prev,
            ref_delta,
        ],
        dtype=float,
    )
    return tokens * config.token_scale


def validate_tokens(tokens: np.ndarray, clip: float) -> Tuple[bool, float]:
    if not np.all(np.isfinite(tokens)):
        return False, float("inf")
    max_abs = float(np.max(np.abs(tokens)))
    return max_abs <= clip, max_abs


def compute_token_stats(token_matrix: np.ndarray) -> TokenStats:
    mean = np.mean(token_matrix, axis=0)
    std = np.std(token_matrix, axis=0) + 1e-8
    return TokenStats(mean=mean, std=std)


def default_constraint_tokens() -> List[ConstraintToken]:
    return [ActuatorLimitToken("T_ACTUATOR_LIMIT"), AccLimitToken("T_ACC_LIMIT")]


def token_activation_stats(tokens: List[ConstraintToken]) -> Dict[str, int]:
    return {token.name: 0 for token in tokens}
