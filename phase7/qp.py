"""Simple QP projection utilities for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class QPResult:
    action: np.ndarray
    violated: bool
    violation_mag: float
    projected: bool


def project_qp(
    u_des: np.ndarray,
    u_min: np.ndarray,
    u_max: np.ndarray,
    sum_u_max: float,
) -> QPResult:
    u = np.clip(u_des, u_min, u_max)
    projected = not np.allclose(u, u_des)
    sum_u = float(u[0] + u[1])
    if sum_u <= sum_u_max:
        return QPResult(action=u, violated=False, violation_mag=0.0, projected=projected)

    # Project onto u0 + u1 = sum_u_max, then clip to bounds
    w = np.array([1.0, 1.0])
    excess = (sum_u - sum_u_max) / (w @ w)
    u_proj = u - excess * w
    u_proj = np.clip(u_proj, u_min, u_max)
    sum_u = float(u_proj[0] + u_proj[1])
    violated = sum_u > sum_u_max + 1e-8
    violation_mag = max(0.0, sum_u - sum_u_max)
    return QPResult(action=u_proj, violated=violated, violation_mag=violation_mag, projected=True)
