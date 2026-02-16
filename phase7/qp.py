"""1D closed-form QP projection solver for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class QPResult:
    action: float
    L: float
    U: float
    projected: bool
    proj_mag: float
    feasible: bool
    infeasible: bool


def _interval_from_constraints(constraints: List[Tuple[float, float]]) -> Tuple[float, float]:
    L = -np.inf
    U = np.inf
    for a, b in constraints:
        if a == 0:
            if b < 0:
                return (1.0, 0.0)
            continue
        bound = b / a
        if a > 0:
            U = min(U, bound)
        else:
            L = max(L, bound)
    return L, U


def project_qp(
    u_prop: float,
    constraints: List[Tuple[float, float]],
    u_bounds: Tuple[float, float],
) -> QPResult:
    Lc, Uc = _interval_from_constraints(constraints)
    L = max(Lc, u_bounds[0])
    U = min(Uc, u_bounds[1])
    feasible = L <= U
    if not feasible:
        u_star = float(np.clip(u_prop, u_bounds[0], u_bounds[1]))
        proj_mag = abs(u_star - u_prop)
        return QPResult(
            action=u_star,
            L=L,
            U=U,
            projected=True,
            proj_mag=proj_mag,
            feasible=False,
            infeasible=True,
        )
    u_star = float(np.clip(u_prop, L, U))
    projected = not np.isclose(u_star, u_prop)
    proj_mag = abs(u_star - u_prop)
    return QPResult(
        action=u_star,
        L=L,
        U=U,
        projected=projected,
        proj_mag=proj_mag,
        feasible=True,
        infeasible=False,
    )
