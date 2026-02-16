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
    slack: float = 0.0


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


def project_qp_soft_acc(
    u_prop: float,
    hard_bounds: Tuple[float, float],
    soft_bounds: Tuple[float, float],
    rho: float,
) -> QPResult:
    L_soft, U_soft = soft_bounds
    L_hard, U_hard = hard_bounds
    if L_hard > U_hard:
        u_star = float(np.clip(u_prop, L_hard, U_hard))
        proj_mag = abs(u_star - u_prop)
        return QPResult(
            action=u_star,
            L=L_hard,
            U=U_hard,
            projected=True,
            proj_mag=proj_mag,
            feasible=False,
            infeasible=True,
            slack=0.0,
        )

    if u_prop < L_soft:
        u_candidate = (u_prop + rho * L_soft) / (1.0 + rho)
        u_star = min(u_candidate, L_soft)
        xi = L_soft - u_star
    elif u_prop > U_soft:
        u_candidate = (u_prop + rho * U_soft) / (1.0 + rho)
        u_star = max(u_candidate, U_soft)
        xi = u_star - U_soft
    else:
        u_star = u_prop
        xi = 0.0

    u_star = float(np.clip(u_star, L_hard, U_hard))
    proj_mag = abs(u_star - u_prop)
    projected = not np.isclose(u_star, u_prop)
    return QPResult(
        action=u_star,
        L=L_soft,
        U=U_soft,
        projected=projected,
        proj_mag=proj_mag,
        feasible=True,
        infeasible=False,
        slack=float(max(0.0, xi)),
    )
