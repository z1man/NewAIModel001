"""Teacher policies for Phase 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Phase6Config
from .env_runner import EnvParams, EnvState, clamp, reference_signal, step_dynamics


@dataclass
class ActionGrid:
    actions: List[Tuple[float, float]]

    @classmethod
    def from_config(cls, config: Phase6Config) -> "ActionGrid":
        actions = [(dr, a) for dr in config.dr_grid for a in config.a_grid]
        return cls(actions=actions)


def apply_action(
    r: float, state: EnvState, dr_max: float, a_scale: float, config: Phase6Config
) -> Tuple[float, EnvState]:
    r_scaled = a_scale * r
    dr = clamp(r_scaled - state.r_feasible, -dr_max * config.dt, dr_max * config.dt)
    r_feasible_next = state.r_feasible + dr
    next_state = state.copy()
    next_state.r_feasible = r_feasible_next
    setattr(next_state, "dr_max", dr_max)
    setattr(next_state, "a_scale", a_scale)
    return clamp(r_feasible_next, -1.0, 1.0), next_state


def horizon_teacher_action(
    r: float,
    state: EnvState,
    dist_scale: float,
    config: Phase6Config,
    params: EnvParams,
    action_grid: ActionGrid,
    t_step: int,
    ref_type: str,
    ref_param,
) -> Tuple[int, float, EnvState]:
    best_cost = float("inf")
    best_idx = 0
    best_r_use = r
    best_state = state.copy()

    for idx, (dr_max, a_scale) in enumerate(action_grid.actions):
        sim_state = state.copy()
        r_use, sim_state = apply_action(r, sim_state, dr_max, a_scale, config)
        sim_state, err, sat = step_dynamics(sim_state, r_use, dist_scale, config, params, t_step)
        state_after_step = sim_state.copy()
        cost = err**2 + config.teacher_sat_penalty * (sat**2)
        cost += config.teacher_shape_penalty * (r_use - sim_state.r_feasible) ** 2

        for h in range(1, config.teacher_horizon):
            r_future = reference_signal(ref_type, ref_param, (t_step + h) * config.dt)
            r_use_h, sim_state = apply_action(r_future, sim_state, dr_max, a_scale, config)
            sim_state, err_h, sat_h = step_dynamics(
                sim_state, r_use_h, dist_scale, config, params, t_step + h
            )
            cost += err_h**2 + config.teacher_sat_penalty * (sat_h**2)

        if cost < best_cost:
            best_cost = cost
            best_idx = idx
            best_r_use = r_use
            best_state = state_after_step

    return best_idx, best_r_use, best_state


def action_to_params(action_grid: ActionGrid, action_idx: int) -> Tuple[float, float]:
    return action_grid.actions[action_idx]


def action_targets(action_grid: ActionGrid) -> np.ndarray:
    return np.array(action_grid.actions)
