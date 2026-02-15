"""Environment runner utilities for Phase 5."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

from .config import Phase5Config


@dataclass
class EnvState:
    x: float
    v: float
    r_feasible: float
    sat_prev: float
    u_buffer: List[float]

    def copy(self) -> "EnvState":
        return EnvState(
            x=self.x,
            v=self.v,
            r_feasible=self.r_feasible,
            sat_prev=self.sat_prev,
            u_buffer=list(self.u_buffer),
        )


@dataclass
class SimTrace:
    errors: np.ndarray
    sats: np.ndarray
    r_vals: np.ndarray
    r_feasible_vals: np.ndarray
    dr_max_vals: np.ndarray
    a_scale_vals: np.ndarray


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def reference_signal(ref_type: str, ref_param, t_sec: float) -> float:
    if ref_type == "step":
        return 0.0 if t_sec < 2.0 else ref_param
    if ref_type == "ramp":
        slope = ref_param
        return max(0.0, min(0.75, slope * max(0.0, t_sec - 2.0)))
    if ref_type == "sine":
        amp, freq = ref_param
        return amp * math.sin(2 * math.pi * freq * t_sec)
    if ref_type == "custom":
        seq = ref_param["seq"]
        dt = ref_param.get("dt", 0.01)
        idx = min(int(t_sec / dt), len(seq) - 1)
        return float(seq[idx])
    return 0.0


def initial_state(config: Phase5Config) -> EnvState:
    return EnvState(
        x=0.0,
        v=0.0,
        r_feasible=0.0,
        sat_prev=0.0,
        u_buffer=[0.0 for _ in range(config.delay_steps + 1)],
    )


def step_dynamics(
    state: EnvState,
    r_use: float,
    dist_scale: float,
    config: Phase5Config,
    t_step: int,
) -> Tuple[EnvState, float, int]:
    e = r_use - state.x
    u_cmd = config.kp * e - config.kd * state.v
    u_cmd_clip = clamp(u_cmd, -config.u_max, config.u_max)
    sat = 1 if (u_cmd_clip != u_cmd) else 0

    state.u_buffer.append(u_cmd_clip)
    u_applied = state.u_buffer.pop(0)

    d_quad = config.c_true * state.v * abs(state.v)
    bias = (3.0 * dist_scale) if t_step < config.steps // 2 else (-3.0 * dist_scale)

    a_true = (u_applied - config.friction * state.v - d_quad + bias) / config.mass
    v_next = state.v + a_true * config.dt
    x_next = state.x + v_next * config.dt

    next_state = EnvState(
        x=x_next,
        v=v_next,
        r_feasible=state.r_feasible,
        sat_prev=float(sat),
        u_buffer=state.u_buffer,
    )
    return next_state, e, sat


def rollout(
    ref_type: str,
    ref_param,
    dist_scale: float,
    config: Phase5Config,
    policy_fn: Callable[[float, EnvState, float, Phase5Config], Tuple[float, EnvState]],
) -> SimTrace:
    state = initial_state(config)
    errors = []
    sats = []
    r_vals = []
    r_feasible_vals = []
    dr_max_vals = []
    a_scale_vals = []

    for t in range(config.steps):
        r = reference_signal(ref_type, ref_param, t * config.dt)
        r_use, state = policy_fn(r, state, dist_scale, config)
        state, e, sat = step_dynamics(state, r_use, dist_scale, config, t)
        if hasattr(state, "sat_hist"):
            state.sat_hist.append(sat)
        errors.append(e)
        sats.append(sat)
        r_vals.append(r)
        r_feasible_vals.append(state.r_feasible)
        dr_max_vals.append(getattr(state, "dr_max", config.dr_cap))
        a_scale_vals.append(getattr(state, "a_scale", config.a_max))

    return SimTrace(
        errors=np.array(errors),
        sats=np.array(sats),
        r_vals=np.array(r_vals),
        r_feasible_vals=np.array(r_feasible_vals),
        dr_max_vals=np.array(dr_max_vals),
        a_scale_vals=np.array(a_scale_vals),
    )
