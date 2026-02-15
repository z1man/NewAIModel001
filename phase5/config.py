"""Phase 5 config helpers."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Phase5Config:
    run_name: str = "phase5_lcg_prod"
    dt: float = 0.01
    steps: int = 2000
    delay_steps: int = 2
    mass: float = 1.0
    friction: float = 0.35
    quad_drag_base: float = 0.2
    drag_mult: float = 6.0
    kp: float = 18.0
    kd: float = 6.0
    u_max: float = 1.0
    recovery_eps: float = 0.05
    dist_scales: List[float] = None
    dr_min: float = 0.05
    dr_cap: float = 2.0
    a_min: float = 0.4
    a_max: float = 1.0
    teacher_horizon: int = 5
    teacher_sat_penalty: float = 5.0
    teacher_shape_penalty: float = 0.5
    dr_grid: List[float] = None
    a_grid: List[float] = None
    train_epochs: int = 80
    train_lr: float = 5e-4
    train_l2: float = 5e-4
    dagger_epochs: int = 40
    dagger_lr: float = 2e-4
    dagger_l2: float = 5e-4
    dagger_iterations: int = 1
    eval_sine_amps: List[float] = None
    eval_sine_freqs: List[float] = None

    def __post_init__(self) -> None:
        if self.dist_scales is None:
            self.dist_scales = [0.15, 0.25, 0.35]
        if self.dr_grid is None:
            self.dr_grid = [0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
        if self.a_grid is None:
            self.a_grid = [0.4, 0.6, 0.8, 1.0]
        if self.eval_sine_amps is None:
            self.eval_sine_amps = [0.25, 0.5, 0.75]
        if self.eval_sine_freqs is None:
            self.eval_sine_freqs = [0.1, 0.2, 0.3, 0.5, 1.0]

    @property
    def c_true(self) -> float:
        return self.quad_drag_base * self.drag_mult


def load_config(path: str | Path) -> Phase5Config:
    with open(path, "r") as f:
        raw = json.load(f)
    return Phase5Config(**raw)


def save_config(config: Phase5Config, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
