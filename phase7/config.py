"""Phase 7 configuration utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Phase7Config:
    run_name: str = "phase7_token_qp"
    seed: int = 7
    dt: float = 0.01
    steps: int = 1200
    state_dim: int = 2
    action_dim: int = 1
    ref_amp: float = 0.8
    ref_freq: float = 0.2
    damping: float = 0.35
    spring: float = 0.25
    mass: float = 1.0
    kp: float = 18.0
    kd: float = 6.0
    kv_nom: float = 0.5
    u_min: List[float] = None
    u_max: List[float] = None
    a_max: float = 1.0
    token_clip: float = 3.0
    token_scale: float = 1.0
    band: float = 0.08
    m2_violation_max: float = 0.002
    m3_rmse_max: float = 0.12
    m4_sat_max: float = 0.12
    m5_mean_abs_max: float = 0.07
    m5_in_band_min: float = 0.7

    def __post_init__(self) -> None:
        if self.u_min is None:
            self.u_min = [-1.0]
        if self.u_max is None:
            self.u_max = [1.0]


def load_config(path: str | Path) -> Phase7Config:
    with open(path, "r") as f:
        raw = json.load(f)
    return Phase7Config(**raw)


def save_config(config: Phase7Config, path: str | Path) -> None:
    with open(path, "w") as f:
        json.dump(config.__dict__, f, indent=2)
