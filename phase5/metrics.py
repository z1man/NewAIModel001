"""Metrics and KPI calculations."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Phase5Config
from .env_runner import SimTrace


@dataclass
class KPIResult:
    sat_total: float
    rmse: float
    mean_abs: float
    recovery: float
    in_band: float
    max_abs: float


def compute_kpis(trace: SimTrace, config: Phase5Config) -> KPIResult:
    errors = trace.errors
    sats = trace.sats
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors ** 2)))
    mean_abs = float(np.mean(np.abs(errors)))
    max_abs = float(np.max(np.abs(errors)))

    recovery = float("inf")
    window = int(0.2 / config.dt)
    start = int(2.0 / config.dt)
    if len(errors) > start + window:
        for i in range(start, len(errors) - window):
            if np.all(np.abs(errors[i : i + window]) < config.recovery_eps):
                recovery = (i - start) * config.dt
                break

    in_band = float(np.mean(np.abs(errors) < config.recovery_eps))
    return KPIResult(
        sat_total=sat_total,
        rmse=rmse,
        mean_abs=mean_abs,
        recovery=recovery,
        in_band=in_band,
        max_abs=max_abs,
    )


def kpis_to_dict(kpis: KPIResult) -> Dict[str, float]:
    return {
        "sat_total": kpis.sat_total,
        "rmse": kpis.rmse,
        "mean_abs": kpis.mean_abs,
        "recovery": kpis.recovery,
        "in_band": kpis.in_band,
        "max_abs": kpis.max_abs,
    }
