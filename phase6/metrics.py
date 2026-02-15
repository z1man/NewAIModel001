"""Metrics and KPI calculations for Phase 6."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Phase6Config
from .env_runner import SimTrace


@dataclass
class KPIResult:
    sat_total: float
    rmse: float
    mean_abs: float
    recovery: float
    in_band: float
    max_abs: float
    mean_ref_diff: float
    rms_rf_over_r: float
    pct_dr_min: float
    pct_a_min: float


def compute_kpis(trace: SimTrace, config: Phase6Config) -> KPIResult:
    errors = trace.errors
    sats = trace.sats
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))
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
    ref_diff = np.abs(trace.r_vals - trace.r_feasible_vals)
    mean_ref_diff = float(np.mean(ref_diff))
    rms_rf = float(np.sqrt(np.mean(trace.r_feasible_vals**2)))
    rms_r = float(np.sqrt(np.mean(trace.r_vals**2)))
    rms_rf_over_r = rms_rf / rms_r if rms_r > 0 else float("nan")
    pct_dr_min = float(np.mean(trace.dr_max_vals <= config.dr_min + 1e-6))
    pct_a_min = float(np.mean(trace.a_scale_vals <= config.a_min + 1e-6))

    return KPIResult(
        sat_total=sat_total,
        rmse=rmse,
        mean_abs=mean_abs,
        recovery=recovery,
        in_band=in_band,
        max_abs=max_abs,
        mean_ref_diff=mean_ref_diff,
        rms_rf_over_r=rms_rf_over_r,
        pct_dr_min=pct_dr_min,
        pct_a_min=pct_a_min,
    )


def kpis_to_dict(kpis: KPIResult) -> Dict[str, float]:
    return {
        "sat_total": kpis.sat_total,
        "rmse": kpis.rmse,
        "mean_abs": kpis.mean_abs,
        "recovery": kpis.recovery,
        "in_band": kpis.in_band,
        "max_abs": kpis.max_abs,
        "mean_ref_diff": kpis.mean_ref_diff,
        "rms_rf_over_r": kpis.rms_rf_over_r,
        "pct_dr_min": kpis.pct_dr_min,
        "pct_a_min": kpis.pct_a_min,
    }
