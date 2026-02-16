"""Metrics and gate checks for Phase 7.1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Phase7Config


@dataclass
class MetricResult:
    rmse: float
    mean_abs: float
    max_abs: float
    in_band: float
    sat_rate: float
    hard_violation_rate: float
    infeasible_rate: float
    proj_mean: float
    proj_p90: float


def compute_metrics(
    errors: np.ndarray,
    sat_flags: np.ndarray,
    hard_violation_flags: np.ndarray,
    infeasible_flags: np.ndarray,
    proj_mag: np.ndarray,
    config: Phase7Config,
) -> MetricResult:
    rmse = math.sqrt(float(np.mean(errors**2)))
    mean_abs = float(np.mean(np.abs(errors)))
    max_abs = float(np.max(np.abs(errors)))
    in_band = float(np.mean(np.abs(errors) < config.band))
    sat_rate = float(np.mean(sat_flags))
    hard_violation_rate = float(np.mean(hard_violation_flags))
    infeasible_rate = float(np.mean(infeasible_flags))
    proj_mean = float(np.mean(proj_mag))
    proj_p90 = float(np.quantile(proj_mag, 0.9))

    return MetricResult(
        rmse=rmse,
        mean_abs=mean_abs,
        max_abs=max_abs,
        in_band=in_band,
        sat_rate=sat_rate,
        hard_violation_rate=hard_violation_rate,
        infeasible_rate=infeasible_rate,
        proj_mean=proj_mean,
        proj_p90=proj_p90,
    )


def metrics_to_dict(metrics: MetricResult) -> Dict[str, float]:
    return {
        "rmse": metrics.rmse,
        "mean_abs": metrics.mean_abs,
        "max_abs": metrics.max_abs,
        "in_band": metrics.in_band,
        "sat_rate": metrics.sat_rate,
        "hard_violation_rate": metrics.hard_violation_rate,
        "infeasible_rate": metrics.infeasible_rate,
        "proj_mean": metrics.proj_mean,
        "proj_p90": metrics.proj_p90,
    }


def evaluate_gates(metrics: MetricResult, config: Phase7Config) -> Dict[str, Dict[str, float | bool]]:
    gates = {
        "M1_hard": {
            "rate": metrics.hard_violation_rate,
            "limit": 0.0,
            "pass": metrics.hard_violation_rate <= 0.0,
        },
        "M2_infeasible": {
            "rate": metrics.infeasible_rate,
            "limit": config.m2_violation_max,
            "pass": metrics.infeasible_rate <= config.m2_violation_max,
        },
        "M3_rmse": {
            "value": metrics.rmse,
            "limit": config.m3_rmse_max,
            "pass": metrics.rmse <= config.m3_rmse_max,
        },
        "M4_sat": {
            "rate": metrics.sat_rate,
            "limit": config.m4_sat_max,
            "pass": metrics.sat_rate <= config.m4_sat_max,
        },
        "M5_kpi": {
            "mean_abs": metrics.mean_abs,
            "in_band": metrics.in_band,
            "mean_abs_limit": config.m5_mean_abs_max,
            "in_band_min": config.m5_in_band_min,
            "pass": metrics.mean_abs <= config.m5_mean_abs_max
            and metrics.in_band >= config.m5_in_band_min,
        },
    }
    return gates
