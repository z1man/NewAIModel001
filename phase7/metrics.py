"""Metrics and gate evaluation for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Phase7Config


@dataclass
class Phase7Metrics:
    rmse: float
    mean_abs: float
    max_abs: float
    in_band: float
    sat_rate: float
    qp_violation_rate: float
    token_clip_rate: float


def compute_metrics(
    errors: np.ndarray,
    sat_flags: np.ndarray,
    qp_violations: np.ndarray,
    token_clips: np.ndarray,
    config: Phase7Config,
) -> Phase7Metrics:
    rmse = float(np.sqrt(np.mean(errors**2)))
    mean_abs = float(np.mean(np.abs(errors)))
    max_abs = float(np.max(np.abs(errors)))
    in_band = float(np.mean(np.abs(errors) <= config.band))
    sat_rate = float(np.mean(sat_flags))
    qp_violation_rate = float(np.mean(qp_violations))
    token_clip_rate = float(np.mean(token_clips))
    return Phase7Metrics(
        rmse=rmse,
        mean_abs=mean_abs,
        max_abs=max_abs,
        in_band=in_band,
        sat_rate=sat_rate,
        qp_violation_rate=qp_violation_rate,
        token_clip_rate=token_clip_rate,
    )


def metrics_to_dict(metrics: Phase7Metrics) -> Dict[str, float]:
    return {
        "rmse": metrics.rmse,
        "mean_abs": metrics.mean_abs,
        "max_abs": metrics.max_abs,
        "in_band": metrics.in_band,
        "sat_rate": metrics.sat_rate,
        "qp_violation_rate": metrics.qp_violation_rate,
        "token_clip_rate": metrics.token_clip_rate,
    }


def evaluate_gates(metrics: Phase7Metrics, config: Phase7Config) -> Dict[str, Dict[str, float | bool]]:
    gates = {
        "M1_tokens": {
            "pass": metrics.token_clip_rate <= 0.0,
            "rate": metrics.token_clip_rate,
            "limit": 0.0,
        },
        "M2_qp": {
            "pass": metrics.qp_violation_rate <= config.m2_violation_max,
            "rate": metrics.qp_violation_rate,
            "limit": config.m2_violation_max,
        },
        "M3_rmse": {
            "pass": metrics.rmse <= config.m3_rmse_max,
            "value": metrics.rmse,
            "limit": config.m3_rmse_max,
        },
        "M4_sat": {
            "pass": metrics.sat_rate <= config.m4_sat_max,
            "rate": metrics.sat_rate,
            "limit": config.m4_sat_max,
        },
        "M5_kpi": {
            "pass": (metrics.mean_abs <= config.m5_mean_abs_max)
            and (metrics.in_band >= config.m5_in_band_min),
            "mean_abs": metrics.mean_abs,
            "in_band": metrics.in_band,
            "mean_abs_limit": config.m5_mean_abs_max,
            "in_band_min": config.m5_in_band_min,
        },
    }
    return gates
