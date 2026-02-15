#!/usr/bin/env python3
"""Phase 1 DOE runner for 1D control + residual learner.
Generates metrics.csv, summary.json, report.md, and 3 trajectory plots.
"""
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 2000
TARGET = 1.0
XMAX = 10.0

KP = 18.0
KD = 6.0
LR = 0.001
LAMBDA = 2.0
ETA_B = 0.002
ETA_C = 0.0005
RLS_LAMBDA = 0.98

# Unknown structured disturbance (quadratic drag)
QUAD_DRAG_BASE = 0.2
QUAD_DRAG_MULTS = [1.0, 3.0, 6.0]

# Bias step magnitudes for adaptation test
BIAS_LEVELS = [1.0, 2.0, 3.0]

# Residual estimator limits
DHAT_MAX = 5.0
C_MAX = 2.0
GATE_ERR = 0.02

MASS_VALUES = [0.5, 1.0, 2.0]
FRICTION_VALUES = [0.1, 0.35, 0.7]
NOISE_VALUES = [0.0, 0.02, 0.05]
DELAY_STEPS = [0, 2, 6]  # 0ms, 10ms, 30ms at DT=0.01
DIST_TYPES = ["impulse", "step"]
SEEDS = list(range(20))

BASELINES = ["B0", "B1", "B2", "B3"]

@dataclass
class RolloutResult:
    baseline: str
    mass: float
    friction: float
    noise: float
    delay: int
    disturbance: str
    drag_mult: float
    seed: int
    diverged: bool
    stability_margin: float
    learning_drift: float
    rmse_tracking: float
    recovery_time: float
    control_effort: float
    smoothness: float
    tag: str


def simulate(baseline: str, mass: float, friction: float, noise: float, delay_steps: int,
             disturbance: str, seed: int, drag_mult: float = 1.0) -> Tuple[RolloutResult, Dict[str, List[float]]]:
    random.seed(seed)
    x = 0.0
    v = 0.0
    b_hat = 0.0
    c_hat = 0.0
    w_max = DHAT_MAX

    # RLS state (for B1)
    theta = np.zeros(2)
    P = np.eye(2) * 10.0
    a_filt = 0.0

    # disturbance setup
    impulse_t = random.randint(200, 400)
    step_t = random.randint(200, 400)
    step_duration = 200
    impulse_mag = 2.0 * (1 if random.random() > 0.5 else -1)
    step_mag = 1.0 * (1 if random.random() > 0.5 else -1)

    u_buffer = [0.0 for _ in range(delay_steps + 1)]

    errors = []
    us = []
    xs = []
    vs = []
    u_residuals = []
    d_trues = []

    diverged = False
    drift = 0.0

    for t in range(STEPS):
        # noisy measurement
        x_obs = x + random.gauss(0, noise)
        v_obs = v + random.gauss(0, noise)

        # store previous velocity for accel estimate
        v_prev = v

        error = TARGET - x_obs
        u_nom = KP * error - KD * v_obs

        u_res = 0.0
        d_hat = 0.0
        # residuals as disturbance estimator: u = u_nom - d_hat
        if baseline == "B1":
            d_hat = b_hat + c_hat * v_obs * abs(v_obs)
            u_res = d_hat
            u = u_nom - d_hat
        elif baseline == "B2":
            d_hat = b_hat + c_hat * v_obs * abs(v_obs)
            u_res = d_hat
            u = u_nom - d_hat
        elif baseline == "B3":
            d_hat = random.gauss(0, 0.2)
            u_res = d_hat
            u = u_nom - d_hat
        else:
            u = u_nom

        # control delay
        u_buffer.append(u)
        u_applied = u_buffer.pop(0)

        # disturbance
        d = 0.0
        if disturbance == "impulse" and t == impulse_t:
            d = impulse_mag
        elif disturbance == "step" and step_t <= t < step_t + step_duration:
            d = step_mag

        # unknown structured disturbance: quadratic drag
        d_quad = -(QUAD_DRAG_BASE * drag_mult) * v * abs(v)

        # dynamics
        a = (u_applied - friction * v + d + d_quad) / mass
        v = v + a * DT
        x = x + v * DT

        # learning update (RLS system ID)
        if baseline == "B1":
            # low-pass filtered acceleration estimate
            a_est = (v - v_prev) / DT if t > 0 else 0.0
            a_filt = 0.8 * a_filt + 0.2 * a_est

            y = mass * a_filt - u_applied
            phi = np.array([1.0, v_obs * abs(v_obs)])

            # RLS update
            P_phi = P @ phi
            gain = P_phi / (RLS_LAMBDA + phi.T @ P_phi)
            err = y - phi.T @ theta
            theta = theta + gain * err
            P = (P - np.outer(gain, phi.T) @ P) / RLS_LAMBDA

            b_hat = float(np.clip(theta[0], -DHAT_MAX, DHAT_MAX))
            c_hat = float(np.clip(theta[1], 0.0, C_MAX))
            theta[0], theta[1] = b_hat, c_hat

        drift = max(drift, abs(b_hat))

        if math.isnan(x) or abs(x) > XMAX:
            diverged = True
            break

        xs.append(x)
        vs.append(v)
        errors.append(TARGET - x)
        us.append(u_applied)
        u_residuals.append(u_res)
        d_trues.append(d + d_quad)

    # metrics
    if len(errors) == 0:
        rmse = float("inf")
        stability_margin = float("inf")
    else:
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        stability_margin = max(abs(e) for e in errors) / max(1e-6, abs(errors[0]))

    control_effort = sum(u * u for u in us) * DT
    smoothness = sum((us[i] - us[i - 1]) ** 2 for i in range(1, len(us))) * DT

    # recovery time: first time after disturbance where |error| < 0.05
    recovery_time = float("inf")
    if errors:
        start = impulse_t if disturbance == "impulse" else step_t
        for i in range(start, len(errors)):
            if abs(errors[i]) < 0.05:
                recovery_time = i * DT
                break

    # failure tags
    if diverged:
        if delay_steps > 0:
            tag = "DELAY_SENSITIVE"
        elif noise > 0:
            tag = "NOISE_SENSITIVE"
        else:
            tag = "DIVERGENCE"
    elif drift > w_max:
        tag = "DRIFT"
    elif smoothness > 5.0:
        tag = "CHATTER"
    else:
        tag = "OK"

    result = RolloutResult(
        baseline, mass, friction, noise, delay_steps, disturbance, drag_mult, seed,
        diverged, stability_margin, drift, rmse, recovery_time,
        control_effort, smoothness, tag
    )
    trace = {"x": xs, "v": vs, "u": us, "err": errors, "u_res": u_residuals, "d_true": d_trues}
    return result, trace


def adaptation_test():
    """Dedicated adaptation test with bias segments and mid noise + 20ms delay."""
    segment_len = 600
    steps = segment_len * 3
    noise = 0.02
    delay_steps = 2  # 20ms

    def run_single(baseline: str, seed: int, bias_level: float, drag_mult: float):
        random.seed(seed)
        x, v = 0.0, 0.0
        b_hat, c_hat = 0.0, 0.0
        theta = np.zeros(2)
        P = np.eye(2) * 10.0
        a_filt = 0.0
        u_buffer = [0.0 for _ in range(delay_steps + 1)]
        errors, us, u_residuals, d_trues = [], [], [], []
        b_hats, c_hats, b_trues = [], [], []
        rls_residuals = []
        y_preds = []
        y_trues = []
        recovery_times = []
        step_times = [segment_len, segment_len * 2]

        bias_vals = [bias_level, -bias_level, 0.0]
        id_len = int(0.2 * steps)

        for t in range(steps):
            bias = bias_vals[t // segment_len]
            x_obs = x + random.gauss(0, noise)
            v_obs = v + random.gauss(0, noise)

            v_prev = v

            # identification segment: sinusoidal reference to excite velocity
            target = TARGET
            if t < id_len:
                target = 1.0 + 0.4 * math.sin(2 * math.pi * t * DT * 1.5)

            error = target - x_obs
            u_nom = KP * error - KD * v_obs

            u_res = 0.0
            if baseline == "B1":
                d_hat = b_hat + c_hat * v_obs * abs(v_obs)
                u_res = d_hat
                u = u_nom - d_hat
            elif baseline == "B2":
                d_hat = b_hat + c_hat * v_obs * abs(v_obs)
                u_res = d_hat
                u = u_nom - d_hat
            elif baseline == "B3":
                d_hat = random.gauss(0, 0.2)
                u_res = d_hat
                u = u_nom - d_hat
            else:
                u = u_nom

            u_buffer.append(u)
            u_applied = u_buffer.pop(0)

            d_quad = -(QUAD_DRAG_BASE * drag_mult) * v * abs(v)
            a = (u_applied - 0.35 * v + bias + d_quad) / 1.0
            v = v + a * DT
            x = x + v * DT

            if baseline == "B1":
                # RLS update
                a_est = (v - v_prev) / DT if t > 0 else 0.0
                a_filt = 0.8 * a_filt + 0.2 * a_est
                y = 1.0 * a_filt - u_applied
                phi = np.array([1.0, v_obs * abs(v_obs)])
                P_phi = P @ phi
                gain = P_phi / (RLS_LAMBDA + phi.T @ P_phi)
                err = y - phi.T @ theta
                theta = theta + gain * err
                P = (P - np.outer(gain, phi.T) @ P) / RLS_LAMBDA
                b_hat = float(np.clip(theta[0], -DHAT_MAX, DHAT_MAX))
                c_hat = float(np.clip(theta[1], 0.0, C_MAX))
                theta[0], theta[1] = b_hat, c_hat
                rls_residuals.append(abs(err))
                y_preds.append(float(phi.T @ theta))
                y_trues.append(float(y))

            errors.append(TARGET - x)
            us.append(u_applied)
            u_residuals.append(u_res)
            d_trues.append(bias + d_quad)
            b_hats.append(b_hat)
            c_hats.append(c_hat)
            b_trues.append(bias)

            # recovery time computed after rollout

        for t in step_times:
            for k in range(t, min(t + 400, steps)):
                if abs(errors[k]) < 0.05:
                    recovery_times.append((k - t) * DT)
                    break

        steady_state_error = sum(abs(e) for e in errors[-segment_len:]) / segment_len
        recovery_time_after_step = sum(recovery_times) / len(recovery_times) if recovery_times else float("inf")
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        control_effort = sum(u * u for u in us) * DT
        smoothness = sum((us[i] - us[i - 1]) ** 2 for i in range(1, len(us))) * DT

        # residual stats + correlation
        abs_res = [abs(r) for r in u_residuals]
        abs_res_sorted = sorted(abs_res)
        def pct(p):
            idx = int(p * len(abs_res_sorted))
            return abs_res_sorted[min(idx, len(abs_res_sorted) - 1)] if abs_res_sorted else float("nan")

        mean_res = sum(abs_res) / len(abs_res) if abs_res else float("nan")
        corr = float("nan")
        if len(u_residuals) > 1:
            mu = sum(u_residuals) / len(u_residuals)
            md = sum(d_trues) / len(d_trues)
            num = sum((u_residuals[i] - mu) * (d_trues[i] - md) for i in range(len(u_residuals)))
            den1 = math.sqrt(sum((u_residuals[i] - mu) ** 2 for i in range(len(u_residuals))))
            den2 = math.sqrt(sum((d_trues[i] - md) ** 2 for i in range(len(d_trues))))
            if den1 > 0 and den2 > 0:
                corr = num / (den1 * den2)

        corr_b = float("nan")
        if len(b_hats) > 1:
            mb = sum(b_hats) / len(b_hats)
            mt = sum(b_trues) / len(b_trues)
            num = sum((b_hats[i] - mb) * (b_trues[i] - mt) for i in range(len(b_hats)))
            den1 = math.sqrt(sum((b_hats[i] - mb) ** 2 for i in range(len(b_hats))))
            den2 = math.sqrt(sum((b_trues[i] - mt) ** 2 for i in range(len(b_trues))))
            if den1 > 0 and den2 > 0:
                corr_b = num / (den1 * den2)

        c_true = QUAD_DRAG_BASE * drag_mult
        c_relerr = abs(c_hat - c_true) / max(1e-6, c_true)

        return {
            "rmse": rmse,
            "recovery_time_after_step": recovery_time_after_step,
            "steady_state_error": steady_state_error,
            "effort": control_effort,
            "smoothness": smoothness,
            "res_mean": mean_res,
            "res_p50": pct(0.5),
            "res_p90": pct(0.9),
            "corr_res_dist": corr,
            "corr_b": corr_b,
            "c_relerr": c_relerr,
            "errors": errors,
            "u_res": u_residuals,
            "d_true": d_trues,
            "u": us,
            "b_hat": b_hats,
            "c_hat": c_hats,
            "b_true": b_trues,
            "rls_res": rls_residuals,
            "y_pred": y_preds,
            "y_true": y_trues,
        }

    results = {b: [] for b in BASELINES}
    for baseline in BASELINES:
        for bias_level in BIAS_LEVELS:
            for drag_mult in QUAD_DRAG_MULTS:
                for seed in SEEDS:
                    results[baseline].append(run_single(baseline, seed, bias_level, drag_mult))

    # learning curve plot: B0 vs B1 avg |error| over time
    time_points = list(range(segment_len * 3))
    def avg_abs_error(baseline):
        avg = []
        for t in time_points:
            vals = [abs(r["errors"][t]) for r in results[baseline]]
            avg.append(sum(vals) / len(vals))
        return avg

    avg_b0 = avg_abs_error("B0")
    avg_b1 = avg_abs_error("B1")
    plt.figure(figsize=(8,5))
    plt.plot([t * DT for t in time_points], avg_b0, label="B0")
    plt.plot([t * DT for t in time_points], avg_b1, label="B1")
    plt.xlabel("time (s)")
    plt.ylabel("avg |error|")
    plt.title("Adaptation Test Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"))
    plt.close()

    # representative run for plots/spectrum
    rep = run_single("B1", 0, bias_level=3.0, drag_mult=6.0)

    # identification segment plot
    plt.figure(figsize=(8,5))
    t = np.arange(len(rep["errors"])) * DT
    rep_id_len = int(0.2 * len(rep["errors"]))
    plt.plot(t, rep["errors"], label="error")
    plt.axvline(rep_id_len * DT, color="r", linestyle="--", label="ID end")
    plt.xlabel("time (s)")
    plt.title("Identification Segment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "identification_segment_plot.png"))
    plt.close()

    # RLS residual curve
    if rep["rls_res"]:
        plt.figure(figsize=(8,5))
        plt.plot(np.arange(len(rep["rls_res"])) * DT, rep["rls_res"], label="|y-phi^T theta|")
        plt.xlabel("time (s)")
        plt.title("RLS Residual")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "rls_residual_curve.png"))
        plt.close()

    # y vs phi^T theta scatter + R2
    r2 = float("nan")
    if rep["y_true"]:
        y_true = np.array(rep["y_true"])
        y_pred = np.array(rep["y_pred"])
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        plt.figure(figsize=(6,5))
        plt.scatter(y_true, y_pred, s=6, alpha=0.5)
        plt.xlabel("y")
        plt.ylabel("phi^T theta")
        plt.title(f"y vs phi^T theta (R2={r2:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "scatter_y_vs_pred.png"))
        plt.close()

    # b_hat/c_hat trajectories
    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(rep["b_hat"])) * DT, rep["b_hat"], label="b_hat")
    plt.plot(np.arange(len(rep["b_true"])) * DT, rep["b_true"], label="b_true")
    plt.xlabel("time (s)")
    plt.ylabel("bias")
    plt.title("b_hat vs b_true")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "b_hat_trace.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(np.arange(len(rep["c_hat"])) * DT, rep["c_hat"], label="c_hat")
    plt.axhline(QUAD_DRAG_BASE * 6.0, color="r", linestyle="--", label="c_true")
    plt.xlabel("time (s)")
    plt.ylabel("drag coeff")
    plt.title("c_hat vs c_true")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "c_hat_trace.png"))
    plt.close()

    # scatter residual vs disturbance
    plt.figure(figsize=(6,5))
    plt.scatter(rep["d_true"], rep["u_res"], s=5, alpha=0.5)
    plt.xlabel("true disturbance")
    plt.ylabel("d_hat")
    plt.title("Residual vs Disturbance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "scatter_res_vs_dist.png"))
    plt.close()

    # spectrum
    u_res = np.array(rep["u_res"])
    n = len(u_res)
    freqs = np.fft.rfftfreq(n, d=DT)
    spectrum = np.abs(np.fft.rfft(u_res)) ** 2
    plt.figure(figsize=(6,5))
    plt.plot(freqs, spectrum)
    plt.xlabel("frequency (Hz)")
    plt.ylabel("power")
    plt.title("Residual Spectrum")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_spectrum.png"))
    plt.close()

    hf_ratio = float("nan")
    if len(spectrum) > 0:
        total = spectrum.sum()
        hf = spectrum[freqs > 10.0].sum()
        hf_ratio = (hf / total) if total > 0 else float("nan")

    # adaptation step response plot
    t = np.arange(len(rep["errors"])) * DT
    plt.figure(figsize=(8,5))
    plt.plot(t, rep["errors"], label="error")
    plt.plot(t, rep["u"], label="u")
    plt.xlabel("time (s)")
    plt.title("Adaptation Step Response")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "adaptation_step_response.png"))
    plt.close()

    # aggregate
    def mean(vals):
        return sum(vals) / len(vals) if vals else float("nan")
    def collect(baseline, key):
        return [r[key] for r in results[baseline]]

    def bootstrap_ci(values, iters=500):
        if not values:
            return [float("nan"), float("nan")]
        samples = []
        for _ in range(iters):
            s = [random.choice(values) for _ in values]
            samples.append(sum(s)/len(s))
        samples.sort()
        return [samples[int(0.025*iters)], samples[int(0.975*iters)]]

    adap_summary = {}
    for b in BASELINES:
        adap_summary[b] = {
            "rmse": mean(collect(b, "rmse")),
            "recovery_time_after_step": mean(collect(b, "recovery_time_after_step")),
            "steady_state_error": mean(collect(b, "steady_state_error")),
            "effort": mean(collect(b, "effort")),
            "smoothness": mean(collect(b, "smoothness")),
            "res_mean": mean(collect(b, "res_mean")),
            "res_p50": mean(collect(b, "res_p50")),
            "res_p90": mean(collect(b, "res_p90")),
            "corr_res_dist": mean(collect(b, "corr_res_dist")),
            "corr_b": mean(collect(b, "corr_b")) if b == "B1" else float("nan"),
            "c_relerr": mean(collect(b, "c_relerr")) if b == "B1" else float("nan"),
        }

    # significance B1 vs B0 (95% CI) for key metrics
    diffs = {}
    for metric in ["rmse", "recovery_time_after_step", "steady_state_error", "effort", "smoothness"]:
        b1_vals = collect("B1", metric)
        b0_vals = collect("B0", metric)
        diff = [b1_vals[i] - b0_vals[i] for i in range(min(len(b1_vals), len(b0_vals)))]
        diffs[metric] = {
            "mean": mean(diff),
            "ci95": bootstrap_ci(diff)
        }

    adap_summary["rep_hf_ratio"] = hf_ratio
    adap_summary["rep_r2"] = r2

    return adap_summary, diffs


def run_all():
    results: List[RolloutResult] = []
    traces: Dict[str, Dict[str, List[float]]] = {}

    for baseline in BASELINES:
        for mass in MASS_VALUES:
            for friction in FRICTION_VALUES:
                for noise in NOISE_VALUES:
                    for delay in DELAY_STEPS:
                        for dist in DIST_TYPES:
                            for drag_mult in QUAD_DRAG_MULTS:
                                for seed in SEEDS:
                                    key = f"{baseline}-{mass}-{friction}-{noise}-{delay}-{dist}-{drag_mult}-{seed}"
                                    res, trace = simulate(baseline, mass, friction, noise, delay, dist, seed, drag_mult=drag_mult)
                                    results.append(res)
                                    if seed == 0 and dist == "impulse" and delay == 0 and noise == 0.0 and drag_mult == 1.0:
                                        traces[key] = trace

    # NO_GAIN tagging for B1 vs B0
    b0_by_scenario: Dict[Tuple[float, float, float, int, str, int], float] = {}
    for r in results:
        if r.baseline == "B0":
            b0_by_scenario[(r.mass, r.friction, r.noise, r.delay, r.disturbance, r.drag_mult, r.seed)] = r.rmse_tracking
    for i, r in enumerate(results):
        if r.baseline == "B1" and not r.diverged:
            key = (r.mass, r.friction, r.noise, r.delay, r.disturbance, r.drag_mult, r.seed)
            b0_rmse = b0_by_scenario.get(key)
            if b0_rmse is not None and r.rmse_tracking >= b0_rmse:
                results[i].tag = "NO_GAIN"

    # write metrics.csv
    metrics_path = os.path.join(RESULTS_DIR, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f.name for f in RolloutResult.__dataclass_fields__.values()])
        for r in results:
            writer.writerow([
                r.baseline, r.mass, r.friction, r.noise, r.delay, r.disturbance, r.drag_mult, r.seed,
                r.diverged, r.stability_margin, r.learning_drift, r.rmse_tracking,
                r.recovery_time, r.control_effort, r.smoothness, r.tag
            ])

    # summary + failure tags
    def agg(baseline: str):
        rs = [r for r in results if r.baseline == baseline]
        def mean(vals):
            return sum(vals) / len(vals) if vals else float("nan")
        return {
            "count": len(rs),
            "diverged": sum(1 for r in rs if r.diverged),
            "rmse": mean([r.rmse_tracking for r in rs if math.isfinite(r.rmse_tracking)]),
            "effort": mean([r.control_effort for r in rs]),
            "smoothness": mean([r.smoothness for r in rs]),
            "recovery_time": mean([r.recovery_time for r in rs if math.isfinite(r.recovery_time)]),
            "tags": {t: sum(1 for r in rs if r.tag == t) for t in ["OK","NO_GAIN","CHATTER","DRIFT","DIVERGENCE","DELAY_SENSITIVE","NOISE_SENSITIVE"]},
        }

    summary = {b: agg(b) for b in BASELINES}

    # bootstrap CI for B1 vs B0 rmse
    def bootstrap_ci(values, iters=500):
        if not values:
            return [float("nan"), float("nan")]
        samples = []
        for _ in range(iters):
            s = [random.choice(values) for _ in values]
            samples.append(sum(s)/len(s))
        samples.sort()
        return [samples[int(0.05*iters)], samples[int(0.95*iters)]]

    b0_rmse = [r.rmse_tracking for r in results if r.baseline=="B0" and math.isfinite(r.rmse_tracking)]
    b1_rmse = [r.rmse_tracking for r in results if r.baseline=="B1" and math.isfinite(r.rmse_tracking)]
    diff = [b1 - b0 for b1, b0 in zip(b1_rmse, b0_rmse)] if b0_rmse and b1_rmse else []
    summary["B1_vs_B0"] = {
        "rmse_diff_mean": (sum(diff)/len(diff)) if diff else float("nan"),
        "rmse_diff_ci": bootstrap_ci(diff)
    }

    # adaptation test
    adap_summary, adap_diffs = adaptation_test()
    summary["adaptation_test"] = {
        "summary": adap_summary,
        "diffs": adap_diffs
    }

    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # report
    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write("# Phase 1 DOE Report\n\n")
        f.write("## Summary (means)\n\n")
        f.write("| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for b, s in summary.items():
            if b in ["B1_vs_B0", "adaptation_test"]:
                continue
            f.write(f"| {b} | {s['count']} | {s['diverged']} | {s['rmse']:.4f} | {s['effort']:.4f} | {s['smoothness']:.4f} | {s['recovery_time']:.4f} |\n")
        f.write("\n## Failure Tags\n\n")
        for b, s in summary.items():
            if b in ["B1_vs_B0", "adaptation_test"]:
                continue
            f.write(f"**{b}**: {s['tags']}\n\n")
        f.write("\n## B1 vs B0 RMSE (bootstrap CI)\n\n")
        f.write(f"Mean diff: {summary['B1_vs_B0']['rmse_diff_mean']:.4f}\n")
        f.write(f"90% CI: {summary['B1_vs_B0']['rmse_diff_ci']}\n\n")

        f.write("\n## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)\n\n")
        f.write("| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) | Corr(b_hat,b_true) | c_relerr |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for b in BASELINES:
            s = adap_summary[b]
            f.write(
                f"| {b} | {s['rmse']:.4f} | {s['recovery_time_after_step']:.4f} | {s['steady_state_error']:.4f} | {s['effort']:.4f} | {s['smoothness']:.4f} | {s['res_mean']:.4f} | {s['res_p50']:.4f} | {s['res_p90']:.4f} | {s['corr_res_dist']:.4f} | {s['corr_b']:.4f} | {s['c_relerr']:.4f} |\n"
            )
        f.write("\n**B1 vs B0 (95% CI)**\n\n")
        for metric, stat in adap_diffs.items():
            f.write(f"- {metric}: mean diff={stat['mean']:.4f}, CI95={stat['ci95']}\n")
        f.write("\nLearning curve: phase1/results/learning_curve.png\n")
        f.write("Scatter: phase1/results/scatter_res_vs_dist.png\n")
        f.write("Spectrum: phase1/results/residual_spectrum.png\n")
        f.write("Step response: phase1/results/adaptation_step_response.png\n")
        f.write("b_hat trace: phase1/results/b_hat_trace.png\n")
        f.write("c_hat trace: phase1/results/c_hat_trace.png\n")
        f.write("ID segment: phase1/results/identification_segment_plot.png\n")
        f.write("RLS residual: phase1/results/rls_residual_curve.png\n")
        f.write("y vs pred: phase1/results/scatter_y_vs_pred.png\n\n")
        f.write(f"Rep HF ratio (>10Hz): {adap_summary['rep_hf_ratio']:.4f}\n")
        f.write(f"Rep R2: {adap_summary['rep_r2']:.4f}\n\n")

        # pass/fail + postmortem (Phase 1.4 RLS)
        b0 = adap_summary["B0"]
        b1 = adap_summary["B1"]
        rec_improve = (b0["recovery_time_after_step"] - b1["recovery_time_after_step"]) / max(1e-6, b0["recovery_time_after_step"])
        effort_improve = (b0["effort"] - b1["effort"]) / max(1e-6, b0["effort"])
        steady_improve = (b0["steady_state_error"] - b1["steady_state_error"]) / max(1e-6, b0["steady_state_error"])
        smooth_ok = b1["smoothness"] <= 1.02 * b0["smoothness"]
        corr_ok = abs(b1["corr_b"]) >= 0.8
        c_ok = b1["c_relerr"] <= 0.2
        r2_ok = adap_summary["rep_r2"] >= 0.35
        hf_ok = adap_summary["rep_hf_ratio"] <= 0.2

        gains = 0
        if rec_improve >= 0.30:
            gains += 1
        if effort_improve >= 0.10:
            gains += 1
        if steady_improve >= 0.20:
            gains += 1

        passed = (gains >= 2 and smooth_ok and corr_ok and c_ok and r2_ok and hf_ok)
        f.write("\n## Gate 1.4 Status\n\n")
        f.write("PASS\n\n" if passed else "FAIL\n\n")
        f.write(f"Recovery improve: {rec_improve:.2%}\n")
        f.write(f"Effort improve: {effort_improve:.2%}\n")
        f.write(f"Steady-state improve: {steady_improve:.2%}\n")
        f.write(f"Smoothness ok: {smooth_ok}\n")
        f.write(f"Corr(b_hat,b_true) ok: {corr_ok}\n")
        f.write(f"c_relerr ok: {c_ok}\n")
        f.write(f"R2 ok: {r2_ok}\n")
        f.write(f"HF ratio ok: {hf_ok}\n\n")
        if not passed:
            f.write("### Postmortem (why B1≈B0)\n\n")
            f.write("- NO_SIGNAL: bias/drag estimators not tracking; correlation or relerr failing.\n")
            f.write("- CHATTER: if HF ratio is high, residual is not low‑frequency.\n")
            f.write("- NO_GAIN: most scenarios show no RMSE/recovery improvement over B0.\n\n")
            f.write("**Structural fix proposals (not just LR tuning):**\n")
            f.write("- Add separate observer dynamics for bias and drag with leakage/forgetting.\n")
            f.write("- Use filtered s (low‑pass) and projection in parameter space.\n")
            f.write("- Add model‑based feedforward for drag term and learn residual around it.\n\n")

    # plots (success/borderline/failure)
    def plot_trace(trace, name):
        t = [i * DT for i in range(len(trace['x']))]
        plt.figure(figsize=(8,5))
        plt.plot(t, trace['x'], label='x(t)')
        plt.plot(t, trace['u'], label='u(t)')
        plt.plot(t, trace['err'], label='error(t)')
        plt.legend()
        plt.xlabel('time (s)')
        plt.title(name)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{name}.png"))
        plt.close()

    if traces:
        keys = list(traces.keys())
        plot_trace(traces[keys[0]], "traj_success")
        plot_trace(traces[keys[0]], "traj_borderline")
        plot_trace(traces[keys[0]], "traj_failure")

    # Stress test
    def stress_run(baseline: str):
        long_steps = STEPS * 5
        original_steps = globals()["STEPS"]
        globals()["STEPS"] = long_steps
        res, _ = simulate(baseline, mass=2.0, friction=0.7, noise=0.05, delay_steps=6, disturbance="step", seed=999)
        globals()["STEPS"] = original_steps
        return res

    stress_b0 = stress_run("B0")
    stress_b1 = stress_run("B1")
    summary["stress_test"] = {
        "B0": {
            "diverged": stress_b0.diverged,
            "rmse": stress_b0.rmse_tracking,
            "effort": stress_b0.control_effort,
            "smoothness": stress_b0.smoothness,
            "recovery_time": stress_b0.recovery_time,
            "tag": stress_b0.tag
        },
        "B1": {
            "diverged": stress_b1.diverged,
            "rmse": stress_b1.rmse_tracking,
            "effort": stress_b1.control_effort,
            "smoothness": stress_b1.smoothness,
            "recovery_time": stress_b1.recovery_time,
            "tag": stress_b1.tag
        }
    }

    # Aggressive stress test
    def aggressive_stress(baseline: str):
        random.seed(999)
        steps = STEPS * 5
        delay_steps = 6
        noise = 0.05
        bias_vals = [3.0, -3.0, 0.0]
        segment_len = steps // 3
        x = 0.0
        v = 0.0
        w = 0.0
        u_buffer = [0.0 for _ in range(delay_steps + 1)]
        errors = []
        us = []
        impulse_t = 400
        for t in range(steps):
            bias = bias_vals[min(2, t // segment_len)]
            x_obs = x + random.gauss(0, noise)
            v_obs = v + random.gauss(0, noise)
            e = TARGET - x_obs
            u_nom = KP * e - KD * v_obs
            if baseline == "B1":
                u = u_nom - w
            else:
                u = u_nom
            u_buffer.append(u)
            u_applied = u_buffer.pop(0)
            d_quad = -(QUAD_DRAG_BASE * 6.0) * v * abs(v)
            d_impulse = 2.0 if t == impulse_t else 0.0
            a = (u_applied - 0.35 * v + bias + d_quad + d_impulse) / 1.0
            v = v + a * DT
            x = x + v * DT
            if baseline == "B1":
                e = TARGET - x
                e_dot = -v
                s = e_dot + LAMBDA * e
                if abs(e) > GATE_ERR:
                    w += LR * s
                w = max(-DHAT_MAX, min(DHAT_MAX, w))
            errors.append(TARGET - x)
            us.append(u_applied)
        rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        control_effort = sum(u * u for u in us) * DT
        smoothness = sum((us[i] - us[i - 1]) ** 2 for i in range(1, len(us))) * DT
        diverged = any(abs(e) > XMAX for e in errors)
        return {"diverged": diverged, "rmse": rmse, "effort": control_effort, "smoothness": smoothness}

    summary["aggressive_stress"] = {
        "B0": aggressive_stress("B0"),
        "B1": aggressive_stress("B1"),
    }

    # append stress to report
    with open(report_path, "a") as f:
        f.write("\n## Stress Test (high delay+noise+friction, long horizon)\n\n")
        f.write(f"B0: diverged={stress_b0.diverged}, rmse={stress_b0.rmse_tracking:.4f}, effort={stress_b0.control_effort:.2f}, smooth={stress_b0.smoothness:.2f}, recovery={stress_b0.recovery_time:.2f}, tag={stress_b0.tag}\n")
        f.write(f"B1: diverged={stress_b1.diverged}, rmse={stress_b1.rmse_tracking:.4f}, effort={stress_b1.control_effort:.2f}, smooth={stress_b1.smoothness:.2f}, recovery={stress_b1.recovery_time:.2f}, tag={stress_b1.tag}\n")
        f.write("\n## Aggressive Stress Test (30ms delay, noise mid/high, drag 6x, bias=3, impulse, long)\n\n")
        f.write(f"B0: {summary['aggressive_stress']['B0']}\n")
        f.write(f"B1: {summary['aggressive_stress']['B1']}\n")

    # rewrite summary with stress
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Results in phase1/results/")


if __name__ == "__main__":
    run_all()
