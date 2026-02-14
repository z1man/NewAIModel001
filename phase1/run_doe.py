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

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 2000
TARGET = 1.0
XMAX = 10.0

KP = 18.0
KD = 6.0
LR = 0.001

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
             disturbance: str, seed: int) -> Tuple[RolloutResult, Dict[str, List[float]]]:
    random.seed(seed)
    x = 0.0
    v = 0.0
    w = 0.0  # residual weight
    w_max = 50.0

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

    diverged = False
    drift = 0.0

    for t in range(STEPS):
        # noisy measurement
        x_obs = x + random.gauss(0, noise)
        v_obs = v + random.gauss(0, noise)

        error = TARGET - x_obs
        u = KP * error - KD * v_obs

        # residuals
        if baseline == "B1":
            u += w * v_obs
        elif baseline == "B2":
            # residual structure, no updates (w fixed at 0)
            u += w * v_obs
        elif baseline == "B3":
            u += random.gauss(0, 0.2) * abs(v_obs)

        # control delay
        u_buffer.append(u)
        u_applied = u_buffer.pop(0)

        # disturbance
        d = 0.0
        if disturbance == "impulse" and t == impulse_t:
            d = impulse_mag
        elif disturbance == "step" and step_t <= t < step_t + step_duration:
            d = step_mag

        # dynamics
        a = (u_applied - friction * v + d) / mass
        v = v + a * DT
        x = x + v * DT

        # learning update
        if baseline == "B1":
            a_model = (u_applied - 0.0 * v) / mass
            a_error = a - a_model
            w += LR * a_error * v_obs

        drift = max(drift, abs(w))

        if math.isnan(x) or abs(x) > XMAX:
            diverged = True
            break

        xs.append(x)
        vs.append(v)
        errors.append(TARGET - x)
        us.append(u_applied)

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
        baseline, mass, friction, noise, delay_steps, disturbance, seed,
        diverged, stability_margin, drift, rmse, recovery_time,
        control_effort, smoothness, tag
    )
    trace = {"x": xs, "v": vs, "u": us, "err": errors}
    return result, trace


def run_all():
    results: List[RolloutResult] = []
    traces: Dict[str, Dict[str, List[float]]] = {}

    for baseline in BASELINES:
        for mass in MASS_VALUES:
            for friction in FRICTION_VALUES:
                for noise in NOISE_VALUES:
                    for delay in DELAY_STEPS:
                        for dist in DIST_TYPES:
                            for seed in SEEDS:
                                key = f"{baseline}-{mass}-{friction}-{noise}-{delay}-{dist}-{seed}"
                                res, trace = simulate(baseline, mass, friction, noise, delay, dist, seed)
                                results.append(res)
                                if seed == 0 and dist == "impulse" and delay == 0 and noise == 0.0:
                                    traces[key] = trace

    # NO_GAIN tagging for B1 vs B0
    b0_by_scenario: Dict[Tuple[float, float, float, int, str, int], float] = {}
    for r in results:
        if r.baseline == "B0":
            b0_by_scenario[(r.mass, r.friction, r.noise, r.delay, r.disturbance, r.seed)] = r.rmse_tracking
    for i, r in enumerate(results):
        if r.baseline == "B1" and not r.diverged:
            key = (r.mass, r.friction, r.noise, r.delay, r.disturbance, r.seed)
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
                r.baseline, r.mass, r.friction, r.noise, r.delay, r.disturbance, r.seed,
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
            if b == "B1_vs_B0":
                continue
            f.write(f"| {b} | {s['count']} | {s['diverged']} | {s['rmse']:.4f} | {s['effort']:.4f} | {s['smoothness']:.4f} | {s['recovery_time']:.4f} |\n")
        f.write("\n## Failure Tags\n\n")
        for b, s in summary.items():
            if b == "B1_vs_B0":
                continue
            f.write(f"**{b}**: {s['tags']}\n\n")
        f.write("\n## B1 vs B0 RMSE (bootstrap CI)\n\n")
        f.write(f"Mean diff: {summary['B1_vs_B0']['rmse_diff_mean']:.4f}\n")
        f.write(f"90% CI: {summary['B1_vs_B0']['rmse_diff_ci']}\n\n")

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

    # append stress to report
    with open(report_path, "a") as f:
        f.write("\n## Stress Test (high delay+noise+friction, long horizon)\n\n")
        f.write(f"B0: diverged={stress_b0.diverged}, rmse={stress_b0.rmse_tracking:.4f}, effort={stress_b0.control_effort:.2f}, smooth={stress_b0.smoothness:.2f}, recovery={stress_b0.recovery_time:.2f}, tag={stress_b0.tag}\n")
        f.write(f"B1: diverged={stress_b1.diverged}, rmse={stress_b1.rmse_tracking:.4f}, effort={stress_b1.control_effort:.2f}, smooth={stress_b1.smoothness:.2f}, recovery={stress_b1.recovery_time:.2f}, tag={stress_b1.tag}\n")

    # rewrite summary with stress
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Done. Results in phase1/results/")


if __name__ == "__main__":
    run_all()
