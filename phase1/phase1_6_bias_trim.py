#!/usr/bin/env python3
"""Slow bias trim using b_hat under steady conditions."""
import math
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 2000
DELAY_STEPS = 2

MASS = 1.0
FRICTION = 0.35

QUAD_DRAG_BASE = 0.2
DRAG_MULT = 6.0
C_TRUE = QUAD_DRAG_BASE * DRAG_MULT

KP = 18.0
KD = 6.0

# frozen params from Step A
C_HAT = 1.2061
F_HAT = 0.3455

BIAS_MAX = 0.8


def run(gamma, tau):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    b_dc = 0.0
    bias_hat = 0.0

    errors = []
    u_total = []
    u_bias = []
    u_base = []

    enabled = 0
    steady = False
    count = 0
    N = int(0.2 / DT)
    e_eps = 0.2

    alpha = DT / tau

    for t in range(STEPS):
        target = 1.0
        e = target - x
        u_base_cmd = KP * e - KD * v

        u_buffer.append(u_base_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0

        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        b_hat = y - C_HAT * v * abs(v) - F_HAT * v
        b_dc = (1 - alpha) * b_dc + alpha * b_hat

        # steady detection (window)
        if abs(e) < e_eps:
            count += 1
        else:
            count = 0
        if count >= N:
            steady = True
        if abs(e) > 1.5 * e_eps:
            steady = False

        if steady:
            bias_hat = (1 - gamma) * bias_hat + gamma * b_dc
            enabled += 1
        else:
            bias_hat = (1 - 0.001) * bias_hat  # small leak

        bias_hat = max(-BIAS_MAX, min(BIAS_MAX, bias_hat))

        u_total_cmd = u_base_cmd + bias_hat
        u_base.append(u_base_cmd)
        u_bias.append(bias_hat)
        u_total.append(u_total_cmd)
        errors.append(target - x)

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    effort_total = sum(u * u for u in u_total) * DT
    smooth = sum((u_total[i] - u_total[i - 1]) ** 2 for i in range(1, len(u_total))) * DT
    recovery = float("inf")
    window = int(0.2 / DT)
    for i in range(0, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery = i * DT
            break
    steady_err = sum(abs(e) for e in errors[-window:]) / window

    abs_e = np.abs(errors)
    p50 = float(np.percentile(abs_e, 50))
    p90 = float(np.percentile(abs_e, 90))

    return {
        "gamma": gamma,
        "tau": tau,
        "rmse": rmse,
        "recovery": recovery,
        "steady": steady_err,
        "effort_total": effort_total,
        "smooth": smooth,
        "bias_final": bias_hat,
        "mean_bias": float(np.mean(np.abs(u_bias))),
        "enabled_pct": enabled / STEPS,
        "p50_abs_e": p50,
        "p90_abs_e": p90,
    }


def main():
    results = []
    for gamma in [0.001, 0.005, 0.01]:
        for tau in [0.5, 2.0]:
            results.append(run(gamma, tau))

    with open(os.path.join(RESULTS_DIR, "bias_trim_report.md"), "w") as f:
        f.write("# Bias trim report\n\n")
        for r in results:
            f.write(
                f"gamma={r['gamma']} tau={r['tau']} => RMSE={r['rmse']:.4f}, recovery={r['recovery']:.4f}, steady={r['steady']:.4f}, effort_total={r['effort_total']:.2f}, smooth={r['smooth']:.4f}, bias_final={r['bias_final']:.4f}, mean|u_bias|={r['mean_bias']:.4f}, enabled={r['enabled_pct']:.2%}, p50|e|={r['p50_abs_e']:.4f}, p90|e|={r['p90_abs_e']:.4f}\n"
            )

    print("Bias trim complete.")


if __name__ == "__main__":
    main()
