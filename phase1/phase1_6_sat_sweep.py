#!/usr/bin/env python3
"""Saturation dominance validation via u_max and disturbance sweeps."""
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

BASE_U_MAX = 1.0
BASE_BIAS = 3.0


def run(u_max_scale=1.0, disturbance_scale=1.0):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    u_max = BASE_U_MAX * u_max_scale

    errors = []
    u_applieds = []
    u_cmds = []

    for t in range(STEPS):
        target = 1.0
        e = target - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-u_max, min(u_max, u_cmd))

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        bias = (BASE_BIAS * disturbance_scale) if t < STEPS // 2 else (-BASE_BIAS * disturbance_scale)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(e)
        u_cmds.append(u_cmd)
        u_applieds.append(u_applied)

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)

    sat_upper = float(np.mean(u_applieds >= u_max - 1e-6))
    sat_lower = float(np.mean(u_applieds <= -u_max + 1e-6))
    sat_total = sat_upper + sat_lower

    rmse = math.sqrt(float(np.mean(errors**2)))
    mean_abs = float(np.mean(np.abs(errors)))
    median_abs = float(np.median(np.abs(errors)))

    # recovery time with sustained window
    recovery = float("inf")
    window = int(0.2 / DT)
    for i in range(0, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery = i * DT
            break

    steady = float(np.mean(np.abs(errors[-window:])))

    effort = float(np.sum(u_applieds**2) * DT)
    smooth = float(np.sum((u_applieds[1:] - u_applieds[:-1])**2) * DT)

    return {
        "u_max_scale": u_max_scale,
        "disturbance_scale": disturbance_scale,
        "sat_upper": sat_upper,
        "sat_lower": sat_lower,
        "sat_total": sat_total,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "median_abs": median_abs,
        "recovery": recovery,
        "steady": steady,
        "effort": effort,
        "smooth": smooth,
    }


def main():
    u_max_scales = [1.0, 1.5, 2.0, 3.0]
    dist_scales = [1.0, 0.75, 0.5, 0.25]

    results_a = [run(u_max_scale=s, disturbance_scale=1.0) for s in u_max_scales]
    results_b = [run(u_max_scale=1.0, disturbance_scale=s) for s in dist_scales]

    with open(os.path.join(RESULTS_DIR, "sat_sweep_report.md"), "w") as f:
        f.write("# Saturation sweep report\n\n")
        f.write("## Experiment A: u_max sweep\n")
        for r in results_a:
            f.write(
                f"u_max_scale={r['u_max_scale']} => sat_total={r['sat_total']:.2%} (upper={r['sat_upper']:.2%}, lower={r['sat_lower']:.2%}), RMSE={r['rmse']:.4f}, mean|e|={r['mean_abs']:.4f}, median|e|={r['median_abs']:.4f}, recovery={r['recovery']:.4f}, steady={r['steady']:.4f}, effort={r['effort']:.2f}, smooth={r['smooth']:.4f}\n"
            )
        f.write("\n## Experiment B: disturbance sweep\n")
        for r in results_b:
            f.write(
                f"dist_scale={r['disturbance_scale']} => sat_total={r['sat_total']:.2%} (upper={r['sat_upper']:.2%}, lower={r['sat_lower']:.2%}), RMSE={r['rmse']:.4f}, mean|e|={r['mean_abs']:.4f}, median|e|={r['median_abs']:.4f}, recovery={r['recovery']:.4f}, steady={r['steady']:.4f}, effort={r['effort']:.2f}, smooth={r['smooth']:.4f}\n"
            )

    print("Saturation sweep complete.")


if __name__ == "__main__":
    main()
