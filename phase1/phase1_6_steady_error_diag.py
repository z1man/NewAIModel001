#!/usr/bin/env python3
"""Diagnose steady error source."""
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

U_MAX = 3.0  # assumed clip from earlier setup


def simulate(k_i=0.0, I_max=2.0):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    I = 0.0

    errors = []
    u_cmds = []
    u_applieds = []

    for t in range(STEPS):
        target = 1.0
        e = target - x
        I += k_i * e * DT
        I = max(-I_max, min(I_max, I))
        u_cmd = KP * e - KD * v + I

        # simulate clip
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(e)
        u_cmds.append(u_cmd)
        u_applieds.append(u_applied)

    return np.array(errors), np.array(u_cmds), np.array(u_applieds)


def main():
    errors, u_cmds, u_applieds = simulate(k_i=0.0)

    # A) error signal sanity
    unique, counts = np.unique(np.round(errors, 6), return_counts=True)
    top_idx = np.argsort(counts)[-10:][::-1]
    top_vals = [(float(unique[i]), int(counts[i])) for i in top_idx]

    # B) saturation
    sat_upper = np.mean(u_applieds >= U_MAX - 1e-6)
    sat_lower = np.mean(u_applieds <= -U_MAX + 1e-6)

    # C) integral tests
    k_is = [0.1, 0.2, 0.5]
    integral_results = []
    for k_i in k_is:
        e, u_cmd, u_app = simulate(k_i=k_i)
        median_abs = float(np.median(np.abs(e)))
        I_sat = np.mean(np.abs(u_cmd) >= U_MAX - 1e-6)
        integral_results.append((k_i, median_abs, I_sat))

    with open(os.path.join(RESULTS_DIR, "steady_error_diag_report.md"), "w") as f:
        f.write("# Steady Error Diagnosis\n\n")
        f.write("## A) Error signal sanity\n")
        f.write(f"unique_count(e)={len(unique)}\n")
        f.write(f"top10={top_vals}\n")
        f.write(f"min={float(np.min(errors)):.4f}, max={float(np.max(errors)):.4f}, mean={float(np.mean(errors)):.4f}, mean_abs={float(np.mean(np.abs(errors))):.4f}\n\n")
        f.write("## B) Actuator saturation\n")
        f.write(f"sat_upper={sat_upper:.2%}, sat_lower={sat_lower:.2%}, mean(u_applied)={float(np.mean(u_applieds)):.4f}\n\n")
        f.write("## C) Integral test\n")
        for k_i, med, sat in integral_results:
            f.write(f"k_i={k_i}: median|e|={med:.4f}, sat_fraction={sat:.2%}\n")

    print("Steady error diagnostics complete.")


if __name__ == "__main__":
    main()
