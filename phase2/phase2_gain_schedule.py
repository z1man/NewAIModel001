#!/usr/bin/env python3
"""Saturation-aware gain scheduling at OP20."""
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

U_MAX = 1.0
DIST_SCALE = 0.25
REF_AMP = 0.75


def simulate(p=1):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    sats = []

    for t in range(STEPS):
        r = REF_AMP
        e = r - x

        # saturation-aware gain schedule
        u_cmd_raw = KP * e - KD * v
        scale = max(0.0, 1.0 - abs(u_cmd_raw) / max(1e-6, U_MAX))
        k_eff = scale ** p
        u_cmd = (KP * k_eff) * e - (KD * k_eff) * v

        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        sat = (u_cmd_clip != u_cmd)

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * DIST_SCALE) if t < STEPS // 2 else (-3.0 * DIST_SCALE)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(r - x)
        u_applieds.append(u_applied)
        sats.append(1 if sat else 0)

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)

    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))

    recovery = float("inf")
    window = int(0.2 / DT)
    for i in range(0, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery = i * DT
            break

    effort = float(np.sum(u_applieds**2) * DT)
    smooth = float(np.sum((u_applieds[1:] - u_applieds[:-1])**2) * DT)

    return {
        "p": p,
        "sat_total": sat_total,
        "rmse": rmse,
        "recovery": recovery,
        "effort": effort,
        "smooth": smooth,
    }


def main():
    results = [simulate(1), simulate(2)]
    with open(os.path.join(RESULTS_DIR, "phase2_gain_schedule.md"), "w") as f:
        f.write("# Phase 2 Gain Scheduling (OP20)\n\n")
        for r in results:
            f.write(f"p={r['p']} => sat_total={r['sat_total']:.2%}, RMSE={r['rmse']:.4f}, recovery={r['recovery']:.4f}, effort={r['effort']:.2f}, smooth={r['smooth']:.4f}\n")

    print("Gain scheduling complete.")


if __name__ == "__main__":
    main()
