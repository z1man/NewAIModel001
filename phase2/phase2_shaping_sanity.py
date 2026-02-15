#!/usr/bin/env python3
"""Phase 2 shaping sanity check for r vs r_shaped."""
import math
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 400
DELAY_STEPS = 2

MASS = 1.0
FRICTION = 0.35
QUAD_DRAG_BASE = 0.2
DRAG_MULT = 6.0
C_TRUE = QUAD_DRAG_BASE * DRAG_MULT

KP = 18.0
KD = 6.0

U_MAX = 1.0


def run(dist_scale, ref_amp, mode, param):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    r_shaped = ref_amp

    r_vals = []
    r_shape_vals = []

    for t in range(STEPS):
        r = ref_amp
        if mode == "rate":
            dr_max = param
            dr = max(-dr_max * DT, min(dr_max * DT, r - r_shaped))
            r_shaped = r_shaped + dr
        elif mode == "lpf":
            tau = param
            alpha = DT / max(1e-6, tau)
            r_shaped = (1 - alpha) * r_shaped + alpha * r
        else:
            r_shaped = r

        e = r_shaped - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * dist_scale) if t < STEPS // 2 else (-3.0 * dist_scale)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        r_vals.append(r)
        r_shape_vals.append(r_shaped)

    r_vals = np.array(r_vals)
    r_shape_vals = np.array(r_shape_vals)
    diff = np.abs(r_vals - r_shape_vals)
    return float(np.max(diff)), float(np.mean(diff))


def main():
    ops = [
        (0.25, 0.75, "OP20"),
        (0.15, 0.25, "OP5"),
    ]

    configs = [("rate", 0.25), ("rate", 0.5), ("rate", 1.0), ("lpf", 0.2), ("lpf", 0.5), ("lpf", 1.0), ("lpf", 2.0)]

    with open(os.path.join(RESULTS_DIR, "phase2_shaping_sanity.md"), "w") as f:
        f.write("# Phase 2 shaping sanity\n\n")
        for dist_scale, ref_amp, label in ops:
            f.write(f"## {label} (dist_scale={dist_scale}, ref_amp={ref_amp})\n")
            for mode, param in configs:
                maxd, meand = run(dist_scale, ref_amp, mode, param)
                f.write(f"{mode} {param}: max|r-rs|={maxd:.6f}, mean|r-rs|={meand:.6f}\n")

    print("Shaping sanity complete.")


if __name__ == "__main__":
    main()
