#!/usr/bin/env python3
"""Debug Phase1 vs Phase2 baseline mismatch."""
import math
import os
import numpy as np

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
DIST_SCALE = 0.75


def run_baseline():
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    r_vals = []
    u_cmds = []
    u_apps = []
    sat_flags = []

    for t in range(STEPS):
        r = 1.0
        e = r - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-BASE_U_MAX, min(BASE_U_MAX, u_cmd))
        sat = (u_cmd_clip != u_cmd)

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * DIST_SCALE) if t < STEPS // 2 else (-3.0 * DIST_SCALE)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        r_vals.append(r)
        u_cmds.append(u_cmd)
        u_apps.append(u_applied)
        sat_flags.append(1 if sat else 0)

    return {
        "u_max": BASE_U_MAX,
        "dist_scale": DIST_SCALE,
        "r_stats": (float(np.min(r_vals)), float(np.max(r_vals)), float(np.mean(r_vals))),
        "u_cmd_stats": (float(np.min(u_cmds)), float(np.max(u_cmds)), float(np.mean(u_cmds))),
        "u_app_stats": (float(np.min(u_apps)), float(np.max(u_apps)), float(np.mean(u_apps))),
        "sat_total": float(np.mean(sat_flags)),
    }


def main():
    res = run_baseline()
    print(res)


if __name__ == "__main__":
    main()
