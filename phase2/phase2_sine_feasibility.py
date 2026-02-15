#!/usr/bin/env python3
"""Phase 2 sine feasibility grid with/without LPF."""
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

RECOVERY_EPS = 0.05


def simulate(A, f, mode="none", tau=1.0):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    sats = []
    r_shaped = 0.0

    for t in range(STEPS):
        t_sec = t * DT
        r = A * math.sin(2 * math.pi * f * t_sec)
        if mode == "lpf":
            alpha = DT / max(1e-6, tau)
            r_shaped = (1 - alpha) * r_shaped + alpha * r
        else:
            r_shaped = r

        e = r_shaped - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        sat = (u_cmd_clip != u_cmd)

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * DIST_SCALE) if t < STEPS // 2 else (-3.0 * DIST_SCALE)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(r_shaped - x)
        u_applieds.append(u_applied)
        sats.append(1 if sat else 0)

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))

    # recovery sanity: first time |e| < eps
    recovery_idx = next((i for i, e in enumerate(errors) if abs(e) < RECOVERY_EPS), None)

    return {
        "A": A,
        "f": f,
        "mode": mode,
        "tau": tau,
        "sat_total": sat_total,
        "rmse": rmse,
        "recovery_idx": recovery_idx,
    }


def main():
    amps = [0.25, 0.5, 0.75]
    freqs = [0.1, 0.2, 0.3, 0.5, 1.0]

    rows = []
    for A in amps:
        for f in freqs:
            rows.append(simulate(A, f, "none", 1.0))
            rows.append(simulate(A, f, "lpf", 1.0))

    with open(os.path.join(RESULTS_DIR, "phase2_sine_feasibility.md"), "w") as f:
        f.write("# Sine feasibility grid (dist_scale=0.25)\n\n")
        f.write(f"Recovery eps: {RECOVERY_EPS}\n\n")
        f.write("| A | f | mode | sat_total | RMSE | first_idx(|e|<eps) |\n")
        f.write("|---:|---:|---|---:|---:|---:|\n")
        for r in rows:
            idx = r['recovery_idx'] if r['recovery_idx'] is not None else -1
            f.write(f"| {r['A']} | {r['f']} | {r['mode']} | {r['sat_total']:.2%} | {r['rmse']:.4f} | {idx} |\n")

    print("Sine feasibility complete.")


if __name__ == "__main__":
    main()
