#!/usr/bin/env python3
"""Phase 2: reference shaping under u_max=1.0 and disturbance_scale=0.75."""
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
DIST_SCALE = 0.75


def simulate(ref_mode="none", ref_param=0.0):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    u_cmds = []
    sats = []

    # reference shaping state
    r = 1.0
    r_shaped = 1.0

    for t in range(STEPS):
        # base reference
        r = 1.0
        # shaping
        if ref_mode == "rate":
            dr_max = ref_param
            dr = max(-dr_max * DT, min(dr_max * DT, r - r_shaped))
            r_shaped = r_shaped + dr
        elif ref_mode == "lpf":
            tau = ref_param
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

        d_quad = (C_TRUE) * v * abs(v)
        bias = (3.0 * DIST_SCALE) if t < STEPS // 2 else (-3.0 * DIST_SCALE)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(r_shaped - x)
        u_cmds.append(u_cmd)
        u_applieds.append(u_applied)
        sats.append(1 if sat else 0)

        # saturation-aware freeze hook (no estimators in Phase 2)
        if sat:
            pass

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)
    sats = np.array(sats)

    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))
    mean_abs = float(np.mean(np.abs(errors)))

    recovery = float("inf")
    window = int(0.2 / DT)
    for i in range(0, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery = i * DT
            break

    effort = float(np.sum(u_applieds**2) * DT)
    smooth = float(np.sum((u_applieds[1:] - u_applieds[:-1])**2) * DT)

    return {
        "mode": ref_mode,
        "param": ref_param,
        "sat_total": sat_total,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "recovery": recovery,
        "effort": effort,
        "smooth": smooth,
    }


def main():
    results = []
    # baseline
    results.append(simulate("none", 0.0))

    # rate limiter sweep
    for dr_max in [0.25, 0.5, 1.0]:
        results.append(simulate("rate", dr_max))

    # LPF sweep
    for tau in [0.2, 0.5, 1.0]:
        results.append(simulate("lpf", tau))

    best = None
    baseline = results[0]

    for r in results[1:]:
        if r["sat_total"] < 0.05 and r["recovery"] < float("inf") and r["rmse"] <= baseline["rmse"]:
            if best is None or r["sat_total"] < best["sat_total"]:
                best = r

    with open(os.path.join(RESULTS_DIR, "phase2_summary.md"), "w") as f:
        f.write("# Phase 2 Reference Shaping Summary\n\n")
        f.write("| mode | param | sat_total | RMSE | mean|e| | Recovery | Effort | Smooth |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in results:
            f.write(
                f"| {r['mode']} | {r['param']} | {r['sat_total']:.2%} | {r['rmse']:.4f} | {r['mean_abs']:.4f} | {r['recovery']:.4f} | {r['effort']:.2f} | {r['smooth']:.4f} |\n"
            )
        f.write("\n")
        if best:
            f.write(f"**Best config:** mode={best['mode']} param={best['param']}\n")
        else:
            f.write("**Best config:** none (no configuration met acceptance)\n")

    print("Phase 2 sweep complete.")


if __name__ == "__main__":
    main()
