#!/usr/bin/env python3
"""Phase 2 feasibility frontier + shaping at OP20/OP5."""
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


def simulate(disturbance_scale=0.75, ref_amp=1.0, ref_mode="none", ref_param=0.0):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    sats = []

    r_shaped = ref_amp

    for t in range(STEPS):
        r = ref_amp
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

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * disturbance_scale) if t < STEPS // 2 else (-3.0 * disturbance_scale)
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
        "disturbance_scale": disturbance_scale,
        "ref_amp": ref_amp,
        "mode": ref_mode,
        "param": ref_param,
        "sat_total": sat_total,
        "rmse": rmse,
        "recovery": recovery,
        "mean_abs": mean_abs,
        "effort": effort,
        "smooth": smooth,
    }


def main():
    disturbance_scales = [0.75, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
    ref_amps = [1.0, 0.75, 0.5, 0.25]

    frontier = []
    for ds in disturbance_scales:
        for ra in ref_amps:
            frontier.append(simulate(disturbance_scale=ds, ref_amp=ra))

    # identify OP20 and OP5
    op20 = next((r for r in frontier if r["sat_total"] <= 0.20), None)
    op5 = next((r for r in frontier if r["sat_total"] <= 0.05), None)

    # shaping tests at OP20/OP5
    shaping_results = []
    if op20:
        for dr_max in [0.25, 0.5, 1.0]:
            shaping_results.append(simulate(op20["disturbance_scale"], op20["ref_amp"], "rate", dr_max))
        for tau in [0.2, 0.5, 1.0, 2.0]:
            shaping_results.append(simulate(op20["disturbance_scale"], op20["ref_amp"], "lpf", tau))
    if op5:
        for dr_max in [0.25, 0.5, 1.0]:
            shaping_results.append(simulate(op5["disturbance_scale"], op5["ref_amp"], "rate", dr_max))
        for tau in [0.2, 0.5, 1.0, 2.0]:
            shaping_results.append(simulate(op5["disturbance_scale"], op5["ref_amp"], "lpf", tau))

    with open(os.path.join(RESULTS_DIR, "phase2_feasibility.md"), "w") as f:
        f.write("# Phase 2 Feasibility Frontier\n\n")
        f.write("| dist_scale | ref_amp | sat_total | RMSE | mean|e| | recovery |\n")
        f.write("|---:|---:|---:|---:|---:|---:|\n")
        for r in frontier:
            f.write(f"| {r['disturbance_scale']} | {r['ref_amp']} | {r['sat_total']:.2%} | {r['rmse']:.4f} | {r['mean_abs']:.4f} | {r['recovery']:.4f} |\n")
        f.write("\n")
        f.write(f"**OP20:** {op20}\n\n")
        f.write(f"**OP5:** {op5}\n\n")

        f.write("## Shaping results (OP20/OP5)\n")
        f.write("| mode | param | sat_total | RMSE | mean|e| | recovery | effort | smooth |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in shaping_results:
            f.write(f"| {r['mode']} | {r['param']} | {r['sat_total']:.2%} | {r['rmse']:.4f} | {r['mean_abs']:.4f} | {r['recovery']:.4f} | {r['effort']:.2f} | {r['smooth']:.4f} |\n")

    print("Phase 2 feasibility complete.")


if __name__ == "__main__":
    main()
