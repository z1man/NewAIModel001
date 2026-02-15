#!/usr/bin/env python3
"""Phase 2 dynamic reference shaping evaluation."""
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
A = 0.75

T0 = 2.0


def simulate(ref_type, ref_param, shaping, shape_param):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    sats = []
    r_vals = []
    r_shape_vals = []

    r_shaped = 0.0

    for t in range(STEPS):
        t_sec = t * DT
        if ref_type == "step":
            r = 0.0 if t_sec < T0 else A
        elif ref_type == "ramp":
            S = ref_param
            r = max(0.0, min(A, S * max(0.0, t_sec - T0)))
        elif ref_type == "sine":
            f = ref_param
            r = A * math.sin(2 * math.pi * f * t_sec)
        else:
            r = A

        # shaping
        if shaping == "rate":
            dr_max = shape_param
            dr = max(-dr_max * DT, min(dr_max * DT, r - r_shaped))
            r_shaped = r_shaped + dr
        elif shaping == "lpf":
            tau = shape_param
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
        r_vals.append(r)
        r_shape_vals.append(r_shaped)

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))

    # recovery: for step, time to |e|<0.05 for 200ms
    recovery = float("inf")
    window = int(0.2 / DT)
    if ref_type == "step":
        start = int(T0 / DT)
        for i in range(start, len(errors) - window):
            if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
                recovery = (i - start) * DT
                break

    effort = float(np.sum(u_applieds**2) * DT)
    smooth = float(np.sum((u_applieds[1:] - u_applieds[:-1])**2) * DT)
    peak_e = float(np.max(np.abs(errors))) if ref_type == "step" else float("nan")

    diff = np.abs(np.array(r_vals) - np.array(r_shape_vals))
    max_diff = float(np.max(diff))

    return {
        "ref": ref_type,
        "ref_param": ref_param,
        "shape": shaping,
        "shape_param": shape_param,
        "sat_total": sat_total,
        "rmse": rmse,
        "recovery": recovery,
        "peak_e": peak_e,
        "effort": effort,
        "smooth": smooth,
        "max_r_diff": max_diff,
    }


def main():
    results_step = []
    for dr_max in [0.2, 0.5, 1.0]:
        results_step.append(simulate("step", 0.0, "rate", dr_max))
    for tau in [0.2, 0.5, 1.0]:
        results_step.append(simulate("step", 0.0, "lpf", tau))

    results_ramp = []
    for S in [0.1 * A, 0.2 * A, 0.5 * A]:
        for dr_max in [0.2, 0.5, 1.0]:
            results_ramp.append(simulate("ramp", S, "rate", dr_max))

    results_sine = []
    for f in [0.1, 0.3, 1.0]:
        for tau in [0.2, 0.5, 1.0]:
            results_sine.append(simulate("sine", f, "lpf", tau))

    def write_table(path, rows):
        with open(path, "w") as f:
            f.write("| shape | param | sat_total | RMSE | recovery | peak|e| | effort | smooth | max|r-rs| |\n")
            f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(
                    f"| {r['shape']} | {r['shape_param']} | {r['sat_total']:.2%} | {r['rmse']:.4f} | {r['recovery']:.4f} | {r['peak_e']:.4f} | {r['effort']:.2f} | {r['smooth']:.4f} | {r['max_r_diff']:.4f} |\n"
                )

    write_table(os.path.join(RESULTS_DIR, "phase2_step_table.md"), results_step)
    write_table(os.path.join(RESULTS_DIR, "phase2_ramp_table.md"), results_ramp)
    write_table(os.path.join(RESULTS_DIR, "phase2_sine_table.md"), results_sine)

    print("Dynamic ref shaping complete.")


if __name__ == "__main__":
    main()
