#!/usr/bin/env python3
"""Phase 1.6 compensation sweep with b_hat in control loop."""
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

B_MAX = 5.0
C_MAX = 2.0
F_MAX = 2.0

U_COMP_MAX = 3.0
V_EPS = 0.1


def run_sweep(k_b):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    # freeze from Step A
    c_hat = 1.2061
    f_hat = 0.3455

    errors = []
    u_base_list = []
    u_comp_list = []
    u_total_list = []
    b_hat_list = []

    for t in range(STEPS):
        target = 1.0
        e = target - x
        u_base = KP * e - KD * v

        u_buffer.append(u_base)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0

        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        b_hat = y - c_hat * v * abs(v) - f_hat * v

        # compensation
        comp = k_b * b_hat
        if abs(v) < V_EPS:
            comp = 0.0
        comp = max(-U_COMP_MAX, min(U_COMP_MAX, comp))

        u_base_list.append(u_base)
        u_comp_list.append(comp)
        u_total_list.append(u_base + comp)
        b_hat_list.append(b_hat)
        errors.append(target - x)

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    effort_base = sum(u * u for u in u_base_list) * DT
    effort_comp = sum(u * u for u in u_comp_list) * DT
    effort_total = sum(u * u for u in u_total_list) * DT
    smooth = sum((u_total_list[i] - u_total_list[i - 1]) ** 2 for i in range(1, len(u_total_list))) * DT
    steady = sum(abs(e) for e in errors[-int(0.2 / DT):]) / int(0.2 / DT)

    recovery = float("inf")
    window = int(0.2 / DT)
    for i in range(0, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery = i * DT
            break

    return {
        "k_b": k_b,
        "rmse": rmse,
        "recovery": recovery,
        "steady": steady,
        "effort_base": effort_base,
        "effort_comp": effort_comp,
        "effort_total": effort_total,
        "smooth": smooth,
        "mean_bhat": float(np.mean(np.abs(b_hat_list))),
        "mean_ucomp": float(np.mean(np.abs(u_comp_list))),
    }


def main():
    results = []
    for k_b in [0, 0.25, 0.5, 1.0]:
        results.append(run_sweep(k_b))

    with open(os.path.join(RESULTS_DIR, "compensation_sweep_report.md"), "w") as f:
        f.write("# Compensation Sweep Report\n\n")
        for r in results:
            f.write(f"k_b={r['k_b']} => RMSE={r['rmse']:.4f}, recovery={r['recovery']:.4f}, steady={r['steady']:.4f}, effort_base={r['effort_base']:.2f}, effort_comp={r['effort_comp']:.2f}, effort_total={r['effort_total']:.2f}, smooth={r['smooth']:.4f}, mean|b_hat|={r['mean_bhat']:.4f}, mean|u_comp|={r['mean_ucomp']:.4f}\n")

    print("Compensation sweep complete.")


if __name__ == "__main__":
    main()
