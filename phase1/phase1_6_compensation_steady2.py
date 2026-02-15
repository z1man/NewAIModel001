#!/usr/bin/env python3
"""Window-based steady detection with compensation."""
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

U_COMP_MAX = 3.0

# frozen params from Step A
C_HAT = 1.2061
F_HAT = 0.3455


def run(alpha, e_eps, k_b=1.0, b_dead=0.2):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    b_lp = 0.0

    errors = []
    u_total = []
    u_comp = []
    u_base = []
    enabled = 0

    steady = False
    count = 0
    N = int(0.2 / DT)

    for t in range(STEPS):
        target = 1.0
        e = target - x
        u_base_cmd = KP * e - KD * v

        u_buffer.append(u_base_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0

        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        b_hat = y - C_HAT * v * abs(v) - F_HAT * v
        b_lp = (1 - alpha) * b_lp + alpha * b_hat
        if abs(b_lp) < b_dead:
            b_lp_eff = 0.0
        else:
            b_lp_eff = b_lp

        # window-based steady detection
        if abs(e) < e_eps:
            count += 1
        else:
            count = 0
        if count >= N:
            steady = True
        if abs(e) > 1.5 * e_eps:
            steady = False

        if steady:
            comp = k_b * b_lp_eff
            comp = max(-U_COMP_MAX, min(U_COMP_MAX, comp))
            enabled += 1
        else:
            comp = 0.0

        u_total_cmd = u_base_cmd + comp
        u_base.append(u_base_cmd)
        u_comp.append(comp)
        u_total.append(u_total_cmd)
        errors.append(target - x)

    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    effort_total = sum(u * u for u in u_total) * DT
    effort_comp = sum(u * u for u in u_comp) * DT
    smooth = sum((u_total[i] - u_total[i - 1]) ** 2 for i in range(1, len(u_total))) * DT

    steady_err = sum(abs(e) for e in errors[-int(0.2 / DT):]) / int(0.2 / DT)

    return {
        "alpha": alpha,
        "e_eps": e_eps,
        "rmse": rmse,
        "steady": steady_err,
        "effort_total": effort_total,
        "effort_comp": effort_comp,
        "smooth": smooth,
        "enabled_pct": enabled / STEPS,
    }


def main():
    configs = [
        (0.02, 0.20),
        (0.05, 0.20),
        (0.02, 0.15),
        (0.05, 0.15),
    ]

    results = []
    for alpha, e_eps in configs:
        results.append(run(alpha, e_eps))

    with open(os.path.join(RESULTS_DIR, "compensation_steady2_report.md"), "w") as f:
        f.write("# Window-based steady compensation\n\n")
        for r in results:
            f.write(
                f"alpha={r['alpha']} e_eps={r['e_eps']} => RMSE={r['rmse']:.4f}, steady={r['steady']:.4f}, effort_total={r['effort_total']:.2f}, effort_comp={r['effort_comp']:.2f}, smooth={r['smooth']:.4f}, enabled={r['enabled_pct']:.2%}\n"
            )

    print("Steady2 compensation complete.")


if __name__ == "__main__":
    main()
