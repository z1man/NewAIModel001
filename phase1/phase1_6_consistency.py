#!/usr/bin/env python3
"""Phase 1.6 consistency checks for y definition and b_hat computation."""
import math
import os
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 1000
DELAY_STEPS = 2

MASS = 1.0
FRICTION = 0.35

QUAD_DRAG_BASE = 0.2
DRAG_MULT = 6.0
C_TRUE = QUAD_DRAG_BASE * DRAG_MULT

KP = 18.0
KD = 6.0


def simulate_window():
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    y_A = []
    y_B = []
    v_vals = []
    b_true = []

    for t in range(STEPS):
        target = 1.0
        e = target - x
        u_nom = KP * e - KD * v

        u_buffer.append(u_nom)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0

        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        # Step A formula (Option A)
        y_a = u_applied - MASS * a_true
        # Step B formula (same)
        y_b = u_applied - MASS * a_true

        y_A.append(y_a)
        y_B.append(y_b)
        v_vals.append(v)
        b_true.append(bias)

    return np.array(y_A), np.array(y_B), np.array(v_vals), np.array(b_true)


def main():
    y_A, y_B, v_vals, b_true = simulate_window()

    diff = y_A - y_B
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff))
    corr = float(np.corrcoef(y_A, y_B)[0, 1])

    # Invariant b_hat
    c_hat = C_TRUE
    f_hat = FRICTION
    b_hat = y_B - c_hat * v_vals * np.abs(v_vals) - f_hat * v_vals

    corr_bb = float(np.corrcoef(b_hat, b_true)[0, 1])
    corr_neg = float(np.corrcoef(-b_hat, b_true)[0, 1])
    corr_flip = float(np.corrcoef(b_hat, -b_true)[0, 1])

    # linear fit
    alpha, beta = np.polyfit(b_true, b_hat, 1)

    report_path = os.path.join(RESULTS_DIR, "consistency_report.md")
    with open(report_path, "w") as f:
        f.write("# Consistency Check\n\n")
        f.write("## y_A vs y_B\n")
        f.write(f"mean diff: {mean_diff:.6f}\n")
        f.write(f"std diff: {std_diff:.6f}\n")
        f.write(f"corr: {corr:.6f}\n\n")

        f.write("## b_hat invariant check\n")
        f.write(f"corr(b_hat,b_true): {corr_bb:.4f}\n")
        f.write(f"corr(-b_hat,b_true): {corr_neg:.4f}\n")
        f.write(f"corr(b_hat,-b_true): {corr_flip:.4f}\n")
        f.write(f"alpha: {alpha:.4f}, beta: {beta:.4f}\n")

    print("Consistency check complete.")


if __name__ == "__main__":
    main()
