#!/usr/bin/env python3
"""Phase 1.6 Step A2 diagnostics (experiment-design, not estimator tuning)."""
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DT = 0.01
STEPS = 2000
ID_FRAC = 0.2
DELAY_STEPS = 2

MASS = 1.0
FRICTION = 0.35

QUAD_DRAG_BASE = 0.2
DRAG_MULT = 6.0

OPEN_LOOP_AMP = 2.0
OPEN_LOOP_FREQ = 1.5


def rls_update(theta, P, phi, y, lam=0.98):
    P_phi = P @ phi
    gain = P_phi / (lam + phi.T @ P_phi)
    err = y - phi.T @ theta
    theta = theta + gain * err
    P = (P - np.outer(gain, phi.T) @ P) / lam
    return theta, P, float(err)


def a2_open_loop_id(seed=0):
    random.seed(seed)
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    id_len = int(ID_FRAC * STEPS)

    theta = np.zeros(1)
    P = np.eye(1) * 10.0
    c_hat_trace, err_trace, y_trace, phi_trace = [], [], [], []

    for t in range(id_len):
        u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
        u_buffer.append(u_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = -(QUAD_DRAG_BASE * DRAG_MULT) * v * abs(v)
        a_true = (u_applied - FRICTION * v + d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        phi = np.array([v * abs(v)])
        theta, P, err = rls_update(theta, P, phi, y)
        c_hat = float(np.clip(theta[0], 0.0, 2.0))
        theta[0] = c_hat
        c_hat_trace.append(c_hat)
        err_trace.append(abs(err))
        y_trace.append(float(y))
        phi_trace.append(float(phi[0]))

    c_true = QUAD_DRAG_BASE * DRAG_MULT
    c_relerr = abs(c_hat_trace[-1] - c_true) / max(1e-6, c_true)
    y_arr = np.array(y_trace)
    y_pred = np.array(phi_trace) * c_hat_trace[-1]
    ss_res = np.sum((y_arr - y_pred) ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return c_hat_trace, err_trace, c_relerr, r2


def a2_model_mismatch(seed=0):
    random.seed(seed)
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    id_len = int(ID_FRAC * STEPS)

    # OLS fit for [c, f_c, b] using phi=[v|v|, sign(v), 1]
    Phi = []
    Y = []

    for t in range(id_len):
        u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
        u_buffer.append(u_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = -(QUAD_DRAG_BASE * DRAG_MULT) * v * abs(v)
        a_true = (u_applied - FRICTION * v + d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        Phi.append([v * abs(v), math.copysign(1.0, v) if v != 0 else 0.0, 1.0])
        Y.append(y)

    Phi = np.array(Phi)
    Y = np.array(Y)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    c_hat, f_c, b_hat = theta

    c_true = QUAD_DRAG_BASE * DRAG_MULT
    c_relerr = abs(c_hat - c_true) / max(1e-6, c_true)

    return c_relerr, c_hat, f_c, b_hat


def a2_delay_scan(seed=0):
    random.seed(seed)
    results = []
    for delta in [-1, 0, 1]:
        x, v = 0.0, 0.0
        u_buffer = [0.0 for _ in range(DELAY_STEPS + 1 + abs(delta))]
        id_len = int(ID_FRAC * STEPS)

        theta = np.zeros(1)
        P = np.eye(1) * 10.0
        c_hat_trace = []

        for t in range(id_len):
            u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
            u_buffer.append(u_cmd)
            u_applied = u_buffer.pop(0)

            d_quad = -(QUAD_DRAG_BASE * DRAG_MULT) * v * abs(v)
            a_true = (u_applied - FRICTION * v + d_quad) / MASS
            v = v + a_true * DT
            x = x + v * DT

            # apply alignment shift by using u_applied from neighbor index
            if delta == -1 and t > 0:
                u_aligned = u_buffer[-1]
            elif delta == 1:
                u_aligned = u_cmd
            else:
                u_aligned = u_applied

            y = u_aligned - MASS * a_true
            phi = np.array([v * abs(v)])
            theta, P, _ = rls_update(theta, P, phi, y)
            c_hat = float(np.clip(theta[0], 0.0, 2.0))
            theta[0] = c_hat
            c_hat_trace.append(c_hat)

        c_true = QUAD_DRAG_BASE * DRAG_MULT
        c_relerr = abs(c_hat_trace[-1] - c_true) / max(1e-6, c_true)
        results.append((delta, c_relerr))
    return results


def main():
    # A2-1 open-loop ID
    c_hat_trace, err_trace, c_relerr, r2 = a2_open_loop_id()
    plt.figure(figsize=(8,5))
    plt.plot(c_hat_trace)
    plt.title("c_hat (A2-1 open-loop)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "c_hat_open_loop.png"))
    plt.close()

    # A2-2 model mismatch
    c_relerr_mm, c_hat_mm, f_c_mm, b_mm = a2_model_mismatch()

    # A2-3 delay scan
    delay_results = a2_delay_scan()

    with open(os.path.join(RESULTS_DIR, "oracle_id_report_A2.md"), "w") as f:
        f.write("# Phase 1.6 Step A2 Report\n\n")
        f.write("## A2-1 Open-loop ID\n")
        f.write(f"c_relerr: {c_relerr:.4f}\n")
        f.write(f"R2: {r2:.4f}\n\n")
        f.write("## A2-2 Model mismatch (expanded regression)\n")
        f.write(f"c_relerr: {c_relerr_mm:.4f}\n")
        f.write(f"c_hat={c_hat_mm:.4f}, f_c={f_c_mm:.4f}, b={b_mm:.4f}\n\n")
        f.write("## A2-3 Delay scan (delta steps)\n")
        for delta, err in delay_results:
            f.write(f"delta={delta}: c_relerr={err:.4f}\n")

    print("Phase 1.6 A2 diagnostics complete.")


if __name__ == "__main__":
    main()
