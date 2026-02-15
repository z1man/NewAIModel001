#!/usr/bin/env python3
"""Phase 1.6 diagnosis-first milestone.
Step A: Oracle acceleration (a_true) identifiability test.
Step B: KF-based acceleration estimation + adaptation.
Outputs:
- phase1/results/oracle_id_report.md
- phase1/results/kf_estimation_report.md
- plots: a_true_vs_a_est.png, c_hat_oracle.png, c_hat_kf.png
"""
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
DELAY_STEPS = 2  # 20ms

MASS = 1.0
FRICTION = 0.35
TARGET = 1.0

QUAD_DRAG_BASE = 0.2
DRAG_MULT = 6.0
BIAS_LEVEL = 3.0

OPEN_LOOP_AMP = 2.0
OPEN_LOOP_FREQ = 1.5

KP = 18.0
KD = 6.0

RLS_LAMBDA = 0.98
B_MAX = 5.0
C_MAX = 2.0


def rls_update(theta, P, phi, y, lam=RLS_LAMBDA):
    P_phi = P @ phi
    gain = P_phi / (lam + phi.T @ P_phi)
    err = y - phi.T @ theta
    theta = theta + gain * err
    P = (P - np.outer(gain, phi.T) @ P) / lam
    return theta, P, float(err)


def simulate_oracle_id(seed=0):
    random.seed(seed)
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    # single-parameter RLS for c
    theta = np.zeros(1)
    P = np.eye(1) * 10.0

    id_len = int(ID_FRAC * STEPS)
    c_hat_trace = []
    err_trace = []
    y_trace = []
    phi_trace = []

    for t in range(STEPS):
        # ID segment target
        target = TARGET
        if t < id_len:
            target = 1.0 + 0.5 * math.sin(2 * math.pi * t * DT * 1.5)

        e = target - x
        u_nom = KP * e - KD * v

        u_buffer.append(u_nom)
        u_applied = u_buffer.pop(0)

        d_quad = -(QUAD_DRAG_BASE * DRAG_MULT) * v * abs(v)
        bias = 0.0  # locked in ID

        a_true = (u_applied - FRICTION * v + bias + d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        if t < id_len:
            # Option A: y = u_applied - m*a_true = c*v|v| - b (b=0)
            y = u_applied - MASS * a_true
            phi = np.array([v * abs(v)])
            theta, P, err = rls_update(theta, P, phi, y)
            c_hat = float(np.clip(theta[0], 0.0, C_MAX))
            theta[0] = c_hat
            c_hat_trace.append(c_hat)
            err_trace.append(abs(err))
            y_trace.append(float(y))
            phi_trace.append(float(phi[0]))

    c_true = QUAD_DRAG_BASE * DRAG_MULT
    c_relerr = abs(c_hat_trace[-1] - c_true) / max(1e-6, c_true)
    r2 = np.nan
    if len(y_trace) > 5:
        y_arr = np.array(y_trace)
        y_pred = np.array(phi_trace) * c_hat_trace[-1]
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return c_hat_trace, err_trace, c_relerr, r2


def simulate_kf_and_adapt(seed=0):
    random.seed(seed)
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    # KF for [x, v, a]
    X = np.zeros(3)
    Pk = np.eye(3) * 0.1
    Q = np.eye(3) * 0.001
    R = np.array([[0.01]])

    theta = np.zeros(2)  # [b, c]
    Prls = np.eye(2) * 10.0

    id_len = int(ID_FRAC * STEPS)
    c_hat_trace = []
    b_hat_trace = []
    a_true_trace = []
    a_kf_trace = []

    for t in range(STEPS):
        # ID segment target
        target = TARGET
        if t < id_len:
            target = 1.0 + 0.5 * math.sin(2 * math.pi * t * DT * 1.5)

        # bias schedule after ID
        bias = 0.0
        if t >= id_len:
            seg = (t - id_len) // (int((STEPS - id_len) / 3))
            if seg == 0:
                bias = BIAS_LEVEL
            elif seg == 1:
                bias = -BIAS_LEVEL
            else:
                bias = 0.0

        e = target - x
        u_nom = KP * e - KD * v

        u_buffer.append(u_nom)
        u_applied = u_buffer.pop(0)

        d_quad = -(QUAD_DRAG_BASE * DRAG_MULT) * v * abs(v)
        a_true = (u_applied - FRICTION * v + bias + d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        # KF update (measure x)
        F = np.array([[1, DT, 0.5 * DT * DT], [0, 1, DT], [0, 0, 1]])
        X = F @ X
        Pk = F @ Pk @ F.T + Q
        z = np.array([x])
        H = np.array([[1, 0, 0]])
        yk = z - H @ X
        S = H @ Pk @ H.T + R
        K = Pk @ H.T @ np.linalg.inv(S)
        X = X + K @ yk
        Pk = (np.eye(3) - K @ H) @ Pk

        a_kf = X[2]

        if t < id_len:
            # estimate c only (Option A)
            y = u_applied - MASS * a_kf
            phi = np.array([0.0, v * abs(v)])
            theta, Prls, _ = rls_update(theta, Prls, phi, y)
            theta[1] = float(np.clip(theta[1], 0.0, C_MAX))
        else:
            # estimate b only with c frozen (Option A)
            y = u_applied - MASS * a_kf
            phi = np.array([1.0, 0.0])
            theta, Prls, _ = rls_update(theta, Prls, phi, y)
            theta[0] = float(np.clip(theta[0], -B_MAX, B_MAX))

        c_hat_trace.append(theta[1])
        b_hat_trace.append(theta[0])
        a_true_trace.append(a_true)
        a_kf_trace.append(a_kf)

    c_true = QUAD_DRAG_BASE * DRAG_MULT
    c_relerr = abs(c_hat_trace[id_len - 1] - c_true) / max(1e-6, c_true)

    # corr b_hat vs b_true on adaptation segment
    b_true = []
    for t in range(STEPS):
        if t < id_len:
            b_true.append(0.0)
        else:
            seg = (t - id_len) // (int((STEPS - id_len) / 3))
            if seg == 0:
                b_true.append(BIAS_LEVEL)
            elif seg == 1:
                b_true.append(-BIAS_LEVEL)
            else:
                b_true.append(0.0)
    b_true = np.array(b_true)
    b_hat = np.array(b_hat_trace)
    corr_b = np.corrcoef(b_true[id_len:], b_hat[id_len:])[0, 1]

    return (c_hat_trace, b_hat_trace, a_true_trace, a_kf_trace, c_relerr, corr_b)


if __name__ == "__main__":
    # Offline unit test (no delay/no noise, b=0)
    c_true = QUAD_DRAG_BASE * DRAG_MULT
    v_vals = np.linspace(-2, 2, 200)
    a_true = np.zeros_like(v_vals)
    u_vals = c_true * v_vals * np.abs(v_vals) + MASS * a_true
    y = u_vals - MASS * a_true
    phi = v_vals * np.abs(v_vals)
    c_hat = np.sum(phi * y) / max(1e-6, np.sum(phi * phi))
    c_relerr_unit = abs(c_hat - c_true) / max(1e-6, c_true)

    with open(os.path.join(RESULTS_DIR, "offline_unit_test.txt"), "w") as f:
        f.write(f"c_true={c_true:.4f}\n")
        f.write(f"c_hat={c_hat:.4f}\n")
        f.write(f"c_relerr={c_relerr_unit:.4f}\n")

    # Step A
    c_hat_trace, err_trace, c_relerr, r2 = simulate_oracle_id()
    with open(os.path.join(RESULTS_DIR, "oracle_id_report.md"), "w") as f:
        f.write("# Step A – Oracle Identifiability\n\n")
        f.write(f"c_relerr: {c_relerr:.4f}\n\n")
        f.write(f"R2: {r2:.4f}\n")

    plt.figure(figsize=(8,5))
    plt.plot(c_hat_trace)
    plt.title("c_hat (oracle)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "c_hat_oracle.png"))
    plt.close()

    # Step B
    c_hat_trace, b_hat_trace, a_true_trace, a_kf_trace, c_relerr_b, corr_b = simulate_kf_and_adapt()
    with open(os.path.join(RESULTS_DIR, "kf_estimation_report.md"), "w") as f:
        f.write("# Step B – KF Estimation + Adaptation\n\n")
        f.write(f"c_relerr: {c_relerr_b:.4f}\n\n")
        f.write(f"corr(b_hat,b_true): {corr_b:.4f}\n")

    plt.figure(figsize=(8,5))
    plt.plot(a_true_trace, label="a_true")
    plt.plot(a_kf_trace, label="a_kf")
    plt.legend()
    plt.title("a_true vs a_kf")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "a_true_vs_a_est.png"))
    plt.close()

    plt.figure(figsize=(8,5))
    plt.plot(c_hat_trace)
    plt.title("c_hat (kf)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "c_hat_kf.png"))
    plt.close()

    print("Phase 1.6 diagnostics complete.")
