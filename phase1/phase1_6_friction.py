#!/usr/bin/env python3
"""Phase 1.6 friction-aware regression + Step B rerun."""
import math
import os
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
C_TRUE = QUAD_DRAG_BASE * DRAG_MULT

OPEN_LOOP_AMP = 2.0
OPEN_LOOP_FREQ = 1.5

KP = 18.0
KD = 6.0

B_MAX = 5.0
C_MAX = 2.0
F_MAX = 2.0


def corr(a, b):
    if len(a) < 2:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def step_a_friction_regression():
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    id_len = int(ID_FRAC * STEPS)

    Phi = []
    Y = []

    for t in range(id_len):
        u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
        u_buffer.append(u_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        a_true = (u_applied - FRICTION * v - d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        Phi.append([v * abs(v), v])
        Y.append(y)

    Phi = np.array(Phi)
    Y = np.array(Y)
    theta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    c_hat, f_hat = theta

    c_relerr = abs(c_hat - C_TRUE) / max(1e-6, C_TRUE)
    residual = Y - Phi @ theta
    corr_v = corr(residual, Phi[:, 1])

    return c_hat, f_hat, c_relerr, corr_v


def step_b_kf(friction_aware=True):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    # KF for [x,v,a]
    X = np.zeros(3)
    Pk = np.eye(3) * 0.1
    Q = np.eye(3) * 0.001
    R = np.array([[0.01]])

    id_len = int(ID_FRAC * STEPS)
    adapt_len = STEPS - id_len

    # parameters
    c_hat = 0.0
    f_hat = 0.0
    b_hat = 0.0

    errors = []
    us = []
    b_true = []
    b_hats = []

    for t in range(STEPS):
        target = 1.0
        if t < id_len:
            target = 1.0 + 0.5 * math.sin(2 * math.pi * t * DT * 1.5)

        # bias schedule
        bias = 0.0
        if t >= id_len:
            seg = (t - id_len) // (adapt_len // 3)
            if seg == 0:
                bias = 3.0
            elif seg == 1:
                bias = -3.0
            else:
                bias = 0.0

        e = target - x
        u_nom = KP * e - KD * v

        u_buffer.append(u_nom)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        # KF update
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
            y = u_applied - MASS * a_kf
            if friction_aware:
                # regress [c, f]
                Phi = np.array([v * abs(v), v])
                theta = np.array([c_hat, f_hat])
                theta = theta + 0.01 * (y - Phi @ theta) * Phi
                c_hat = float(np.clip(theta[0], 0.0, C_MAX))
                f_hat = float(np.clip(theta[1], -F_MAX, F_MAX))
            else:
                # regress c only
                Phi = v * abs(v)
                c_hat = float(np.clip(c_hat + 0.01 * (y - c_hat * Phi) * Phi, 0.0, C_MAX))
        else:
            # estimate b only with c,f frozen
            y = u_applied - MASS * a_kf
            if friction_aware:
                # Convention: y = c*v|v| + f_v*v + b
                b_hat = float(np.clip(y - c_hat * v * abs(v) - f_hat * v, -B_MAX, B_MAX))
            else:
                b_hat = float(np.clip(y - c_hat * v * abs(v), -B_MAX, B_MAX))

        errors.append(target - x)
        us.append(u_applied)
        b_true.append(-bias)
        b_hats.append(b_hat)

    # metrics
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    control_effort = sum(u * u for u in us) * DT
    smoothness = sum((us[i] - us[i - 1]) ** 2 for i in range(1, len(us))) * DT

    # recovery time with sustained window
    recovery_time = float("inf")
    window = int(0.2 / DT)
    step_start = id_len
    for i in range(step_start, len(errors) - window):
        if all(abs(errors[j]) < 0.05 for j in range(i, i + window)):
            recovery_time = (i - step_start) * DT
            break

    steady = sum(abs(e) for e in errors[-window:]) / window

    b_true_arr = np.array(b_true[id_len:])
    b_hat_arr = np.array(b_hats[id_len:])
    corr_b = corr(b_hat_arr, b_true_arr)
    alpha, beta = np.polyfit(b_true_arr, b_hat_arr, 1)
    return {
        "rmse": rmse,
        "recovery": recovery_time,
        "steady": steady,
        "effort": control_effort,
        "smooth": smoothness,
        "corr_b": corr_b,
        "alpha": float(alpha),
        "beta": float(beta),
    }


def residual_correlation_table(c_hat, f_hat):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    id_len = int(ID_FRAC * STEPS)

    residuals = []
    signv = []
    vv = []
    clip = []

    for t in range(id_len):
        u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
        u_buffer.append(u_cmd)
        u_applied = u_buffer.pop(0)
        d_quad = (C_TRUE) * v * abs(v)
        a_true = (u_applied - FRICTION * v - d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT
        y = u_applied - MASS * a_true
        residuals.append(y - c_hat * v * abs(v) - f_hat * v)
        signv.append(math.copysign(1.0, v) if v != 0 else 0.0)
        vv.append(v)
        clip.append(1 if abs(u_cmd) >= 2.0 - 1e-6 else 0)

    return {
        "corr_v": corr(residuals, vv),
        "corr_sign": corr(residuals, signv),
        "corr_clip": corr(residuals, clip),
    }


def main():
    c_hat, f_hat, c_relerr, corr_v = step_a_friction_regression()
    table = residual_correlation_table(c_hat, f_hat)

    # Step B
    base = step_b_kf(friction_aware=False)
    fric = step_b_kf(friction_aware=True)

    with open(os.path.join(RESULTS_DIR, "oracle_id_report_A2_5.md"), "w") as f:
        f.write("# Step A with friction term\n\n")
        f.write(f"c_hat={c_hat:.4f}, f_v_hat={f_hat:.4f}\n")
        f.write(f"c_relerr={c_relerr:.4f}\n")
        f.write(f"corr(residual,v)={corr_v:.4f}\n\n")
        f.write("# Residual correlation table\n")
        f.write(f"corr(v)={table['corr_v']:.4f}, corr(signv)={table['corr_sign']:.4f}, corr(clip)={table['corr_clip']:.4f}\n\n")
        f.write("# Step B comparison (baseline vs friction-aware)\n")
        f.write(f"baseline: {base}\n")
        f.write(f"friction-aware: {fric}\n")

    print("A2-5 complete.")


if __name__ == "__main__":
    main()
