#!/usr/bin/env python3
"""Phase 4: Learning-based Command Governor (L-CG).

Generates teacher data (CG), trains a lightweight regression model to
approximate CG reference shaping, and evaluates closed-loop performance.
"""
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
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

RECOVERY_EPS = 0.05

DIST_SCALES = [0.15, 0.25, 0.35]


@dataclass
class SimResult:
    sat_total: float
    rmse: float
    mean_abs: float
    recovery: float
    in_band: float


@dataclass
class Model:
    w: np.ndarray
    b: float
    x_mean: np.ndarray
    x_std: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_n = (x - self.x_mean) / self.x_std
        return x_n @ self.w + self.b


def reference_signal(ref_type: str, ref_param, t_sec: float) -> float:
    if ref_type == "step":
        return 0.0 if t_sec < 2.0 else ref_param
    if ref_type == "ramp":
        slope = ref_param
        return max(0.0, min(0.75, slope * max(0.0, t_sec - 2.0)))
    if ref_type == "sine":
        A, f = ref_param
        return A * math.sin(2 * math.pi * f * t_sec)
    return 0.0


def teacher_cg_update(r: float, r_feasible: float, sat_hist: list, dr_max: float) -> Tuple[float, float, float]:
    """Return (r_feasible_next, dr_max_next, sat_rate)."""
    # CG parameters
    dr_min = 0.05
    dr_cap = 2.0
    tau = 0.5
    alpha = DT / tau
    sat_hi = 0.2
    sat_lo = 0.05
    down = 0.9
    up = 1.02

    sat_rate = (1 - alpha) * (sat_hist[-1] if sat_hist else 0.0) + alpha * (1.0 if (len(sat_hist) > 0 and sat_hist[-1] == 1) else 0.0)
    if sat_rate > sat_hi:
        dr_max *= down
    elif sat_rate < sat_lo:
        dr_max *= up
    dr_max = max(dr_min, min(dr_cap, dr_max))
    dr = max(-dr_max * DT, min(dr_max * DT, r - r_feasible))
    r_feasible_next = r_feasible + dr
    return r_feasible_next, dr_max, sat_rate


def simulate_teacher(ref_type: str, ref_param, dist_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate CG teacher and return dataset (X, y) for regression."""
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    r_feasible = 0.0
    dr_max = 1.0
    sat_hist = []
    sat_prev = 0.0

    xs = []
    ys = []

    for t in range(STEPS):
        t_sec = t * DT
        r = reference_signal(ref_type, ref_param, t_sec)

        r_feasible_next, dr_max, sat_rate = teacher_cg_update(r, r_feasible, sat_hist, dr_max)

        # feature vector includes memory + saturation proxy
        feat = np.array([r, x, v, r_feasible, dist_scale, sat_prev], dtype=float)
        xs.append(feat)
        ys.append(r_feasible_next)

        # Apply control with r_feasible_next
        e = r_feasible_next - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        sat = 1 if (u_cmd_clip != u_cmd) else 0

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * dist_scale) if t < STEPS // 2 else (-3.0 * dist_scale)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        r_feasible = r_feasible_next
        sat_hist.append(sat)
        sat_prev = float(sat)

    return np.vstack(xs), np.array(ys)


def generate_dataset() -> Tuple[np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []

    # Step and ramp
    for ds in DIST_SCALES:
        xs, ys = simulate_teacher("step", 0.75, ds)
        xs_all.append(xs)
        ys_all.append(ys)
        xs, ys = simulate_teacher("ramp", 0.5 * 0.75, ds)
        xs_all.append(xs)
        ys_all.append(ys)
        xs, ys = simulate_teacher("ramp", 0.2 * 0.75, ds)
        xs_all.append(xs)
        ys_all.append(ys)

    # Sine grid
    amps = [0.25, 0.5, 0.75]
    freqs = [0.1, 0.2, 0.3, 0.5, 1.0]
    for ds in DIST_SCALES:
        for A in amps:
            for f in freqs:
                xs, ys = simulate_teacher("sine", (A, f), ds)
                xs_all.append(xs)
                ys_all.append(ys)

    X = np.vstack(xs_all)
    y = np.concatenate(ys_all)

    np.savez(os.path.join(DATA_DIR, "teacher_dataset.npz"), X=X, y=y)
    return X, y


def train_model(X: np.ndarray, y: np.ndarray) -> Tuple[Model, Dict[str, list]]:
    np.seterr(over='ignore', divide='ignore', invalid='ignore')
    rng = np.random.default_rng(10)
    idx = rng.permutation(len(X))
    X = X[idx]
    y = y[idx]

    n_train = int(0.7 * len(X))
    n_val = int(0.15 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test, y_test = X[n_train + n_val:], y[n_train + n_val:]

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-6
    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std

    w = np.zeros(X_train.shape[1])
    b = 0.0

    lr = 5e-4
    l2 = 5e-4
    epochs = 80

    train_losses = []
    val_losses = []

    for _ in range(epochs):
        pred = X_train_n @ w + b
        err = pred - y_train
        loss = float(np.mean(err ** 2))
        if not np.isfinite(loss):
            break
        train_losses.append(loss)

        grad_w = (2.0 / len(X_train_n)) * (X_train_n.T @ err) + 2.0 * l2 * w
        grad_b = (2.0 / len(X_train_n)) * np.sum(err)
        # gradient clipping
        grad_norm = float(np.linalg.norm(grad_w))
        if grad_norm > 5.0:
            grad_w = grad_w * (5.0 / grad_norm)
        if abs(grad_b) > 5.0:
            grad_b = 5.0 * np.sign(grad_b)
        w -= lr * grad_w
        b -= lr * grad_b

        val_pred = X_val_n @ w + b
        val_loss = float(np.mean((val_pred - y_val) ** 2))
        val_losses.append(val_loss)

    model = Model(w=w, b=b, x_mean=x_mean, x_std=x_std)
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_rmse": float(np.sqrt(np.mean((X_test_n @ w + b - y_test) ** 2))),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
    np.savez(os.path.join(DATA_DIR, "lcg_model.npz"), w=w, b=b, x_mean=x_mean, x_std=x_std)
    return model, metrics


def simulate_policy(ref_type: str, ref_param, dist_scale: float, policy: str, model: Model = None) -> SimResult:
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    sats = []

    r_feasible = 0.0
    dr_max = 1.0
    sat_hist = []
    sat_prev = 0.0

    for t in range(STEPS):
        t_sec = t * DT
        r = reference_signal(ref_type, ref_param, t_sec)

        if policy == "baseline":
            r_use = r
        elif policy == "cg":
            r_feasible, dr_max, _ = teacher_cg_update(r, r_feasible, sat_hist, dr_max)
            r_use = r_feasible
        elif policy == "lcg":
            feat = np.array([r, x, v, r_feasible, dist_scale, sat_prev], dtype=float)
            r_pred = float(model.predict(feat))
            r_use = max(-1.0, min(1.0, r_pred))
            r_feasible = r_use
        else:
            r_use = r

        e = r_use - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        sat = 1 if (u_cmd_clip != u_cmd) else 0

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * dist_scale) if t < STEPS // 2 else (-3.0 * dist_scale)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(r_use - x)
        sats.append(sat)
        sat_hist.append(sat)
        sat_prev = float(sat)

    errors = np.array(errors)
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors ** 2)))
    mean_abs = float(np.mean(np.abs(errors)))

    recovery = float("inf")
    if ref_type in ["step", "ramp"]:
        window = int(0.2 / DT)
        start = int(2.0 / DT)
        for i in range(start, len(errors) - window):
            if all(abs(errors[j]) < RECOVERY_EPS for j in range(i, i + window)):
                recovery = (i - start) * DT
                break

    in_band = float(np.mean(np.abs(errors) < RECOVERY_EPS))
    return SimResult(sat_total=sat_total, rmse=rmse, mean_abs=mean_abs, recovery=recovery, in_band=in_band)


def plot_training_curve(train_losses, val_losses):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label="train")
    ax.plot(val_losses, label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("L-CG Training Curve")
    ax.legend()
    out_path = os.path.join(RESULTS_DIR, "training_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_summary(metrics: Dict, eval_rows: list, curve_path: str):
    out_path = os.path.join(RESULTS_DIR, "phase4_summary.md")
    with open(out_path, "w") as f:
        f.write("# Phase 4 L-CG Summary\n\n")
        f.write("## Dataset\n")
        f.write(f"- Samples: {metrics['n_train'] + metrics['n_val'] + metrics['n_test']}\n")
        f.write(f"- Train/Val/Test: {metrics['n_train']} / {metrics['n_val']} / {metrics['n_test']}\n")
        f.write(f"- Features: [r, x, v, r_feasible_prev, dist_scale, sat_prev]\n\n")

        f.write("## Training\n")
        f.write(f"- Test RMSE (teacher r_feasible): {metrics['test_rmse']:.5f}\n")
        if curve_path:
            f.write(f"- Training curve: ![curve](./training_curve.png)\n\n")
        else:
            f.write("- Training curve: matplotlib not available\n\n")

        f.write("## Closed-loop Evaluation\n")
        f.write("| test | dist_scale | policy | sat_total | RMSE | recovery | mean|e| | %time(|e|<eps) |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|\n")
        for row in eval_rows:
            f.write("| {test} | {dist} | {policy} | {sat_total:.2%} | {rmse:.4f} | {recovery:.4f} | {mean_abs:.4f} | {in_band:.2%} |\n".format(**row))

        f.write("\n## Notes\n")
        f.write("- L-CG is a lightweight regressor trained to emulate teacher CG reference shaping.\n")
        f.write("- u_max=1.0 and dist_scale in {0.15, 0.25, 0.35} are preserved.\n")
        f.write("- Evaluation compares baseline, teacher CG, and learned L-CG across step/ramp/sine.\n")
    return out_path


def main():
    X, y = generate_dataset()
    model, metrics = train_model(X, y)
    curve_path = plot_training_curve(metrics["train_losses"], metrics["val_losses"])

    eval_rows = []
    for ds in DIST_SCALES:
        for policy in ["baseline", "cg", "lcg"]:
            eval_rows.append({
                "test": "step",
                "dist": ds,
                "policy": policy,
                **simulate_policy("step", 0.75, ds, policy, model).__dict__,
            })
            eval_rows.append({
                "test": "ramp_fast",
                "dist": ds,
                "policy": policy,
                **simulate_policy("ramp", 0.5 * 0.75, ds, policy, model).__dict__,
            })
            eval_rows.append({
                "test": "ramp_med",
                "dist": ds,
                "policy": policy,
                **simulate_policy("ramp", 0.2 * 0.75, ds, policy, model).__dict__,
            })

    amps = [0.25, 0.5, 0.75]
    freqs = [0.1, 0.2, 0.3, 0.5, 1.0]
    for ds in DIST_SCALES:
        for A in amps:
            for f in freqs:
                for policy in ["baseline", "cg", "lcg"]:
                    eval_rows.append({
                        "test": f"sine_A{A}_f{f}",
                        "dist": ds,
                        "policy": policy,
                        **simulate_policy("sine", (A, f), ds, policy, model).__dict__,
                    })

    write_summary(metrics, eval_rows, curve_path)
    print("Phase 4 L-CG complete.")


if __name__ == "__main__":
    main()
