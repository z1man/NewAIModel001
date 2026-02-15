#!/usr/bin/env python3
"""Diagnose gating signals and propose thresholds."""
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


def simulate_signals():
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]
    abs_e = []
    abs_de = []
    abs_v = []

    for t in range(STEPS):
        target = 1.0
        e = target - x
        de = -v
        u_base = KP * e - KD * v

        u_buffer.append(u_base)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = 3.0 if t < STEPS // 2 else -3.0

        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        abs_e.append(abs(e))
        abs_de.append(abs(de))
        abs_v.append(abs(v))

    return np.array(abs_e), np.array(abs_de), np.array(abs_v)


def quantiles(arr):
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def main():
    abs_e, abs_de, abs_v = simulate_signals()
    q_e = quantiles(abs_e)
    q_de = quantiles(abs_de)
    q_v = quantiles(abs_v)

    # candidate thresholds from lower quantiles
    candidates = [0.1, 0.2, 0.3]
    e_eps_list = [float(np.percentile(abs_e, p * 100)) for p in candidates]
    de_eps_list = [float(np.percentile(abs_de, p * 100)) for p in candidates]
    v_min_list = [float(np.percentile(abs_v, p * 100)) for p in candidates]

    rows = []
    for e_eps in e_eps_list:
        for de_eps in de_eps_list:
            for v_min in v_min_list:
                cond = (abs_e < e_eps) & (abs_de < de_eps) & (abs_v > v_min)
                pct = float(np.mean(cond))
                rows.append((e_eps, de_eps, v_min, pct))

    # choose a suggested config closest to 10% enable
    best = min(rows, key=lambda r: abs(r[3] - 0.1))

    with open(os.path.join(RESULTS_DIR, "gating_diagnose_report.md"), "w") as f:
        f.write("# Gating diagnostics\n\n")
        f.write("## Quantiles\n")
        f.write(f"abs_e: {q_e}\n")
        f.write(f"abs_de: {q_de}\n")
        f.write(f"abs_v: {q_v}\n\n")
        f.write("## Candidate threshold grid (pct enabled)\n")
        for e_eps, de_eps, v_min, pct in rows[:20]:
            f.write(f"e_eps={e_eps:.4f}, de_eps={de_eps:.4f}, v_min={v_min:.4f} => {pct:.2%}\n")
        f.write("\n## Suggested\n")
        f.write(f"e_eps={best[0]:.4f}, de_eps={best[1]:.4f}, v_min={best[2]:.4f}, enabled={best[3]:.2%}\n")

    print("Gating diagnostics complete.")


if __name__ == "__main__":
    main()
