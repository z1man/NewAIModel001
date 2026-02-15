#!/usr/bin/env python3
"""Phase 1.6 A2-4: pure identification isolation."""
import math
import os
import random
import numpy as np
import matplotlib.pyplot as plt

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

OPEN_LOOP_AMP = 2.0
OPEN_LOOP_FREQ = 1.5

CLIP_MAX = 3.0


def simulate(disable_friction=False, disable_clip=False, disable_delay=False):
    x, v = 0.0, 0.0
    delay = 0 if disable_delay else DELAY_STEPS
    u_buffer = [0.0 for _ in range(delay + 1)]

    y_chain = []
    d_quad_true = []
    residual = []
    sign_v = []
    v_vals = []
    clip_events = []

    for t in range(STEPS):
        u_cmd = OPEN_LOOP_AMP * math.sin(2 * math.pi * OPEN_LOOP_FREQ * t * DT)
        if not disable_clip:
            u_cmd = max(-CLIP_MAX, min(CLIP_MAX, u_cmd))
        clipped = abs(u_cmd) >= CLIP_MAX - 1e-6

        u_buffer.append(u_cmd)
        u_applied = u_buffer.pop(0)

        d_quad = (C_TRUE) * v * abs(v)  # same sign as plant term
        friction = 0.0 if disable_friction else FRICTION * v

        # plant uses -d_quad
        a_true = (u_applied - friction - d_quad) / MASS
        v = v + a_true * DT
        x = x + v * DT

        y = u_applied - MASS * a_true
        y_chain.append(y)
        d_quad_true.append(d_quad)
        residual.append(y - d_quad)
        sign_v.append(math.copysign(1.0, v) if v != 0 else 0.0)
        v_vals.append(v)
        clip_events.append(1 if clipped else 0)

    return {
        "y_chain": np.array(y_chain),
        "d_quad": np.array(d_quad_true),
        "residual": np.array(residual),
        "sign_v": np.array(sign_v),
        "v": np.array(v_vals),
        "clip": np.array(clip_events),
    }


def regression_c(d_quad, v):
    phi = v * np.abs(v)
    c_hat = np.sum(phi * d_quad) / max(1e-6, np.sum(phi * phi))
    c_relerr = abs(c_hat - C_TRUE) / max(1e-6, C_TRUE)
    # R2
    y_pred = c_hat * phi
    ss_res = np.sum((d_quad - y_pred) ** 2)
    ss_tot = np.sum((d_quad - np.mean(d_quad)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return c_hat, c_relerr, r2


def run():
    data = simulate()
    c_hat, c_relerr, r2 = regression_c(data["d_quad"], data["v"])

    # plots
    plt.figure(figsize=(6,5))
    plt.scatter(data["d_quad"], data["y_chain"], s=6, alpha=0.5)
    plt.xlabel("d_quad_true")
    plt.ylabel("y_chain")
    plt.title("y_chain vs d_quad_true")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "y_chain_vs_d_quad.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(data["sign_v"], data["residual"], s=6, alpha=0.5)
    plt.xlabel("sign(v)")
    plt.ylabel("residual")
    plt.title("residual vs sign(v)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_vs_signv.png"))
    plt.close()

    plt.figure(figsize=(6,5))
    plt.scatter(data["clip"], data["residual"], s=6, alpha=0.5)
    plt.xlabel("clip event")
    plt.ylabel("residual")
    plt.title("residual vs clip events")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "residual_vs_clip.png"))
    plt.close()

    # toggles
    toggles = {
        "baseline": data,
        "no_friction": simulate(disable_friction=True),
        "no_clip": simulate(disable_clip=True),
        "no_delay": simulate(disable_delay=True),
    }

    # residual correlation analysis
    def corr(a, b):
        if len(a) < 2:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    conclusions = []
    for name, d in toggles.items():
        res = d["residual"]
        conclusions.append({
            "name": name,
            "corr_signv": corr(res, d["sign_v"]),
            "corr_v": corr(res, d["v"]),
            "corr_abs_v": corr(res, np.abs(d["v"])),
            "corr_clip": corr(res, d["clip"]),
        })

    # write report
    report_path = os.path.join(RESULTS_DIR, "oracle_id_report_A2_4.md")
    with open(report_path, "w") as f:
        f.write("# Phase 1.6 A2-4 Report\n\n")
        f.write("## A2-4.1 Ground-truth regression\n")
        f.write(f"c_hat={c_hat:.4f}, c_relerr={c_relerr:.4f}, R2={r2:.4f}\n\n")
        f.write("## A2-4.2 Residual analysis (baseline)\n")
        f.write(f"corr(residual, sign(v))={conclusions[0]['corr_signv']:.4f}\n")
        f.write(f"corr(residual, v)={conclusions[0]['corr_v']:.4f}\n")
        f.write(f"corr(residual, |v|)={conclusions[0]['corr_abs_v']:.4f}\n")
        f.write(f"corr(residual, clip)={conclusions[0]['corr_clip']:.4f}\n\n")
        f.write("## A2-4.3 Toggle test\n")
        for c in conclusions:
            f.write(f"{c['name']}: signv={c['corr_signv']:.4f}, v={c['corr_v']:.4f}, |v|={c['corr_abs_v']:.4f}, clip={c['corr_clip']:.4f}\n")

    print("A2-4 complete.")


if __name__ == "__main__":
    run()
