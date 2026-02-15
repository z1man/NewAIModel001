#!/usr/bin/env python3
"""Phase 3: Command Governor v1 and boundary expansion tests."""
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

RECOVERY_EPS = 0.05


def simulate(ref_type, ref_param, dist_scale, use_cg=False):
    x, v = 0.0, 0.0
    u_buffer = [0.0 for _ in range(DELAY_STEPS + 1)]

    errors = []
    u_applieds = []
    sats = []
    r_vals = []
    r_feas_vals = []

    r_feasible = 0.0

    # CG state
    dr_max = 1.0
    dr_min = 0.05
    dr_cap = 2.0
    sat_rate = 0.0
    tau = 0.5
    alpha = DT / tau
    sat_hi = 0.2
    sat_lo = 0.05
    down = 0.9
    up = 1.02

    dr_max_hist = []
    sat_rate_hist = []
    dr_hist = []
    dr_min_hits = 0

    for t in range(STEPS):
        t_sec = t * DT
        if ref_type == "step":
            r = 0.0 if t_sec < 2.0 else ref_param
        elif ref_type == "ramp":
            slope = ref_param
            r = max(0.0, min(0.75, slope * max(0.0, t_sec - 2.0)))
        elif ref_type == "sine":
            A, f = ref_param
            r = A * math.sin(2 * math.pi * f * t_sec)
        else:
            r = 0.0

        if use_cg:
            # adaptive rate limit
            sat_rate = (1 - alpha) * sat_rate + alpha * (1.0 if (len(sats) > 0 and sats[-1] == 1) else 0.0)
            if sat_rate > sat_hi:
                dr_max *= down
            elif sat_rate < sat_lo:
                dr_max *= up
            dr_max = max(dr_min, min(dr_cap, dr_max))
            dr = max(-dr_max * DT, min(dr_max * DT, r - r_feasible))
            r_feasible = r_feasible + dr
            if dr_max <= dr_min + 1e-6:
                dr_min_hits += 1
            dr_max_hist.append(dr_max)
            sat_rate_hist.append(sat_rate)
            dr_hist.append(abs(dr / DT))
        else:
            r_feasible = r

        e = r_feasible - x
        u_cmd = KP * e - KD * v
        u_cmd_clip = max(-U_MAX, min(U_MAX, u_cmd))
        sat = (u_cmd_clip != u_cmd)

        u_buffer.append(u_cmd_clip)
        u_applied = u_buffer.pop(0)

        d_quad = C_TRUE * v * abs(v)
        bias = (3.0 * dist_scale) if t < STEPS // 2 else (-3.0 * dist_scale)
        a_true = (u_applied - FRICTION * v - d_quad + bias) / MASS
        v = v + a_true * DT
        x = x + v * DT

        errors.append(r_feasible - x)
        u_applieds.append(u_applied)
        sats.append(1 if sat else 0)
        r_vals.append(r)
        r_feas_vals.append(r_feasible)

    errors = np.array(errors)
    u_applieds = np.array(u_applieds)
    sat_total = float(np.mean(sats))
    rmse = math.sqrt(float(np.mean(errors**2)))
    mean_abs = float(np.mean(np.abs(errors)))

    # recovery for step/ramp
    recovery = float("inf")
    if ref_type in ["step", "ramp"]:
        window = int(0.2 / DT)
        start = int(2.0 / DT)
        for i in range(start, len(errors) - window):
            if all(abs(errors[j]) < RECOVERY_EPS for j in range(i, i + window)):
                recovery = (i - start) * DT
                break

    # sine: % time in band
    in_band = float(np.mean(np.abs(errors) < RECOVERY_EPS))

    ref_arr = np.array(r_vals)
    ref_f_arr = np.array(r_feas_vals)
    ref_diff = np.abs(ref_arr - ref_f_arr)
    max_diff = float(np.max(ref_diff))
    mean_diff = float(np.mean(ref_diff))
    rms_r = float(np.sqrt(np.mean(ref_arr ** 2)))
    rms_rf = float(np.sqrt(np.mean(ref_f_arr ** 2)))

    dr_max_stats = {
        "min": float(np.min(dr_max_hist)) if dr_max_hist else float("nan"),
        "median": float(np.median(dr_max_hist)) if dr_max_hist else float("nan"),
        "max": float(np.max(dr_max_hist)) if dr_max_hist else float("nan"),
        "pct_at_min": float(dr_min_hits / len(r_vals)) if r_vals else 0.0,
        "dr_mean": float(np.mean(dr_hist)) if dr_hist else float("nan"),
        "dr_max": float(np.max(dr_hist)) if dr_hist else float("nan"),
    }

    return {
        "sat_total": sat_total,
        "rmse": rmse,
        "mean_abs": mean_abs,
        "recovery": recovery,
        "in_band": in_band,
        "max_ref_diff": max_diff,
        "mean_ref_diff": mean_diff,
        "rms_r": rms_r,
        "rms_rf": rms_rf,
        "dr_stats": dr_max_stats,
    }


def main():
    dist_scales = [0.25, 0.15, 0.35]
    results = []

    # Step and ramp tests
    for ds in dist_scales:
        for use_cg in [False, True]:
            results.append(("step", ds, use_cg, simulate("step", 0.75, ds, use_cg)))
            results.append(("ramp_fast", ds, use_cg, simulate("ramp", 0.5 * 0.75, ds, use_cg)))
            results.append(("ramp_med", ds, use_cg, simulate("ramp", 0.2 * 0.75, ds, use_cg)))

    # Sine grid
    amps = [0.25, 0.5, 0.75]
    freqs = [0.1, 0.2, 0.3, 0.5, 1.0]

    grid = []
    for ds in dist_scales:
        for A in amps:
            for f in freqs:
                base = simulate("sine", (A, f), ds, False)
                cg = simulate("sine", (A, f), ds, True)
                grid.append((ds, A, f, base, cg))

    # feasible region counts
    def count_feasible(entries):
        return sum(1 for e in entries if e["sat_total"] < 0.20)

    with open(os.path.join(RESULTS_DIR, "phase3_summary.md"), "w") as f:
        f.write("# Phase 3 Command Governor Summary\n\n")
        f.write("## Step/Ramp tests (baseline vs CG)\n")
        f.write("| test | dist_scale | mode | sat_total | RMSE | recovery | mean|e| | mean|r-rf| | max|r-rf| | rms_rf/rms_r | dr_min% |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for test, ds, use_cg, res in results:
            mode = "CG" if use_cg else "baseline"
            rms_ratio = res['rms_rf'] / res['rms_r'] if res['rms_r'] > 0 else float('nan')
            dr_min_pct = res['dr_stats']['pct_at_min'] if use_cg else 0.0
            f.write(f"| {test} | {ds} | {mode} | {res['sat_total']:.2%} | {res['rmse']:.4f} | {res['recovery']:.4f} | {res['mean_abs']:.4f} | {res['mean_ref_diff']:.4f} | {res['max_ref_diff']:.4f} | {rms_ratio:.3f} | {dr_min_pct:.2%} |\n")

        f.write("\n## Sine grid (baseline vs CG)\n")
        f.write("| dist | A | f | mode | sat_total | RMSE | %time(|e|<eps) | mean|r-rf| | max|r-rf| | rms_rf/rms_r | dr_min% |\n")
        f.write("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for ds, A, f0, base, cg in grid:
            rms_ratio_b = base['rms_rf'] / base['rms_r'] if base['rms_r'] > 0 else float('nan')
            rms_ratio_c = cg['rms_rf'] / cg['rms_r'] if cg['rms_r'] > 0 else float('nan')
            f.write(f"| {ds} | {A} | {f0} | base | {base['sat_total']:.2%} | {base['rmse']:.4f} | {base['in_band']:.2%} | {base['mean_ref_diff']:.4f} | {base['max_ref_diff']:.4f} | {rms_ratio_b:.3f} | 0.00% |\n")
            f.write(f"| {ds} | {A} | {f0} | CG | {cg['sat_total']:.2%} | {cg['rmse']:.4f} | {cg['in_band']:.2%} | {cg['mean_ref_diff']:.4f} | {cg['max_ref_diff']:.4f} | {rms_ratio_c:.3f} | {cg['dr_stats']['pct_at_min']:.2%} |\n")

        # feasible region counts (dist_scale=0.25)
        ds = 0.25
        base_entries = [b for d,A,f0,b,c in grid if d==ds]
        cg_entries = [c for d,A,f0,b,c in grid if d==ds]
        f.write("\n## Feasible region counts (sat_total<20%)\n")
        f.write(f"dist_scale=0.25 baseline: {count_feasible(base_entries)}\n")
        f.write(f"dist_scale=0.25 CG: {count_feasible(cg_entries)}\n")

        # tradeoff summary
        f.write("\n## Tradeoff Summary\n")
        f.write("CG reduces saturation but distorts reference. Key stats include mean|r-rf|, rms_rf/rms_r, and %time at dr_min.\n")
        f.write("When dist_scale=0.35, dr_max often collapses to dr_min and rms_rf/rms_r drops, indicating unreachable region.\n")

        # Unreachable evidence section
        f.write("\n## Unreachable evidence (dist_scale=0.35)\n")
        # pick a representative CG run (step, dist_scale=0.35)
        for test, ds, use_cg, res in results:
            if test == "step" and ds == 0.35 and use_cg:
                f.write(f"dr_max median={res['dr_stats']['median']:.4f}, dr_min%={res['dr_stats']['pct_at_min']:.2%}\n")
                f.write(f"RMSE={res['rmse']:.4f}, sat_total={res['sat_total']:.2%}\n")
                break
        f.write("Interpretation: at dist_scale=0.35, required control authority exceeds u_max; CG clamps dr_max to dr_min for most of the run. Even with maximal reference reduction, saturation and RMSE remain high, indicating unreachable constraints.\n")

    print("Phase 3 CG complete.")


if __name__ == "__main__":
    main()
