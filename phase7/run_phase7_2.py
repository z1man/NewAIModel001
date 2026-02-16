#!/usr/bin/env python3
"""Phase 7.2 diagnosis-first runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .compiler import compile_constraints
from .config import Phase7Config, load_config, save_config
from .metrics import compute_metrics, metrics_to_dict
from .qp import project_qp, project_qp_soft_acc
from .tokens import AccLimitToken, ActuatorLimitToken, reference_signal

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / ".." / "phase7_reports"


def default_run_id(seed: int) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"


def baseline_controller(ref: float, state: np.ndarray, ref_prev: float, config: Phase7Config) -> float:
    r_dot = (ref - ref_prev) / config.dt
    error = ref - state[0]
    u_prop = config.kp * error + config.kd * (r_dot - state[1])
    return float(u_prop)


def simulate_episode(
    config: Phase7Config,
    ref_type: str,
    a_max: float,
    rho: float,
) -> Dict[str, np.ndarray]:
    state = np.array([0.0, 0.0], dtype=float)
    ref_prev = 0.0

    u_prop_raw = []
    u_prop_applied = []
    u_star = []
    u_applied = []
    bounds_L = []
    bounds_U = []
    proj_mag = []
    infeasible_flags = []
    hard_violation_flags = []
    sat_flags = []
    errors = []
    refs = []
    slack_vals = []
    invariant_violations = []

    act_token = ActuatorLimitToken("T_ACTUATOR_LIMIT")
    acc_token = AccLimitToken("T_ACC_LIMIT")

    for t_idx in range(config.steps):
        t = t_idx * config.dt
        if ref_type == "step":
            ref = config.ref_amp if t > 1.0 else 0.0
        else:
            ref = reference_signal(config, t)

        u_prop = baseline_controller(ref, state, ref_prev, config)
        u_prop_apply = u_prop  # no delay model in Phase7.4

        _, hard_bounds = act_token.compile(state, ref, config)
        _, soft_bounds = acc_token.compile(state, ref, config)
        soft_bounds = (soft_bounds[0], soft_bounds[1])

        qp_res = project_qp_soft_acc(u_prop, hard_bounds, soft_bounds, rho)
        u = qp_res.action
        u_apply = u  # applied directly

        # Runtime invariant check (projecting u_prop_raw)
        L, U = qp_res.L, qp_res.U
        if L <= u_prop <= U:
            if abs(u - u_prop) > 1e-6:
                invariant_violations.append((t_idx, u_prop, u, L, U, soft_bounds))
        else:
            u_expected = float(np.clip(u_prop, L, U))
            if abs(u - u_expected) > 1e-6:
                invariant_violations.append((t_idx, u_prop, u, L, U, soft_bounds))

        u_min = float(config.u_min[0])
        u_max = float(config.u_max[0])
        sat = float((u_apply <= u_min + 1e-8) or (u_apply >= u_max - 1e-8))

        accel = (u_apply - config.damping * state[1] - config.spring * state[0]) / config.mass
        state[1] = state[1] + accel * config.dt
        state[0] = state[0] + state[1] * config.dt

        error = ref - state[0]
        ref_prev = ref

        u_prop_raw.append(u_prop)
        u_prop_applied.append(u_prop_apply)
        u_star.append(u)
        u_applied.append(u_apply)
        bounds_L.append(soft_bounds[0])
        bounds_U.append(soft_bounds[1])
        proj_mag.append(qp_res.proj_mag)
        infeasible_flags.append(1.0 if qp_res.infeasible else 0.0)
        hard_violation_flags.append(1.0 if (u_apply < u_min - 1e-8 or u_apply > u_max + 1e-8) else 0.0)
        sat_flags.append(sat)
        errors.append(error)
        refs.append(ref)
        slack_vals.append(qp_res.slack)

    return {
        "u_prop_raw": np.array(u_prop_raw),
        "u_prop_applied": np.array(u_prop_applied),
        "u_star": np.array(u_star),
        "u_applied": np.array(u_applied),
        "L": np.array(bounds_L),
        "U": np.array(bounds_U),
        "proj_mag": np.array(proj_mag),
        "infeasible_flags": np.array(infeasible_flags),
        "hard_violation_flags": np.array(hard_violation_flags),
        "sat_flags": np.array(sat_flags),
        "errors": np.array(errors),
        "refs": np.array(refs),
        "slack": np.array(slack_vals),
        "invariant_violations": invariant_violations,
    }


def run_sweep(config: Phase7Config, a_max_grid: List[float], rho: float) -> Dict[str, Dict[str, float]]:
    results = {}
    for a_max in a_max_grid:
        config.a_max = a_max
        for ref_type in ["step", "sine"]:
            ep = simulate_episode(config, ref_type, a_max, rho)
            metrics = compute_metrics(
                ep["errors"],
                ep["sat_flags"],
                ep["hard_violation_flags"],
                ep["infeasible_flags"],
                ep["proj_mag"],
                config,
            )
            xi_p95 = float(np.quantile(ep["slack"], 0.95))
            u_max = float(config.u_max[0])
            sat_rate_u_star = float(np.mean(np.abs(ep["u_star"]) >= 0.95 * u_max))
            sat_rate_u_applied = float(np.mean(np.abs(ep["u_applied"]) >= 0.95 * u_max))
            results[f"a_max={a_max}_{ref_type}"] = {
                **metrics_to_dict(metrics),
                "xi_p95": xi_p95,
                "sat_rate_u_star": sat_rate_u_star,
                "sat_rate_u_applied": sat_rate_u_applied,
            }
    return results


def run_qp_passthrough_test(out_dir: Path) -> Dict[str, int]:
    rng = np.random.default_rng(7)
    total = 0
    pass_count = 0
    fail_count = 0
    failures = []

    for _ in range(10000):
        L = rng.uniform(-2.0, 0.0)
        U = rng.uniform(0.0, 2.0)
        if L > U:
            L, U = U, L
        u_prop = rng.uniform(L, U)
        res = project_qp(u_prop, [], (L, U))
        total += 1
        if abs(res.action - u_prop) <= 1e-9:
            pass_count += 1
        else:
            fail_count += 1
            failures.append((u_prop, res.action, L, U))

    for _ in range(10000):
        L = rng.uniform(-2.0, 0.0)
        U = rng.uniform(0.0, 2.0)
        if L > U:
            L, U = U, L
        u_prop = rng.uniform(-3.0, 3.0)
        if L <= u_prop <= U:
            continue
        res = project_qp(u_prop, [], (L, U))
        total += 1
        u_expected = float(np.clip(u_prop, L, U))
        if abs(res.action - u_expected) <= 1e-9:
            pass_count += 1
        else:
            fail_count += 1
            failures.append((u_prop, res.action, L, U))

    out_path = out_dir / "qp_psthrough_unit_test.txt"
    with open(out_path, "w") as f:
        f.write(f"pass_count={pass_count}\n")
        f.write(f"fail_count={fail_count}\n")
        f.write(f"total={total}\n")
        if failures:
            f.write("failures (u_prop,u_star,L,U):\n")
            for row in failures[:20]:
                f.write(f"{row}\n")

    return {"pass_count": pass_count, "fail_count": fail_count, "total": total}


def plot_traces(out_dir: Path, ep: Dict[str, np.ndarray], tag: str) -> None:
    import matplotlib.pyplot as plt

    t = np.arange(len(ep["u_star"]))
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(t, ep["u_prop_raw"], label="u_prop_raw")
    axes[0].plot(t, ep["u_prop_applied"], label="u_prop_applied")
    axes[0].plot(t, ep["u_star"], label="u_star")
    axes[0].plot(t, ep["u_applied"], label="u_applied")
    axes[0].plot(t, ep["L"], label="L", linestyle="--")
    axes[0].plot(t, ep["U"], label="U", linestyle="--")
    axes[0].set_ylabel("u")
    axes[0].legend()

    axes[1].plot(t, ep["slack"], label="xi")
    axes[1].set_ylabel("xi")
    axes[1].legend()

    axes[2].plot(t, ep["u_prop_raw"] - ep["u_star"], label="u_prop_raw - u_star")
    axes[2].plot(t, ep["u_star"] - ep["u_applied"], label="u_star - u_applied")
    axes[2].set_ylabel("delta u")
    axes[2].set_xlabel("t")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(out_dir / f"trace_{tag}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7.2 diagnosis")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "phase7_default.json"),
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id for reports")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.seed

    run_id = args.run_id or default_run_id(seed)
    out_dir = REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / f"phase7_2_config_snapshot_{run_id}.json")

    omega = 2 * np.pi * config.ref_freq
    rddot_max = config.ref_amp * omega**2

    rho = 5.0
    a_max_grid = [0.5, 1.0, 2.0]

    # QP passthrough unit test
    unit_test = run_qp_passthrough_test(out_dir)

    sweep = run_sweep(config, a_max_grid, rho)

    # stop if any hard violation
    if any(vals["hard_violation_rate"] > 0 for vals in sweep.values()):
        raise RuntimeError("Hard actuator violation detected; stopping per Phase7.4")

    # plot example traces for a_max=1.0 and 2.0
    ep_step = simulate_episode(config, "step", 1.0, rho)
    ep_sine = simulate_episode(config, "sine", 1.0, rho)
    ep_step_hi = simulate_episode(config, "step", 2.0, rho)
    ep_sine_hi = simulate_episode(config, "sine", 2.0, rho)

    violations = (
        ep_step["invariant_violations"]
        + ep_sine["invariant_violations"]
        + ep_step_hi["invariant_violations"]
        + ep_sine_hi["invariant_violations"]
    )

    plot_traces(out_dir, ep_step, f"step_{run_id}_a1")
    plot_traces(out_dir, ep_sine, f"sine_{run_id}_a1")
    plot_traces(out_dir, ep_step_hi, f"step_{run_id}_a2")
    plot_traces(out_dir, ep_sine_hi, f"sine_{run_id}_a2")

    report_path = out_dir / f"phase7_2_diagnosis_{run_id}.md"
    with open(report_path, "w") as f:
        f.write("# Phase 7.2 Diagnosis Report\n\n")
        f.write("## Analytical Check\n")
        f.write(f"- ref_amp A = {config.ref_amp}\n")
        f.write(f"- omega = {omega:.4f}\n")
        f.write(f"- |r_ddot|_max = A*omega^2 = {rddot_max:.4f}\n")
        f.write(f"- configured a_max = {config.a_max}\n\n")

        f.write("## QP Passthrough Unit Test\n")
        f.write(f"- pass_count = {unit_test['pass_count']} / {unit_test['total']}\n")
        f.write(f"- fail_count = {unit_test['fail_count']}\n\n")

        f.write("## Runtime Invariant Violations (first 20)\n")
        if violations:
            f.write("| t | u_prop | u* | L | U | bounds |\n")
            f.write("|---:|---:|---:|---:|---:|---:|\n")
            for t, u_prop, u_star, L, U, bounds in violations[:20]:
                f.write(
                    f"| {t} | {u_prop:.6f} | {u_star:.6f} | {L:.6f} | {U:.6f} | {bounds} |\n"
                )
        else:
            f.write("- none\n")
        f.write("\n")

        f.write("## a_max Sweep (baseline+QP)\n")
        f.write("| case | RMSE | sat_rate_u_star | sat_rate_u_applied | proj_mean | xi_p95 |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for key, vals in sweep.items():
            f.write(
                f"| {key} | {vals['rmse']:.4f} | {vals['sat_rate_u_star']:.2%} | {vals['sat_rate_u_applied']:.2%} | {vals['proj_mean']:.4f} | {vals['xi_p95']:.4f} |\n"
            )

        f.write("\n## Slack Stats (a_max=1.0)\n")
        f.write(
            f"- step xi_mean={ep_step['slack'].mean():.4f}, p95={np.quantile(ep_step['slack'],0.95):.4f}, p99={np.quantile(ep_step['slack'],0.99):.4f}\n"
        )
        f.write(
            f"- sine xi_mean={ep_sine['slack'].mean():.4f}, p95={np.quantile(ep_sine['slack'],0.95):.4f}, p99={np.quantile(ep_sine['slack'],0.99):.4f}\n\n"
        )

        f.write("## Plots\n")
        f.write(f"- trace_step_a1: trace_step_{run_id}_a1.png\n")
        f.write(f"- trace_sine_a1: trace_sine_{run_id}_a1.png\n")
        f.write(f"- trace_step_a2: trace_step_{run_id}_a2.png\n")
        f.write(f"- trace_sine_a2: trace_sine_{run_id}_a2.png\n")

    print(f"Phase 7.2 diagnosis complete. Report in {report_path}")


if __name__ == "__main__":
    main()
