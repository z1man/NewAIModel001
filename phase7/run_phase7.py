#!/usr/bin/env python3
"""Phase 7.1 unified tokenized QP projection runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .compiler import compile_constraints
from .config import Phase7Config, load_config, save_config
from .metrics import compute_metrics, evaluate_gates, metrics_to_dict
from .qp import project_qp
from .tokens import (
    build_tokens,
    compute_token_stats,
    default_constraint_tokens,
    reference_signal,
    token_activation_stats,
    validate_tokens,
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def default_run_id(seed: int) -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{seed}"


def baseline_controller(ref: float, state: np.ndarray, ref_prev: float, config: Phase7Config) -> float:
    r_dot = (ref - ref_prev) / config.dt
    error = ref - state[0]
    u_prop = config.kp * error + config.kd * (r_dot - state[1])
    return float(u_prop)


def simulate_episode(
    config: Phase7Config,
    policy_name: str,
    use_qp: bool,
    ref_type: str,
    seed: int,
) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    state = np.array([0.0, 0.0], dtype=float)
    error_prev = 0.0
    ref_prev = 0.0

    tokens = default_constraint_tokens()
    token_stats = token_activation_stats(tokens)

    token_list = []
    token_clip_flags = []
    u_props = []
    actions = []
    bounds_L = []
    bounds_U = []
    proj_mag = []
    infeasible_flags = []
    hard_violation_flags = []
    sat_flags = []
    errors = []
    refs = []

    for t_idx in range(config.steps):
        t = t_idx * config.dt
        if ref_type == "step":
            ref = config.ref_amp if t > 1.0 else 0.0
        else:
            ref = reference_signal(config, t)

        feat_tokens = build_tokens(config, t_idx, ref, state, error_prev, ref_prev)
        token_ok, _ = validate_tokens(feat_tokens, config.token_clip)
        token_list.append(feat_tokens)
        token_clip_flags.append(0.0 if token_ok else 1.0)

        u_prop = baseline_controller(ref, state, ref_prev, config)
        u = u_prop
        L = -np.inf
        U = np.inf
        infeasible = False
        proj = 0.0

        if use_qp:
            constraints, bounds = compile_constraints(tokens, state, ref, config)
            for tok in tokens:
                token_stats[tok.name] += 1
            qp_res = project_qp(u_prop, constraints, bounds)
            u = qp_res.action
            L, U = qp_res.L, qp_res.U
            infeasible = qp_res.infeasible
            proj = qp_res.proj_mag

        u_min = float(config.u_min[0])
        u_max = float(config.u_max[0])
        sat = float((u <= u_min + 1e-8) or (u >= u_max - 1e-8))

        accel = (u - config.damping * state[1] - config.spring * state[0]) / config.mass
        state[1] = state[1] + accel * config.dt
        state[0] = state[0] + state[1] * config.dt

        error = ref - state[0]
        error_prev = error
        ref_prev = ref

        u_props.append(u_prop)
        actions.append(u)
        bounds_L.append(L)
        bounds_U.append(U)
        proj_mag.append(proj)
        infeasible_flags.append(1.0 if infeasible else 0.0)
        hard_violation_flags.append(1.0 if (u < u_min - 1e-8 or u > u_max + 1e-8) else 0.0)
        sat_flags.append(sat)
        errors.append(error)
        refs.append(ref)

    return {
        "policy": policy_name,
        "ref_type": ref_type,
        "tokens": np.vstack(token_list),
        "token_clip_flags": np.array(token_clip_flags),
        "u_prop": np.array(u_props),
        "u": np.array(actions),
        "L": np.array(bounds_L),
        "U": np.array(bounds_U),
        "proj_mag": np.array(proj_mag),
        "infeasible_flags": np.array(infeasible_flags),
        "hard_violation_flags": np.array(hard_violation_flags),
        "sat_flags": np.array(sat_flags),
        "errors": np.array(errors),
        "refs": np.array(refs),
        "token_stats": token_stats,
    }


def evaluate_suite(config: Phase7Config, seed: int) -> List[Dict[str, np.ndarray]]:
    episodes = []
    for ref_type in ["step", "sine"]:
        episodes.append(simulate_episode(config, "baseline", False, ref_type, seed))
        episodes.append(simulate_episode(config, "baseline_qp", True, ref_type, seed))
    return episodes


def summarize_metrics(episodes: List[Dict[str, np.ndarray]], config: Phase7Config) -> Dict[str, Dict[str, float]]:
    summary = {}
    for ep in episodes:
        metrics = compute_metrics(
            ep["errors"],
            ep["sat_flags"],
            ep["hard_violation_flags"],
            ep["infeasible_flags"],
            ep["proj_mag"],
            config,
        )
        summary[f"{ep['policy']}_{ep['ref_type']}"] = metrics_to_dict(metrics)
    return summary


def write_summary(
    out_dir: Path,
    config: Phase7Config,
    metrics: Dict[str, Dict[str, float]],
    gates: Dict[str, Dict[str, float | bool]],
) -> None:
    out_path = out_dir / "phase7_summary.md"
    with open(out_path, "w") as f:
        f.write("# Phase 7.1 Tokenized QP Projection Summary\n\n")
        f.write("## Metrics (per policy)\n")
        for name, vals in metrics.items():
            f.write(f"- {name}: RMSE={vals['rmse']:.4f}, mean|e|={vals['mean_abs']:.4f}, "
                    f"sat_rate={vals['sat_rate']:.2%}, hard_violation={vals['hard_violation_rate']:.2%}, "
                    f"proj_mean={vals['proj_mean']:.4f}\n")
        f.write("\n## Gates (M1-M5)\n")
        for name, gate in gates.items():
            status = "PASS" if gate["pass"] else "FAIL"
            detail = ", ".join(f"{k}={v}" for k, v in gate.items() if k != "pass")
            f.write(f"- {name}: {status} ({detail})\n")
        f.write("\n## Notes\n")
        f.write("- Policy = baseline u_prop + QP projection with actuator/acc tokens.\n")


def save_worst_cases(out_dir: Path, episodes: List[Dict[str, np.ndarray]]) -> None:
    worst_dir = out_dir / "worst_case_trajs"
    worst_dir.mkdir(parents=True, exist_ok=True)

    def save_episode(ep: Dict[str, np.ndarray], label: str) -> None:
        np.savez(
            worst_dir / f"{label}.npz",
            u_prop=ep["u_prop"],
            u=ep["u"],
            L=ep["L"],
            U=ep["U"],
            refs=ep["refs"],
            errors=ep["errors"],
            sat_flags=ep["sat_flags"],
            proj_mag=ep["proj_mag"],
        )

    qp_eps = [ep for ep in episodes if ep["policy"] == "baseline_qp"]
    if not qp_eps:
        return
    worst_rmse = max(qp_eps, key=lambda e: float(np.sqrt(np.mean(e["errors"] ** 2))))
    worst_sat = max(qp_eps, key=lambda e: float(np.mean(e["sat_flags"])))
    worst_proj = max(qp_eps, key=lambda e: float(np.max(e["proj_mag"])))

    save_episode(worst_rmse, "worst_rmse")
    save_episode(worst_sat, "worst_sat_rate")
    save_episode(worst_proj, "worst_proj_mag")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7.1 tokenized QP projection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "phase7_default.json"),
        help="Path to config JSON",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id for artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.seed
    np.random.seed(seed)
    run_id = args.run_id or default_run_id(seed)

    out_dir = ARTIFACTS_DIR / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config_snapshot.json")

    episodes = evaluate_suite(config, seed)
    metrics = summarize_metrics(episodes, config)

    baseline_metrics = next(iter(metrics.values()))
    gates = evaluate_gates(
        compute_metrics(
            episodes[1]["errors"],
            episodes[1]["sat_flags"],
            episodes[1]["hard_violation_flags"],
            episodes[1]["infeasible_flags"],
            episodes[1]["proj_mag"],
            config,
        ),
        config,
    )

    token_stats = compute_token_stats(episodes[1]["tokens"])
    token_activation = episodes[1]["token_stats"]

    np.savez(
        out_dir / "token_stream.npz",
        tokens=episodes[1]["tokens"],
        u_prop=episodes[1]["u_prop"],
        u=episodes[1]["u"],
        refs=episodes[1]["refs"],
        errors=episodes[1]["errors"],
        token_clip_flags=episodes[1]["token_clip_flags"],
        infeasible_flags=episodes[1]["infeasible_flags"],
        proj_mag=episodes[1]["proj_mag"],
        sat_flags=episodes[1]["sat_flags"],
        L=episodes[1]["L"],
        U=episodes[1]["U"],
    )

    with open(out_dir / "token_stats.json", "w") as f:
        json.dump({"mean": token_stats.mean.tolist(), "std": token_stats.std.tolist()}, f, indent=2)

    with open(out_dir / "token_activation_stats.json", "w") as f:
        json.dump(token_activation, f, indent=2)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(out_dir / "gates.json", "w") as f:
        json.dump(gates, f, indent=2)

    with open(out_dir / "projection_report.csv", "w") as f:
        f.write("step,ref,error,u_prop,u,L,U,proj_mag,infeasible,sat\n")
        for idx in range(config.steps):
            f.write(
                f"{idx},{episodes[1]['refs'][idx]:.6f},{episodes[1]['errors'][idx]:.6f},"
                f"{episodes[1]['u_prop'][idx]:.6f},{episodes[1]['u'][idx]:.6f},"
                f"{episodes[1]['L'][idx]:.6f},{episodes[1]['U'][idx]:.6f},"
                f"{episodes[1]['proj_mag'][idx]:.6f},{int(episodes[1]['infeasible_flags'][idx])},"
                f"{int(episodes[1]['sat_flags'][idx])}\n"
            )

    write_summary(out_dir, config, metrics, gates)
    save_worst_cases(out_dir, episodes)

    with open(out_dir / "phase7_gate_report.json", "w") as f:
        json.dump(gates, f, indent=2)

    with open(out_dir / "phase7_gate_report.md", "w") as f:
        f.write("# Phase 7.1 Gate Report\n\n")
        for name, gate in gates.items():
            status = "PASS" if gate["pass"] else "FAIL"
            f.write(f"- {name}: {status} ({gate})\n")

    print(f"Phase 7.1 run complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
