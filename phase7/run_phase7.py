#!/usr/bin/env python3
"""Phase 7 tokenized QP projection runner."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np

from .config import Phase7Config, load_config, save_config
from .metrics import compute_metrics, evaluate_gates, metrics_to_dict
from .qp import project_qp
from .tokens import build_tokens, compute_token_stats, reference_signal, validate_tokens

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def policy_action(tokens: np.ndarray) -> np.ndarray:
    weights = np.array(
        [
            [0.0, 1.2, -0.6, -0.4, 0.7, 0.3, 0.5],
            [0.0, 0.4, -0.2, -0.2, 0.3, 0.15, 0.25],
        ],
        dtype=float,
    )
    bias = np.array([0.0, 0.0], dtype=float)
    return weights @ tokens + bias


def simulate(config: Phase7Config) -> Dict[str, np.ndarray]:
    state = np.array([0.0, 0.0], dtype=float)
    error_prev = 0.0
    ref_prev = 0.0

    token_list = []
    token_clip_flags = []
    actions = []
    projected_actions = []
    qp_violation_flags = []
    qp_violation_mag = []
    sat_flags = []
    errors = []
    refs = []

    u_min = np.array(config.u_min, dtype=float)
    u_max = np.array(config.u_max, dtype=float)

    for t_idx in range(config.steps):
        t = t_idx * config.dt
        ref = reference_signal(config, t)
        tokens = build_tokens(config, t_idx, ref, state, error_prev, ref_prev)
        token_ok, token_max_abs = validate_tokens(tokens, config.token_clip)
        token_list.append(tokens)
        token_clip_flags.append(0.0 if token_ok else 1.0)

        u_des = policy_action(tokens)
        qp_res = project_qp(u_des, u_min, u_max, config.sum_u_max)
        u = qp_res.action

        sat = float(np.any((u <= u_min + 1e-8) | (u >= u_max - 1e-8)))

        accel = (
            config.control_gain * u[0]
            + config.bias_gain * u[1]
            - config.damping * state[1]
            - config.spring * state[0]
        )
        state[1] = state[1] + accel * config.dt
        state[0] = state[0] + state[1] * config.dt

        error = ref - state[0]
        error_prev = error
        ref_prev = ref

        actions.append(u_des)
        projected_actions.append(u)
        qp_violation_flags.append(1.0 if qp_res.violated else 0.0)
        qp_violation_mag.append(qp_res.violation_mag)
        sat_flags.append(sat)
        errors.append(error)
        refs.append(ref)

    return {
        "tokens": np.vstack(token_list),
        "token_clip_flags": np.array(token_clip_flags),
        "actions": np.vstack(actions),
        "projected_actions": np.vstack(projected_actions),
        "qp_violation_flags": np.array(qp_violation_flags),
        "qp_violation_mag": np.array(qp_violation_mag),
        "sat_flags": np.array(sat_flags),
        "errors": np.array(errors),
        "refs": np.array(refs),
    }


def write_summary(
    out_dir: Path,
    config: Phase7Config,
    metrics: Dict[str, float],
    gates: Dict[str, Dict[str, float | bool]],
) -> None:
    out_path = out_dir / "phase7_summary.md"
    with open(out_path, "w") as f:
        f.write("# Phase 7 Tokenized QP Projection Summary\n\n")
        f.write("## Metrics\n")
        f.write(f"- RMSE: {metrics['rmse']:.4f}\n")
        f.write(f"- Mean |e|: {metrics['mean_abs']:.4f}\n")
        f.write(f"- Max |e|: {metrics['max_abs']:.4f}\n")
        f.write(f"- In-band: {metrics['in_band']:.2%}\n")
        f.write(f"- Saturation rate: {metrics['sat_rate']:.2%}\n")
        f.write(f"- QP violation rate: {metrics['qp_violation_rate']:.2%}\n")
        f.write(f"- Token clip rate: {metrics['token_clip_rate']:.2%}\n\n")

        f.write("## Gates (M1-M5)\n")
        for name, gate in gates.items():
            status = "PASS" if gate["pass"] else "FAIL"
            detail = ", ".join(
                f"{k}={v}" for k, v in gate.items() if k != "pass"
            )
            f.write(f"- {name}: {status} ({detail})\n")

        f.write("\n## Notes\n")
        f.write("- Tokens are clipped at config.token_clip; any clip triggers M1.\n")
        f.write("- QP projection enforces bounds and sum constraint.\n")
        f.write("- M3-M5 thresholds are configurable in the Phase7 config.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7 tokenized QP projection")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "phase7_default.json"),
        help="Path to config JSON",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id for artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = args.run_id or default_run_id()

    out_dir = ARTIFACTS_DIR / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config_snapshot.json")

    sim = simulate(config)
    token_stats = compute_token_stats(sim["tokens"])

    metrics = compute_metrics(
        sim["errors"],
        sim["sat_flags"],
        sim["qp_violation_flags"],
        sim["token_clip_flags"],
        config,
    )
    metrics_dict = metrics_to_dict(metrics)
    gates = evaluate_gates(metrics, config)

    np.savez(
        out_dir / "token_stream.npz",
        tokens=sim["tokens"],
        actions=sim["actions"],
        projected_actions=sim["projected_actions"],
        refs=sim["refs"],
        errors=sim["errors"],
        token_clip_flags=sim["token_clip_flags"],
        qp_violation_flags=sim["qp_violation_flags"],
        qp_violation_mag=sim["qp_violation_mag"],
        sat_flags=sim["sat_flags"],
    )

    with open(out_dir / "token_stats.json", "w") as f:
        json.dump(
            {
                "mean": token_stats.mean.tolist(),
                "std": token_stats.std.tolist(),
            },
            f,
            indent=2,
        )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)

    with open(out_dir / "gates.json", "w") as f:
        json.dump(gates, f, indent=2)

    with open(out_dir / "projection_report.csv", "w") as f:
        f.write("step,ref,error,u0,u1,u0_proj,u1_proj,token_clip,qp_violation,sat\n")
        for idx in range(config.steps):
            u_des = sim["actions"][idx]
            u_proj = sim["projected_actions"][idx]
            f.write(
                f"{idx},{sim['refs'][idx]:.6f},{sim['errors'][idx]:.6f},"
                f"{u_des[0]:.6f},{u_des[1]:.6f},{u_proj[0]:.6f},{u_proj[1]:.6f},"
                f"{int(sim['token_clip_flags'][idx])},"
                f"{int(sim['qp_violation_flags'][idx])},"
                f"{int(sim['sat_flags'][idx])}\n"
            )

    write_summary(out_dir, config, metrics_dict, gates)

    print(f"Phase 7 run complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
