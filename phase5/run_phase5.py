#!/usr/bin/env python3
"""Phase 5 production L-CG pipeline runner."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import Phase5Config, load_config, save_config
from .env_runner import initial_state, reference_signal, rollout, step_dynamics
from .metrics import compute_kpis, kpis_to_dict
from .policy import baseline_policy, cg_v1_policy, student_policy, train_classifier
from .teacher import ActionGrid, horizon_teacher_action

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_random_reference(config: Phase5Config, seed: int) -> Dict[str, np.ndarray | float]:
    rng = np.random.default_rng(seed)
    seq = np.zeros(config.steps)
    t = np.arange(config.steps) * config.dt
    segments = [
        (rng.uniform(0.1, 0.75), rng.uniform(0.05, 1.0)),
        (rng.uniform(0.1, 0.75), rng.uniform(0.05, 1.0)),
        (rng.uniform(0.1, 0.75), rng.uniform(0.05, 1.0)),
    ]
    seg_len = config.steps // len(segments)
    for i, (amp, freq) in enumerate(segments):
        start = i * seg_len
        end = config.steps if i == len(segments) - 1 else (i + 1) * seg_len
        seq[start:end] = amp * np.sin(2 * np.pi * freq * t[start:end])
    return {"seq": seq, "dt": config.dt}


def teacher_episode(
    ref_type: str,
    ref_param,
    dist_scale: float,
    config: Phase5Config,
    action_grid: ActionGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    state = initial_state(config)
    xs: List[np.ndarray] = []
    ys: List[int] = []

    for t in range(config.steps):
        r = reference_signal(ref_type, ref_param, t * config.dt)
        feat = np.array(
            [r, state.x, state.v, state.r_feasible, dist_scale, state.sat_prev], dtype=float
        )
        action_idx, _, next_state = horizon_teacher_action(
            r,
            state,
            dist_scale,
            config,
            action_grid,
            t,
            ref_type,
            ref_param,
        )
        xs.append(feat)
        ys.append(action_idx)
        state = next_state

    return np.vstack(xs), np.array(ys)


def student_episode(
    ref_type: str,
    ref_param,
    dist_scale: float,
    config: Phase5Config,
    action_grid: ActionGrid,
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    state = initial_state(config)
    xs: List[np.ndarray] = []
    ys: List[int] = []

    for t in range(config.steps):
        r = reference_signal(ref_type, ref_param, t * config.dt)
        feat = np.array(
            [r, state.x, state.v, state.r_feasible, dist_scale, state.sat_prev], dtype=float
        )
        action_idx, _, _ = horizon_teacher_action(
            r,
            state,
            dist_scale,
            config,
            action_grid,
            t,
            ref_type,
            ref_param,
        )
        xs.append(feat)
        ys.append(action_idx)
        r_use, state = student_policy(r, state, dist_scale, config, model, action_grid)
        state, _, _ = step_dynamics(state, r_use, dist_scale, config, t)
    return np.vstack(xs), np.array(ys)


def collect_dataset(config: Phase5Config, action_grid: ActionGrid) -> Tuple[np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []

    for seed in config.train_seeds:
        for ds in config.dist_scales:
            for ref_type, ref_param in [
                ("step", 0.75),
                ("ramp", 0.5 * 0.75),
                ("ramp", 0.2 * 0.75),
            ]:
                xs, ys = teacher_episode(ref_type, ref_param, ds, config, action_grid)
                xs_all.append(xs)
                ys_all.append(ys)

        for ds in config.dist_scales:
            for amp in config.eval_sine_amps:
                for freq in config.eval_sine_freqs:
                    xs, ys = teacher_episode("sine", (amp, freq), ds, config, action_grid)
                    xs_all.append(xs)
                    ys_all.append(ys)

        for idx in range(config.random_ref_count):
            seq = generate_random_reference(config, seed + idx * 100)
            for ds in config.dist_scales:
                xs, ys = teacher_episode("custom", seq, ds, config, action_grid)
                xs_all.append(xs)
                ys_all.append(ys)

    X = np.vstack(xs_all)
    y = np.concatenate(ys_all)
    return X, y


def collect_dagger_data(
    config: Phase5Config, action_grid: ActionGrid, model
) -> Tuple[np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []
    for seed in config.train_seeds:
        for ds in config.dist_scales:
            for ref_type, ref_param in [
                ("step", 0.75),
                ("ramp", 0.5 * 0.75),
                ("ramp", 0.2 * 0.75),
            ]:
                xs, ys = student_episode(ref_type, ref_param, ds, config, action_grid, model)
                xs_all.append(xs)
                ys_all.append(ys)

        for ds in config.dist_scales:
            for amp in config.eval_sine_amps:
                for freq in config.eval_sine_freqs:
                    xs, ys = student_episode("sine", (amp, freq), ds, config, action_grid, model)
                    xs_all.append(xs)
                    ys_all.append(ys)

        for idx in range(config.random_ref_count):
            seq = generate_random_reference(config, seed + idx * 100)
            for ds in config.dist_scales:
                xs, ys = student_episode("custom", seq, ds, config, action_grid, model)
                xs_all.append(xs)
                ys_all.append(ys)

    return np.vstack(xs_all), np.concatenate(ys_all)


def evaluate_policies(config: Phase5Config, action_grid: ActionGrid, model) -> List[Dict]:
    eval_rows = []
    for ds in config.dist_scales:
        for policy_name, policy_fn in [
            ("baseline", baseline_policy),
            ("cg_v1", cg_v1_policy),
            ("lcg_v2", lambda r, s, d, c: student_policy(r, s, d, c, model, action_grid)),
        ]:
            for ref_type, ref_param, label in [
                ("step", 0.75, "step"),
                ("ramp", 0.5 * 0.75, "ramp_fast"),
                ("ramp", 0.2 * 0.75, "ramp_med"),
            ]:
                trace = rollout(ref_type, ref_param, ds, config, policy_fn)
                kpis = compute_kpis(trace, config)
                eval_rows.append(
                    {
                        "test": label,
                        "dist": ds,
                        "policy": policy_name,
                        **kpis_to_dict(kpis),
                    }
                )

    for ds in config.dist_scales:
        for amp in config.eval_sine_amps:
            for freq in config.eval_sine_freqs:
                for policy_name, policy_fn in [
                    ("baseline", baseline_policy),
                    ("cg_v1", cg_v1_policy),
                    ("lcg_v2", lambda r, s, d, c: student_policy(r, s, d, c, model, action_grid)),
                ]:
                    label = f"sine_A{amp}_f{freq}"
                    trace = rollout("sine", (amp, freq), ds, config, policy_fn)
                    kpis = compute_kpis(trace, config)
                    eval_rows.append(
                        {
                            "test": label,
                            "dist": ds,
                            "policy": policy_name,
                            **kpis_to_dict(kpis),
                        }
                    )

    for seed in config.eval_seeds:
        ref_param = generate_random_reference(config, seed)
        for ds in config.dist_scales:
            for policy_name, policy_fn in [
                ("baseline", baseline_policy),
                ("cg_v1", cg_v1_policy),
                ("lcg_v2", lambda r, s, d, c: student_policy(r, s, d, c, model, action_grid)),
            ]:
                label = f"random_seed{seed}"
                trace = rollout("custom", ref_param, ds, config, policy_fn)
                kpis = compute_kpis(trace, config)
                eval_rows.append(
                    {
                        "test": label,
                        "dist": ds,
                        "policy": policy_name,
                        **kpis_to_dict(kpis),
                    }
                )

    return eval_rows


def topline_stats(eval_rows: List[Dict]) -> Dict[str, Dict[str, float]]:
    policies = {row["policy"] for row in eval_rows}
    topline = {policy: {"feasible_count": 0, "avg_distortion": float("nan")} for policy in policies}

    for policy in policies:
        rows = [
            row
            for row in eval_rows
            if row["dist"] == 0.25
            and row["test"].startswith("sine_")
            and row["policy"] == policy
            and row["sat_total"] < 0.2
        ]
        topline[policy]["feasible_count"] = len(rows)
        if rows:
            topline[policy]["avg_distortion"] = float(
                np.mean([row["mean_ref_diff"] for row in rows])
            )

    return topline


def write_summary(
    out_dir: Path,
    config: Phase5Config,
    train_metrics: Dict,
    dagger_stats: Dict,
    eval_rows: List[Dict],
) -> None:
    out_path = out_dir / "phase5_summary.md"
    topline = topline_stats(eval_rows)

    with open(out_path, "w") as f:
        f.write("# Phase 5 L-CG Production Summary\n\n")
        f.write("## Topline KPIs (dist=0.25 sine grid)\n")
        for policy in ["baseline", "cg_v1", "lcg_v2"]:
            stats = topline.get(policy, {"feasible_count": 0, "avg_distortion": float("nan")})
            f.write(
                f"- {policy}: sine feasible_count(sat<20%)={stats['feasible_count']} | "
                f"avg distortion under sat<20%={stats['avg_distortion']:.4f}\n"
            )
        f.write("\n")

        f.write("## Dataset\n")
        f.write(
            f"- Samples: {train_metrics['n_train'] + train_metrics['n_val'] + train_metrics['n_test']}\n"
        )
        f.write(
            f"- Train/Val/Test: {train_metrics['n_train']} / {train_metrics['n_val']} / {train_metrics['n_test']}\n"
        )
        f.write("- Features: [r, x, v, r_feasible_prev, dist_scale, sat_prev]\n")
        f.write("- Targets: discrete action index over (dr_max, A_scale) grid\n\n")

        f.write("## Teacher\n")
        f.write(f"- Horizon: {config.teacher_horizon} steps\n")
        f.write(f"- Action grid size: {len(config.dr_grid) * len(config.a_grid)}\n\n")

        f.write("## DAgger-lite\n")
        f.write(f"- Iterations: {dagger_stats['iterations']}\n")
        f.write(f"- DAgger samples: {dagger_stats['samples']}\n\n")

        f.write("## Training\n")
        f.write(f"- Train accuracy: {train_metrics['train_acc']:.3f}\n")
        f.write(f"- Val accuracy: {train_metrics['val_acc']:.3f}\n")
        f.write(f"- Test accuracy: {train_metrics['test_acc']:.3f}\n\n")

        f.write("## Evaluation KPIs\n")
        f.write(
            "| test | dist_scale | policy | sat_total | RMSE | recovery | mean|e| | max|e| | %time(|e|<eps) | mean|r-rf| | rms_rf/rms_r | %time dr_min | %time a_min |\n"
        )
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in eval_rows:
            f.write(
                "| {test} | {dist} | {policy} | {sat_total:.2%} | {rmse:.4f} | {recovery:.4f} | {mean_abs:.4f} | {max_abs:.4f} | {in_band:.2%} | {mean_ref_diff:.4f} | {rms_rf_over_r:.4f} | {pct_dr_min:.2%} | {pct_a_min:.2%} |\n".format(
                    **row
                )
            )

        f.write("\n## Notes\n")
        f.write("- Teacher performs horizon search over discrete (dr_max, A_scale) grid.\n")
        f.write("- Student imitates teacher with softmax classification (discrete imitation).\n")
        f.write("- Artifacts include run_id with config snapshot, dataset, model, and eval KPIs.\n")


def write_eval_csv(out_dir: Path, eval_rows: List[Dict]) -> None:
    out_path = out_dir / "eval_kpis.csv"
    columns = [
        "test",
        "dist",
        "policy",
        "sat_total",
        "rmse",
        "recovery",
        "mean_abs",
        "max_abs",
        "in_band",
        "mean_ref_diff",
        "rms_rf_over_r",
        "pct_dr_min",
        "pct_a_min",
    ]
    with open(out_path, "w") as f:
        f.write(",".join(columns) + "\n")
        for row in eval_rows:
            f.write(",".join(str(row[c]) for c in columns) + "\n")


def save_model(out_dir: Path, model, action_grid: ActionGrid) -> None:
    np.savez(
        out_dir / "lcg_classifier.npz",
        w=model.w,
        b=model.b,
        x_mean=model.x_mean,
        x_std=model.x_std,
        actions=np.array(action_grid.actions),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 production L-CG pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "phase5_default.json"),
        help="Path to config JSON",
    )
    parser.add_argument("--run-id", type=str, default="", help="Run id for artifacts")
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = args.run_id or default_run_id()

    out_dir = ARTIFACTS_DIR / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    save_config(config, out_dir / "config_snapshot.json")

    action_grid = ActionGrid.from_config(config)

    X, y = collect_dataset(config, action_grid)
    np.savez(out_dir / "teacher_dataset.npz", X=X, y=y)

    model, train_metrics = train_classifier(
        X,
        y,
        epochs=config.train_epochs,
        lr=config.train_lr,
        l2=config.train_l2,
    )

    X_dag = np.empty((0, X.shape[1]))
    y_dag = np.empty((0,), dtype=int)
    for _ in range(config.dagger_iterations):
        X_dag, y_dag = collect_dagger_data(config, action_grid, model)
        X = np.vstack([X, X_dag])
        y = np.concatenate([y, y_dag])
        model, train_metrics = train_classifier(
            X,
            y,
            epochs=config.dagger_epochs,
            lr=config.dagger_lr,
            l2=config.dagger_l2,
        )

    save_model(out_dir, model, action_grid)

    eval_rows = evaluate_policies(config, action_grid, model)
    write_eval_csv(out_dir, eval_rows)

    dagger_stats = {
        "iterations": config.dagger_iterations,
        "samples": len(X_dag) if config.dagger_iterations else 0,
    }
    write_summary(out_dir, config, train_metrics, dagger_stats, eval_rows)

    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(out_dir / "eval_kpis.json", "w") as f:
        json.dump(eval_rows, f, indent=2)

    print(f"Phase 5 L-CG run complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
