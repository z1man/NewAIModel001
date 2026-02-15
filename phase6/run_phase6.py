#!/usr/bin/env python3
"""Phase 6 robust L-CG pipeline runner."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import Phase6Config, load_config, save_config
from .env_runner import (
    EnvParams,
    initial_state,
    make_env_params,
    reference_signal,
    rollout,
    sample_env_params,
    step_dynamics,
)
from .metrics import compute_kpis, kpis_to_dict
from .policy import (
    adaptive_student_policy,
    baseline_policy,
    cg_v1_policy,
    safe_policy,
    student_policy,
    train_classifier,
)
from .teacher import ActionGrid, horizon_teacher_action

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def default_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def generate_random_reference(config: Phase6Config, seed: int) -> Dict[str, np.ndarray | float]:
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
    config: Phase6Config,
    params: EnvParams,
    action_grid: ActionGrid,
) -> Tuple[np.ndarray, np.ndarray]:
    state = initial_state(config, params)
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
            params,
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
    config: Phase6Config,
    params: EnvParams,
    action_grid: ActionGrid,
    model,
) -> Tuple[np.ndarray, np.ndarray]:
    state = initial_state(config, params)
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
            params,
            action_grid,
            t,
            ref_type,
            ref_param,
        )
        xs.append(feat)
        ys.append(action_idx)
        r_use, state = student_policy(r, state, dist_scale, config, params, model, action_grid)
        state, _, _ = step_dynamics(state, r_use, dist_scale, config, params, t)
    return np.vstack(xs), np.array(ys)


def collect_dataset(config: Phase6Config, action_grid: ActionGrid) -> Tuple[np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []

    for seed in config.train_seeds:
        rng = np.random.default_rng(seed)
        for ds in config.dist_scales:
            for ref_type, ref_param in [
                ("step", 0.75),
                ("ramp", 0.5 * 0.75),
                ("ramp", 0.2 * 0.75),
            ]:
                params = sample_env_params(config, rng)
                xs, ys = teacher_episode(ref_type, ref_param, ds, config, params, action_grid)
                xs_all.append(xs)
                ys_all.append(ys)

        for ds in config.dist_scales:
            for amp in config.eval_sine_amps:
                for freq in config.eval_sine_freqs:
                    params = sample_env_params(config, rng)
                    xs, ys = teacher_episode("sine", (amp, freq), ds, config, params, action_grid)
                    xs_all.append(xs)
                    ys_all.append(ys)

        for idx in range(config.random_ref_count):
            seq = generate_random_reference(config, seed + idx * 100)
            for ds in config.dist_scales:
                params = sample_env_params(config, rng)
                xs, ys = teacher_episode("custom", seq, ds, config, params, action_grid)
                xs_all.append(xs)
                ys_all.append(ys)

    X = np.vstack(xs_all)
    y = np.concatenate(ys_all)
    return X, y


def collect_dagger_data(
    config: Phase6Config, action_grid: ActionGrid, model
) -> Tuple[np.ndarray, np.ndarray]:
    xs_all = []
    ys_all = []
    for seed in config.train_seeds:
        rng = np.random.default_rng(seed + 50)
        for ds in config.dist_scales:
            for ref_type, ref_param in [
                ("step", 0.75),
                ("ramp", 0.5 * 0.75),
                ("ramp", 0.2 * 0.75),
            ]:
                params = sample_env_params(config, rng)
                xs, ys = student_episode(
                    ref_type, ref_param, ds, config, params, action_grid, model
                )
                xs_all.append(xs)
                ys_all.append(ys)

        for ds in config.dist_scales:
            for amp in config.eval_sine_amps:
                for freq in config.eval_sine_freqs:
                    params = sample_env_params(config, rng)
                    xs, ys = student_episode(
                        "sine", (amp, freq), ds, config, params, action_grid, model
                    )
                    xs_all.append(xs)
                    ys_all.append(ys)

        for idx in range(config.random_ref_count):
            seq = generate_random_reference(config, seed + idx * 100)
            for ds in config.dist_scales:
                params = sample_env_params(config, rng)
                xs, ys = student_episode(
                    "custom", seq, ds, config, params, action_grid, model
                )
                xs_all.append(xs)
                ys_all.append(ys)

    return np.vstack(xs_all), np.concatenate(ys_all)


def build_benchmark_suite(config: Phase6Config) -> List[Dict[str, EnvParams]]:
    nominal = make_env_params(config)
    suite = [
        {"name": "in_dist", "params": nominal},
        {
            "name": "ood_mass_high",
            "params": EnvParams(
                mass=nominal.mass * 1.4,
                friction=nominal.friction,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult,
                bias_scale=nominal.bias_scale,
                delay_steps=nominal.delay_steps,
            ),
        },
        {
            "name": "ood_friction_high",
            "params": EnvParams(
                mass=nominal.mass,
                friction=nominal.friction * 1.8,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult,
                bias_scale=nominal.bias_scale,
                delay_steps=nominal.delay_steps,
            ),
        },
        {
            "name": "ood_drag_high",
            "params": EnvParams(
                mass=nominal.mass,
                friction=nominal.friction,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult * 1.7,
                bias_scale=nominal.bias_scale,
                delay_steps=nominal.delay_steps,
            ),
        },
        {
            "name": "ood_delay",
            "params": EnvParams(
                mass=nominal.mass,
                friction=nominal.friction,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult,
                bias_scale=nominal.bias_scale,
                delay_steps=nominal.delay_steps + 2,
            ),
        },
        {
            "name": "ood_bias",
            "params": EnvParams(
                mass=nominal.mass,
                friction=nominal.friction,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult,
                bias_scale=nominal.bias_scale * 1.5,
                delay_steps=nominal.delay_steps,
            ),
        },
        {
            "name": "ood_combo",
            "params": EnvParams(
                mass=nominal.mass * 1.25,
                friction=nominal.friction * 1.4,
                quad_drag_base=nominal.quad_drag_base,
                drag_mult=nominal.drag_mult * 1.4,
                bias_scale=nominal.bias_scale * 1.4,
                delay_steps=nominal.delay_steps + 1,
            ),
        },
    ]
    return suite


def evaluate_benchmarks(
    config: Phase6Config, action_grid: ActionGrid, model
) -> List[Dict[str, float]]:
    eval_rows = []
    benchmarks = build_benchmark_suite(config)

    policies = [
        ("baseline", baseline_policy),
        ("cg_v1", cg_v1_policy),
        ("lcg_v2", lambda r, s, d, c, p: student_policy(r, s, d, c, p, model, action_grid)),
        (
            "lcg_v2_safe",
            lambda r, s, d, c, p: safe_policy(
                lambda r2, s2, d2, c2, p2: student_policy(
                    r2, s2, d2, c2, p2, model, action_grid
                ),
                r,
                s,
                d,
                c,
                p,
            ),
        ),
    ]

    for bench in benchmarks:
        params = bench["params"]
        for ds in config.dist_scales:
            for policy_name, policy_fn in policies:
                for ref_type, ref_param, label in [
                    ("step", 0.75, "step"),
                    ("ramp", 0.5 * 0.75, "ramp_fast"),
                    ("ramp", 0.2 * 0.75, "ramp_med"),
                ]:
                    trace = rollout(ref_type, ref_param, ds, config, params, policy_fn)
                    kpis = compute_kpis(trace, config)
                    eval_rows.append(
                        {
                            "benchmark": bench["name"],
                            "test": label,
                            "dist": ds,
                            "policy": policy_name,
                            **kpis_to_dict(kpis),
                        }
                    )

        for ds in config.dist_scales:
            for amp in config.eval_sine_amps:
                for freq in config.eval_sine_freqs:
                    for policy_name, policy_fn in policies:
                        label = f"sine_A{amp}_f{freq}"
                        trace = rollout("sine", (amp, freq), ds, config, params, policy_fn)
                        kpis = compute_kpis(trace, config)
                        eval_rows.append(
                            {
                                "benchmark": bench["name"],
                                "test": label,
                                "dist": ds,
                                "policy": policy_name,
                                **kpis_to_dict(kpis),
                            }
                        )

        for seed in config.eval_seeds:
            ref_param = generate_random_reference(config, seed)
            for ds in config.dist_scales:
                for policy_name, policy_fn in policies:
                    label = f"random_seed{seed}"
                    trace = rollout("custom", ref_param, ds, config, params, policy_fn)
                    kpis = compute_kpis(trace, config)
                    eval_rows.append(
                        {
                            "benchmark": bench["name"],
                            "test": label,
                            "dist": ds,
                            "policy": policy_name,
                            **kpis_to_dict(kpis),
                        }
                    )

    return eval_rows


def adaptation_demo(
    config: Phase6Config, action_grid: ActionGrid, model
) -> List[Dict[str, float]]:
    ood_params = EnvParams(
        mass=config.mass * 1.3,
        friction=config.friction * 1.5,
        quad_drag_base=config.quad_drag_base,
        drag_mult=config.drag_mult * 1.5,
        bias_scale=config.bias_scale * 1.4,
        delay_steps=config.delay_steps + 1,
    )
    dist_scale = config.dist_scales[1] if len(config.dist_scales) > 1 else config.dist_scales[0]
    ref_param = (0.6, 0.2)
    rows = []

    policies = [
        ("lcg_v2", lambda r, s, d, c, p: student_policy(r, s, d, c, p, model, action_grid)),
        (
            "lcg_v2_adapt",
            lambda r, s, d, c, p: adaptive_student_policy(
                r, s, d, c, p, model, action_grid
            ),
        ),
    ]

    for name, fn in policies:
        trace = rollout("sine", ref_param, dist_scale, config, ood_params, fn)
        kpis = compute_kpis(trace, config)
        rows.append({"policy": name, **kpis_to_dict(kpis)})

    return rows


def write_eval_csv(out_dir: Path, eval_rows: List[Dict]) -> None:
    out_path = out_dir / "ood_eval_kpis.csv"
    columns = [
        "benchmark",
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


def write_summary(
    out_dir: Path,
    config: Phase6Config,
    train_metrics: Dict,
    dagger_stats: Dict,
    eval_rows: List[Dict],
    adaptation_rows: List[Dict],
) -> None:
    out_path = out_dir / "phase6_summary.md"
    with open(out_path, "w") as f:
        f.write("# Phase 6 Robust L-CG Summary\n\n")
        f.write("## Dataset\n")
        f.write(
            f"- Samples: {train_metrics['n_train'] + train_metrics['n_val'] + train_metrics['n_test']}\n"
        )
        f.write(
            f"- Train/Val/Test: {train_metrics['n_train']} / {train_metrics['n_val']} / {train_metrics['n_test']}\n"
        )
        f.write("- Features: [r, x, v, r_feasible_prev, dist_scale, sat_prev]\n\n")

        f.write("## Domain Randomization\n")
        f.write(f"- Mass range: {config.domain_mass_range}\n")
        f.write(f"- Friction range: {config.domain_friction_range}\n")
        f.write(f"- Drag mult range: {config.domain_drag_mult_range}\n")
        f.write(f"- Bias scale range: {config.domain_bias_scale_range}\n")
        f.write(f"- Delay steps range: {config.domain_delay_steps_range}\n\n")

        f.write("## DAgger-lite\n")
        f.write(f"- Iterations: {dagger_stats['iterations']}\n")
        f.write(f"- DAgger samples: {dagger_stats['samples']}\n\n")

        f.write("## Training\n")
        f.write(f"- Train accuracy: {train_metrics['train_acc']:.3f}\n")
        f.write(f"- Val accuracy: {train_metrics['val_acc']:.3f}\n")
        f.write(f"- Test accuracy: {train_metrics['test_acc']:.3f}\n\n")

        f.write("## OOD Benchmark Suite (sample)\n")
        sample_rows = [row for row in eval_rows if row["test"] == "sine_A0.5_f0.2"]
        for row in sample_rows[:12]:
            f.write(
                f"- {row['benchmark']} | {row['policy']} | dist={row['dist']} | "
                f"sat={row['sat_total']:.2%} | rmse={row['rmse']:.4f}\n"
            )
        f.write("\n")

        f.write("## Adaptation Demo (OOD combo sine)\n")
        for row in adaptation_rows:
            f.write(
                f"- {row['policy']}: sat={row['sat_total']:.2%} | rmse={row['rmse']:.4f} | "
                f"mean|e|={row['mean_abs']:.4f}\n"
            )

        f.write("\n## Notes\n")
        f.write("- Domain randomization samples mass/friction/drag/bias/delay per episode.\n")
        f.write("- Safety layer scales dr_max/a_scale when saturation rate spikes.\n")
        f.write("- Adaptation demo uses an online a_scale adjustment wrapper.\n")


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
    parser = argparse.ArgumentParser(description="Phase 6 robust L-CG pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(BASE_DIR / "configs" / "phase6_default.json"),
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

    eval_rows = evaluate_benchmarks(config, action_grid, model)
    write_eval_csv(out_dir, eval_rows)

    adaptation_rows = adaptation_demo(config, action_grid, model)

    dagger_stats = {
        "iterations": config.dagger_iterations,
        "samples": len(X_dag) if config.dagger_iterations else 0,
    }
    write_summary(out_dir, config, train_metrics, dagger_stats, eval_rows, adaptation_rows)

    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump(train_metrics, f, indent=2)

    with open(out_dir / "ood_eval_kpis.json", "w") as f:
        json.dump(eval_rows, f, indent=2)

    with open(out_dir / "adaptation_demo.json", "w") as f:
        json.dump(adaptation_rows, f, indent=2)

    print(f"Phase 6 robust run complete. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
