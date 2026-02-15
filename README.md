# NewAIModel001

## Phase 5: Production L-CG Pipeline

Phase 5 introduces a modular, production-ready L-CG pipeline with discrete imitation, horizon teacher, action grids, DAgger-lite finetuning, and KPI evaluation.

### Run (one command)

```bash
python -m phase5.run_phase5 --config phase5/configs/phase5_default.json
```

Artifacts are written to `phase5/artifacts/run_<run_id>/` and include:
- `config_snapshot.json`
- `teacher_dataset.npz`
- `lcg_classifier.npz`
- `train_metrics.json`
- `eval_kpis.json` / `eval_kpis.csv`
- `phase5_summary.md`

### Config

Edit `phase5/configs/phase5_default.json` to adjust the action grid, horizon length, training hyperparameters, or evaluation grid.

## Phase 6: Robust L-CG + OOD Benchmarks

Phase 6 adds domain-randomized training, a safety layer, an OOD benchmark suite runner, and an adaptation demo that adjusts tracking scale online.

### Run (one command)

```bash
python -m phase6.run_phase6 --config phase6/configs/phase6_default.json
```

Artifacts are written to `phase6/artifacts/run_<run_id>/` and include:
- `config_snapshot.json`
- `teacher_dataset.npz`
- `lcg_classifier.npz`
- `train_metrics.json`
- `ood_eval_kpis.json` / `ood_eval_kpis.csv`
- `adaptation_demo.json`
- `phase6_summary.md`

### Config

Edit `phase6/configs/phase6_default.json` to adjust the domain randomization ranges, safety thresholds, adaptation gains, and evaluation grid.
