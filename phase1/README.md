# Phase 1 – Artificial Cerebellum (Minimum Viable Control Loop)

This phase is a minimal, runnable experiment inspired by the “人工大脑与小脑” notes.
We treat LLMs as **cortex** (planning, language, abstraction) and build a tiny **cerebellum**
module for real‑time control: a physics model + residual learner in a closed loop.

## Goals
- Build a **closed‑loop controller** that is stable and low‑latency.
- Use a **physics model** for baseline control.
- Add a **residual learner** to compensate model error (friction/disturbance).
- Show measurable improvement vs baseline.

## Files
- `summary.md` – distilled notes + architecture outline
- `experiment.py` – 1D control loop with residual learning

## Run
```bash
python3 phase1/experiment.py
```

## Expected output
- Prints baseline tracking error
- Prints residual‑learned tracking error (should be lower)
