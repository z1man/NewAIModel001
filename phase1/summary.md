# Phase 1 Summary – Cortex vs Cerebellum (工程化视角)

**Key idea from the shared notes**
- Today’s LLMs ≈ **人工大脑皮层**: language, reasoning, planning.
- What’s missing for embodied agents is **人工小脑**: fast, stable, low‑latency control.
- The cerebellum is not just “reward learning.” It is a **predictive, local error‑driven controller**
  that stabilizes motion, smooths trajectories, and compensates model errors.

## Why current RL/LLM aren’t enough
- LLMs solve **symbolic/statistical** problems, not continuous control.
- RL often optimizes rewards **offline** and is unstable for real‑time control.
- Cerebellar learning is **local, event‑driven**, not global backprop across long horizons.

## Engineering interpretation
A practical “artificial cerebellum” should:
1. Run at high frequency (sub‑10ms loop).
2. Use a **physics model** for baseline control (forward model).
3. Learn residual errors (friction, latency, actuator bias) with a lightweight model.
4. Be **stateful** and **body‑specific** (not one model fits all).

## Minimum Viable Cerebellum (MVC)
Closed‑loop architecture:
```
Sensors -> State Estimation -> Baseline Controller (physics)
                                + Residual Learner -> Control Output
```

## Phase‑1 Experiment
- Simulate a 1D mass system with **unknown friction**.
- Baseline controller assumes no friction.
- Residual learner observes model error and learns a correction.
- Result: lower tracking error and smoother control.

## Next Steps (Phase 2)
- Replace linear residual with small NN.
- Add latency / actuator saturation.
- Move to 2D system or cart‑pole.
- Integrate with a planner (LLM provides target trajectory).
