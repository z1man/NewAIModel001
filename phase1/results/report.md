# Phase 1 DOE Report

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B1 | 9720 | 0 | 0.1198 | 48.1322 | 17.1931 | 2.8710 |
| B2 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B3 | 9720 | 0 | 0.1198 | 49.0349 | 19.0588 | 2.8710 |

## Failure Tags

**B0**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 479, 'NO_GAIN': 8250, 'CHATTER': 991, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 1176, 'NO_GAIN': 0, 'CHATTER': 8544, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: 0.0000
90% CI: [4.573054085296982e-05, 4.75269128883802e-05]


## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)

| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1482 | 0.2231 | 0.0078 | 93.1986 | 8.5495 | 0.0000 | 0.0000 | 0.0000 | nan |
| B1 | 0.1482 | 0.2282 | 0.0080 | 93.0932 | 8.5494 | 0.0700 | 0.0575 | 0.1375 | 0.0103 |
| B2 | 0.1482 | 0.2231 | 0.0078 | 93.1986 | 8.5495 | 0.0000 | 0.0000 | 0.0000 | nan |
| B3 | 0.1482 | 0.2251 | 0.0082 | 93.9970 | 10.0648 | 0.1594 | 0.1359 | 0.3278 | -0.0027 |

**B1 vs B0 (95% CI)**

- rmse: mean diff=-0.0000, CI95=[-2.4070678257544992e-05, -2.0859783631954158e-05]
- recovery_time_after_step: mean diff=0.0051, CI95=[0.004361111111111115, 0.005694444444444447]
- steady_state_error: mean diff=0.0003, CI95=[0.0001987191551005065, 0.00030445082034328717]
- effort: mean diff=-0.1054, CI95=[-0.11096305762532555, -0.09988341916852943]
- smoothness: mean diff=-0.0001, CI95=[-9.290556474808772e-05, -6.989379124159436e-05]

Learning curve: phase1/results/learning_curve.png
Scatter: phase1/results/scatter_res_vs_dist.png
Spectrum: phase1/results/residual_spectrum.png
Step response: phase1/results/adaptation_step_response.png

Rep HF ratio (>10Hz): 0.0000


## Gate 1.2 Status

FAIL

Recovery improve: -2.27%
Effort improve: 0.11%
Steady-state improve: -3.27%
Smoothness ok: True
Corr ok: False
HF ratio ok: True

### Postmortem (why B1≈B0)

- NO_SIGNAL: residual output magnitude is tiny; correlation with true disturbance is weak.
- CHATTER: if HF ratio is high, residual is not low‑frequency.
- NO_GAIN: most scenarios show no RMSE/recovery improvement over B0.

**Structural fix proposals (not just LR tuning):**
- Add bias estimator state (integral/observer) tied to disturbance channels with projection.
- Use a richer disturbance model (piecewise constant + quadratic drag term).
- Use low‑pass filtering on s and residual update with explicit bandwidth limit.


## Stress Test (high delay+noise+friction, long horizon)

B0: diverged=False, rmse=0.0613, effort=165.04, smooth=181.22, recovery=3.73, tag=CHATTER
B1: diverged=False, rmse=0.0614, effort=164.74, smooth=181.22, recovery=3.73, tag=CHATTER

## Aggressive Stress Test (30ms delay, noise mid/high, drag 6x, bias=3, impulse, long)

B0: {'diverged': False, 'rmse': 0.14516851646267434, 'effort': 733.5304327153644, 'smoothness': 181.76609492760255}
B1: {'diverged': False, 'rmse': 0.1448698358291126, 'effort': 733.0935113502176, 'smoothness': 181.76578950794544}
