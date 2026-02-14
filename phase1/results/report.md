# Phase 1 DOE Report

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B1 | 9720 | 0 | 0.1198 | 48.0215 | 17.1930 | 2.8710 |
| B2 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B3 | 9720 | 0 | 0.1198 | 49.0349 | 19.0588 | 2.8710 |

## Failure Tags

**B0**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 340, 'NO_GAIN': 8644, 'CHATTER': 736, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 1176, 'NO_GAIN': 0, 'CHATTER': 8544, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: 0.0001
90% CI: [0.00011275382997791995, 0.00011686071076210388]


## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)

| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) | Corr(b_hat,b_true) | c_relerr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1482 | 0.2231 | 0.0078 | 93.1986 | 8.5495 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B1 | 0.1481 | 0.2277 | 0.0100 | 92.9810 | 8.5493 | 0.1578 | 0.1289 | 0.2877 | 0.0865 | 0.1697 | 1.0000 |
| B2 | 0.1482 | 0.2231 | 0.0078 | 93.1986 | 8.5495 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B3 | 0.1482 | 0.2251 | 0.0082 | 93.9970 | 10.0648 | 0.1594 | 0.1359 | 0.3278 | -0.0027 | nan | nan |

**B1 vs B0 (95% CI)**

- rmse: mean diff=-0.0001, CI95=[-0.00010167820514093969, -8.111185542896608e-05]
- recovery_time_after_step: mean diff=0.0046, CI95=[0.0033611111111111185, 0.0058888888888888975]
- steady_state_error: mean diff=0.0022, CI95=[0.0020200980702043594, 0.0023541183167294297]
- effort: mean diff=-0.2176, CI95=[-0.22872868802163898, -0.20653552061007924]
- smoothness: mean diff=-0.0001, CI95=[-0.0001515028910600316, -0.00010296999094659359]

Learning curve: phase1/results/learning_curve.png
Scatter: phase1/results/scatter_res_vs_dist.png
Spectrum: phase1/results/residual_spectrum.png
Step response: phase1/results/adaptation_step_response.png
b_hat trace: phase1/results/b_hat_trace.png
c_hat trace: phase1/results/c_hat_trace.png

Rep HF ratio (>10Hz): 0.0000


## Gate 1.3 Status

FAIL

Recovery improve: -2.04%
Effort improve: 0.23%
Steady-state improve: -28.21%
Smoothness ok: True
Corr(b_hat,b_true) ok: False
c_relerr ok: False
HF ratio ok: True

### Postmortem (why B1≈B0)

- NO_SIGNAL: bias/drag estimators not tracking; correlation or relerr failing.
- CHATTER: if HF ratio is high, residual is not low‑frequency.
- NO_GAIN: most scenarios show no RMSE/recovery improvement over B0.

**Structural fix proposals (not just LR tuning):**
- Add separate observer dynamics for bias and drag with leakage/forgetting.
- Use filtered s (low‑pass) and projection in parameter space.
- Add model‑based feedforward for drag term and learn residual around it.


## Stress Test (high delay+noise+friction, long horizon)

B0: diverged=False, rmse=0.0613, effort=165.04, smooth=181.22, recovery=3.73, tag=CHATTER
B1: diverged=False, rmse=0.0618, effort=164.41, smooth=181.22, recovery=3.73, tag=CHATTER

## Aggressive Stress Test (30ms delay, noise mid/high, drag 6x, bias=3, impulse, long)

B0: {'diverged': False, 'rmse': 0.14516851646267434, 'effort': 733.5304327153644, 'smoothness': 181.76609492760255}
B1: {'diverged': False, 'rmse': 0.1448698358291126, 'effort': 733.0935113502176, 'smoothness': 181.76578950794544}
