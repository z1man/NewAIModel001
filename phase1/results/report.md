# Phase 1 DOE Report

**Recovery metric update:** recovery_time_after_step uses sustained recovery: after a step, the earliest time where |e|<eps holds continuously for 200ms.

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B1 | 9720 | 0 | 0.1177 | 54.4079 | 17.3241 | 2.8710 |
| B2 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B3 | 9720 | 0 | 0.1198 | 48.9538 | 18.7860 | 2.8710 |

## Failure Tags

**B0**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 2649, 'NO_GAIN': 1711, 'CHATTER': 5360, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 2783, 'NO_GAIN': 0, 'CHATTER': 6937, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: -0.0020
90% CI: [-0.00206478350626457, -0.00199051898194455]


## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)

| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) | Corr(b_hat,b_true) | c_relerr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1472 | 0.2546 | 0.0078 | 227.3079 | 9.8430 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B1 | 0.1312 | 0.4462 | 0.0169 | 220.2147 | 9.9009 | 0.9730 | 0.8090 | 2.0040 | 0.6910 | 0.7509 | 3.9942 |
| B2 | 0.1472 | 0.2546 | 0.0078 | 227.3079 | 9.8430 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B3 | 0.1472 | 0.2581 | 0.0083 | 228.0323 | 11.1407 | 0.1524 | 0.1321 | 0.3101 | -0.0031 | nan | nan |

**B1 vs B0 (95% CI)**

- rmse: mean diff=-0.0160, CI95=[-0.0173433428000041, -0.014421680973597032]
- recovery_time_after_step: mean diff=0.1916, CI95=[0.16299999999999987, 0.2207499999999998]
- steady_state_error: mean diff=0.0091, CI95=[0.008363648993924505, 0.009745720369880563]
- effort: mean diff=-7.0932, CI95=[-7.860383787559685, -6.397463338402897]
- smoothness: mean diff=0.0578, CI95=[0.05311703018816565, 0.06302037851627089]

Learning curve: phase1/results/learning_curve.png
Scatter: phase1/results/scatter_res_vs_dist.png
Spectrum: phase1/results/residual_spectrum.png
Step response: phase1/results/adaptation_step_response.png
b_hat trace: phase1/results/b_hat_trace.png
c_hat trace: phase1/results/c_hat_trace.png
ID segment: phase1/results/identification_segment_plot.png
RLS residual: phase1/results/rls_residual_curve.png
y vs pred: phase1/results/scatter_y_vs_pred.png

Rep HF ratio (>10Hz): 0.0001
Rep R2: 0.5048


## Gate 1.4 Status

FAIL

Recovery improve: -75.27%
Effort improve: 3.12%
Steady-state improve: -116.90%
Smoothness ok: True
Corr(b_hat,b_true) ok: False
c_relerr ok: False
R2 ok: True
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
B1: diverged=False, rmse=0.0617, effort=170.77, smooth=181.98, recovery=3.73, tag=CHATTER

## Aggressive Stress Test (30ms delay, noise mid/high, drag 6x, bias=3, impulse, long)

B0: {'diverged': False, 'rmse': 0.14516851646267434, 'effort': 733.5304327153644, 'smoothness': 181.76609492760255}
B1: {'diverged': False, 'rmse': 0.1448698358291126, 'effort': 733.0935113502176, 'smoothness': 181.76578950794544}
