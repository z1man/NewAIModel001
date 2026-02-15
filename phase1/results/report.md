# Phase 1 DOE Report

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B1 | 9720 | 0 | 0.1168 | 58.5118 | 17.5624 | 2.8710 |
| B2 | 9720 | 0 | 0.1197 | 48.2307 | 17.1933 | 2.8710 |
| B3 | 9720 | 0 | 0.1198 | 49.0349 | 19.0588 | 2.8710 |

## Failure Tags

**B0**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 2885, 'NO_GAIN': 1005, 'CHATTER': 5830, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 3240, 'NO_GAIN': 0, 'CHATTER': 6480, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 1176, 'NO_GAIN': 0, 'CHATTER': 8544, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: -0.0029
90% CI: [-0.0029669550215111287, -0.002891000536050374]


## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)

| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) | Corr(b_hat,b_true) | c_relerr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1472 | 0.2231 | 0.0078 | 227.3079 | 9.8430 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B1 | 0.1200 | 0.0000 | 0.0168 | 232.8503 | 10.2838 | 1.3103 | 1.7901 | 2.0310 | 0.7409 | 0.6254 | 1.7478 |
| B2 | 0.1472 | 0.2231 | 0.0078 | 227.3079 | 9.8430 | 0.0000 | 0.0000 | 0.0000 | nan | nan | nan |
| B3 | 0.1472 | 0.2252 | 0.0082 | 228.0614 | 11.3716 | 0.1594 | 0.1359 | 0.3278 | -0.0027 | nan | nan |

**B1 vs B0 (95% CI)**

- rmse: mean diff=-0.0272, CI95=[-0.02946293566972534, -0.024624495378019487]
- recovery_time_after_step: mean diff=-0.2231, CI95=[-0.23591666666666677, -0.20713888888888915]
- steady_state_error: mean diff=0.0091, CI95=[0.008347664321534881, 0.009729134375752777]
- effort: mean diff=5.5424, CI95=[4.554070413198045, 6.497625439930241]
- smoothness: mean diff=0.4407, CI95=[0.42678415818424636, 0.4536241607528047]

Learning curve: phase1/results/learning_curve.png
Scatter: phase1/results/scatter_res_vs_dist.png
Spectrum: phase1/results/residual_spectrum.png
Step response: phase1/results/adaptation_step_response.png
b_hat trace: phase1/results/b_hat_trace.png
c_hat trace: phase1/results/c_hat_trace.png
ID segment: phase1/results/identification_segment_plot.png
RLS residual: phase1/results/rls_residual_curve.png
y vs pred: phase1/results/scatter_y_vs_pred.png

Rep HF ratio (>10Hz): 0.0007
Rep R2: 0.7464


## Gate 1.4 Status

FAIL

Recovery improve: 100.00%
Effort improve: -2.44%
Steady-state improve: -116.68%
Smoothness ok: False
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
B1: diverged=False, rmse=0.0615, effort=174.00, smooth=182.22, recovery=3.73, tag=CHATTER

## Aggressive Stress Test (30ms delay, noise mid/high, drag 6x, bias=3, impulse, long)

B0: {'diverged': False, 'rmse': 0.14516851646267434, 'effort': 733.5304327153644, 'smoothness': 181.76609492760255}
B1: {'diverged': False, 'rmse': 0.1448698358291126, 'effort': 733.0935113502176, 'smoothness': 181.76578950794544}
