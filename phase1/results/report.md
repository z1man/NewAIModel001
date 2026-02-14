# Phase 1 DOE Report

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3240 | 0 | 0.1173 | 50.5184 | 17.2243 | 2.8710 |
| B1 | 3240 | 0 | 0.1175 | 50.7533 | 17.2998 | 2.8710 |
| B2 | 3240 | 0 | 0.1173 | 50.5184 | 17.2243 | 2.8710 |
| B3 | 3240 | 0 | 0.1173 | 50.4836 | 17.5337 | 2.8710 |

## Failure Tags

**B0**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 360, 'NO_GAIN': 2160, 'CHATTER': 720, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: 0.0002
90% CI: [0.00018671428719269816, 0.0002021441967625234]


## Adaptation Test (bias +/−/0, noise=mid, delay=20ms)

| Baseline | RMSE | Recovery Time | Steady State Error | Effort | Smoothness | ResMean | ResP50 | ResP90 | Corr(res,dist) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 0.1244 | 0.0813 | 0.0047 | 50.1602 | 8.5420 | 0.0000 | 0.0000 | 0.0000 | nan |
| B1 | 0.1244 | 0.0813 | 0.0047 | 50.1553 | 8.5426 | 0.0009 | 0.0000 | 0.0004 | -0.1125 |
| B2 | 0.1244 | 0.0813 | 0.0047 | 50.1602 | 8.5420 | 0.0000 | 0.0000 | 0.0000 | nan |
| B3 | 0.1244 | 0.0853 | 0.0050 | 50.2153 | 8.7119 | 0.0137 | 0.0022 | 0.0147 | -0.0013 |

**B1 vs B0 (95% CI)**

- rmse: mean diff=0.0000, CI95=[1.5457566477646012e-06, 2.219461423213359e-06]
- recovery_time_after_step: mean diff=0.0000, CI95=[0.0, 0.0]
- steady_state_error: mean diff=-0.0000, CI95=[-4.1641082828437066e-07, -1.8965313174253152e-08]
- effort: mean diff=-0.0050, CI95=[-0.005329412166238967, -0.004553746429371586]
- smoothness: mean diff=0.0006, CI95=[0.0005540406505192052, 0.000724608745113553]

Learning curve: phase1/results/learning_curve.png


## Gate 1.1 Status

PASS


## Stress Test (high delay+noise+friction, long horizon)

B0: diverged=False, rmse=0.0613, effort=165.04, smooth=181.22, recovery=3.73, tag=CHATTER
B1: diverged=False, rmse=0.0613, effort=165.12, smooth=181.68, recovery=3.73, tag=CHATTER
