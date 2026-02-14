# Phase 1 DOE Report

## Summary (means)

| Baseline | Count | Diverged | RMSE | Effort | Smoothness | Recovery Time |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 3240 | 0 | 0.1162 | 53.2273 | 17.2507 | 2.8710 |
| B1 | 3240 | 0 | 0.1163 | 53.3604 | 17.2904 | 2.8710 |
| B2 | 3240 | 0 | 0.1162 | 53.2273 | 17.2507 | 2.8710 |
| B3 | 3240 | 0 | 0.1162 | 53.1830 | 17.5685 | 2.8710 |

## Failure Tags

**B0**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B1**: {'OK': 360, 'NO_GAIN': 2135, 'CHATTER': 745, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B2**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}

**B3**: {'OK': 1080, 'NO_GAIN': 0, 'CHATTER': 2160, 'DRIFT': 0, 'DIVERGENCE': 0, 'DELAY_SENSITIVE': 0, 'NOISE_SENSITIVE': 0}


## B1 vs B0 RMSE (bootstrap CI)

Mean diff: 0.0001
90% CI: [8.434856207809479e-05, 9.375355150120519e-05]


## Stress Test (high delay+noise+friction, long horizon)

B0: diverged=False, rmse=0.0613, effort=168.32, smooth=181.23, recovery=3.73, tag=CHATTER
B1: diverged=False, rmse=0.0613, effort=168.35, smooth=181.57, recovery=3.73, tag=CHATTER
