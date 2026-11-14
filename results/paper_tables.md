# Conference tables

Generated from `summary.json` by `paper_assets.py`. Values are mean ± 95% CI across seeds unless the column says otherwise.

## Table 1 — Regime comparison

| Regime | Accuracy | ROC AUC | ECE raw | ECE federated binning | EO gap before → after | Signed top-1 flip | Mean margin R | Mean P(R>0) | Robust at 95% |
|---|---|---|---|---|---|---|---|---|---|
| age | 0.762 ± 0.044 | 0.837 ± 0.032 | 0.068 ± 0.020 | 0.060 ± 0.025 | 0.413 ± 0.241 → 0.307 ± 0.114 | 0.226 ± 0.050 | +0.337 ± 0.018 | 0.775 ± 0.043 | 0.690 ± 0.051 |
| age (pooled) | 0.778 ± 0.029 | 0.842 ± 0.030 | 0.064 ± 0.010 | — | — | — | — | — | — |
| iid | 0.768 ± 0.051 | 0.838 ± 0.028 | 0.091 ± 0.013 | 0.056 ± 0.016 | 0.420 ± 0.212 → 0.358 ± 0.086 | 0.049 ± 0.023 | +0.332 ± 0.052 | 0.925 ± 0.013 | 0.784 ± 0.021 |
| iid (pooled) | 0.778 ± 0.029 | 0.843 ± 0.030 | 0.062 ± 0.011 | — | — | — | — | — | — |

## Table 2 — Baseline-robustness diagnostic

| Regime | Flip rate (95% CI across seeds) | Mean margin R (95% CI) | P(R>0) | Robust at 95% | Fragile at 50% | Split-half flip Δ | Split-half margin Δ |
|---|---|---|---|---|---|---|---|
| age | 0.226 ± 0.050 | +0.337 ± 0.018 | 0.775 | 0.690 | 0.227 | 0.0544 | 0.0642 |
| iid | 0.049 ± 0.023 | +0.332 ± 0.052 | 0.925 | 0.784 | 0.057 | 0.0274 | 0.0515 |

## Table 3 — Aggregation weighting ablation

| Partition | Weighting | Client weights | Accuracy | ECE | EO gap after | Mean margin R | P(R>0) |
|---|---|---|---|---|---|---|---|
| age | sample | 0.527 / 0.296 / 0.176 | 0.762 ± 0.044 | 0.068 ± 0.020 | 0.307 ± 0.114 | +0.337 ± 0.018 | 0.775 ± 0.043 |
| age | uniform | 0.333 / 0.333 / 0.333 | 0.774 ± 0.040 | 0.082 ± 0.019 | 0.317 ± 0.140 | +0.323 ± 0.028 | 0.778 ± 0.034 |
| age | positive | 0.327 / 0.428 / 0.245 | 0.774 ± 0.044 | 0.092 ± 0.016 | 0.310 ± 0.156 | +0.333 ± 0.033 | 0.777 ± 0.027 |
| iid | sample | 0.333 / 0.333 / 0.333 | 0.768 ± 0.051 | 0.091 ± 0.013 | 0.358 ± 0.086 | +0.332 ± 0.052 | 0.925 ± 0.013 |
| iid | uniform | 0.333 / 0.333 / 0.333 | 0.768 ± 0.051 | 0.091 ± 0.013 | 0.358 ± 0.086 | +0.332 ± 0.052 | 0.925 ± 0.013 |
| iid | positive | 0.306 / 0.319 / 0.375 | 0.769 ± 0.054 | 0.095 ± 0.016 | 0.340 ± 0.108 | +0.333 ± 0.053 | 0.927 ± 0.012 |