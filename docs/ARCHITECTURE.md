# Architecture

```
data/pima_diabetes.csv      768 real PIMA records (primary dataset)
data/uci_early_diabetes.csv 520 real UCI Early Stage records (external validation)
data_loader.py              registry, loading, train-only imputation, age-band partition
federated_scaler.py         federated mean/variance from (n, sum, sumsq) statistics
federated_engine.py         FedAvg loop; returns metrics + 3 audits per round
federated_metrics.py        accuracy, AUC, Brier, NLL, ECE, MCE, reliability, bootstrap CI
federated_calibration.py    exact federated histogram binning + temperature comparison
federated_fairness.py       subgroup counts, equalised-odds thresholds, repair cost
federated_explain.py        exact linear Shapley, signed top-1, robustness margin,
                            background-uncertainty P(R>0), split-half reliability
federated_theory.py         half-space decomposition, concavity, union bound, phase sweep
client_valuation.py         exact Shapley over all 8 coalitions, per metric
experiment_store.py         SQLite persistence + CSV export
experiments.py              reproducible multi-seed harness -> results/
paper_assets.py             summary.json -> conference tables (Markdown/LaTeX)
app.py                      Flask routes, per-user RunSession, JSON serialisation
templates/                  5-tab audit dashboard + patient explanation form
```

The Flask app still runs the primary PIMA cohort by default. The external dataset is selected
by the experiment harness (`--dataset uci_early`), which keeps the clinical UI unchanged while
making the audit reproducible on a second instrument.

### Clients are age cohorts, not random shards

`config.AGE_BANDS` splits PIMA training rows into `<30`, `30-45`, `45+`. Each simulated client
serves one cohort. Prevalence differs sharply (`0.22`, `0.50`, `0.49` in the default run), which
is what produces the calibration drift and subgroup gaps being measured. `PARTITION_MODE="iid"`
gives random shards for comparison, and the harness runs both.

UCI's age distribution is shifted upward, so its bands are `<35`, `35-50`, `50+`. With the PIMA
bands the youngest UCI cohort has only 20 rows and cannot support a local calibration split.
`partition_client_indices` reports any fallback to IID explicitly (`partition_fallback`,
`partition_reason` in `summary()`), so an "age" run can never silently be an IID run.

### What a client sends

- Model coefficients and bias (FedAvg, as usual).
- For standardisation, three numbers per feature: `n`, the column sum, and the column sum of
  squares. These are additive, so the server recovers the pooled mean and variance without
  ever seeing a row.
- Per-bin **integer counts**: how many calibration patients fell in each of 10 fixed
  probability bins, and how many of them were positive.
- Per-cohort, per-threshold **four integers**: TP, FP, FN, TN.
- Per-cohort **`(n, mean, covariance)`** for the local background. The covariance is an
  aggregate statistic; it lets the server propagate background uncertainty without a row.

Never a patient row. The dashboard has no code path that uploads one.

### How much weight a cohort gets

`AGGREGATION_WEIGHTING` (or `--weighting`) selects how the server combines updates:

| Mode | Weight | Effect |
|---|---|---|
| `sample` (default) | ∝ rows | standard FedAvg |
| `uniform` | equal | small cohorts get a full vote |
| `positive` (alias `prevalence`) | ∝ rows × prevalence | influence tracks the number of diabetic patients contributed |

Under the default age split the sample-weighted scheme gives the `45+` cohort — which has the
highest prevalence — only `0.176` of the aggregate weight, versus `0.527` for `<30`. The `e6`
ablation reports what that mechanical choice does to accuracy, ECE and the equal-opportunity gap.

### Dashboard

Five tabs, all fed by real round output — no mocked numbers:

1. **Performance** — accuracy, AUC, ECE per round; per-client cohort cards; confusion matrix;
   federated vs centralised comparison with the accuracy cost of federation.
2. **Calibration** — reliability diagram, six-method comparison, exactness gap.
3. **Subgroup Fairness** — per-cohort TPR before/after repair, fairness-right frontier, cost, and
   a badge showing which split selected the thresholds.
4. **Explanations** — signed top-1 flip rate with its bootstrap CI, mean robustness margin,
   background-uncertainty `P(R > 0)`, share of patients robust at 95%, the robustness curve,
   the baseline phase diagram (flip rate, survival and union bound vs `α`), and a per-cohort
   breakdown. The old flip count is still there, but only as the sign of `R`.
5. **Client Value** — Shapley by metric, ranking agreement, coalition table.

Runs persist to SQLite (`users.db`) round by round and export to CSV from the dashboard header.
