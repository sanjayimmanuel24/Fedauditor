# Project overview

This page preserves the full project overview, claims, and long-form home-page explanation.

Every claim below is implemented in code and covered by the experiment harness. Claims that
are *not* implemented are listed in [`LIMITATIONS.md`](LIMITATIONS.md).

| Claim | Where | Verified by |
|---|---|---|
| Federated histogram binning is **exactly** equal to centralised binning | `federated_calibration.py` | `exactness_gap == 0.0` on every seed |
| Temperature scaling has **no** exact federated form; the gap is measurable | `federated_calibration.py` | `federated_temperature` vs `centralised_temperature` |
| Standardisation is fitted from **additive client statistics**, not pooled rows | `federated_scaler.py` | `(n, sum, sumsq)` pooled per client; matches `StandardScaler` |
| Imputation statistics are fitted on the **training split only** | `data_loader.py`, `federated_engine.py` | test rows cannot influence the fill value |
| Equalised-odds repair works from **integer confusion counts only** | `federated_fairness.py` | `pool_group_tables` sums counts, no rows |
| Fairness thresholds are selected on **client calibration splits**, not the test set | `federated_fairness.py` | `fairness.tuning_source == "client_calibration"` |
| Explanations are **cohort-relative**: the top risk factor changes with the baseline | `federated_explain.py`, `templates/predict.html` | signed top-1 flip rate under non-IID vs IID |
| Explanation stability is a **continuous margin**, not a single flip count | `federated_explain.py` | `R(x)`, the robustness curve, and `P(R > 0)` |
| Baseline uncertainty is estimated from **aggregate `(n, mean, covariance)` statistics** | `federated_engine.py`, `federated_explain.py` | per-patient `P(R > 0)`; no row leaves a client |
| The diagnostic has **measurable sampling reliability** | `federated_explain.split_half_reliability` | split-half disagreement, bootstrap CI, across-seed CI |
| Client value **depends on the metric** you value | `client_valuation.py` | Kendall τ between metric rankings |
| Exact Shapley is tractable for 3 clients | `client_valuation.py` | efficiency residual `0.0` |
| The flip set is a **union of half-spaces** and the margin is **concave** in the baseline | `federated_theory.py` | `e7_phase_sensitivity.csv`; unit tests |
| A **critical baseline multiplier** `α*` separates stability from flipping | `federated_theory.py`, `federated_explain.py` | median `α*`, survival curve |
| The audit is **computable on a second public dataset** | `data/uci_early_diabetes.csv`, `results_second/` | `--dataset uci_early` |
| Federation has a **measurable cost** against a pooled model | `federated_engine.centralised_baseline()` | `e0_centralised.csv` vs `e1_rounds.csv` |
| FedAvg's sample weighting **downweights the most prevalent cohort** | `federated_engine._aggregation_weights` | `e6_weighting.csv` ablation |

The home page is intentionally hero-only. This is the longer explanation it used to carry,
kept here so the detail is still available without pushing a first-time visitor past the
headline.

### Three audits, no shared records

Every number below is produced by clients that transmit aggregate sufficient statistics only.

**Calibration without a server dataset.** A centralised pipeline would fit calibration on a
validation set. A federated server is not allowed to have one. Fixed-edge histogram binning
solves this: per-bin positive counts are a sufficient statistic, so pooled counts reproduce
the centralised fit exactly. Reported through ECE, Brier, NLL and reliability diagrams.

**Subgroup fairness from counts.** Clients report only confusion counts per age cohort and
threshold. The server aggregates them and chooses per-cohort thresholds that equalise
true-positive rates, then reports the accuracy it cost to do so.

**Explanations are cohort-relative.** For a linear model the Shapley value is exact:
`w · (x − background)`. The interesting variable is the background. Explain the same patient
against the global cohort and their local cohort, then measure the *margin* by which the global
top factor survives the baseline change. The flip rate is only the sign of that margin; the
dashboard also reports its uncertainty and a robustness curve. Local-first, with exact Shapley
and client valuation.

### What leaves a client

- **Model coefficients** — weighted averages of local optimiser output, exactly as in FedAvg.
- **Per-bin integer counts** — how many calibration patients fell in each probability bin, and
  how many were positive.
- **Confusion counts per cohort** — four integers per subgroup per threshold. Enough to audit
  and repair fairness.
- **Never: patient rows.** The dashboard has no code path that uploads a record, and no claim is
  made that FedAvg alone guarantees privacy.

### Dataset provenance

| Property | Value |
| --- | --- |
| Primary source | PIMA Indians Diabetes (public, UCI) |
| Primary records | 768 · 8 features |
| External source | UCI Early Stage Diabetes Risk Prediction (public, UCI) |
| External records | 520 · 16 features (age, sex, 14 binary symptom items) |
| Clients | 3 age cohorts (non-IID); random shards for comparison |
| Model | Logistic regression, FedAvg |

Honesty about the data matters more than the size of the accuracy number. PIMA is a
single-population, all-female cohort. Subgroup analysis is therefore reported as age-cohort
analysis, not as demographic fairness. The external dataset is a different questionnaire, not a
re-split of PIMA, so it validates the audit rather than the clinical model. Screening output is
not a diagnosis.
