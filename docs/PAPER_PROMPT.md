# Paper-writing prompt — FedHealth baseline-robustness audit

Use this prompt with a strong writing model (Claude Sonnet or equivalent) after the
experiments and tables in this repository have been regenerated. It is intentionally
strict about what may be claimed. Replace only the bracketed placeholders such as
`[VENUE]` and `[AUTHOR]`; do not invent numbers.

---

## Role

You are writing a 4–6 page conference paper for **[VENUE]** in the measurement,
evaluation, or audit track. The paper reports an empirical audit of a federated
clinical screening model. It is **not** a new-method paper, **not** a clinical
validation study, and **not** a privacy paper.

Write in the voice of a careful empirical researcher. Prefer "we measure", "we
report", "the diagnostic separates", and "this is consistent with". Avoid
"revolutionary", "state of the art", "we solve", "guarantees privacy", and
"clinically validated".

## Working title

> **When the Explanation Depends on the Cohort: A Baseline-Robustness Diagnostic
> for Federated Clinical Models**

Alternative if the venue prefers a broader framing:

> **Beyond Accuracy in Federated Diabetes Screening: Calibration, Subgroup
> Fairness, and Explanation Robustness as First-Class Measurements**

## One-sentence claim

> In a federated clinical model, two legitimate cohort baselines can disagree about
> a patient's top risk factor; the disagreement is a measurable, uncertainty-aware
> property of the cohort structure, and it is invisible to accuracy, calibration,
> and aggregate fairness metrics.

## Absolute rules

1. **Use only measured results.** Every number must come from
   `results/summary.json`, `results/e*.csv`, `results_second/summary.json`, or
   `results/paper_tables.md`. Never estimate, round from memory, or invent a
   number. If a number is missing, write "not measured" and explain why.
2. **Real data only.** The PIMA run uses `data/pima_diabetes.csv`. The external
   run uses `data/uci_early_diabetes.csv`. If a run is flagged synthetic, do not
   report it as a finding.
3. **Clients are age cohorts, not hospitals.** Each simulated client serves one
   age band (`<30`, `30–45`, `45+`). The IID regime uses random shards, not
   cohorts. Do not write "hospital" unless you are explicitly describing a
   deployment analogy, and even then say "simulated client".
4. **Global and local baselines are both legitimate.** The diagnostic measures
   sensitivity to that choice. It does not declare one baseline correct.
5. **The flip rate is not the headline.** The headline is the continuous margin
   `R`, its uncertainty, and its phase behaviour. The flip rate is exactly
   `P(R < 0)`.
6. **No privacy guarantee.** Coefficients, integer counts, and aggregate
   sufficient statistics travel in the clear. The honest claim is "patient rows
   are never centralised", not "privacy is guaranteed".
7. **No clinical claim.** Screening output is a decision aid, not a diagnosis.
   There is no clinician adjudication, no causal estimand, and no external
   clinical cohort.
8. **No new-method claim.** Linear Shapley, FedAvg, equalised odds, and exact
   coalition Shapley are standard. The contribution is the measurement layer,
   the geometry, and the validation protocol.

## Study design

- **Primary dataset:** PIMA Indians Diabetes, 768 records, 8 numeric features.
  Single population, all female, single site. This limits external validity and
  must be stated.
- **External dataset:** UCI Early Stage Diabetes Risk Prediction, 520 records,
  16 questionnaire features (age, sex, and binary symptom items). It is a
  different instrument, not a re-split of PIMA. Use it for external validation
  of the audit profile, not for a leaderboard comparison.
- **Model:** L2-regularised logistic regression trained with federated averaging.
  Deliberately simple, because it makes exact linear Shapley and exact
  count-based calibration possible.
- **Clients:** 3 simulated clients. Under the age partition each client serves a
  distinct age cohort, producing clinically motivated non-IID data. The IID
  partition is random shards and is the comparison regime.
- **Runs:** 5 seeds × 15 communication rounds × 2 partition regimes. Report
  means with 95% confidence intervals across seeds. The external dataset uses
  the same protocol; the weighting ablation may be omitted there and that must
  be stated.
- **Train/test:** stratified 80/20 split, seed-dependent. Imputation statistics
  are fitted on the training split only. The federated scaler is fitted from
  additive `(n, sum, sum of squares)` statistics. Calibration and fairness
  thresholds are selected on client-local calibration splits, never on the test
  labels.

## Measurements to report

1. **Predictive performance:** accuracy, ROC AUC, Brier, NLL, ECE, MCE, recall,
   precision. Include the pooled centralised reference and state that it is not
   compute-matched to FedAvg, so the gap is an upper bound on the cost of
   federation.
2. **Calibration:** raw vs federated histogram binning vs temperature scaling.
   The exactness gap for histogram binning is a property of the sufficient
   statistic; the benefit of calibration is regime-dependent and must be
   reported as such.
3. **Subgroup fairness:** per-cohort TPR/FPR before and after equalised-odds
   threshold repair, the accuracy cost, and the split used to select thresholds.
   The repair is imperfect; say so.
4. **Explanation instability:** signed top-1 flip rate, mean and median margin
   `R`, lower-tail quantiles, per-cohort breakdown, bootstrap CI, split-half
   test–retest, and across-seed CI.
5. **Background uncertainty:** per-patient `P(R > 0)` under a Gaussian model of
   cohort-mean uncertainty, estimated from aggregate `(n, mean, covariance)`
   sufficient statistics. State that model weights are fixed and patient-level
   sampling noise is not included.
6. **Aggregation weighting ablation:** sample vs uniform vs positive-case
   weighting. Report the accuracy/calibration trade-off and the fairness null
   result honestly.
7. **Client valuation:** exact Shapley over all 8 coalitions for 3 clients, per
   metric, with Kendall τ between metric rankings. The value of a client depends
   on which metric you value.
8. **Phase sensitivity:** see the theory and figure specification below.

## Theory to include

For a linear model under feature independence, the exact Shapley value against
background `b` is

```
phi_j(x; b) = w_j * (x_j - b_j).
```

Fix the global leader `i = argmax_j phi_j(x; b_g)` using **signed** values. A
large negative contribution is a strong protective factor, not the top risk
factor. Define the baseline shift

```
Delta = phi_l - phi_g = w * (b_g - b_l)
```

and, for each competitor `j != i`,

```
a_j = phi_g,i - phi_g,j          (>= 0)
c_j = Delta_i - Delta_j
R(x) = min_{j != i} (a_j + c_j).
```

Then:

- `R > 0` means the global top factor survives the local baseline.
- `R < 0` means it flips.
- The conventional flip rate is exactly `P(R < 0)`.

**Concavity and half-spaces.** `R` is a minimum of affine functions of the
baseline, hence concave. With `R(0) >= 0`, the stable set is an interval and the
flip set is a union of pairwise half-spaces `{a_j + c_j < 0}`. Therefore

```
P(flip) <= sum_{j != i} P(a_j + c_j < 0).
```

The empirical version of the right-hand side is a deterministic upper bound on
the observed flip rate. Report it next to the empirical curve; do not claim the
bound is tight.

**Critical baseline multiplier.** Move the local baseline along

```
b_l(alpha) = b_g + alpha * (b_l - b_g),   alpha in [0, 3].
```

`alpha = 0` is the global baseline, `alpha = 1` is the observed local baseline,
and `alpha > 1` asks how much further the cohort baseline would have to move
before the explanation changes. The per-patient critical multiplier is

```
alpha* = min_{j: c_j < 0} a_j / (-c_j),
```

with `alpha* = +inf` when no competitor can ever overtake the leader. Then
`flip(alpha) = 1{alpha* < alpha}` and the observed flip rate is
`P(alpha* < 1)`. This is the phase diagram: flip rate, survival
`P(alpha* >= alpha)`, mean margin, and the union bound as functions of `alpha`.

State clearly that the sweep is a **sensitivity analysis**, not a causal
intervention and not a claim that any `alpha` is clinically meaningful.

## External validation: how to frame it

The UCI Early Stage dataset is a **different questionnaire**, not a second split
of PIMA. Its features are age, sex, and 14 binary symptom items; PIMA has
pregnancies, glucose, blood pressure, skin thickness, insulin, BMI, pedigree
function, and age. Therefore:

- Do **not** compare feature importances across datasets.
- Do **not** claim the model transfers clinically.
- Do **compare the audit profile**: whether the same qualitative separation
  between age-cohort non-IID and IID appears, and whether the margin/phase
  diagnostic is computable and stable on a second instrument.
- If the external dataset shows weaker or absent fragility, report that as a
  boundary condition. A diagnostic that does not fire everywhere is more
  credible, not less.
- Mention sample size: PIMA has 154 test rows per seed; UCI has 104. Both are
  small, so report intervals and avoid three-decimal overprecision.

## Figures and tables

Build tables with `python paper_assets.py --results results --external
results_second --latex`. Do not retype cells. Suggested paper assets:

| Item | Source | Message |
|---|---|---|
| Table 1 | `results/paper_tables.md` Table 1 | Same accuracy, different audit profile |
| Table 2 | Table 2 | Margin, background uncertainty, split-half reliability, union bound |
| Table 3 | Table 3 | Reweighting buys accuracy but not explanation stability |
| Table 4 | Table 4 | Phase sensitivity and critical multiplier |
| Table 5 | Table 5 | External validation on UCI Early Stage |
| Fig. 1 | `results/e4_explanations.csv` | Robustness curve or margin distribution, age vs IID |
| Fig. 2 | `results/e7_phase_sensitivity.csv` | Flip rate and survival vs `alpha`, with union bound |
| Fig. 3 | `results/e6_weighting.csv` | Reweighting does not repair fragility |
| Appendix | `results/summary.json` | Full CIs and per-cohort breakdowns |

Figure captions must state the regime, dataset, number of seeds, and what the
shaded band or error bar represents. Never show a single-seed curve as if it
were the main result.

## Paper structure

1. **Abstract** (150–200 words): problem, measurement, two headline regimes,
   external validation, claim. No method novelty language.
2. **Introduction:** accuracy hides baseline-dependent explanations; define the
   gap; list contributions as measurement contributions.
3. **Related work:** Shapley/feature attribution, explanation stability,
   federated auditing, calibration and subgroup fairness. Position the paper as
   measurement, not a new attribution formula.
4. **Data and federated model:** both datasets, age-cohort vs IID partition,
   preprocessing, FedAvg, exactness properties, what leaves a client.
5. **Audit measurements:** calibration, fairness, signed margin, background
   uncertainty, validation protocol.
6. **Geometry and phase diagram:** the `a/c` decomposition, concavity,
   half-spaces, union bound, `alpha*`.
7. **Results:** primary dataset, both regimes, all audits; then phase results.
8. **External validation:** UCI Early Stage; same audit, different instrument;
   boundary conditions.
9. **Discussion:** what an auditor should do with the diagnostic; why
   reweighting is not a repair; what would change the conclusion.
10. **Limitations and ethics:** single-population PIMA; small n; linear model;
    Gaussian background approximation; no privacy guarantee; no clinical
    validation; age cohorts are not demographic fairness categories.
11. **Conclusion:** a diagnostic for when an explanation is a property of the
    patient versus a property of the cohort definition.
12. **References:** gather real citations; do not fabricate BibTeX.

## Allowed claims

- "Under an age-cohort partition, the signed top factor changes for a
  substantially larger share of patients than under random sharding, while
  accuracy is statistically indistinguishable."
- "The margin is a continuous, threshold-free replacement for the flip count;
  the flip rate is its sign."
- "Background uncertainty can be estimated from aggregate sufficient statistics
  without centralising patient rows."
- "The diagnostic has measurable sampling reliability at this sample size, and
  individual patient statuses remain probabilistic."
- "Reweighting FedAvg toward the small high-prevalence cohort does not repair
  explanation fragility in this study."
- "The phase diagram makes the half-space geometry and the union bound
  empirically visible."
- "The audit is computable on a second public questionnaire, with the caveat
  that the instrument differs."

## Forbidden claims

- A new federated optimisation algorithm.
- A privacy or security guarantee.
- Clinical validity, causal effect, or treatment recommendation.
- Demographic fairness by sex, ethnicity, or insurance status.
- "Hospitals" as the unit of analysis when the clients are age cohorts.
- That the global baseline is correct, or that the local baseline is correct.
- That a stable explanation implies a correct prediction.
- That the union bound is tight, or that `alpha*` is a clinical threshold.
- That PIMA results generalise to the UCI questionnaire or to deployment.

## Quality gates before returning the draft

- [ ] Every number in the draft is traceable to a file in `results/` or
      `results_second/`.
- [ ] The draft says "age cohort" or "simulated client", not "hospital".
- [ ] The headline is the margin, not the flip rate.
- [ ] The union bound is labelled as a bound, not an estimate.
- [ ] The external dataset is framed as a different instrument.
- [ ] The limitations section names PIMA's single-population restriction, the
      small test sets, the linear model, and the Gaussian background
      approximation.
- [ ] No privacy guarantee, no clinical claim, no new-method claim.
- [ ] All tables are generated from `summary.json`, not retyped.
- [ ] Figure captions state regime, dataset, seeds, and error bars.
- [ ] The references are real and verifiable.

## Output format

Return:

1. A complete LaTeX draft with sections, captions, and `\cite{}` placeholders
   for references you cannot verify.
2. A short "numbers used" appendix mapping each headline number to its source
   file and field.
3. A list of claims that still need evidence, if any.
4. A one-paragraph cover note to **[AUTHOR]** summarising the strongest result
   and the weakest claim.

Do not return a generic ML paper. Return this measurement paper.
