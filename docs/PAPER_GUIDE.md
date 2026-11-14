# Paper guide

This is an **empirical measurement / audit paper**, not a new-model paper. The model is a
deliberately simple linear classifier; the contribution is the measurement layer and the
negative results it exposes. That framing is defensible at a clinical-informatics or
trustworthy-ML venue. It is not defensible as "we invented a new federated learning method".

### Working title

> *When the Explanation Depends on the Cohort: A Baseline-Robustness Diagnostic for Federated
> Clinical Models*

Alternative, if the venue prefers a broader framing:

> *Beyond Accuracy in Federated Diabetes Screening: Calibration, Subgroup Fairness, and
> Explanation Robustness as First-Class Measurements*

### One-sentence claim

> In a federated clinical model, two legitimate cohort baselines can disagree about a patient's
> top risk factor; the disagreement is a measurable, uncertainty-aware property of the cohort
> structure, and it is invisible to accuracy, calibration, and aggregate fairness metrics.

### Contributions (the honest list)

1. **A continuous diagnostic.** Replace the top-1 flip indicator with a signed margin
   `R(x) = min_{j≠i}(φ_l,i − φ_l,j)`, where the flip rate is exactly `P(R < 0)`. The margin is
   threshold-free and its sign is the old metric.
2. **Federated uncertainty without rows.** Estimate the sampling covariance of each cohort mean
   from aggregate `(n, mean, covariance)` sufficient statistics and report per-patient
   `P(R > 0)`. The server never receives a patient record.
3. **A validation protocol for the measurement itself.** Bootstrap CIs, split-half test–retest,
   per-cohort breakdowns, and a five-seed across-run interval. The paper reports whether the
   diagnostic is stable, not only whether it is statistically significant.
4. **A negative result with practical value.** Reweighting FedAvg toward the small,
   high-prevalence cohort buys about 1.2 accuracy points but worsens ECE and leaves the
   equal-opportunity gap flat. Explanation fragility is not repaired by the obvious aggregation
   knob.
5. **A half-space geometry and phase diagram for the diagnostic.** The margin is a minimum of
   affine functions of the baseline, so it is concave, the stable set is an interval, and the
   flip set is a union of pairwise half-spaces. That yields a deterministic union bound and a
   per-patient critical multiplier `α*`. The empirical sweep makes the geometry visible and
   shows the two regimes separate by an order of magnitude in `α*` (median 2.36 vs 20.62).
6. **An external-validation run on a second public instrument.** The same audit is computed on
   the UCI Early Stage questionnaire. The coarse flip count happens to be identical across
   regimes there, while the continuous margin, `P(R > 0)`, robust-at-95 share and median `α*`
   still separate them in the expected direction. This is reported as a boundary condition, not
   as a stronger version of the primary claim.

### What the paper must not claim

- It does **not** propose a new federated optimisation algorithm.
- It does **not** prove a privacy guarantee. Coefficients and counts travel in the clear.
- It does **not** validate explanations against clinical ground truth. There is no clinician
  adjudication and no causal estimand.
- It does **not** generalise beyond PIMA's single-population, all-female cohort.
- The diagnostic is about **baseline sensitivity**, not about whether a prediction is correct.

### Suggested venue and format

| Venue type | Fit | Format |
|---|---|---|
| IEEE ICHI / IEEE BHI | Clinical informatics, audit and trust | 4–6 page short paper |
| IEEE CBMS | Applied clinical ML with a measurement story | 4–6 page paper |
| IEEE TrustCom / workshop on trustworthy FL | Explanation stability as a trust property | workshop paper |
| Local graduate symposium / department seminar | Fast feedback before a full submission | 15-minute talk |

The safest first submission is a **short paper with one strong table and two figures**, not a
long paper that overclaims a method. If the venue asks for a "novel method", this project is a
weaker fit; if it asks for "evaluation, audit, or deployment lessons", it is a strong fit.

### Figure and table plan

| Item | Source | Message |
|---|---|---|
| Table 1 | `results/paper_tables.md` Table 1 | Same accuracy, different audit profile |
| Table 2 | `results/paper_tables.md` Table 2 | Margin, uncertainty, split-half reliability, union bound |
| Table 3 | `results/paper_tables.md` Table 3 | Reweighting buys accuracy but not explanation stability |
| Table 4 | `results/paper_tables.md` Table 4 | Phase sensitivity and the critical multiplier `α*` |
| Table 5 | `results/paper_tables.md` Table 5 | External validation on the UCI Early Stage instrument |
| Fig. 1 | `e4_explanations.csv` | Robustness curve for age vs IID |
| Fig. 2 | `e4_explanations.csv` | Per-cohort margin distribution / lower tail |
| Fig. 3 | `e7_phase_sensitivity.csv` | Flip rate and survival vs `α`, with the union bound |
| Fig. 4 | `e6_weighting.csv` | Reweighting does not repair fragility |
| Appendix | `summary.json` | Full CIs and split-half reliability |

### Reviewer objections and honest responses

| Objection | Response |
|---|---|
| "This is just SHAP with a different background." | Correct. The paper is not a new attribution method. It is a measurement of how much the attribution depends on a legitimate modelling choice, with a validation protocol. |
| "The flip rate is a threshold artefact." | Agreed, which is why the headline is the continuous margin and the robustness curve. The flip rate is reported only as `P(R < 0)`. |
| "Your background uncertainty is Gaussian and ignores model uncertainty." | True and stated. It is a lower-bound-style sensitivity check on the cohort mean, not a full posterior. |
| "PIMA is not a fairness benchmark." | True. The paper reports age-cohort analysis, not demographic fairness, and says so in the limitations. |
| "Only three clients." | True. Exact Shapley and exact count-based calibration are the reason; scaling is future work. |
| "Why not just use the global background?" | That is a modelling choice, not a ground truth. The diagnostic quantifies the cost of the choice rather than hiding it. |
| "Is the diagnostic itself reliable?" | Split-half disagreement and across-seed CIs are reported in Table 2. At n=154 the aggregate is usable; patient-level status is a probability, not a verdict. |

### 60-second pitch

> We built a federated diabetes screener, but the paper is not about the screener. It is about a
> measurement problem. The same model can explain the same patient against the whole population
> or against their own age cohort, and those are both legitimate. Under an age-cohort split, the
> top risk factor changes for about one patient in four; under random sharding it changes for
> about one in twenty. Accuracy cannot see this, calibration cannot see it, and the fairness gap
> does not move when you reweight the cohorts. So we turned the flip count into a continuous
> margin, propagated the uncertainty in each cohort mean from aggregate statistics, derived the
> half-space geometry that gives a union bound and a critical baseline multiplier, and validated
> the measurement with split-half and across-seed intervals. We then repeated the audit on a
> second public questionnaire, where the coarse flip count happened to be identical across
> regimes but the continuous diagnostics still separated them. The result is a diagnostic that
> tells you when an explanation is a property of the patient and when it is a property of the
> cohort.
