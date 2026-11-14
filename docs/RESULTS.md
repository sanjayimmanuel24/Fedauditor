# Results

All numbers below come from the reproducible harness in `experiments.py` and the saved outputs in `results/` and `results_second/`. See `SETUP.md` for the exact commands.

5 seeds × 15 rounds × 2 partitioning regimes, 95% confidence intervals.
Reproduce with `python experiments.py --seeds 5 --rounds 15`.

| | Age-cohort split (non-IID) | Random shards (IID) | Pooled (centralised) |
|---|---|---|---|
| Accuracy | 0.762 ± 0.044 | 0.768 ± 0.051 | 0.778 ± 0.029 |
| ROC AUC | 0.837 ± 0.032 | 0.838 ± 0.028 | 0.842 ± 0.030 |
| ECE, uncalibrated | 0.068 ± 0.020 | 0.091 ± 0.013 | 0.064 / 0.062 |
| ECE, federated binning | 0.060 ± 0.025 | 0.056 ± 0.016 | — |
| Binning exactness gap | **0.0** | **0.0** | — |
| Equal-opportunity gap, before | 0.413 ± 0.241 | 0.420 ± 0.212 | — |
| Equal-opportunity gap, after | 0.307 ± 0.114 | 0.358 ± 0.086 | — |
| Accuracy change from repair | +0.010 | +0.001 | — |
| **Signed** top-1 flip rate | **22.6% ± 5.0** | **4.9% ± 2.3** | — |
| Mean Spearman ρ (global vs local) | 0.809 ± 0.028 | 0.985 ± 0.004 | — |
| Mean robustness margin R | +0.337 ± 0.018 | +0.332 ± 0.052 | — |
| Mean P(R > 0) under background uncertainty | **0.775 ± 0.043** | **0.925 ± 0.013** | — |
| Share robust at 95% | **0.690 ± 0.051** | **0.784 ± 0.021** | — |
| Split-half flip-rate disagreement | 0.054 ± 0.005 | 0.027 ± 0.007 | — |
| Split-half margin disagreement | 0.064 ± 0.006 | 0.052 ± 0.005 | — |
| Union bound at α = 1 | 0.445 ± 0.076 | 0.057 ± 0.017 | — |
| Median critical multiplier α* | **2.36 ± 0.73** | **20.62 ± 15.95** | — |
| Share with finite α* | 0.942 ± 0.089 | 0.849 ± 0.155 | — |

The flip rate is now computed from a **signed** top factor. A large negative contribution is a
strong protective factor, not the top risk factor; the old `argmax(|φ|)` rule conflated the two.
The qualitative conclusion is unchanged, but the honest number falls from 25.3% to 22.6%.

"Pooled" is a fully pooled, unregularised logistic regression fitted on the same split with the
same federated scaler the clients used. Because that scaler is fitted on each regime's client
training rows, the pooled model is *matched to its regime* rather than being one fixed model —
which is why its ECE differs slightly between the two columns. It is not compute-matched to
FedAvg, so treat the accuracy gap as an upper bound on what federation costs: about 1.0–1.6
points.

Six things worth reading off that table:

1. **The two federated regimes are indistinguishable on accuracy** (0.762 vs 0.768, heavily
   overlapping CIs) but diverge on every audit metric. Federation itself costs 1.0–1.6 points
   against pooling; the *partitioning regime* costs almost nothing on accuracy and a great deal
   on explanations. An accuracy-only evaluation cannot see the difference that matters.
2. **The value of federated calibration depends on the split.** Histogram binning improves ECE
   from 0.091 to 0.056 under IID (a 39% reduction) but only from 0.068 to 0.060 under the age
   split (12%). Exactness is guaranteed in both cases (`gap == 0.0`); the *benefit* is not.
   Under the age split the local calibration splits are small and already reasonably calibrated,
   so there is less for the projection to fix.
3. **Explanation instability is a tail phenomenon, not a shift in the average explanation.**
   The same weights and the same patients produce a mean margin of +0.337 under the age split
   and +0.332 under IID: the middle of the distribution looks identical. What differs is the
   lower tail. 22.6% of age-split patients have `R < 0`, versus 4.9% under IID, and the
   uncertainty-aware probability separates them more cleanly than the mean (mean `P(R > 0)`
   0.775 vs 0.925). Reporting only a mean attribution or a single flip count would miss this.
   This is the project's central measurement result.
4. **The diagnostic is reliable enough to quote at this sample size, but only with its CI.**
   Split-half resampling moves the flip rate by 0.054 under the age split and 0.027 under IID;
   the across-seed CI is ±0.050 and ±0.023. The between-regime difference is larger than either,
   so the ranking is stable. Individual patient statuses are not: the patient-level P(R > 0)
   is a probability, not a verdict.
5. **Post-repair fairness is now measured honestly, and it is worse than it used to look.**
   Thresholds are selected on client calibration splits and scored on the untouched test set, so
   the gap only falls 0.413 → 0.307 instead of the old 0.425 → 0.093 obtained by tuning on the
   test labels themselves. The remaining gap is the real one: 28–81 calibration rows per cohort
   are not enough to pin a threshold precisely. Accuracy does not fall — it rises slightly,
   within noise.
6. **The geometry is visible in the numbers.** The flip set is a union of pairwise half-spaces,
   so the per-patient union bound (0.445 ± 0.076 age, 0.057 ± 0.017 IID) must sit above the
   observed flip rate (0.226, 0.049) — and it does. The median critical multiplier `α*` is
   **2.36 ± 0.73** under the age split versus **20.62 ± 15.95** under random shards: the local
   baseline would have to move more than twice as far before the typical age-split explanation
   changes, and more than twenty times as far under IID. `α*` is a sensitivity coordinate, not a
   clinical threshold, and 5.8% of age-split patients have no finite `α*` at all.

### Aggregation weighting ablation (`e6_weighting.csv`)

Under the age split, standard sample-weighted FedAvg gives the cohorts weights
`[0.527, 0.296, 0.176]` — the `45+` cohort, with the highest prevalence (0.49), gets the least
influence. Re-running the same seeds with the other rules:

| Weighting | Client weights | Accuracy | ECE | EO gap after repair |
|---|---|---|---|---|
| `sample` | 0.527 / 0.296 / 0.176 | 0.762 | **0.068** | 0.307 |
| `uniform` | 0.333 / 0.333 / 0.333 | **0.774** | 0.082 | 0.317 |
| `positive` | 0.327 / 0.428 / 0.245 | **0.774** | 0.092 | 0.310 |

Two honest readings, both of which are negative results for the obvious hypothesis:

- **Reweighting buys accuracy at the cost of calibration.** Giving the small high-prevalence
  cohort more influence is associated with about 1.2 extra accuracy points, but ECE degrades
  from 0.068 to 0.082–0.092. The direction is consistent across seeds, but with 5 seeds both
  movements are comparable to their intervals: read this as a trade-off worth investigating,
  not a settled ranking.
- **Reweighting does not fix subgroup parity.** The equal-opportunity gap after repair is flat
  across all three rules (0.307, 0.317, 0.310). The parity gap is therefore not an artefact of
  FedAvg's sample weighting; it comes from the cohorts themselves. Under IID the same ablation is
  almost inert (`uniform` is identical to `sample` by construction; `positive` moves the gap
  0.358 → 0.340 within noise). With 5 seeds the between-seed spread in EO gap (±0.11–0.16) is
  larger than the differences between modes, so this is a null result, not evidence of equality.

### External validation on a second public dataset (`results_second/`)

The UCI Early Stage Diabetes Risk Prediction dataset is a different questionnaire: 520 records,
age, sex, and 14 binary symptom items. It is **not** a second split of PIMA and its feature
importances are not comparable. The point of the run is to ask whether the audit — not the
clinical model — is computable and informative on a second instrument. Its age bands are
`<35`, `35-50`, `50+` because the PIMA bands leave only 20 rows in the youngest cohort.

| | UCI age-cohort split | UCI random shards |
|---|---|---|
| Accuracy | 0.904 ± 0.058 | 0.894 ± 0.036 |
| ROC AUC | 0.975 ± 0.011 | 0.977 ± 0.005 |
| ECE, uncalibrated | 0.116 ± 0.018 | 0.119 ± 0.017 |
| Signed top-1 flip rate | 0.138 ± 0.162 | 0.138 ± 0.095 |
| Mean margin `R` | **0.197 ± 0.060** | 0.177 ± 0.014 |
| Mean `P(R > 0)` | **0.842 ± 0.112** | 0.817 ± 0.059 |
| Share robust at 95% | **0.621 ± 0.168** | 0.533 ± 0.062 |
| Union bound at α = 1 | 0.158 ± 0.169 | 0.138 ± 0.095 |
| Median `α*` | **3.69 ± 1.53** | 6.48 ± 3.23 |

The honest reading is a **boundary condition**: the signed flip rate is identical in the two
external regimes (0.138), so the coarse count does not separate them. The continuous
diagnostics do, in the expected direction — the age-cohort split has a higher mean `P(R > 0)`,
a larger share robust at 95%, and a much smaller median critical multiplier `α*`. But the
across-seed intervals are wide at n = 104 test rows, and the external result is weaker and less
clean than PIMA. That is what an external validation should look like: same diagnostic,
different instrument, not a stronger version of the primary claim. The weighting ablation was
not repeated on this dataset and is reported only for PIMA.
