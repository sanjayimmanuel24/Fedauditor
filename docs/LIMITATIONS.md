# Limitations

Read this page before quoting any number from the project.

- **No differential privacy, no secure aggregation, no encryption.** Coefficients and counts
  travel in the clear. FedAvg is not a privacy guarantee; membership inference and gradient
  inversion are live threats in this exact setting. The honest description is "records are not
  centralised", not "privacy is guaranteed".
- **Fairness thresholds are selected on small, distribution-shifted splits.** Thresholds now come
  from the clients' local calibration splits (28–81 rows per cohort here), then are scored on the
  test set. This removes the old bias of tuning and testing on the same labels, but the tuning
  splits are small and are not a random sample of the test population, so the post-repair gap
  (0.31–0.36) is a realistic estimate with a wide CI, not a guarantee. A deployment needs a
  dedicated audit set or cross-fitting.
- **The centralised baseline is not compute-matched.** Pooled logistic regression is
  unregularised and run to convergence, while FedAvg uses fixed local epochs and an L2 penalty.
  The 1.0–1.6 accuracy-point gap is therefore an upper bound on the cost of federation, not a
  like-for-like comparison.
- **The weighting ablation has wide intervals.** With 5 seeds the between-seed spread in EO gap
  (±0.11–0.16) is larger than the differences between weighting modes. The accuracy/calibration
  trade-off is consistent across seeds; the fairness null result is consistent with "no effect"
  but cannot rule out a small one.
- **The explanation diagnostic measures baseline stability, not correctness.** `R` answers
  "would a different but legitimate cohort baseline change the top factor?" It does not answer
  "is the explanation right?" or "is the prediction right?". The background uncertainty is a
  Gaussian approximation to the sampling variability of the cohort mean; it holds the model
  weights fixed and does not include patient-level sampling noise. A stable explanation of a
  wrong prediction is still wrong.
- **Global and local are both legitimate baselines.** `R` quantifies sensitivity to that choice.
  It does not claim the global cohort is the correct reference, and it does not claim the local
  cohort is. The clinical question determines which comparison is meaningful.
- **The phase sweep is a sensitivity analysis, not an intervention.** `alpha` is a mathematical
  coordinate on the segment between two observed baselines. `alpha*` is not a clinical threshold,
  and the union bound is a bound, not an estimate. The concavity and half-space statements hold
  for this linear model and this baseline family; they should not be transported to a nonlinear
  model without redoing the argument.
- **External validation uses a different instrument.** The UCI Early Stage dataset is not a
  second split of PIMA. Its features are age, sex and symptom items, so feature importances are
  not comparable across the two datasets. The external run tests whether the *audit* is
  computable and directionally consistent on a second questionnaire, not whether the clinical
  model transfers. Its test set is 104 rows, so the across-seed intervals are wide.
- **PIMA is one population: all female, Native American ancestry, single site.** Age-cohort
  subgroups are not demographic fairness categories. Nothing here supports a claim about
  fairness by sex, ethnicity, or insurance status.
- **n = 154 test rows (PIMA) and 104 (UCI).** ECE at those sample sizes is noisy; hence the
  bootstrap CIs and the 5-seed harness. Single-run numbers should not be quoted to three
  decimals.
- **Linear model only.** Exact Shapley and exact count-based calibration are properties of this
  model class. A neural network would need approximate Shapley and would lose the exactness
  guarantee.
- **Screening aid, not a diagnosis.** The UI says this and it should stay said.
