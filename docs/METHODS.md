# Methods

### Calibration without a server dataset

A centralised pipeline fits calibration on a validation set. A federated server is not allowed
to have one. With **fixed** equal-width bin edges, the per-bin positive and total counts are a
sufficient statistic: summing them across clients reproduces the centralised fit bit for bit.
The isotonic projection (PAVA) and Laplace smoothing are deterministic functions of those
counts, so exactness survives both.

That is only true for histogram binning. Temperature scaling's likelihood has no
finite-dimensional sufficient statistic, so the federated variant is an approximation built from
per-bin logit means — and the size of that approximation is reported rather than hidden.

### Standardisation without pooling

A `StandardScaler` fitted on the pooled training frame would contradict the claim that only
weights and counts leave a client, because the server would have been handed every client's
column means and variances. Column standardisation only needs `(n, Σx, Σx²)`, and those are
additive:

```
mean = Σx / n            var = Σx² / n − mean²
```

`federated_scaler.FederatedScaler` aggregates those three quantities per client and reproduces
`sklearn`'s population variance exactly. The arithmetic and the privacy story now agree. The
scaler is fitted on client **training** rows; calibration rows are excluded.

Missing-value imputation follows the same rule: `load_dataset` leaves impossible zeros as NaN,
and `fit_medians` is applied *after* the train/test split so test rows cannot shift a median.

### Fairness from counts

`federated_fairness.py` sweeps a threshold grid per subgroup, so each client reports a small
integer table. The server pools the tables, computes the pooled true-positive rate at the
default threshold, and picks a per-cohort threshold that matches it. The cost — accuracy,
recall, predicted-positive rate — is reported next to the benefit.

Threshold **selection** uses counts pooled from the clients' local calibration splits. The test
set is only used to score the resulting thresholds, so the post-repair gap is no longer
optimistically biased. `fairness_report` records which split was used in `tuning_source`, and
the dashboard shows it as a badge. The honest caveat is that the calibration splits are small
(28–81 rows per cohort here) and not drawn from exactly the same distribution as the test set,
so the repaired gap is a realistic estimate rather than a guaranteed one.

### Cohort-relative explanations and the robustness margin

For a linear model under feature independence the Shapley value is closed form:

```
phi_j(x) = w_j * (x_j - E_bg[x_j])
```

`w_j * x_j`, which the original dashboard used, is the same quantity with a background of zero.
The interesting variable is the background. The UI lets you explain the same patient against
the global cohort or their local cohort, side by side.

A top-1 flip count is a coarse summary. It discards how close the leader was, it ignores the
size of the margin, and it depends on a bug-prone convention. The project therefore reports a
continuous **baseline-robustness margin**.

#### Signed top factor

Write `i = argmax_j phi_j` over **signed** contributions. A large negative contribution is a
strong protective factor, not the top risk factor. The old rule `argmax(|phi|)` labelled the
largest absolute influence as "top risk", which is wrong whenever the strongest factor is
protective. All top-factor and flip calculations now use the signed rule. The display order in
the UI still uses `|phi|`, because that is the right order for a bar chart.

#### The margin

Let `phi_g` be the explanation against the global cohort and `phi_l` against the patient's local
cohort. The baseline shift is patient-independent:

```
Delta = phi_l - phi_g = w * (b_g - b_l)
```

The margin is the local lead of the global leader:

```
R(x) = min_{j != i} (phi_l,i - phi_l,j)
     = min_{j != i} ((phi_g,i - phi_g,j) + (Delta_i - Delta_j))
```

`R > 0` means the global top factor is still the top factor under the local baseline.
`R < 0` means it flipped. The conventional flip rate is exactly `P(R < 0)`; the margin is the
continuous quantity and the flip rate is only its sign. A patient with `R = 0.001` and a patient
with `R = 2.0` both count as "stable" under a flip rate, but they are not equally stable. The
robustness curve `P(R > delta)` exposes that difference.

#### Background uncertainty from aggregate statistics

The local background is a cohort mean estimated from a finite sample, so it is itself uncertain.
A patient's top factor can flip because the cohort mean would move if the cohort were sampled
again. The server can quantify this without seeing a row: each client contributes the aggregate
sufficient statistics `(n, mean, covariance)`, and the sampling covariance of the cohort mean is
`Sigma / n`. Drawing plausible cohort means from that distribution gives a per-patient
`P(R > 0)`: the probability that the top factor survives re-estimation of the local cohort mean
at the observed cohort size.

This is a Gaussian approximation. It covers uncertainty in the **background**, not uncertainty
in the model weights or in the patient's own features. It is reported as such.

#### Validating the diagnostic itself

A measurement paper has to show that the measurement is stable enough to interpret:

- **Bootstrap CI** over patients for the flip rate and mean margin.
- **Split-half test–retest**: two random halves of the test set are scored independently; the
  mean absolute disagreement is the sampling noise a reader should expect from the aggregate.
- **Across-seed CI**: the harness reruns the whole federated fit for five seeds and reports a
  95% interval, so the between-run spread is visible rather than hidden.
- **Cohort breakdown**: the same diagnostic is reported per age cohort, not only pooled.

What this validates is reproducibility of the *measurement*. It is not external validation of
the explanation, and it is not evidence that the model is clinically right.

#### Half-space geometry and the phase diagram

The margin has a useful algebraic form. Fix the global leader `i` and write
`a_j = phi_g,i - phi_g,j >= 0` and `c_j = Delta_i - Delta_j` for every competitor `j != i`.
Then

```
R(alpha) = min_{j != i} (a_j + alpha * c_j)
```

under the baseline path `b_l(alpha) = b_g + alpha * (b_l - b_g)`. Because `R` is a minimum of
affine functions of `alpha`, it is **concave**. With `R(0) >= 0` the stable set is an interval,
so stability persists until a single critical multiplier

```
alpha* = min_{j: c_j < 0} a_j / (-c_j)
```

and then the leader flips for every `alpha > alpha*`. Equivalently, the flip set is a union of
pairwise half-spaces `{a_j + alpha c_j < 0}`, which gives the union bound
`P(flip) <= sum_j P(a_j + alpha c_j < 0)`. `federated_theory.py` implements this decomposition,
and `experiments.py` writes the sweep over `alpha in [0, 3]` to `e7_phase_sensitivity.csv`.
`alpha = 0` is the global baseline, `alpha = 1` is the observed local baseline, and `alpha > 1`
asks how much further the cohort baseline would have to move before the explanation changes.
This is a sensitivity analysis, not a causal intervention, and `alpha*` is not a clinical
threshold.

### Client valuation

With three clients there are only 8 coalitions, so Shapley is computed exactly rather than
sampled. The value of a coalition is the improvement of the model trained on its union over the
**untrained** model (zero weights, p = 0.5). A prevalence-only baseline is deliberately avoided:
it scores a perfect ECE of 0, which would make every client look like it worsens calibration.

Loss metrics are sign-flipped so a positive number always means "better". Efficiency holds
exactly (`sum(phi) == v(full) - v(empty)`).
