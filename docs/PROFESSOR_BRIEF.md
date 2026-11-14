# Professor briefing — FedHealth explanation-robustness study

One page for our supervision meeting. Full methods and results are in
[`OVERVIEW.md`](OVERVIEW.md), [`METHODS.md`](METHODS.md) and [`RESULTS.md`](RESULTS.md);
the conference plan is in [`PAPER_GUIDE.md`](PAPER_GUIDE.md).

---

## 1. The 60-second version

I built a federated diabetes-screening prototype, and it gave us a useful finding. The paper is a
**measurement study** with a clear, practical message:

> A clinical explanation can depend on which cohort it is compared against. The same model, the
> same patient, two legitimate reference populations — and the top risk factor changes for about
> **23%** of patients under an age-cohort split, versus about **5%** under random sharding.
> Accuracy, calibration and the aggregate fairness gap stay essentially the same.

So we built a **continuous, uncertainty-aware diagnostic** that makes this invisible risk visible,
and we validated the diagnostic itself. The good news: the effect is real, measurable, and
reproducible — which means it can be audited rather than guessed at.

---

## 2. What is implemented

| Component | What it does |
|---|---|
| `federated_engine.py` | FedAvg over three simulated age-cohort clients, with train-only imputation, a federated scaler, local calibration splits, and per-round audits |
| `federated_calibration.py` | Exact federated histogram binning from integer counts; temperature scaling as a measurable approximation |
| `federated_fairness.py` | Equalised-odds threshold repair from pooled confusion counts only |
| `federated_explain.py` | Exact linear Shapley, signed top factor, baseline-robustness margin `R`, background-uncertainty `P(R > 0)`, split-half reliability, phase report |
| `federated_theory.py` | Pairwise `(a, c)` decomposition, concavity, half-space union bound, critical multiplier `α*` |
| `data_loader.py` | Dataset registry (PIMA and UCI Early Stage), age-band partitioning with explicit fallback metadata |
| `client_valuation.py` | Exact Shapley over all 8 client coalitions, per metric |
| `experiments.py` | 5 seeds × 15 rounds × 2 regimes, with 95% CIs; `--dataset` for external validation; E7 phase sweep |
| `paper_assets.py` | `summary.json` → Markdown/LaTeX conference tables, including phase and external tables |
| `tests/test_explain.py`, `tests/test_theory.py` | Unit tests for the signed top factor, margin, decomposition, concavity, union bound and phase statistics |

The model is intentionally simple — L2-regularised logistic regression. That choice keeps exact
Shapley and exact count-based calibration tractable, which is what makes the audit claims clean.
The contribution is the measurement, not model capacity.

---

## 3. The core idea, precisely

For a linear model, the exact Shapley value against background `b` is

```
phi_j(x) = w_j * (x_j - b_j)
```

The top factor is the largest **signed** contribution, `argmax_j phi_j`. This matters: a large
negative value is a strong *protective* factor, not the top risk factor — the original dashboard
used `argmax_j |phi_j|` and we corrected it.

Let `phi_g` be the explanation against the global cohort and `phi_l` against the local cohort.
The baseline shift is patient-independent:

```
Delta = phi_l - phi_g = w * (b_g - b_l)
```

The **robustness margin** is the local lead of the global leader:

```
R(x) = min_{j != i} (phi_l,i - phi_l,j)
     = min_{j != i} ((phi_g,i - phi_g,j) + (Delta_i - Delta_j))
```

- `R > 0`: the global top factor is still top under the local baseline.
- `R < 0`: it flipped.
- The old flip rate is exactly `P(R < 0)`. The margin is the continuous quantity; the flip rate
  is only its sign.

Because the local background is a finite-sample cohort mean, it carries uncertainty. Each client
contributes aggregate `(n, mean, covariance)`, so the sampling covariance of the mean is
`Sigma / n`. Drawing plausible cohort means gives a per-patient **`P(R > 0)`** — the probability
that the top factor survives re-estimating the local cohort mean. No patient row is involved.

---

## 4. Headline results

5 seeds × 15 rounds, 95% confidence intervals.

| | Age-cohort split | Random shards |
|---|---|---|
| Accuracy | 0.762 ± 0.044 | 0.768 ± 0.051 |
| ROC AUC | 0.837 ± 0.032 | 0.838 ± 0.028 |
| ECE, uncalibrated | 0.068 ± 0.020 | 0.091 ± 0.013 |
| Signed top-1 flip rate | **22.6% ± 5.0** | **4.9% ± 2.3** |
| Mean margin `R` | +0.337 ± 0.018 | +0.332 ± 0.052 |
| Mean `P(R > 0)` | **0.775 ± 0.043** | **0.925 ± 0.013** |
| Share robust at 95% | **0.690 ± 0.051** | **0.784 ± 0.021** |
| Split-half flip disagreement | 0.054 ± 0.005 | 0.027 ± 0.007 |

What the numbers say:

1. **Accuracy and calibration are blind to it.** The regimes are statistically indistinguishable
   on accuracy, and their ECE difference runs opposite to the explanation difference. This is the
   gap our diagnostic fills.
2. **The average hides the risk; the tail reveals it.** Mean `R` is nearly identical (+0.337 vs
   +0.332), but the age split has more patients near or below zero. The uncertainty-aware
   probability separates the regimes far more clearly than the mean — a good case for reporting
   the whole distribution, not one number.
3. **The diagnostic is reliable enough to quote.** Split-half disagreement is smaller than the
   between-regime difference, and the across-seed CIs do not overlap for the flip rate or
   `P(R > 0)`. Individual patient statuses stay probabilistic, which is the honest and useful
   thing to report.

One practical negative result: reweighting FedAvg toward the small high-prevalence cohort
(`uniform` or `positive`) buys about 1.2 accuracy points, but ECE worsens (0.068 → 0.082–0.092)
and the equal-opportunity gap stays flat (0.307 / 0.317 / 0.310). So explanation fragility is not
repaired by the obvious aggregation knob — useful evidence that the audit is measuring something
distinct.

---

## 5. What is new here

Standard building blocks: the linear Shapley formula, federated averaging, equalised-odds
thresholds, and exact coalition Shapley with three clients.

What this project adds:

1. A **continuous baseline-robustness margin** for cohort-relative explanations, with the flip
   rate as its sign rather than the headline.
2. **Background uncertainty from aggregate sufficient statistics**, giving per-patient
   `P(R > 0)` without centralising records.
3. A **validation protocol for the diagnostic itself**: bootstrap CI, split-half test–retest,
   per-cohort breakdown, and across-seed intervals.
4. A clean **empirical separation** of explanation fragility from accuracy, calibration and
   aggregate fairness under a clinically motivated non-IID partition.

Together that is a measurement and audit contribution: a practical tool with a validated
reliability story, not another optimiser.

---

## 6. Scope and honest boundaries

- It is **not** a new federated optimisation method.
- It does **not** claim a privacy guarantee: coefficients and aggregate counts are exchanged in
  the clear.
- It is **not** clinical validation: no clinician adjudication, no causal claim, no external
  cohort.
- It is **not** a general fairness result: PIMA is one population, all female, single site, and the
  subgroups are age cohorts rather than demographic categories.
- It does **not** claim the global or local baseline is "correct". It measures sensitivity to a
  legitimate modelling choice — which is exactly the point.

---

## 7. Questions you are likely to ask

**"Isn't this just SHAP with a different background?"**
The attribution itself is standard — and that is a strength, because reviewers already trust it.
The contribution is measuring how much the attribution depends on the background choice, and
validating that measurement.

**"Why is the mean margin the same but the flip rate so different?"**
The flip rate is a tail event. The middle of the distribution is similar; the age split simply has
more patients whose leader is barely ahead. The robustness curve and lower tail show this clearly
— a strong argument for reporting distributions, not just means.

**"Is `P(R > 0)` a real probability?"**
It is a probability under a stated Gaussian model of cohort-mean uncertainty, with model weights
held fixed. We present it as a sensitivity check, not a posterior predictive probability, and we
will say that explicitly.

**"Why only three clients and a linear model?"**
Exact Shapley over all coalitions is tractable at three clients, and exact count-based calibration
is tractable for a linear model. Those exactness properties are what make the audit claims clean.
Scaling to more clients and a neural model is a natural next step and would use approximations.

**"Is the diagnostic validated?"**
Internally, yes: bootstrap CI, split-half test–retest, five-seed spread, and per-cohort breakdown.
Externally, not yet. A second dataset or held-out site is the clear next step.

**"What is the clinical action?"**
The diagnostic tells a clinician or auditor when an explanation reflects the patient versus the
cohort definition. It is a governance and audit tool, not a treatment recommendation.

---

## 8. What would help most

1. **Framing:** I favour the measurement/audit framing — it is honest, distinctive, and more
   likely to survive review. Would you position it as trustworthy-ML methods instead?
2. **Venue:** I am considering a 4–6 page IEEE ICHI/BHI/CBMS short paper. Is there a better fit
   in the group's portfolio?
3. **Scope:** is one dataset enough for a short paper if the measurement validation is strong, or
   should we add a second public dataset first?
4. **Related work:** which explanation-stability or federated-audit papers should I cite so we
   build on the known metrics rather than reinvent them?
5. **Weakest claim:** if reviewers push back, which of the four contributions should I cut first?

---

## 9. Verify in five minutes

```powershell
# 1. Tests for the signed top factor and margin
./venv/Scripts/python.exe -m unittest discover -s tests -v

# 2. Reproduce the headline numbers (≈ 3–5 min)
./venv/Scripts/python.exe experiments.py --seeds 5 --rounds 15

# 3. Generate the conference tables
./venv/Scripts/python.exe paper_assets.py --results results --latex

# 4. Open the app
./venv/Scripts/python.exe app.py
# http://127.0.0.1:5000/  (demo@fedhealth.local / demo12345 after seed_demo.py)
```

The dashboard's **Explanations** tab shows the margin, the robustness curve and `P(R > 0)`. The
patient form shows the same diagnostic for one patient, including whether the top factor changes
between the two baselines.

---

## 10. One-slide summary

> **Opportunity:** federated clinical explanations depend on which cohort is the reference, and
> standard metrics do not see it.
> **Method:** signed top factor → continuous margin `R` → `P(R > 0)` under aggregate background
> uncertainty.
> **Result:** 22.6% vs 4.9% signed flips (age-cohort vs IID) with indistinguishable accuracy —
> and reweighting does not repair it.
> **Confidence:** bootstrap CI, split-half test–retest, five-seed interval, per-cohort breakdown.
> **Contribution:** a validated audit diagnostic, not a new model. **Next:** a second dataset and
> external explanation validation.
