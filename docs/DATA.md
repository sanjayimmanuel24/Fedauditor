# Data and provenance

## Dataset registry

| Property | Primary | External validation |
|---|---|---|
| Name | PIMA Indians Diabetes | UCI Early Stage Diabetes Risk Prediction |
| File | `data/pima_diabetes.csv` | `data/uci_early_diabetes.csv` |
| Records | 768 | 520 |
| Features | 8 clinical measurements | Age, sex, 14 binary symptom items |
| Target | `Outcome` (0/1) | `class` (`Positive` / `Negative`) |
| Clients | 3 age cohorts | 3 shifted age cohorts |
| Role | Primary audit | Different-instrument validation |

Both datasets are public research datasets from the UCI Machine Learning
Repository.

## PIMA Indians Diabetes

- Source: UCI Machine Learning Repository, PIMA Indians Diabetes Database
- Citation: Smith et al., PIMA Indians Diabetes Database (UCI)
- Feature columns: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- Target: `Outcome`

The standard convention is used: a recorded zero in `Glucose`,
`BloodPressure`, `SkinThickness`, `Insulin`, or `BMI` is physiologically
impossible and is treated as missing. Median imputation is fitted on the
training split only, so test rows cannot influence the fill value.

PIMA is a single-population, all-female cohort. Age-cohort subgroups are not
demographic fairness categories. Nothing in this repository supports a claim
about fairness by sex, ethnicity, or insurance status.

If `data/pima_diabetes.csv` is missing, the loader falls back to a documented
synthetic generator so the app can boot. The dashboard labels that mode
`SYNTHETIC FALLBACK`; do not report results from it.

## UCI Early Stage Diabetes Risk Prediction

- Source: UCI Machine Learning Repository, dataset 529
- Citation: S. Islam et al., Early Stage Diabetes Risk Prediction Dataset
  (UCI Machine Learning Repository, 2020)
- Inputs: age, sex, and 14 binary symptom items
- Target: `class` (`Positive` / `Negative`)

This is a different questionnaire, not a second split of PIMA. Its age
distribution is shifted upward, so the age bands are `<35`, `35–50`, and `50+`
rather than PIMA's `<30`, `30–45`, and `45+`. The loader never replaces this
dataset with synthetic data.

The external run asks whether the audit is computable and directionally
consistent on a second public instrument. It does not establish that the
clinical model transfers.

## Licensing

Both datasets are distributed publicly through the UCI Machine Learning
Repository. Check the authoritative UCI pages for license and citation terms
before redistributing the CSV files. The repository's MIT license covers the
code, not the datasets.

## What leaves a client

The application is federated in the sense that patient rows are never
centralised. Clients send:

- Model coefficients and bias, as in FedAvg.
- `(n, sum, sum-of-squares)` per feature for standardisation.
- Per-bin integer counts for calibration.
- Per-cohort, per-threshold confusion counts for fairness.
- Per-cohort `(n, mean, covariance)` for background uncertainty.

No patient row is uploaded. This is not a privacy guarantee: coefficients and
counts travel in the clear, and membership-inference or gradient-inversion
attacks remain live threats in this setting.
