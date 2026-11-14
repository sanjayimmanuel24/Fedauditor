# Data

This folder contains the two public datasets used by the project. No patient
records are uploaded by the application; these files are read locally by the
data loader.

## Primary dataset — PIMA Indians Diabetes

- File: `pima_diabetes.csv`
- Records: 768
- Features: 8 (`Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`,
  `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`)
- Target: `Outcome` (0/1)
- Source: UCI Machine Learning Repository, PIMA Indians Diabetes Database
- Citation: Smith et al., PIMA Indians Diabetes Database (UCI)

The standard preprocessing convention is used: a recorded zero in `Glucose`,
`BloodPressure`, `SkinThickness`, `Insulin`, or `BMI` is physiologically
impossible and is treated as missing. Imputation medians are fitted on the
training split only.

If this file is missing, the loader falls back to a documented synthetic
generator so the app can still boot. Results from that mode are labelled
`SYNTHETIC FALLBACK` and must not be reported as findings.

## External-validation dataset — UCI Early Stage Diabetes Risk Prediction

- File: `uci_early_diabetes.csv`
- Records: 520
- Features: age, sex, and 14 binary symptom items
- Target: `class` (`Positive` / `Negative`)
- Source: UCI Machine Learning Repository, dataset 529
- Citation: S. Islam et al., Early Stage Diabetes Risk Prediction Dataset
  (UCI Machine Learning Repository, 2020)

This is a different questionnaire, not a second split of PIMA. It is used to
test whether the audit is computable and directionally consistent on a second
public instrument. The loader never replaces this dataset with synthetic data.

## Licensing and provenance

Both datasets are public research datasets distributed through the UCI Machine
Learning Repository. Check the UCI pages for the authoritative license and
citation requirements before redistributing the CSV files. The repository's MIT
license covers the code, not the datasets.

PIMA is a single-population, all-female cohort. Age-cohort subgroup results are
not demographic fairness claims. Screening output is not a diagnosis.
