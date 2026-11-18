# Beyond Accuracy in Federated Diabetes Screening

**A measurement study of calibration, subgroup fairness, and explanation robustness.**

A federated learning system for diabetes risk screening that treats **calibration, subgroup
fairness, explanation stability, and client valuation** as first-class outputs rather than
afterthoughts.

The base model is deliberately simple: L2-regularised logistic regression trained with
federated averaging over three simulated age-cohort clients. The contribution is not the model.
It is the measurement layer that shows what a respectable accuracy number hides.

The explanation layer reports a **continuous baseline-robustness diagnostic**. The flip rate is
still reported, but only as the sign of a margin; the margin itself, its uncertainty, and its
sampling reliability are the measurement.

## What this project measures

- **Federated calibration without a server dataset.** Fixed-edge histogram binning is exactly
  reproducible from per-bin integer counts.
- **Subgroup fairness from confusion counts only.** Equalised-odds repair works from integer
  counts and reports the accuracy cost.
- **Cohort-relative explanations.** For a linear model the Shapley value is closed form; the
  interesting variable is the background cohort. The project measures how far the top factor
  survives a change of baseline.
- **Client value depends on the metric.** With three clients, exact Shapley values are tractable
  for every subset.

Every claim below is implemented in code and covered by the experiment harness. Full methods and
results are in [`docs/METHODS.md`](METHODS.md) and [`docs/RESULTS.md`](RESULTS.md).

## Headline result

Five seeds × fifteen rounds × two partitioning regimes, with 95% confidence intervals:

| Metric | Age-cohort split (non-IID) | Random shards (IID) |
|---|---|---|
| Accuracy | 0.762 ± 0.044 | 0.768 ± 0.051 |
| Signed top-1 flip rate | **22.6% ± 5.0** | **4.9% ± 2.3** |
| Mean `P(R > 0)` | **0.775 ± 0.043** | **0.925 ± 0.013** |
| Median critical multiplier `α*` | **2.36 ± 0.73** | **20.62 ± 15.95** |

The two regimes are almost indistinguishable on accuracy, but they separate on explanation
stability. Accuracy, calibration, and the aggregate fairness gap do not see the difference that
matters.

## Quickstart

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_hex(32))"
# paste the value into .env as FEDHEALTH_SECRET_KEY

python seed_demo.py
python app.py
```

Open <http://127.0.0.1:5000/> and log in with `demo@fedhealth.local` / `demo12345`.

Full installation, environment, experiment, and troubleshooting instructions are in
[`docs/SETUP.md`](SETUP.md).

## Reproduce the results

```powershell
python experiments.py --seeds 5 --rounds 15 --out results
python experiments.py --dataset uci_early --seeds 5 --rounds 15 --skip-weighting --out results_second
python paper_assets.py --results results --external results_second --latex
```

The saved outputs are in [`results/`](../results/) and [`results_second/`](../results_second/). Every
cell in the paper tables is read from `summary.json`; nothing is typed by hand.

Run the measurement tests:

```powershell
./venv/Scripts/python.exe -m unittest discover -s tests -v
```

## Project structure

```text
.
├── app.py                     Flask routes, per-user RunSession, JSON serialisation
├── config.py                  paths, datasets, model and simulation settings
├── data_loader.py             dataset registry, loading, imputation, age-band partition
├── federated_scaler.py        federated mean/variance from additive statistics
├── federated_engine.py        FedAvg loop and per-round audits
├── federated_calibration.py   exact federated histogram binning
├── federated_fairness.py      subgroup counts and equalised-odds repair
├── federated_explain.py       linear Shapley, robustness margin, background uncertainty
├── federated_theory.py        half-space geometry and phase sensitivity
├── federated_metrics.py       accuracy, AUC, Brier, NLL, ECE, reliability, bootstrap CI
├── client_valuation.py        exact Shapley over client coalitions
├── experiment_store.py        SQLite persistence and CSV export
├── experiments.py             multi-seed experiment harness
├── paper_assets.py            summary.json -> Markdown/LaTeX tables
├── seed_demo.py               create a known demo login
├── data/                      public datasets and provenance notes
├── docs/                      setup, methods, results, limitations, paper notes
├── notebooks/                 exploratory notebook and legacy artifacts
├── results/                   primary experiment outputs
├── results_second/            external-validation outputs
├── static/                    CSS
├── templates/                 Flask templates and dashboard
└── tests/                     unit tests for explanations and theory
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/SETUP.md`](SETUP.md) | Install, `.env`, demo login, run, experiments, tests, troubleshooting |
| [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) | Module map, request flow, persistence, dashboard, privacy boundary |
| [`docs/METHODS.md`](METHODS.md) | Federated calibration, scaler, fairness, explanations, theory, valuation |
| [`docs/RESULTS.md`](RESULTS.md) | Main results, weighting ablation, external validation |
| [`docs/LIMITATIONS.md`](LIMITATIONS.md) | What must not be claimed, and why |
| [`docs/DATA.md`](DATA.md) | Dataset provenance, licensing, preprocessing, what leaves a client |
| [`docs/OVERVIEW.md`](OVERVIEW.md) | Full claims table and long-form project overview |
| [`docs/CHANGELOG.md`](CHANGELOG.md) | What changed from the original tutorial version |
| [`docs/PAPER_GUIDE.md`](PAPER_GUIDE.md) | Conference framing, venue plan, figure/table plan |
| [`docs/PAPER_PROMPT.md`](PAPER_PROMPT.md) | Strict prompt for drafting the paper from saved results |
| [`docs/PROFESSOR_BRIEF.md`](PROFESSOR_BRIEF.md) | One-page supervision briefing |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Local development, tests, style, PR checklist |
| [`data/README.md`](../data/README.md) | Dataset details and citations |
| [`results/README.md`](../results/README.md) | Primary output file index |
| [`results_second/README.md`](../results_second/README.md) | External-validation output index |
| [`notebooks/README.md`](../notebooks/README.md) | Exploratory notebook and legacy artifacts |

## Limitations

Read [`docs/LIMITATIONS.md`](LIMITATIONS.md) before quoting any number. The short version:

- No differential privacy, secure aggregation, or encryption. Records are not centralised;
  privacy is not guaranteed.
- Fairness thresholds are selected on small, distribution-shifted calibration splits.
- The centralised baseline is not compute-matched to FedAvg.
- The explanation diagnostic measures baseline stability, not correctness.
- External validation uses a different instrument, not a second PIMA split.
- PIMA is a single-population, all-female cohort. Age-cohort results are not demographic
  fairness claims.
- The model is linear. Exact Shapley and exact count-based calibration are properties of this
  model class.
- Screening output is not a diagnosis.

## License and citation

Code is released under the [MIT License](../LICENSE). The datasets are public UCI research datasets;
see [`docs/DATA.md`](DATA.md) for provenance and citation guidance.

If you use this work, see [`CITATION.cff`](../CITATION.cff).
