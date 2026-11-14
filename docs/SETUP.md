# Setup and reproduction guide

This guide covers installation, local configuration, running the Flask app,
reproducing the experiments, and troubleshooting.

## Requirements

- Python 3.12 (tested on 3.12.10)
- Runtime dependencies: Flask, Werkzeug, numpy, pandas, scikit-learn, python-dotenv
- No GPU, TensorFlow, PyTorch, Flower, or Qiskit is required

## 1. Create and activate a virtual environment

From the repository root:

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
```

Command Prompt:

```bat
venv\Scripts\activate.bat
```

If PowerShell blocks the activation script, allow it for the current shell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 2. Install dependencies

```powershell
pip install -r requirements.txt
```

## 3. Configure local environment variables

Copy the example file and edit it locally:

```powershell
Copy-Item .env.example .env
```

Generate a session secret:

```powershell
python -c "import secrets; print(secrets.token_hex(32))"
```

Put the value in `.env`:

```dotenv
PORT=5000
FLASK_DEBUG=0
FEDHEALTH_SECRET_KEY=your-generated-value
```

`.env` is ignored by git. Only `.env.example` is committed. If
`FEDHEALTH_SECRET_KEY` is missing, the app generates a process-local key, which
means sessions reset when the process restarts.

## 4. Create a login

There is no built-in account. Either sign up at `/signup`, or create the demo
account:

```powershell
python seed_demo.py
```

Demo credentials:

```text
email:    demo@fedhealth.local
password: demo12345
```

`seed_demo.py` is idempotent. Use `--reset` to overwrite the password, or pass
`--email` / `--password` to choose your own. The script verifies the credentials
before reporting success.

## 5. Run the app

```powershell
python app.py
```

Open <http://127.0.0.1:5000/>.

Environment overrides:

| Variable | Default | Meaning |
|---|---|---|
| `PORT` | `5000` | HTTP port |
| `FLASK_DEBUG` | `0` | Set to `1` to enable the Flask reloader |
| `FEDHEALTH_SECRET_KEY` | random per process | Session signing key |

## 6. First run

1. Log in at `/login`.
2. The dashboard opens with five warm-up rounds already executed.
3. **Trigger FedAvg Round** runs one more communication round.
4. **Cold Start** begins a fresh run with zero weights and zero rounds. The
   previous run remains in SQLite and remains exportable.
5. **Export CSV** downloads the per-round metric history of the current run.

Dashboard tabs:

| Tab | What it shows |
|---|---|
| Performance | Accuracy, AUC, ECE per round; per-client cards; federated vs centralised comparison |
| Calibration | Reliability diagram and six-method comparison |
| Subgroup Fairness | Per-cohort TPR before/after repair and the cost of repair |
| Explanations | Baseline-robustness margin, signed top-1 flip rate, `P(R > 0)`, robustness curve |
| Client Value | Exact Shapley client valuation by metric |

The patient form at `/predict` scores one patient and shows the exact linear
Shapley attribution. The **Explain against** toggle compares global-cohort and
local-cohort explanations for the same patient. The robustness panel reports
`R` and `P(R > 0)`; this is an explanation-stability diagnostic, not a clinical
risk score.

## 7. Reproduce the experiments

Quick run (5 seeds, 15 rounds, both partition regimes):

```powershell
python experiments.py --seeds 5 --rounds 15
```

Longer run:

```powershell
python experiments.py --seeds 10 --rounds 25
```

Age-cohort partition only:

```powershell
python experiments.py --partitions age --seeds 10 --out results_age
```

Choose an aggregation rule:

```powershell
python experiments.py --weighting uniform
python experiments.py --weighting positive
python experiments.py --skip-weighting
```

External validation on the UCI dataset:

```powershell
python experiments.py --dataset uci_early --seeds 5 --rounds 15 --skip-weighting --out results_second
```

Outputs land in `results/` (or the folder passed to `--out`):

| File | Contents |
|---|---|
| `e0_centralised.csv` | Pooled logistic-regression reference |
| `e1_rounds.csv` | Metrics per round |
| `e2_calibration.csv` | Calibration methods per round |
| `e3_fairness.csv` | Per-cohort rates before/after repair |
| `e4_explanations.csv` | Robustness margin, `P(R > 0)`, flip rate, split-half reliability |
| `e5_valuation.csv` | Shapley value per client per metric |
| `e5_rank_agreement.csv` | Kendall tau between metric rankings |
| `e6_weighting.csv` | Sample vs uniform vs positive weighting |
| `e7_phase_sensitivity.csv` | Baseline-multiplier sweep |
| `summary.json` | Means with 95% confidence intervals |

A full 5-seed run (both regimes, all three weightings) takes roughly 4–6 minutes
on a laptop CPU.

## 8. Build paper tables

```powershell
python paper_assets.py --results results --external results_second --latex
```

This writes `results/paper_tables.md` and `results/paper_tables.tex`. Every cell
is read from `summary.json`; nothing is typed in by hand.

## 9. Run the tests

```powershell
./venv/Scripts/python.exe -m unittest discover -s tests -v
```

## 10. Reset state

Results and accounts are stored in the SQLite database `users.db`. The file is
runtime state and is ignored by git. To wipe accounts and run history, delete
`users.db` and restart the app; the schema is recreated automatically. Recreate
the demo account with:

```powershell
python seed_demo.py
```

Legacy notebook artifacts live in `notebooks/artifacts/`. They are not read by
the current engine.

## 11. Troubleshooting

**`ModuleNotFoundError: No module named 'sklearn'`**

You are using the system Python, not the virtual environment. Activate the venv
or call `./venv/Scripts/python.exe` explicitly.

**`ModuleNotFoundError: No module named 'flask'`**

Dependencies were installed into a different interpreter:

```powershell
./venv/Scripts/python.exe -m pip install -r requirements.txt
```

**Dashboard shows `SYNTHETIC FALLBACK`**

`data/pima_diabetes.csv` is missing. Restore it before collecting results.

**Port 5000 is already in use**

```powershell
$env:PORT=8000
python app.py
```

**Login does not persist across restarts**

Set `FEDHEALTH_SECRET_KEY` in `.env`. Without it, the app generates a fresh
process-local key on every start.

## 12. Honest scope

This project does **not** implement differential privacy, secure aggregation, or
encrypted weight exchange. Model coefficients and aggregate counts are exchanged
in the clear. The correct claim is "patient records are never centralised", not
"privacy is guaranteed".

What leaves a client: weights, per-bin integer counts, per-cohort confusion
counts, `(n, sum, sum-of-squares)` per feature for standardisation, and
`(n, mean, covariance)` per cohort for background uncertainty. No row is ever
uploaded.

Fairness thresholds are selected on the clients' local calibration splits and
scored on the untouched test set. Those splits are small, so the repaired gap
has a wide interval.

The centralised reference is unregularised and converged while FedAvg is not, so
the reported accuracy gap is an upper bound on the cost of federation rather than
a like-for-like comparison.

PIMA is a single-population, all-female cohort. Subgroup results here are
age-cohort results, not demographic fairness claims. Screening output is not a
diagnosis.
