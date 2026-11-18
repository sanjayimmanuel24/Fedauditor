# FedAuditor

**An AI agent that audits a federated clinical model without ever seeing patient data.**

In federated learning, hospitals or regions train one shared model without sharing their records.
That protects privacy, but it also means nobody can look at the data to check whether a site is
faulty. FedAuditor gives an LLM agent a small set of audit tools that return only noisy,
privacy-budgeted summary statistics. The agent screens the federation, finds faulty sites, names
the fault and when it started, and writes a report. Every number in that report must point to a
logged tool result, and an automatic fact-checker verifies it.

To measure this we built **FedFault**: 8 planted fault types, each at 3 severities over 5 seeds,
on a federation built from the CDC BRFSS 2023 diabetes survey, with each of the 52 US states
and territories as one client.

| Auditor (125 test scenarios, logistic regression) | Precision | Recall | Fault type correct | Claims verified | Cost |
|---|---|---|---|---|---|
| Rule baseline (same tools and budget) | 0.99 | 0.88 | 0.87 | 100% | free |
| Claude Haiku 5.5, prompt v5 | 0.97 | 0.89 | 0.91 | 99.9% | US$1.16 total |

The agent is much better at subgroup bias (finds 100% of affected sites at medium severity vs 60%
for the rules). Pooled AUC, the usual monitoring number, hides 7 of the 8 faults.

This project builds on **FedHealth**, a federated diabetes-screening system with calibration,
fairness and explanation measurements. Its original README is in
[`docs/FEDHEALTH_README.md`](docs/FEDHEALTH_README.md), and its measurement modules are reused as the
agent's tools.

---

## Setup

### 1. Requirements

- **Python 3.12** (3.10 or newer should work)
- **Git**
- About **1 GB** of free disk space (Python packages + BRFSS data)
- Optional: **LaTeX** (MiKTeX or TeX Live) to rebuild the paper PDF
- Optional: an **Anthropic API key**, only to run *new* LLM audits (all saved audits are included)

### 2. Get the code

```bash
git clone https://github.com/sanjayimmanuel24/Fedauditor.git
cd Fedauditor
```

### 3. Create a virtual environment and install packages

Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call
`.\venv\Scripts\python.exe` directly in the commands below.

### 4. Create your `.env` file

```powershell
Copy-Item .env.example .env          # Windows
```

```bash
cp .env.example .env                 # macOS / Linux
```

Then put a random value in `FEDHEALTH_SECRET_KEY` (generate one with
`python -c "import secrets; print(secrets.token_hex(32))"`). If you want to run new paid LLM
audits, also uncomment `ANTHROPIC_API_KEY` and add your key. **Never commit `.env`**; it is
already in `.gitignore`.

### 5. Download the BRFSS data (once)

```bash
python -m fedauditor.prepare_brfss --year 2023
```

This downloads the CDC's public BRFSS 2023 file (about 100 MB) and writes a recoded cache to
`data/brfss/` (not stored in git). It is needed for the walkthrough and for re-running the
federation, faults and audits. The web app, the tests and the paper work without it.

### 6. Create the demo login and start the app

```bash
python seed_demo.py
python app.py
```

Open <http://127.0.0.1:5000> and log in with **demo@fedhealth.local / demo12345**.

- **FedAuditor** (top menu, or <http://127.0.0.1:5000/fedauditor>): pick any saved audit; every
  claim in the report is a link to the exact logged tool output it cites.
- **Dashboard**: the original FedHealth measurement dashboard (PIMA data).

### 7. Check that everything works

```bash
python -m unittest discover -s tests
```

All 78 tests should pass (about 30 seconds, no data download or API key needed).

---

## Running the project

All commands run from the repository root with the virtual environment active.

| What | Command | Cost / time |
|---|---|---|
| Live demo of the whole pipeline (data, model, fault, tools, audit, fact-check) | `python -m fedauditor.walkthrough` | free, ~1 min |
| Clean federated baseline (LR, XGBoost, MLP) | `python -m fedauditor.run_baseline --dataset brfss` | free, a few min |
| FedFault benchmark (375 federated runs) | `python -m fedauditor.run_fedfault --dataset brfss --models lr mlp xgb --workers 4` | free, long |
| Rule-baseline audits on the grid | `python -m fedauditor.run_audit_grid --agents rules --models lr` | free |
| One LLM audit | `python -m fedauditor.run_audit --fault poisoning --severity low --llm claude-haiku-5-5` | **paid** (~1 cent with Haiku) |
| Compare auditors | `python -m fedauditor.compare_agents --model lr --labels rules claude-haiku-5-5 claude-haiku-5-5-p5 --conf 0 0.51` | free |
| Paper tables and figures | `python -m fedauditor.make_figures` | free |

Runners are resumable: finished scenarios are skipped. Saved outputs are already in
`results/fedauditor/`, so you only need to re-run something if you change the code.

### Rebuild the paper

```bash
cd paper
pdflatex fedauditor
bibtex fedauditor
pdflatex fedauditor
pdflatex fedauditor
```

Run `python -m fedauditor.make_figures` first if the results changed; the paper reads its tables
from `results/fedauditor/figures/`.

---

## What is in the repository

```
fedauditor/             FedAuditor package
  brfss.py, data.py       BRFSS download and recoding; per-state train/calibration/test splits
  federation.py, models.py  Flower FedAvg loop; logistic regression, MLP, XGBoost
  faults.py               FedFault: plants faults and records the ground truth
  audit_env.py, tools.py  8 privacy-budgeted audit tools, Laplace noise, privacy accountant
  agent.py, baseline.py   Claude LLM auditor and rule-based auditor
  openai_agent.py         Same auditor for open models (Ollama / vLLM)
  report.py               Report format, fact-checker and scoring
  run_*.py, compare_agents.py, make_figures.py, walkthrough.py   command-line entry points
app.py, templates/      Flask web app, including the FedAuditor report page
tests/                  Unit tests (78)
results/fedauditor/     Benchmark runs, all saved audits, comparison tables, figures
paper/                  6-page IEEE conference paper (LaTeX source + PDF)
docs/walkthrough/       Step-by-step project walkthrough (PDF)
docs/presentation/      Guide-review slides (PPTX)
notebooks/              Kaggle notebook for running open models (Qwen)
data/                   PIMA and UCI datasets; BRFSS cache goes in data/brfss/ (not in git)
```

## Troubleshooting

- **`ModuleNotFoundError`**: the virtual environment is not active, or `pip install -r
  requirements.txt` did not finish. Activate it and install again.
- **Login fails on a new machine**: the user database is not in git. Run `python seed_demo.py`.
- **Port 5000 is busy**: set `PORT=5001` in `.env` and open that port instead.
- **BRFSS download fails**: download `LLCP2023XPT.zip` from the CDC BRFSS 2023 page, unzip it and
  run `python -m fedauditor.prepare_brfss --year 2023 --xpt path/to/LLCP2023.XPT`.
- **`latexmk` complains about Perl** (MiKTeX): use the `pdflatex` / `bibtex` sequence above.

## Working on two machines

Run `git pull` before you start and `git push` when you finish. Keep your `.env` file on each
machine separately; it is never pushed.

## Licence

See [`LICENSE`](LICENSE). The BRFSS data is public and comes from the US Centers for Disease
Control and Prevention.
