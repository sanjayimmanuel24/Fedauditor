# Primary results

This folder holds the reproducible outputs of `experiments.py` for the primary
PIMA dataset. Regenerate it with:

```powershell
python experiments.py --seeds 5 --rounds 15 --out results
```

| File | Contents |
|---|---|
| `e0_centralised.csv` | Pooled logistic-regression reference |
| `e1_rounds.csv` | Metrics per round and partition |
| `e2_calibration.csv` | Calibration methods per round |
| `e3_fairness.csv` | Per-cohort rates before/after repair |
| `e4_explanations.csv` | Robustness margin, `P(R > 0)`, flip rate, split-half reliability |
| `e5_valuation.csv` | Shapley value per client per metric |
| `e5_rank_agreement.csv` | Kendall tau between metric rankings |
| `e6_weighting.csv` | Sample vs uniform vs positive aggregation weighting |
| `e7_phase_sensitivity.csv` | Baseline-multiplier sweep for the phase diagram |
| `summary.json` | Means with 95% confidence intervals |
| `paper_tables.md` | Conference tables generated from `summary.json` |
| `paper_tables.tex` | LaTeX version of the tables |

Do not edit these files by hand. Every value is produced by the harness.
