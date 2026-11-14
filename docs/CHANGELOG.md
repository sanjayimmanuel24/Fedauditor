# Changelog

The original repository was a FedAvg + PIMA + logistic-regression tutorial. It also asserted,
in the UI, encrypted weight exchange, zero data leakage and local differential privacy — none of
which existed in the code.

Fixed along the way:

- Data was regenerated with `np.random` and never actually read from disk.
- The client calibration split had `train_test_split`'s return values inverted (62 training
  rows, 143 calibration rows).
- The scaler was fit on synthetic data and applied to real data in the legacy notebook path.
- Simulation state was a module-level global shared by every logged-in user.
- "Reset" did not reset the weights.
- `requirements.txt` was a ~200-package environment dump including TensorFlow, PyTorch and
  Qiskit, none of which the project imports.
- Reliability, subgroup rates and Shapley values were never computed — only coefficient
  magnitudes were displayed and labelled as explanations.

Second pass (correctness of the chimera between "federated" and "centralised"):

- Median imputation was fitted on the whole frame before the split. It now uses
  `fit_medians` / `apply_medians` on training rows only.
- The scaler was a pooled `StandardScaler`, so the server implicitly saw every client's feature
  moments. It is now fitted from additive `(n, sum, sumsq)` updates in `federated_scaler.py`.
- Fairness thresholds were tuned and evaluated on the same test labels. They are now selected
  from pooled client-calibration confusion counts and scored on the test set.
- Added a centralised reference and an aggregation-weighting ablation, both reported by the
  harness and surfaced in the dashboard.

Third pass (explanation measurement):

- The top factor is signed. `argmax(|φ|)` could label a strong protective factor as the top
  risk factor; the corrected rule is `argmax(φ)`.
- The flip count is now the sign of a continuous baseline-robustness margin `R`, with a
  robustness curve and per-patient `P(R > 0)` under aggregate background uncertainty.
- The diagnostic is validated with bootstrap CIs, split-half test–retest, per-cohort breakdowns
  and five-seed intervals. `paper_assets.py` turns the summary into conference tables.

Fourth pass (geometry, phase sensitivity, external validation):

- The margin is decomposed into pairwise `(a, c)` terms. `federated_theory.py` derives the
  concavity, the interval stable set, the half-space union bound and the critical multiplier
  `α*`; the harness sweeps `α ∈ [0, 3]` into `e7_phase_sensitivity.csv`.
- The UCI Early Stage dataset is added as a second public instrument (`--dataset uci_early`,
  `results_second/`). It uses shifted age bands (`<35`, `35–50`, `50+`) and explicit
  `partition_fallback` / `partition_reason` metadata, so an age run cannot silently become IID.
- [`PAPER_PROMPT.md`](PAPER_PROMPT.md) records the paper-writing rules, including real-data-only, no privacy
  guarantee, no clinical claim, and the different-instrument framing for external validation.
