# External-validation results

This folder holds the external-validation run on the UCI Early Stage Diabetes
Risk Prediction questionnaire. Regenerate it with:

```powershell
python experiments.py --dataset uci_early --seeds 5 --rounds 15 --skip-weighting --out results_second
```

The files have the same names and meaning as in `results/`. The external run
tests whether the audit is computable and directionally consistent on a second
public instrument; it is not a second split of PIMA and its feature importances
are not comparable with PIMA's.

Do not edit these files by hand.
