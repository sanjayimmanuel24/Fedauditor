"""Subgroup fairness auditing and repair from pooled confusion counts.

Every client reports only ``(group, threshold, TP, FP, FN, TN)`` integers. The
server sums them, so the audit and the repair both run without any record ever
leaving a hospital.

Repair is the standard equalised-odds post-processing idea: choose a separate
decision threshold per subgroup so that true-positive rates line up. The cost
of that choice (accuracy, calibration) is reported alongside the benefit.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from config import AGE_BANDS
from federated_metrics import binarize, expected_calibration_error, full_metrics


def threshold_grid(steps: int = 19) -> np.ndarray:
    return np.linspace(0.05, 0.95, steps)


def ordered_groups(groups: Iterable) -> List[str]:
    present = {str(g) for g in np.asarray(list(groups)).tolist()}
    order = [label for label, _, _ in AGE_BANDS if label in present]
    extras = sorted(present - set(order))
    return order + extras


def group_counts_over_grid(
    y: np.ndarray,
    p: np.ndarray,
    groups: Iterable,
    grid: np.ndarray,
) -> Dict[str, Dict[str, list]]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray([str(g) for g in groups])

    table: Dict[str, Dict[str, list]] = {}
    for group in ordered_groups(groups):
        mask = groups == group
        yy, pp = y[mask], p[mask]
        tp, fp, fn, tn = [], [], [], []
        for threshold in grid:
            y_hat = (pp >= threshold).astype(int)
            tp.append(int(np.sum((yy == 1) & (y_hat == 1))))
            fp.append(int(np.sum((yy == 0) & (y_hat == 1))))
            fn.append(int(np.sum((yy == 1) & (y_hat == 0))))
            tn.append(int(np.sum((yy == 0) & (y_hat == 0))))
        table[group] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return table


def pool_group_tables(
    tables: List[Dict[str, Dict[str, list]]]
) -> Dict[str, Dict[str, list]]:
    pooled: Dict[str, Dict[str, list]] = {}
    for table in tables:
        for group, counts in table.items():
            if group not in pooled:
                pooled[group] = {
                    key: [0] * len(value) for key, value in counts.items()
                }
            for key in ("tp", "fp", "fn", "tn"):
                pooled[group][key] = [
                    a + b for a, b in zip(pooled[group][key], counts[key])
                ]
    return pooled


def rates(counts: Dict[str, list]) -> Dict[str, np.ndarray]:
    tp = np.asarray(counts["tp"], dtype=float)
    fp = np.asarray(counts["fp"], dtype=float)
    fn = np.asarray(counts["fn"], dtype=float)
    tn = np.asarray(counts["tn"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpr = np.where(tp + fn > 0, tp / (tp + fn), np.nan)
        fpr = np.where(fp + tn > 0, fp / (fp + tn), np.nan)
        ppv = np.where(tp + fp > 0, tp / (tp + fp), np.nan)
    return {"tpr": tpr, "fpr": fpr, "ppv": ppv}


def _pooled_rates(pooled: Dict[str, Dict[str, list]]) -> Dict[str, np.ndarray]:
    totals = {
        key: np.sum(
            [np.asarray(value[key], dtype=float) for value in pooled.values()], axis=0
        )
        for key in ("tp", "fp", "fn", "tn")
    }
    return rates(totals)


def equalized_odds_thresholds(
    pooled: Dict[str, Dict[str, list]],
    grid: np.ndarray,
    reference_threshold: float = 0.5,
) -> Dict[str, float]:
    """Pick a per-group threshold whose TPR matches the pooled reference TPR."""
    reference_index = int(np.argmin(np.abs(grid - reference_threshold)))
    pooled_rates = _pooled_rates(pooled)
    target_tpr = float(pooled_rates["tpr"][reference_index])
    target_fpr = float(pooled_rates["fpr"][reference_index])

    if np.isnan(target_tpr):
        target_tpr = 1.0
    if np.isnan(target_fpr):
        target_fpr = 0.0

    thresholds: Dict[str, float] = {}
    for group, counts in pooled.items():
        group_rates = rates(counts)
        tpr = np.nan_to_num(group_rates["tpr"], nan=0.0)
        fpr = np.nan_to_num(group_rates["fpr"], nan=0.0)

        # Primary objective: match TPR. Secondary: keep FPR close.
        # Tertiary: stay near the default threshold so we do not choose absurd
        # cut-offs purely to win a tie.
        cost = (
            np.abs(tpr - target_tpr)
            + 0.5 * np.abs(fpr - target_fpr)
            + 0.02 * np.abs(grid - reference_threshold)
        )
        thresholds[group] = float(grid[int(np.argmin(cost))])

    return thresholds


def apply_group_thresholds(
    p: np.ndarray, groups: Iterable, thresholds: Dict[str, float]
) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    groups = np.asarray([str(g) for g in groups])
    y_hat = np.zeros_like(p, dtype=int)
    fallback = float(np.median(list(thresholds.values()))) if thresholds else 0.5
    for group, threshold in thresholds.items():
        mask = groups == group
        y_hat[mask] = (p[mask] >= threshold).astype(int)
    unassigned = ~np.isin(groups, list(thresholds.keys()))
    if np.any(unassigned):
        y_hat[unassigned] = (p[unassigned] >= fallback).astype(int)
    return y_hat


def _group_block(
    y: np.ndarray, p: np.ndarray, groups: np.ndarray, y_hat: np.ndarray
) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for group in ordered_groups(groups):
        mask = groups == group
        yy = y[mask]
        pp = p[mask]
        yh = y_hat[mask]
        tp = int(np.sum((yy == 1) & (yh == 1)))
        fp = int(np.sum((yy == 0) & (yh == 1)))
        fn = int(np.sum((yy == 1) & (yh == 0)))
        tn = int(np.sum((yy == 0) & (yh == 0)))
        out[group] = {
            "n": int(mask.sum()),
            "positives": int(yy.sum()),
            "prevalence": round(float(yy.mean()), 4) if len(yy) else 0.0,
            "predicted_positive_rate": round(float(yh.mean()), 4) if len(yh) else 0.0,
            "tpr": round(tp / (tp + fn), 4) if (tp + fn) else None,
            "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "ppv": round(tp / (tp + fp), 4) if (tp + fp) else None,
            "ece": round(expected_calibration_error(yy, pp), 4) if len(yy) else None,
        }
    return out


def _gaps(block: Dict[str, dict]) -> Dict[str, float]:
    def spread(key):
        values = [v[key] for v in block.values() if v[key] is not None]
        return round(float(max(values) - min(values)), 4) if len(values) >= 2 else 0.0

    return {
        "equal_opportunity": spread("tpr"),
        "equalized_odds_fpr": spread("fpr"),
        "demographic_parity": spread("predicted_positive_rate"),
        "calibration": spread("ece"),
    }


def fairness_report(
    y: np.ndarray,
    p: np.ndarray,
    groups: Iterable,
    grid: np.ndarray | None = None,
    default_threshold: float = 0.5,
    tuning_counts: Dict[str, Dict[str, list]] | None = None,
) -> Dict[str, object]:
    """Audit subgroup error rates, optionally repairing on a separate split.

    ``tuning_counts`` is a pooled table of ``(group, threshold, TP, FP, FN, TN)``
    integers gathered from the clients' local calibration splits. When supplied,
    per-cohort thresholds are selected on that data and scored on ``y``/``p``
    here, which removes the optimistic bias of tuning and evaluating on the same
    rows. When omitted, thresholds fall back to the evaluation set and the
    report says so through ``tuning_source``.
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    groups = np.asarray([str(g) for g in groups])
    grid = threshold_grid() if grid is None else np.asarray(grid)

    counts = group_counts_over_grid(y, p, groups, grid)

    baseline_hat = binarize(p, default_threshold)
    baseline = _group_block(y, p, groups, baseline_hat)
    baseline_gaps = _gaps(baseline)

    if tuning_counts is None:
        selection_counts = counts
        tuning_source = "evaluation_set"
    else:
        selection_counts = tuning_counts
        tuning_source = "client_calibration"

    thresholds = equalized_odds_thresholds(selection_counts, grid, default_threshold)
    repaired_hat = apply_group_thresholds(p, groups, thresholds)
    repaired = _group_block(y, p, groups, repaired_hat)
    repaired_gaps = _gaps(repaired)

    baseline_metrics = full_metrics(y, p, default_threshold)
    repaired_metrics = {
        "accuracy": round(float(np.mean(repaired_hat == y)), 4),
        "ece": round(expected_calibration_error(y, p), 4),
        "recall": round(
            float(
                np.sum((y == 1) & (repaired_hat == 1))
                / max(np.sum(y == 1), 1)
            ),
            4,
        ),
    }

    frontier = []
    for threshold in grid:
        y_hat = binarize(p, threshold)
        block = _group_block(y, p, groups, y_hat)
        gaps = _gaps(block)
        frontier.append(
            {
                "threshold": round(float(threshold), 3),
                "accuracy": round(float(np.mean(y_hat == y)), 4),
                "equal_opportunity": gaps["equal_opportunity"],
                "demographic_parity": gaps["demographic_parity"],
            }
        )

    reference_index = int(np.argmin(np.abs(grid - default_threshold)))
    tuning_rates = _pooled_rates(selection_counts)
    eval_rates = _pooled_rates(counts)

    return {
        "group_order": ordered_groups(groups),
        "threshold_grid": [round(float(t), 3) for t in grid],
        "tuning_source": tuning_source,
        "tuning_reference_tpr": round(float(tuning_rates["tpr"][reference_index]), 4),
        "eval_reference_tpr": round(float(eval_rates["tpr"][reference_index]), 4),
        "baseline": baseline,
        "baseline_gaps": baseline_gaps,
        "repaired": repaired,
        "repaired_gaps": repaired_gaps,
        "repaired_thresholds": {k: round(float(v), 3) for k, v in thresholds.items()},
        "cost": {
            "accuracy_before": round(baseline_metrics["accuracy"], 4),
            "accuracy_after": repaired_metrics["accuracy"],
            "delta_accuracy": round(
                repaired_metrics["accuracy"] - baseline_metrics["accuracy"], 4
            ),
            "eo_gap_before": baseline_gaps["equal_opportunity"],
            "eo_gap_after": repaired_gaps["equal_opportunity"],
            "delta_eo_gap": round(
                repaired_gaps["equal_opportunity"]
                - baseline_gaps["equal_opportunity"],
                4,
            ),
        },
        "frontier": frontier,
        "client_counts": counts,
        "default_threshold": default_threshold,
    }
