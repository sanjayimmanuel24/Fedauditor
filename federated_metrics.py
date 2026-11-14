"""Metrics used by the clinical-depth experiments.

Accuracy is deliberately not treated as the headline number. Calibration
(Brier, NLL, ECE, MCE), discrimination (ROC AUC), and subgroup-conditional
rates are first-class citizens here.
"""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-12


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def binarize(probabilities: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probabilities) >= threshold).astype(int)


def safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y) ** 2))


def negative_log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def confusion_counts(y: np.ndarray, y_hat: np.ndarray) -> Dict[str, int]:
    y = np.asarray(y).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    tp = int(np.sum((y == 1) & (y_hat == 1)))
    fp = int(np.sum((y == 0) & (y_hat == 1)))
    fn = int(np.sum((y == 1) & (y_hat == 0)))
    tn = int(np.sum((y == 0) & (y_hat == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(counts: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def uniform_edges(n_bins: int = 10) -> np.ndarray:
    """Fixed equal-width bin edges on [0, 1].

    Fixed edges (rather than quantile edges) are what make federated histogram
    binning *exactly* equivalent to centralised binning: every client bins on
    the same constants, so summing per-bin counts is lossless.
    """
    return np.linspace(0.0, 1.0, n_bins + 1)


def reliability_curve(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    edges: np.ndarray | None = None,
) -> Dict[str, list]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = uniform_edges(n_bins) if edges is None else np.asarray(edges)
    n_bins = len(edges) - 1

    bin_index = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    counts, confidence, accuracy = [], [], []
    for b in range(n_bins):
        mask = bin_index == b
        n_b = int(mask.sum())
        counts.append(n_b)
        confidence.append(float(p[mask].mean()) if n_b else 0.0)
        accuracy.append(float(y[mask].mean()) if n_b else 0.0)

    counts = np.asarray(counts)
    confidence = np.asarray(confidence)
    accuracy = np.asarray(accuracy)
    gaps = np.where(counts > 0, np.abs(accuracy - confidence), 0.0)

    return {
        "edges": [round(float(e), 4) for e in edges],
        "counts": counts.tolist(),
        "confidence": [round(float(c), 4) for c in confidence],
        "accuracy": [round(float(a), 4) for a in accuracy],
        "gaps": [round(float(g), 4) for g in gaps],
    }


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    curve = reliability_curve(y, p, n_bins)
    counts = np.asarray(curve["counts"], dtype=float)
    gaps = np.asarray(curve["gaps"], dtype=float)
    total = counts.sum()
    if total == 0:
        return 0.0
    return float(np.sum(counts / total * gaps))


def maximum_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10
) -> float:
    curve = reliability_curve(y, p, n_bins)
    counts = np.asarray(curve["counts"], dtype=float)
    gaps = np.asarray(curve["gaps"], dtype=float)
    if not np.any(counts > 0):
        return 0.0
    return float(np.max(gaps[counts > 0]))


def full_metrics(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float = 0.5,
    n_bins: int = 10,
) -> Dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    counts = confusion_counts(y, binarize(p, threshold))
    prf = precision_recall_f1(counts)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean(binarize(p, threshold) == y)),
        "auc": safe_auc(y, p),
        "brier": brier_score(y, p),
        "nll": negative_log_likelihood(y, p),
        "ece": expected_calibration_error(y, p, n_bins),
        "mce": maximum_calibration_error(y, p, n_bins),
        "precision": prf["precision"],
        "recall": prf["recall"],
        "f1": prf["f1"],
        **counts,
    }


def group_metrics(
    y: np.ndarray,
    p: np.ndarray,
    groups: Iterable,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    groups = np.asarray(list(groups))
    out: Dict[str, Dict[str, float]] = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        if mask.sum() == 0:
            continue
        out[str(group)] = full_metrics(y[mask], p[mask], threshold)
        out[str(group)]["positives"] = int(y[mask].sum())
        out[str(group)]["prevalence"] = float(y[mask].mean())
    return out


def tpr_fpr(counts: Dict[str, int]) -> tuple[float, float]:
    tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return tpr, fpr


def equal_opportunity_gap(
    y: np.ndarray, p: np.ndarray, groups: Iterable, threshold: float = 0.5
) -> float:
    per_group = group_metrics(y, p, groups, threshold)
    tprs = [v["tp"] / (v["tp"] + v["fn"]) for v in per_group.values() if (v["tp"] + v["fn"])]
    if len(tprs) < 2:
        return 0.0
    return float(max(tprs) - min(tprs))


def demographic_parity_gap(
    y: np.ndarray, p: np.ndarray, groups: Iterable, threshold: float = 0.5
) -> float:
    groups = np.asarray(list(groups))
    p = np.asarray(p, dtype=float)
    y_hat = binarize(p, threshold)
    rates = [float(y_hat[groups == g].mean()) for g in sorted(set(groups.tolist()))]
    if len(rates) < 2:
        return 0.0
    return float(max(rates) - min(rates))


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metric: str = "ece",
    n_boot: int = 300,
    alpha: float = 0.05,
    seed: int = 0,
    threshold: float = 0.5,
) -> tuple[float, float]:
    """Percentile bootstrap CI. Essential at n~154, where ECE is noisy."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)

    def compute(yy, pp):
        if metric == "ece":
            return expected_calibration_error(yy, pp)
        if metric == "brier":
            return brier_score(yy, pp)
        if metric == "nll":
            return negative_log_likelihood(yy, pp)
        if metric == "accuracy":
            return float(np.mean(binarize(pp, threshold) == yy))
        if metric == "auc":
            return safe_auc(yy, pp)
        raise ValueError(f"unsupported metric {metric}")

    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2 and metric == "auc":
            continue
        stats.append(compute(yy, pp))

    if not stats:
        return float("nan"), float("nan")
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi
