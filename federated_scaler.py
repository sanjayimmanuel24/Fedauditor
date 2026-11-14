"""Federated standardisation from per-client sufficient statistics.

The previous pipeline fit a single ``sklearn`` ``StandardScaler`` on the pooled
training frame. That is statistically correct but contradicts the privacy claim:
the server was implicitly handed every client's column means and variances, so
"only model weights and aggregate counts leave a hospital" was not literally
true.

Column standardisation only needs ``(n, sum, sum of squares)`` per feature, and
those are additive across clients. Summing them reproduces the pooled mean and
population variance exactly, so the arithmetic and the privacy story now agree.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def local_sufficient_statistics(X) -> Dict[str, list]:
    """What a client computes over its own rows and nothing else."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("expected a 2-D feature matrix")
    return {
        "n": int(X.shape[0]),
        "sum": X.sum(axis=0).tolist(),
        "sumsq": (X**2).sum(axis=0).tolist(),
    }


def pool_sufficient_statistics(updates: List[Dict[str, list]]) -> Dict[str, object]:
    """Server-side aggregation of client statistics."""
    if not updates:
        raise ValueError("no client statistics supplied")

    n = 0
    total = None
    total_sq = None
    width = None

    for update in updates:
        part_sum = np.asarray(update["sum"], dtype=float)
        part_sq = np.asarray(update["sumsq"], dtype=float)
        if width is None:
            width = part_sum.shape[0]
        elif part_sum.shape[0] != width:
            raise ValueError("clients reported inconsistent feature counts")

        n += int(update["n"])
        total = part_sum if total is None else total + part_sum
        total_sq = part_sq if total_sq is None else total_sq + part_sq

    return {"n": n, "sum": total.tolist(), "sumsq": total_sq.tolist()}


class FederatedScaler:
    """Drop-in replacement for ``StandardScaler`` with federated fitting.

    Matches ``sklearn`` semantics: population variance (``ddof=0``) and a unit
    scale for constant columns.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.var_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_samples_seen_: int = 0

    def fit_from_statistics(self, pooled: Dict[str, object]) -> "FederatedScaler":
        n = int(pooled["n"])
        if n <= 0:
            raise ValueError("cannot fit a scaler on zero samples")

        total = np.asarray(pooled["sum"], dtype=float)
        total_sq = np.asarray(pooled["sumsq"], dtype=float)

        mean = total / n
        # Guard against tiny negative values from floating-point cancellation.
        var = np.maximum(total_sq / n - mean**2, 0.0)

        self.mean_ = mean
        self.var_ = var
        self.scale_ = np.where(var > 0.0, np.sqrt(var), 1.0)
        self.n_samples_seen_ = n
        return self

    def fit(self, X) -> "FederatedScaler":
        return self.fit_from_statistics(local_sufficient_statistics(X))

    def transform(self, X) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("scaler is not fitted")
        return np.asarray(Z, dtype=float) * self.scale_ + self.mean_

    def to_dict(self) -> dict:
        mean = [] if self.mean_ is None else self.mean_
        scale = [] if self.scale_ is None else self.scale_
        return {
            "fit": "federated sufficient statistics",
            "n_samples": int(self.n_samples_seen_),
            "mean": [round(float(v), 6) for v in mean],
            "scale": [round(float(v), 6) for v in scale],
        }
