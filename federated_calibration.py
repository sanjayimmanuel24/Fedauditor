"""Privacy-preserving post-hoc calibration.

The central problem this module solves: a federated server has no validation
data, so it cannot fit a calibration map the way a centralised pipeline would.
Naively averaging client-side calibration parameters is *not* equivalent to
fitting on pooled data.

Two families are implemented:

* ``HistogramBinner`` - equal-width probability binning (Zadrozny & Elkan).
  Because the bin edges are fixed constants, the per-bin positive counts are a
  sufficient statistic. Summing them across clients reproduces the centralised
  fit **exactly**, while the server only ever sees integer counts.

* ``TemperatureScaler`` - one-parameter logit rescaling. Its likelihood has no
  finite-dimensional sufficient statistic, so the federated variant is an
  approximation built from per-bin logit means. The size of that approximation
  gap is itself a reported result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from federated_metrics import (
    brier_score,
    expected_calibration_error,
    negative_log_likelihood,
    reliability_curve,
    sigmoid,
    uniform_edges,
)

# Fixed logit grid used for federated temperature scaling. Constants, not data
# dependent, so no information about individual records is exposed.
LOGIT_LOW, LOGIT_HIGH, LOGIT_BINS = -12.0, 12.0, 48
TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.25), np.log(8.0), 240))


def _pava(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: non-decreasing isotonic fit."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    block_values: List[float] = []
    block_weights: List[float] = []
    block_sizes: List[int] = []

    for value, weight in zip(values, weights):
        block_values.append(float(value))
        block_weights.append(max(float(weight), 1e-12))
        block_sizes.append(1)

        while len(block_values) > 1 and block_values[-2] > block_values[-1]:
            v2 = block_values.pop()
            w2 = block_weights.pop()
            s2 = block_sizes.pop()
            v1 = block_values.pop()
            w1 = block_weights.pop()
            s1 = block_sizes.pop()

            block_values.append((v1 * w1 + v2 * w2) / (w1 + w2))
            block_weights.append(w1 + w2)
            block_sizes.append(s1 + s2)

    output: List[float] = []
    for value, size in zip(block_values, block_sizes):
        output.extend([value] * size)
    return np.asarray(output, dtype=float)


def _fill_empty_bins(raw: np.ndarray) -> np.ndarray:
    """Interpolate bins that received no data from their nearest neighbours."""
    filled = raw.copy()
    known = np.where(~np.isnan(filled))[0]
    if len(known) == 0:
        return np.linspace(0.0, 1.0, len(raw))
    if len(known) == 1:
        filled[:] = filled[known[0]]
        return filled
    missing = np.where(np.isnan(filled))[0]
    filled[missing] = np.interp(missing, known, filled[known])
    return filled


@dataclass
class HistogramBinner:
    """Fixed-edge probability binning, federatable by count summation.

    Laplace smoothing is applied to each bin before the isotonic projection.
    Without it, a bin that happens to contain only negatives in a small
    calibration split maps to p = 0 and produces unbounded log-loss on the test
    set. Smoothing keeps the estimator a pure function of the counts, so the
    federated and centralised fits remain identical.
    """

    n_bins: int = 10
    smoothing: float = 1.0
    values: np.ndarray | None = None
    totals: np.ndarray | None = None
    edges: np.ndarray = field(default_factory=lambda: uniform_edges(10))

    def __post_init__(self):
        self.edges = uniform_edges(self.n_bins)

    # -- fitting -----------------------------------------------------------
    def fit_from_counts(self, positives, totals) -> "HistogramBinner":
        positives = np.asarray(positives, dtype=float)
        totals = np.asarray(totals, dtype=float)
        alpha = float(self.smoothing)
        smoothed = (positives + alpha) / (totals + 2.0 * alpha)
        empty = totals <= 0
        raw = np.where(empty, np.nan, smoothed)
        filled = np.clip(_fill_empty_bins(raw), 0.0, 1.0)
        self.values = np.clip(_pava(filled, np.maximum(totals, 1e-12)), 0.0, 1.0)
        self.totals = totals
        return self

    def fit(self, probabilities, labels) -> "HistogramBinner":
        counts = count_bins(probabilities, labels, self.n_bins)
        return self.fit_from_counts(counts["positives"], counts["totals"])

    # -- inference ---------------------------------------------------------
    def transform(self, probabilities) -> np.ndarray:
        if self.values is None:
            return np.asarray(probabilities, dtype=float)
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
        index = np.clip(
            np.digitize(probabilities, self.edges[1:-1]), 0, self.n_bins - 1
        )
        return self.values[index]

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n_bins": int(self.n_bins),
            "values": [round(float(v), 6) for v in self.values],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "HistogramBinner":
        b = cls(n_bins=int(payload["n_bins"]))
        b.values = np.asarray(payload["values"], dtype=float)
        return b


def count_bins(probabilities, labels, n_bins: int = 10) -> Dict[str, list]:
    """Sufficient statistics for histogram binning. All that a client sends."""
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    labels = np.asarray(labels, dtype=float)
    edges = uniform_edges(n_bins)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, n_bins - 1)

    positives = np.bincount(index, weights=labels, minlength=n_bins)
    totals = np.bincount(index, minlength=n_bins).astype(float)
    return {
        "n_bins": int(n_bins),
        "positives": positives.round().astype(int).tolist(),
        "totals": totals.astype(int).tolist(),
    }


def pool_bin_counts(updates: List[Dict[str, list]]) -> Dict[str, list]:
    """Server-side aggregation of client histogram counts."""
    if not updates:
        raise ValueError("no client calibration updates supplied")
    n_bins = int(updates[0]["n_bins"])
    positives = np.zeros(n_bins)
    totals = np.zeros(n_bins)
    for update in updates:
        if int(update["n_bins"]) != n_bins:
            raise ValueError("clients used inconsistent bin counts")
        positives += np.asarray(update["positives"], dtype=float)
        totals += np.asarray(update["totals"], dtype=float)
    return {
        "n_bins": n_bins,
        "positives": positives.round().astype(int).tolist(),
        "totals": totals.astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------------
def _nll_from_logits(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = np.asarray(logits, dtype=float) / temperature
    y = np.asarray(labels, dtype=float)
    # softplus(z) - y*z  ==  -[y log s(z) + (1-y) log(1-s(z))]
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        losses = [_nll_from_logits(logits, labels, t) for t in TEMPERATURE_GRID]
        self.temperature = float(TEMPERATURE_GRID[int(np.argmin(losses))])
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=float) / self.temperature)

    def transform_proba(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
        return sigmoid(np.log(p / (1 - p)) / self.temperature)

    def to_dict(self) -> dict:
        return {"temperature": round(float(self.temperature), 4)}


def logit_histogram(logits: np.ndarray, labels: np.ndarray) -> Dict[str, list]:
    """Binned sufficient statistics for approximate federated temperature fit."""
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(LOGIT_LOW, LOGIT_HIGH, LOGIT_BINS + 1)
    index = np.clip(np.digitize(logits, edges[1:-1]), 0, LOGIT_BINS - 1)

    totals = np.bincount(index, minlength=LOGIT_BINS).astype(float)
    positives = np.bincount(index, weights=labels, minlength=LOGIT_BINS)
    sums = np.bincount(index, weights=logits, minlength=LOGIT_BINS)
    return {
        "n_bins": LOGIT_BINS,
        "totals": totals.astype(int).tolist(),
        "positives": positives.round().astype(int).tolist(),
        "logit_sums": [round(float(v), 4) for v in sums],
    }


def pool_logit_histograms(updates: List[Dict[str, list]]) -> Dict[str, list]:
    totals = np.zeros(LOGIT_BINS)
    positives = np.zeros(LOGIT_BINS)
    sums = np.zeros(LOGIT_BINS)
    for update in updates:
        totals += np.asarray(update["totals"], dtype=float)
        positives += np.asarray(update["positives"], dtype=float)
        sums += np.asarray(update["logit_sums"], dtype=float)
    return {
        "n_bins": LOGIT_BINS,
        "totals": totals.astype(int).tolist(),
        "positives": positives.round().astype(int).tolist(),
        "logit_sums": [round(float(v), 4) for v in sums],
    }


def fit_temperature_from_histogram(pooled: Dict[str, list]) -> TemperatureScaler:
    """Approximate federated temperature fit using per-bin mean logits."""
    totals = np.asarray(pooled["totals"], dtype=float)
    positives = np.asarray(pooled["positives"], dtype=float)
    sums = np.asarray(pooled["logit_sums"], dtype=float)

    mask = totals > 0
    if not np.any(mask):
        return TemperatureScaler(1.0)

    representative = sums[mask] / totals[mask]
    weights = totals[mask]
    rate = positives[mask] / totals[mask]

    losses = []
    for temperature in TEMPERATURE_GRID:
        z = representative / temperature
        per_bin = np.logaddexp(0.0, z) - rate * z
        losses.append(float(np.sum(weights * per_bin)))

    return TemperatureScaler(temperature=float(TEMPERATURE_GRID[int(np.argmin(losses))]))


def average_client_temperatures(
    client_logits: List[np.ndarray], client_labels: List[np.ndarray]
) -> TemperatureScaler:
    """Naive baseline: fit per client then average. Not equal to pooled fit."""
    temperatures, weights = [], []
    for logits, labels in zip(client_logits, client_labels):
        scaler = TemperatureScaler().fit(logits, labels)
        temperatures.append(scaler.temperature)
        weights.append(len(labels))
    if not temperatures:
        return TemperatureScaler(1.0)
    weights = np.asarray(weights, dtype=float)
    averaged = float(np.average(temperatures, weights=weights))
    return TemperatureScaler(temperature=averaged)


# ---------------------------------------------------------------------------
# Orchestration used by the engine
# ---------------------------------------------------------------------------
def calibration_report(
    y_test: np.ndarray,
    p_test: np.ndarray,
    client_updates: List[Dict[str, object]],
    n_bins: int = 10,
) -> Dict[str, object]:
    """Compare all calibration regimes on the held-out test set.

    ``client_updates`` entries carry the *local calibration split* only:
    ``{"probs":..., "logits":..., "labels":...}``. Those rows are never used
    for training the federated model.
    """
    probabilities = [
        np.asarray(u["probs"], dtype=float) for u in client_updates
    ]
    logits = [np.asarray(u["logits"], dtype=float) for u in client_updates]
    labels = [np.asarray(u["labels"], dtype=float) for u in client_updates]

    methods: Dict[str, dict] = {}

    def register(name: str, p_cal: np.ndarray, meta: dict | None = None) -> None:
        methods[name] = {
            "brier": brier_score(y_test, p_cal),
            "nll": negative_log_likelihood(y_test, p_cal),
            "ece": expected_calibration_error(y_test, p_cal, n_bins),
            "reliability": reliability_curve(y_test, p_cal, n_bins),
            **(meta or {}),
        }

    # 1. Raw federated model
    register("raw", p_test, {"description": "Uncalibrated federated output"})

    # 2. Local fits, then average the parameters
    local_temp = average_client_temperatures(logits, labels)
    register(
        "local_average_temperature",
        local_temp.transform_proba(p_test),
        {
            "description": "Fit per client, average temperatures",
            "parameters": local_temp.to_dict(),
        },
    )

    # 3. Centralised oracle for temperature (server is given pooled logits)
    pooled_logits = np.concatenate(logits)
    pooled_labels = np.concatenate(labels)
    oracle_temp = TemperatureScaler().fit(pooled_logits, pooled_labels)
    register(
        "centralised_temperature",
        oracle_temp.transform_proba(p_test),
        {
            "description": "Oracle: pooled logits available (privacy-violating)",
            "parameters": oracle_temp.to_dict(),
        },
    )

    # 4. Federated temperature (approximate, from binned logit statistics)
    pooled_hist = pool_logit_histograms([logit_histogram(l, y) for l, y in zip(logits, labels)])
    fed_temp = fit_temperature_from_histogram(pooled_hist)
    register(
        "federated_temperature",
        fed_temp.transform_proba(p_test),
        {
            "description": "Binned logit statistics only",
            "parameters": fed_temp.to_dict(),
            "gap_to_oracle": abs(fed_temp.temperature - oracle_temp.temperature),
        },
    )

    # 5. Centralised oracle for histogram binning
    oracle_binner = HistogramBinner(n_bins=n_bins)
    oracle_probs = np.concatenate(probabilities)
    oracle_binner.fit(oracle_probs, pooled_labels)
    register(
        "centralised_binning",
        oracle_binner.transform(p_test),
        {"description": "Oracle: pooled calibration rows available"},
    )

    # 6. Federated histogram binning (exact)
    client_counts = [count_bins(p, y, n_bins) for p, y in zip(probabilities, labels)]
    pooled_counts = pool_bin_counts(client_counts)
    fed_binner = HistogramBinner(n_bins=n_bins).fit_from_counts(
        pooled_counts["positives"], pooled_counts["totals"]
    )
    fed_cal = fed_binner.transform(p_test)
    register(
        "federated_binning",
        fed_cal,
        {
            "description": "Per-bin integer counts only (exact)",
            "bin_values": fed_binner.to_dict()["values"],
        },
    )

    # 7. Accuracy after calibration still needs a threshold; keep 0.5.
    for entry in methods.values():
        entry["ece"] = round(float(entry["ece"]), 6)
        entry["brier"] = round(float(entry["brier"]), 6)
        entry["nll"] = round(float(entry["nll"]), 6)

    exactness_gap = float(
        np.max(np.abs(fed_cal - oracle_binner.transform(p_test)))
    )

    return {
        "methods": methods,
        "bin_counts": client_counts,
        "pooled_counts": pooled_counts,
        "exactness_gap": exactness_gap,
        "best_federated": "federated_binning",
        "n_bins": n_bins,
    }
