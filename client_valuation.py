"""Client valuation via exact Shapley values.

The value of a coalition is the quality of the federated model trained on the
union of its members' local data. With three clients the full coalition space is
only eight subsets, so the Shapley value can be computed exactly rather than
sampled.

The interesting question is not "who is most valuable" but "does the answer
depend on the metric". A hospital can be the largest contributor to accuracy
while actively degrading calibration, which is exactly the incentive problem
this module exposes.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial
from typing import Dict, Iterable, List

import numpy as np

from config import VALUATION_ROUNDS
from federated_metrics import full_metrics


class ClientValuation:
    def __init__(self, engine, n_rounds: int | None = None):
        self.engine = engine
        self.n_rounds = int(n_rounds or VALUATION_ROUNDS)
        self._model_cache: Dict[frozenset, tuple] = {}
        self._metric_cache: Dict[frozenset, dict] = {}

    # -- model training ----------------------------------------------------
    def train_coalition(self, members: Iterable[int]) -> tuple:
        key = frozenset(int(m) for m in members)
        if key in self._model_cache:
            return self._model_cache[key]

        if not key:
            raise ValueError("empty coalition has no trainable model")

        coef = np.zeros((1, len(self.engine.feature_names)))
        intercept = np.zeros(1)

        for _ in range(self.n_rounds):
            updates = []
            for client_id in sorted(key):
                (
                    local_coef,
                    local_intercept,
                    _,
                    n_samples,
                    _,
                    _,
                    prevalence,
                ) = self.engine.train_local_client(client_id, coef, intercept)
                updates.append(
                    (local_coef, local_intercept, n_samples, prevalence)
                )
            coef, intercept = self.engine.federated_average(updates)

        self._model_cache[key] = (coef, intercept)
        return coef, intercept

    # -- raw coalition quality --------------------------------------------
    def _baseline_metrics(self) -> dict:
        """Quality of the global model before any client contributes.

        The empty coalition is the *untrained* model (zero weights, p = 0.5),
        not a prevalence-only predictor. A prevalence predictor scores a
        misleadingly perfect ECE of 0, which would make every client look like
        it worsens calibration. The untrained model is the honest zero point.
        """
        if not hasattr(self, "_baseline_cache"):
            coef, intercept = self.engine.initial_parameters()
            probabilities = self.engine.predict_proba(
                coef, intercept, self.engine.X_test_scaled
            )
            self._baseline_cache = full_metrics(self.engine.y_test, probabilities)
        return self._baseline_cache

    def metrics_for(self, members: Iterable[int]) -> dict:
        """Raw evaluation metrics of the model trained on ``members``."""
        key = frozenset(int(m) for m in members)
        if key in self._metric_cache:
            return self._metric_cache[key]

        if not key:
            metrics = self._baseline_metrics()
        else:
            coef, intercept = self.train_coalition(key)
            probabilities = self.engine.predict_proba(
                coef, intercept, self.engine.X_test_scaled
            )
            metrics = full_metrics(self.engine.y_test, probabilities)

        self._metric_cache[key] = metrics
        return metrics

    # -- value function ----------------------------------------------------
    # A positive value always means "better than contributing nothing", for
    # every metric. Loss metrics are sign-flipped so that a ranking of clients
    # is unambiguous regardless of whether the metric is accuracy or ECE.
    _DIRECTION = {"accuracy": 1.0, "auc": 1.0, "ece": -1.0, "brier": -1.0, "nll": -1.0}

    def value(self, members: Iterable[int]) -> Dict[str, float]:
        baseline = self._baseline_metrics()
        current = self.metrics_for(members)
        out: Dict[str, float] = {}
        for metric, direction in self._DIRECTION.items():
            before = baseline.get(metric)
            after = current.get(metric)
            if before is None or after is None:
                continue
            if before != before or after != after:  # NaN guard
                continue
            out[metric] = float(direction * (after - before))
        return out

    # -- Shapley -----------------------------------------------------------
    def exact_shapley(self, metric: str = "accuracy") -> Dict[int, float]:
        clients = list(range(self.engine.n_clients))
        n = len(clients)
        values = {
            frozenset(subset): self.value(subset)[metric]
            for r in range(n + 1)
            for subset in combinations(clients, r)
        }

        phi: Dict[int, float] = {}
        for client in clients:
            others = [c for c in clients if c != client]
            total = 0.0
            for r in range(len(others) + 1):
                for subset in combinations(others, r):
                    coalition = frozenset(subset)
                    weight = (
                        factorial(len(coalition))
                        * factorial(n - len(coalition) - 1)
                        / factorial(n)
                    )
                    total += weight * (
                        values[coalition | {client}] - values[coalition]
                    )
            phi[client] = total
        return phi

    def permutation_shapley(self, metric: str = "accuracy", n_permutations: int = 200, seed: int = 0):
        """Monte-Carlo fallback for more than ~6 clients."""
        clients = list(range(self.engine.n_clients))
        rng = np.random.default_rng(seed)
        totals = {c: 0.0 for c in clients}

        for _ in range(n_permutations):
            order = list(rng.permutation(clients))
            coalition: List[int] = []
            previous = self.value(coalition)[metric]
            for client in order:
                coalition.append(client)
                current = self.value(coalition)[metric]
                totals[client] += current - previous
                previous = current

        return {c: totals[c] / n_permutations for c in clients}

    # -- reporting ---------------------------------------------------------
    def report(self, metrics: Iterable[str] = ("accuracy", "auc", "ece", "brier")) -> dict:
        clients = list(range(self.engine.n_clients))
        metrics = list(metrics)

        shapley = {metric: self.exact_shapley(metric) for metric in metrics}

        coalitions = []
        for r in range(len(clients) + 1):
            for subset in combinations(clients, r):
                raw = self.metrics_for(subset)
                coalitions.append(
                    {
                        "members": list(subset),
                        "label": "+".join(f"C{c + 1}" for c in subset) or "none",
                        "accuracy": round(raw["accuracy"], 4),
                        "auc": (
                            round(raw["auc"], 4)
                            if raw["auc"] == raw["auc"]
                            else None
                        ),
                        "ece": round(raw["ece"], 4),
                        "brier": round(raw["brier"], 4),
                    }
                )

        rankings = {}
        for metric in metrics:
            ordering = sorted(clients, key=lambda c: shapley[metric][c], reverse=True)
            rankings[metric] = ordering

        full_coalition = frozenset(clients)
        efficiency = {
            metric: {
                "sum_shapley": round(float(sum(shapley[metric].values())), 6),
                "value_full": round(float(self.value(full_coalition)[metric]), 6),
                "value_empty": round(float(self.value([])[metric]), 6),
            }
            for metric in metrics
        }
        for metric, entry in efficiency.items():
            entry["residual"] = round(
                entry["sum_shapley"] - (entry["value_full"] - entry["value_empty"]), 6
            )

        rank_disagreement = {}
        reference = rankings.get("accuracy", clients)
        for metric in metrics:
            if metric == "accuracy":
                continue
            ordering = rankings[metric]
            pairs = 0
            agreements = 0
            for i, a in enumerate(reference):
                for b in reference[i + 1 :]:
                    pairs += 1
                    if ordering.index(a) < ordering.index(b):
                        agreements += 1
            kendall = (2 * agreements - pairs) / pairs if pairs else 1.0
            rank_disagreement[f"accuracy_vs_{metric}"] = round(kendall, 4)

        return {
            "clients": [
                {
                    "id": c + 1,
                    "label": f"Client {c + 1}",
                    "cohort": self.engine.client_meta[c]["dominant_band"],
                    "n_samples": self.engine.client_meta[c]["n_samples"],
                    "prevalence": self.engine.client_meta[c]["prevalence"],
                }
                for c in clients
            ],
            "metrics": metrics,
            "shapley": {
                metric: {f"Client {c + 1}": round(float(v), 6) for c, v in values.items()}
                for metric, values in shapley.items()
            },
            "rankings": {metric: [f"Client {c + 1}" for c in order] for metric, order in rankings.items()},
            "rank_agreement": rank_disagreement,
            "efficiency": efficiency,
            "coalitions": coalitions,
            "rounds_per_coalition": self.n_rounds,
        }
