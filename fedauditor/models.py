"""The three model families, each exchanging Flower ``Parameters``.

* ``lr``  - logistic regression, minibatch SGD with L2 (numpy).
* ``mlp`` - one hidden ReLU layer, minibatch SGD with L2 (numpy).
* ``xgb`` - XGBoost, federated by tree bagging: each round every client grows
  one tree on top of the global ensemble and the server appends the trees
  (Flower's ``FedXgbBagging``). The learning rate is divided by the number of
  clients in the round so the bagged step matches one boosting step.

LR and MLP are written in numpy rather than torch so every weight is visible to
the audit tools and runs are bit-for-bit reproducible from a seed.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from federated_metrics import sigmoid


class NumpyModel:
    """Shared plumbing for models whose state is a list of numpy arrays."""

    kind = "numpy"
    weights: List[np.ndarray]

    def to_parameters(self) -> Parameters:
        return ndarrays_to_parameters(self.weights)

    def from_parameters(self, parameters: Parameters) -> None:
        self.weights = [w.copy() for w in parameters_to_ndarrays(parameters)]

    def update_parameters(self) -> Parameters:
        """What a client uploads after local training: its full weights."""
        return self.to_parameters()

    def start_round(self, global_model: "NumpyModel", X_train: np.ndarray) -> None:
        """Begin local training from the current global model."""
        self.weights = [w.copy() for w in global_model.weights]

    def update_norm(self, global_model: "NumpyModel") -> float:
        """L2 distance between this client's upload and the global model."""
        return float(np.sqrt(sum(np.sum((a - b) ** 2)
                                 for a, b in zip(self.weights, global_model.weights))))

    def _batches(self, n: int, rng: np.random.Generator):
        order = rng.permutation(n)
        for start in range(0, n, self.batch_size):
            yield order[start:start + self.batch_size]


class LogisticModel(NumpyModel):
    kind = "lr"

    def __init__(self, n_features: int, seed: int = 0, lr: float = 0.05,
                 l2: float = 1e-4, batch_size: int = 64):
        self.lr, self.l2, self.batch_size = lr, l2, batch_size
        self.weights = [np.zeros(n_features), np.zeros(1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        coef, intercept = self.weights
        return sigmoid(X @ coef + intercept[0])

    def fit(self, X, y, epochs: int, rng: np.random.Generator, **_) -> None:
        coef, intercept = self.weights
        for _ in range(epochs):
            for idx in self._batches(len(y), rng):
                g = sigmoid(X[idx] @ coef + intercept[0]) - y[idx]
                coef -= self.lr * (X[idx].T @ g / len(idx) + self.l2 * coef)
                intercept -= self.lr * g.mean()


class MLPModel(NumpyModel):
    kind = "mlp"

    def __init__(self, n_features: int, seed: int = 0, hidden: int = 32,
                 lr: float = 0.05, l2: float = 1e-4, batch_size: int = 64):
        self.lr, self.l2, self.batch_size = lr, l2, batch_size
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / n_features), (n_features, hidden)),
            np.zeros(hidden),
            rng.normal(0, np.sqrt(1.0 / hidden), hidden),
            np.zeros(1),
        ]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        W1, b1, w2, b2 = self.weights
        return sigmoid(np.maximum(X @ W1 + b1, 0) @ w2 + b2[0])

    def fit(self, X, y, epochs: int, rng: np.random.Generator, **_) -> None:
        W1, b1, w2, b2 = self.weights
        for _ in range(epochs):
            for idx in self._batches(len(y), rng):
                Xb = X[idx]
                h = np.maximum(Xb @ W1 + b1, 0)
                g = (sigmoid(h @ w2 + b2[0]) - y[idx]) / len(idx)
                dh = np.outer(g, w2) * (h > 0)
                w2 -= self.lr * (h.T @ g + self.l2 * w2)
                b2 -= self.lr * g.sum()
                W1 -= self.lr * (Xb.T @ dh + self.l2 * W1)
                b1 -= self.lr * dh.sum(axis=0)


def tree_leaves(booster: xgb.Booster) -> np.ndarray:
    """Leaf values of every tree in ``booster``."""
    model = json.loads(bytes(booster.save_raw("json")))
    out = []
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        values = tree["split_conditions"]
        out.extend(values[i] for i, left in enumerate(tree["left_children"]) if left == -1)
    return np.asarray(out, dtype=float)


def scale_tree_leaves(booster: xgb.Booster, factor: float,
                      noise: Optional[np.ndarray] = None) -> xgb.Booster:
    """A copy of ``booster`` with every leaf value multiplied by ``factor``.

    If ``noise`` is given, leaf ``k`` (in ``tree_leaves`` order) is instead
    multiplied by ``factor * (1 + noise[k])``. Used to plant update-level faults.
    """
    model = json.loads(bytes(booster.save_raw("json")))
    k = 0
    for tree in model["learner"]["gradient_booster"]["model"]["trees"]:
        for i, left in enumerate(tree["left_children"]):
            if left != -1:
                continue
            f = factor * (1.0 + (noise[k] if noise is not None else 0.0))
            tree["split_conditions"][i] *= f
            tree["base_weights"][i] *= f
            k += 1
    out = xgb.Booster()
    out.load_model(bytearray(json.dumps(model), "utf-8"))
    return out


class XGBModel:
    """XGBoost federated by bagging.

    Speed: the global model caches its output margin per input array (keyed by
    array identity) and only predicts the trees added since the last call, and a
    client starts each round from that cached margin (``start_round``) instead
    of deserialising and re-predicting the whole ensemble. This is valid
    because bagging only ever appends trees; ``from_parameters`` with fewer
    trees than a cached margin clears the cache.
    """

    kind = "xgb"
    _CACHE_SIZE = 512

    def __init__(self, n_features: int, seed: int = 0, eta: float = 0.3,
                 max_depth: int = 4, nthread: int = 1):
        self.eta = eta
        # base_score 0.5 -> zero base margin, so per-tree margins simply add up.
        self.params = {
            "objective": "binary:logistic", "max_depth": max_depth,
            "tree_method": "hist", "nthread": nthread, "seed": seed,
            "base_score": 0.5, "verbosity": 0,
        }
        self.booster: Optional[xgb.Booster] = None
        self._update: Optional[xgb.Booster] = None
        self._base_margin: Optional[np.ndarray] = None
        self._margins: Dict[int, tuple] = {}

    @staticmethod
    def _raw(booster: Optional[xgb.Booster]) -> bytes:
        return bytes(booster.save_raw("json")) if booster is not None else b""

    def to_parameters(self) -> Parameters:
        return Parameters(tensor_type="", tensors=[self._raw(self.booster)])

    def from_parameters(self, parameters: Parameters) -> None:
        raw = parameters.tensors[0] if parameters.tensors else b""
        if not raw:
            self.booster = None
            self._margins.clear()
            return
        booster = xgb.Booster(params={"nthread": self.params["nthread"]})
        booster.load_model(bytearray(raw))
        self.booster = booster
        if any(n > self.n_trees for _, n, _ in self._margins.values()):
            self._margins.clear()

    def update_parameters(self) -> Parameters:
        """Only the tree grown this round (the whole model in round one)."""
        return Parameters(tensor_type="", tensors=[self._raw(self._update)])

    def clear_cache(self) -> None:
        self._margins.clear()

    @property
    def n_trees(self) -> int:
        return 0 if self.booster is None else self.booster.num_boosted_rounds()

    def margin(self, X: np.ndarray) -> np.ndarray:
        """Output margin (log-odds), predicting only trees not yet cached for ``X``."""
        if self.booster is None:
            return np.zeros(len(X))
        n = self.n_trees
        hit = self._margins.get(id(X))
        if hit is not None and hit[0] is X and hit[1] <= n:
            _, done, m = hit
        else:
            done, m = 0, np.zeros(len(X))
        if done < n:
            m = m + self.booster.predict(xgb.DMatrix(X), output_margin=True,
                                         iteration_range=(done, n)).astype(float)
        if len(self._margins) >= self._CACHE_SIZE and id(X) not in self._margins:
            self._margins.pop(next(iter(self._margins)))
        self._margins[id(X)] = (X, n, m)
        return m

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.booster is None:
            return np.full(len(X), 0.5)
        return sigmoid(self.margin(X))

    def start_round(self, global_model: "XGBModel", X_train: np.ndarray) -> None:
        """Begin local training from the global ensemble's cached margin on ``X_train``.

        The local ``booster`` is the (shared, unmodified) global ensemble; the
        tree grown in ``fit`` is held in ``_update`` only.
        """
        self.booster = global_model.booster
        self._base_margin = None if self.booster is None else global_model.margin(X_train)

    def update_norm(self, global_model=None) -> float:
        """L2 norm of the uploaded tree's leaf values."""
        return 0.0 if self._update is None else float(np.linalg.norm(tree_leaves(self._update)))

    def fit(self, X, y, epochs: int = 1, rng=None, n_clients: int = 1, **_) -> None:
        params = dict(self.params, eta=self.eta / max(n_clients, 1))
        if self._base_margin is not None:
            # Same gradients as continuing the global booster, without reloading it.
            dtrain = xgb.DMatrix(X, label=y, base_margin=self._base_margin)
            self._update = xgb.train(params, dtrain, num_boost_round=1)
            self._base_margin = None
            return
        before = self.n_trees
        booster = xgb.train(params, xgb.DMatrix(X, label=y), num_boost_round=1,
                            xgb_model=self.booster)
        self._update = booster if before == 0 else booster[before:before + 1]
        self.booster = booster


MODELS = {"lr": LogisticModel, "mlp": MLPModel, "xgb": XGBModel}


def make_model(kind: str, n_features: int, seed: int = 0, **kwargs):
    if kind not in MODELS:
        raise KeyError(f"unknown model {kind!r}; expected one of {sorted(MODELS)}")
    return MODELS[kind](n_features, seed=seed, **kwargs)
