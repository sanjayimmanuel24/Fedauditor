"""Seeded in-process federation driven by Flower strategies.

Aggregation is done by Flower's own strategy classes (``FedAvg``,
``FedMedian``, ``FedTrimmedAvg``, ``FedXgbBagging``), fed with real
``FitRes`` messages. The round loop runs in-process instead of through Flower's
Ray-based simulation engine because FedFault needs thousands of short runs
that must be exactly reproducible from a seed, and Ray adds neither.

``update_hook(round, client, local_model, global_model)`` runs after each
client's local training and before upload. Phase 2 uses it to plant faults
(poisoning, free-riding); it may modify ``local_model`` in place.

``data_schedule(round) -> FederatedDataset`` optionally swaps the data seen in a
round (same clients, possibly changed rows); Phase 2 uses it for drift.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
from flwr.common import Code, FitRes, Parameters, Status
from flwr.server.strategy import FedAvg, FedMedian, FedTrimmedAvg, FedXgbBagging

from federated_metrics import full_metrics
from fedauditor.data import FederatedDataset
from fedauditor.models import NumpyModel, make_model

_NO_METRICS = {"fit_metrics_aggregation_fn": lambda _: {}}
NUMPY_STRATEGIES = {
    "fedavg": lambda init: FedAvg(initial_parameters=init, **_NO_METRICS),
    "fedmedian": lambda init: FedMedian(initial_parameters=init, **_NO_METRICS),
    "fedtrimmedavg": lambda init: FedTrimmedAvg(initial_parameters=init, beta=0.2, **_NO_METRICS),
}

UpdateHook = Callable[[int, str, object, object], None]
DataSchedule = Callable[[int], FederatedDataset]


class BatchedXgbBagging(FedXgbBagging):
    """Flower's ``FedXgbBagging`` with the same result, but the global model is
    kept parsed between rounds instead of being re-parsed for every client.
    ``aggregate`` in Flower costs O(ensemble size) per client; this is O(1)."""

    _model: Optional[dict] = None

    def aggregate_fit(self, server_round, results, failures):
        if not results or (not self.accept_failures and failures):
            return None, {}
        model, raw = self._model, None
        for _, fit_res in results:
            for bst in fit_res.parameters.tensors:
                if model is None:
                    model, raw = json.loads(bytearray(bst)), bst
                    continue
                curr = json.loads(bytearray(bst))
                gb = model["learner"]["gradient_booster"]["model"]
                n_prev = int(gb["gbtree_model_param"]["num_trees"])
                n_par = int(curr["learner"]["gradient_booster"]["model"]
                            ["gbtree_model_param"]["num_parallel_tree"])
                gb["gbtree_model_param"]["num_trees"] = str(n_prev + n_par)
                gb["iteration_indptr"].append(gb["iteration_indptr"][-1] + n_par)
                trees = curr["learner"]["gradient_booster"]["model"]["trees"]
                for k in range(n_par):
                    trees[k]["id"] = n_prev + k
                    gb["trees"].append(trees[k])
                    gb["tree_info"].append(0)
                raw = None
        self._model = model
        self.global_model = raw if raw is not None else bytes(json.dumps(model), "utf-8")
        return Parameters(tensor_type="", tensors=[self.global_model]), {}


@dataclass
class FederationResult:
    dataset: str
    model_kind: str
    strategy: str
    seed: int
    rounds: int
    model: object
    history: List[dict] = field(default_factory=list)
    final: Dict[str, dict] = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset, "model": self.model_kind,
            "strategy": self.strategy, "seed": self.seed, "rounds": self.rounds,
            "seconds": round(self.seconds, 2), "history": self.history,
            "final": self.final,
        }


def _make_strategy(name: str, model_kind: str, initial):
    if model_kind == "xgb":
        if name not in ("fedavg", "bagging"):
            raise ValueError("xgb is federated by tree bagging only")
        return BatchedXgbBagging(), "bagging"
    if name not in NUMPY_STRATEGIES:
        raise KeyError(f"unknown strategy {name!r}; expected {sorted(NUMPY_STRATEGIES)}")
    return NUMPY_STRATEGIES[name](initial), name


def evaluate(model, data: FederatedDataset, split: str = "test", full: bool = False) -> Dict[str, dict]:
    """Per-client and pooled metrics on one split."""
    out: Dict[str, dict] = {}
    probs, labels = [], []
    for client in data.clients:
        p = model.predict_proba(client.X[split])
        y = client.y[split]
        probs.append(p)
        labels.append(y)
        m = full_metrics(y, p)
        out[client.name] = m if full else {"n": m["n"], "auc": m["auc"], "brier": m["brier"]}
    m = full_metrics(np.concatenate(labels), np.concatenate(probs))
    out["_pooled"] = m if full else {"n": m["n"], "auc": m["auc"], "brier": m["brier"]}
    return out


def run_federation(
    data: FederatedDataset,
    model: str = "lr",
    rounds: int = 20,
    strategy: str = "fedavg",
    local_epochs: int = 1,
    seed: int = 20,
    fraction_fit: float = 1.0,
    model_kwargs: Optional[dict] = None,
    update_hook: Optional[UpdateHook] = None,
    eval_every: int = 1,
    data_schedule: Optional[DataSchedule] = None,
    record_weights: bool = False,
) -> FederationResult:
    start = time.perf_counter()
    model_kwargs = model_kwargs or {}
    rng = np.random.default_rng(seed)
    global_model = make_model(model, data.n_features, seed=seed, **model_kwargs)
    params = global_model.to_parameters()
    strat, strategy = _make_strategy(strategy, model, params)

    n_select = max(1, int(round(fraction_fit * len(data.clients))))
    result = FederationResult(data.key, model, strategy, seed, rounds, global_model)

    current = data
    for rnd in range(1, rounds + 1):
        current = data_schedule(rnd) if data_schedule is not None else data
        chosen = sorted(rng.choice(len(current.clients), size=n_select, replace=False))
        results, norms = [], {}
        for i in chosen:
            client = current.clients[i]
            local = make_model(model, current.n_features, seed=seed, **model_kwargs)
            local.start_round(global_model, client.X["train"])
            local.fit(client.X["train"], client.y["train"], epochs=local_epochs,
                      rng=np.random.default_rng([seed, rnd, i]), n_clients=n_select)
            if update_hook is not None:
                update_hook(rnd, client.name, local, global_model)
            norms[client.name] = local.update_norm(global_model)
            fit_res = FitRes(Status(Code.OK, ""), local.update_parameters(),
                             client.n_train, {})
            results.append((None, fit_res))

        params, _ = strat.aggregate_fit(rnd, results, [])
        global_model.from_parameters(params)

        if rnd % eval_every == 0 or rnd == rounds:
            entry = {"round": rnd, "clients": [current.clients[i].name for i in chosen],
                     "test": evaluate(global_model, current), "update_norm": norms}
            if record_weights and isinstance(global_model, NumpyModel):
                entry["weights"] = [w.tolist() for w in global_model.weights]
            result.history.append(entry)

    result.final = evaluate(global_model, current, full=True)
    if hasattr(global_model, "clear_cache"):
        global_model.clear_cache()
    result.seconds = time.perf_counter() - start
    return result
