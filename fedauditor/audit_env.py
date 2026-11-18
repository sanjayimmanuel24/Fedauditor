"""The run an auditor inspects: one FedFault scenario, re-run with recording.

``AuditRun.from_scenario`` plants the scenario, runs the federation and keeps
what a server could log during training, without any patient rows leaving the
clients:

* the global model after every round (``global_at(r)``; round 0 is the start);
* every client's uploaded update per round, as a vector: the parameter delta for
  numpy models, and for XGBoost the uploaded tree's margins on a fixed set of
  synthetic probe points (no patient data), so update directions are comparable;
* which dataset each client held in each round (drift swaps it mid-training).

The audit tools (``fedauditor.tools``) compute aggregates from this state at the
clients and release only noisy summaries. ``truth`` is kept here for scoring and
is never shown to the auditor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import xgboost as xgb

from fedauditor.data import FederatedDataset
from fedauditor.faults import Planted, Scenario, plant
from fedauditor.federation import FederationResult, run_federation
from fedauditor.models import NumpyModel, make_model

N_PROBE = 256


def _flat(weights) -> np.ndarray:
    return np.concatenate([np.asarray(w, dtype=float).ravel() for w in weights])


@dataclass
class AuditRun:
    scenario: Scenario
    planted: Planted
    result: FederationResult
    probe: np.ndarray
    snapshots: Dict[int, object] = field(default_factory=dict)       # round -> weights | n_trees
    updates: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    data_by_round: Dict[int, FederatedDataset] = field(default_factory=dict)

    # ------------------------------------------------------------ construction
    @classmethod
    def from_scenario(cls, scenario: Scenario, data: FederatedDataset) -> "AuditRun":
        planted = plant(scenario, data)
        kwargs = planted.run_kwargs()
        inner = kwargs.pop("update_hook", None)
        schedule = kwargs.pop("data_schedule", None)
        probe = np.random.default_rng([scenario.seed, 777]).standard_normal((N_PROBE, data.n_features))
        snapshots: Dict[int, object] = {}
        updates: Dict[int, Dict[str, np.ndarray]] = {}
        by_round: Dict[int, FederatedDataset] = {}

        def record_schedule(rnd):
            d = schedule(rnd) if schedule is not None else planted.data
            by_round[rnd] = d
            return d

        def hook(rnd, name, local, global_model):
            if rnd - 1 not in snapshots:      # first upload this round: global after rnd-1
                snapshots[rnd - 1] = _snapshot(global_model)
            if inner is not None:
                inner(rnd, name, local, global_model)
            if isinstance(local, NumpyModel):
                vec = _flat(local.weights) - _flat(global_model.weights)
            else:
                vec = (np.zeros(N_PROBE) if local._update is None else
                       local._update.predict(xgb.DMatrix(probe), output_margin=True).astype(float))
            updates.setdefault(rnd, {})[name] = vec

        result = run_federation(planted.data, update_hook=hook, data_schedule=record_schedule, **kwargs)
        snapshots[scenario.rounds] = _snapshot(result.model)
        return cls(scenario, planted, result, probe, snapshots, updates, by_round)

    # ------------------------------------------------------------------ access
    @property
    def rounds(self) -> int:
        return self.scenario.rounds

    @property
    def truth(self) -> dict:
        return self.planted.truth

    @property
    def final_data(self) -> FederatedDataset:
        return self.data_by_round.get(self.rounds, self.planted.data)

    def data_at(self, rnd: int) -> FederatedDataset:
        rnd = max(1, min(self.rounds, rnd))
        return self.data_by_round.get(rnd, self.planted.data)

    def global_at(self, rnd: int):
        """The global model after round ``rnd`` (0 = initial model)."""
        rnd = max(0, min(self.rounds, rnd))
        snap = self.snapshots[rnd]
        model = self.result.model
        if isinstance(model, NumpyModel):
            m = make_model(self.scenario.model, self.planted.data.n_features, seed=self.scenario.seed)
            m.weights = [w.copy() for w in snap]
            return m
        return _TreePrefix(model, snap)

    def margin(self, model, X: np.ndarray) -> np.ndarray:
        if isinstance(model, _TreePrefix):
            return model.margin(X)
        p = np.clip(model.predict_proba(X), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p))


def _snapshot(model):
    if isinstance(model, NumpyModel):
        return [w.copy() for w in model.weights]
    return model.n_trees


class _TreePrefix:
    """The first ``n_trees`` trees of a bagged XGBoost ensemble (the global model at a round)."""

    def __init__(self, model, n_trees: int):
        self.model, self.n_trees = model, n_trees

    def margin(self, X: np.ndarray) -> np.ndarray:
        if self.model.booster is None or self.n_trees == 0:
            return np.zeros(len(X))
        return self.model.booster.predict(xgb.DMatrix(X), output_margin=True,
                                          iteration_range=(0, self.n_trees)).astype(float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.margin(X)))
