"""FedFault: planted faults with known ground truth.

A ``Scenario`` names one fault, a severity, a seed and the run settings. ``plant``
turns it into what ``run_federation`` needs (a dataset, an optional
``data_schedule`` and an optional ``update_hook``) plus the ground truth the
auditor is scored against: which clients are faulty, from which round, and
the fault's parameters.

Faults (severity values are ``low / medium / high``):

* ``label_noise``      - a fraction of the client's labels flipped (0.1/0.2/0.3).
* ``drift``            - from mid-training, the three features most correlated
                         with the outcome shift by k SD (0.5/1/2).
* ``poisoning``        - uploaded update replaced by -k times itself (1/3/10).
* ``free_rider``       - in a fraction of rounds the client uploads the global
                         model (plus tiny noise) without training (0.3/0.6/1.0).
* ``prevalence_shift`` - client keeps only 1/k of its negatives, multiplying the
                         outcome odds by k (2/4/8).
* ``subgroup_bias``    - a fraction of positives in one subgroup recorded as
                         negative, i.e. under-diagnosis (0.3/0.6/0.9).
* ``missing_artefact`` - a fraction of rows have two features exported as a
                         sentinel code instead of missing (0.1/0.25/0.5).
* ``explanation_instability`` - random noise added to every upload, k times
                         the update's own norm (1/3/10): the model's accuracy
                         barely moves but its attributions jitter round to round.

Data faults change every split of the faulty client (it is that site's data);
``clean_data`` in the result keeps the untouched data for measuring harm.
Affected clients are drawn from clients with at least median training size and
are the same across severities for a given seed, so severities are comparable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Optional

import numpy as np

from fedauditor.data import SPLITS, ClientData, FederatedDataset
from fedauditor.models import NumpyModel, scale_tree_leaves, tree_leaves

SEVERITY_LEVELS = ("low", "medium", "high")
SEVERITIES: Dict[str, Dict[str, float]] = {
    "label_noise": {"low": 0.1, "medium": 0.2, "high": 0.3},
    "drift": {"low": 0.5, "medium": 1.0, "high": 2.0},
    "poisoning": {"low": 1.0, "medium": 3.0, "high": 10.0},
    "free_rider": {"low": 0.3, "medium": 0.6, "high": 1.0},
    "prevalence_shift": {"low": 2.0, "medium": 4.0, "high": 8.0},
    "subgroup_bias": {"low": 0.3, "medium": 0.6, "high": 0.9},
    "missing_artefact": {"low": 0.1, "medium": 0.25, "high": 0.5},
    "explanation_instability": {"low": 1.0, "medium": 3.0, "high": 10.0},
}
FAULTS = tuple(SEVERITIES)
CLEAN = "clean"


@dataclass(frozen=True)
class Scenario:
    fault: str                       # one of FAULTS, or "clean"
    severity: str = "none"           # one of SEVERITY_LEVELS ("none" when clean)
    seed: int = 20
    dataset: str = "brfss"
    model: str = "lr"
    rounds: int = 20
    strategy: str = "fedavg"
    n_affected: Optional[int] = None  # default: ~5% of clients, at least 1, at most 3

    def __post_init__(self):
        if self.fault == CLEAN:
            if self.severity != "none":
                raise ValueError("clean scenarios have severity 'none'")
        elif self.fault not in SEVERITIES:
            raise KeyError(f"unknown fault {self.fault!r}; expected {FAULTS}")
        elif self.severity not in SEVERITY_LEVELS:
            raise KeyError(f"unknown severity {self.severity!r}; expected {SEVERITY_LEVELS}")

    @property
    def id(self) -> str:
        out = (f"{self.dataset}-{self.model}-{self.strategy}-r{self.rounds}-"
               f"{self.fault}-{self.severity}-s{self.seed}")
        return out + (f"-n{self.n_affected}" if self.n_affected else "")

    @property
    def level(self) -> float:
        return 0.0 if self.fault == CLEAN else SEVERITIES[self.fault][self.severity]

    def to_dict(self) -> dict:
        return {**asdict(self), "id": self.id, "level": self.level}


@dataclass
class Planted:
    scenario: Scenario
    data: FederatedDataset            # data at the end of the run (what an auditor sees)
    clean_data: FederatedDataset
    truth: dict
    data_schedule: Optional[Callable[[int], FederatedDataset]] = None
    update_hook: Optional[Callable] = None

    def run_kwargs(self) -> dict:
        s = self.scenario
        out = {"model": s.model, "rounds": s.rounds, "strategy": s.strategy, "seed": s.seed}
        if self.data_schedule is not None:
            out["data_schedule"] = self.data_schedule
        if self.update_hook is not None:
            out["update_hook"] = self.update_hook
        return out


# --------------------------------------------------------------------- helpers

def _rng(scenario: Scenario, *extra: int) -> np.random.Generator:
    """Fault RNG: depends on seed and fault, not severity (same clients across severities)."""
    return np.random.default_rng([scenario.seed, FAULTS.index(scenario.fault) + 1, *extra])


def default_n_affected(n_clients: int) -> int:
    return int(max(1, min(3, round(0.05 * n_clients))))


def choose_clients(data: FederatedDataset, n: int, rng: np.random.Generator) -> List[str]:
    sizes = np.array([c.n_train for c in data.clients])
    eligible = [c.name for c in data.clients if c.n_train >= np.median(sizes)]
    n = min(n, len(eligible))
    return sorted(rng.choice(eligible, size=n, replace=False).tolist())


def _copy_client(c: ClientData) -> ClientData:
    return ClientData(
        c.name,
        {s: c.X[s].copy() for s in SPLITS},
        {s: c.y[s].copy() for s in SPLITS},
        {s: {a: v.copy() for a, v in c.groups[s].items()} for s in SPLITS},
    )


def _subset(c: ClientData, keep: Dict[str, np.ndarray]) -> ClientData:
    return ClientData(
        c.name,
        {s: c.X[s][keep[s]] for s in SPLITS},
        {s: c.y[s][keep[s]] for s in SPLITS},
        {s: {a: v[keep[s]] for a, v in c.groups[s].items()} for s in SPLITS},
    )


def _with_clients(data: FederatedDataset, new: Dict[str, ClientData], tag: str) -> FederatedDataset:
    clients = [new.get(c.name, c) for c in data.clients]
    return replace(data, clients=clients, notes=list(data.notes) + [f"FedFault: {tag}"])


def _raw_train(data: FederatedDataset) -> np.ndarray:
    """Pooled training features back on the original scale (via the federated scaler)."""
    X, _ = data.pooled("train")
    mean = np.asarray(data.scaler.get("mean") or np.zeros(X.shape[1]))
    scale = np.asarray(data.scaler.get("scale") or np.ones(X.shape[1]))
    return X * scale + mean


def _to_scaled(data: FederatedDataset, j: int, raw: float) -> float:
    mean = data.scaler.get("mean") or [0.0] * data.n_features
    scale = data.scaler.get("scale") or [1.0] * data.n_features
    return float((raw - mean[j]) / scale[j])


def top_features(data: FederatedDataset, k: int) -> List[int]:
    """Indices of the ``k`` features most correlated with the outcome (pooled clean train)."""
    X, y = data.pooled("train")
    yc = y - y.mean()
    corr = np.abs((X - X.mean(axis=0)).T @ yc) / (X.std(axis=0) * yc.std() * len(y) + 1e-12)
    return [int(j) for j in np.argsort(corr)[::-1][:k]]


def _numpy_noise_like(weights, rng, scale: float):
    return [rng.normal(0.0, scale, w.shape) for w in weights]


# ------------------------------------------------------------------ data faults

def _label_noise(c, rate, rng, **_):
    c = _copy_client(c)
    for s in SPLITS:
        flip = rng.random(len(c.y[s])) < rate
        c.y[s][flip] = 1 - c.y[s][flip]
    return c, {"flip_rate": rate}


def _prevalence_shift(c, odds_ratio, rng, **_):
    keep = {s: (c.y[s] == 1) | (rng.random(len(c.y[s])) < 1.0 / odds_ratio) for s in SPLITS}
    before = float(c.y["train"].mean())
    out = _subset(c, keep)
    return out, {"odds_ratio": odds_ratio, "train_prevalence_before": round(before, 4),
                 "train_prevalence_after": round(float(out.y["train"].mean()), 4)}


def subgroup_definition(data: FederatedDataset) -> dict:
    """The subgroup used by ``subgroup_bias``: a binary attribute's minority value,
    otherwise the lower half of the first sensitive attribute."""
    attrs = list(data.clients[0].groups["train"].keys())
    if not attrs:
        raise ValueError("subgroup_bias needs at least one sensitive attribute")
    for name in ("Sex", "Gender"):
        if name in attrs:
            attr = name
            break
    else:
        attr = attrs[0]
    values = np.concatenate([c.groups["train"][attr] for c in data.clients])
    uniq, counts = np.unique(values, return_counts=True)
    if len(uniq) <= 2:
        return {"attribute": attr, "rule": "==", "value": uniq[np.argmin(counts)].item()}
    return {"attribute": attr, "rule": "<=", "value": float(np.median(values.astype(float)))}


def in_subgroup(values: np.ndarray, sub: dict) -> np.ndarray:
    if sub["rule"] == "==":
        return values == sub["value"]
    return values.astype(float) <= sub["value"]


def _subgroup_bias(c, rate, rng, subgroup, **_):
    c = _copy_client(c)
    for s in SPLITS:
        mask = in_subgroup(c.groups[s][subgroup["attribute"]], subgroup) & (c.y[s] == 1)
        c.y[s][mask & (rng.random(len(c.y[s])) < rate)] = 0
    return c, {"miss_rate": rate, "subgroup": subgroup}


def artefact_features(data: FederatedDataset, k: int = 2) -> List[dict]:
    """Features most like continuous measurements, each with a sentinel code that
    is out of range: 0 if 0 never occurs, else 99, else 999."""
    raw = _raw_train(data)
    n_unique = [len(np.unique(raw[:, j].round(6))) for j in range(raw.shape[1])]
    out = []
    for j in np.argsort(n_unique)[::-1][:k]:
        lo, hi = raw[:, j].min(), raw[:, j].max()
        sentinel = 0.0 if lo > 0.5 else (99.0 if hi < 99 else 999.0)
        out.append({"feature": data.feature_names[j], "index": int(j), "sentinel": sentinel,
                    "sentinel_scaled": _to_scaled(data, int(j), sentinel)})
    return out


def _missing_artefact(c, frac, rng, features, **_):
    c = _copy_client(c)
    for s in SPLITS:
        rows = rng.random(len(c.y[s])) < frac
        for f in features:
            c.X[s][rows, f["index"]] = f["sentinel_scaled"]
    return c, {"row_fraction": frac, "features": features}


DATA_FAULTS = {
    "label_noise": _label_noise,
    "prevalence_shift": _prevalence_shift,
    "subgroup_bias": _subgroup_bias,
    "missing_artefact": _missing_artefact,
}


# ---------------------------------------------------------------- update faults

def _poison_hook(targets, factor):
    def hook(rnd, name, local, global_model):
        if name not in targets:
            return
        if isinstance(local, NumpyModel):
            local.weights = [g - factor * (l - g) for l, g in zip(local.weights, global_model.weights)]
        else:
            local._update = scale_tree_leaves(local._update, -factor)
    return hook


def _free_rider_hook(targets, rounds, scenario):
    def hook(rnd, name, local, global_model):
        if name not in targets or rnd not in rounds:
            return
        rng = _rng(scenario, rnd, targets.index(name))
        if isinstance(local, NumpyModel):
            local.weights = [g + n for g, n in
                             zip(global_model.weights, _numpy_noise_like(global_model.weights, rng, 1e-4))]
        else:
            local._update = scale_tree_leaves(local._update, 0.0)
    return hook


def _instability_hook(targets, sigma, scenario):
    def hook(rnd, name, local, global_model):
        if name not in targets:
            return
        rng = _rng(scenario, rnd, targets.index(name))
        if isinstance(local, NumpyModel):
            delta = [l - g for l, g in zip(local.weights, global_model.weights)]
            norm = float(np.sqrt(sum(np.sum(d ** 2) for d in delta)))
            size = sum(d.size for d in delta)
            noise = _numpy_noise_like(delta, rng, sigma * norm / np.sqrt(size))
            local.weights = [g + d + n for g, d, n in zip(global_model.weights, delta, noise)]
        else:
            n_leaves = len(tree_leaves(local._update))
            local._update = scale_tree_leaves(local._update, 1.0, rng.normal(0.0, sigma, n_leaves))
    return hook


# ------------------------------------------------------------------------ plant

def plant(scenario: Scenario, data: FederatedDataset) -> Planted:
    """Plant ``scenario``'s fault into ``data`` (which must be loaded with the scenario's seed)."""
    truth = {"fault": scenario.fault, "severity": scenario.severity, "level": scenario.level,
             "clients": [], "onset_round": None, "details": {}}
    if scenario.fault == CLEAN:
        return Planted(scenario, data, data, truth)

    rng = _rng(scenario)
    n = scenario.n_affected or default_n_affected(len(data.clients))
    targets = choose_clients(data, n, rng)
    level = scenario.level
    truth.update(clients=targets, onset_round=1)
    fault = scenario.fault

    if fault in DATA_FAULTS:
        ctx = {}
        if fault == "subgroup_bias":
            ctx["subgroup"] = subgroup_definition(data)
        if fault == "missing_artefact":
            ctx["features"] = artefact_features(data)
        new, details = {}, {}
        for i, name in enumerate(targets):
            new[name], d = DATA_FAULTS[fault](data.client(name), level,
                                              _rng(scenario, 100 + i), **ctx)
            details[name] = d
        truth["details"] = details
        faulty = _with_clients(data, new, scenario.id)
        return Planted(scenario, faulty, data, truth)

    if fault == "drift":
        onset = scenario.rounds // 2 + 1
        idx = sorted(top_features(data, 3))
        signs = rng.choice([-1.0, 1.0], size=len(idx))
        new = {}
        for name in targets:
            c = _copy_client(data.client(name))
            for s in SPLITS:
                c.X[s][:, idx] += level * signs
            new[name] = c
        drifted = _with_clients(data, new, scenario.id)
        truth.update(onset_round=onset, details={
            "shift_sd": level, "features": [data.feature_names[j] for j in idx],
            "signs": signs.tolist()})
        return Planted(scenario, drifted, data, truth,
                       data_schedule=lambda rnd: drifted if rnd >= onset else data)

    if fault == "poisoning":
        truth["details"] = {"scale": -level}
        return Planted(scenario, data, data, truth, update_hook=_poison_hook(targets, level))

    if fault == "free_rider":
        k = max(1, int(round(level * scenario.rounds)))
        rounds = sorted((rng.choice(scenario.rounds, size=k, replace=False) + 1).tolist())
        truth.update(onset_round=rounds[0], details={"rounds": rounds})
        return Planted(scenario, data, data, truth,
                       update_hook=_free_rider_hook(targets, set(rounds), scenario))

    if fault == "explanation_instability":
        truth["details"] = {"noise_to_update_ratio": level}
        return Planted(scenario, data, data, truth,
                       update_hook=_instability_hook(targets, level, scenario))

    raise AssertionError(fault)


def benchmark_grid(dataset: str, model: str, seeds: List[int], rounds: int = 20,
                   strategy: str = "fedavg", faults=FAULTS,
                   severities=SEVERITY_LEVELS, clean_seeds: Optional[List[int]] = None) -> List[Scenario]:
    """All FedFault scenarios: every fault x severity x seed, plus clean runs."""
    base = dict(dataset=dataset, model=model, rounds=rounds, strategy=strategy)
    grid = [Scenario(CLEAN, seed=s, **base) for s in (clean_seeds or seeds)]
    grid += [Scenario(f, sev, seed=s, **base) for f in faults for sev in severities for s in seeds]
    return grid
