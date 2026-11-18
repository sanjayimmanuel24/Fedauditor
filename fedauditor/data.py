"""Federated datasets: one ``ClientData`` per client, each with local splits.

Every client splits its own rows into train / calibration / test, so audits can
be run per client and the pooled test set is simply the union of client test
sets. Two preprocessing steps need cross-client statistics and both use only
aggregates, never rows:

* imputation of missing values with the federated mean of each feature, and
* standardisation with ``FederatedScaler`` (per-client n, sum, sum of squares).

Datasets:
* ``brfss``     - CDC BRFSS, one client per US state (main benchmark).
* ``pima``      - PIMA, three age-cohort clients (continuity with FedHealth).
* ``uci_early`` - UCI Early Stage, three age-cohort clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import load_dataset, partition_client_indices
from federated_scaler import (
    FederatedScaler,
    local_sufficient_statistics,
    pool_sufficient_statistics,
)
from fedauditor import brfss

SPLITS = ("train", "cal", "test")


@dataclass
class ClientData:
    name: str
    X: Dict[str, np.ndarray]                 # split -> standardised features
    y: Dict[str, np.ndarray]                 # split -> 0/1 labels
    groups: Dict[str, Dict[str, np.ndarray]]  # split -> attribute -> raw values

    @property
    def n_train(self) -> int:
        return int(len(self.y["train"]))

    @property
    def prevalence(self) -> float:
        return float(self.y["train"].mean())


@dataclass
class FederatedDataset:
    key: str
    name: str
    feature_names: List[str]
    clients: List[ClientData]
    scaler: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def client_names(self) -> List[str]:
        return [c.name for c in self.clients]

    def client(self, name: str) -> ClientData:
        for c in self.clients:
            if c.name == name:
                return c
        raise KeyError(name)

    def pooled(self, split: str = "test") -> tuple[np.ndarray, np.ndarray]:
        X = np.vstack([c.X[split] for c in self.clients])
        y = np.concatenate([c.y[split] for c in self.clients])
        return X, y

    def merged(self) -> "FederatedDataset":
        """All clients as one: the centralised reference (same rows, same splits)."""
        X = {s: np.vstack([c.X[s] for c in self.clients]) for s in SPLITS}
        y = {s: np.concatenate([c.y[s] for c in self.clients]) for s in SPLITS}
        attrs = self.clients[0].groups["train"].keys()
        groups = {
            s: {a: np.concatenate([c.groups[s][a] for c in self.clients]) for a in attrs}
            for s in SPLITS
        }
        return FederatedDataset(
            key=self.key, name=f"{self.name} (centralised)",
            feature_names=list(self.feature_names),
            clients=[ClientData("central", X, y, groups)],
            scaler=self.scaler, notes=list(self.notes),
        )

    def summary(self) -> List[dict]:
        return [
            {"client": c.name, "n_train": c.n_train, "n_cal": len(c.y["cal"]),
             "n_test": len(c.y["test"]), "prevalence": round(c.prevalence, 4)}
            for c in self.clients
        ]


def _split(frame: pd.DataFrame, target: str, seed: int, test_size: float, cal_fraction: float):
    y = frame[target].to_numpy()
    strat = y if np.bincount(y, minlength=2).min() >= 4 else None
    rest, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=strat)
    y_rest = rest[target].to_numpy()
    strat = y_rest if np.bincount(y_rest, minlength=2).min() >= 4 else None
    train, cal = train_test_split(rest, test_size=cal_fraction, random_state=seed, stratify=strat)
    return {"train": train, "cal": cal, "test": test}


def build_from_frame(
    frame: pd.DataFrame,
    *,
    key: str,
    name: str,
    client_column: str,
    features: List[str],
    target: str = "Outcome",
    sensitive: Iterable[str] = (),
    seed: int = 20,
    test_size: float = 0.2,
    cal_fraction: float = 0.25,
    min_rows: int = 40,
    min_positives: int = 5,
    max_rows_per_client: Optional[int] = None,
    clients: Optional[List[str]] = None,
) -> FederatedDataset:
    """Turn a frame with a client column into a ``FederatedDataset``."""
    notes: List[str] = []
    sensitive = [s for s in sensitive if s in frame.columns]
    names = clients if clients is not None else sorted(frame[client_column].dropna().unique())

    raw: Dict[str, Dict[str, pd.DataFrame]] = {}
    for client in names:
        part = frame[frame[client_column] == client]
        if max_rows_per_client and len(part) > max_rows_per_client:
            part = part.sample(n=max_rows_per_client, random_state=seed)
        positives = int(part[target].sum())
        if len(part) < min_rows or positives < min_positives or len(part) - positives < min_positives:
            notes.append(f"skipped client {client}: {len(part)} rows, {positives} positive")
            continue
        raw[str(client)] = _split(part, target, seed, test_size, cal_fraction)
    if len(raw) < 2:
        raise ValueError(f"only {len(raw)} usable clients; need at least 2")

    # Federated mean imputation: clients share per-feature non-missing counts and sums.
    counts = np.zeros(len(features))
    sums = np.zeros(len(features))
    for parts in raw.values():
        values = parts["train"][features].to_numpy(dtype=float)
        counts += (~np.isnan(values)).sum(axis=0)
        sums += np.nansum(values, axis=0)
    fill = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    def matrix(part: pd.DataFrame) -> np.ndarray:
        values = part[features].to_numpy(dtype=float)
        return np.where(np.isnan(values), fill, values)

    stats = [local_sufficient_statistics(matrix(p["train"])) for p in raw.values()]
    scaler = FederatedScaler().fit_from_statistics(pool_sufficient_statistics(stats))

    built = []
    for client, parts in raw.items():
        built.append(ClientData(
            name=client,
            X={s: scaler.transform(matrix(parts[s])) for s in SPLITS},
            y={s: parts[s][target].to_numpy(dtype=int) for s in SPLITS},
            groups={s: {a: parts[s][a].to_numpy() for a in sensitive} for s in SPLITS},
        ))
    return FederatedDataset(
        key=key, name=name, feature_names=list(features), clients=built,
        scaler=scaler.to_dict(), notes=notes,
    )


def load_federated(key: str = "brfss", seed: int = 20, **kwargs) -> FederatedDataset:
    """Load a registered dataset as a federation.

    BRFSS kwargs: ``year``, ``states`` (list of postal codes), ``min_rows``,
    ``max_rows_per_client``. PIMA/UCI kwargs: ``n_clients``.
    """
    if key == "brfss":
        year = kwargs.pop("year", 2023)
        frame = kwargs.pop("frame", None)
        if frame is None:
            frame = brfss.load_brfss(year)
        return build_from_frame(
            frame, key="brfss", name=f"CDC BRFSS {year}",
            client_column=brfss.CLIENT_COLUMN, features=brfss.feature_names(frame),
            sensitive=brfss.SENSITIVE, seed=seed,
            clients=kwargs.pop("states", None),
            min_rows=kwargs.pop("min_rows", 1000), **kwargs,
        )

    if key in ("pima", "uci_early"):
        n_clients = kwargs.pop("n_clients", 3)
        ds = load_dataset(key=key)
        chunks, status = partition_client_indices(
            ds.frame["Age"], ds.y, n_clients=n_clients, mode="age",
            seed=seed, bands=ds.age_bands, return_status=True,
        )
        frame = ds.frame.copy()
        frame["Client"] = None
        for i, chunk in enumerate(chunks):
            frame.iloc[chunk, frame.columns.get_loc("Client")] = f"cohort_{i + 1}"
        sensitive = [c for c in ("Age", "Gender", "AgeBand") if c in frame.columns]
        out = build_from_frame(
            frame, key=key, name=ds.name, client_column="Client",
            features=ds.features, sensitive=sensitive, seed=seed, **kwargs,
        )
        out.notes = ds.notes + [f"partition: {status['reason']}"] + out.notes
        return out

    raise KeyError(f"unknown dataset {key!r}; expected brfss, pima or uci_early")
