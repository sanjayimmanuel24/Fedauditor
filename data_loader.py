"""Dataset loading, cleaning and client partitioning.

Two public datasets are supported:

* ``pima`` (default) -- the PIMA Indians Diabetes dataset, 768 records.
* ``uci_early`` -- the UCI Early Stage Diabetes Risk Prediction dataset, 520
  questionnaire records. Used only for external validation of the audit, not
  as a replacement for the primary dataset.

Both are normalised to the same shape: the dataset's feature columns plus an
``Outcome`` target and a derived ``AgeBand`` column. The synthetic generator is
retained only as a PIMA fallback so the application can boot on a machine
without the CSV; every result computed from it is flagged as synthetic and
should not be reported as a finding. The second dataset never falls back.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from config import (
    AGE_BANDS,
    DATASETS,
    DEFAULT_DATASET,
    TARGET,
    ZERO_AS_MISSING,
)

TARGET_COLUMN = "Outcome"


@dataclass
class Dataset:
    frame: pd.DataFrame
    key: str = DEFAULT_DATASET
    name: str = "PIMA Indians Diabetes"
    features: List[str] = field(default_factory=list)
    feature_labels: Dict[str, str] = field(default_factory=dict)
    zero_as_missing: List[str] = field(default_factory=list)
    age_bands: List[tuple] = field(default_factory=lambda: list(AGE_BANDS))
    citation: str = ""
    source: str = "pima"
    notes: List[str] = field(default_factory=list)

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.features]

    @property
    def y(self) -> pd.Series:
        return self.frame[TARGET_COLUMN]

    @property
    def age_band(self) -> pd.Series:
        return self.frame["AgeBand"]


def _generate_synthetic(n_samples: int = 768, seed: int = 42) -> pd.DataFrame:
    """Realistic stand-in for PIMA. Used only when the CSV is absent."""
    rng = np.random.default_rng(seed)

    pregnancies = rng.poisson(3.8, n_samples)
    glucose = np.clip(rng.normal(121, 31, n_samples), 44, 199)
    blood_pressure = np.clip(rng.normal(69, 12, n_samples), 24, 122)
    skin = np.clip(rng.normal(20.5, 15, n_samples), 7, 99)
    insulin = np.clip(rng.exponential(80, n_samples), 14, 846)
    bmi = np.clip(rng.normal(32, 7, n_samples), 18, 67)
    dpf = rng.beta(2, 5, n_samples) * 2.5
    age = rng.exponential(12, n_samples) + 21

    logits = (
        -5.5
        + 0.12 * pregnancies
        + 0.035 * glucose
        - 0.012 * blood_pressure
        + 0.005 * skin
        - 0.0005 * insulin
        + 0.08 * bmi
        + 0.95 * dpf
        + 0.02 * age
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    outcome = (probs > rng.uniform(0, 1, n_samples)).astype(int)

    return pd.DataFrame(
        {
            "Pregnancies": pregnancies,
            "Glucose": glucose.round(1),
            "BloodPressure": blood_pressure.round(1),
            "SkinThickness": skin.round(1),
            "Insulin": insulin.round(1),
            "BMI": bmi.round(1),
            "DiabetesPedigreeFunction": dpf.round(3),
            "Age": age.round().astype(int),
            "Outcome": outcome,
        }
    )


def age_band_of(age: float, bands: List[tuple] | None = None) -> str:
    bands = list(bands) if bands is not None else list(AGE_BANDS)
    for label, low, high in bands:
        if low <= age < high:
            return label
    return bands[-1][0]


def dataset_keys() -> List[str]:
    """Registered dataset keys, primary dataset first."""
    keys = [key for key in DATASETS if key != DEFAULT_DATASET]
    return [DEFAULT_DATASET] + sorted(keys)


def _resolve_spec(key: str | None, path: str | None) -> tuple[str, dict]:
    key = (key or DEFAULT_DATASET).lower()
    if key not in DATASETS:
        raise ValueError(
            f"unknown dataset {key!r}; choose from {sorted(DATASETS)}"
        )
    spec = dict(DATASETS[key])
    if path:
        spec["path"] = path
    return key, spec


def _encode_yes_no(values: pd.Series, column: str) -> pd.Series:
    cleaned = values.astype(str).str.strip().str.lower()
    encoded = cleaned.map({"yes": 1, "no": 0})
    if encoded.isna().any():
        unexpected = sorted(set(cleaned[encoded.isna()]))
        raise ValueError(
            f"column {column!r} contains values that are neither Yes nor No: "
            f"{unexpected[:5]}"
        )
    return encoded.astype(int)


def _normalise_uci(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Encode the UCI questionnaire into numeric columns and a 0/1 target."""
    frame = frame.rename(columns=lambda name: str(name).strip()).copy()
    source_target = spec["source_target"]
    required = list(spec["features"]) + [source_target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"UCI dataset is missing columns: {missing}")

    for column in spec.get("binary_columns", []):
        frame[column] = _encode_yes_no(frame[column], column)

    gender_column = spec.get("gender_column")
    if gender_column:
        gender = frame[gender_column].astype(str).str.strip().str.lower()
        encoded = gender.map({"male": 1, "female": 0})
        if encoded.isna().any():
            unexpected = sorted(set(gender[encoded.isna()]))
            raise ValueError(
                f"column {gender_column!r} contains unexpected values: "
                f"{unexpected[:5]}"
            )
        frame[gender_column] = encoded.astype(int)

    labels = frame[source_target].astype(str).str.strip().str.lower()
    outcome = labels.map({"positive": 1, "negative": 0})
    if outcome.isna().any():
        unexpected = sorted(set(labels[outcome.isna()]))
        raise ValueError(
            f"target {source_target!r} contains unexpected values: "
            f"{unexpected[:5]}"
        )
    frame[TARGET_COLUMN] = outcome.astype(int)
    frame["Age"] = pd.to_numeric(frame["Age"], errors="coerce")
    if frame["Age"].isna().any():
        raise ValueError("UCI dataset has non-numeric or missing Age values")
    return frame[list(spec["features"]) + [TARGET_COLUMN]]


def _normalise_pima(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    source_target = spec["source_target"]
    required = list(spec["features"]) + [source_target]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"PIMA dataset is missing columns: {missing}")
    frame = frame.rename(columns={source_target: TARGET_COLUMN}).copy()
    try:
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError("PIMA target column must be 0/1") from exc
    return frame[list(spec["features"]) + [TARGET_COLUMN]]


def load_dataset(
    path: str | None = None, key: str | None = None
) -> Dataset:
    """Load and clean a registered dataset, masking impossible zeros.

    Missing values are *not* filled here: the imputation statistics must be
    fitted after the train/test split, so this function leaves NaN in place.
    """
    key, spec = _resolve_spec(key, path)
    path = spec["path"]
    notes: List[str] = []

    if os.path.exists(path):
        frame = pd.read_csv(path)
        source = key
        if key == "uci_early":
            frame = _normalise_uci(frame, spec)
        else:
            frame = _normalise_pima(frame, spec)
    else:
        if not spec.get("allow_synthetic", False):
            raise FileNotFoundError(
                f"Dataset {key!r} not found at {path}. This dataset is used for "
                "external validation and is never replaced by synthetic data. "
                "Download the CSV from the UCI Machine Learning Repository "
                "(dataset 529) and save it at that path."
            )
        frame = _generate_synthetic()
        source = "synthetic"
        notes.append(
            "data/pima_diabetes.csv not found - running on the synthetic "
            "fallback. Results are illustrative only."
        )

    frame = frame.copy()

    # Treat physiologically impossible zeros as missing. The imputation values
    # themselves are fitted on the training split only (``fit_medians``);
    # filling them here would leak test-set information into training.
    for column in spec.get("zero_as_missing", []):
        if column in frame.columns:
            frame[column] = frame[column].mask(frame[column] == 0)

    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(int)
    bands = list(spec.get("age_bands", AGE_BANDS))
    frame["AgeBand"] = frame["Age"].apply(
        lambda age: age_band_of(age, bands)
    )

    return Dataset(
        frame=frame,
        key=key,
        name=spec["name"],
        features=list(spec["features"]),
        feature_labels=dict(spec.get("feature_labels", {})),
        zero_as_missing=list(spec.get("zero_as_missing", [])),
        age_bands=bands,
        citation=spec.get("citation", ""),
        source=source,
        notes=notes,
    )


def fit_medians(frame: pd.DataFrame, columns=None) -> dict:
    """Column medians fitted on the training split and nothing else."""
    columns = list(columns) if columns is not None else list(ZERO_AS_MISSING)
    medians: dict[str, float] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        value = frame[column].median()
        # A column that is entirely missing would yield NaN; fall back to zero
        # so the caller never has to special-case it.
        medians[column] = float(value) if value == value else 0.0
    return medians


def apply_medians(frame: pd.DataFrame, medians: dict) -> pd.DataFrame:
    """Fill missing values using medians fitted elsewhere."""
    out = frame.copy()
    for column, value in medians.items():
        if column in out.columns:
            out[column] = out[column].fillna(value)
    return out


def _band_to_client_map(n_clients: int, bands: List[tuple] | None = None) -> dict:
    """Assign age bands to simulated clients.

    With three bands and three clients each client serves exactly one cohort,
    which keeps the subgroup audit and the client list identical and therefore
    easy to reason about. For other client counts the bands are merged into
    contiguous buckets.
    """
    bands = list(bands) if bands is not None else list(AGE_BANDS)
    labels = [label for label, _, _ in bands]
    mapping: dict[str, int] = {}
    if n_clients >= len(labels):
        for index, label in enumerate(labels):
            mapping[label] = min(index, n_clients - 1)
    else:
        for client_id, bucket in enumerate(np.array_split(np.arange(len(labels)), n_clients)):
            for index in bucket:
                mapping[labels[int(index)]] = client_id
    return mapping


def partition_client_indices(
    ages: pd.Series,
    labels: pd.Series,
    n_clients: int,
    mode: str = "age",
    seed: int = 20,
    bands: List[tuple] | None = None,
    return_status: bool = False,
):
    """Split training rows across simulated clients.

    ``age`` mode assigns rows by clinical age cohort, giving each client a
    genuinely different patient population. This is the non-IID regime that
    makes calibration drift and subgroup harm measurable.

    ``iid`` mode is a uniform random shard, provided for comparison.

    If an age cohort is too small or class-degenerate the function falls back to
    IID shards. The fallback is reported in ``status`` when
    ``return_status=True`` so a run can never silently claim to be an age split
    when it is not.
    """
    rng = np.random.default_rng(seed)
    n = len(ages)
    indices = np.arange(n)
    requested_mode = mode

    def finish(chunks: List[np.ndarray], fallback: bool, reason: str):
        status = {
            "requested_mode": requested_mode,
            "effective_mode": "iid" if (requested_mode == "iid" or fallback) else "age",
            "fallback": bool(fallback),
            "reason": reason,
        }
        if return_status:
            return chunks, status
        return chunks

    if mode == "iid" or n_clients < 2:
        shuffled = rng.permutation(indices)
        chunks = [np.sort(chunk) for chunk in np.array_split(shuffled, n_clients)]
        return finish(chunks, fallback=False, reason="requested IID partition")

    labels_by_band = list(bands) if bands is not None else list(AGE_BANDS)
    band_values = ages.apply(lambda age: age_band_of(age, labels_by_band)).to_numpy()
    mapping = _band_to_client_map(n_clients, labels_by_band)

    chunks: List[np.ndarray] = []
    for client_id in range(n_clients):
        assigned = [band for band, owner in mapping.items() if owner == client_id]
        mask = np.isin(band_values, assigned)
        chunks.append(np.sort(indices[mask]))

    # Guard against a cohort that is empty, too small, or class-degenerate.
    # Falling back to equal-size shards is preferable to an untrainable client.
    y = labels.to_numpy()
    for client_id, chunk in enumerate(chunks):
        positives = int(y[chunk].sum())
        negatives = len(chunk) - positives
        if len(chunk) < 40 or positives < 5 or negatives < 5:
            shuffled = rng.permutation(indices)
            chunks = [
                np.sort(c) for c in np.array_split(shuffled, n_clients)
            ]
            return finish(
                chunks,
                fallback=True,
                reason=(
                    f"client {client_id + 1} cohort has {len(chunk)} rows "
                    f"({positives} positive, {negatives} negative); "
                    "fell back to IID shards"
                ),
            )

    for chunk in chunks:
        rng.shuffle(chunk)

    return finish(chunks, fallback=False, reason="age-cohort partition")
