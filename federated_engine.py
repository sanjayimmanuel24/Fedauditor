"""Federated Averaging engine with an emphasis on auditability.

Design notes:

* A registered public dataset is loaded from disk instead of being regenerated
  from ``np.random``. The default is PIMA; ``dataset_key="uci_early"`` loads
  the UCI Early Stage questionnaire for external validation.
* Clients are partitioned by **age cohort**, producing the non-IID regime that
  makes calibration drift and subgroup harm measurable. An IID mode remains
  available for comparison.
* Missing-value imputation is fitted on the training split only.
* The feature scaler is fitted **federatedly** from per-client
  ``(n, sum, sum of squares)`` statistics, so the server never sees the pooled
  data matrix.
* Each client holds out a local calibration split. Those rows are used for
  calibration *and* for selecting fairness thresholds, so the evaluation set is
  never used to choose a threshold.
* Every round also reports a centralised reference model, so the cost of
  federation is visible.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import (
    AGGREGATION_WEIGHTING,
    BATCH_SIZE,
    BOOTSTRAP_SAMPLES,
    CALIBRATION_BINS,
    CLIENT_CALIBRATION_FRACTION,
    DECISION_THRESHOLD,
    DEFAULT_DATASET,
    DEFAULT_WARMUP_ROUNDS,
    FAIRNESS_THRESHOLD_STEPS,
    L2_PENALTY,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    N_CLIENTS,
    PARTITION_MODE,
    RANDOM_SEED,
    TEST_SIZE,
    WEIGHTING_MODES,
)
from data_loader import (
    apply_medians,
    fit_medians,
    load_dataset,
    partition_client_indices,
)
from federated_calibration import calibration_report
from federated_explain import contributions, instability_report, patient_robustness
from federated_fairness import (
    fairness_report,
    group_counts_over_grid,
    pool_group_tables,
    threshold_grid,
)
from federated_metrics import (
    binarize,
    confusion_counts,
    full_metrics,
    sigmoid,
)
from federated_scaler import (
    FederatedScaler,
    local_sufficient_statistics,
    pool_sufficient_statistics,
)


class FederatedEngine:
    def __init__(
        self,
        data_path: str | None = None,
        partition_mode: str | None = None,
        n_clients: int | None = None,
        seed: int | None = None,
        weighting: str | None = None,
        dataset_key: str | None = None,
    ):
        self.seed = int(RANDOM_SEED if seed is None else seed)
        self.partition_mode = partition_mode or PARTITION_MODE
        self.n_clients = int(n_clients or N_CLIENTS)
        self.dataset_key = dataset_key or DEFAULT_DATASET

        weighting = (weighting or AGGREGATION_WEIGHTING).lower()
        # "prevalence" is accepted as a readable alias for positive-case
        # weighting; "positive" is the canonical name.
        if weighting == "prevalence":
            weighting = "positive"
        if weighting not in WEIGHTING_MODES:
            raise ValueError(
                f"unknown aggregation weighting {weighting!r}; "
                f"choose from {WEIGHTING_MODES}"
            )
        self.weighting = weighting

        self.local_epochs = LOCAL_EPOCHS
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.l2 = L2_PENALTY

        dataset = load_dataset(data_path, key=self.dataset_key)
        self.data_source = dataset.source
        self.data_notes = dataset.notes
        self.dataset_name = dataset.name
        self.dataset_citation = dataset.citation
        self.feature_names = list(dataset.features)
        self.feature_labels = dict(dataset.feature_labels)
        self.zero_as_missing = list(dataset.zero_as_missing)
        self.age_bands = list(dataset.age_bands)
        self.frame = dataset.frame
        self.requested_partition_mode = self.partition_mode
        self.partition_fallback = False
        self.partition_reason = ""

        self._centralised_cache: dict | None = None
        self._prepare()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _prepare(self) -> None:
        frame = self.frame
        # AgeBand travels with the split so that subgroup labels and client
        # cohorts stay aligned with the rows; it is never a model feature.
        X = frame[self.feature_names + ["AgeBand"]]
        y = frame["Outcome"]

        X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=self.seed,
            stratify=y,
        )

        X_train_df = X_train_df.reset_index(drop=True)
        X_test_df = X_test_df.reset_index(drop=True)

        # Imputation statistics come from the training split only, so test rows
        # cannot influence how missing values are filled.
        self.impute_medians = fit_medians(X_train_df, self.zero_as_missing)
        self.X_train_raw = apply_medians(
            X_train_df[self.feature_names], self.impute_medians
        )
        self.X_test_raw = apply_medians(
            X_test_df[self.feature_names], self.impute_medians
        )
        self.y_train = y_train_series.to_numpy().astype(int)
        self.y_test = y_test_series.to_numpy().astype(int)
        self.test_bands = X_test_df["AgeBand"].to_numpy()

        requested_mode = self.partition_mode if self.n_clients > 1 else "iid"
        client_row_indices, self.partition_status = partition_client_indices(
            X_train_df["Age"].reset_index(drop=True),
            pd.Series(self.y_train),
            self.n_clients,
            mode=requested_mode,
            seed=self.seed,
            bands=self.age_bands,
            return_status=True,
        )
        self.requested_partition_mode = self.partition_status["requested_mode"]
        self.partition_mode = self.partition_status["effective_mode"]
        self.partition_fallback = bool(self.partition_status["fallback"])
        self.partition_reason = str(self.partition_status["reason"])

        # Split each client into a local training part and a local calibration
        # part before fitting anything, so the scaler is fitted on training rows
        # only.
        client_train_rows: List[np.ndarray] = []
        client_cal_rows: List[np.ndarray] = []

        for client_id, rows in enumerate(client_row_indices):
            labels = self.y_train[rows]
            if len(np.unique(labels)) < 2:
                raise ValueError(
                    f"client {client_id} received a single-class shard; "
                    "increase data or change partition mode"
                )

            try:
                train_rows, cal_rows = train_test_split(
                    rows,
                    test_size=CLIENT_CALIBRATION_FRACTION,
                    random_state=self.seed + client_id,
                    stratify=labels,
                )
            except ValueError:
                train_rows, cal_rows = train_test_split(
                    rows,
                    test_size=CLIENT_CALIBRATION_FRACTION,
                    random_state=self.seed + client_id,
                )

            client_train_rows.append(np.sort(train_rows))
            client_cal_rows.append(np.sort(cal_rows))

        # ------------------------------------------------------------------
        # Federated standardisation
        # ------------------------------------------------------------------
        # Each "hospital" computes column sums and sums of squares over its own
        # training rows. The server sums those and derives the pooled mean and
        # variance. This is exactly the sklearn fit, without pooling the data.
        self.scaler_update = [
            local_sufficient_statistics(self.X_train_raw.iloc[rows])
            for rows in client_train_rows
        ]
        pooled_statistics = pool_sufficient_statistics(self.scaler_update)
        self.scaler = FederatedScaler().fit_from_statistics(pooled_statistics)

        self.X_train_scaled = self.scaler.transform(self.X_train_raw)
        self.X_test_scaled = self.scaler.transform(self.X_test_raw)

        # ------------------------------------------------------------------
        # Per-client views
        # ------------------------------------------------------------------
        self.client_train: List[Tuple[np.ndarray, np.ndarray]] = []
        self.client_cal: List[Tuple[np.ndarray, np.ndarray]] = []
        self.client_cal_bands: List[np.ndarray] = []
        self.client_meta: List[Dict] = []
        self.client_row_indices: List[np.ndarray] = []

        for client_id, rows in enumerate(client_row_indices):
            train_rows = client_train_rows[client_id]
            cal_rows = client_cal_rows[client_id]

            self.client_row_indices.append(rows)
            self.client_train.append(
                (self.X_train_scaled[train_rows], self.y_train[train_rows])
            )
            self.client_cal.append(
                (self.X_train_scaled[cal_rows], self.y_train[cal_rows])
            )
            # Group labels for the calibration rows, needed to build the
            # per-cohort confusion tables used for fairness threshold selection.
            self.client_cal_bands.append(
                X_train_df["AgeBand"].to_numpy()[cal_rows]
            )

            subset = X_train_df.iloc[rows]
            self.client_meta.append(
                {
                    "n_samples": int(len(rows)),
                    "n_train": int(len(train_rows)),
                    "n_calibration": int(len(cal_rows)),
                    "age_min": float(subset["Age"].min()),
                    "age_max": float(subset["Age"].max()),
                    "age_mean": round(float(subset["Age"].mean()), 1),
                    "prevalence": round(float(self.y_train[rows].mean()), 4),
                    "positives": int(self.y_train[rows].sum()),
                    "dominant_band": subset["AgeBand"].value_counts().idxmax(),
                    "aggregation_weight": round(
                        self._aggregation_weights(
                            np.array([len(r) for r in client_train_rows]),
                            np.array(
                                [
                                    float(self.y_train[r].mean())
                                    for r in client_train_rows
                                ]
                            ),
                        )[client_id],
                        4,
                    ),
                }
            )

        # Cohort backgrounds in scaled space: what a local explanation is
        # relative to. The sample-weighted mean gives the global background.
        #
        # The covariance is an aggregate sufficient statistic too: it lets the
        # server quantify how precisely a cohort mean is estimated without ever
        # receiving a row. ``cov / n`` is the sampling covariance of the mean.
        self.client_backgrounds: Dict[str, np.ndarray] = {}
        self.client_background_cov: Dict[str, np.ndarray] = {}
        self.client_background_n: Dict[str, int] = {}
        weighted_sum = np.zeros(len(self.feature_names))
        total = 0
        for client_id, (X_c, _) in enumerate(self.client_train):
            mean_vector = X_c.mean(axis=0)
            covariance = (
                np.atleast_2d(np.cov(X_c, rowvar=False, ddof=1))
                if len(X_c) > 1
                else np.zeros((self.n_features, self.n_features))
            )
            self.client_meta[client_id]["background"] = mean_vector
            self.client_meta[client_id]["background_cov"] = covariance
            self.client_meta[client_id]["background_n"] = int(len(X_c))
            band = self.client_meta[client_id]["dominant_band"]
            self.client_backgrounds[band] = mean_vector
            self.client_background_cov[band] = covariance
            self.client_background_n[band] = int(len(X_c))
            weighted_sum += mean_vector * len(X_c)
            total += len(X_c)

        self.global_background = weighted_sum / max(total, 1)

        # Map every age band onto the nearest client cohort so a new patient
        # always has a local background and a local background covariance.
        self.background_by_band: Dict[str, np.ndarray] = {}
        self.background_cov_by_band: Dict[str, np.ndarray] = {}
        self.background_n_by_band: Dict[str, int] = {}
        band_centres = {}
        for band in self.frame["AgeBand"].unique():
            ages = self.frame.loc[self.frame["AgeBand"] == band, "Age"]
            band_centres[band] = float(ages.mean())
        for band, centre in band_centres.items():
            if band in self.client_backgrounds:
                self.background_by_band[band] = self.client_backgrounds[band]
                self.background_cov_by_band[band] = self.client_background_cov[band]
                self.background_n_by_band[band] = self.client_background_n[band]
                continue
            nearest = min(
                range(self.n_clients),
                key=lambda c: abs(self.client_meta[c]["age_mean"] - centre),
            )
            nearest_band = self.client_meta[nearest]["dominant_band"]
            self.background_by_band[band] = self.client_meta[nearest]["background"]
            self.background_cov_by_band[band] = self.client_background_cov[nearest_band]
            self.background_n_by_band[band] = self.client_background_n[nearest_band]

        self.threshold_grid = threshold_grid(FAIRNESS_THRESHOLD_STEPS)

    # ------------------------------------------------------------------
    # Model helpers
    # ------------------------------------------------------------------
    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def initial_parameters(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros((1, self.n_features)), np.zeros(1)

    @staticmethod
    def logits(coef: np.ndarray, intercept: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float).ravel() + float(
            np.asarray(intercept).ravel()[0]
        )

    def predict_proba(
        self, coef: np.ndarray, intercept: np.ndarray, X: np.ndarray
    ) -> np.ndarray:
        return sigmoid(self.logits(coef, intercept, X))

    # ------------------------------------------------------------------
    # Aggregation weighting
    # ------------------------------------------------------------------
    def _aggregation_weights(
        self, sizes: np.ndarray, prevalences: np.ndarray
    ) -> np.ndarray:
        """Normalised client weights for the configured aggregation scheme."""
        sizes = np.asarray(sizes, dtype=float).ravel()
        prevalences = np.asarray(prevalences, dtype=float).ravel()

        if self.weighting == "uniform":
            weights = np.ones_like(sizes)
        elif self.weighting == "positive":
            # Influence proportional to the number of diabetic patients a
            # hospital contributes, not just the number of rows.
            weights = sizes * np.clip(prevalences, 1e-6, 1.0)
        else:  # "sample": standard FedAvg
            weights = sizes

        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            weights = np.ones_like(sizes)
            total = float(weights.sum())
        return weights / total

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------
    def _fit_local(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coef: np.ndarray,
        intercept: np.ndarray,
        client_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.seed + client_id)
        coef = np.asarray(coef, dtype=float).reshape(1, -1).copy()
        intercept = np.asarray(intercept, dtype=float).reshape(1).copy()
        n = len(y)

        for _ in range(self.local_epochs):
            order = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch = order[start : start + self.batch_size]
                Xb = X[batch]
                yb = y[batch].astype(float)
                z = Xb @ coef.ravel() + intercept[0]
                error = sigmoid(z) - yb
                grad_w = Xb.T @ error / len(batch) + self.l2 * coef.ravel()
                grad_b = float(error.mean())
                coef -= self.learning_rate * grad_w.reshape(1, -1)
                intercept -= self.learning_rate * grad_b

        return coef, intercept

    def train_local_client(
        self, client_id: int, coef_init: np.ndarray, intercept_init: np.ndarray
    ) -> tuple:
        """Train one client and return its update plus local holdout data.

        Returns ``(coef, intercept, local_accuracy, n_samples, cal_X, cal_y,
        prevalence)``. The prevalence slot lets aggregation schemes that need
        per-client outcome rates work without a second call.
        """
        X_c, y_c = self.client_train[client_id]
        coef, intercept = self._fit_local(X_c, y_c, coef_init, intercept_init, client_id)

        cal_X, cal_y = self.client_cal[client_id]
        local_probabilities = self.predict_proba(coef, intercept, cal_X)
        local_accuracy = float(
            np.mean(binarize(local_probabilities, DECISION_THRESHOLD) == cal_y)
        )
        prevalence = float(np.mean(y_c))
        return (
            coef,
            intercept,
            local_accuracy,
            len(y_c),
            cal_X,
            cal_y,
            prevalence,
        )

    def federated_average(self, updates) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted parameter average.

        ``updates`` entries are ``(coef, intercept, n_samples, prevalence)``.
        """
        sizes = np.array([update[2] for update in updates], dtype=float)
        prevalences = np.array([update[3] for update in updates], dtype=float)
        weights = self._aggregation_weights(sizes, prevalences)

        coef = np.zeros((1, self.n_features))
        intercept = np.zeros(1)
        for weight, update in zip(weights, updates):
            local_coef = np.asarray(update[0], dtype=float).reshape(1, -1)
            local_intercept = np.asarray(update[1], dtype=float).reshape(1)
            coef += local_coef * weight
            intercept += local_intercept * weight
        return coef, intercept

    # ------------------------------------------------------------------
    # Centralised reference
    # ------------------------------------------------------------------
    def centralised_baseline(self) -> dict:
        """Fully pooled logistic regression, the standard upper-bound reference.

        Unregularised and fitted with L-BFGS to convergence. Its purpose is to
        answer "what did federation cost?", not to be compute-matched to FedAvg.
        """
        if self._centralised_cache is None:
            model = LogisticRegression(penalty=None, max_iter=5000)
            model.fit(self.X_train_scaled, self.y_train)
            probabilities = model.predict_proba(self.X_test_scaled)[:, 1]
            metrics = full_metrics(self.y_test, probabilities, DECISION_THRESHOLD)
            self._centralised_cache = {
                "metrics": metrics,
                "coef": model.coef_.ravel().tolist(),
                "intercept": float(model.intercept_[0]),
                "n_train": int(len(self.y_train)),
                "n_test": int(len(self.y_test)),
                "regularisation": "none (converged)",
            }
        return self._centralised_cache

    # ------------------------------------------------------------------
    # One federated round
    # ------------------------------------------------------------------
    def run_round(
        self,
        current_coef: np.ndarray,
        current_intercept: np.ndarray,
        round_no: int | None = None,
    ) -> Dict[str, object]:
        updates = []
        local_accuracies = []
        local_meta = []
        cal_sets = []

        for client_id in range(self.n_clients):
            (
                coef_c,
                intercept_c,
                local_acc,
                n_samples,
                cal_X,
                cal_y,
                prevalence,
            ) = self.train_local_client(client_id, current_coef, current_intercept)
            updates.append((coef_c, intercept_c, n_samples, prevalence))
            local_accuracies.append(local_acc)
            local_meta.append(
                {
                    "client": client_id + 1,
                    "samples": int(n_samples),
                    "accuracy": round(local_acc, 6),
                    "weight": round(
                        float(
                            self._aggregation_weights(
                                np.array([u[2] for u in updates]),
                                np.array([u[3] for u in updates]),
                            )[-1]
                        ),
                        4,
                    ),
                }
            )
            cal_sets.append((cal_X, cal_y))

        new_coef, new_intercept = self.federated_average(updates)

        probabilities = self.predict_proba(new_coef, new_intercept, self.X_test_scaled)
        metrics = full_metrics(self.y_test, probabilities, DECISION_THRESHOLD)
        counts = confusion_counts(self.y_test, binarize(probabilities, DECISION_THRESHOLD))

        # Client-side calibration statistics, computed against the *aggregated*
        # model. Only these aggregate numbers would be transmitted.
        client_updates = []
        for cal_X, cal_y in cal_sets:
            client_updates.append(
                {
                    "probs": self.predict_proba(new_coef, new_intercept, cal_X),
                    "logits": self.logits(new_coef, new_intercept, cal_X),
                    "labels": cal_y,
                }
            )

        calibration = calibration_report(
            self.y_test, probabilities, client_updates, CALIBRATION_BINS
        )

        # Fairness thresholds are selected on pooled confusion counts from the
        # clients' local calibration splits, then scored on the test set. The
        # evaluation labels never take part in threshold selection.
        tuning_tables = []
        for client_id, (cal_X, cal_y) in enumerate(cal_sets):
            cal_probabilities = self.predict_proba(new_coef, new_intercept, cal_X)
            tuning_tables.append(
                group_counts_over_grid(
                    cal_y,
                    cal_probabilities,
                    self.client_cal_bands[client_id],
                    self.threshold_grid,
                )
            )
        pooled_tuning = pool_group_tables(tuning_tables)

        fairness = fairness_report(
            self.y_test,
            probabilities,
            self.test_bands,
            grid=self.threshold_grid,
            default_threshold=DECISION_THRESHOLD,
            tuning_counts=pooled_tuning,
        )
        explanation = instability_report(
            new_coef.ravel(),
            self.X_test_scaled,
            self.test_bands,
            self.background_by_band,
            self.global_background,
            self.feature_names,
            background_covariances=self.background_cov_by_band,
            background_counts=self.background_n_by_band,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=self.seed + (round_no or 0),
        )

        return {
            "round": round_no,
            "new_coef": new_coef,
            "new_intercept": new_intercept,
            "coef": new_coef.ravel().tolist(),
            "intercept": float(new_intercept[0]),
            "global_accuracy": metrics["accuracy"],
            "metrics": metrics,
            "local_accuracies": local_accuracies,
            "local_meta": local_meta,
            "confusion_matrix": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
            "probabilities": probabilities,
            "calibration": calibration,
            "fairness": fairness,
            "explanation": explanation,
            "centralised": self.centralised_baseline(),
            "fairness_tuning": fairness["tuning_source"],
        }

    # ------------------------------------------------------------------
    # Single-patient inference
    # ------------------------------------------------------------------
    def predict(
        self,
        raw_features: Dict[str, float],
        coef: np.ndarray,
        intercept: np.ndarray,
        background: str = "global",
        threshold: float = DECISION_THRESHOLD,
    ) -> Dict[str, object]:
        frame = pd.DataFrame(
            [
                {
                    name: float(raw_features.get(name, 0.0))
                    for name in self.feature_names
                }
            ]
        )

        # A recorded 0 in these columns is physiologically impossible and was
        # treated as missing during training, so handle it the same way here
        # rather than scaling a literal 0.
        for column in self.zero_as_missing:
            if column in frame.columns and frame.loc[0, column] == 0:
                frame.loc[0, column] = np.nan
        frame = apply_medians(frame, self.impute_medians)

        scaled = self.scaler.transform(frame[self.feature_names])[0]
        probability = float(self.predict_proba(coef, intercept, scaled.reshape(1, -1))[0])
        prediction = int(probability >= threshold)

        age = float(raw_features.get("Age", 0.0))
        from data_loader import age_band_of

        band = age_band_of(age, self.age_bands)
        local_background = self.background_by_band.get(band, self.global_background)

        if background == "local":
            used_background = local_background
            used_label = f"local cohort ({band})"
        else:
            used_background = self.global_background
            used_label = "global cohort"

        phi = contributions(
            np.asarray(coef).ravel(), scaled, used_background, self.feature_names
        )
        phi_global = contributions(
            np.asarray(coef).ravel(), scaled, self.global_background, self.feature_names
        )
        phi_local = contributions(
            np.asarray(coef).ravel(), scaled, local_background, self.feature_names
        )

        local_covariance = self.background_cov_by_band.get(band)
        local_n = self.background_n_by_band.get(band)
        robustness = patient_robustness(
            np.asarray(coef).ravel(),
            scaled,
            self.global_background,
            local_background,
            local_covariance=local_covariance,
            local_n=local_n,
            n_boot=BOOTSTRAP_SAMPLES,
            seed=self.seed,
        )
        top_global = self.feature_names[int(robustness["top_global"])]
        top_local = self.feature_names[int(robustness["top_local"])]

        return {
            "prediction": "Diabetic" if prediction else "Not Diabetic",
            "probability_diabetic": probability,
            "confidence": probability if prediction else 1.0 - probability,
            "threshold": threshold,
            "age_band": band,
            "background_used": used_label,
            "contributions": {row["feature"]: row["contribution"] for row in phi},
            "ranked_contributions": phi,
            "contributions_global": {
                row["feature"]: row["contribution"] for row in phi_global
            },
            "contributions_local": {
                row["feature"]: row["contribution"] for row in phi_local
            },
            "top_global": top_global,
            "top_local": top_local,
            "explanation_changed": bool(robustness["changed"]),
            "robustness": {
                "margin": round(float(robustness["margin"]), 5),
                "probability_robust": round(float(robustness["probability_robust"]), 4),
                "status": robustness["status"],
                "margin_ci_low": round(float(robustness["margin_ci_low"]), 5),
                "margin_ci_high": round(float(robustness["margin_ci_high"]), 5),
                "top_global": top_global,
                "top_local": top_local,
                "changed": bool(robustness["changed"]),
                "local_background_n": robustness["local_background_n"],
                "background_draws": robustness["background_draws"],
                "curve_thresholds": robustness["curve_thresholds"],
                "curve_probability": robustness["curve_probability"],
                "interpretation": (
                    "Probability that this patient's signed top factor stays the same "
                    "when the local cohort mean is re-estimated at the observed cohort size."
                ),
            },
            "imputed_features": {
                column: round(float(value), 4)
                for column, value in self.impute_medians.items()
            },
            "explanation_units": "log-odds contribution (w_j * (x_j - background_j))",
        }

    # ------------------------------------------------------------------
    # Metadata for the UI
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "dataset_key": self.dataset_key,
            "dataset_name": self.dataset_name,
            "dataset_citation": self.dataset_citation,
            "data_source": self.data_source,
            "data_notes": self.data_notes,
            "partition_mode": self.partition_mode,
            "partition_requested": self.requested_partition_mode,
            "partition_fallback": self.partition_fallback,
            "partition_reason": self.partition_reason,
            "aggregation_weighting": self.weighting,
            "n_train": int(len(self.y_train)),
            "n_test": int(len(self.y_test)),
            "n_features": self.n_features,
            "test_prevalence": round(float(np.mean(self.y_test)), 4),
            "feature_names": self.feature_names,
            "imputation": {
                "fit": "training split only",
                "medians": {k: round(float(v), 4) for k, v in self.impute_medians.items()},
            },
            "scaler": self.scaler.to_dict(),
            "clients": [
                {
                    "id": client_id + 1,
                    "n_samples": meta["n_samples"],
                    "n_train": meta["n_train"],
                    "n_calibration": meta["n_calibration"],
                    "age_min": meta["age_min"],
                    "age_max": meta["age_max"],
                    "age_mean": meta["age_mean"],
                    "prevalence": meta["prevalence"],
                    "positives": meta["positives"],
                    "cohort": meta["dominant_band"],
                    "aggregation_weight": meta["aggregation_weight"],
                    "background_n": meta.get("background_n"),
                    "background_se": round(
                        float(
                            np.sqrt(
                                np.mean(np.diag(meta["background_cov"]))
                                / max(meta.get("background_n", 1), 1)
                            )
                        ),
                        4,
                    ),
                }
                for client_id, meta in enumerate(self.client_meta)
            ],
            "age_bands": sorted(set(self.test_bands.tolist())),
            "hyperparameters": {
                "local_epochs": self.local_epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "l2": self.l2,
                "warmup_rounds": DEFAULT_WARMUP_ROUNDS,
                "aggregation_weighting": self.weighting,
            },
        }
