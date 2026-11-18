"""Phase 1: BRFSS recoding, federated datasets and Flower federation."""
from __future__ import annotations

import unittest
import warnings

import numpy as np
import pandas as pd

from fedauditor import brfss
from fedauditor.data import load_federated
from fedauditor.federation import run_federation


def synthetic_brfss(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "State": rng.choice(["AL", "CA", "NY", "TX"], n),
        "BMI": rng.normal(28, 6, n),
        "HighBP": rng.integers(0, 2, n).astype(float),
        "AgeGroup": rng.integers(1, 14, n).astype(float),
        "Sex": rng.integers(0, 2, n).astype(float),
        "Income": rng.integers(1, 8, n).astype(float),
    })
    z = 0.12 * (frame.BMI - 28) + frame.HighBP + 0.2 * frame.AgeGroup - 3.5
    frame["Outcome"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return frame


class TestBRFSSRecode(unittest.TestCase):
    def test_codes_map_to_schema(self):
        raw = pd.DataFrame({
            "_STATE": [1, 6, 36, 72],
            "DIABETE4": [1, 2, 4, 9],          # yes, pregnancy-only, pre-diabetes, refused
            "_RFHYPE6": [2, 1, 9, 2],
            "TOLDHI3": [1, 2, np.nan, 7],      # blank = never checked -> 0
            "_BMI5": [2512, 3000, np.nan, 9999],
            "SMOKE100": [1, 2, 7, 1], "CVDSTRK3": [2, 2, 2, 1], "_MICHD": [2, 1, 2, 2],
            "_TOTINDA": [1, 2, 9, 1], "_RFDRHV8": [1, 2, 1, 1], "_HLTHPL1": [1, 2, 1, 9],
            "GENHLTH": [3, 5, 7, 1], "MENTHLTH": [88, 5, 77, 30], "PHYSHLTH": [2, 88, 99, 88],
            "DIFFWALK": [2, 1, 2, 2], "SEXVAR": [1, 2, 2, 1], "_AGEG5YR": [5, 13, 14, 1],
            "EDUCA": [4, 6, 9, 2], "_INCOMG1": [3, 9, 7, 1],
        })
        out, used = brfss.recode_frame(raw)
        self.assertEqual(out["State"].tolist(), ["AL", "CA", "NY", "PR"])
        self.assertEqual(out["Outcome"].tolist()[:3], [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(out["Outcome"].iloc[3]))
        self.assertEqual(out["HighBP"].tolist()[:2], [1.0, 0.0])
        self.assertEqual(out["HighChol"].tolist()[:3], [1.0, 0.0, 0.0])
        self.assertTrue(np.isnan(out["HighChol"].iloc[3]))
        self.assertAlmostEqual(out["BMI"].iloc[0], 25.12)
        self.assertEqual(out["MentHlth"].tolist()[0], 0.0)
        self.assertTrue(np.isnan(out["MentHlth"].iloc[2]))
        self.assertTrue(np.isnan(out["AgeGroup"].iloc[2]))
        self.assertEqual(used["HvyAlcohol"], "_RFDRHV8")
        self.assertNotIn("Fruits", out.columns)  # optional module absent

    def test_missing_required_column_raises(self):
        with self.assertRaises(KeyError):
            brfss.recode_frame(pd.DataFrame({"_STATE": [1], "DIABETE4": [1]}))


class TestFederatedDataset(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_brfss()
        self.data = load_federated("brfss", frame=self.frame, min_rows=100, seed=3)

    def test_one_client_per_state_and_disjoint_splits(self):
        self.assertEqual(self.data.client_names, ["AL", "CA", "NY", "TX"])
        total = sum(len(c.y[s]) for c in self.data.clients for s in ("train", "cal", "test"))
        self.assertEqual(total, len(self.frame))
        for c in self.data.clients:
            self.assertEqual(set(c.groups["test"]), {"Sex", "AgeGroup", "Income"})

    def test_federated_scaling_standardises_pooled_train(self):
        X, _ = self.data.pooled("train")
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(X.std(axis=0), 1, atol=1e-6)

    def test_small_states_are_skipped_and_noted(self):
        counts = self.frame["State"].value_counts()
        data = load_federated("brfss", frame=self.frame, min_rows=int(counts.min()) + 1)
        self.assertNotIn(counts.idxmin(), data.client_names)
        self.assertGreaterEqual(len(data.clients), 2)
        self.assertTrue(any("skipped" in n for n in data.notes))

    def test_pima_age_cohorts(self):
        data = load_federated("pima")
        self.assertEqual(len(data.clients), 3)


class TestFederation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.data = load_federated("brfss", frame=synthetic_brfss(), min_rows=100)

    def test_all_models_learn(self):
        for model in ("lr", "mlp", "xgb"):
            with self.subTest(model=model):
                result = run_federation(self.data, model=model, rounds=8)
                self.assertGreater(result.final["_pooled"]["auc"], 0.7)
                self.assertEqual(len(result.history), 8)

    def test_seeded_runs_are_identical(self):
        a = run_federation(self.data, model="mlp", rounds=3, seed=5)
        b = run_federation(self.data, model="mlp", rounds=3, seed=5)
        for wa, wb in zip(a.model.weights, b.model.weights):
            np.testing.assert_array_equal(wa, wb)

    def test_xgb_bagging_adds_one_tree_per_client_per_round(self):
        result = run_federation(self.data, model="xgb", rounds=3)
        self.assertEqual(result.model.n_trees, 3 * len(self.data.clients))

    def test_robust_strategies_and_client_sampling(self):
        for strategy in ("fedmedian", "fedtrimmedavg"):
            result = run_federation(self.data, model="lr", rounds=4, strategy=strategy,
                                    fraction_fit=0.5)
            self.assertEqual(len(result.history[0]["clients"]), 2)
            self.assertGreater(result.final["_pooled"]["auc"], 0.7)

    def test_update_hook_can_free_ride(self):
        def free_rider(rnd, name, local, global_model):
            if name == "AL":
                local.weights = [w.copy() for w in global_model.weights]

        result = run_federation(self.data, model="lr", rounds=3, update_hook=free_rider)
        norms = result.history[-1]["update_norm"]
        self.assertEqual(norms["AL"], 0.0)
        self.assertGreater(norms["CA"], 0.0)


if __name__ == "__main__":
    unittest.main()
