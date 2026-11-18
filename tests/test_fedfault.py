"""Phase 2: FedFault planting and the federated XGBoost speed-up."""
from __future__ import annotations

import json
import unittest
import warnings

import numpy as np
import xgboost as xgb
from flwr.common import Code, FitRes, Status
from flwr.server.strategy import FedXgbBagging

from fedauditor.data import load_federated
from fedauditor.faults import (CLEAN, FAULTS, SEVERITY_LEVELS, Scenario, benchmark_grid,
                               in_subgroup, plant)
from fedauditor.federation import BatchedXgbBagging, run_federation
from fedauditor.models import make_model, tree_leaves
from test_fedauditor_phase1 import synthetic_brfss


def _data():
    frame = synthetic_brfss(n=6000)
    frame["State"] = np.random.default_rng(1).choice(["AL", "CA", "NY", "TX", "WA", "OR"], len(frame))
    return load_federated("brfss", frame=frame, min_rows=100)


class TestXgbSpeedup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.data = _data()

    def test_batched_bagging_matches_flower(self):
        flower, batched = FedXgbBagging(), BatchedXgbBagging()
        g = make_model("xgb", self.data.n_features)
        for rnd in range(1, 4):
            results = []
            for c in self.data.clients:
                local = make_model("xgb", self.data.n_features)
                local.start_round(g, c.X["train"])
                local.fit(c.X["train"], c.y["train"], n_clients=len(self.data.clients))
                results.append((None, FitRes(Status(Code.OK, ""), local.update_parameters(),
                                             c.n_train, {})))
            pf, _ = flower.aggregate_fit(rnd, results, [])
            pb, _ = batched.aggregate_fit(rnd, results, [])
            self.assertEqual(json.loads(pf.tensors[0]), json.loads(pb.tensors[0]))
            g.from_parameters(pb)

    def test_margin_start_matches_continued_training(self):
        c = self.data.clients[0]
        g = make_model("xgb", self.data.n_features)
        for _ in range(3):
            g.fit(c.X["train"], c.y["train"])
        fast = make_model("xgb", self.data.n_features)
        fast.start_round(g, c.X["train"])
        fast.fit(c.X["train"], c.y["train"])
        slow = make_model("xgb", self.data.n_features)
        slow.booster = g.booster.copy()
        slow.fit(c.X["train"], c.y["train"])
        np.testing.assert_allclose(tree_leaves(fast._update), tree_leaves(slow._update), rtol=1e-5)

    def test_cached_margin_equals_full_prediction(self):
        result = run_federation(self.data, model="xgb", rounds=3)
        X = self.data.clients[1].X["test"]
        full = result.model.booster.predict(xgb.DMatrix(X))
        np.testing.assert_allclose(result.model.predict_proba(X), full, rtol=1e-5)


class TestFaults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        warnings.filterwarnings("ignore")
        cls.data = _data()

    def plant(self, fault, severity="high", model="lr", rounds=4):
        return plant(Scenario(fault, severity, seed=20, model=model, rounds=rounds), self.data)

    def test_grid_size(self):
        grid = benchmark_grid("brfss", "lr", [20, 21, 22, 23, 24])
        self.assertEqual(len(grid), len(FAULTS) * 3 * 5 + 5)
        self.assertEqual(len({s.id for s in grid}), len(grid))

    def test_clean_is_untouched(self):
        p = plant(Scenario(CLEAN, seed=20), self.data)
        self.assertIs(p.data, self.data)
        self.assertEqual(p.truth["clients"], [])

    def test_targets_fixed_across_severities_and_clean_data_untouched(self):
        for fault in FAULTS:
            with self.subTest(fault=fault):
                planted = [self.plant(fault, sev) for sev in SEVERITY_LEVELS]
                targets = planted[0].truth["clients"]
                self.assertEqual(len(targets), 1)
                for p in planted:
                    self.assertEqual(p.truth["clients"], targets)
                    self.assertIs(p.clean_data, self.data)
                json.dumps(planted[0].truth)  # ground truth must be serialisable

    def test_data_faults_change_only_targets(self):
        for fault in ("label_noise", "prevalence_shift", "subgroup_bias", "missing_artefact"):
            with self.subTest(fault=fault):
                p = self.plant(fault)
                for c in p.data.clients:
                    same = (c.y["train"].shape == self.data.client(c.name).y["train"].shape
                            and np.array_equal(c.X["train"], self.data.client(c.name).X["train"])
                            and np.array_equal(c.y["train"], self.data.client(c.name).y["train"]))
                    self.assertEqual(same, c.name not in p.truth["clients"], c.name)

    def test_data_fault_effects(self):
        name = self.plant("label_noise").truth["clients"][0]
        clean = self.data.client(name)

        p = self.plant("label_noise")
        flipped = np.mean(p.data.client(name).y["train"] != clean.y["train"])
        self.assertAlmostEqual(flipped, 0.3, delta=0.05)

        p = self.plant("prevalence_shift")
        name = p.truth["clients"][0]
        clean = self.data.client(name)
        self.assertGreater(p.data.client(name).prevalence, clean.prevalence + 0.1)

        p = self.plant("subgroup_bias")
        name = p.truth["clients"][0]
        clean = self.data.client(name)
        sub = p.truth["details"][name]["subgroup"]
        c = p.data.client(name)
        mask = in_subgroup(c.groups["train"][sub["attribute"]], sub)
        self.assertLess(c.y["train"][mask].sum(), 0.3 * clean.y["train"][mask].sum())
        self.assertEqual(c.y["train"][~mask].sum(), clean.y["train"][~mask].sum())

        p = self.plant("missing_artefact")
        name = p.truth["clients"][0]
        clean = self.data.client(name)
        f = p.truth["details"][name]["features"][0]
        hit = np.isclose(p.data.client(name).X["train"][:, f["index"]], f["sentinel_scaled"])
        self.assertAlmostEqual(hit.mean(), 0.5, delta=0.06)

    def test_drift_starts_mid_training(self):
        p = self.plant("drift", rounds=6)
        self.assertEqual(p.truth["onset_round"], 4)
        self.assertIs(p.data_schedule(3), self.data)
        self.assertIs(p.data_schedule(4), p.data)
        name = p.truth["clients"][0]
        diff = p.data.client(name).X["train"] - self.data.client(name).X["train"]
        self.assertEqual(int((np.abs(diff).max(axis=0) > 0).sum()), 3)

    def test_update_faults_show_in_update_norms(self):
        for model in ("lr", "xgb"):
            clean = run_federation(self.data, model=model, rounds=4, seed=20)
            for fault in ("poisoning", "free_rider", "explanation_instability"):
                with self.subTest(model=model, fault=fault):
                    p = self.plant(fault, model=model)
                    result = run_federation(p.data, **p.run_kwargs())
                    name = p.truth["clients"][0]
                    rnd = p.truth["details"].get("rounds", [4])[-1]
                    got = result.history[rnd - 1]["update_norm"][name]
                    ref = clean.history[rnd - 1]["update_norm"][name]
                    if fault == "free_rider":
                        self.assertLess(got, 0.01 * ref)
                    else:
                        self.assertGreater(got, 2 * ref)

    def test_every_fault_runs_end_to_end(self):
        for model in ("lr", "mlp", "xgb"):
            for fault in FAULTS:
                with self.subTest(model=model, fault=fault):
                    p = self.plant(fault, "medium", model=model, rounds=3)
                    result = run_federation(p.data, **p.run_kwargs())
                    self.assertTrue(np.isfinite(result.final["_pooled"]["auc"]))

    def test_planted_runs_are_reproducible(self):
        for fault in ("label_noise", "explanation_instability"):
            a = run_federation(**{"data": self.plant(fault).data, **self.plant(fault).run_kwargs()})
            b = run_federation(**{"data": self.plant(fault).data, **self.plant(fault).run_kwargs()})
            for wa, wb in zip(a.model.weights, b.model.weights):
                np.testing.assert_array_equal(wa, wb)


if __name__ == "__main__":
    unittest.main()
