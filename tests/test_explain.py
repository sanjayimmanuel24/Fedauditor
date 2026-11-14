"""Unit tests for the baseline-robustness diagnostic.

Run with::

    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from federated_explain import (
    instability_report,
    leader_margin,
    linear_phi,
    patient_robustness,
    robustness_margin,
    signed_top1,
)


class SignedTopFactorTests(unittest.TestCase):
    def test_large_protective_factor_is_not_top_risk(self):
        phi = np.array([-5.0, 0.2, 0.1])
        self.assertEqual(signed_top1(phi), 1)
        # The old abs-based rule would have selected feature 0.
        self.assertEqual(int(np.argmax(np.abs(phi))), 0)

    def test_leader_margin_is_lead_over_best_competitor(self):
        values = np.array([0.4, 0.1, -0.2])
        self.assertAlmostEqual(leader_margin(values, 0), 0.3)
        self.assertAlmostEqual(leader_margin(values), 0.3)


class MarginDefinitionTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.coef = rng.normal(size=5) * 0.4
        self.x = rng.normal(size=5)
        self.global_bg = rng.normal(size=5)
        self.local_bg = rng.normal(size=5)

    def test_margin_equals_local_lead_of_global_top(self):
        result = robustness_margin(
            self.coef, self.x, self.global_bg, self.local_bg
        )
        phi_global = linear_phi(self.coef, self.x, self.global_bg)
        phi_local = linear_phi(self.coef, self.x, self.local_bg)
        top = int(np.argmax(phi_global))
        expected = phi_local[top] - np.max(np.delete(phi_local, top))
        self.assertAlmostEqual(result["margin"], expected, places=12)
        self.assertEqual(result["changed"], int(top != int(np.argmax(phi_local))))

    def test_sign_of_margin_is_flip_indicator(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            coef = rng.normal(size=6)
            x = rng.normal(size=6)
            bg = rng.normal(size=6)
            local = rng.normal(size=6)
            result = robustness_margin(coef, x, bg, local)
            self.assertEqual(result["changed"], int(result["margin"] < 0))


class BackgroundUncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.coef = np.array([0.4, -0.2, 0.1, 0.05])
        self.x = np.array([0.3, -0.4, 0.2, 0.1])
        self.global_bg = np.zeros(4)

    def test_probability_is_bounded_and_deterministic(self):
        local = np.array([0.02, -0.01, 0.0, 0.01])
        covariance = np.eye(4) * 0.25
        first = patient_robustness(
            self.coef, self.x, self.global_bg, local,
            local_covariance=covariance, local_n=80, n_boot=200, seed=11,
        )
        second = patient_robustness(
            self.coef, self.x, self.global_bg, local,
            local_covariance=covariance, local_n=80, n_boot=200, seed=11,
        )
        self.assertGreaterEqual(first["probability_robust"], 0.0)
        self.assertLessEqual(first["probability_robust"], 1.0)
        self.assertAlmostEqual(
            first["probability_robust"], second["probability_robust"], places=12
        )
        self.assertIn(first["status"], {"stable", "fragile", "inconclusive"})

    def test_more_background_uncertainty_cannot_increase_robustness(self):
        local = np.array([0.02, -0.01, 0.0, 0.01])
        tight = patient_robustness(
            self.coef, self.x, self.global_bg, local,
            local_covariance=np.eye(4) * 0.01, local_n=80, n_boot=300, seed=5,
        )
        loose = patient_robustness(
            self.coef, self.x, self.global_bg, local,
            local_covariance=np.eye(4) * 1.0, local_n=80, n_boot=300, seed=5,
        )
        self.assertGreaterEqual(
            tight["probability_robust"], loose["probability_robust"]
        )


class ReportConsistencyTests(unittest.TestCase):
    def test_report_fields_agree_with_patient_margins(self):
        rng = np.random.default_rng(21)
        n_features = 4
        coef = rng.normal(size=n_features) * 0.5
        X = rng.normal(size=(30, n_features))
        groups = np.array(["<30"] * 10 + ["30-45"] * 10 + ["45+"] * 10)
        global_bg = np.zeros(n_features)
        backgrounds = {
            "<30": np.array([0.1, 0.0, -0.1, 0.0]),
            "30-45": np.array([0.0, 0.1, 0.0, -0.1]),
            "45+": np.array([-0.1, 0.0, 0.1, 0.0]),
        }
        covariances = {band: np.eye(n_features) * 0.2 for band in backgrounds}
        counts = {band: 50 for band in backgrounds}

        report = instability_report(
            coef, X, groups, backgrounds, global_bg,
            [f"f{i}" for i in range(n_features)],
            background_covariances=covariances,
            background_counts=counts,
            n_boot=40,
            seed=2,
        )
        margins = np.asarray(report["patient_margins"])
        self.assertEqual(len(margins), len(X))
        self.assertAlmostEqual(
            report["top1_flip_rate"], float(np.mean(margins < 0)), places=12
        )
        self.assertAlmostEqual(
            report["frac_negative_margin"], float(np.mean(margins < 0)), places=12
        )
        self.assertAlmostEqual(report["mean_margin"], float(np.mean(margins)), places=4)
        self.assertEqual(report["n_patients"], len(X))
        self.assertAlmostEqual(
            sum(row["n"] for row in report["cohort_summary"].values()), len(X)
        )


if __name__ == "__main__":
    unittest.main()
