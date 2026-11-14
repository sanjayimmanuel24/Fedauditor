"""Unit tests for the half-space geometry and phase sensitivity.

Run with::

    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest

import numpy as np

from federated_theory import (
    critical_alpha,
    pairwise_terms,
    phase_margin,
    phase_statistics,
)


class PairwiseDecompositionTests(unittest.TestCase):
    def setUp(self):
        self.coef = np.array([0.8, 0.3, -0.2, 0.1])
        self.x = np.array([0.5, -0.2, 1.1, 0.4])
        self.global_bg = np.zeros(4)
        self.local_bg = np.array([0.1, -0.3, 0.4, 0.2])

    def test_leader_is_signed_and_a_is_non_negative(self):
        terms = pairwise_terms(
            self.coef, self.x, self.global_bg, self.local_bg
        )
        phi_global = self.coef * (self.x - self.global_bg)
        self.assertEqual(terms["leader"], int(np.argmax(phi_global)))
        self.assertTrue(np.all(np.asarray(terms["a"]) >= -1e-12))

    def test_direct_margin_equals_pairwise_formula(self):
        terms = pairwise_terms(
            self.coef, self.x, self.global_bg, self.local_bg
        )
        phi_local = self.coef * (self.x - self.local_bg)
        expected = np.min(
            phi_local[terms["leader"]] - np.delete(phi_local, terms["leader"])
        )
        self.assertAlmostEqual(
            phase_margin(terms["a"], terms["c"], 1.0), expected, places=12
        )


class CriticalAlphaTests(unittest.TestCase):
    def test_matches_brute_force_scan(self):
        rng = np.random.default_rng(19)
        for _ in range(50):
            a = rng.uniform(0.0, 2.0, size=5)
            c = rng.normal(size=5)
            alpha = critical_alpha(a, c)
            grid = np.linspace(0.0, 5.0, 5001)
            margins = np.array([phase_margin(a, c, value) for value in grid])
            if np.isfinite(alpha) and alpha > grid[-1]:
                # The critical multiplier lies beyond the scanned window.
                self.assertTrue(np.all(margins >= -1e-10))
                continue
            first_flip = grid[np.argmax(margins < 0)] if np.any(margins < 0) else np.inf
            if np.isfinite(alpha):
                self.assertLessEqual(alpha, first_flip + 0.002)
                self.assertGreaterEqual(alpha, first_flip - 0.002)
            else:
                self.assertTrue(np.all(margins >= -1e-10))

    def test_competitor_that_never_overtakes_gives_infinite_alpha(self):
        self.assertEqual(critical_alpha([0.2, 0.3], [0.0, 1.0]), float("inf"))

    def test_margin_is_concave_and_stable_set_is_an_interval(self):
        rng = np.random.default_rng(4)
        alphas = np.linspace(0.0, 4.0, 801)
        for _ in range(40):
            a = rng.uniform(0.0, 1.5, size=6)
            c = rng.normal(size=6)
            margins = np.array([phase_margin(a, c, value) for value in alphas])
            # Concavity: second differences are non-positive.
            self.assertTrue(np.all(np.diff(margins, n=2) <= 1e-10))
            # Once the margin is negative it cannot become positive again.
            negative = np.flatnonzero(margins < 0)
            if negative.size:
                self.assertTrue(np.all(margins[negative[0] :] < 1e-10))


class PhaseStatisticsTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(31)
        self.n_features = 5
        self.coef = rng.normal(size=self.n_features) * 0.6
        self.X = rng.normal(size=(40, self.n_features))
        self.groups = np.array(["a"] * 20 + ["b"] * 20)
        self.global_bg = np.zeros(self.n_features)
        self.backgrounds = {
            "a": np.array([0.3, -0.2, 0.1, 0.0, 0.2]),
            "b": np.array([-0.2, 0.3, 0.0, 0.2, -0.1]),
        }

    def test_alpha_zero_is_stable_and_alpha_one_matches_direct_margins(self):
        report = phase_statistics(
            self.coef,
            self.X,
            self.groups,
            self.backgrounds,
            self.global_bg,
            alphas=(0.0, 0.5, 1.0, 2.0),
        )
        self.assertEqual(report["flip_rate"][0], 0.0)

        margins = []
        for row, group in zip(self.X, self.groups):
            terms = pairwise_terms(
                self.coef, row, self.global_bg, self.backgrounds[group]
            )
            margins.append(phase_margin(terms["a"], terms["c"], 1.0))
        expected = float(np.mean(np.asarray(margins) < 0))
        self.assertAlmostEqual(report["flip_rate"][2], expected, places=12)
        self.assertAlmostEqual(
            report["observed_flip_rate"], expected, places=12
        )

    def test_union_bound_dominates_empirical_flip_rate(self):
        report = phase_statistics(
            self.coef,
            self.X,
            self.groups,
            self.backgrounds,
            self.global_bg,
        )
        for flip, bound in zip(report["flip_rate"], report["union_bound"]):
            self.assertGreaterEqual(bound + 1e-12, flip)

    def test_survival_is_complement_of_phase_flip(self):
        report = phase_statistics(
            self.coef,
            self.X,
            self.groups,
            self.backgrounds,
            self.global_bg,
        )
        for flip, survival in zip(report["flip_rate"], report["survival"]):
            self.assertAlmostEqual(flip, 1.0 - survival, places=12)


if __name__ == "__main__":
    unittest.main()
