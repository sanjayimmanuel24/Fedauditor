"""Half-space geometry and phase sensitivity for the baseline-robustness margin.

This module turns the verbal argument in the paper into executable algebra.
For a linear model the cohort-relative Shapley value is

    phi_j(x; b) = w_j * (x_j - b_j).

Fix the global leader ``i = argmax_j phi_j(x; b_g)``. For any other feature
``j`` define

    a_j = phi_g,i - phi_g,j                  (>= 0 by construction)
    c_j = Delta_i - Delta_j,  Delta = w * (b_g - b_l)

so the local margin is

    R(x) = min_{j != i} (a_j + c_j).

The global top factor flips exactly when ``R(x) < 0``, i.e. when at least one
of the pairwise half-space constraints ``a_j + c_j < 0`` is violated. Two
consequences follow immediately:

1. **Concavity.** ``R`` is a minimum of affine functions of the baseline, so it
   is concave. With ``R(0) >= 0`` the stable set is an interval, and a single
   critical baseline multiplier ``alpha*`` separates stability from flipping.
2. **Union bound.** The flip event is a union of pairwise events, so
   ``P(flip) <= sum_j P(a_j + c_j < 0)``. The empirical version of that sum is
   a deterministic upper bound on the observed flip rate.

The phase sweep moves the local baseline along the segment from the global
baseline to the observed local baseline::

    b_l(alpha) = b_g + alpha * (b_l - b_g),   alpha in [0, 3].

``alpha = 0`` is the global baseline, ``alpha = 1`` is the observed local
baseline, and ``alpha > 1`` asks how much further the cohort baseline would
have to move before the explanation changes. The critical multiplier is

    alpha* = min_{j: c_j < 0} a_j / (-c_j),

with ``alpha* = +inf`` when no competitor can ever overtake the leader. Then
``flip(alpha) = 1{alpha* < alpha}`` and the observed flip rate is
``P(alpha* < 1)``.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np

DEFAULT_PHASE_GRID = tuple(
    float(value) for value in np.round(np.arange(0.0, 3.0001, 0.25), 2)
)


def linear_phi(
    coef: np.ndarray, x_scaled: np.ndarray, background: np.ndarray
) -> np.ndarray:
    coef = np.asarray(coef, dtype=float).ravel()
    x_scaled = np.asarray(x_scaled, dtype=float).ravel()
    background = np.asarray(background, dtype=float).ravel()
    return coef * (x_scaled - background)


def pairwise_terms(
    coef: np.ndarray,
    x_scaled: np.ndarray,
    global_background: np.ndarray,
    local_background: np.ndarray,
) -> Dict[str, object]:
    """Return the leader and the pairwise ``(a, c)`` decomposition.

    ``a`` and ``c`` are indexed over the competitors in feature order, with the
    leader removed. ``a >= 0`` always; a non-negative ``c_j`` means competitor
    ``j`` can never overtake the leader as the local baseline moves away from
    the global one.
    """
    coef = np.asarray(coef, dtype=float).ravel()
    phi_global = linear_phi(coef, x_scaled, global_background)
    phi_local = linear_phi(coef, x_scaled, local_background)
    delta = phi_local - phi_global

    leader = int(np.argmax(phi_global))
    competitors = np.array(
        [index for index in range(phi_global.size) if index != leader], dtype=int
    )
    a = phi_global[leader] - phi_global[competitors]
    c = delta[leader] - delta[competitors]
    return {
        "leader": leader,
        "competitors": competitors,
        "a": a,
        "c": c,
        "delta": delta,
        "phi_global": phi_global,
        "phi_local": phi_local,
    }


def phase_margin(a: Sequence[float], c: Sequence[float], alpha: float) -> float:
    """Margin ``R(alpha) = min_j (a_j + alpha c_j)``; ``+inf`` if no rivals."""
    a = np.asarray(a, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    if a.size == 0:
        return float("inf")
    return float(np.min(a + float(alpha) * c))


def critical_alpha(a: Sequence[float], c: Sequence[float]) -> float:
    """Smallest ``alpha >= 0`` at which any competitor overtakes the leader."""
    a = np.asarray(a, dtype=float).ravel()
    c = np.asarray(c, dtype=float).ravel()
    if a.size == 0:
        return float("inf")
    overtaking = c < 0
    if not np.any(overtaking):
        return float("inf")
    return float(np.min(a[overtaking] / (-c[overtaking])))


def phase_statistics(
    coef: np.ndarray,
    X_scaled: np.ndarray,
    groups: Iterable,
    client_backgrounds: Dict[str, np.ndarray],
    global_background: np.ndarray,
    alphas: Sequence[float] = DEFAULT_PHASE_GRID,
    observed_alpha: float = 1.0,
) -> Dict[str, object]:
    """Sweep the local baseline and summarise the flip phase diagram."""
    coef = np.asarray(coef, dtype=float).ravel()
    X_scaled = np.asarray(X_scaled, dtype=float)
    groups = np.asarray([str(group) for group in groups])
    fallback = np.asarray(global_background, dtype=float).ravel()
    alphas = np.asarray(alphas, dtype=float).ravel()

    a_rows = []
    c_rows = []
    alpha_stars = []
    for index, row in enumerate(X_scaled):
        background = client_backgrounds.get(groups[index], fallback)
        background = np.asarray(background, dtype=float).ravel()
        if background.shape != row.shape:
            background = fallback
        terms = pairwise_terms(coef, row, fallback, background)
        a = np.asarray(terms["a"], dtype=float)
        c = np.asarray(terms["c"], dtype=float)
        a_rows.append(a)
        c_rows.append(c)
        alpha_stars.append(critical_alpha(a, c))

    n_patients = int(len(X_scaled))
    n_features = int(coef.size)
    alpha_stars = np.asarray(alpha_stars, dtype=float)

    if n_patients and n_features > 1:
        A = np.vstack(a_rows)
        C = np.vstack(c_rows)
        # One row per patient, one column per alpha.
        margins = np.min(A[:, None, :] + alphas[None, :, None] * C[:, None, :], axis=2)
        violations = np.sum(
            A[:, None, :] + alphas[None, :, None] * C[:, None, :] < 0.0, axis=2
        )
        flip_rate = np.mean(margins < 0.0, axis=0)
        mean_margin = np.mean(margins, axis=0)
        union_bound = np.minimum(1.0, np.mean(violations, axis=0))
    else:
        flip_rate = np.zeros_like(alphas)
        mean_margin = np.full_like(alphas, np.nan)
        union_bound = np.zeros_like(alphas)

    finite = alpha_stars[np.isfinite(alpha_stars)]
    critical_summary: Dict[str, float | int | None] = {
        "n_patients": n_patients,
        "n_finite": int(finite.size),
        "fraction_finite": (
            round(float(finite.size / n_patients), 6) if n_patients else None
        ),
        "median": round(float(np.median(finite)), 6) if finite.size else None,
        "q10": round(float(np.quantile(finite, 0.10)), 6) if finite.size else None,
        "q25": round(float(np.quantile(finite, 0.25)), 6) if finite.size else None,
        "q75": round(float(np.quantile(finite, 0.75)), 6) if finite.size else None,
        "q90": round(float(np.quantile(finite, 0.90)), 6) if finite.size else None,
        "fraction_flip_by_observed": (
            round(float(np.mean(alpha_stars < observed_alpha)), 6)
            if n_patients
            else None
        ),
    }

    observed_index = int(np.argmin(np.abs(alphas - observed_alpha)))
    observed_flip = float(flip_rate[observed_index])
    observed_union = float(union_bound[observed_index])
    observed_margin = float(mean_margin[observed_index])

    return {
        "alpha_grid": [float(value) for value in alphas],
        "flip_rate": [round(float(value), 6) for value in flip_rate],
        "union_bound": [round(float(value), 6) for value in union_bound],
        "mean_margin": [
            None if not np.isfinite(value) else round(float(value), 6)
            for value in mean_margin
        ],
        "survival": [
            round(float(np.mean(alpha_stars >= alpha)), 6) for alpha in alphas
        ],
        "observed_alpha": float(observed_alpha),
        "observed_flip_rate": round(observed_flip, 6),
        "observed_union_bound": round(observed_union, 6),
        "observed_mean_margin": round(observed_margin, 6),
        "critical_alpha": critical_summary,
        # ``None`` marks a patient whose leader can never be overtaken by
        # moving the local baseline further away from the global one.
        "alpha_stars": [
            round(float(value), 6) if np.isfinite(value) else None
            for value in alpha_stars
        ],
        "n_patients": n_patients,
        "n_features": n_features,
    }
