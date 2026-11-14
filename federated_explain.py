"""Background-relative explanations and a baseline-robustness diagnostic.

For a linear model the Shapley value is available in closed form under the
interventional (feature-independence) assumption:

    phi_j(x) = w_j * (x_j - E_bg[x_j])

The original dashboard was already computing ``w_j * x_j``, which is the same
quantity with a zero background. The contribution here is not "we added SHAP";
it is that the *background* is a choice, and in a federated system the local
cohort and the global cohort are different. This module turns that observation
into a continuous, auditable measurement rather than a single flip count.

Definitions
-----------
Let ``phi_g`` be the explanation against the global cohort and ``phi_l`` the
explanation against the patient's local cohort. The patient-independent
baseline shift is::

    Delta = phi_l - phi_g = w * (b_g - b_l)

Write ``i = argmax_j phi_g,j`` for the signed top factor under the global
baseline (the largest *increase* in log-odds; it may be negative when every
factor is protective). The baseline-robustness margin is::

    R(x) = min_{j != i} (phi_l,i - phi_l,j)
         = min_{j != i} ((phi_g,i - phi_g,j) + (Delta_i - Delta_j))

``R(x) > 0`` means the global top factor is still the top factor under the
local baseline. ``R(x) < 0`` means the top factor changed. The conventional
top-1 flip rate is therefore ``P(R < 0)``: the margin is the continuous
quantity, the flip rate is only its sign. A positive but tiny margin is not
evidence of stability, so the diagnostic also propagates uncertainty in the
local cohort mean, estimated from aggregate ``(n, mean, covariance)``
sufficient statistics, and reports ``P(R > 0)`` per patient.

The diagnostic is an *explanation-stability* diagnostic. It says nothing about
whether a prediction is clinically correct.

Geometry
--------
Fix the global leader ``i`` and write ``a_j = phi_g,i - phi_g,j >= 0`` and
``c_j = Delta_i - Delta_j``. Then ``R = min_j (a_j + c_j)``. Since ``R`` is a
minimum of affine functions of the baseline, it is concave, the stable set is an
interval, and the flip set is a union of pairwise half-spaces. That geometry
gives the union bound and the critical baseline multiplier ``alpha*`` reported
in ``phase_diagram``; see ``federated_theory.py`` for the derivation.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from federated_theory import phase_statistics

# Thresholds at which the robustness curve is evaluated. 0 is the top-1 flip
# boundary; larger values ask how far the leader must be ahead to count as
# meaningfully stable.
DEFAULT_ROBUSTNESS_GRID: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)

STABLE_THRESHOLD = 0.95
FRAGILE_THRESHOLD = 0.50


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = np.sqrt(np.sum(ra**2) * np.sum(rb**2))
    if denominator == 0:
        return 0.0
    return float(np.sum(ra * rb) / denominator)


def linear_phi(coef: np.ndarray, x_scaled: np.ndarray, background: np.ndarray) -> np.ndarray:
    coef = np.asarray(coef, dtype=float).ravel()
    x_scaled = np.asarray(x_scaled, dtype=float).ravel()
    background = np.asarray(background, dtype=float).ravel()
    return coef * (x_scaled - background)


def contributions(
    coef: np.ndarray,
    x_scaled: np.ndarray,
    background: np.ndarray,
    features: Iterable[str],
    top_k: int | None = None,
) -> List[dict]:
    """Signed linear Shapley values, ordered for display by magnitude.

    Display order is by ``|phi|`` so the largest factors are easy to see, but
    the ``direction`` field is signed: positive means "raises log-odds relative
    to this background", negative means "lowers it". Any use of a *top* factor
    must use ``signed_top1`` rather than ``argmax(abs(phi))``.
    """
    phi = linear_phi(coef, x_scaled, background)
    rows = [
        {
            "feature": str(feature),
            "contribution": float(value),
            "direction": "risk" if value > 0 else "protective",
        }
        for feature, value in zip(features, phi)
    ]
    rows.sort(key=lambda row: abs(row["contribution"]), reverse=True)
    if top_k is not None:
        rows = rows[:top_k]
    return rows


def signed_top1(phi: np.ndarray) -> int:
    """Index of the largest signed contribution.

    This is intentionally *not* ``argmax(abs(phi))``. A large negative value is
    a strong protective factor, not the top risk factor. Using absolute values
    conflates "strongest influence" with "largest increase in risk", which is
    the bug the margin diagnostic exists to avoid.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    if phi.size == 0:
        raise ValueError("cannot select a top factor from an empty explanation")
    return int(np.argmax(phi))


def leader_margin(values: np.ndarray, leader: int | None = None) -> float:
    """Signed lead of ``leader`` over the best other entry."""
    values = np.asarray(values, dtype=float).ravel()
    if values.size < 2:
        return 0.0
    if leader is None:
        leader = signed_top1(values)
    others = np.delete(values, int(leader))
    return float(values[int(leader)] - np.max(others))


def robustness_margin(
    coef: np.ndarray,
    x_scaled: np.ndarray,
    global_background: np.ndarray,
    local_background: np.ndarray,
) -> Dict[str, float | int]:
    """Closed-form baseline-robustness margin for one patient.

    Returns the global/local top indices, the margin ``R``, and the global
    leader margin (how far the global leader is ahead *before* changing
    baseline). ``R`` is the local margin of the global leader; its sign is
    exactly the top-1 flip indicator.
    """
    phi_global = linear_phi(coef, x_scaled, global_background)
    phi_local = linear_phi(coef, x_scaled, local_background)
    top_global = signed_top1(phi_global)
    top_local = signed_top1(phi_local)
    return {
        "top_global": top_global,
        "top_local": top_local,
        "margin": leader_margin(phi_local, top_global),
        "global_margin": leader_margin(phi_global, top_global),
        "changed": int(top_global != top_local),
    }


def mean_abs_attribution(
    coef: np.ndarray, X_scaled: np.ndarray, background: np.ndarray, features: Iterable[str]
) -> Dict[str, float]:
    phi = np.array(
        [linear_phi(coef, row, background) for row in np.asarray(X_scaled, dtype=float)]
    )
    return {
        str(feature): round(float(value), 5)
        for feature, value in zip(features, np.mean(np.abs(phi), axis=0))
    }


def _regularised_covariance(covariance: np.ndarray | None, n_features: int) -> np.ndarray:
    """Return a positive-semidefinite covariance, regularised if necessary."""
    if covariance is None:
        return np.zeros((n_features, n_features), dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.shape != (n_features, n_features):
        return np.zeros((n_features, n_features), dtype=float)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.clip(eigenvalues, 1e-9, None)
    return (eigenvectors * eigenvalues) @ eigenvectors.T


def _bootstrap_backgrounds(
    mean: np.ndarray,
    covariance: np.ndarray | None,
    n: int | None,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Draw plausible local-cohort means from aggregate sufficient statistics.

    The sampling covariance of a cohort mean is ``Sigma / n``. ``Sigma`` and
    ``n`` are aggregate statistics; no patient row is needed. The server-side
    simulation can therefore quantify baseline uncertainty without moving data.
    """
    if covariance is None or n is None or int(n) <= 0 or n_boot <= 0:
        return None
    mean = np.asarray(mean, dtype=float).ravel()
    sigma = _regularised_covariance(covariance, mean.size)
    try:
        return rng.multivariate_normal(mean, sigma / float(n), size=int(n_boot))
    except (ValueError, np.linalg.LinAlgError):
        # A fallback diagonal draw keeps the diagnostic available even for a
        # degenerate cohort covariance.
        return mean + rng.normal(
            0.0, np.sqrt(np.clip(np.diag(sigma) / float(n), 1e-12, None)),
            size=(int(n_boot), mean.size),
        )


def patient_robustness(
    coef: np.ndarray,
    x_scaled: np.ndarray,
    global_background: np.ndarray,
    local_background: np.ndarray,
    local_covariance: np.ndarray | None = None,
    local_n: int | None = None,
    n_boot: int = 200,
    seed: int = 0,
    grid: Sequence[float] = DEFAULT_ROBUSTNESS_GRID,
    rng: np.random.Generator | None = None,
) -> Dict[str, object]:
    """Robustness diagnostic for one patient, including background uncertainty."""
    coef = np.asarray(coef, dtype=float).ravel()
    x_scaled = np.asarray(x_scaled, dtype=float).ravel()
    global_background = np.asarray(global_background, dtype=float).ravel()
    local_background = np.asarray(local_background, dtype=float).ravel()

    point = robustness_margin(coef, x_scaled, global_background, local_background)
    top_global = int(point["top_global"])
    margin = float(point["margin"])

    rng = rng or np.random.default_rng(seed)
    draws = _bootstrap_backgrounds(
        local_background, local_covariance, local_n, n_boot, rng
    )

    if draws is None:
        probabilities = np.array([1.0 if margin > 0 else 0.0])
        margins = np.array([margin])
        curve = np.array([1.0 if margin > threshold else 0.0 for threshold in grid])
    else:
        # Vectorised: explanation under every plausible local background.
        phi_draws = coef[None, :] * (x_scaled[None, :] - draws)
        leader = phi_draws[:, top_global]
        competitor = phi_draws.copy()
        competitor[:, top_global] = -np.inf
        margins = np.min(leader[:, None] - competitor, axis=1)
        probabilities = (margins > 0).astype(float)
        curve = np.array([np.mean(margins > threshold) for threshold in grid], dtype=float)

    probability_robust = float(np.mean(probabilities))
    ci_low = float(np.quantile(margins, 0.025))
    ci_high = float(np.quantile(margins, 0.975))
    if probability_robust >= STABLE_THRESHOLD:
        status = "stable"
    elif probability_robust <= FRAGILE_THRESHOLD:
        status = "fragile"
    else:
        status = "inconclusive"

    return {
        **point,
        "margin": margin,
        "probability_robust": probability_robust,
        "margin_ci_low": ci_low,
        "margin_ci_high": ci_high,
        "background_draws": int(len(margins)),
        "local_background_n": int(local_n) if local_n is not None else None,
        "status": status,
        "curve_thresholds": [float(t) for t in grid],
        "curve_probability": [float(v) for v in curve],
    }


def _bootstrap_ci(
    values: np.ndarray,
    statistic,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0 or n_boot <= 0:
        return float("nan"), float("nan")
    statistics = [
        float(statistic(values[rng.integers(0, values.size, values.size)]))
        for _ in range(int(n_boot))
    ]
    if not statistics:
        return float("nan"), float("nan")
    return (
        float(np.quantile(statistics, alpha / 2)),
        float(np.quantile(statistics, 1 - alpha / 2)),
    )


def split_half_reliability(
    margins: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Test-retest behaviour of the aggregate statistics at this sample size.

    Two random halves of the same test set are scored independently. The
    distribution of their disagreement is the sampling noise a reader should
    expect from the reported flip rate and mean margin. It is a reliability
    check on the *measurement*, not on individual patients.
    """
    margins = np.asarray(margins, dtype=float).ravel()
    if margins.size < 4 or n_boot <= 0:
        return {
            "flip_rate_half_abs_diff": float("nan"),
            "flip_rate_half_abs_diff_q95": float("nan"),
            "margin_half_abs_diff": float("nan"),
            "margin_half_abs_diff_q95": float("nan"),
        }

    rng = np.random.default_rng(seed)
    flip_differences = []
    margin_differences = []
    half = margins.size // 2
    for _ in range(int(n_boot)):
        order = rng.permutation(margins.size)
        first = margins[order[:half]]
        second = margins[order[half : 2 * half]]
        flip_differences.append(
            abs(float(np.mean(first < 0)) - float(np.mean(second < 0)))
        )
        margin_differences.append(abs(float(np.mean(first)) - float(np.mean(second))))

    return {
        "flip_rate_half_abs_diff": round(float(np.mean(flip_differences)), 5),
        "flip_rate_half_abs_diff_q95": round(float(np.quantile(flip_differences, 0.95)), 5),
        "margin_half_abs_diff": round(float(np.mean(margin_differences)), 5),
        "margin_half_abs_diff_q95": round(float(np.quantile(margin_differences, 0.95)), 5),
    }


def instability_report(
    coef: np.ndarray,
    X_scaled: np.ndarray,
    groups: Iterable,
    client_backgrounds: Dict[str, np.ndarray],
    global_background: np.ndarray,
    features: Iterable[str],
    max_examples: int = 6,
    background_covariances: Dict[str, np.ndarray] | None = None,
    background_counts: Dict[str, int] | None = None,
    n_boot: int = 200,
    seed: int = 0,
    grid: Sequence[float] = DEFAULT_ROBUSTNESS_GRID,
) -> Dict[str, object]:
    """Compare global vs local explanations and measure baseline robustness.

    ``top1_flip_rate`` is retained for backward compatibility but is now the
    *signed* top-1 flip rate. It equals ``P(R < 0)`` and should be read as the
    sign of the margin distribution, not as the headline number.
    """
    features = [str(f) for f in features]
    X_scaled = np.asarray(X_scaled, dtype=float)
    groups = np.asarray([str(g) for g in groups])
    fallback = np.asarray(global_background, dtype=float)
    rng = np.random.default_rng(seed)

    margins: List[float] = []
    global_margins: List[float] = []
    correlations: List[float] = []
    shifts: List[float] = []
    probabilities: List[float] = []
    margins_ci: List[Tuple[float, float]] = []
    per_patient_top_global: List[str] = []
    per_patient_top_local: List[str] = []
    per_patient_status: List[str] = []
    per_patient_curve: List[List[float]] = []
    flips = 0
    examples: List[dict] = []

    for index, row in enumerate(X_scaled):
        background = client_backgrounds.get(groups[index], fallback)
        background = np.asarray(background, dtype=float)
        if background.shape != row.shape:
            background = fallback

        covariance = None
        count = None
        if background_covariances is not None:
            covariance = background_covariances.get(groups[index])
        if background_counts is not None:
            count = background_counts.get(groups[index])

        patient = patient_robustness(
            coef,
            row,
            fallback,
            background,
            local_covariance=covariance,
            local_n=count,
            n_boot=n_boot,
            seed=seed + index,
            grid=grid,
            rng=rng,
        )

        phi_global = linear_phi(coef, row, fallback)
        phi_local = linear_phi(coef, row, background)
        top_global = int(patient["top_global"])
        top_local = int(patient["top_local"])
        flipped = bool(patient["changed"])
        flips += int(flipped)

        correlation = _spearman(phi_global, phi_local)
        correlations.append(correlation)
        shifts.append(float(np.mean(np.abs(phi_global - phi_local))))
        margins.append(float(patient["margin"]))
        global_margins.append(float(patient["global_margin"]))
        probabilities.append(float(patient["probability_robust"]))
        margins_ci.append((float(patient["margin_ci_low"]), float(patient["margin_ci_high"])))
        per_patient_top_global.append(features[top_global])
        per_patient_top_local.append(features[top_local])
        per_patient_status.append(str(patient["status"]))
        per_patient_curve.append([float(v) for v in patient["curve_probability"]])

        if len(examples) < max_examples and (flipped or patient["status"] == "fragile"):
            examples.append(
                {
                    "age_band": groups[index],
                    "top_global": features[top_global],
                    "top_local": features[top_local],
                    "spearman": round(correlation, 4),
                    "margin": round(float(patient["margin"]), 4),
                    "probability_robust": round(float(patient["probability_robust"]), 4),
                    "status": patient["status"],
                    "global_margin": round(float(patient["global_margin"]), 4),
                    "global": {
                        f: round(float(v), 4)
                        for f, v in zip(features, phi_global)
                    },
                    "local": {
                        f: round(float(v), 4)
                        for f, v in zip(features, phi_local)
                    },
                }
            )

    n = max(len(correlations), 1)
    margins_array = np.asarray(margins, dtype=float)
    probabilities_array = np.asarray(probabilities, dtype=float)
    curve_array = (
        np.asarray(per_patient_curve, dtype=float)
        if per_patient_curve
        else np.zeros((0, len(grid)))
    )
    uncertainty_curve = (
        curve_array.mean(axis=0).tolist() if curve_array.size else []
    )
    point_curve = [
        float(np.mean(margins_array > threshold)) if margins_array.size else 0.0
        for threshold in grid
    ]

    flip_low, flip_high = _bootstrap_ci(
        (margins_array < 0).astype(float), np.mean, n_boot, rng
    )
    margin_low, margin_high = _bootstrap_ci(margins_array, np.mean, n_boot, rng)
    split_half = split_half_reliability(margins_array, n_boot=n_boot, seed=seed + 1)

    # Phase diagram: move the local baseline along the segment from the global
    # to the observed local baseline and record when each patient's leader
    # flips. The sign of the margin at alpha=1 is the signed flip indicator
    # reported above; the curve and the union bound are the geometry behind it.
    phase_diagram = phase_statistics(
        coef,
        X_scaled,
        groups,
        client_backgrounds,
        fallback,
    )

    cohort_summary = {}
    for band in sorted(set(groups.tolist())):
        mask = groups == band
        if not np.any(mask):
            continue
        band_margins = margins_array[mask]
        band_probabilities = probabilities_array[mask]
        band_ci = [margins_ci[int(i)] for i in np.flatnonzero(mask)]
        cohort_summary[band] = {
            "n": int(mask.sum()),
            "flip_rate": round(float(np.mean(band_margins < 0)), 4),
            "mean_margin": round(float(np.mean(band_margins)), 4),
            "mean_probability_robust": round(float(np.mean(band_probabilities)), 4),
            "fraction_robust_95": round(float(np.mean(band_probabilities >= STABLE_THRESHOLD)), 4),
            "fraction_ci_excludes_zero": round(
                float(
                    np.mean(
                        [
                            float(not (low <= 0.0 <= high))
                            for low, high in band_ci
                        ]
                    )
                ),
                4,
            ),
        }

    return {
        "n_patients": int(len(X_scaled)),
        # Backward-compatible field; now signed top-1.
        "top1_flip_rate": round(flips / n, 4),
        "signed_top1_flip_rate": round(flips / n, 4),
        "flip_rate_ci_low": round(flip_low, 4),
        "flip_rate_ci_high": round(flip_high, 4),
        "mean_spearman": round(float(np.mean(correlations)) if correlations else 1.0, 4),
        "min_spearman": round(float(np.min(correlations)) if correlations else 1.0, 4),
        "mean_abs_shift": round(float(np.mean(shifts)) if shifts else 0.0, 5),
        # The continuous diagnostic.
        "mean_margin": round(float(np.mean(margins_array)), 4),
        "median_margin": round(float(np.median(margins_array)), 4),
        "margin_q10": round(float(np.quantile(margins_array, 0.10)), 4),
        "margin_q90": round(float(np.quantile(margins_array, 0.90)), 4),
        "margin_ci_low": round(margin_low, 4),
        "margin_ci_high": round(margin_high, 4),
        "frac_negative_margin": round(float(np.mean(margins_array < 0)), 4),
        "mean_global_margin": round(float(np.mean(global_margins)), 4),
        # Background uncertainty.
        "background_uncertainty": {
            "draws_per_patient": int(n_boot),
            "mean_probability_robust": round(float(np.mean(probabilities_array)), 4),
            "median_probability_robust": round(float(np.median(probabilities_array)), 4),
            "fraction_robust_95": round(float(np.mean(probabilities_array >= STABLE_THRESHOLD)), 4),
            "fraction_fragile_50": round(float(np.mean(probabilities_array <= FRAGILE_THRESHOLD)), 4),
            "mean_ci_width": round(
                float(np.mean([high - low for low, high in margins_ci])), 4
            ),
        },
        "robustness_curve": {
            "thresholds": [float(t) for t in grid],
            "point_survival": [round(float(v), 4) for v in point_curve],
            "uncertainty_survival": [round(float(v), 4) for v in uncertainty_curve],
        },
        "phase_diagram": phase_diagram,
        "cohort_summary": cohort_summary,
        "split_half": split_half,
        "patient_margins": [round(float(v), 5) for v in margins],
        "patient_status": per_patient_status,
        "patient_top_global": per_patient_top_global,
        "patient_top_local": per_patient_top_local,
        "examples": examples,
        "global_mean_abs": mean_abs_attribution(coef, X_scaled, fallback, features),
        "client_mean_abs": {
            group: mean_abs_attribution(coef, X_scaled[groups == group], background, features)
            for group, background in client_backgrounds.items()
            if np.any(groups == group)
        },
        "feature_order": features,
    }
