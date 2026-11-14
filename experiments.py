"""Reproducible experiment harness for the clinical-depth study.

Runs the five experiments described in the project write-up across multiple
random seeds and both partitioning regimes (age-cohort non-IID vs IID shards),
then writes tidy CSVs plus a summary with 95% confidence intervals.

    python experiments.py --seeds 5 --rounds 15
    python experiments.py --partitions age --seeds 10 --out results

Outputs (in ``results/`` by default):

    e0_centralised.csv        fully pooled logistic-regression reference
    e1_rounds.csv             accuracy / AUC / ECE / Brier / NLL per round
    e2_calibration.csv        calibration method comparison per round
    e3_fairness.csv           per-cohort TPR before and after repair
    e4_explanations.csv       explanation instability per round
    e5_valuation.csv          Shapley client values per metric
    e5_rank_agreement.csv     Kendall tau between metric rankings
    e6_weighting.csv          FedAvg weighting ablation (sample/uniform/positive)
    e7_phase_sensitivity.csv  baseline-multiplier sweep, flip phase and union bound
    summary.json              means and 95% CIs at the final round

``--dataset`` selects a registered dataset. The default is PIMA; the external
validation run uses ``uci_early`` and should be written to a separate folder
(``--out results_second``) so the primary results are not overwritten.

``e6`` answers a question the other experiments cannot: standard FedAvg weights
a hospital by its row count, so under the age partition the smallest and most
prevalent cohort receives the least influence. The ablation re-runs each seed
with ``sample``, ``uniform`` and ``positive`` (prevalence-scaled) weights and
reports what that choice costs in accuracy, calibration and subgroup parity.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from math import sqrt
from typing import Dict, List

import numpy as np

from client_valuation import ClientValuation
from config import AGGREGATION_WEIGHTING, DATASETS, DEFAULT_DATASET, RANDOM_SEED, WEIGHTING_MODES
from data_loader import dataset_keys
from federated_engine import FederatedEngine


def mean_ci(values: List[float]) -> Dict[str, float] | None:
    clean = np.asarray([v for v in values if v is not None and v == v], dtype=float)
    if clean.size == 0:
        return None
    mean = float(clean.mean())
    if clean.size == 1:
        return {"mean": mean, "ci95": 0.0, "n": 1, "sd": 0.0}
    sd = float(clean.std(ddof=1))
    se = sd / sqrt(clean.size)
    try:
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.975, clean.size - 1))
    except Exception:
        critical = 1.96
    return {
        "mean": mean,
        "ci95": critical * se,
        "n": int(clean.size),
        "sd": sd,
    }


def run_seed(
    seed: int,
    rounds: int,
    partition: str,
    weighting: str | None = None,
    with_valuation: bool = True,
    dataset: str | None = None,
) -> Dict[str, object]:
    engine = FederatedEngine(
        partition_mode=partition,
        seed=seed,
        weighting=weighting,
        dataset_key=dataset,
    )
    coef, intercept = engine.initial_parameters()
    history: List[dict] = []

    for round_no in range(1, rounds + 1):
        result = engine.run_round(coef, intercept, round_no=round_no)
        coef, intercept = result["new_coef"], result["new_intercept"]

        calibration = result["calibration"]
        fairness = result["fairness"]
        explanation = result["explanation"]

        history.append(
            {
                "round": round_no,
                "metrics": result["metrics"],
                "calibration": {
                    name: {
                        "ece": entry["ece"],
                        "brier": entry["brier"],
                        "nll": entry["nll"],
                    }
                    for name, entry in calibration["methods"].items()
                },
                "exactness_gap": calibration["exactness_gap"],
                "fairness": {
                    "baseline": fairness["baseline"],
                    "repaired": fairness["repaired"],
                    "thresholds": fairness["repaired_thresholds"],
                    "gaps": fairness["baseline_gaps"],
                    "repaired_gaps": fairness["repaired_gaps"],
                    "cost": fairness["cost"],
                    "tuning_source": fairness["tuning_source"],
                },
                "explanation": {
                    "flip_rate": explanation["top1_flip_rate"],
                    "flip_rate_ci_low": explanation["flip_rate_ci_low"],
                    "flip_rate_ci_high": explanation["flip_rate_ci_high"],
                    "mean_spearman": explanation["mean_spearman"],
                    "mean_abs_shift": explanation["mean_abs_shift"],
                    "mean_margin": explanation["mean_margin"],
                    "median_margin": explanation["median_margin"],
                    "margin_q10": explanation["margin_q10"],
                    "margin_q90": explanation["margin_q90"],
                    "margin_ci_low": explanation["margin_ci_low"],
                    "margin_ci_high": explanation["margin_ci_high"],
                    "frac_negative_margin": explanation["frac_negative_margin"],
                    "mean_probability_robust": explanation["background_uncertainty"][
                        "mean_probability_robust"
                    ],
                    "fraction_robust_95": explanation["background_uncertainty"][
                        "fraction_robust_95"
                    ],
                    "fraction_fragile_50": explanation["background_uncertainty"][
                        "fraction_fragile_50"
                    ],
                    "mean_margin_ci_width": explanation["background_uncertainty"][
                        "mean_ci_width"
                    ],
                    "split_half_flip_abs_diff": explanation["split_half"][
                        "flip_rate_half_abs_diff"
                    ],
                    "split_half_margin_abs_diff": explanation["split_half"][
                        "margin_half_abs_diff"
                    ],
                    "phase": explanation["phase_diagram"],
                },
            }
        )

    return {
        "seed": seed,
        "partition": partition,
        "weighting": engine.weighting,
        "summary": engine.summary(),
        "centralised": engine.centralised_baseline(),
        "history": history,
        "valuation": ClientValuation(engine).report() if with_valuation else None,
    }


def _write_csv(path: str, fieldnames: List[str], rows: List[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="FedHealth experiment harness")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--partitions", default="age,iid")
    parser.add_argument("--out", default="results")
    parser.add_argument("--seed-start", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=dataset_keys(),
        help="registered dataset key (default: pima; use uci_early for external validation)",
    )
    parser.add_argument(
        "--weighting",
        default=AGGREGATION_WEIGHTING,
        choices=list(WEIGHTING_MODES),
        help="aggregation weighting for the main experiments (E1-E5)",
    )
    parser.add_argument(
        "--skip-weighting",
        action="store_true",
        help="skip the E6 weighting ablation",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    seeds = [args.seed_start + i for i in range(args.seeds)]

    E0, E1, E2, E3, E4, E5, E5R, E6, E7 = [], [], [], [], [], [], [], [], []
    summary: Dict[str, object] = {
        "dataset": args.dataset,
        "dataset_name": DATASETS[args.dataset]["name"],
        "seeds": seeds,
        "rounds": args.rounds,
        "aggregation_weighting": args.weighting,
        "partitions": {},
    }

    for partition in partitions:
        runs = []
        for seed in seeds:
            print(f"[{partition}] seed {seed} ...", flush=True)
            run = run_seed(
                seed,
                args.rounds,
                partition,
                weighting=args.weighting,
                dataset=args.dataset,
            )
            runs.append(run)

            central = run["centralised"]["metrics"]
            E0.append(
                {
                    "partition": partition,
                    "seed": seed,
                    "accuracy": round(central["accuracy"], 6),
                    "auc": round(central["auc"], 6),
                    "brier": round(central["brier"], 6),
                    "nll": round(central["nll"], 6),
                    "ece": round(central["ece"], 6),
                    "mce": round(central["mce"], 6),
                    "recall": round(central["recall"], 6),
                    "precision": round(central["precision"], 6),
                }
            )

            for entry in run["history"]:
                metrics = entry["metrics"]
                E1.append(
                    {
                        "partition": partition,
                        "seed": seed,
                        "round": entry["round"],
                        "accuracy": round(metrics["accuracy"], 6),
                        "auc": round(metrics["auc"], 6),
                        "brier": round(metrics["brier"], 6),
                        "nll": round(metrics["nll"], 6),
                        "ece": round(metrics["ece"], 6),
                        "mce": round(metrics["mce"], 6),
                        "recall": round(metrics["recall"], 6),
                        "precision": round(metrics["precision"], 6),
                    }
                )

                for method, values in entry["calibration"].items():
                    E2.append(
                        {
                            "partition": partition,
                            "seed": seed,
                            "round": entry["round"],
                            "method": method,
                            "ece": round(values["ece"], 6),
                            "brier": round(values["brier"], 6),
                            "nll": round(values["nll"], 6),
                        }
                    )

                fairness = entry["fairness"]
                for cohort, row in fairness["baseline"].items():
                    repaired = fairness["repaired"][cohort]
                    E3.append(
                        {
                            "partition": partition,
                            "seed": seed,
                            "round": entry["round"],
                            "cohort": cohort,
                            "n": row["n"],
                            "prevalence": row["prevalence"],
                            "tpr_before": row["tpr"],
                            "fpr_before": row["fpr"],
                            "tpr_after": repaired["tpr"],
                            "fpr_after": repaired["fpr"],
                            "threshold": fairness["thresholds"].get(cohort),
                            "eo_gap_before": fairness["gaps"]["equal_opportunity"],
                            "eo_gap_after": fairness["repaired_gaps"]["equal_opportunity"],
                            "accuracy_before": fairness["cost"]["accuracy_before"],
                            "accuracy_after": fairness["cost"]["accuracy_after"],
                            "tuning_source": fairness["tuning_source"],
                        }
                    )

                E4.append(
                    {
                        "partition": partition,
                        "seed": seed,
                        "round": entry["round"],
                        "flip_rate": entry["explanation"]["flip_rate"],
                        "flip_rate_ci_low": entry["explanation"]["flip_rate_ci_low"],
                        "flip_rate_ci_high": entry["explanation"]["flip_rate_ci_high"],
                        "mean_spearman": entry["explanation"]["mean_spearman"],
                        "mean_abs_shift": entry["explanation"]["mean_abs_shift"],
                        "mean_margin": entry["explanation"]["mean_margin"],
                        "median_margin": entry["explanation"]["median_margin"],
                        "margin_q10": entry["explanation"]["margin_q10"],
                        "margin_q90": entry["explanation"]["margin_q90"],
                        "margin_ci_low": entry["explanation"]["margin_ci_low"],
                        "margin_ci_high": entry["explanation"]["margin_ci_high"],
                        "frac_negative_margin": entry["explanation"]["frac_negative_margin"],
                        "mean_probability_robust": entry["explanation"][
                            "mean_probability_robust"
                        ],
                        "fraction_robust_95": entry["explanation"]["fraction_robust_95"],
                        "fraction_fragile_50": entry["explanation"]["fraction_fragile_50"],
                        "split_half_flip_abs_diff": entry["explanation"][
                            "split_half_flip_abs_diff"
                        ],
                    }
                )

            for metric in ("accuracy", "auc", "ece", "brier"):
                for client, value in run["valuation"]["shapley"][metric].items():
                    E5.append(
                        {
                            "partition": partition,
                            "seed": seed,
                            "client": client,
                            "metric": metric,
                            "shapley": round(float(value), 6),
                        }
                    )
            for comparison, tau in run["valuation"]["rank_agreement"].items():
                E5R.append(
                    {
                        "partition": partition,
                        "seed": seed,
                        "comparison": comparison,
                        "kendall_tau": round(float(tau), 4),
                    }
                )

            phase = run["history"][-1]["explanation"]["phase"]
            for alpha, flip, bound, margin, survival in zip(
                phase["alpha_grid"],
                phase["flip_rate"],
                phase["union_bound"],
                phase["mean_margin"],
                phase["survival"],
            ):
                E7.append(
                    {
                        "partition": partition,
                        "seed": seed,
                        "alpha": alpha,
                        "flip_rate": flip,
                        "union_bound": bound,
                        "mean_margin": margin,
                        "survival": survival,
                        "n_patients": phase["n_patients"],
                    }
                )

        final = [run["history"][-1] for run in runs]
        methods = final[0]["calibration"].keys()
        summary["partitions"][partition] = {
            "clients": runs[0]["summary"]["clients"],
            "n_test": runs[0]["summary"]["n_test"],
            "final_accuracy": mean_ci([r["metrics"]["accuracy"] for r in final]),
            "final_auc": mean_ci([r["metrics"]["auc"] for r in final]),
            "final_ece": mean_ci([r["metrics"]["ece"] for r in final]),
            "final_brier": mean_ci([r["metrics"]["brier"] for r in final]),
            "calibration": {
                method: {
                    "ece": mean_ci([r["calibration"][method]["ece"] for r in final]),
                    "brier": mean_ci([r["calibration"][method]["brier"] for r in final]),
                }
                for method in methods
            },
            "exactness_gap": mean_ci([r["exactness_gap"] for r in final]),
            "fairness": {
                "eo_gap_before": mean_ci([r["fairness"]["gaps"]["equal_opportunity"] for r in final]),
                "eo_gap_after": mean_ci(
                    [r["fairness"]["repaired_gaps"]["equal_opportunity"] for r in final]
                ),
                "accuracy_cost": mean_ci(
                    [r["fairness"]["cost"]["delta_accuracy"] for r in final]
                ),
            },
            "explanation": {
                "flip_rate": mean_ci([r["explanation"]["flip_rate"] for r in final]),
                "mean_spearman": mean_ci([r["explanation"]["mean_spearman"] for r in final]),
                "mean_margin": mean_ci([r["explanation"]["mean_margin"] for r in final]),
                "median_margin": mean_ci(
                    [r["explanation"]["median_margin"] for r in final]
                ),
                "margin_q10": mean_ci([r["explanation"]["margin_q10"] for r in final]),
                "margin_q90": mean_ci([r["explanation"]["margin_q90"] for r in final]),
                "frac_negative_margin": mean_ci(
                    [r["explanation"]["frac_negative_margin"] for r in final]
                ),
                "mean_probability_robust": mean_ci(
                    [r["explanation"]["mean_probability_robust"] for r in final]
                ),
                "fraction_robust_95": mean_ci(
                    [r["explanation"]["fraction_robust_95"] for r in final]
                ),
                "fraction_fragile_50": mean_ci(
                    [r["explanation"]["fraction_fragile_50"] for r in final]
                ),
                "split_half_flip_abs_diff": mean_ci(
                    [r["explanation"]["split_half_flip_abs_diff"] for r in final]
                ),
                "split_half_margin_abs_diff": mean_ci(
                    [r["explanation"]["split_half_margin_abs_diff"] for r in final]
                ),
            },
            "valuation": {
                metric: {
                    client: mean_ci(
                        [run["valuation"]["shapley"][metric][client] for run in runs]
                    )
                    for client in runs[0]["valuation"]["shapley"][metric]
                }
                for metric in ("accuracy", "auc", "ece", "brier")
            },
        }

        # -- E7: phase diagram -----------------------------------------
        # The local baseline is moved along b_g + alpha (b_l - b_g). Flip rate,
        # union bound and mean margin are reported at every alpha, then
        # averaged across seeds. ``alpha*`` is the per-patient critical
        # multiplier; its distribution is summarised separately.
        phases = [run["history"][-1]["explanation"]["phase"] for run in runs]
        alpha_grid = phases[0]["alpha_grid"]
        phase_block = {
            "alpha_grid": alpha_grid,
            "flip_rate": [
                mean_ci([phase["flip_rate"][i] for phase in phases])
                for i in range(len(alpha_grid))
            ],
            "union_bound": [
                mean_ci([phase["union_bound"][i] for phase in phases])
                for i in range(len(alpha_grid))
            ],
            "mean_margin": [
                mean_ci(
                    [
                        phase["mean_margin"][i]
                        for phase in phases
                        if phase["mean_margin"][i] is not None
                    ]
                )
                for i in range(len(alpha_grid))
            ],
            "survival": [
                mean_ci([phase["survival"][i] for phase in phases])
                for i in range(len(alpha_grid))
            ],
            "observed_alpha": phases[0]["observed_alpha"],
            "observed_flip_rate": mean_ci(
                [phase["observed_flip_rate"] for phase in phases]
            ),
            "observed_union_bound": mean_ci(
                [phase["observed_union_bound"] for phase in phases]
            ),
            "observed_mean_margin": mean_ci(
                [phase["observed_mean_margin"] for phase in phases]
            ),
            "critical_alpha": {
                key: mean_ci(
                    [
                        phase["critical_alpha"][key]
                        for phase in phases
                        if phase["critical_alpha"][key] is not None
                    ]
                )
                for key in (
                    "median",
                    "q10",
                    "q25",
                    "q75",
                    "q90",
                    "fraction_finite",
                    "fraction_flip_by_observed",
                )
            },
        }
        summary["partitions"][partition]["explanation"]["phase"] = phase_block

        # -- centralised reference -------------------------------------
        # Same split, same federated scaler, one model trained on every row.
        # The gap against FedAvg is the price of federation, not a tuning gap:
        # this model is unregularised and converged, so it is an upper bound.
        summary["partitions"][partition]["centralised"] = {
            metric: mean_ci(
                [run["centralised"]["metrics"][metric] for run in runs]
            )
            for metric in ("accuracy", "auc", "ece", "brier", "nll")
        }

        # -- E6: aggregation weighting ablation -------------------------
        if args.skip_weighting:
            continue

        weighting_block = {}
        for weighting in WEIGHTING_MODES:
            if weighting == runs[0]["weighting"]:
                probes = runs
            else:
                probes = []
                for seed in seeds:
                    print(
                        f"[{partition}] weighting={weighting} seed {seed} ...",
                        flush=True,
                    )
                    probes.append(
                        run_seed(
                            seed,
                            args.rounds,
                            partition,
                            weighting=weighting,
                            with_valuation=False,
                            dataset=args.dataset,
                        )
                    )

            finals = []
            for probe in probes:
                for entry in probe["history"]:
                    metrics = entry["metrics"]
                    fairness = entry["fairness"]
                    E6.append(
                        {
                            "partition": partition,
                            "seed": probe["seed"],
                            "weighting": weighting,
                            "round": entry["round"],
                            "accuracy": round(metrics["accuracy"], 6),
                            "auc": round(metrics["auc"], 6),
                            "ece": round(metrics["ece"], 6),
                            "eo_gap_before": round(
                                fairness["gaps"]["equal_opportunity"], 6
                            ),
                            "eo_gap_after": round(
                                fairness["repaired_gaps"]["equal_opportunity"], 6
                            ),
                            "accuracy_cost": round(
                                fairness["cost"]["delta_accuracy"], 6
                            ),
                            "flip_rate": round(entry["explanation"]["flip_rate"], 6),
                            "mean_spearman": round(
                                entry["explanation"]["mean_spearman"], 6
                            ),
                            "mean_margin": round(
                                entry["explanation"]["mean_margin"], 6
                            ),
                            "mean_probability_robust": round(
                                entry["explanation"]["mean_probability_robust"], 6
                            ),
                        }
                    )
                finals.append(probe["history"][-1])

            weighting_block[weighting] = {
                "client_weights": [
                    client["aggregation_weight"]
                    for client in probes[0]["summary"]["clients"]
                ],
                "accuracy": mean_ci([r["metrics"]["accuracy"] for r in finals]),
                "auc": mean_ci([r["metrics"]["auc"] for r in finals]),
                "ece": mean_ci([r["metrics"]["ece"] for r in finals]),
                "eo_gap_before": mean_ci(
                    [r["fairness"]["gaps"]["equal_opportunity"] for r in finals]
                ),
                "eo_gap_after": mean_ci(
                    [r["fairness"]["repaired_gaps"]["equal_opportunity"] for r in finals]
                ),
                "accuracy_cost": mean_ci(
                    [r["fairness"]["cost"]["delta_accuracy"] for r in finals]
                ),
                "flip_rate": mean_ci([r["explanation"]["flip_rate"] for r in finals]),
                "mean_spearman": mean_ci(
                    [r["explanation"]["mean_spearman"] for r in finals]
                ),
                "mean_margin": mean_ci(
                    [r["explanation"]["mean_margin"] for r in finals]
                ),
                "mean_probability_robust": mean_ci(
                    [r["explanation"]["mean_probability_robust"] for r in finals]
                ),
            }

        summary["partitions"][partition]["weighting"] = weighting_block

    _write_csv(
        os.path.join(args.out, "e0_centralised.csv"),
        [
            "partition", "seed", "accuracy", "auc", "brier", "nll", "ece", "mce",
            "recall", "precision",
        ],
        E0,
    )
    _write_csv(
        os.path.join(args.out, "e1_rounds.csv"),
        ["partition", "seed", "round", "accuracy", "auc", "brier", "nll", "ece", "mce", "recall", "precision"],
        E1,
    )
    _write_csv(
        os.path.join(args.out, "e2_calibration.csv"),
        ["partition", "seed", "round", "method", "ece", "brier", "nll"],
        E2,
    )
    _write_csv(
        os.path.join(args.out, "e3_fairness.csv"),
        [
            "partition", "seed", "round", "cohort", "n", "prevalence", "tpr_before", "fpr_before",
            "tpr_after", "fpr_after", "threshold", "eo_gap_before", "eo_gap_after",
            "accuracy_before", "accuracy_after", "tuning_source",
        ],
        E3,
    )
    _write_csv(
        os.path.join(args.out, "e4_explanations.csv"),
        [
            "partition", "seed", "round", "flip_rate", "flip_rate_ci_low",
            "flip_rate_ci_high", "mean_spearman", "mean_abs_shift", "mean_margin",
            "median_margin", "margin_q10", "margin_q90", "margin_ci_low",
            "margin_ci_high", "frac_negative_margin", "mean_probability_robust",
            "fraction_robust_95", "fraction_fragile_50", "split_half_flip_abs_diff",
        ],
        E4,
    )
    _write_csv(
        os.path.join(args.out, "e5_valuation.csv"),
        ["partition", "seed", "client", "metric", "shapley"],
        E5,
    )
    _write_csv(
        os.path.join(args.out, "e5_rank_agreement.csv"),
        ["partition", "seed", "comparison", "kendall_tau"],
        E5R,
    )
    if not args.skip_weighting:
        _write_csv(
            os.path.join(args.out, "e6_weighting.csv"),
            [
                "partition", "seed", "weighting", "round", "accuracy", "auc", "ece",
                "eo_gap_before", "eo_gap_after", "accuracy_cost", "flip_rate",
                "mean_spearman", "mean_margin", "mean_probability_robust",
            ],
            E6,
        )
    _write_csv(
        os.path.join(args.out, "e7_phase_sensitivity.csv"),
        [
            "partition", "seed", "alpha", "flip_rate", "union_bound",
            "mean_margin", "survival", "n_patients",
        ],
        E7,
    )

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Console report
    for partition, block in summary["partitions"].items():
        print(f"\n=== {partition.upper()} partition ===")
        for key in ("final_accuracy", "final_auc", "final_ece", "final_brier"):
            value = block[key]
            if value:
                print(f"{key:18s} {value['mean']:.4f} +/- {value['ci95']:.4f}")
        print("calibration method     ECE (mean +/- ci)")
        for method, values in block["calibration"].items():
            ece = values["ece"]
            if ece:
                print(f"  {method:26s} {ece['mean']:.4f} +/- {ece['ci95']:.4f}")
        gap = block["exactness_gap"]
        if gap:
            print(f"  {'exactness gap':26s} {gap['mean']:.2e} +/- {gap['ci95']:.2e}")
        fairness = block["fairness"]
        print(
            f"EO gap {fairness['eo_gap_before']['mean']:.4f} -> "
            f"{fairness['eo_gap_after']['mean']:.4f}  "
            f"(accuracy cost {fairness['accuracy_cost']['mean']:+.4f})"
        )
        explanation = block["explanation"]
        print(
            f"explanation flip rate {explanation['flip_rate']['mean']:.4f}, "
            f"mean margin {explanation['mean_margin']['mean']:+.4f}, "
            f"P(robust) {explanation['mean_probability_robust']['mean']:.4f}, "
            f"robust@95 {explanation['fraction_robust_95']['mean']:.4f}"
        )
        print(
            f"  split-half |diff|: flip {explanation['split_half_flip_abs_diff']['mean']:.4f}, "
            f"margin {explanation['split_half_margin_abs_diff']['mean']:.4f}"
        )
        phase = explanation.get("phase", {})
        if phase:
            critical = phase["critical_alpha"]
            median_alpha = critical.get("median")
            print(
                "  phase diagram: "
                f"flip@alpha=1 {phase['observed_flip_rate']['mean']:.4f}, "
                f"union bound {phase['observed_union_bound']['mean']:.4f}, "
                f"median alpha* "
                + (f"{median_alpha['mean']:.3f}" if median_alpha else "—")
            )
        print("Shapley (accuracy):")
        for client, value in block["valuation"]["accuracy"].items():
            print(f"  {client}: {value['mean']:+.4f} +/- {value['ci95']:.4f}")

        central = block.get("centralised", {})
        if central:
            print(
                "centralised reference  "
                f"accuracy {central['accuracy']['mean']:.4f}, "
                f"AUC {central['auc']['mean']:.4f}, "
                f"ECE {central['ece']['mean']:.4f}  "
                f"(federated accuracy gap "
                f"{block['final_accuracy']['mean'] - central['accuracy']['mean']:+.4f})"
            )

        weighting = block.get("weighting", {})
        if weighting:
            print("aggregation weighting  accuracy   ECE      EO gap (after)  weights")
            for mode, values in weighting.items():
                weights = ", ".join(f"{w:.3f}" for w in values["client_weights"])
                print(
                    f"  {mode:18s} {values['accuracy']['mean']:.4f}    "
                    f"{values['ece']['mean']:.4f}   "
                    f"{values['eo_gap_after']['mean']:.4f}          [{weights}]"
                )

    print(f"\nWrote {len(E1)} round rows and summary to {args.out}/")


if __name__ == "__main__":
    main()
