"""Turn the experiment outputs into conference-ready tables.

Usage::

    python paper_assets.py --results results --external results_second
    python paper_assets.py --results results --external results_second --latex

Reads ``summary.json`` and writes ``paper_tables.md`` and, with ``--latex``,
a ``paper_tables.tex`` containing the same tables. ``--external`` adds the
second-dataset validation table. The script deliberately does not invent
numbers: every cell comes from the harness.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _mean_ci(entry: Dict | None, digits: int = 3, signed: bool = False) -> str:
    if not entry:
        return "—"
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    mean = fmt.format(entry["mean"])
    return f"{mean} ± {entry['ci95']:.{digits}f}"


def _mean(entry: Dict | None, digits: int = 3, signed: bool = False) -> str:
    if not entry:
        return "—"
    return f"{entry['mean']:+.{digits}f}" if signed else f"{entry['mean']:.{digits}f}"


def _table_main(summary: Dict) -> List[str]:
    lines = [
        "| Regime | Accuracy | ROC AUC | ECE raw | ECE federated binning | EO gap before → after | Signed top-1 flip | Mean margin R | Mean P(R>0) | Robust at 95% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for partition, block in summary.get("partitions", {}).items():
        calibration = block.get("calibration", {})
        raw = calibration.get("raw", {}).get("ece")
        binned = calibration.get("federated_binning", {}).get("ece")
        fairness = block.get("fairness", {})
        explanation = block.get("explanation", {})
        lines.append(
            "| {regime} | {acc} | {auc} | {raw} | {binned} | {eo_before} → {eo_after} | {flip} | {margin} | {prob} | {robust} |".format(
                regime=partition,
                acc=_mean_ci(block.get("final_accuracy")),
                auc=_mean_ci(block.get("final_auc")),
                raw=_mean_ci(raw),
                binned=_mean_ci(binned),
                eo_before=_mean_ci(fairness.get("eo_gap_before")),
                eo_after=_mean_ci(fairness.get("eo_gap_after")),
                flip=_mean_ci(explanation.get("flip_rate")),
                margin=_mean_ci(explanation.get("mean_margin"), signed=True),
                prob=_mean_ci(explanation.get("mean_probability_robust")),
                robust=_mean_ci(explanation.get("fraction_robust_95")),
            )
        )
        central = block.get("centralised", {})
        if central:
            lines.append(
                "| {regime} (pooled) | {acc} | {auc} | {ece} | — | — | — | — | — | — |".format(
                    regime=partition,
                    acc=_mean_ci(central.get("accuracy")),
                    auc=_mean_ci(central.get("auc")),
                    ece=_mean_ci(central.get("ece")),
                )
            )
    return lines


def _table_diagnostic(summary: Dict) -> List[str]:
    lines = [
        "| Regime | Flip rate (95% CI) | Mean margin R (95% CI) | P(R>0) | Robust at 95% | Fragile at 50% | Split-half flip Δ | Union bound at α=1 | Median α* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for partition, block in summary.get("partitions", {}).items():
        explanation = block.get("explanation", {})
        phase = explanation.get("phase", {})
        critical = phase.get("critical_alpha", {})
        lines.append(
            "| {regime} | {flip} | {margin} | {prob} | {robust} | {fragile} | {split_flip} | {bound} | {alpha} |".format(
                regime=partition,
                flip=_mean_ci(explanation.get("flip_rate"), 3),
                margin=_mean_ci(explanation.get("mean_margin"), 3, signed=True),
                prob=_mean(explanation.get("mean_probability_robust"), 3),
                robust=_mean(explanation.get("fraction_robust_95"), 3),
                fragile=_mean(explanation.get("fraction_fragile_50"), 3),
                split_flip=_mean(explanation.get("split_half_flip_abs_diff"), 4),
                bound=_mean_ci(phase.get("observed_union_bound"), 3),
                alpha=_mean_ci(critical.get("median"), 3),
            )
        )
    return lines


def _phase_value(phase: Dict, key: str, alpha: float) -> str:
    grid = phase.get("alpha_grid", [])
    if not grid:
        return "—"
    index = min(range(len(grid)), key=lambda i: abs(grid[i] - alpha))
    return _mean_ci(phase.get(key, [None] * len(grid))[index], 3)


def _table_phase(summary: Dict) -> List[str]:
    lines = [
        "| Dataset | Regime | Flip @ α=0.5 | Flip @ α=1 | Flip @ α=1.5 | Flip @ α=2 | Union bound @ α=1 | Median α* | Finite α* |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    dataset = summary.get("dataset_name", summary.get("dataset", "primary"))
    for partition, block in summary.get("partitions", {}).items():
        phase = block.get("explanation", {}).get("phase", {})
        critical = phase.get("critical_alpha", {})
        lines.append(
            "| {dataset} | {regime} | {f05} | {f1} | {f15} | {f2} | {bound} | {median} | {finite} |".format(
                dataset=dataset,
                regime=partition,
                f05=_phase_value(phase, "flip_rate", 0.5),
                f1=_phase_value(phase, "flip_rate", 1.0),
                f15=_phase_value(phase, "flip_rate", 1.5),
                f2=_phase_value(phase, "flip_rate", 2.0),
                bound=_mean_ci(phase.get("observed_union_bound"), 3),
                median=_mean_ci(critical.get("median"), 3),
                finite=_mean(critical.get("fraction_finite"), 3),
            )
        )
    return lines


def _table_external(primary: Dict, external: Dict) -> List[str]:
    lines = [
        "| Dataset | Regime | n test | Accuracy | ROC AUC | ECE raw | Signed flip | Mean margin R | P(R>0) | Robust at 95% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for summary in (primary, external):
        dataset = summary.get("dataset_name", summary.get("dataset", "dataset"))
        for partition, block in summary.get("partitions", {}).items():
            calibration = block.get("calibration", {})
            raw = calibration.get("raw", {}).get("ece")
            explanation = block.get("explanation", {})
            lines.append(
                "| {dataset} | {regime} | {n} | {acc} | {auc} | {ece} | {flip} | {margin} | {prob} | {robust} |".format(
                    dataset=dataset,
                    regime=partition,
                    n=block.get("n_test", "—"),
                    acc=_mean_ci(block.get("final_accuracy")),
                    auc=_mean_ci(block.get("final_auc")),
                    ece=_mean_ci(raw),
                    flip=_mean_ci(explanation.get("flip_rate")),
                    margin=_mean_ci(explanation.get("mean_margin"), signed=True),
                    prob=_mean_ci(explanation.get("mean_probability_robust")),
                    robust=_mean_ci(explanation.get("fraction_robust_95")),
                )
            )
    return lines


def _table_weighting(summary: Dict) -> List[str]:
    lines = [
        "| Partition | Weighting | Client weights | Accuracy | ECE | EO gap after | Mean margin R | P(R>0) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for partition, block in summary.get("partitions", {}).items():
        for mode, values in block.get("weighting", {}).items():
            weights = " / ".join(f"{w:.3f}" for w in values.get("client_weights", []))
            lines.append(
                "| {regime} | {mode} | {weights} | {acc} | {ece} | {eo} | {margin} | {prob} |".format(
                    regime=partition,
                    mode=mode,
                    weights=weights,
                    acc=_mean_ci(values.get("accuracy")),
                    ece=_mean_ci(values.get("ece")),
                    eo=_mean_ci(values.get("eo_gap_after")),
                    margin=_mean_ci(values.get("mean_margin"), signed=True),
                    prob=_mean_ci(values.get("mean_probability_robust")),
                )
            )
    return lines


def _latex_escape(cell: str) -> str:
    return (
        cell.replace("±", r"$\pm$")
        .replace("→", r"$\rightarrow$")
        .replace("—", "--")
        .replace("%", r"\%")
        .replace("Δ", r"$\Delta$")
        .replace("α", r"$\alpha$")
        .replace(">", r"$>$")
        .replace("<", r"$<$")
    )


def _to_latex(markdown_lines: List[str], caption: str, label: str) -> str:
    header = markdown_lines[0].strip("|").split("|")
    rows = []
    for line in markdown_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        cells = [_latex_escape(cell) for cell in cells]
        rows.append(" & ".join(cells) + r" \\")
    header = [_latex_escape(cell.replace("|", r"$\vert$")) for cell in header]
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{" + "l" + "r" * (len(header) - 1) + "}",
            r"\toprule",
            " & ".join(cell.strip() for cell in header) + r" \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{_latex_escape(caption)}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build conference tables from results/")
    parser.add_argument("--results", default="results")
    parser.add_argument(
        "--external",
        default=None,
        help="optional second results folder for external validation (e.g. results_second)",
    )
    parser.add_argument("--latex", action="store_true", help="also write paper_tables.tex")
    args = parser.parse_args()

    with open(os.path.join(args.results, "summary.json"), encoding="utf-8") as handle:
        summary = json.load(handle)

    external = None
    if args.external:
        with open(
            os.path.join(args.external, "summary.json"), encoding="utf-8"
        ) as handle:
            external = json.load(handle)

    main_table = _table_main(summary)
    diagnostic_table = _table_diagnostic(summary)
    weighting_table = _table_weighting(summary)
    phase_table = _table_phase(summary)
    tables = [
        "## Table 1 — Regime comparison\n\n" + "\n".join(main_table),
        "## Table 2 — Baseline-robustness diagnostic\n\n"
        + "\n".join(diagnostic_table),
        "## Table 3 — Aggregation weighting ablation\n\n"
        + "\n".join(weighting_table),
        "## Table 4 — Baseline phase sensitivity\n\n" + "\n".join(phase_table),
    ]
    if external is not None:
        external_table = _table_external(summary, external)
        tables.append(
            "## Table 5 — External validation\n\n" + "\n".join(external_table)
        )

    markdown = "\n\n".join(
        [
            "# Conference tables",
            "Generated from `summary.json` by `paper_assets.py`. Values are mean ± 95% CI across seeds unless the column says otherwise.",
            *tables,
        ]
    )
    output_path = os.path.join(args.results, "paper_tables.md")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(markdown)
    print(f"Wrote {output_path}")

    if args.latex:
        latex_tables = [
            _to_latex(main_table, "Federated audit metrics by partitioning regime.", "tab:regimes"),
            _to_latex(
                diagnostic_table,
                "Baseline-robustness diagnostic. The margin R is continuous; the flip rate is P(R<0).",
                "tab:robustness",
            ),
            _to_latex(
                weighting_table,
                "Aggregation-weighting ablation.",
                "tab:weighting",
            ),
            _to_latex(
                phase_table,
                "Baseline phase sensitivity. The local baseline moves from the global baseline (alpha=0) to the observed local baseline (alpha=1) and beyond.",
                "tab:phase",
            ),
        ]
        if external is not None:
            latex_tables.append(
                _to_latex(
                    _table_external(summary, external),
                    "External validation on a second public dataset.",
                    "tab:external",
                )
            )
        tex = "\n\n".join(latex_tables)
        tex_path = os.path.join(args.results, "paper_tables.tex")
        with open(tex_path, "w", encoding="utf-8") as handle:
            handle.write(tex)
        print(f"Wrote {tex_path}")


if __name__ == "__main__":
    main()
