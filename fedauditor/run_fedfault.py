"""Run the FedFault benchmark: every fault x severity x seed, plus clean runs.

    python -m fedauditor.run_fedfault --dataset brfss --models lr --workers 4
    python -m fedauditor.run_fedfault --dataset pima --models lr xgb mlp --faults poisoning drift

Appends one JSON line per run to results/fedauditor/fedfault_<dataset>_<model>.jsonl
and skips runs already in the file, so it can be stopped and resumed. Each line
holds the scenario, its ground truth, the round history (per-client test AUC
and update norms), final metrics on the data the auditor sees (``final``) and on
the untouched data (``clean_eval``, for measuring harm).
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

from config import BASE_DIR
from fedauditor.data import load_federated
from fedauditor.faults import FAULTS, SEVERITY_LEVELS, Scenario, benchmark_grid, plant
from fedauditor.federation import evaluate, run_federation

OUT_DIR = os.path.join(BASE_DIR, "results", "fedauditor")


@lru_cache(maxsize=2)
def _data(dataset: str, seed: int, extra: tuple):
    return load_federated(dataset, seed=seed, **dict(extra))


def run_scenario(scenario: Scenario, extra: tuple = ()) -> dict:
    warnings.filterwarnings("ignore")
    planted = plant(scenario, _data(scenario.dataset, scenario.seed, extra))
    result = run_federation(planted.data, **planted.run_kwargs())
    clean = evaluate(result.model, planted.clean_data, full=True)
    return {
        "scenario": scenario.to_dict(), "truth": planted.truth,
        "seconds": round(result.seconds, 2), "history": result.history,
        "final": result.final,
        "clean_eval": {"_pooled": clean["_pooled"],
                       **{k: {"auc": v["auc"], "brier": v["brier"], "ece": v["ece"]}
                          for k, v in clean.items() if k != "_pooled"}},
    }


def _done_ids(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as fh:
        return {json.loads(line)["scenario"]["id"] for line in fh if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="FedFault benchmark")
    parser.add_argument("--dataset", default="brfss", choices=["brfss", "pima", "uci_early"])
    parser.add_argument("--models", nargs="+", default=["lr"])
    parser.add_argument("--faults", nargs="+", default=list(FAULTS))
    parser.add_argument("--severities", nargs="+", default=list(SEVERITY_LEVELS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    parser.add_argument("--clean-seeds", type=int, nargs="+", default=None,
                        help="seeds for clean runs (default: --seeds)")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--strategy", default="fedavg")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-rows-per-client", type=int, default=None)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    extra = (("max_rows_per_client", args.max_rows_per_client),) if args.max_rows_per_client else ()
    os.makedirs(OUT_DIR, exist_ok=True)
    for model in args.models:
        path = os.path.join(OUT_DIR, f"fedfault_{args.dataset}_{model}.jsonl")
        grid = benchmark_grid(args.dataset, model, args.seeds, args.rounds, args.strategy,
                              args.faults, args.severities, args.clean_seeds)
        done = _done_ids(path)
        todo = [s for s in grid if s.id not in done]
        # Group by seed so each worker loads a seed's data once.
        todo.sort(key=lambda s: s.seed)
        print(f"{model}: {len(todo)} to run, {len(grid) - len(todo)} already in {path}")
        with open(path, "a", encoding="utf-8") as fh:
            if args.workers <= 1:
                results = (run_scenario(s, extra) for s in todo)
            else:
                pool = ProcessPoolExecutor(args.workers)
                results = (f.result() for f in as_completed(
                    [pool.submit(run_scenario, s, extra) for s in todo]))
            for k, rec in enumerate(results, 1):
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                sc, ev = rec["scenario"], rec["clean_eval"]["_pooled"]
                print(f"[{k}/{len(todo)}] {sc['fault']:>24} {sc['severity']:>6} s{sc['seed']} "
                      f"clean AUC {ev['auc']:.3f} ECE {ev['ece']:.3f} "
                      f"clients {rec['truth']['clients']} ({rec['seconds']:.0f}s)")
            if args.workers > 1:
                pool.shutdown()


if __name__ == "__main__":
    main()
