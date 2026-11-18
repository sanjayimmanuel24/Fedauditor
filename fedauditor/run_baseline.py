"""Clean (fault-free) federated baselines, with the centralised reference.

    python -m fedauditor.run_baseline --dataset brfss --models lr xgb mlp --rounds 20
    python -m fedauditor.run_baseline --dataset pima --seeds 20 21 22

Writes results/fedauditor/baseline_<dataset>.json and prints pooled test AUC,
Brier and ECE for each model, federated and centralised.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

from config import BASE_DIR
from fedauditor.data import load_federated
from fedauditor.federation import run_federation

OUT_DIR = os.path.join(BASE_DIR, "results", "fedauditor")


def main() -> None:
    parser = argparse.ArgumentParser(description="FedAuditor clean baselines")
    parser.add_argument("--dataset", default="brfss", choices=["brfss", "pima", "uci_early"])
    parser.add_argument("--models", nargs="+", default=["lr", "xgb", "mlp"])
    parser.add_argument("--strategy", default="fedavg")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20])
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--max-rows-per-client", type=int, default=None)
    parser.add_argument("--no-central", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    extra = {"year": args.year} if args.dataset == "brfss" else {}
    if args.max_rows_per_client:
        extra["max_rows_per_client"] = args.max_rows_per_client

    runs = []
    print(f"{'seed':>4} {'model':>5} {'setting':>11} {'AUC':>6} {'Brier':>6} {'ECE':>6} {'sec':>6}")
    for seed in args.seeds:
        data = load_federated(args.dataset, seed=seed, **extra)
        settings = [("federated", data)] + ([] if args.no_central else [("centralised", data.merged())])
        for model in args.models:
            for setting, ds in settings:
                result = run_federation(ds, model=model, rounds=args.rounds,
                                        strategy=args.strategy, seed=seed,
                                        local_epochs=args.local_epochs)
                pooled = result.final["_pooled"]
                print(f"{seed:>4} {model:>5} {setting:>11} {pooled['auc']:>6.3f} "
                      f"{pooled['brier']:>6.3f} {pooled['ece']:>6.3f} {result.seconds:>6.1f}")
                runs.append({"setting": setting, **result.to_dict()})

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"baseline_{args.dataset}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"args": vars(args), "clients": data.summary(), "notes": data.notes,
                   "runs": runs}, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
