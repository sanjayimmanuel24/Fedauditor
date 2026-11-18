"""Does the ``explanation_instability`` fault change the *global* model's attributions?

    python -m fedauditor.attribution_check --dataset brfss --models lr xgb --workers 3

For each seed it trains the clean model, a noise-floor model (clean data, only
the minibatch order changed: federation seed + 1000) and the faulty model at
each severity, then compares global attributions with the clean model's:

* ``auc_delta``       - pooled test AUC minus clean (does accuracy move?).
* ``rank_corr``       - Spearman of mean |attribution| per feature.
* ``top3_overlap``    - shared features among the three largest, /3.
* ``patient_top1``    - share of test patients whose signed top factor is unchanged.
* ``rel_change``      - ||A - A_clean|| / ||A_clean|| for mean |attribution|.

For LR it also measures round-to-round churn over the second half of training
(``r2r_*``: the same quantities between consecutive rounds). Attributions are
linear Shapley values against the pooled training mean (LR) and TreeSHAP
(``pred_contribs``, XGB), on a fixed 10k-patient sample of the pooled test set.
Writes results/fedauditor/attribution_check_<dataset>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import xgboost as xgb

from config import BASE_DIR
from federated_explain import _spearman
from federated_metrics import safe_auc
from fedauditor.data import load_federated
from fedauditor.faults import CLEAN, SEVERITY_LEVELS, Scenario, plant
from fedauditor.federation import run_federation

OUT_DIR = os.path.join(BASE_DIR, "results", "fedauditor")
FAULT = "explanation_instability"
KINDS = ("floor",) + SEVERITY_LEVELS
METRICS = ("auc_delta", "rank_corr", "top3_overlap", "patient_top1", "rel_change")


@lru_cache(maxsize=2)
def _data(dataset: str, seed: int):
    return load_federated(dataset, seed=seed)


def _sample(data, seed, n=10_000):
    X, y = data.pooled("test")
    idx = np.random.default_rng(seed).permutation(len(y))[:n]
    return X[idx], y[idx]


def _attr(model_kind, model, X, b):
    """Per-patient signed attributions (n x features)."""
    if model_kind == "lr":
        return (X - b) * model.weights[0]
    return model.booster.predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def _compare(phi, phi_ref, auc, auc_ref) -> dict:
    A, R = np.abs(phi).mean(axis=0), np.abs(phi_ref).mean(axis=0)
    top = lambda v: set(np.argsort(v)[::-1][:3].tolist())
    return {
        "auc_delta": auc - auc_ref,
        "rank_corr": _spearman(A, R),
        "top3_overlap": len(top(A) & top(R)) / 3,
        "patient_top1": float(np.mean(phi.argmax(axis=1) == phi_ref.argmax(axis=1))),
        "rel_change": float(np.linalg.norm(A - R) / np.linalg.norm(R)),
    }


def run_one(dataset: str, model: str, seed: int, kind: str, rounds: int) -> dict:
    warnings.filterwarnings("ignore")
    data = _data(dataset, seed)
    if kind in ("clean", "floor"):
        planted = plant(Scenario(CLEAN, seed=seed, dataset=dataset, model=model, rounds=rounds), data)
    else:
        planted = plant(Scenario(FAULT, kind, seed=seed, dataset=dataset, model=model,
                                 rounds=rounds), data)
    kwargs = planted.run_kwargs()
    if kind == "floor":
        kwargs["seed"] = seed + 1000
    result = run_federation(planted.data, record_weights=(model == "lr"), **kwargs)

    X, y = _sample(data, seed)
    b = data.pooled("train")[0].mean(axis=0)
    phi = _attr(model, result.model, X, b)
    out = {"kind": kind, "seed": seed, "phi": phi,
           "auc": safe_auc(y, result.model.predict_proba(X)), "r2r": None}
    if model == "lr":
        half = [h for h in result.history if h["round"] > rounds // 2]
        steps = []
        prev = None
        for h in half:
            w = np.asarray(h["weights"][0])
            p = (X - b) * w
            if prev is not None:
                c = _compare(p, prev, 0.0, 0.0)
                c.pop("auc_delta")
                steps.append(c)
            prev = p
        out["r2r"] = {k: float(np.mean([s[k] for s in steps])) for k in steps[0]}
    return out


def _summary(values):
    v = np.asarray(values, dtype=float)
    return {"mean": round(float(v.mean()), 4), "sd": round(float(v.std()), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="brfss")
    parser.add_argument("--models", nargs="+", default=["lr", "xgb"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    report = {"args": vars(args), "models": {}}
    with ProcessPoolExecutor(args.workers) as pool:
        for model in args.models:
            jobs = {(s, k): pool.submit(run_one, args.dataset, model, s, k, args.rounds)
                    for s in args.seeds for k in ("clean",) + KINDS}
            res = {key: f.result() for key, f in jobs.items()}
            rows = {}
            for kind in KINDS:
                per_seed = [_compare(res[(s, kind)]["phi"], res[(s, "clean")]["phi"],
                                     res[(s, kind)]["auc"], res[(s, "clean")]["auc"])
                            for s in args.seeds]
                row = {m: _summary([p[m] for p in per_seed]) for m in METRICS}
                if model == "lr":
                    for m in res[(args.seeds[0], kind)]["r2r"]:
                        row[f"r2r_{m}"] = _summary([res[(s, kind)]["r2r"][m] for s in args.seeds])
                rows[kind] = row
            if model == "lr":
                rows["clean"] = {f"r2r_{m}": _summary([res[(s, "clean")]["r2r"][m] for s in args.seeds])
                                 for m in res[(args.seeds[0], "clean")]["r2r"]}
            report["models"][model] = rows

            cols = [m for m in rows["high"]]
            print(f"\n{model.upper()} (mean over {len(args.seeds)} seeds; vs clean model of same seed)")
            print(f"{'':>7} " + " ".join(f"{c:>17}" for c in cols))
            for kind, row in rows.items():
                print(f"{kind:>7} " + " ".join(
                    f"{row[c]['mean']:>10.4f}+-{row[c]["sd"]:<5.4f}" if c in row else f"{'':>17}"
                    for c in cols))

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"attribution_check_{args.dataset}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
