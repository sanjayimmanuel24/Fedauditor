"""Compare auditors on the FedFault grid, optionally at confidence cut-offs.

    python -m fedauditor.compare_agents --model lr --labels rules claude-haiku-5-5 --conf 0 0.5 0.7

Reads ``results/fedauditor/audit_grid_<ds>_<model>_<label>.jsonl`` and re-scores each
scenario keeping only findings with confidence >= the cut-off (the rule baseline always
reports confidence 0.7). Prints overall precision / recall / scenario type accuracy / onset
error / clean false-alarm rate / claim support / cost, then recall per fault x severity.
Only scenarios present for every label are compared. Writes
``results/fedauditor/compare_<ds>_<model>.json``.
"""
from __future__ import annotations

import argparse
import json
import os

from fedauditor.faults import FAULTS, SEVERITY_LEVELS
from fedauditor.report import aggregate, score
from fedauditor.run_audit_grid import RES_DIR, _load, grid_path


def rescore(rec: dict, conf: float) -> dict:
    kept = [f for f in rec["findings"] if (f.get("confidence") or 0) >= conf]
    return score({"findings": kept}, rec["truth"])


def compare(dataset: str, model: str, labels, confs) -> dict:
    data = {lab: {r["id"]: r for r in _load(grid_path(dataset, model, lab))} for lab in labels}
    common = sorted(set.intersection(*(set(d) for d in data.values()))) if data else []
    out = {"n_common": len(common), "rows": {}}
    for lab in labels:
        recs = [data[lab][i] for i in common]
        claims = sum(r["fact_check"]["claims"] for r in recs)
        sup = sum(r["fact_check"]["supported"] for r in recs)
        for conf in (confs if lab != "rules" else [0.0]):
            scores = [rescore(r, conf) for r in recs]
            cells = {}
            for f in FAULTS:
                for s in SEVERITY_LEVELS:
                    sub = [x for x in scores if x["fault"] == f and x["severity"] == s]
                    if sub:
                        cells[f"{f}/{s}"] = aggregate(sub)["recall"]
            out["rows"][f"{lab}@{conf:g}"] = {
                **aggregate(scores), "claim_support": sup / claims if claims else None,
                "cost_usd": round(sum(r["cost_usd"] for r in recs), 3), "by_cell": cells}
    return out


def _f(x):
    return "  -  " if x is None else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brfss")
    ap.add_argument("--model", default="lr", choices=["lr", "mlp", "xgb"])
    ap.add_argument("--labels", nargs="+", default=["rules", "claude-haiku-5-5"])
    ap.add_argument("--conf", type=float, nargs="+", default=[0.0, 0.5, 0.7])
    args = ap.parse_args()
    res = compare(args.dataset, args.model, args.labels, args.conf)
    print(f"{res['n_common']} scenarios present for all of {args.labels}\n")
    w = max(len(k) for k in res["rows"]) if res["rows"] else 10
    print(f"{'agent@conf':{w}s}  prec   rec  type onset   FAR facts   cost")
    for k, r in res["rows"].items():
        print(f"{k:{w}s} {_f(r['precision'])} {_f(r['recall'])} {_f(r['scenario_type_accuracy'])} "
              f"{_f(r['onset_abs_error'])} {_f(r['false_alarm_rate'])} {_f(r['claim_support'])} ${r['cost_usd']:.2f}")
    print("\nRecall by fault x severity (low medium high):")
    for f in FAULTS:
        print(f"  {f:24s} " + "  |  ".join(
            f"{k.split('@')[0][:6]}@{k.split('@')[1]} " + " ".join(_f(r["by_cell"].get(f"{f}/{s}")) for s in SEVERITY_LEVELS)
            for k, r in res["rows"].items()))
    path = os.path.join(RES_DIR, f"compare_{args.dataset}_{args.model}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"\nSaved {os.path.relpath(path)}")


if __name__ == "__main__":
    main()
