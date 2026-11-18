"""Run one FedAuditor audit on one FedFault scenario and print fact-check and score.

    python -m fedauditor.run_audit --fault poisoning --severity low --seed 20 --model lr

Saves the full record (report, tool log, fact-check, score, token usage) to
``results/fedauditor/audits/<scenario id>__<agent>.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import warnings

from fedauditor.agent import ClaudeAuditor, MODEL, audit
from fedauditor.audit_env import AuditRun
from fedauditor.data import load_federated
from fedauditor.faults import CLEAN, FAULTS, SEVERITY_LEVELS, Scenario
from fedauditor.tools import DEFAULT_BUDGET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "results", "fedauditor", "audits")
# Claude Opus 5.5 list prices, USD per million tokens (cache reads 0.1x, writes 1.25x input).
PRICE = {"input": 4.0, "output": 20.0, "cache_read": 0.4, "cache_write": 5.0}


def print_summary(rec: dict) -> None:
    s, fc, t = rec["score"], rec["fact_check"], rec["truth"]
    print(f"\nTruth   : {t['fault']} ({t['severity']}) clients={t['clients']} onset={t['onset_round']}")
    for f in rec["report"]["findings"]:
        print(f"Finding : {f['client']} {f['fault_type']} onset={f['onset_round']} "
              f"conf={f['confidence']} claims={len(f['claims'])}")
    if not rec["report"]["findings"]:
        print("Finding : none (clean)")
    print(f"Score   : precision={s['precision']} recall={s['recall']} type_acc={s['type_accuracy']} "
          f"onset_err={s['onset_abs_error']} false_alarm={s['false_alarm']}")
    print(f"Facts   : {fc['supported']}/{fc['claims']} claims supported")
    for u in fc["unsupported"][:5]:
        print(f"  unsupported: {u['client']} {u['claim']} -> {u['reason']}")
    u = rec["usage"]
    if u:
        cost = sum(u[k] * PRICE[k] for k in PRICE) / 1e6
        print(f"Agent   : {rec['stop']} after {rec['turns']} turns, {len(rec['log'])} tool calls, "
              f"epsilon {rec['budget']['spent']}/{rec['budget']['total']}, "
              f"tokens in={u['input']} out={u['output']} cache_read={u['cache_read']}, ~${cost:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brfss")
    ap.add_argument("--model", default="lr", choices=["lr", "mlp", "xgb"])
    ap.add_argument("--fault", required=True, choices=list(FAULTS) + [CLEAN])
    ap.add_argument("--severity", default="medium", choices=list(SEVERITY_LEVELS) + ["none"])
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--llm", default=MODEL)
    ap.add_argument("--effort", default="high")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    sev = "none" if args.fault == CLEAN else args.severity
    scenario = Scenario(args.fault, sev, seed=args.seed, dataset=args.dataset,
                        model=args.model, rounds=args.rounds)
    print(f"Scenario {scenario.id}: running federation with recording...")
    run = AuditRun.from_scenario(scenario, load_federated(args.dataset, seed=args.seed))
    print(f"  done in {run.result.seconds:.1f}s; auditing with {args.llm}")
    rec = audit(run, ClaudeAuditor(model=args.llm, effort=args.effort), budget=args.budget)
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{scenario.id}__{args.llm}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    print_summary(rec)
    print(f"Saved {os.path.relpath(path, BASE_DIR)}")


if __name__ == "__main__":
    main()
