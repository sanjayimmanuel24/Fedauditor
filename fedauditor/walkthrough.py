"""Step-by-step demo of the whole FedAuditor pipeline, for showing the project (no API cost).

    python -m fedauditor.walkthrough

Runs live, on BRFSS seed 20: (1) the federation, (2) a clean model, (3) a planted fault,
(4) the privacy-budgeted tools, (5) a full audit by the rule auditor with fact-check and score.
Then (6) prints the saved Phase 4 results (LLM audits are not re-run, they cost money).
Each step's console output is also saved to ``results/fedauditor/walkthrough/stepN.txt``.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import textwrap
import warnings

from fedauditor.agent import audit
from fedauditor.audit_env import AuditRun
from fedauditor.baseline import RuleAuditor
from fedauditor.compare_agents import compare
from fedauditor.data import load_federated
from fedauditor.faults import Scenario
from fedauditor.run_audit_grid import RES_DIR
from fedauditor.tools import AuditSession

OUT = os.path.join(RES_DIR, "walkthrough")
SEED = 20


def _rows(table, n=None):
    cols = table["columns"]
    out = ["  " + " | ".join(cols)]
    for r in table["rows"][:n]:
        out.append("  " + " | ".join(str(v) for v in r))
    if n is not None and len(table["rows"]) > n:
        out.append(f"  ... ({len(table['rows']) - n} more rows)")
    return "\n".join(out)


def step(n, title, fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print(f"=== Step {n}: {title} ===")
        result = fn()
    # wrap long lines so the saved output fits a printed page
    text = "\n".join(textwrap.fill(line, 100, subsequent_indent=" " * 19, break_on_hyphens=False)
                     if len(line) > 100 else line for line in buf.getvalue().split("\n"))
    print(text)
    with open(os.path.join(OUT, f"step{n}.txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return result


def s1_data():
    data = load_federated("brfss", seed=SEED)
    n = {c.name: sum(len(y) for y in c.y.values()) for c in data.clients}
    total = sum(n.values())
    prev = sum(c.y[s].sum() for c in data.clients for s in c.y) / total
    small = min(n, key=n.get)
    big = max(n, key=n.get)
    print(f"Dataset          : {data.name}")
    print(f"Clients (states) : {len(data.clients)}")
    print(f"Records          : {total:,} (train/cal/test split inside every client)")
    print(f"Diabetes rate    : {prev:.1%}")
    print(f"Smallest client  : {small} ({n[small]:,} records); largest: {big} ({n[big]:,})")
    print(f"Features ({data.n_features}) : {', '.join(data.feature_names)}")
    return data


def s2_clean(data):
    run = AuditRun.from_scenario(Scenario("clean", "none", seed=SEED, dataset="brfss", model="lr"), data)
    fin = run.result.final
    aucs = {k: v["auc"] for k, v in fin.items() if not k.startswith("_")}
    print("FedAvg, logistic regression, 20 rounds, 52 clients")
    print(f"Training time    : {run.result.seconds:.1f} s")
    print(f"Pooled test AUC  : {fin['_pooled']['auc']:.3f}   ECE: {fin['_pooled']['ece']:.3f}")
    print(f"Per-state AUC    : {min(aucs.values()):.3f} - {max(aucs.values()):.3f}")
    return run


def s3_fault(data, clean):
    sc = Scenario("poisoning", "low", seed=SEED, dataset="brfss", model="lr")
    run = AuditRun.from_scenario(sc, data)
    t = run.truth
    fin, cfin = run.result.final, clean.result.final
    print(f"Scenario         : {sc.id}")
    print(f"Planted fault    : {t['fault']} ({t['severity']}): update replaced by -1 x itself")
    print(f"Faulty clients   : {', '.join(t['clients'])} from round {t['onset_round']}")
    print(f"Pooled AUC       : clean {cfin['_pooled']['auc']:.4f} -> faulty {fin['_pooled']['auc']:.4f}"
          f"  (change {fin['_pooled']['auc'] - cfin['_pooled']['auc']:+.4f})")
    print("=> The pooled number hides the fault; an auditor has to look per client.")
    return run


def s4_tools(run):
    s = AuditSession(run, budget=20.0, seed=0)
    ov = s.call("federation_overview")
    print(f"c1 federation_overview (free): {ov['n_clients']} clients, {ov['rounds']} rounds, "
          f"model {ov['model']}, budget left {ov['budget_left']}")
    cu = s.call("client_updates", {"epsilon": 3})
    print(f"\nc2 client_updates (epsilon 3, budget left {cu['budget_left']}):"
          " noisy update statistics per client; only outliers with |z|>3 carry flags")
    tbl = cu["clients"]
    cols = tbl["columns"]
    fi = cols.index("flags")
    flagged = {"columns": cols, "rows": [r for r in tbl["rows"] if r[fi]]}
    print(_rows(flagged))
    print(f"  ({len(tbl['rows'])} clients in the table, {len(flagged['rows'])} flagged)")
    bad = s.call("client_updates", {"epsilon": 50})
    print(f"\nc3 client_updates (epsilon 50) -> refused: {bad.get('error')}")
    print(f"Budget spent {s.accountant.spent:g} of {s.accountant.total:g}; every call is logged as c1, c2, ...")


def s5_audit(run):
    rec = audit(run, RuleAuditor())
    print(f"Auditor          : rule baseline (same tools and budget as the LLM agent)")
    print(f"Tool calls       : {', '.join(e['id'] + ' ' + e['tool'] for e in rec['log'])}")
    print(f"Budget           : epsilon {rec['budget']['spent']:g} of {rec['budget']['total']:g}")
    for f in rec["report"]["findings"]:
        print(f"Finding          : {f['client']} -> {f['fault_type']}, onset round {f['onset_round']}, "
              f"confidence {f['confidence']}")
        for c in f["claims"][:2]:
            keys = ", ".join(f"{k}={c[k]}" for k in ("client", "round", "feature", "group") if k in c)
            print(f"   claim: {c['metric']} = {c['value']} ({keys}) cites {c['call_id']}")
    fc, sc = rec["fact_check"], rec["score"]
    print(f"Fact-check       : {fc['supported']}/{fc['claims']} claims found in the tool log")
    print(f"Score vs truth   : precision {sc['precision']}, recall {sc['recall']}, "
          f"type correct {sc['scenario_type_correct']}, onset error {sc['onset_abs_error']}")
    with open(os.path.join(OUT, "rule_audit_poisoning_low_s20.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1, default=str)
    return rec


def s6_saved():
    res = compare("brfss", "lr", ["rules", "claude-haiku-5-5", "claude-haiku-5-5-p5"], [0.0])
    print(f"{res['n_common']} evaluation scenarios (8 faults x 3 severities x 5 seeds + 5 clean), LR")
    print(f"{'auditor':24s} prec   rec  type onset  FAR  verified   cost")
    names = {"rules@0": "Rule baseline", "claude-haiku-5-5@0": "Haiku 5.5 prompt v4",
             "claude-haiku-5-5-p5@0": "Haiku 5.5 prompt v5"}
    for k, r in res["rows"].items():
        print(f"{names[k]:24s} {r['precision']:.2f}  {r['recall']:.2f}  {r['scenario_type_accuracy']:.2f}"
              f"  {r['onset_abs_error']:.2f}  {r['false_alarm_rate']:.2f}  {r['claim_support']:.3f}"
              f"    ${r['cost_usd']:.2f}")


def main() -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    data = step(1, "Data: BRFSS 2023, one client per US state", s1_data)
    clean = step(2, "Clean federated model", lambda: s2_clean(data))
    faulty = step(3, "Plant a fault (FedFault benchmark)", lambda: s3_fault(data, clean))
    step(4, "Privacy-budgeted audit tools", lambda: s4_tools(faulty))
    step(5, "Full audit, fact-check and score", lambda: s5_audit(faulty))
    step(6, "Saved Phase 4 results (125 audits per auditor)", s6_saved)


if __name__ == "__main__":
    main()
