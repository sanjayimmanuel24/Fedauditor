"""Phase 4: run auditors over the FedFault grid and summarise.

    python -m fedauditor.run_audit_grid --models lr --agents rules --workers 4
    python -m fedauditor.run_audit_grid --models lr --agents claude --llm claude-haiku-5-5 --max-cost 2
    python -m fedauditor.run_audit_grid --models lr --agents openai --llm Qwen/Qwen3-8B-AWQ \
        --base-url http://localhost:8000/v1 --workers 4
    python -m fedauditor.run_audit_grid --summary-only

One line per (scenario, agent) goes to
``results/fedauditor/audit_grid_<ds>_<model>_<label>.jsonl`` where the label is
``rules``, ``claude`` (or the Claude model id when it is not Opus 5.5) or, for
``openai``, ``--label`` / a slug of ``--llm``. Runs are resumable: scenario ids
already in the file are skipped. LLM audits also keep their full record (report
+ tool log) in ``results/fedauditor/audits/``. The summary (overall, per fault x
severity, per fault) covers every label on disk and goes to
``results/fedauditor/audit_grid_summary_<ds>.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache

import numpy as np

from fedauditor.faults import FAULTS, SEVERITY_LEVELS, Scenario, benchmark_grid
from fedauditor.report import aggregate, score
from fedauditor.run_audit import OUT_DIR as AUDIT_DIR, PRICE
from fedauditor.tools import DEFAULT_BUDGET

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES_DIR = os.path.join(BASE_DIR, "results", "fedauditor")
AGENTS = ("rules", "claude", "openai")
CLAUDE_DEFAULT = "claude-opus-5-5"
# USD per million tokens for Claude models other than Opus 5.5 (cache read 0.1x, write 1.25x input).
CLAUDE_PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}


def grid_path(dataset: str, model: str, label: str) -> str:
    return os.path.join(RES_DIR, f"audit_grid_{dataset}_{model}_{label}.jsonl")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.split("/")[-1].lower()).strip("-")


def agent_label(agent: str, cfg: dict) -> str:
    if agent == "rules":
        return "rules"
    if agent == "claude":
        if cfg.get("label"):
            return cfg["label"]
        return "claude" if cfg["llm"] == CLAUDE_DEFAULT else slug(cfg["llm"])
    return cfg.get("label") or slug(cfg["llm"])


def cost_usd(usage: dict, llm: str) -> float:
    if not usage:
        return 0.0
    if llm in CLAUDE_PRICES:
        i, o = CLAUDE_PRICES[llm]
        price = {"input": i, "output": o, "cache_read": 0.1 * i, "cache_write": 1.25 * i}
    else:
        price = PRICE
    return sum(usage.get(k, 0) * price[k] for k in price) / 1e6


@lru_cache(maxsize=2)
def _data(dataset: str, seed: int):
    from fedauditor.data import load_federated
    return load_federated(dataset, seed=seed)


def _make_agent(name: str, cfg: dict):
    if name == "rules":
        from fedauditor.baseline import RuleAuditor
        return RuleAuditor()
    if name == "claude":
        from fedauditor.agent import ClaudeAuditor
        return ClaudeAuditor(model=cfg["llm"], effort=cfg["effort"], verbose=False)
    from fedauditor.openai_agent import OpenAIAuditor
    return OpenAIAuditor(model=cfg["llm"], base_url=cfg["base_url"], thinking=cfg["thinking"],
                         max_tokens=cfg["max_tokens"], label=agent_label("openai", cfg),
                         transcript_dir=AUDIT_DIR, verbose=False)


def run_one(scenario: Scenario, agents, budget: float, cfg: dict) -> list:
    """Re-run one scenario with recording and audit it with each agent."""
    warnings.filterwarnings("ignore")
    from fedauditor.agent import audit
    from fedauditor.audit_env import AuditRun
    run = AuditRun.from_scenario(scenario, _data(scenario.dataset, scenario.seed))
    out = []
    for name in agents:
        label = agent_label(name, cfg)
        rec = audit(run, _make_agent(name, cfg), budget=budget, seed=scenario.seed)
        if name != "rules":
            os.makedirs(AUDIT_DIR, exist_ok=True)
            with open(os.path.join(AUDIT_DIR, f"{scenario.id}__{label}.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, default=str)
        fc = rec["fact_check"]
        out.append((label, {
            "id": scenario.id, "scenario": rec["scenario"], "truth": rec["truth"], "agent": name,
            "llm": cfg["llm"] if name != "rules" else None, "stop": rec["stop"], "turns": rec["turns"],
            "calls": len(rec["log"]), "epsilon": rec["budget"]["spent"], "seconds": rec["seconds"],
            "usage": rec["usage"], "report_errors": len(rec["report_errors"]),
            "cost_usd": round(cost_usd(rec["usage"], cfg["llm"]), 4) if name == "claude" else 0.0,
            "findings": [{k: f.get(k) for k in ("client", "fault_type", "onset_round", "confidence")}
                         for f in rec["report"]["findings"]],
            "fact_check": {"claims": fc["claims"], "supported": fc["supported"]},
            "score": rec["score"],
        }))
    return out


def _load(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _labels_on_disk(dataset: str, model: str) -> list:
    prefix = f"audit_grid_{dataset}_{model}_"
    labels = [os.path.basename(p)[len(prefix):-len(".jsonl")]
              for p in glob.glob(os.path.join(RES_DIR, prefix + "*.jsonl"))]
    return sorted((l for l in labels if "_v1" not in l), key=lambda l: (l != "rules", l))


def summarise(dataset: str) -> dict:
    summary = {}
    for model in ("lr", "mlp", "xgb"):
        for label in _labels_on_disk(dataset, model):
            recs = _load(grid_path(dataset, model, label))
            if not recs:
                continue
            scores = [score({"findings": r["findings"]}, r["truth"]) for r in recs]   # re-score: current scorer
            claims = sum(r["fact_check"]["claims"] for r in recs)
            sup = sum(r["fact_check"]["supported"] for r in recs)
            stops = defaultdict(int)
            for r in recs:
                stops[str(r.get("stop", "?")).split(":")[0]] += 1
            block = {"overall": {**aggregate(scores), "claims": claims,
                                 "claim_support": sup / claims if claims else None,
                                 "mean_calls": float(np.mean([r["calls"] for r in recs])),
                                 "mean_epsilon": float(np.mean([r["epsilon"] for r in recs])),
                                 "stops": dict(stops),
                                 "cost_usd": round(sum(r["cost_usd"] for r in recs), 2)}}
            by_cell, by_fault = defaultdict(list), defaultdict(list)
            for s in scores:
                if s["fault"] != "clean":
                    by_cell[f"{s['fault']}/{s['severity']}"].append(s)
                    by_fault[s["fault"]].append(s)
            block["by_fault"] = {k: aggregate(v) for k, v in sorted(by_fault.items())}
            block["by_cell"] = {k: aggregate(v) for k, v in sorted(by_cell.items())}
            summary[f"{model}/{label}"] = block
    path = os.path.join(RES_DIR, f"audit_grid_summary_{dataset}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def _fmt(x):
    return "  -  " if x is None else f"{x:5.2f}"


def print_summary(summary: dict) -> None:
    w = max([14] + [len(k) for k in summary])
    print(f"\n{'model/agent':{w}s} {'n':>4s} {'prec':>5s} {'rec':>5s} {'type':>5s} {'onset':>5s} "
          f"{'FAR':>5s} {'facts':>5s} {'cost':>7s}  stops")
    for key, b in summary.items():
        o = b["overall"]
        print(f"{key:{w}s} {o['n_faulty'] + o['n_clean']:4d} {_fmt(o['precision'])} {_fmt(o['recall'])} "
              f"{_fmt(o['scenario_type_accuracy'])} {_fmt(o['onset_abs_error'])} "
              f"{_fmt(o['false_alarm_rate'])} {_fmt(o['claim_support'])} ${o['cost_usd']:6.2f}  "
              f"{o.get('stops', {})}")
    for key, b in summary.items():
        print(f"\nRecall by fault x severity ({key}):")
        for f in FAULTS:
            cells = [b["by_cell"].get(f"{f}/{s}", {}).get("recall") for s in SEVERITY_LEVELS]
            typ = b["by_fault"].get(f, {}).get("scenario_type_accuracy")
            print(f"  {f:24s} " + " ".join(_fmt(c) for c in cells) + f"   type {_fmt(typ)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="brfss")
    ap.add_argument("--models", nargs="+", default=["lr"], choices=["lr", "mlp", "xgb"])
    ap.add_argument("--agents", nargs="+", default=["rules"], choices=list(AGENTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    ap.add_argument("--faults", nargs="+", default=list(FAULTS), choices=list(FAULTS))
    ap.add_argument("--severities", nargs="+", default=list(SEVERITY_LEVELS), choices=list(SEVERITY_LEVELS))
    ap.add_argument("--no-clean", action="store_true", help="skip the clean runs")
    ap.add_argument("--limit", type=int, default=None, help="audit at most this many scenarios (pilot runs)")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET)
    ap.add_argument("--llm", default=None, help=f"model id (claude default {CLAUDE_DEFAULT}; required for openai)")
    ap.add_argument("--effort", default="high", help="Claude effort")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible server (openai agent)")
    ap.add_argument("--label", default=None, help="results label for an LLM agent (default: slug of --llm; use e.g. <model>-pilot for pilots)")
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3 thinking mode (slower)")
    ap.add_argument("--max-tokens", type=int, default=4096, help="max output tokens per turn (openai agent)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-cost", type=float, default=None,
                    help="stop submitting Claude audits once this many USD are spent in this invocation")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    if "openai" in args.agents and not args.llm:
        ap.error("--agents openai needs --llm (the model name the server serves)")
    if "claude" in args.agents and "openai" in args.agents:
        ap.error("run claude and openai agents in separate invocations (they need different --llm)")
    cfg = {"llm": args.llm or CLAUDE_DEFAULT, "effort": args.effort, "base_url": args.base_url,
           "label": args.label, "thinking": args.thinking, "max_tokens": args.max_tokens}

    if not args.summary_only:
        jobs = []
        for model in args.models:
            grid = benchmark_grid(args.dataset, model, args.seeds, args.rounds, faults=args.faults,
                                  severities=args.severities)
            if args.no_clean:
                grid = [s for s in grid if s.fault != "clean"]
            done = {a: {r["id"] for r in _load(grid_path(args.dataset, model, agent_label(a, cfg)))}
                    for a in args.agents}
            for sc in grid:
                todo = [a for a in args.agents if sc.id not in done[a]]
                if todo:
                    jobs.append((sc, todo))
        if args.limit is not None:
            jobs = jobs[:args.limit]
        print(f"{len(jobs)} scenarios to audit with {args.agents} ({args.workers} workers)", flush=True)
        spent, t0, n = 0.0, time.time(), 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending = {}
            queue = list(jobs)

            def submit():
                while queue and len(pending) < args.workers:
                    if args.max_cost is not None and spent >= args.max_cost:
                        queue.clear()
                        print(f"Cost cap ${args.max_cost} reached; stopping.", flush=True)
                        return
                    sc, todo = queue.pop(0)
                    pending[pool.submit(run_one, sc, todo, args.budget, cfg)] = sc

            submit()
            while pending:
                fut = next(as_completed(pending))
                sc = pending.pop(fut)
                n += 1
                try:
                    results = fut.result()
                except Exception as e:  # keep the grid going; the scenario stays undone
                    print(f"[{n}/{len(jobs)}] {sc.id} FAILED: {type(e).__name__}: {e}", flush=True)
                    submit()
                    continue
                for label, line in results:
                    with open(grid_path(sc.dataset, sc.model, label), "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(line, default=str) + "\n")
                    spent += line["cost_usd"]
                    s = line["score"]
                    print(f"[{n}/{len(jobs)}] {sc.model} {label:10s} {sc.fault:24s} {sc.severity:6s} "
                          f"s{sc.seed} flagged={s['flagged']} truth={s['true_clients']} "
                          f"facts {line['fact_check']['supported']}/{line['fact_check']['claims']} "
                          f"{line['stop']} {line['seconds']:.0f}s ${line['cost_usd']:.3f}", flush=True)
                submit()
        print(f"Done in {time.time() - t0:.0f}s, Claude spend this run ${spent:.2f}", flush=True)
    print_summary(summarise(args.dataset))


if __name__ == "__main__":
    main()
