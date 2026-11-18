"""Turn a saved audit (``results/fedauditor/audits/*.json``) into a view for the dashboard.

Each claim is resolved to the exact logged table row it cites, using the same matching rules
as the fact-checker (``report.check_claim``), so the page can link a claim to that row.
"""
from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Tuple

from fedauditor.report import ABS_TOL, KEYS, REL_TOL, _same, _tables, check_claim, parse_flags
from fedauditor.run_audit_grid import RES_DIR

AUDIT_DIR = os.path.join(RES_DIR, "audits")
DEFAULT_AUDIT = "brfss-lr-fedavg-r20-subgroup_bias-high-s20__claude-haiku-5-5-p5"
_NAME = re.compile(r"^[A-Za-z0-9_.\-]+$")


def list_audits() -> List[str]:
    """Audit names (file stems), newest label grouping kept simple: sorted by name."""
    if not os.path.isdir(AUDIT_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(AUDIT_DIR) if f.endswith(".json"))


def locate_claim(claim: dict, entry: dict) -> Optional[Tuple[str, int]]:
    """(table name, row index) of the logged row that supports ``claim``, or None."""
    if not entry or not entry.get("ok"):
        return None
    metric = claim.get("metric")
    try:
        value = float(claim.get("value"))
    except (TypeError, ValueError):
        return None
    for tname, table in _tables(entry["result"]):
        cols = table["columns"]
        if metric in cols:
            mi = cols.index(metric)
            get = lambda row: row[mi]
        elif str(metric).endswith("_z") and "flags" in cols:
            fi = cols.index("flags")
            get = lambda row: parse_flags(row[fi]).get(metric[:-2])
        else:
            continue
        for i, row in enumerate(table["rows"]):
            if any(k in claim and k in cols and not _same(row[cols.index(k)], claim[k]) for k in KEYS):
                continue
            logged = get(row)
            if isinstance(logged, (int, float)) and abs(value - logged) <= ABS_TOL + REL_TOL * abs(logged):
                return tname, i
    return None


def anchor(call_id: str, table: Optional[str] = None, row: Optional[int] = None) -> str:
    return f"call-{call_id}" if table is None else f"call-{call_id}-{table}-{row}"


def load_view(name: str) -> Dict:
    """View model for one audit. ``name`` must be one of ``list_audits()``."""
    if not _NAME.match(name or "") or name not in list_audits():
        raise KeyError(name)
    with open(os.path.join(AUDIT_DIR, name + ".json"), encoding="utf-8") as fh:
        a = json.load(fh)
    log_by_id = {e["id"]: e for e in a.get("log", [])}
    cited: Dict[str, set] = {}          # call id -> {(table, row)}
    findings = []
    for f in a.get("report", {}).get("findings", []):
        claims = []
        for c in f.get("claims", []):
            chk = check_claim(c, log_by_id)
            loc = locate_claim(c, log_by_id.get(c.get("call_id")))
            if loc:
                cited.setdefault(c["call_id"], set()).add(loc)
            entry = log_by_id.get(c.get("call_id"))
            claims.append({
                "call_id": c.get("call_id"), "tool": entry["tool"] if entry else None,
                "keys": {k: c[k] for k in KEYS if k in c}, "metric": c.get("metric"),
                "value": c.get("value"), "supported": chk["supported"],
                "logged": chk.get("logged"), "reason": chk.get("reason"),
                "href": "#" + (anchor(c["call_id"], *loc) if loc else anchor(c.get("call_id", ""))),
            })
        findings.append({**{k: f.get(k) for k in ("client", "fault_type", "onset_round", "confidence",
                                                  "rationale")}, "claims": claims})
    calls = []
    for e in a.get("log", []):
        res = e.get("result") or {}
        tables = [{"name": tn, "columns": t["columns"],
                   "rows": [{"cells": r, "anchor": anchor(e["id"], tn, i),
                             "cited": (tn, i) in cited.get(e["id"], set())}
                            for i, r in enumerate(t["rows"])]}
                  for tn, t in _tables(res)]
        scalars = {k: v for k, v in res.items()
                   if k not in {t["name"] for t in tables} and not isinstance(v, (dict, list))
                   and k not in ("call_id", "tool")}
        calls.append({"id": e["id"], "tool": e.get("tool"), "args": e.get("args") or {},
                      "ok": e.get("ok"), "anchor": anchor(e["id"]), "tables": tables,
                      "scalars": scalars, "n_cited": len(cited.get(e["id"], ())),
                      "error": None if e.get("ok") else res.get("error", res)})
    return {"name": name, "scenario": a.get("scenario", {}), "truth": a.get("truth", {}),
            "score": a.get("score", {}), "fact_check": a.get("fact_check", {}),
            "budget": a.get("budget", {}), "agent": a.get("agent"), "turns": a.get("turns"),
            "summary": a.get("report", {}).get("summary", ""), "findings": findings, "calls": calls}
