"""Audit report: schema, validation, fact-checking and scoring against FedFault truth.

A report names the faulty clients it found, each with a fault type, an onset
round and evidence claims. A claim cites one logged tool call and one number in
its result::

    {"call_id": "c4", "metric": "mean_cos", "value": -0.98, "client": "TX"}

``fact_check`` accepts a claim only if the cited call succeeded and one row of
one of its tables has the claimed client / round / feature / group (when the
claim gives them and the table has that column) and a ``metric`` value within
tolerance. ``score`` compares the flagged clients and fault types with
``Planted.truth``.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np

from fedauditor.faults import FAULTS

KEYS = ("client", "round", "feature", "group")
ABS_TOL = 0.005
REL_TOL = 0.02

CLAIM_SCHEMA = {
    "type": "object",
    "properties": {
        "call_id": {"type": "string", "description": "Id of the tool call whose result holds the number, e.g. 'c3'."},
        "metric": {"type": "string", "description": "Column name in that result, e.g. 'mean_cos'."},
        "value": {"type": "number", "description": "The number exactly as the tool returned it."},
        "client": {"type": "string"}, "round": {"type": "integer"},
        "feature": {"type": "string"}, "group": {"type": "string"},
    },
    "required": ["call_id", "metric", "value"],
    "additionalProperties": False,
}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "description": "One entry per faulty client. Empty if the federation looks clean.",
                     "items": {"type": "object", "properties": {
                         "client": {"type": "string"},
                         "fault_type": {"type": "string", "enum": list(FAULTS)},
                         "onset_round": {"type": "integer", "description": "First faulty round; 1 if from the start."},
                         "confidence": {"type": "number", "description": "0..1"},
                         "rationale": {"type": "string"},
                         "claims": {"type": "array", "items": CLAIM_SCHEMA},
                     }, "required": ["client", "fault_type", "onset_round", "confidence", "rationale", "claims"],
                         "additionalProperties": False}},
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
    "additionalProperties": False,
}


def validate_report(report: dict, clients: Iterable[str], rounds: int) -> List[str]:
    """Problems that make a report unusable (empty list = valid)."""
    errs = []
    clients = set(clients)
    if not isinstance(report, dict) or not isinstance(report.get("findings"), list):
        return ["report must be an object with a 'findings' list"]
    for i, f in enumerate(report["findings"]):
        where = f"findings[{i}]"
        if not isinstance(f, dict):
            errs.append(f"{where}: must be an object")
            continue
        if f.get("client") not in clients:
            errs.append(f"{where}: unknown client {f.get('client')!r}")
        if f.get("fault_type") not in FAULTS:
            errs.append(f"{where}: fault_type must be one of {list(FAULTS)}")
        onset = f.get("onset_round")
        if not isinstance(onset, int) or not 1 <= onset <= rounds:
            errs.append(f"{where}: onset_round must be an integer in 1..{rounds}")
        claims = f.get("claims")
        if not isinstance(claims, list) or not claims:
            errs.append(f"{where}: needs at least one claim")
            continue
        for j, c in enumerate(claims):
            if not isinstance(c, dict) or not {"call_id", "metric", "value"} <= set(c):
                errs.append(f"{where}.claims[{j}]: needs call_id, metric and value")
            elif not isinstance(c["value"], (int, float)):
                errs.append(f"{where}.claims[{j}]: value must be a number")
    return errs


# ------------------------------------------------------------------ fact-check

def _tables(result: dict):
    for key, value in result.items():
        if isinstance(value, dict) and "columns" in value and "rows" in value:
            yield key, value


def _same(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    return str(a) == str(b)


def parse_flags(text) -> Dict[str, float]:
    """{'mean_cos': -8.2, ...} from a tool `flags` cell like 'mean_cos z=-8.2; min_cos z=-5.7'."""
    out = {}
    for part in str(text or "").split(";"):
        bits = part.strip().split(" z=")
        if len(bits) == 2:
            try:
                out[bits[0].strip()] = float(bits[1])
            except ValueError:
                pass
    return out


def check_claim(claim: dict, log_by_id: Dict[str, dict]) -> dict:
    entry = log_by_id.get(claim.get("call_id"))
    if entry is None:
        return {"supported": False, "reason": "no such call id"}
    if not entry.get("ok"):
        return {"supported": False, "reason": "cited call failed"}
    metric, value = claim["metric"], float(claim["value"])
    nearest = None
    for _, table in _tables(entry["result"]):
        cols = table["columns"]
        if metric in cols:
            mi = cols.index(metric)
            get = lambda row: row[mi]
        elif metric.endswith("_z") and "flags" in cols:     # a cited flag, e.g. mean_cos_z = -8.2
            fi = cols.index("flags")
            get = lambda row: parse_flags(row[fi]).get(metric[:-2])
        else:
            continue
        for row in table["rows"]:
            if any(k in claim and k in cols and not _same(row[cols.index(k)], claim[k]) for k in KEYS):
                continue
            logged = get(row)
            if not isinstance(logged, (int, float)):
                continue
            err = abs(value - logged)
            if err <= ABS_TOL + REL_TOL * abs(logged):
                return {"supported": True, "logged": logged}
            nearest = logged if nearest is None or err < abs(value - nearest) else nearest
    if nearest is None:
        return {"supported": False, "reason": f"no row with metric {metric!r} for the claimed keys"}
    return {"supported": False, "reason": f"value mismatch (closest logged {nearest})"}


def fact_check(report: dict, log: List[dict]) -> dict:
    log_by_id = {e["id"]: e for e in log}
    checked, unsupported = 0, []
    for f in report.get("findings", []):
        for c in f.get("claims", []):
            checked += 1
            res = check_claim(c, log_by_id)
            if not res["supported"]:
                unsupported.append({"client": f.get("client"), "claim": c, "reason": res["reason"]})
    return {"claims": checked, "supported": checked - len(unsupported),
            "unsupported": unsupported,
            "support_rate": None if checked == 0 else (checked - len(unsupported)) / checked}


# --------------------------------------------------------------------- scoring

def score(report: dict, truth: dict) -> dict:
    """Client-level detection, fault-type accuracy and onset error for one scenario."""
    findings = report.get("findings", [])
    flagged = {f["client"] for f in findings}
    true = set(truth.get("clients") or [])
    tp = flagged & true
    is_clean = truth.get("fault") == "clean"
    out = {
        "fault": truth.get("fault"), "severity": truth.get("severity"),
        "flagged": sorted(flagged), "true_clients": sorted(true),
        "tp": len(tp), "fp": len(flagged - true), "fn": len(true - flagged),
        "precision": len(tp) / len(flagged) if flagged else (None if true else 1.0),
        "recall": len(tp) / len(true) if true else None,
        "false_alarm": bool(flagged) if is_clean else None,
    }
    hits = [f for f in findings if f["client"] in true]
    out["type_accuracy"] = (np.mean([f["fault_type"] == truth["fault"] for f in hits]).item()
                            if hits else None)
    onset = truth.get("onset_round")
    errs = [abs(f["onset_round"] - onset) for f in hits if onset is not None and f.get("onset_round")]
    out["onset_abs_error"] = float(np.mean(errs)) if errs else None
    # Most common reported type; a tie between k types gives credit 1/k if the true type is
    # among them (deterministic, unlike max(set(...)) whose tie-break follows string hashing).
    types = [f["fault_type"] for f in findings]
    if is_clean:
        out["scenario_type_correct"] = None
    elif not types:
        out["scenario_type_correct"] = 0.0
    else:
        top = max(types.count(t) for t in set(types))
        modes = {t for t in set(types) if types.count(t) == top}
        out["scenario_type_correct"] = (1.0 / len(modes)) if truth["fault"] in modes else 0.0
    return out


def aggregate(scores: List[dict]) -> dict:
    """Micro-averaged precision/recall over faulty scenarios, false-alarm rate over clean ones."""
    faulty = [s for s in scores if s["fault"] != "clean"]
    clean = [s for s in scores if s["fault"] == "clean"]
    tp = sum(s["tp"] for s in faulty)
    fp = sum(s["fp"] for s in faulty)
    fn = sum(s["fn"] for s in faulty)

    def mean(key, rows):
        vals = [s[key] for s in rows if s.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "n_faulty": len(faulty), "n_clean": len(clean),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "type_accuracy": mean("type_accuracy", faulty),
        "scenario_type_accuracy": mean("scenario_type_correct", faulty),
        "onset_abs_error": mean("onset_abs_error", faulty),
        "false_alarm_rate": mean("false_alarm", clean),
    }
