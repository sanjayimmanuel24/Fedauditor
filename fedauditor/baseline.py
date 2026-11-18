"""Rule-based baseline auditor for FedFault.

It uses exactly the agent's tools, the agent's privacy budget and the agent's
report format, but no LLM: a fixed query plan followed by fixed thresholds on
robust z-scores across clients. Every finding cites the tool rows it used, so
its reports go through the same fact-checker and scorer as the agent's.

Query plan (epsilon, default budget 20):
    federation_overview (free), client_performance (3), client_updates (4),
    drift_by_round (4), feature_summary (3), subgroup_outcomes (3, split over
    the sensitive attributes), then client_updates per suspected free-rider
    (1 each, at most 3) to date the onset.

Rules per client; if several match, the label follows ``PRIORITY`` (data-level
faults before poisoning). Thresholds were set on dev seed
10 (the evaluation grid uses seeds 20-24). ``zn`` is a robust z across clients
whose scale also includes the client's own Laplace noise SD (the mechanism is
public), so tiny clients are not flagged for noise alone.
    label_noise      local AUC zn < -Z
    prevalence_shift CITL zn < -Z and prevalence above the median
    free_rider       min_norm_ratio < FREE_RIDE_RATIO (onset dated per client)
    poisoning        mean_cos < 0
    explanation_inst. min_norm_ratio z > Z (large update every round, not opposed)
    drift            drift onset >= 2 and max_shift_sd > 0.3 + 3 noise SD
    missing_artefact p01/p99 outlier (noise-adjusted z) of a non-binary feature,
                     outside the federation's median [p01, p99] widened by half its width
    subgroup_bias    O/E or outcome-rate noise-adjusted z < -Z in exactly one subgroup
v2 (2026-11-15): v1 applied the noise correction only to performance and drift;
on eval seeds its clean false-alarm rate was 0.6 (LR) / 0.8 (MLP), all from small
states hit by the sentinel and subgroup rules, so v2 corrects those two rules too.
v1 results are kept as ``audit_grid_*_rules_v1.jsonl``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from fedauditor.tools import DRIFT_CLIP_SD, AuditSession, _robust_z

Z = 4.0
FREE_RIDE_RATIO = 0.05
DRIFT_MIN = 0.3            # drift_by_round's own onset threshold (SD units)
PLAN = {"client_updates": 4.0, "client_performance": 3.0, "feature_summary": 3.0,
        "drift_by_round": 4.0, "subgroup_outcomes": 3.0, "free_rider_check": 1.0}
SKIP = {"(median)", "_pooled"}
SQ2 = float(np.sqrt(2.0))
# When a client matches several rules, a data-level explanation beats "malicious":
# label noise, sentinel codes or subgroup bias also make a client's update oppose the others.
PRIORITY = ["label_noise", "prevalence_shift", "free_rider", "drift", "missing_artefact",
            "subgroup_bias", "poisoning", "explanation_instability"]


def rows(table: Optional[dict]) -> List[dict]:
    if not table:
        return []
    return [dict(zip(table["columns"], r)) for r in table["rows"]]


def _z(rs: List[dict], key: str) -> Dict[str, float]:
    vals = np.array([np.nan if r[key] is None else r[key] for r in rs], dtype=float)
    return dict(zip([r["client"] for r in rs], _robust_z(vals)))


def _zn(rs: List[dict], key: str, noise_sd: Dict[str, float]) -> Dict[str, float]:
    """Robust z whose scale adds each client's own noise SD in quadrature."""
    vals = np.array([np.nan if r[key] is None else r[key] for r in rs], dtype=float)
    ok = np.isfinite(vals)
    if ok.sum() < 3:
        return {r["client"]: 0.0 for r in rs}
    med = np.median(vals[ok])
    scale = max(1.4826 * np.median(np.abs(vals[ok] - med)), 1e-3 + 0.02 * abs(med))
    return {r["client"]: (0.0 if not np.isfinite(v) else
                          float((v - med) / np.hypot(scale, noise_sd.get(r["client"], 0.0))))
            for r, v in zip(rs, vals)}


def _adj_z(row: dict, median_row: dict, noise_sd: float) -> float:
    """Re-scale a screened row's robust z to include the client's own noise SD.
    The tool's z = (v - med) / scale, so scale = |v - med| / |z|."""
    stat = row["outlier_stat"]
    v, med, z = row[stat], median_row.get(stat), row["z"]
    if v is None or med is None or not z:
        return 0.0
    scale = abs(v - med) / abs(z)
    return float((v - med) / np.hypot(scale, noise_sd))


def _claim(res: dict, row: dict, metric: str, **keys) -> dict:
    c = {"call_id": res["call_id"], "metric": metric, "value": row[metric], "client": row["client"]}
    c.update({k: v for k, v in keys.items() if v is not None})
    return c


class RuleAuditor:
    """``agent.run(session)``-compatible rule baseline (see module docstring)."""

    def __init__(self, z: float = Z, free_ride_ratio: float = FREE_RIDE_RATIO, plan: dict = None):
        self.z, self.free_ride_ratio = z, free_ride_ratio
        self.plan = {**PLAN, **(plan or {})}

    def run(self, session: AuditSession):
        from fedauditor.agent import AgentOutcome
        return AgentOutcome(self.policy(session), "rules", 0)

    def policy(self, session: AuditSession) -> dict:
        Zt, plan = self.z, self.plan
        ov = session.call("federation_overview")
        n_train = {r["client"]: r["n_train"] for r in rows(ov["clients"])}
        R = ov["rounds"]
        evidence: Dict[str, list] = {}   # client -> [(fault, onset, claims, why)] in priority order

        def add(client, fault, claims, why, onset=1):
            if client not in SKIP:
                evidence.setdefault(client, []).append((fault, onset, claims, why))

        # local performance: label noise, prevalence shift (Laplace b = c/n * m/eps, m = 5)
        eps = plan["client_performance"]
        cp = session.call("client_performance", {"epsilon": eps})
        cr = [r for r in rows(cp.get("clients")) if r["client"] not in SKIP
              and None not in (r["auc"], r["citl"], r["prevalence"])]
        b = {r["client"]: 5 / (eps * max(r["n"], 1)) for r in cr}
        b_auc = {r["client"]: 5 / (eps * max(1.0, min(r["prevalence"], 1 - r["prevalence"]) * r["n"]))
                 for r in cr}
        za = _zn(cr, "auc", {c: SQ2 * v for c, v in b_auc.items()})
        zt = _zn(cr, "citl", {c: 2 * v for c, v in b.items()})
        med_prev = float(np.median([r["prevalence"] for r in cr])) if cr else 0.0
        for r in cr:
            c = r["client"]
            if za[c] < -Zt:
                add(c, "label_noise", [_claim(cp, r, "auc")], "low local AUC")
            elif zt[c] < -Zt and r["prevalence"] > med_prev:
                add(c, "prevalence_shift", [_claim(cp, r, "citl"), _claim(cp, r, "prevalence")],
                    "over-predicted risk with high observed prevalence")

        # updates: poisoning, free-riding, noisy uploads
        up = session.call("client_updates", {"epsilon": plan["client_updates"]})
        ur = [r for r in rows(up.get("clients"))
              if None not in (r["mean_cos"], r["median_norm_ratio"], r["min_norm_ratio"])]
        zm = _z(ur, "min_norm_ratio")
        for r in ur:
            c = r["client"]
            if r["min_norm_ratio"] < self.free_ride_ratio:
                add(c, "free_rider", [_claim(up, r, "min_norm_ratio")], "near-zero update in some round")
            elif r["mean_cos"] < 0:
                add(c, "poisoning", [_claim(up, r, "mean_cos")], "update points against the median update")
            elif zm[c] > Zt:
                add(c, "explanation_instability", [_claim(up, r, "min_norm_ratio")],
                    "update norm large in every round without opposing the others (noisy uploads)")

        # drift over rounds (shift_sd: c = 4 * clip, m = 3R)
        eps = plan["drift_by_round"]
        dr = session.call("drift_by_round", {"epsilon": eps})
        for r in rows(dr.get("clients")):
            c, on, shift = r["client"], r["onset_round"], r["max_shift_sd"]
            sd = SQ2 * 4 * DRIFT_CLIP_SD * 3 * R / (eps * max(n_train.get(c, 1), 1))
            if on is not None and on >= 2 and shift is not None and shift > DRIFT_MIN + 3 * sd:
                add(c, "drift", [_claim(dr, r, "max_shift_sd"), _claim(dr, r, "onset_round")],
                    f"feature shift up to {shift:.2f} SD from round {on}", onset=int(on))

        # sentinel codes: a tail percentile far outside the federation's usual range
        eps = plan["feature_summary"]
        fs = session.call("feature_summary", {"epsilon": eps})
        med = {r["feature"]: r for r in rows(fs.get("medians"))}
        n_feat = len(ov.get("features") or med) or 1
        for r in rows(fs.get("outliers")):
            m = med.get(r["feature"])
            if r["outlier_stat"] not in ("p01", "p99") or not m or None in (m["p01"], m["p99"], m["sd"]):
                continue
            width = m["p99"] - m["p01"]
            v = r[r["outlier_stat"]]
            # tool noise: c = clip range (~16 federation SDs), m = 4 stats per feature
            sd = SQ2 * 16 * m["sd"] * 4 * n_feat / (eps * max(n_train.get(r["client"], 1), 1))
            if width > 2 and v is not None and abs(_adj_z(r, m, sd)) > Zt and (
                    v < m["p01"] - width / 2 or v > m["p99"] + width / 2):
                add(r["client"], "missing_artefact",
                    [_claim(fs, r, r["outlier_stat"], feature=r["feature"])],
                    f"{r['feature']} {r['outlier_stat']} = {v:.1f}, outside the usual "
                    f"[{m['p01']:.1f}, {m['p99']:.1f}] (sentinel code?)")

        # subgroup under-diagnosis
        attrs = list(ov.get("sensitive_attributes") or {})
        for a in attrs:
            eps = plan["subgroup_outcomes"] / len(attrs)
            so = session.call("subgroup_outcomes", {"attribute": a, "epsilon": eps})
            meds = {r["group"]: r for r in rows(so.get("medians"))}
            n_groups = max(len(meds), 1)
            low: Dict[str, list] = {}
            for r in rows(so.get("outliers")):
                st, m = r["outlier_stat"], meds.get(r["group"])
                if st not in ("oe_ratio", "outcome_rate") or r["z"] >= 0 or not m:
                    continue
                # tool noise on the rate: c = 1, m = 4 stats per group; O/E divides by mean_pred
                sd = SQ2 * 4 * n_groups / (eps * max(r["n"] or 1, 1))
                if st == "oe_ratio":
                    sd /= max(r["mean_pred"] or 0.0, 1e-3)
                if _adj_z(r, m, sd) < -Zt:
                    low.setdefault(r["client"], []).append(r)
            for c, rs in low.items():
                if len({r["group"] for r in rs}) == 1:
                    r = rs[0]
                    add(c, "subgroup_bias", [_claim(so, r, r["outlier_stat"], group=r["group"])],
                        f"low {r['outlier_stat']} only in {a}={r['group']}")

        # date free-riding onsets
        riders = [c for c, ev in evidence.items() if any(e[0] == "free_rider" for e in ev)][:3]
        for c in riders:
            pr = session.call("client_updates", {"client": c, "epsilon": plan["free_rider_check"]})
            low = [r for r in rows(pr.get("per_round"))
                   if r["norm_ratio"] is not None and r["norm_ratio"] < self.free_ride_ratio]
            if low:
                i = next(i for i, e in enumerate(evidence[c]) if e[0] == "free_rider")
                fault, _, claims, why = evidence[c][i]
                evidence[c][i] = (fault, int(low[0]["round"]),
                                  claims + [_claim(pr, low[0], "norm_ratio", round=int(low[0]["round"]))], why)

        findings = []
        for c, ev in sorted(evidence.items()):
            ev.sort(key=lambda e: PRIORITY.index(e[0]))
            fault, onset, claims, why = ev[0]
            others = sorted({e[0] for e in ev[1:]} - {fault})
            findings.append({"client": c, "fault_type": fault, "onset_round": onset, "confidence": 0.7,
                             "rationale": why + (f"; also matched {others}" if others else ""),
                             "claims": claims})
        summary = (f"Rule baseline: {len(findings)} flagged client(s)." if findings
                   else "Rule baseline: no client exceeded the thresholds.")
        return {"findings": findings, "summary": summary}
