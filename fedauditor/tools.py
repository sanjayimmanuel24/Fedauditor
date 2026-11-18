"""Audit tools: noisy, budgeted aggregate queries over a federated run.

The auditor never sees patient rows. Every tool computes statistics at the
clients (reusing the FedHealth modules: ``federated_metrics``,
``federated_fairness``, ``federated_explain``, ``federated_scaler``) and releases
only per-client or pooled aggregates with Laplace noise. Each call spends
``epsilon`` from a per-audit ``PrivacyAccountant``; once the budget is spent,
calls are refused.

Noise model (basic composition, parallel across clients). A call that releases
``m`` statistics per client splits its ``epsilon`` evenly over them, and each
statistic gets Laplace noise of scale ``c / n * m / epsilon``, where ``n`` is the
number of records it is computed over and ``c`` the statistic's range after
clipping (features are clipped to +-8 SD of the federation mean, +-4 SD in the
drift tool). For means and rates this is the exact sensitivity; for AUC, ECE,
quantiles, correlations and update statistics ``c / n`` is a per-record
influence heuristic, not a worst-case bound. The accounting is therefore a
calibrated budget, not a formal DP proof; a formal mechanism is Phase 4 work.

Every result is a dict of compact tables (``columns`` + ``rows``). Every call,
result and refusal is logged with an id (``c1``, ``c2``...) so the fact-checker
can verify each claim in the final report.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np

from federated_explain import _spearman, linear_phi
from federated_fairness import group_counts_over_grid
from federated_metrics import full_metrics, group_metrics
from federated_scaler import local_sufficient_statistics
from fedauditor.audit_env import AuditRun
from fedauditor.models import NumpyModel

DEFAULT_BUDGET = 20.0
DEFAULT_EPSILON = 1.0
SCREEN_Z = 3.5
FEATURE_CLIP_SD = 8.0
DRIFT_CLIP_SD = 4.0
UPDATE_C = 5.0           # per-record influence constant for update statistics
ATTR_ROWS = 300          # rows per client sampled for attribution statistics
PHI_CLIP = 4.0


class BudgetExhausted(Exception):
    pass


class PrivacyAccountant:
    def __init__(self, total: float = DEFAULT_BUDGET):
        self.total = float(total)
        self.spent = 0.0
        self.ledger: List[dict] = []

    @property
    def remaining(self) -> float:
        return round(self.total - self.spent, 6)

    def charge(self, epsilon: float, call_id: str, tool: str) -> None:
        if not epsilon > 0:
            raise ValueError("epsilon must be positive")
        if self.spent + epsilon > self.total + 1e-9:
            raise BudgetExhausted(f"privacy budget exhausted: {self.remaining} left, "
                                  f"{epsilon} requested")
        self.spent += epsilon
        self.ledger.append({"call_id": call_id, "tool": tool, "epsilon": epsilon})

    def refund(self, call_id: str) -> None:
        for i, e in enumerate(self.ledger):
            if e["call_id"] == call_id:
                self.spent -= e["epsilon"]
                del self.ledger[i]
                return


def _r(x, digits: int = 4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    return float(f"{float(x):.{digits}g}")


def _table(columns: List[str], rows: List[list]) -> dict:
    return {"columns": columns, "rows": [[_r(v) if isinstance(v, (float, np.floating)) else v
                                          for v in row] for row in rows]}


class _Noise:
    def __init__(self, rng: np.random.Generator, epsilon: float, m: int):
        self.rng, self.factor = rng, max(1, m) / epsilon

    def __call__(self, value: float, c: float, n: float, lo=None, hi=None):
        if value is None or not np.isfinite(value):
            return None
        out = float(value + self.rng.laplace(0.0, c / max(n, 1) * self.factor))
        if lo is not None:
            out = max(lo, out)
        if hi is not None:
            out = min(hi, out)
        return out


def _robust_z(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    ok = np.isfinite(v)
    z = np.zeros_like(v)
    if ok.sum() < 3:
        return z
    med = np.median(v[ok])
    mad = 1.4826 * np.median(np.abs(v[ok] - med))
    scale = max(mad, 1e-3 + 0.02 * abs(med))
    z[ok] = (v[ok] - med) / scale
    return z


SQ2 = float(np.sqrt(2.0))
FLAG_Z = 3.0             # noise-adjusted |z| above which a stat is listed in `flags`
SCREEN_Z_NOISE = 3.0     # screened outlier rows with noise-adjusted |z| at or below this are dropped


def laplace_sd(c: float, n: float, m: int, epsilon: float) -> float:
    """SD of the Laplace noise _Noise adds to a statistic with influence c over n records,
    when the call's epsilon is split over m statistics (public mechanism; no privacy cost)."""
    return SQ2 * c * max(1, m) / (max(n, 1) * epsilon)


def _noise_z(values, sds) -> np.ndarray:
    """Deviation of each client from the other clients, in units that combine the spread
    across clients with that client's own noise SD. Post-processing of released values."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    s = np.nan_to_num(np.array([np.nan if x is None else x for x in sds], dtype=float))
    out = np.zeros_like(v)
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return out
    med = np.median(v[ok])
    scale = max(1.4826 * np.median(np.abs(v[ok] - med)), 1e-3 + 0.02 * abs(med))
    out[ok] = (v[ok] - med) / np.hypot(scale, s[ok])
    out[~ok] = np.nan
    return out


def _flags(zs_by_stat: Dict[str, np.ndarray], i: int) -> str:
    """'stat z=+5.1; stat z=-3.2' for the stats of row i whose noise-adjusted |z| exceeds FLAG_Z."""
    items = [(s, float(z[i])) for s, z in zs_by_stat.items() if np.isfinite(z[i]) and abs(z[i]) > FLAG_Z]
    items.sort(key=lambda t: -abs(t[1]))
    return "; ".join(f"{s} z={z:+.1f}" for s, z in items)


def _screen(rows: List[dict], stats: List[str], by: Optional[str], sd_of=None) -> tuple:
    """Keep rows that are robust outliers (|z| > SCREEN_Z) on any stat within their
    ``by`` group; also return the per-group median rows. With ``sd_of(row, stat)`` (the
    row's noise SD), kept rows also get ``z_noise``: the noise-adjusted z of their
    outlier stat (selection itself is unchanged)."""
    groups: Dict[object, List[dict]] = {}
    for row in rows:
        groups.setdefault(row.get(by) if by else None, []).append(row)
    kept, medians = [], []
    for key, members in groups.items():
        zs = {s: _robust_z(np.array([np.nan if r[s] is None else r[s] for r in members]))
              for s in stats}
        zn = ({s: _noise_z([r[s] for r in members], [sd_of(r, s) for r in members]) for s in stats}
              if sd_of else None)
        med = {"client": "(median)", **({by: key} if by else {})}
        for s in stats:
            vals = [r[s] for r in members if r[s] is not None]
            med[s] = float(np.median(vals)) if vals else None
        medians.append(med)
        for i, r in enumerate(members):
            best = max(stats, key=lambda s: abs(zs[s][i]))
            if abs(zs[best][i]) > SCREEN_Z:
                row = {**r, "outlier_stat": best, "z": float(zs[best][i])}
                if zn is not None:
                    znb = float(zn[best][i]) if np.isfinite(zn[best][i]) else None
                    if znb is not None and abs(znb) <= SCREEN_Z_NOISE:
                        continue    # an outlier only because this client is small: within its own noise
                    row["z_noise"] = znb
                kept.append(row)
    return kept, medians


def _rows_to_table(rows: List[dict], columns: List[str]) -> dict:
    return _table(columns, [[r.get(c) for c in columns] for r in rows])


# ----------------------------------------------------------------- tool specs

def _eps_prop():
    return {"type": "number", "description": f"Privacy budget to spend on this call "
            f"(default {DEFAULT_EPSILON}). More epsilon = less noise. The call's epsilon is "
            "split over the statistics it releases per client, so narrower queries are more precise."}


TOOL_SPECS = [
    {"name": "federation_overview",
     "description": "Free (no budget). Model type, number of rounds, client names with their "
                    "training sizes (the server knows these for FedAvg weighting), feature names, "
                    "sensitive attributes and their values, remaining privacy budget.",
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "global_performance",
     "description": "Pooled test metrics of the global model after each round: AUC, Brier, "
                    "observed prevalence, mean predicted risk, calibration-in-the-large (CITL = "
                    "mean predicted - observed).",
     "input_schema": {"type": "object", "properties": {"epsilon": _eps_prop()},
                      "additionalProperties": False}},
    {"name": "client_performance",
     "description": "Each client's local metrics of the global model on its own held-out data "
                    "at one round (default: final): AUC, ECE, Brier, observed prevalence, mean "
                    "predicted risk, CITL. Includes a pooled row. `flags` lists the stats on which the client stands out after allowing for its own noise (noise-adjusted z beyond +/-3, e.g. \"mean_cos z=-9.2\": a z-score, not the value of mean_cos); empty = within noise. To cite a flag, use metric \"<stat>_z\" (e.g. mean_cos_z) with the z value.",
     "input_schema": {"type": "object", "properties": {
         "round": {"type": "integer", "description": "Round (1..R); default final."},
         "epsilon": _eps_prop()}, "additionalProperties": False}},
    {"name": "client_updates",
     "description": "Statistics of each client's uploaded model updates across rounds. "
                    "norm_ratio = update norm / median norm of all clients that round; cos = cosine "
                    "between the client's update and the coordinate-wise median update of all clients. "
                    "Without `client`: one summary row per client (median and min norm_ratio, mean "
                    "and min cos, plus `flags`). With `client`: that client's per-round values. Update norms grow with client size (more local steps), so the norm-ratio flags are size-adjusted: they compare a client with what its size predicts. `flags` lists the stats on which the client stands out after allowing for its own noise (noise-adjusted z beyond +/-3, e.g. \"mean_cos z=-9.2\": a z-score, not the value of mean_cos); empty = within noise. To cite a flag, use metric \"<stat>_z\" (e.g. mean_cos_z) with the z value.",
     "input_schema": {"type": "object", "properties": {
         "client": {"type": "string"}, "epsilon": _eps_prop()}, "additionalProperties": False}},
    {"name": "feature_summary",
     "description": "Per-client feature statistics in raw units (mean, sd, 1st and 99th "
                    "percentile) on the clients' training data at a round (default final). "
                    "Without `clients`: only client-feature rows that are robust outliers versus "
                    "the other clients, plus the median row per feature. With `clients`: full rows "
                    "for those clients. Restrict `features` for less noise. Outlier rows carry `z` (robust z vs the other clients) and `z_noise` (the same deviation in units that include that client's own noise; only rows with |z_noise| > 3 are listed).",
     "input_schema": {"type": "object", "properties": {
         "features": {"type": "array", "items": {"type": "string"}},
         "clients": {"type": "array", "items": {"type": "string"}},
         "round": {"type": "integer"}, "epsilon": _eps_prop()}, "additionalProperties": False}},
    {"name": "subgroup_outcomes",
     "description": "Per-client, per-subgroup outcomes on held-out data for one sensitive "
                    "attribute: n, observed outcome rate, mean predicted risk, O/E ratio (observed/"
                    "expected) and false-negative rate at `threshold` (default 0.2). Without "
                    "`clients`: only outlier rows plus the median row per subgroup. Outlier rows carry `z` (robust z vs the other clients) and `z_noise` (the same deviation in units that include that client's own noise; only rows with |z_noise| > 3 are listed).",
     "input_schema": {"type": "object", "properties": {
         "attribute": {"type": "string"},
         "clients": {"type": "array", "items": {"type": "string"}},
         "threshold": {"type": "number"}, "epsilon": _eps_prop()},
         "required": ["attribute"], "additionalProperties": False}},
    {"name": "drift_by_round",
     "description": "Data drift over training rounds. shift_sd = largest absolute change of "
                    "any feature's client mean since round 1, in federation SDs (shift_feature names "
                    "it); citl = calibration-in-the-large of that round's global model on the "
                    "client's held-out data; prevalence = observed outcome rate. Without `clients`: "
                    "one summary row per client (max and mean shift_sd, first round with two "
                    "consecutive shift_sd > 0.3, citl in the first and last round) plus a pooled "
                    "per-round table. With `clients`: per-round rows for those clients. `flags` lists the stats on which the client stands out after allowing for its own noise (noise-adjusted z beyond +/-3, e.g. \"mean_cos z=-9.2\": a z-score, not the value of mean_cos); empty = within noise. To cite a flag, use metric \"<stat>_z\" (e.g. mean_cos_z) with the z value.",
     "input_schema": {"type": "object", "properties": {
         "clients": {"type": "array", "items": {"type": "string"}},
         "epsilon": _eps_prop()}, "additionalProperties": False}},
    {"name": "attribution_churn",
     "description": "How the global model's explanations change from round to round, from "
                    "per-patient attributions computed at the clients (signed contribution of each "
                    "feature to the log-odds, relative to the federation mean). Per round r >= 2: "
                    "rank correlation of global feature importance with round r-1, top-3 overlap, "
                    "share of patients whose top risk factor is unchanged, relative L2 change of "
                    "importance; plus the top 3 features at each round.",
     "input_schema": {"type": "object", "properties": {"epsilon": _eps_prop()},
                      "additionalProperties": False}},
]
TOOL_NAMES = [t["name"] for t in TOOL_SPECS]


# --------------------------------------------------------------------- tools

class AuditTools:
    def __init__(self, run: AuditRun):
        self.run = run
        d = run.planted.data
        self.names = d.client_names
        self.features = list(d.feature_names)
        self.mean = np.asarray(d.scaler["mean"], dtype=float)
        self.scale = np.asarray(d.scaler["scale"], dtype=float)
        self.attrs = list(d.clients[0].groups["train"].keys())

    # helpers
    def _client(self, data, name):
        return data.client(name)

    def _held_out(self, c):
        X = np.vstack([c.X["cal"], c.X["test"]])
        y = np.concatenate([c.y["cal"], c.y["test"]])
        g = {a: np.concatenate([c.groups["cal"][a], c.groups["test"][a]]) for a in self.attrs}
        return X, y, g

    def _check_clients(self, clients):
        bad = [c for c in clients if c not in self.names]
        if bad:
            raise KeyError(f"unknown clients {bad}")

    # ------------------------------------------------------------- free
    def federation_overview(self, **_):
        d = self.run.planted.data
        values = {}
        for a in self.attrs:
            v = np.unique(np.concatenate([c.groups["train"][a] for c in d.clients]).astype(str))
            values[a] = v.tolist() if len(v) <= 20 else v[:20].tolist() + ["..."]
        return {
            "model": self.run.scenario.model, "strategy": self.run.result.strategy,
            "rounds": self.run.rounds, "n_clients": len(self.names),
            "clients": _table(["client", "n_train"], [[c.name, c.n_train] for c in d.clients]),
            "features": self.features, "sensitive_attributes": values,
        }

    # ------------------------------------------------------------- paid
    def global_performance(self, noise_rng, epsilon, **_):
        R = self.run.rounds
        nz = _Noise(noise_rng, epsilon, 5 * R)
        rows = []
        for r in range(1, R + 1):
            data, model = self.run.data_at(r), self.run.global_at(r)
            X, y = data.pooled("test")
            p = model.predict_proba(X)
            m = full_metrics(y, p)
            n, npos = len(y), int(y.sum())
            prev, mp = float(y.mean()), float(p.mean())
            rows.append([r, nz(m["auc"], 1, min(npos, n - npos), 0, 1), nz(m["brier"], 1, n, 0),
                         nz(prev, 1, n, 0, 1), nz(mp, 1, n, 0, 1), None])
            rows[-1][5] = rows[-1][4] - rows[-1][3]
        return {"by_round": _table(["round", "auc", "brier", "prevalence", "mean_pred", "citl"], rows)}

    def client_performance(self, noise_rng, epsilon, round=None, **_):
        r = self.run.rounds if round is None else int(round)
        if not 1 <= r <= self.run.rounds:
            raise ValueError(f"round must be in 1..{self.run.rounds}")
        data, model = self.run.data_at(r), self.run.global_at(r)
        nz = _Noise(noise_rng, epsilon, 5)
        rows, ys, ps = [], [], []
        for c in data.clients:
            y, p = c.y["test"], model.predict_proba(c.X["test"])
            ys.append(y)
            ps.append(p)
            rows.append([c.name, *self._perf(nz, y, p)])
        # noise-adjusted outlier flags (columns: client n auc ece brier prevalence mean_pred citl)
        sd = {"auc": lambda r: laplace_sd(1, min(r[5], 1 - r[5]) * r[1] if r[5] is not None else 1, 5, epsilon),
              "ece": lambda r: laplace_sd(2, r[1], 5, epsilon),
              "citl": lambda r: SQ2 * laplace_sd(1, r[1], 5, epsilon),
              "prevalence": lambda r: laplace_sd(1, r[1], 5, epsilon)}
        col = {"auc": 2, "ece": 3, "prevalence": 5, "citl": 7}
        zs = {s: _noise_z([r[col[s]] for r in rows], [f(r) for r in rows]) for s, f in sd.items()}
        for i, row in enumerate(rows):
            row.append(_flags(zs, i))
        rows.append(["_pooled", *self._perf(nz, np.concatenate(ys), np.concatenate(ps)), ""])
        return {"round": r, "clients": _table(
            ["client", "n", "auc", "ece", "brier", "prevalence", "mean_pred", "citl", "flags"], rows)}

    @staticmethod
    def _perf(nz, y, p):
        m = full_metrics(y, p)
        n, npos = len(y), int(y.sum())
        prev, mp = nz(float(y.mean()), 1, n, 0, 1), nz(float(p.mean()), 1, n, 0, 1)
        return [n, nz(m["auc"], 1, min(npos, n - npos), 0, 1), nz(m["ece"], 2, n, 0, 1),
                nz(m["brier"], 1, n, 0), prev, mp, mp - prev]

    def _update_stats(self):
        """Per round: {client: (norm_ratio, cos)} from the recorded uploads."""
        out = {}
        for r, ups in sorted(self.run.updates.items()):
            names = list(ups)
            U = np.stack([ups[n] for n in names])
            norms = np.linalg.norm(U, axis=1)
            med = np.median(norms) or 1e-12
            ref = np.median(U, axis=0)       # robust to a few large or flipped uploads
            ref_norm = np.linalg.norm(ref)
            stats = {}
            for i, n in enumerate(names):
                denom = norms[i] * ref_norm
                cos = float(U[i] @ ref / denom) if denom > 0 else 0.0
                stats[n] = (float(norms[i] / med), cos)
            out[r] = stats
        return out

    def client_updates(self, noise_rng, epsilon, client=None, **_):
        stats = self._update_stats()
        sizes = {c.name: c.n_train for c in self.run.planted.data.clients}
        R = len(stats)
        if client is not None:
            self._check_clients([client])
            nz = _Noise(noise_rng, epsilon, 2 * R)
            n = sizes[client]
            rows = []
            for r, s in stats.items():
                if client in s:
                    ratio, cos = s[client]
                    rows.append([client, r, nz(min(ratio, 20.0), UPDATE_C, n, 0),
                                 nz(cos, 2 * UPDATE_C, n, -1, 1)])
            return {"per_round": _table(["client", "round", "norm_ratio", "cos"], rows)}
        nz = _Noise(noise_rng, epsilon, 4)
        rows = []
        for name in self.names:
            vals = [s[name] for s in stats.values() if name in s]
            if not vals:
                continue
            ratio = np.minimum(np.array([v[0] for v in vals]), 20.0)
            cos = np.array([v[1] for v in vals])
            n = sizes[name]
            rows.append([name, len(vals),
                         nz(float(np.median(ratio)), UPDATE_C, n, 0),
                         nz(float(ratio.min()), UPDATE_C, n, 0),
                         nz(float(cos.mean()), 2 * UPDATE_C, n, -1, 1),
                         nz(float(cos.min()), 2 * UPDATE_C, n, -1, 1)])
        zs = {}
        for k, name in ((4, "mean_cos"), (5, "min_cos")):
            zs[name] = _noise_z([r[k] for r in rows], [laplace_sd(2 * UPDATE_C, sizes[r[0]], 4, epsilon) for r in rows])
        # Update norms grow with client size (fixed local epochs => more SGD steps), so the norm
        # flags compare each client with the size trend: log(ratio) minus a fit on log(n_train).
        log_n = np.log([sizes[r[0]] for r in rows])
        for k, name in ((2, "median_norm_ratio"), (3, "min_norm_ratio")):
            v = np.array([np.nan if r[k] is None else max(r[k], 1e-3) for r in rows], dtype=float)
            ok = np.isfinite(v)
            resid = np.full(len(v), np.nan)
            if ok.sum() >= 3:
                slope, icpt = np.polyfit(log_n[ok], np.log(v[ok]), 1)
                resid[ok] = np.log(v[ok]) - (icpt + slope * log_n[ok])
            sds = [laplace_sd(UPDATE_C, sizes[r[0]], 4, epsilon) / max(r[k] or 1e-3, 1e-3) for r in rows]
            zs[name] = _noise_z(list(resid), sds)
        for i, row in enumerate(rows):
            row.append(_flags(zs, i))
        return {"clients": _table(["client", "rounds", "median_norm_ratio", "min_norm_ratio",
                                   "mean_cos", "min_cos", "flags"], rows)}

    def feature_summary(self, noise_rng, epsilon, features=None, clients=None, round=None, **_):
        r = self.run.rounds if round is None else int(round)
        data = self.run.data_at(r)
        feats = features or self.features
        bad = [f for f in feats if f not in self.features]
        if bad:
            raise KeyError(f"unknown features {bad}")
        idx = [self.features.index(f) for f in feats]
        targets = clients or self.names
        self._check_clients(targets)
        nz = _Noise(noise_rng, epsilon, 4 * len(idx))
        lo = self.mean - FEATURE_CLIP_SD * self.scale
        hi = self.mean + FEATURE_CLIP_SD * self.scale
        rows = []
        for name in targets:
            X = data.client(name).X["train"][:, idx]
            raw = np.clip(X * self.scale[idx] + self.mean[idx], lo[idx], hi[idx])
            st = local_sufficient_statistics(raw)
            n = st["n"]
            mean = np.asarray(st["sum"]) / n
            sd = np.sqrt(np.maximum(np.asarray(st["sumsq"]) / n - mean ** 2, 0.0))
            q01, q99 = np.percentile(raw, [1, 99], axis=0)
            for k, j in enumerate(idx):
                c = hi[j] - lo[j]
                rows.append({"client": name, "feature": self.features[j],
                             "mean": nz(mean[k], c, n), "sd": nz(sd[k], c, n, 0),
                             "p01": nz(q01[k], c, n), "p99": nz(q99[k], c, n),
                             "_noise_sd": laplace_sd(c, n, 4 * len(idx), epsilon)})
        cols = ["client", "feature", "mean", "sd", "p01", "p99"]
        if clients:
            return {"round": r, "rows": _rows_to_table(rows, cols)}
        kept, med = _screen(rows, ["mean", "sd", "p01", "p99"], "feature",
                            sd_of=lambda row, stat: row["_noise_sd"])
        return {"round": r, "outliers": _rows_to_table(kept, cols + ["outlier_stat", "z", "z_noise"]),
                "medians": _rows_to_table(med, cols)}

    def subgroup_outcomes(self, noise_rng, epsilon, attribute, clients=None, threshold=0.2, **_):
        if attribute not in self.attrs:
            raise KeyError(f"unknown attribute {attribute!r}; expected {self.attrs}")
        threshold = 0.2 if threshold is None else float(threshold)
        data, model = self.run.final_data, self.run.global_at(self.run.rounds)
        targets = clients or self.names
        self._check_clients(targets)
        n_groups = len(np.unique(np.concatenate([c.groups["test"][attribute] for c in data.clients])))
        nz = _Noise(noise_rng, epsilon, 4 * n_groups)
        rows = []
        for name in targets:
            X, y, g = self._held_out(data.client(name))
            p = model.predict_proba(X)
            groups = g[attribute].astype(str)
            gm = group_metrics(y, p, groups, threshold)
            counts = group_counts_over_grid(y, p, groups, np.array([threshold]))
            for grp, m in gm.items():
                n = m["n"]
                npos = m["positives"]
                rate = nz(m["prevalence"], 1, n, 0, 1)
                mp = nz(float(p[groups == grp].mean()), 1, n, 0, 1)
                tp, fn = counts[grp]["tp"][0], counts[grp]["fn"][0]
                fnr = nz(fn / (tp + fn), 1, npos, 0, 1) if tp + fn else None
                rows.append({"client": name, "group": grp, "n": int(round(nz(n, 1, 1, 0))),
                             "outcome_rate": rate, "mean_pred": mp,
                             "oe_ratio": rate / mp if mp else None, "fnr": fnr})
        cols = ["client", "group", "n", "outcome_rate", "mean_pred", "oe_ratio", "fnr"]
        if clients:
            return {"attribute": attribute, "threshold": threshold, "rows": _rows_to_table(rows, cols)}
        def sd_of(row, stat):   # from released values only: n, rate, mean_pred
            n, m = max(row["n"] or 1, 1), 4 * n_groups
            if stat == "fnr":
                return laplace_sd(1, max(n * (row["outcome_rate"] or 0), 1), m, epsilon)
            base = laplace_sd(1, n, m, epsilon)
            return base / max(row["mean_pred"] or 0, 1e-3) if stat == "oe_ratio" else base

        kept, med = _screen(rows, ["outcome_rate", "oe_ratio", "fnr"], "group", sd_of=sd_of)
        return {"attribute": attribute, "threshold": threshold,
                "outliers": _rows_to_table(kept, cols + ["outlier_stat", "z", "z_noise"]),
                "medians": _rows_to_table(med, cols)}

    def drift_by_round(self, noise_rng, epsilon, clients=None, **_):
        R = self.run.rounds
        targets = clients or self.names
        self._check_clients(targets)
        nz = _Noise(noise_rng, epsilon, 3 * R)
        c_rng = 2 * DRIFT_CLIP_SD
        per_client: Dict[str, list] = {}
        pooled = {r: [0.0, 0.0, 0] for r in range(1, R + 1)}   # sum y, sum p, n
        for name in targets:
            base = None
            rows = []
            for r in range(1, R + 1):
                c = self.run.data_at(r).client(name)
                Xt = np.clip(c.X["train"], -DRIFT_CLIP_SD, DRIFT_CLIP_SD)
                mean = np.asarray(local_sufficient_statistics(Xt)["sum"]) / len(Xt)
                base = mean if base is None else base
                shift = np.abs(mean - base)
                n = len(Xt)
                j = int(np.argmax(shift + noise_rng.laplace(0, 2 * c_rng / n * nz.factor, len(shift))))
                s = nz(shift[j], 2 * c_rng, n, 0)
                y, p = c.y["test"], self.run.global_at(r).predict_proba(c.X["test"])
                nt = len(y)
                prev = nz(float(y.mean()), 1, nt, 0, 1)
                citl = nz(float(p.mean() - y.mean()), 2, nt)
                pooled[r][0] += y.sum()
                pooled[r][1] += p.sum()
                pooled[r][2] += nt
                rows.append([name, r, s, self.features[j], citl, prev])
            per_client[name] = rows
        pooled_rows = [[r, nz(v[0] / v[2], 1, v[2], 0, 1), nz((v[1] - v[0]) / v[2], 2, v[2])]
                       for r, v in pooled.items()]
        out = {"pooled": _table(["round", "prevalence", "citl"], pooled_rows)}
        cols = ["client", "round", "shift_sd", "shift_feature", "citl", "prevalence"]
        if clients:
            out["per_round"] = _table(cols, [row for name in targets for row in per_client[name]])
            return out
        summary = []
        for name, rows in per_client.items():
            shifts = [row[2] for row in rows]
            onset = next((rows[i][1] for i in range(len(rows) - 1)
                          if shifts[i] > 0.3 and shifts[i + 1] > 0.3), None)
            top = rows[int(np.argmax(shifts))]
            summary.append([name, max(shifts), float(np.mean(shifts)), top[3], onset,
                            rows[0][4], rows[-1][4]])
        n_train = {c.name: c.n_train for c in self.run.planted.data.clients}
        sds = [laplace_sd(2 * c_rng, n_train[row[0]], 3 * R, epsilon) for row in summary]
        zs = {"max_shift_sd": _noise_z([row[1] for row in summary], sds),
              "mean_shift_sd": _noise_z([row[2] for row in summary], sds)}
        for i, row in enumerate(summary):
            row.append(_flags(zs, i))
        out["clients"] = _table(["client", "max_shift_sd", "mean_shift_sd", "max_shift_feature",
                                 "onset_round", "citl_first", "citl_last", "flags"], summary)
        return out

    def _attributions(self, model, X):
        if isinstance(model, NumpyModel) and self.run.scenario.model == "lr":
            coef = model.weights[0]
            phi = np.array([linear_phi(coef, x, np.zeros_like(x)) for x in X])
        else:  # occlusion to the federation mean (0 in scaled units), in log-odds
            base = self.run.margin(model, X)
            phi = np.empty_like(X)
            for j in range(X.shape[1]):
                Xj = X.copy()
                Xj[:, j] = 0.0
                phi[:, j] = base - self.run.margin(model, Xj)
        return np.clip(phi, -PHI_CLIP, PHI_CLIP)

    def attribution_churn(self, noise_rng, epsilon, **_):
        R = self.run.rounds
        data = self.run.final_data
        rng = np.random.default_rng([self.run.scenario.seed, 99])
        X = np.vstack([c.X["test"][rng.permutation(len(c.y["test"]))[:ATTR_ROWS]] for c in data.clients])
        N = len(X)
        F = X.shape[1]
        nz = _Noise(noise_rng, epsilon, F * R + R)
        imps, tops, prev_top = [], [], None
        agree = []
        for r in range(1, R + 1):
            phi = self._attributions(self.run.global_at(r), X)
            imp = np.array([nz(v, PHI_CLIP, N, 0) for v in np.abs(phi).mean(axis=0)])
            imps.append(imp)
            top = phi.argmax(axis=1)            # signed top risk factor (federated_explain.signed_top1)
            agree.append(None if prev_top is None else nz(float(np.mean(top == prev_top)), 1, N, 0, 1))
            prev_top = top
            tops.append([self.features[j] for j in np.argsort(imp)[::-1][:3]])
        rows = []
        for r in range(2, R + 1):
            a, b = imps[r - 1], imps[r - 2]
            overlap = len(set(tops[r - 1]) & set(tops[r - 2])) / 3
            rows.append([r, _spearman(a, b), overlap, agree[r - 1],
                         float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
                         ", ".join(tops[r - 1])])
        return {"by_round": _table(["round", "rank_corr_prev", "top3_overlap_prev",
                                    "patient_top1_agree_prev", "rel_change_prev", "top3"], rows)}


# ------------------------------------------------------------------- session

class AuditSession:
    """One audit: tools + accountant + call log. Agents call ``call(name, args)``."""

    FREE = {"federation_overview"}

    def __init__(self, run: AuditRun, budget: float = DEFAULT_BUDGET, seed: int = 0):
        self.run = run
        self.tools = AuditTools(run)
        self.accountant = PrivacyAccountant(budget)
        self.rng = np.random.default_rng([run.scenario.seed, seed, 4242])
        self.log: List[dict] = []

    def call(self, name: str, args: Optional[dict] = None) -> dict:
        args = dict(args or {})
        call_id = f"c{len(self.log) + 1}"
        entry = {"id": call_id, "tool": name, "args": args}
        try:
            if name not in TOOL_NAMES:
                raise KeyError(f"unknown tool {name!r}; expected {TOOL_NAMES}")
            fn = getattr(self.tools, name)
            if name in self.FREE:
                result = fn()
                eps = 0.0
            else:
                eps = float(args.pop("epsilon", None) or DEFAULT_EPSILON)
                if eps <= 0:
                    raise ValueError("epsilon must be positive")
                self.accountant.charge(eps, call_id, name)
                try:
                    result = fn(noise_rng=self.rng, epsilon=eps, **args)
                except Exception:
                    self.accountant.refund(call_id)   # invalid arguments spend nothing
                    raise
            out = {"call_id": call_id, "tool": name, "epsilon_spent": eps,
                   "budget_left": self.accountant.remaining, **result}
            entry.update(ok=True, result=out)
        except (BudgetExhausted, KeyError, ValueError, TypeError) as e:
            out = {"call_id": call_id, "tool": name, "error": str(e).strip("'\""),
                   "budget_left": self.accountant.remaining}
            entry.update(ok=False, result=out)
        self.log.append(entry)
        return out

    @staticmethod
    def render(result: dict) -> str:
        return json.dumps(result, separators=(",", ":"))
