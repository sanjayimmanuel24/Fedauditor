"""Paper figures and tables for FedAuditor (Phase 5), from existing results only (no API spend).

    python -m fedauditor.make_figures

Reads ``results/fedauditor/compare_brfss_lr.json`` (made by ``fedauditor.compare_agents``), the
LR audit grids and one worked-example audit, and writes to ``results/fedauditor/figures/``:

* ``main_table.{md,tex,csv}``      rules vs Haiku 5.5 prompt v4 / v5 (all findings and conf > .5)
* ``recall_heatmap.{svg,tex,csv}`` recall per fault x severity, rules vs Haiku v5
* ``type_confusion.{svg,tex,csv}`` true fault vs reported fault type, Haiku v4 vs v5
* ``worked_example.md``            one Haiku v5 audit: findings, claims and the tool calls they cite

Every number is re-derived from the grids and asserted equal to ``compare_brfss_lr.json``.
SVGs and LaTeX need no plotting library (LaTeX tables use ``xcolor[table]`` cell colours).
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter

from fedauditor.compare_agents import compare
from fedauditor.faults import FAULTS, SEVERITY_LEVELS
from fedauditor.report import check_claim
from fedauditor.run_audit_grid import RES_DIR, _load, grid_path

DATASET, MODEL = "brfss", "lr"
LABELS = ["rules", "claude-haiku-5-5", "claude-haiku-5-5-p5"]
CONFS = [0.0, 0.51]
OUT = os.path.join(RES_DIR, "figures")
EXAMPLE = "brfss-lr-fedavg-r20-subgroup_bias-high-s20"

# (row key in compare json, display name)
ROWS = [
    ("rules@0", "Rule baseline"),
    ("claude-haiku-5-5@0", "Haiku 5.5, prompt v4"),
    ("claude-haiku-5-5@0.51", "Haiku 5.5, prompt v4, conf > .5"),
    ("claude-haiku-5-5-p5@0", "Haiku 5.5, prompt v5"),
    ("claude-haiku-5-5-p5@0.51", "Haiku 5.5, prompt v5, conf > .5"),
]
FAULT_NAMES = {
    "label_noise": "Label noise", "drift": "Drift", "poisoning": "Poisoning",
    "free_rider": "Free-rider", "prevalence_shift": "Prevalence shift",
    "subgroup_bias": "Subgroup bias", "missing_artefact": "Missing-data artefact",
    "explanation_instability": "Expl. instability", "clean": "Clean (no fault)", "none": "Nothing flagged",
}
# Sequential blue ramp, steps 100 -> 700 (light -> dark).
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
EMPTY = "#f0efec"
# Dark variant for the dashboard: same blue hue, mirrored so that low values sit
# close to the dark surface and high values are bright.
DARK_RAMP = ["#141c30", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]
LIGHT_THEME = dict(bg="#ffffff", title="#1b1b1b", label="#3d3d3a", muted="#6b6b66", empty=EMPTY, ramp=RAMP,
                   ink_on_high="#ffffff", ink_on_low="#1b1b1b", high_from=0.5)
DARK_THEME = dict(bg=None, title="#eef0f6", label="#b9becf", muted="#8a93ab", empty="#121626", ramp=DARK_RAMP,
                  ink_on_high="#04050c", ink_on_low="#eef0f6", high_from=0.55)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def color(v: float, ramp=None) -> str:
    """Interpolate the ramp for v in [0, 1]."""
    ramp = ramp or RAMP
    v = min(max(v, 0.0), 1.0) * (len(ramp) - 1)
    i = min(int(v), len(ramp) - 2)
    a, b, t = _hex(ramp[i]), _hex(ramp[i + 1]), v - i
    return "#%02x%02x%02x" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def ink(v: float) -> str:
    return "#ffffff" if v >= 0.5 else "#1b1b1b"


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def _write(name, text):
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def _csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _close(a, b):
    return (a is None and b is None) or (a is not None and b is not None and abs(a - b) < 1e-9)


# ------------------------------------------------------------------ data

def load_grids():
    grids = {lab: {r["id"]: r for r in _load(grid_path(DATASET, MODEL, lab))} for lab in LABELS}
    common = sorted(set.intersection(*(set(g) for g in grids.values())))
    return grids, common


def check_against_json(fresh: dict) -> dict:
    path = os.path.join(RES_DIR, f"compare_{DATASET}_{MODEL}.json")
    with open(path, encoding="utf-8") as fh:
        saved = json.load(fh)
    assert saved["n_common"] == fresh["n_common"], "scenario count differs from compare json"
    for key, _ in ROWS:
        a, b = saved["rows"][key], fresh["rows"][key]
        for m in ("precision", "recall", "scenario_type_accuracy", "onset_abs_error",
                  "false_alarm_rate", "claim_support", "cost_usd"):
            assert _close(a[m], b[m]), f"{key} {m}: json {a[m]} vs recomputed {b[m]}"
        for cell, v in a["by_cell"].items():
            assert _close(v, b["by_cell"][cell]), f"{key} {cell}"
    return saved


# ------------------------------------------------------------------ main table

def main_table(res: dict, n: int):
    hdr = ["Auditor", "Precision", "Recall", "Type acc.", "Onset err.", "Clean FAR",
           "Claims verified", "Cost (USD)"]
    rows = []
    for key, name in ROWS:
        r = res["rows"][key]
        far = f"{round(r['false_alarm_rate'] * r['n_clean'])}/{r['n_clean']}"
        rows.append([name, fmt(r["precision"]), fmt(r["recall"]), fmt(r["scenario_type_accuracy"]),
                     fmt(r["onset_abs_error"]), far, f"{r['claim_support']:.1%}", f"{r['cost_usd']:.2f}"])
    _csv("main_table.csv", hdr, rows)
    md = [f"Main result: BRFSS, logistic regression, {n} evaluation scenarios "
          "(8 faults x 3 severities x 5 seeds + 5 clean). Precision/recall are client-level, "
          "micro-averaged; type accuracy = most-reported fault type correct (ties 1/k); "
          "onset error in rounds; clean FAR = clean runs with any finding.", "",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    _write("main_table.md", "\n".join(md) + "\n")
    tex = [r"\begin{table*}[t]", r"\centering", r"\small",
           r"\caption{FedFault results on BRFSS with logistic regression (" + str(n) +
           r" evaluation scenarios). All auditors use the same tools and privacy budget "
           r"($\varepsilon=20$). ``conf $>$ .5'' keeps only findings with confidence above 0.5 "
           r"(a post-hoc cut, sensitivity only).}",
           r"\label{tab:main}", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
           " & ".join(["Auditor", "Prec.", "Rec.", "Type", "Onset", "FAR", "Verified", r"Cost (\$)"]) + r" \\",
           r"\midrule"]
    for r in rows:
        tex.append(" & ".join(c.replace("%", r"\%").replace("–", "--") for c in r) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    _write("main_table.tex", "\n".join(tex) + "\n")
    return rows


# ------------------------------------------------------------------ heatmaps (SVG + LaTeX)

def svg_grid(panels, row_keys, col_keys, col_labels, title, note, cell_fmt, vmax=1.0, cw=64, ch=26,
             theme=None):
    """panels: [(panel title, {(row, col): value or None})]. Returns SVG text."""
    T = theme or LIGHT_THEME
    left, top, gap = 168, 64, 28
    pw = cw * len(col_keys)
    width = left + len(panels) * pw + (len(panels) - 1) * gap + 16
    notes = note.split("\n")
    height = top + ch * len(row_keys) + 30 + 14 * len(notes)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
         f'height="{height}" font-family="Onest, Helvetica, Arial, sans-serif" font-size="11">']
    if T["bg"]:
        s.append(f'<rect width="{width}" height="{height}" fill="{T["bg"]}"/>')
    s.append(f'<text x="8" y="18" font-size="13" font-weight="bold" fill="{T["title"]}">{title}</text>')
    for i, rk in enumerate(row_keys):
        y = top + i * ch + ch / 2 + 4
        s.append(f'<text x="{left - 8}" y="{y}" text-anchor="end" fill="{T['label']}">{FAULT_NAMES.get(rk, rk)}</text>')
    for p, (ptitle, vals) in enumerate(panels):
        x0 = left + p * (pw + gap)
        s.append(f'<text x="{x0 + pw / 2}" y="{top - 26}" text-anchor="middle" font-weight="bold" '
                 f'fill="{T["title"]}">{ptitle}</text>')
        for j, lab in enumerate(col_labels):
            s.append(f'<text x="{x0 + j * cw + cw / 2}" y="{top - 8}" text-anchor="middle" '
                     f'fill="{T["muted"]}">{lab}</text>')
        for i, rk in enumerate(row_keys):
            for j, ck in enumerate(col_keys):
                v = vals.get((rk, ck))
                x, y = x0 + j * cw, top + i * ch
                fill = T["empty"] if v is None else color(v / vmax, T["ramp"])
                tip = f"{ptitle}: {FAULT_NAMES.get(rk, rk)} / {col_labels[j]} = {cell_fmt(v)}"
                s.append(f'<g><title>{tip}</title><rect x="{x + 1}" y="{y + 1}" width="{cw - 2}" '
                         f'height="{ch - 2}" rx="2" fill="{fill}"/>')
                if v is not None:
                    s.append(f'<text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" '
                             f'fill="{T["ink_on_high"] if v / vmax >= T["high_from"] else T["ink_on_low"]}">{cell_fmt(v)}</text>')
                s.append("</g>")
    for k, line in enumerate(notes):
        s.append(f'<text x="8" y="{height - 12 - 14 * (len(notes) - 1 - k)}" fill="{T['muted']}">{line}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def tex_grid(panels, row_keys, col_keys, col_labels, caption, label, cell_fmt, vmax=1.0, wide=False,
             titles=None, theme=None):
    T = theme or LIGHT_THEME
    env = "table*" if wide else "table"
    defs = [r"% needs \usepackage[table]{xcolor} and \usepackage{booktabs}"]
    cols = "l" + ("|" + "c" * len(col_keys)) * len(panels)
    head1 = " & ".join([""] + [rf"\multicolumn{{{len(col_keys)}}}{{c}}{{{t}}}"
                               for t in (titles or [t for t, _ in panels])]) + r" \\"
    head2 = " & ".join(["Fault"] + list(col_labels) * len(panels)) + r" \\"
    body = []
    for rk in row_keys:
        cells = [FAULT_NAMES.get(rk, rk)]
        for _, vals in panels:
            for ck in col_keys:
                v = vals.get((rk, ck))
                if v is None:
                    cells.append("--")
                else:
                    r, g, b = _hex(color(v / vmax, T["ramp"]))
                    txt = cell_fmt(v)
                    if theme is None:     # light: unchanged output (white text on dark cells)
                        txt = rf"\textcolor{{white}}{{{txt}}}" if v / vmax >= 0.5 else txt
                    else:
                        ink_hex = T["ink_on_high"] if v / vmax >= T["high_from"] else T["ink_on_low"]
                        txt = rf"\textcolor[HTML]{{{ink_hex.lstrip('#').upper()}}}{{{txt}}}"
                    cells.append(rf"\cellcolor[RGB]{{{r},{g},{b}}}{txt}")
        body.append(" & ".join(cells) + r" \\")
    return "\n".join(defs + [rf"\begin{{{env}}}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
                             rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                             rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head1, head2, r"\midrule"]
                     + body + [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]) + "\n"


def recall_heatmap(res: dict):
    sev = list(SEVERITY_LEVELS)
    panels = []
    for key, name in [("rules@0", "Rule baseline"), ("claude-haiku-5-5-p5@0", "Haiku 5.5, prompt v5")]:
        cells = res["rows"][key]["by_cell"]
        panels.append((name, {(f, s): cells.get(f"{f}/{s}") for f in FAULTS for s in sev}))
    two = lambda v: fmt(v)
    for suffix, theme in (("", LIGHT_THEME), ("_dark", DARK_THEME)):
        _write(f"recall_heatmap{suffix}.svg", svg_grid(
            panels, FAULTS, sev, sev, "Recall of planted clients by fault and severity (BRFSS, LR)",
            "Recall = planted clients flagged / planted clients (3 per scenario, 5 seeds per cell).", two,
            theme=theme))
    for suffix, theme in (("", None), ("_dark", DARK_THEME)):
        _write(f"recall_heatmap{suffix}.tex", tex_grid(
            panels, FAULTS, sev, ["L", "M", "H"],
            "Recall of the planted clients by fault type and severity (L/M/H), BRFSS, logistic regression, "
            "5 seeds per cell. Left: rule baseline; right: Haiku 5.5 with prompt v5.",
            "tab:recall", two, titles=["Rules", "Haiku v5"], theme=theme))
    _csv("recall_heatmap.csv", ["fault"] + [f"{p}:{s}" for p, _ in panels for s in sev],
         [[f] + [fmt(v[(f, s)], 4) for _, v in panels for s in sev] for f in FAULTS])
    return panels


# ------------------------------------------------------------------ type confusion

def scenario_type_votes(rec: dict) -> dict:
    """Reported type(s) for one scenario: the most common finding type (ties split 1/k)."""
    types = [f["fault_type"] for f in rec["findings"]]
    if not types:
        return {"none": 1.0}
    top = max(Counter(types).values())
    modes = sorted(t for t, c in Counter(types).items() if c == top)
    return {t: 1.0 / len(modes) for t in modes}


def type_confusion(grids, common, res):
    true_keys = list(FAULTS) + ["clean"]
    pred_keys = list(FAULTS) + ["none"]
    panels = []
    for lab, name, key in [("claude-haiku-5-5", "Haiku 5.5, prompt v4", "claude-haiku-5-5@0"),
                           ("claude-haiku-5-5-p5", "Haiku 5.5, prompt v5", "claude-haiku-5-5-p5@0")]:
        m = {(t, p): 0.0 for t in true_keys for p in pred_keys}
        for i in common:
            rec = grids[lab][i]
            for p, w in scenario_type_votes(rec).items():
                m[(rec["truth"]["fault"], p)] += w
        diag = sum(m[(f, f)] for f in FAULTS)
        n_faulty = sum(1 for i in common if grids[lab][i]["truth"]["fault"] != "clean")
        assert _close(diag / n_faulty, res["rows"][key]["scenario_type_accuracy"]), \
            f"{lab}: confusion diagonal {diag / n_faulty} != scenario type accuracy"
        panels.append((name, {k: (v if v else None) for k, v in m.items()}))
    cnt = lambda v: "" if v is None else (f"{v:g}" if float(v).is_integer() else f"{v:.1f}")
    short = ["LN", "DR", "PO", "FR", "PS", "SB", "MA", "EI", "–"]
    vmax = 15.0
    for suffix, theme in (("", LIGHT_THEME), ("_dark", DARK_THEME)):
        _write(f"type_confusion{suffix}.svg", svg_grid(
            panels, true_keys, pred_keys, short,
            "Planted fault (rows) vs fault type the agent reported (columns), scenarios",
            "LN label noise, DR drift, PO poisoning, FR free-rider, PS prevalence shift,\n"
            "SB subgroup bias, MA missing-data artefact, EI expl. instability, – nothing flagged. 15 scenarios per fault.",
            cnt, vmax=vmax, cw=30, theme=theme))
    for suffix, theme in (("", None), ("_dark", DARK_THEME)):
        _write(f"type_confusion{suffix}.tex", tex_grid(
            panels, true_keys, pred_keys, short,
            "Fault-type confusion for Haiku 5.5 with prompt v4 (left) and v5 (right): planted fault (rows) "
            "vs the most-reported type per scenario (ties split). 15 scenarios per fault, 5 clean.",
            "tab:confusion", cnt, vmax=vmax, wide=True, titles=["Prompt v4", "Prompt v5"], theme=theme))
    _csv("type_confusion.csv", ["prompt", "true"] + pred_keys,
         [[name, t] + [fmt(v.get((t, p)) or 0.0, 2) for p in pred_keys] for name, v in panels for t in true_keys])
    return panels


# ------------------------------------------------------------------ worked example

def _cited_row(entry: dict, claim: dict):
    """The logged table row a claim points at (for display)."""
    for tname, table in ((k, v) for k, v in entry["result"].items()
                         if isinstance(v, dict) and "columns" in v and "rows" in v):
        cols = table["columns"]
        for row in table["rows"]:
            if all(str(row[cols.index(k)]) == str(claim[k]) for k in ("client", "round", "feature", "group")
                   if k in claim and k in cols):
                if claim["metric"] in cols or (claim["metric"].endswith("_z") and "flags" in cols):
                    return tname, dict(zip(cols, row))
    return None, None


def worked_example():
    path = os.path.join(RES_DIR, "audits", f"{EXAMPLE}__claude-haiku-5-5-p5.json")
    with open(path, encoding="utf-8") as fh:
        a = json.load(fh)
    log = {e["id"]: e for e in a["log"]}
    sc, truth, rep = a["scenario"], a["truth"], a["report"]
    out = [f"# Worked example: `{EXAMPLE}` audited by Haiku 5.5 (prompt v5)", "",
           f"**Planted fault:** {truth['fault']} ({truth['severity']}) in "
           f"{', '.join(truth['clients'])} from round {truth['onset_round']}.  ",
           f"**Score:** precision {a['score']['precision']:.2f}, recall {a['score']['recall']:.2f}, "
           f"type {'correct' if a['score']['scenario_type_correct'] else 'wrong'}, onset error "
           f"{a['score']['onset_abs_error']}.  ",
           f"**Fact-check:** {a['fact_check']['supported']}/{a['fact_check']['claims']} claims supported.  ",
           f"**Budget:** ε {a['budget']['spent']:g} of {a['budget']['total']:g}; {len(a['log'])} tool calls, "
           f"{a['turns']} turns.", "", "## Tool calls", "", "| id | tool | args | ε |", "|---|---|---|---|"]
    for e in a["log"]:
        out.append(f"| {e['id']} | {e['tool']} | {json.dumps(e['args']) if e['args'] else ''} | "
                   f"{e['result'].get('epsilon_spent', 0):g} |")
    out += ["", "## Agent summary", "", rep.get("summary", ""), "", "## Findings and the logged values they cite"]
    for f in rep["findings"]:
        out += ["", f"### {f['client']}: {f['fault_type']} (onset {f['onset_round']}, confidence "
                f"{f['confidence']})", "", f["rationale"], "",
                "| claim | metric | claimed | logged | cites | row |", "|---|---|---|---|---|---|"]
        for c in f["claims"]:
            chk = check_claim(c, log)
            tname, row = _cited_row(log[c["call_id"]], c) if c["call_id"] in log else (None, None)
            keys = ", ".join(f"{k}={c[k]}" for k in ("client", "feature", "group", "round") if k in c)
            logged = chk.get("logged", "✗ " + chk.get("reason", ""))
            out.append(f"| {keys} | {c['metric']} | {c['value']} | {logged} | "
                       f"{c['call_id']} {log[c['call_id']]['tool'] if c['call_id'] in log else '?'} | "
                       f"{tname or ''} |")
    _write("worked_example.md", "\n".join(out) + "\n")
    return a


# ------------------------------------------------------------------ extra tables for the paper

def _tex_table(name, caption, label, header, rows, align, wide=False):
    env = "table*" if wide else "table"
    tex = [rf"\begin{{{env}}}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           rf"\caption{{{caption}}}", rf"\label{{{label}}}", rf"\begin{{tabular}}{{{align}}}",
           r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    tex += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    tex += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    _write(name + ".tex", "\n".join(tex) + "\n")
    _csv(name + ".csv", header, rows)


def pooled_auc_table():
    """Change in pooled AUC (clean test data) caused by each fault, paired with the same-seed clean run."""
    models = ["lr", "mlp", "xgb"]
    delta = {}            # (model, fault, severity) -> mean paired delta over seeds
    clean_range = {}
    for m in models:
        recs = _load(os.path.join(RES_DIR, f"fedfault_{DATASET}_{m}.jsonl"))
        clean = {r["scenario"]["seed"]: r["clean_eval"]["_pooled"]["auc"]
                 for r in recs if r["scenario"]["fault"] == "clean"}
        clean_range[m] = (min(clean.values()), max(clean.values()))
        acc = {}
        for r in recs:
            sc = r["scenario"]
            if sc["fault"] != "clean":
                acc.setdefault((sc["fault"], sc["severity"]), []).append(
                    r["clean_eval"]["_pooled"]["auc"] - clean[sc["seed"]])
        for k, v in acc.items():
            delta[(m,) + k] = sum(v) / len(v)
    sev = list(SEVERITY_LEVELS)
    f3 = lambda x: f"{x:+.3f}".replace("+0.000", "0.000").replace("-0.000", "0.000")
    rows = [[FAULT_NAMES[f]] + [f3(delta[("lr", f, s)]) for s in sev]
            + [f3(delta[("mlp", f, "high")]), f3(delta[("xgb", f, "high")])] for f in FAULTS]
    rows.append([r"\textit{Clean, seed range}", "", "", f"{clean_range['lr'][1] - clean_range['lr'][0]:.3f}",
                 f"{clean_range['mlp'][1] - clean_range['mlp'][0]:.3f}",
                 f"{clean_range['xgb'][1] - clean_range['xgb'][0]:.3f}"])
    _tex_table("pooled_auc", "Change in pooled test AUC caused by each planted fault, relative to the "
               "clean run with the same seed (mean of 5 seeds). LR at low/medium/high severity, MLP and "
               "XGBoost at high severity. The last row is the spread of clean-run AUC across seeds.",
               "tab:pooled", ["Fault", "LR L", "LR M", "LR H", "MLP H", "XGB H"], rows, "lrrrrr")
    return delta, clean_range


def rules_models_table():
    rows = []
    for m, name in [("lr", "Logistic regression"), ("mlp", "MLP"), ("xgb", "XGBoost")]:
        r = compare(DATASET, m, ["rules"], [0.0])["rows"]["rules@0"]
        rows.append([name, fmt(r["precision"]), fmt(r["recall"]), fmt(r["scenario_type_accuracy"]),
                     fmt(r["onset_abs_error"]), f"{round(r['false_alarm_rate'] * r['n_clean'])}/{r['n_clean']}"])
    _tex_table("rules_models", "Rule baseline on the three model families (125 evaluation scenarios each, "
               "thresholds tuned on LR development runs only).", "tab:models",
               ["Model", "Prec.", "Rec.", "Type", "Onset", "FAR"], rows, "lrrrrr")
    return rows


def behaviour_table(grids, common):
    """How each auditor spends its budget: calls, turns, time, epsilon, findings, claims, cost."""
    rows = []
    for lab, name in [("rules", "Rule baseline"), ("claude-haiku-5-5", "Haiku 5.5, prompt v4"),
                      ("claude-haiku-5-5-p5", "Haiku 5.5, prompt v5")]:
        recs = [grids[lab][i] for i in common]
        n = len(recs)
        mean = lambda k: sum(r[k] for r in recs) / n
        claims = sum(r["fact_check"]["claims"] for r in recs)
        sup = sum(r["fact_check"]["supported"] for r in recs)
        rows.append([name, f"{mean('calls'):.1f}", f"{mean('turns'):.1f}" if lab != "rules" else "--",
                     f"{mean('seconds'):.0f}", f"{mean('epsilon'):.1f}",
                     f"{sum(len(r['findings']) for r in recs) / n:.2f}", f"{claims / n:.1f}",
                     f"{claims - sup}", f"{100 * mean('cost_usd'):.2f}"])
    _tex_table("behaviour", "Audit behaviour per scenario (means over the 125 evaluation audits). "
               "``Unsupp.'' is the total number of claims the fact-checker rejected; cost in US cents.",
               "tab:behaviour", ["Auditor", "Calls", "Turns", "Time (s)", r"$\varepsilon$", "Findings",
                                 "Claims", "Unsupp.", r"Cost (\textcent)"], rows, "lrrrrrrrr", wide=True)
    return rows


# ------------------------------------------------------------------ main

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    grids, common = load_grids()
    fresh = compare(DATASET, MODEL, LABELS, CONFS)
    res = check_against_json(fresh)
    rows = main_table(res, res["n_common"])
    recall_heatmap(res)
    type_confusion(grids, common, res)
    ex = worked_example()
    pooled_auc_table()
    rules_models_table()
    behaviour_table(grids, common)
    print(f"{res['n_common']} common scenarios; numbers match compare_{DATASET}_{MODEL}.json")
    for r in rows:
        print("  " + " | ".join(r))
    print(f"Worked example {EXAMPLE}: {ex['fact_check']['supported']}/{ex['fact_check']['claims']} claims")
    print(f"Wrote {os.path.relpath(OUT)}: " + ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
