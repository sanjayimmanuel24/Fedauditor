// FedAuditor guide-review deck (12 slides) in the app's "deep ink" design.
// Run (pptxgenjs, react-icons, react, react-dom and sharp must be resolvable, e.g. via NODE_PATH):
//   node build_deck.js <out.pptx> <docs/design dir>
// The design dir holds ink_cover.png, ink_haze.png (make_ink_art.py) and screens/*.jpg.
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const OUT = process.argv[2];
const DESIGN = process.argv[3];
const SCREENS = path.join(DESIGN, "screens");

// Deep-ink tokens (same as static/css/style.css)
const C = {
  base: "04050C", fog: "EEF0F6", muted: "B9BECF", dim: "8A93AB",
  cyan: "67E8F9", blue: "7AA2FF", violet: "A78BFA", magenta: "F0ABFC",
  good: "4ADE80", risk: "FB7185", warn: "FBBF24",
  glassLine: "2A3045", tHead: "161B2C", tRowA: "0E1220", tRowB: "0A0D18", tHi: "10303A",
};
const THEME = {
  name: "FedAuditor Deep Ink",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: {
    dk1: C.base, lt1: "FFFFFF", dk2: "0E1220", lt2: C.fog,
    accent1: C.cyan, accent2: C.violet, accent3: C.magenta, accent4: C.risk,
    accent5: C.dim, accent6: C.good, hlink: C.cyan, folHlink: C.violet,
  },
};
const DARK_RAMP = ["141C30", "0D366B", "184F95", "256ABF", "3987E5", "6DA7EC", "9EC5F4", "CDE2FB"];

async function icon(Comp, color) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  return "image/png;base64," + (await sharp(Buffer.from(svg)).png().toBuffer()).toString("base64");
}
// Tinted glass circle with an accent-coloured icon
async function iconCircle(s, Comp, x, y, d = 0.6, color = C.cyan, name = "icon") {
  s.addShape("ellipse", { x, y, w: d, h: d, fill: { color, transparency: 82 }, line: { color, transparency: 55, width: 0.75 }, objectName: name + "-circle" });
  const p = d * 0.25;
  s.addImage({ data: await icon(Comp, color), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name, altText: name });
}
// Frosted glass card
function card(s, x, y, w, h, name, strong = false) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: 0.12, fill: { color: "FFFFFF", transparency: strong ? 90 : 94 },
    line: { color: "FFFFFF", transparency: 84, width: 0.75 }, objectName: name });
}
function pill(s, text, x, y, w, h, name, opts = {}) {
  s.addShape("roundRect", { x, y, w, h, rectRadius: h / 2, fill: { color: opts.fill || "FFFFFF" }, line: { color: opts.fill || "FFFFFF" }, objectName: name });
  s.addText(text, { x, y, w, h, isTextBox: true, margin: 0, align: "center", valign: "middle", fontFace: "Arial",
    fontSize: opts.size || 12, bold: true, color: opts.color || "2F2F33", objectName: name + "-text" });
}
function txt(s, text, o) {
  s.addText(text, { isTextBox: true, fontFace: "Arial", color: C.fog, valign: "top", margin: 0, ...o });
}
function bullets(items, size = 15, color = C.fog) {
  return items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, fontSize: size, color,
    paraSpaceAfter: 6, breakLine: i < items.length - 1 } }));
}
function heat(v) {
  const t = Math.max(0, Math.min(1, v)) * (DARK_RAMP.length - 1);
  const i = Math.min(Math.floor(t), DARK_RAMP.length - 2), f = t - i;
  const a = DARK_RAMP[i].match(/../g).map((h) => parseInt(h, 16)), b = DARK_RAMP[i + 1].match(/../g).map((h) => parseInt(h, 16));
  return a.map((x, k) => Math.round(x + (b[k] - x) * f).toString(16).padStart(2, "0")).join("").toUpperCase();
}
const hdr = (t, align = "left") => ({ text: t, options: { bold: true, color: C.fog, fill: { color: C.tHead }, fontSize: 13, align } });
const cell = (t, i, o = {}) => ({ text: t, options: { fontSize: 12, color: C.fog, fill: { color: i % 2 ? C.tRowB : C.tRowA }, ...o } });
const TABLE = { border: { type: "solid", pt: 0.75, color: C.glassLine }, valign: "middle", fontFace: "Arial", margin: [3, 7, 3, 7] };

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";                    // 13.333 x 7.5 in
  pres.title = "FedAuditor - Guide Review";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "COVER", background: { path: path.join(DESIGN, "ink_cover.png") },
    objects: [
      { rect: { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.base, transparency: 35 } } },
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.3, w: 9.6, h: 2.1,
        fontFace: "Arial", fontSize: 40, bold: true, color: C.fog, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.5, w: 9.6, h: 0.8,
        fontFace: "Arial", fontSize: 18, color: C.muted, valign: "top", align: "left" }, text: "" } },
    ],
  });
  const contentMaster = (title, closing) => pres.defineSlideMaster({
    title, background: { path: path.join(DESIGN, closing ? "ink_cover.png" : "ink_haze.png") },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      ...(closing ? [{ rect: { x: 0, y: 0, w: 13.333, h: 7.5, fill: { color: C.base, transparency: 22 } } }] : []),
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.32, w: 11.4, h: 0.8,
        fontFace: "Arial", fontSize: 30, bold: true, color: C.fog, valign: "middle", align: "left", margin: 0 }, text: "" } },
      { text: { text: "FedAuditor  ·  Guide Review", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontFace: "Arial", fontSize: 10, color: C.dim, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontFace: "Arial", fontSize: 10, color: C.dim, align: "right" },
  });
  contentMaster("CONTENT", false);
  contentMaster("CLOSING", true);

  const content = (title, section, closing = false) => {
    const s = pres.addSlide({ masterName: closing ? "CLOSING" : "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };

  // =============== 1. Title ===============
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "COVER", sectionTitle: "Introduction" });
  s.addText("FedAuditor: Auditing Federated Clinical Models with a Privacy-Budgeted, Fact-Checked AI Agent", { placeholder: "title" });
  s.addText("Final Year Project  ·  Guide Review", { placeholder: "body" });
  txt(s, [{ text: "FedHealth", options: { bold: true } }, { text: "  ·  FedAuditor", options: { color: C.muted } }],
    { x: 0.8, y: 0.6, w: 6, h: 0.4, fontSize: 16 });
  card(s, 0.8, 5.55, 7.6, 1.25, "presenter-card");
  txt(s, [
    { text: "Presented by: ", options: { bold: true } }, { text: "Student Name (Reg. No.), Student Name (Reg. No.)", options: { breakLine: true } },
    { text: "Guide: ", options: { bold: true } }, { text: "Guide Name, Designation", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, Institution Name", options: { color: C.muted } },
  ], { x: 1.05, y: 5.68, w: 7.2, h: 1.0, fontSize: 14, paraSpaceAfter: 3 });
  s.addNotes("Introduce the project title, team and guide. One line: we built an AI auditor that checks a federated diabetes model without ever seeing patient data.");

  // =============== 2. Agenda ===============
  s = content("Agenda", "Introduction");
  const agenda = [
    [fa.FaFileAlt, "Abstract"], [fa.FaBullseye, "Problem Definition, Scope and Objectives"],
    [fa.FaBook, "Literature Survey"], [fa.FaProjectDiagram, "Proposed System Architecture"],
    [fa.FaCode, "Implementation"], [fa.FaChartBar, "Results and Discussion"],
    [fa.FaFlagCheckered, "Conclusion"], [fa.FaListOl, "References"],
    [fa.FaAward, "Outcome Specific Supporting Evidence"],
  ];
  const band = [C.cyan, C.blue, C.violet, C.magenta];
  for (let i = 0; i < agenda.length; i++) {
    const col = i < 5 ? 0 : 1, row = i < 5 ? i : i - 5;
    const x = 0.6 + col * 6.2, y = 1.45 + row * 1.05;
    card(s, x, y, 5.9, 0.85, "agenda-card-" + (i + 1));
    await iconCircle(s, agenda[i][0], x + 0.18, y + 0.13, 0.58, band[i % 4], "agenda-icon-" + (i + 1));
    txt(s, [{ text: String(i + 1).padStart(2, "0") + "   ", options: { bold: true, color: band[i % 4] } },
            { text: agenda[i][1], options: { color: C.fog } }],
      { x: x + 0.98, y: y + 0.1, w: 4.8, h: 0.65, fontSize: 17, valign: "middle" });
  }
  s.addNotes("Walk through the nine parts of the review.");

  // =============== 3. Abstract ===============
  pres.addSection({ title: "Background" });
  s = content("Abstract", "Background");
  txt(s, [
    { text: "Federated learning lets hospitals train one shared model without sharing patient records. The price is that nobody can look at the data to check whether a site is faulty, and pooled accuracy hides most faults.", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "FedAuditor is an AI agent that audits a federated model using only noisy, privacy-budgeted summary statistics. It never sees a patient record, and every number in its report is checked against the tool log by an automatic fact-checker.", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "We built FedFault, a benchmark that plants 8 kinds of fault in a 52-state federation built from the CDC BRFSS 2023 diabetes survey, and compared the agent with a rule-based auditor on 125 test scenarios." },
  ], { x: 0.6, y: 1.45, w: 6.6, h: 3.7, fontSize: 15, color: C.muted });
  const flow = [[fa.FaLock, "Noisy, budgeted tools", C.cyan], [fa.FaRobot, "AI agent audits", C.violet], [fa.FaCheckDouble, "Fact-checked report", C.magenta]];
  for (let i = 0; i < flow.length; i++) {
    const x = 0.6 + i * 2.25;
    card(s, x, 5.45, 1.95, 1.3, "abstract-flow-" + i);
    await iconCircle(s, flow[i][0], x + 0.7, 5.57, 0.55, flow[i][2], "abstract-flow-icon-" + i);
    txt(s, flow[i][1], { x: x + 0.08, y: 6.2, w: 1.79, h: 0.45, fontSize: 12.5, bold: true, align: "center" });
    if (i < flow.length - 1)
      s.addShape("rightArrow", { x: x + 1.98, y: 5.97, w: 0.24, h: 0.26, fill: { color: C.dim }, line: { color: C.dim }, objectName: "abstract-arrow-" + i });
  }
  const stats = [["52", "states as clients", C.cyan], ["290,737", "survey records", C.blue], ["8 x 3 x 5", "faults x severities x seeds", C.violet],
                 ["0.97 / 0.89", "precision / recall", C.magenta], ["99.9%", "claims verified", C.good], ["US$1.16", "for all 125 audits", C.cyan]];
  for (let i = 0; i < stats.length; i++) {
    const x = 7.65 + (i % 2) * 2.6, y = 1.45 + Math.floor(i / 2) * 1.75;
    card(s, x, y, 2.4, 1.55, "stat-card-" + (i + 1));
    txt(s, stats[i][0], { x: x + 0.1, y: y + 0.24, w: 2.2, h: 0.7, fontSize: 26, bold: true, color: stats[i][2], align: "center" });
    txt(s, stats[i][1], { x: x + 0.1, y: y + 0.97, w: 2.2, h: 0.45, fontSize: 12, color: C.muted, align: "center" });
  }
  s.addNotes("Summarise the whole project in three sentences, then point at the key numbers on the right.");

  // =============== 4. Problem, scope, objectives ===============
  s = content("Problem Definition, Scope and Objectives", "Background");
  card(s, 0.6, 1.4, 4.4, 5.35, "problem-card", true);
  await iconCircle(s, fa.FaExclamationTriangle, 0.85, 1.6, 0.6, C.risk, "problem-icon");
  txt(s, "Problem", { x: 1.6, y: 1.65, w: 3.2, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
  txt(s, "A faulty site changes what the shared model learns, but the coordinator cannot see the data. Pooled test AUC barely moves:", { x: 0.85, y: 2.35, w: 3.95, h: 1.15, fontSize: 13.5, color: C.muted });
  s.addChart(pres.charts.BAR, [{ name: "Drop in pooled AUC", labels: ["Label noise", "Drift", "Poisoning", "Free-rider", "Prev. shift", "Subgroup bias", "Missing data", "Expl. instab."],
      values: [0.000, 0.001, 0.184, 0.000, 0.000, 0.001, 0.000, 0.000] }], {
    x: 0.7, y: 3.45, w: 4.2, h: 2.75, barDir: "bar", chartColors: [C.risk],
    showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 9, dataLabelColor: C.fog, dataLabelPosition: "outEnd",
    catAxisLabelFontSize: 9, valAxisHidden: true, catAxisLabelColor: C.muted, catAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    catAxisLineShow: false, valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
    showTitle: true, title: "Drop in pooled AUC (high severity, LR)", titleFontSize: 10, titleColor: C.fog, titleFontFace: "+mn-lt", valAxisMaxVal: 0.25,
  });
  txt(s, "Only strong poisoning is visible. Normal seed-to-seed variation is 0.008.", { x: 0.85, y: 6.2, w: 3.95, h: 0.45, fontSize: 11, italic: true, color: C.dim });
  const cols = [
    [fa.FaCrosshairs, "Scope", C.cyan, [
      "Federated diabetes screening on CDC BRFSS 2023; 52 US states and territories as clients",
      "Three model types: logistic regression, neural network (MLP), XGBoost",
      "8 fault types at 3 severities; clean runs to measure false alarms",
      "Auditor sees only noisy aggregates under a fixed privacy budget (epsilon = 20)",
    ]],
    [fa.FaBullseye, "Objectives", C.violet, [
      "Detect which clients are faulty and when the fault started",
      "Name the type of fault correctly",
      "Never expose patient records to the auditor",
      "Make every number in the report verifiable",
      "Compare an AI agent with a rule-based auditor fairly",
    ]],
  ];
  for (let i = 0; i < cols.length; i++) {
    const x = 5.3 + i * 3.8;
    card(s, x, 1.4, 3.55, 5.35, "col-card-" + i);
    await iconCircle(s, cols[i][0], x + 0.25, 1.6, 0.6, cols[i][2], "col-icon-" + i);
    txt(s, cols[i][1], { x: x + 1.0, y: 1.65, w: 2.4, h: 0.5, fontSize: 20, bold: true, valign: "middle" });
    txt(s, bullets(cols[i][3], 13.5, C.muted), { x: x + 0.25, y: 2.4, w: 3.1, h: 4.2 });
  }
  s.addNotes("The chart is the motivation: seven of eight faults are invisible in pooled AUC. Then read the scope and objectives.");

  // =============== 5. Literature survey ===============
  s = content("Literature Survey", "Background");
  const lit = [
    ["McMahan et al., 2017", "FedAvg: train one model across sites without sharing data", "No way to check data quality at the sites"],
    ["Blanchard et al., 2017; Yin et al., 2018", "Robust aggregation (Krum, median, trimmed mean)", "Silently down-weights bad updates; no explanation; misses data-level faults"],
    ["Fraboni et al., 2021", "Free-rider attacks on FL", "Covers one fault type only"],
    ["Saha et al., 2025 (FedAgentBench)", "LLM agents set up federated medical imaging pipelines", "Agents run training; they do not audit it"],
    ["Jarczewski et al., 2026 (Agentic-FL)", "Agents as orchestrators of federated training", "Position paper; no auditing or benchmark"],
    ["Wu et al., 2026 (FL-MAESTRO)", "Multi-agent LLM picks topology, aggregation, resources", "Optimises training cost, not model soundness"],
    ["Kuhn and Buettner, 2025 (ModelAuditor)", "Agent audits a clinical imaging model", "Needs direct data access; not federated"],
  ];
  s.addTable([[hdr("Work"), hdr("What it does"), hdr("Gap")]].concat(lit.map((r, i) => r.map((c, j) => cell(c, i, { bold: j === 0, color: j === 2 ? C.muted : C.fog })))),
    { x: 0.6, y: 1.4, w: 8.4, colW: [2.6, 3.0, 2.8], rowH: 0.66, ...TABLE, objectName: "literature-table" });
  card(s, 9.35, 1.4, 3.4, 5.35, "gap-card", true);
  await iconCircle(s, fa.FaSearch, 9.6, 1.65, 0.6, C.magenta, "gap-icon");
  txt(s, "Research gap", { x: 10.35, y: 1.7, w: 2.3, h: 0.5, fontSize: 19, bold: true, valign: "middle" });
  txt(s, bullets([
    "Existing agent systems run federated training; none of them audits the trained model",
    "Robust aggregation hides faults instead of reporting them",
    "No benchmark with known, planted faults to measure an auditor",
    "LLM reports can contain invented numbers",
  ], 13.5, C.muted), { x: 9.6, y: 2.5, w: 2.95, h: 4.1 });
  s.addNotes("Two groups of prior work: defences inside FL, and LLM agents for FL. The agents orchestrate training; ModelAuditor audits but needs data access. FedAuditor fills the gap.");

  // =============== 6. Architecture ===============
  pres.addSection({ title: "System" });
  s = content("Proposed System Architecture", "System");
  const boxes = [
    ["52 clients", "local patient data stays at each site", C.cyan, fa.FaHospital],
    ["8 audit tools", "statistics computed locally + Laplace noise", C.blue, fa.FaTools],
    ["Privacy budget", "accountant charges epsilon per call; refuses at 20", C.warn, fa.FaLock],
    ["AI agent", "plans queries, compares clients, writes report", C.violet, fa.FaRobot],
    ["Fact-checker", "matches every claim to the logged result", C.magenta, fa.FaCheckDouble],
    ["Verified report", "dashboard links each claim to its source", C.good, fa.FaDesktop],
  ];
  const bw = 1.85, gap = 0.22, by = 1.5;
  for (let i = 0; i < boxes.length; i++) {
    const x = 0.6 + i * (bw + gap);
    card(s, x, by, bw, 2.25, "arch-box-" + (i + 1));
    await iconCircle(s, boxes[i][3], x + bw / 2 - 0.32, by + 0.2, 0.64, boxes[i][2], "arch-icon-" + (i + 1));
    txt(s, boxes[i][0], { x: x + 0.08, y: by + 0.96, w: bw - 0.16, h: 0.4, fontSize: 14, bold: true, align: "center" });
    txt(s, boxes[i][1], { x: x + 0.1, y: by + 1.38, w: bw - 0.2, h: 0.8, fontSize: 11, color: C.muted, align: "center" });
    if (i < boxes.length - 1)
      s.addShape("rightArrow", { x: x + bw + 0.02, y: by + 0.98, w: gap - 0.04, h: 0.3, fill: { color: C.dim }, line: { color: C.dim }, objectName: "arch-arrow-" + (i + 1) });
  }
  txt(s, "The agent can only ask questions through the tools; every call and its noisy answer is logged as c1, c2, ...", { x: 0.6, y: 3.92, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: C.dim, align: "center" });
  const tools = [
    ["client_performance", "AUC, calibration, prevalence, mean prediction", "label noise, prevalence shift"],
    ["client_updates", "update size and direction vs. the other clients", "poisoning, free-riding"],
    ["feature_summary / drift_by_round", "feature means and percentiles; shift and onset round", "drift, missing-data artefact"],
    ["subgroup_outcomes / attribution_churn", "observed vs. expected outcomes per subgroup; explanation change", "subgroup bias, explanation instability"],
  ];
  s.addTable([[hdr("Tool"), hdr("What it returns (per client, with noise)"), hdr("Faults it reveals")]]
      .concat(tools.map((r, i) => r.map((c, j) => cell(c, i, { bold: j === 0, color: j === 0 ? C.cyan : (j === 2 ? C.muted : C.fog) })))),
    { x: 0.6, y: 4.45, w: 12.1, colW: [3.4, 5.1, 3.6], rowH: 0.42, ...TABLE, objectName: "tools-table" });
  s.addNotes("Read the flow left to right. Stress the two guarantees: the agent never sees records (only noisy tools under a budget), and nothing in the report is trusted unless the fact-checker finds it in the log.");

  // =============== 7. Implementation ===============
  s = content("Implementation", "System");
  const mods = [
    ["brfss.py, data.py", "Download and recode BRFSS; split each state into train / calibration / test"],
    ["federation.py, models.py", "Flower FedAvg loop; logistic regression, MLP, XGBoost"],
    ["faults.py", "FedFault: plants 8 faults with ground truth (clients, onset round)"],
    ["tools.py", "8 audit tools, Laplace noise, privacy accountant, call log"],
    ["agent.py, baseline.py", "Claude AI agent (tool-use loop) and rule-based auditor"],
    ["report.py", "Report schema, fact-checker, scoring against ground truth"],
    ["app.py, templates/, static/", "Flask web app: dark glass UI, live WebGL ink hero, FedAuditor page"],
  ];
  s.addTable([[hdr("Module"), hdr("Role")]].concat(mods.map((r, i) => [
      cell(r[0], i, { bold: true, color: C.cyan, fontFace: "Consolas", fontSize: 12 }), cell(r[1], i, { fontSize: 12.5 })])),
    { x: 0.6, y: 1.4, w: 7.0, colW: [2.75, 4.25], rowH: 0.62, ...TABLE, objectName: "modules-table" });
  txt(s, "Technology stack", { x: 7.95, y: 1.4, w: 4.8, h: 0.45, fontSize: 18, bold: true });
  const stack = [[fa.FaPython, "Python 3.12, NumPy", C.cyan], [fa.FaNetworkWired, "Flower 1.43 (FL)", C.blue], [fa.FaTree, "XGBoost 3.4", C.good],
                 [fa.FaRobot, "Claude Haiku 5.5 API", C.violet], [fa.FaGlobe, "Flask + WebGL UI", C.magenta], [fa.FaVial, "unittest: 78 tests", C.warn]];
  for (let i = 0; i < stack.length; i++) {
    const x = 7.95 + (i % 2) * 2.45, y = 1.95 + Math.floor(i / 2) * 0.72;
    card(s, x, y, 2.3, 0.58, "stack-chip-" + i);
    await iconCircle(s, stack[i][0], x + 0.1, y + 0.08, 0.42, stack[i][2], "stack-icon-" + i);
    txt(s, stack[i][1], { x: x + 0.62, y: y + 0.06, w: 1.62, h: 0.46, fontSize: 12, valign: "middle" });
  }
  card(s, 7.95, 4.25, 4.75, 2.5, "fedfault-card", true);
  txt(s, "FedFault benchmark", { x: 8.2, y: 4.4, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: C.cyan });
  txt(s, bullets([
    "8 faults x 3 severities x 5 test seeds + 5 clean runs = 125 scenarios per model",
    "375 federated runs (LR, MLP, XGBoost), 20 rounds each",
    "Seed 10 kept for development only, so tuning never sees the test set",
  ], 13, C.muted), { x: 8.2, y: 4.9, w: 4.3, h: 1.8 });
  s.addNotes("Each module maps to one block of the architecture. The existing FedHealth measurement modules were reused as the tools, not rewritten.");

  // =============== 8. Results 1 ===============
  pres.addSection({ title: "Results" });
  s = content("Results and Discussion: Overall Comparison", "Results");
  const main = [
    ["Rule baseline", "0.99", "0.88", "0.87", "0.35", "0 / 5", "100%", "0.00"],
    ["Haiku 5.5, prompt v4", "0.98", "0.88", "0.79", "0.33", "2 / 5", "99.6%", "1.11"],
    ["Haiku 5.5, prompt v5", "0.97", "0.89", "0.91", "0.24", "2 / 5", "99.9%", "1.16"],
    ["Haiku 5.5, v5, conf > 0.5", "1.00", "0.84", "0.88", "0.17", "0 / 5", "99.9%", "1.16"],
  ];
  const mh = ["Auditor", "Precision", "Recall", "Type acc.", "Onset err.", "False alarms", "Claims verified", "Cost (US$)"].map((t, j) => hdr(t, j ? "center" : "left"));
  s.addTable([mh].concat(main.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 13, align: j ? "center" : "left", bold: j === 0 || i === 2,
      color: i === 2 ? C.cyan : C.fog, fill: { color: i === 2 ? C.tHi : (i % 2 ? C.tRowB : C.tRowA) } } })))),
    { x: 0.6, y: 1.4, w: 12.1, colW: [3.0, 1.2, 1.1, 1.2, 1.2, 1.4, 1.6, 1.4], rowH: 0.45, ...TABLE, objectName: "main-results-table" });
  txt(s, "125 test scenarios, logistic regression, same tools and privacy budget for every auditor.", { x: 0.6, y: 3.75, w: 12.1, h: 0.3, fontSize: 11, italic: true, color: C.dim });
  card(s, 0.6, 4.1, 6.4, 2.75, "chart-card");
  s.addChart(pres.charts.BAR, [
    { name: "Rule baseline", labels: ["Precision", "Recall", "Type accuracy"], values: [0.99, 0.88, 0.87] },
    { name: "Haiku 5.5 (prompt v5)", labels: ["Precision", "Recall", "Type accuracy"], values: [0.97, 0.89, 0.91] },
  ], { x: 0.75, y: 4.2, w: 6.1, h: 2.55, barDir: "col", barGapWidthPct: 60, chartColors: [C.dim, C.cyan],
    showValue: true, dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: C.fog, dataLabelPosition: "outEnd",
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", legendColor: C.muted,
    valAxisHidden: true, valAxisMinVal: 0.5, valAxisMaxVal: 1.08, catAxisLabelColor: C.muted, catAxisLabelFontSize: 11, catAxisLineShow: false,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 11 });
  card(s, 7.35, 4.1, 5.35, 2.75, "takeaway-card", true);
  await iconCircle(s, fa.FaLightbulb, 7.55, 4.3, 0.55, C.warn, "takeaway-icon");
  txt(s, "What this shows", { x: 8.25, y: 4.35, w: 4.3, h: 0.45, fontSize: 17, bold: true, valign: "middle" });
  txt(s, bullets([
    "The agent finds faulty clients as well as tuned rules, and names the fault more often (0.91 vs 0.87)",
    "It dates the fault more precisely (0.24 vs 0.35 rounds)",
    "Both false alarms: Guam, the smallest client (confidence 0.5)",
  ], 13, C.muted), { x: 7.6, y: 4.95, w: 4.95, h: 1.8 });
  s.addNotes("Highlighted row is the final agent. Explain each column briefly. Cost: under one cent per audit.");

  // =============== 9. Results 2 ===============
  s = content("Results and Discussion: Fault-wise Analysis", "Results");
  const faults = ["Label noise", "Drift", "Poisoning", "Free-rider", "Prevalence shift", "Subgroup bias", "Missing-data artefact", "Expl. instability"];
  const rec = [[1, 1, 1, 1, 1, 1], [0.87, 1, 1, 0.6, 1, 1], [1, 1, 1, 0.93, 1, 1], [1, 1, 1, 0.87, 1, 1],
               [0.8, 1, 1, 0.87, 1, 1], [0.13, 0.6, 0.93, 0.13, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0.87, 1, 0, 1, 1]];
  const sub = (t) => ({ text: t, options: { bold: true, color: C.muted, fill: { color: C.tRowA }, fontSize: 12, align: "center" } });
  const recRows = [
    [{ text: "Recall", options: { bold: true, color: C.fog, fill: { color: C.tHead }, fontSize: 13, rowspan: 2 } },
     { text: "Rule baseline", options: { bold: true, color: C.fog, fill: { color: C.tHead }, fontSize: 13, colspan: 3, align: "center" } },
     { text: "Haiku 5.5 (prompt v5)", options: { bold: true, color: C.cyan, fill: { color: C.tHead }, fontSize: 13, colspan: 3, align: "center" } }],
    ["Low", "Med", "High", "Low", "Med", "High"].map(sub),
  ].concat(faults.map((f, i) => [{ text: f, options: { fontSize: 12.5, bold: true, color: C.fog, fill: { color: C.tRowB } } }]
    .concat(rec[i].map((v) => ({ text: v.toFixed(2), options: { fontSize: 12.5, align: "center", bold: true,
      color: v >= 0.55 ? C.base : C.fog, fill: { color: heat(v) } } })))));
  s.addTable(recRows, { x: 0.6, y: 1.4, w: 6.9, colW: [2.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], rowH: 0.43,
    border: { type: "solid", pt: 1.5, color: C.base }, valign: "middle", fontFace: "Arial", margin: [2, 5, 2, 5], objectName: "recall-heatmap" });
  txt(s, "Share of planted clients found, 5 seeds per cell (brighter = higher).", { x: 0.6, y: 6.35, w: 6.9, h: 0.3, fontSize: 11, italic: true, color: C.dim });
  const disc = [
    [fa.FaUsers, C.cyan, "Subgroup bias", "Agent finds 100% at medium severity; rules find 60%. It needs combining a subgroup query with the client screen."],
    [fa.FaMagic, C.violet, "Prompt v5 fix", "Label noise was called poisoning in 12 of 15 cases. One rule (check local AUC first) fixed all 15."],
    [fa.FaWaveSquare, C.risk, "Weak spots", "Faint drift (0.60 vs 0.87) and lowest severities, where the signal is below the privacy noise."],
    [fa.FaMicrochip, C.magenta, "Open models", "Qwen3 8B / 30B failed with the same tools: fixated on one state or read noise as faults."],
  ];
  for (let i = 0; i < disc.length; i++) {
    const y = 1.4 + i * 1.3;
    card(s, 7.85, y, 4.85, 1.15, "discussion-card-" + i);
    await iconCircle(s, disc[i][0], 8.0, y + 0.27, 0.6, disc[i][1], "discussion-icon-" + i);
    txt(s, disc[i][2], { x: 8.75, y: y + 0.1, w: 3.85, h: 0.32, fontSize: 15, bold: true });
    txt(s, disc[i][3], { x: 8.75, y: y + 0.43, w: 3.85, h: 0.68, fontSize: 11.5, color: C.muted });
  }
  s.addNotes("Read the heatmap row by row. The agent's clear win is subgroup bias. The prompt change is the single biggest improvement. Lowest severities are limited by privacy noise for both auditors.");

  // =============== 10. Conclusion ===============
  pres.addSection({ title: "Closing" });
  s = content("Conclusion", "Closing", true);
  const concl = [
    [fa.FaCheckCircle, "An AI agent can audit a federated clinical model through noisy, budgeted statistics alone, without seeing patient data", C.cyan],
    [fa.FaBalanceScale, "It matches a tuned rule baseline at finding faulty sites and is better at naming the fault and at subgroup bias", C.blue],
    [fa.FaCheckDouble, "The fact-checker makes reports trustworthy: 2,262 of 2,264 numbers verified; wrong numbers are caught", C.violet],
    [fa.FaCoins, "Practical cost: under one US cent per audit", C.magenta],
  ];
  for (let i = 0; i < concl.length; i++) {
    const y = 1.4 + i * 1.0;
    await iconCircle(s, concl[i][0], 0.7, y, 0.62, concl[i][2], "conclusion-icon-" + i);
    txt(s, concl[i][1], { x: 1.55, y: y - 0.02, w: 6.3, h: 0.9, fontSize: 16, valign: "middle" });
  }
  const cstats = [["0.91", "fault-type accuracy"], ["100%", "subgroup bias found (medium)"], ["< 1 cent", "cost per audit"]];
  for (let i = 0; i < cstats.length; i++) {
    const x = 0.7 + i * 2.45;
    card(s, x, 5.65, 2.25, 1.1, "conclusion-stat-" + i, true);
    txt(s, cstats[i][0], { x: x + 0.1, y: 5.74, w: 2.05, h: 0.55, fontSize: 24, bold: true, align: "center" });
    txt(s, cstats[i][1], { x: x + 0.1, y: 6.3, w: 2.05, h: 0.4, fontSize: 11.5, color: C.muted, align: "center" });
  }
  card(s, 8.35, 1.4, 4.35, 5.35, "future-card", true);
  txt(s, "Future work", { x: 8.65, y: 1.6, w: 3.8, h: 0.5, fontSize: 20, bold: true, color: C.cyan });
  txt(s, bullets([
    "Formal differential-privacy guarantee for every tool",
    "Run the agent on the MLP and XGBoost federations",
    "Second dataset: eICU intensive-care records (journal version)",
    "Check the wording of reports, not only the numbers",
    "Submit the 6-page paper to an IEEE conference",
  ], 14, C.fog), { x: 8.65, y: 2.25, w: 3.8, h: 4.3 });
  s.addNotes("Close with the four takeaways and the plan for the journal version.");

  // =============== 11. References ===============
  s = content("References", "Closing");
  const refs = [
    "B. McMahan et al., \"Communication-efficient learning of deep networks from decentralized data,\" AISTATS, 2017.",
    "P. Kairouz et al., \"Advances and open problems in federated learning,\" Found. Trends Mach. Learn., 2021.",
    "P. Blanchard et al., \"Machine learning with adversaries: Byzantine tolerant gradient descent,\" NeurIPS, 2017.",
    "D. Yin et al., \"Byzantine-robust distributed learning: Towards optimal statistical rates,\" ICML, 2018.",
    "Y. Fraboni et al., \"Free-rider attacks on model aggregation in federated learning,\" AISTATS, 2021.",
    "C. Dwork et al., \"Calibrating noise to sensitivity in private data analysis,\" TCC, 2006.",
    "P. Saha et al., \"FedAgentBench: Towards automating real-world federated medical image analysis with server-client LLM agents,\" arXiv:2509.23803, 2025.",
    "R. O. Jarczewski et al., \"Agentic federated learning: The future of distributed training orchestration,\" arXiv:2604.04895, 2026.",
    "J. Wu et al., \"FL-MAESTRO: Multi-agent LLM orchestration for resource-constrained federated learning,\" arXiv:2608.20518, 2026.",
    "L. Kuhn and F. Buettner, \"An autonomous agent for auditing and improving the reliability of clinical AI models,\" arXiv:2507.05755, 2025.",
    "S. Yao et al., \"ReAct: Synergizing reasoning and acting in language models,\" ICLR, 2023.",
    "Centers for Disease Control and Prevention, \"Behavioral Risk Factor Surveillance System survey data, 2023.\"",
    "D. J. Beutel et al., \"Flower: A friendly federated learning research framework,\" arXiv:2007.14390, 2020.",
    "Z. Obermeyer et al., \"Dissecting racial bias in an algorithm used to manage the health of populations,\" Science, 2019.",
  ];
  for (let c = 0; c < 2; c++) {
    const part = refs.slice(c * 7, c * 7 + 7);
    txt(s, part.map((r, i) => ([
      { text: `[${c * 7 + i + 1}]  `, options: { bold: true, color: C.cyan } },
      { text: r, options: { color: C.muted, breakLine: i < part.length - 1, paraSpaceAfter: 10 } }])).flat(),
      { x: 0.6 + c * 6.2, y: 1.45, w: 5.9, h: 5.35, fontSize: 13 });
  }
  s.addNotes("Key references. The full paper cites 29 sources.");

  // =============== 12. Outcome evidence ===============
  s = content("Outcome Specific Supporting Evidence", "Closing");
  const ev = [
    ["Working prototype", "Flask web app in a dark glass design; FedAuditor page links every claim to its logged tool call", "Done"],
    ["Benchmark", "FedFault: 375 federated runs with known, planted faults", "Done"],
    ["Experiments", "375 rule-based audits + 250 AI-agent audits, scored automatically", "Done"],
    ["Research paper", "6-page IEEE conference paper, 29 references, tables generated from results", "Ready"],
    ["Testing", "78 automated unit tests, all passing", "Done"],
    ["Documentation", "Step-by-step walkthrough PDF with live console output", "Done"],
  ];
  s.addTable([[hdr("Outcome"), hdr("Evidence"), hdr("Status", "center")]].concat(ev.map((r, i) => [
    cell(r[0], i, { bold: true, fontSize: 12.5 }), cell(r[1], i, { color: C.muted }),
    { text: r[2], options: { bold: true, fontSize: 12.5, align: "center", color: C.base, fill: { color: r[2] === "Done" ? C.good : C.warn } } },
  ])), { x: 0.6, y: 1.4, w: 6.9, colW: [1.75, 4.15, 1.0], rowH: 0.62, ...TABLE, objectName: "evidence-table" });
  const shotW = 5.0, shotH = shotW * 1800 / 2880;
  card(s, 7.72, 1.32, shotW + 0.16, shotH + 0.16, "screenshot-frame");
  s.addImage({ path: path.join(SCREENS, "3_cited_row.jpg"), x: 7.8, y: 1.4, w: shotW, h: shotH, objectName: "dashboard-screenshot",
    altText: "FedAuditor page in the dark design: a clicked claim opens the cited tool call and highlights the logged row" });
  txt(s, "FedAuditor page: clicking claim c6 opens the cited tool call and highlights the exact logged row (MA, O/E ratio 0.0).",
    { x: 7.8, y: 1.62 + shotH, w: shotW, h: 0.6, fontSize: 11, italic: true, color: C.dim });
  card(s, 7.8, 5.95, shotW, 0.8, "demo-card", true);
  await iconCircle(s, fa.FaPlayCircle, 7.95, 6.07, 0.56, C.cyan, "demo-icon");
  txt(s, [{ text: "Live demo  ", options: { bold: true } }, { text: "python -m fedauditor.walkthrough  (free, ~1 minute)", options: { color: C.muted } }],
    { x: 8.65, y: 6.05, w: 4.05, h: 0.6, fontSize: 12.5, valign: "middle" });
  s.addNotes("Show the evidence for each outcome. If time allows, run the walkthrough script or open the dashboard live.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
