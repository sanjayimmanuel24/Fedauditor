// FedAuditor guide-review deck (12 slides). Run: node build.js <outPath>
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");
const { applyTheme } = require(process.env.PPTX_SKILL + "/scripts/apply_theme.js");

const OUT = process.argv[2];
const IMG = process.argv[3];          // folder with dashboard screenshots

const THEME = {
  name: "FedAuditor",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2A30", lt1: "FFFFFF", dk2: "0B3C49", lt2: "EAF3F4",
    accent1: "1F7A8C", accent2: "F2A541", accent3: "5FA8A0", accent4: "C8553D",
    accent5: "8AA1A8", accent6: "2E5E6B", hlink: "1F7A8C", folHlink: "5FA8A0",
  },
};
const K = THEME.colors;
const MUTED = "5B6B70";

async function icon(Comp, color = "FFFFFF") {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + color, size: 256 }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

// teal circle with a white icon
async function iconCircle(slide, Comp, x, y, d = 0.6, fill = K.accent1, name = "icon") {
  slide.addShape("ellipse", { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill }, objectName: name + "-circle" });
  const p = d * 0.24;
  slide.addImage({ data: await icon(Comp), x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p, objectName: name, altText: name });
}

function shadow() { return { type: "outer", color: "0B3C49", opacity: 0.12, blur: 6, offset: 2, angle: 90 }; }

function card(slide, x, y, w, h, name, fill = "FFFFFF") {
  slide.addShape("roundRect", { x, y, w, h, rectRadius: 0.08, fill: { color: fill },
    line: { color: "D5E4E6", width: 0.75 }, shadow: shadow(), objectName: name });
}

function txt(slide, text, opts) {
  slide.addText(text, { isTextBox: true, fontFace: "Calibri", color: K.dk1, valign: "top", margin: 0, ...opts });
}

function bullets(items, size = 15, color = K.dk1) {
  return items.map((t, i) => ({ text: t, options: { bullet: { indent: 16 }, fontSize: size, color,
    paraSpaceAfter: 6, breakLine: i < items.length - 1 } }));
}

function heatHex(v) {   // light tint -> deep teal
  const a = [0xE3, 0xF1, 0xF2], b = [0x0B, 0x3C, 0x49];
  const t = Math.max(0, Math.min(1, v));
  return a.map((x, i) => Math.round(x + (b[i] - x) * t).toString(16).padStart(2, "0")).join("").toUpperCase();
}

(async () => {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE";                     // 13.333 x 7.5
  pres.title = "FedAuditor - Guide Review";
  pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };

  // ---------- layouts ----------
  pres.defineSlideMaster({
    title: "TITLE_DARK", background: { color: K.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.0, w: 11.7, h: 1.9,
        fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: K.lt1, valign: "bottom", align: "left" }, text: "" } },
      { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.05, w: 11.7, h: 0.9,
        fontSize: 20, color: "CFE3E6", valign: "top" }, text: "" } },
    ],
  });
  pres.defineSlideMaster({
    title: "CONTENT", background: { color: "FFFFFF" },
    margin: [0.5, 0.6, 0.6, 0.6],
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85,
        fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: K.dk2, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "FedAuditor  |  Guide Review", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: MUTED, margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: MUTED, align: "right" },
  });
  pres.defineSlideMaster({
    title: "CONTENT_DARK", background: { color: K.dk2 },
    objects: [
      { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 0.3, w: 12.1, h: 0.85,
        fontFace: THEME.headFontFace, fontSize: 32, bold: true, color: K.lt1, valign: "middle", margin: 0 }, text: "" } },
      { text: { text: "FedAuditor  |  Guide Review", options: { x: 0.6, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: "9FC0C6", margin: 0 } } },
    ],
    slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: "9FC0C6", align: "right" },
  });

  const content = (title, section, dark = false) => {
    const s = pres.addSlide({ masterName: dark ? "CONTENT_DARK" : "CONTENT", sectionTitle: section });
    s.addText(title, { placeholder: "title" });
    return s;
  };

  // =============== 1. Title ===============
  pres.addSection({ title: "Introduction" });
  let s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Introduction" });
  s.addText("FedAuditor: Auditing Federated Clinical Models with a Privacy-Budgeted, Fact-Checked AI Agent", { placeholder: "title" });
  s.addText("Final Year Project  -  Guide Review", { placeholder: "body" });
  await iconCircle(s, fa.FaShieldAlt, 0.8, 0.8, 0.9, K.accent1, "logo-icon");
  txt(s, [
    { text: "Presented by: ", options: { bold: true } }, { text: "Student Name (Reg. No.), Student Name (Reg. No.)", options: { breakLine: true } },
    { text: "Guide: ", options: { bold: true } }, { text: "Guide Name, Designation", options: { breakLine: true } },
    { text: "Department of Computer Science and Engineering, Institution Name" },
  ], { x: 0.8, y: 5.4, w: 11.7, h: 1.2, fontSize: 16, color: "DDEBED", paraSpaceAfter: 4 });
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
  for (let i = 0; i < agenda.length; i++) {
    const col = i < 5 ? 0 : 1, row = i < 5 ? i : i - 5;
    const x = 0.9 + col * 6.2, y = 1.55 + row * 1.03;
    card(s, x, y, 5.7, 0.82, "agenda-card-" + (i + 1), K.lt2);
    await iconCircle(s, agenda[i][0], x + 0.18, y + 0.13, 0.56, K.accent1, "agenda-icon-" + (i + 1));
    txt(s, [{ text: String(i + 1).padStart(2, "0") + "   ", options: { bold: true, color: K.accent1 } },
            { text: agenda[i][1], options: { color: K.dk2 } }],
      { x: x + 0.95, y: y + 0.1, w: 4.6, h: 0.62, fontSize: 18, valign: "middle" });
  }
  s.addNotes("Walk through the nine parts of the review.");

  // =============== 3. Abstract ===============
  pres.addSection({ title: "Background" });
  s = content("Abstract", "Background");
  txt(s, [
    { text: "Federated learning lets hospitals train one shared model without sharing patient records. The price is that nobody can look at the data to check whether a site is faulty, and pooled accuracy hides most faults.", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "FedAuditor is an AI agent that audits a federated model using only noisy, privacy-budgeted summary statistics. It never sees a patient record, and every number in its report is checked against the tool log by an automatic fact-checker.", options: { breakLine: true, paraSpaceAfter: 10 } },
    { text: "We built FedFault, a benchmark that plants 8 kinds of fault in a 52-state federation built from the CDC BRFSS 2023 diabetes survey, and compared the agent with a rule-based auditor on 125 test scenarios." },
  ], { x: 0.6, y: 1.45, w: 6.6, h: 3.7, fontSize: 16 });
  const flow = [[fa.FaLock, "Noisy, budgeted tools"], [fa.FaRobot, "AI agent audits"], [fa.FaCheckDouble, "Fact-checked report"]];
  for (let i = 0; i < flow.length; i++) {
    const x = 0.6 + i * 2.25;
    card(s, x, 5.45, 1.95, 1.3, "abstract-flow-" + i, K.lt2);
    await iconCircle(s, flow[i][0], x + 0.7, 5.57, 0.55, K.accent1, "abstract-flow-icon-" + i);
    txt(s, flow[i][1], { x: x + 0.08, y: 6.2, w: 1.79, h: 0.45, fontSize: 13, bold: true, color: K.dk2, align: "center" });
    if (i < flow.length - 1)
      s.addShape("rightArrow", { x: x + 1.98, y: 5.97, w: 0.24, h: 0.26, fill: { color: K.accent5 }, line: { color: K.accent5 }, objectName: "abstract-arrow-" + i });
  }
  const stats = [["52", "states as clients"], ["290,737", "survey records"], ["8 x 3 x 5", "faults x severities x seeds"],
                 ["0.97 / 0.89", "precision / recall"], ["99.9%", "claims verified"], ["US$1.16", "for all 125 audits"]];
  for (let i = 0; i < stats.length; i++) {
    const x = 7.65 + (i % 2) * 2.6, y = 1.45 + Math.floor(i / 2) * 1.75;
    card(s, x, y, 2.4, 1.55, "stat-card-" + (i + 1));
    txt(s, stats[i][0], { x: x + 0.1, y: y + 0.22, w: 2.2, h: 0.7, fontSize: 28, bold: true, color: K.accent1, align: "center", fontFace: "Cambria" });
    txt(s, stats[i][1], { x: x + 0.1, y: y + 0.95, w: 2.2, h: 0.45, fontSize: 13, color: MUTED, align: "center" });
  }
  s.addNotes("Summarise the whole project in three sentences, then point at the key numbers on the right.");

  // =============== 4. Problem, scope, objectives ===============
  s = content("Problem Definition, Scope and Objectives", "Background");
  card(s, 0.6, 1.4, 4.4, 5.35, "problem-card", K.lt2);
  await iconCircle(s, fa.FaExclamationTriangle, 0.85, 1.6, 0.6, K.accent4, "problem-icon");
  txt(s, "Problem", { x: 1.6, y: 1.65, w: 3.2, h: 0.5, fontSize: 20, bold: true, color: K.dk2, fontFace: "Cambria", valign: "middle" });
  txt(s, "A faulty site changes what the shared model learns, but the coordinator cannot see the data. Pooled test AUC barely moves:", { x: 0.85, y: 2.35, w: 3.95, h: 1.15, fontSize: 14 });
  s.addChart(pres.charts.BAR, [{ name: "Drop in pooled AUC", labels: ["Label noise", "Drift", "Poisoning", "Free-rider", "Prev. shift", "Subgroup bias", "Missing data", "Expl. instab."],
      values: [0.000, 0.001, 0.184, 0.000, 0.000, 0.001, 0.000, 0.000] }], {
    x: 0.7, y: 3.45, w: 4.2, h: 2.75, barDir: "bar", chartColors: [K.accent4],
    showValue: true, dataLabelFormatCode: "0.000", dataLabelFontSize: 9, dataLabelColor: K.dk1, dataLabelPosition: "outEnd",
    catAxisLabelFontSize: 9, valAxisHidden: true, catAxisLabelColor: K.dk1, catAxisLabelFontFace: "+mn-lt", dataLabelFontFace: "+mn-lt",
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: false, catAxisOrientation: "maxMin",
    showTitle: true, title: "Drop in pooled AUC (high severity, LR)", titleFontSize: 10, titleColor: K.dk2, titleFontFace: "+mn-lt", valAxisMaxVal: 0.25,
  });
  txt(s, "Only strong poisoning is visible. Normal seed-to-seed variation is 0.008.", { x: 0.85, y: 6.2, w: 3.95, h: 0.45, fontSize: 11, italic: true, color: MUTED });

  const cols = [
    [fa.FaCrosshairs, "Scope", [
      "Federated diabetes screening on CDC BRFSS 2023; 52 US states and territories as clients",
      "Three model types: logistic regression, neural network (MLP), XGBoost",
      "8 fault types at 3 severities; clean runs to measure false alarms",
      "Auditor sees only noisy aggregates under a fixed privacy budget (epsilon = 20)",
    ]],
    [fa.FaBullseye, "Objectives", [
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
    await iconCircle(s, cols[i][0], x + 0.25, 1.6, 0.6, K.accent1, "col-icon-" + i);
    txt(s, cols[i][1], { x: x + 1.0, y: 1.65, w: 2.4, h: 0.5, fontSize: 20, bold: true, color: K.dk2, fontFace: "Cambria", valign: "middle" });
    txt(s, bullets(cols[i][2], 14), { x: x + 0.25, y: 2.4, w: 3.1, h: 4.2 });
  }
  s.addNotes("The chart is the motivation: seven of eight faults are invisible in pooled AUC. Then read the scope and objectives.");

  // =============== 5. Literature survey ===============
  s = content("Literature Survey", "Background");
  const hdr = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: K.dk2 }, fontSize: 13 } });
  const lit = [
    ["McMahan et al., 2017", "FedAvg: train one model across sites without sharing data", "No way to check data quality at the sites"],
    ["Blanchard et al., 2017; Yin et al., 2018", "Robust aggregation (Krum, median, trimmed mean)", "Silently down-weights bad updates; no explanation; misses data-level faults"],
    ["Fraboni et al., 2021", "Free-rider attacks on FL", "Covers one fault type only"],
    ["Saha et al., 2025 (FedAgentBench)", "LLM agents set up federated medical imaging pipelines", "Agents run training; they do not audit it"],
    ["Jarczewski et al., 2026 (Agentic-FL)", "Agents as orchestrators of federated training", "Position paper; no auditing or benchmark"],
    ["Wu et al., 2026 (FL-MAESTRO)", "Multi-agent LLM picks topology, aggregation, resources", "Optimises training cost, not model soundness"],
    ["Kuhn and Buettner, 2025 (ModelAuditor)", "Agent audits a clinical imaging model", "Needs direct data access; not federated"],
  ];
  const litRows = [[hdr("Work"), hdr("What it does"), hdr("Gap")]].concat(lit.map((r, i) => r.map((c, j) => ({
    text: c, options: { fontSize: 12, color: K.dk1, bold: j === 0, fill: { color: i % 2 ? "FFFFFF" : K.lt2 } } }))));
  s.addTable(litRows, { x: 0.6, y: 1.4, w: 8.4, colW: [2.6, 3.0, 2.8], border: { type: "solid", pt: 0.5, color: "D5E4E6" },
    rowH: 0.66, valign: "middle", fontFace: "Calibri", margin: [3, 6, 3, 6], objectName: "literature-table" });
  card(s, 9.35, 1.4, 3.4, 5.35, "gap-card", K.dk2);
  await iconCircle(s, fa.FaSearch, 9.6, 1.65, 0.6, K.accent2, "gap-icon");
  txt(s, "Research gap", { x: 10.35, y: 1.7, w: 2.3, h: 0.5, fontSize: 20, bold: true, color: "FFFFFF", fontFace: "Cambria", valign: "middle" });
  txt(s, bullets([
    "Existing agent systems run federated training; none of them audits the trained model",
    "Robust aggregation hides faults instead of reporting them",
    "No benchmark with known, planted faults to measure an auditor",
    "LLM reports can contain invented numbers",
  ], 14, "FFFFFF"), { x: 9.6, y: 2.5, w: 2.95, h: 4.1 });
  s.addNotes("Two groups of prior work: defences inside FL, and LLM agents for FL. The agents orchestrate training; ModelAuditor audits but needs data access. FedAuditor fills the gap.");

  // =============== 6. Architecture ===============
  pres.addSection({ title: "System" });
  s = content("Proposed System Architecture", "System");
  const boxes = [
    ["52 clients", "local patient data stays at each site", K.accent6, fa.FaHospital],
    ["8 audit tools", "statistics computed locally + Laplace noise", K.accent1, fa.FaTools],
    ["Privacy accountant", "charges epsilon per call; refuses at 20", K.accent2, fa.FaLock],
    ["AI agent", "plans queries, compares clients, writes report", K.accent1, fa.FaRobot],
    ["Fact-checker", "matches every claim to the logged result", K.accent4, fa.FaCheckDouble],
    ["Verified report", "dashboard links each claim to its source", K.accent6, fa.FaDesktop],
  ];
  const bw = 1.85, gap = 0.22, by = 1.55;
  for (let i = 0; i < boxes.length; i++) {
    const x = 0.6 + i * (bw + gap);
    card(s, x, by, bw, 2.2, "arch-box-" + (i + 1));
    await iconCircle(s, boxes[i][3], x + bw / 2 - 0.32, by + 0.18, 0.64, boxes[i][2], "arch-icon-" + (i + 1));
    txt(s, boxes[i][0], { x: x + 0.08, y: by + 0.92, w: bw - 0.16, h: 0.4, fontSize: 15, bold: true, color: K.dk2, align: "center" });
    txt(s, boxes[i][1], { x: x + 0.1, y: by + 1.33, w: bw - 0.2, h: 0.8, fontSize: 11.5, color: MUTED, align: "center" });
    if (i < boxes.length - 1)
      s.addShape("rightArrow", { x: x + bw + 0.02, y: by + 0.95, w: gap - 0.04, h: 0.3, fill: { color: K.accent5 }, line: { color: K.accent5 }, objectName: "arch-arrow-" + (i + 1) });
  }
  txt(s, "The agent can only ask questions through the tools; every call and its noisy answer is logged as c1, c2, ...", { x: 0.6, y: 3.9, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: MUTED, align: "center" });
  const toolHdr = [hdr("Tool"), hdr("What it returns (per client, with noise)"), hdr("Faults it reveals")];
  const tools = [
    ["client_performance", "AUC, calibration, prevalence, mean prediction", "label noise, prevalence shift"],
    ["client_updates", "update size and direction vs. the other clients", "poisoning, free-riding"],
    ["feature_summary / drift_by_round", "feature means and percentiles; shift and onset round", "drift, missing-data artefact"],
    ["subgroup_outcomes / attribution_churn", "observed vs. expected outcomes per subgroup; explanation change", "subgroup bias, explanation instability"],
  ];
  s.addTable([toolHdr].concat(tools.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 12, color: K.dk1,
    bold: j === 0, fill: { color: i % 2 ? "FFFFFF" : K.lt2 } } })))), {
    x: 0.6, y: 4.45, w: 12.1, colW: [3.4, 5.1, 3.6], rowH: 0.42, border: { type: "solid", pt: 0.5, color: "D5E4E6" },
    valign: "middle", fontFace: "Calibri", margin: [3, 6, 3, 6], objectName: "tools-table" });
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
    ["app.py, templates/", "Flask dashboard with the FedAuditor report page"],
  ];
  s.addTable([[hdr("Module"), hdr("Role")]].concat(mods.map((r, i) => r.map((c, j) => ({ text: c, options: {
    fontSize: 12.5, color: K.dk1, bold: j === 0, fontFace: j === 0 ? "Consolas" : "Calibri", fill: { color: i % 2 ? "FFFFFF" : K.lt2 } } })))), {
    x: 0.6, y: 1.4, w: 7.0, colW: [2.75, 4.25], rowH: 0.62, border: { type: "solid", pt: 0.5, color: "D5E4E6" },
    valign: "middle", fontFace: "Calibri", margin: [3, 6, 3, 6], objectName: "modules-table" });
  txt(s, "Technology stack", { x: 7.95, y: 1.4, w: 4.8, h: 0.45, fontSize: 18, bold: true, color: K.dk2, fontFace: "Cambria" });
  const stack = [[fa.FaPython, "Python 3.12, NumPy"], [fa.FaNetworkWired, "Flower 1.43 (FL)"], [fa.FaTree, "XGBoost 3.4"],
                 [fa.FaRobot, "Claude Haiku 5.5 API"], [fa.FaGlobe, "Flask web app"], [fa.FaVial, "unittest: 78 tests"]];
  for (let i = 0; i < stack.length; i++) {
    const x = 7.95 + (i % 2) * 2.45, y = 1.95 + Math.floor(i / 2) * 0.72;
    card(s, x, y, 2.3, 0.58, "stack-chip-" + i, K.lt2);
    await iconCircle(s, stack[i][0], x + 0.1, y + 0.08, 0.42, K.accent1, "stack-icon-" + i);
    txt(s, stack[i][1], { x: x + 0.62, y: y + 0.06, w: 1.62, h: 0.46, fontSize: 12.5, color: K.dk2, valign: "middle" });
  }
  card(s, 7.95, 4.25, 4.75, 2.5, "fedfault-card", K.dk2);
  txt(s, "FedFault benchmark", { x: 8.2, y: 4.4, w: 4.3, h: 0.4, fontSize: 16, bold: true, color: "FFFFFF", fontFace: "Cambria" });
  txt(s, bullets([
    "8 faults x 3 severities x 5 test seeds + 5 clean runs = 125 scenarios per model",
    "375 federated runs (LR, MLP, XGBoost), 20 rounds each",
    "Seed 10 kept for development only, so tuning never sees the test set",
  ], 13.5, "FFFFFF"), { x: 8.2, y: 4.9, w: 4.3, h: 1.8 });
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
  const mh = ["Auditor", "Precision", "Recall", "Type acc.", "Onset err.", "False alarms", "Claims verified", "Cost (US$)"]
    .map((t, j) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: K.dk2 }, fontSize: 13, align: j ? "center" : "left" } }));
  s.addTable([mh].concat(main.map((r, i) => r.map((c, j) => ({ text: c, options: { fontSize: 13, color: K.dk1,
    bold: j === 0 || i === 2, align: j ? "center" : "left", fill: { color: i === 2 ? "FCE9C8" : (i % 2 ? "FFFFFF" : K.lt2) } } })))), {
    x: 0.6, y: 1.4, w: 12.1, colW: [3.0, 1.2, 1.1, 1.2, 1.2, 1.4, 1.6, 1.4], rowH: 0.45, border: { type: "solid", pt: 0.5, color: "D5E4E6" },
    valign: "middle", fontFace: "Calibri", margin: [3, 6, 3, 6], objectName: "main-results-table" });
  txt(s, "125 test scenarios, logistic regression, same tools and privacy budget for every auditor.", { x: 0.6, y: 3.75, w: 12.1, h: 0.3, fontSize: 11, italic: true, color: MUTED });
  s.addChart(pres.charts.BAR, [
    { name: "Rule baseline", labels: ["Precision", "Recall", "Type accuracy"], values: [0.99, 0.88, 0.87] },
    { name: "Haiku 5.5 (prompt v5)", labels: ["Precision", "Recall", "Type accuracy"], values: [0.97, 0.89, 0.91] },
  ], { x: 0.6, y: 4.1, w: 6.4, h: 2.75, barDir: "col", barGapWidthPct: 60, chartColors: [K.accent5, K.accent1],
    showValue: true, dataLabelFormatCode: "0.00", dataLabelFontSize: 10, dataLabelColor: K.dk1, dataLabelPosition: "outEnd",
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt",
    valAxisHidden: true, valAxisMinVal: 0.5, valAxisMaxVal: 1.08, catAxisLabelColor: K.dk1, catAxisLabelFontSize: 11,
    valGridLine: { style: "none" }, catGridLine: { style: "none" }, showLegend: true, legendPos: "t", legendFontSize: 11 });
  card(s, 7.35, 4.15, 5.35, 2.65, "takeaway-card", K.lt2);
  await iconCircle(s, fa.FaLightbulb, 7.55, 4.35, 0.55, K.accent2, "takeaway-icon");
  txt(s, "What this shows", { x: 8.25, y: 4.4, w: 4.3, h: 0.45, fontSize: 17, bold: true, color: K.dk2, fontFace: "Cambria", valign: "middle" });
  txt(s, bullets([
    "The agent finds faulty clients as well as tuned rules, and names the fault more often (0.91 vs 0.87)",
    "It dates the fault more precisely (0.24 vs 0.35 rounds)",
    "Both false alarms: Guam, the smallest client (confidence 0.5)",
  ], 13.5), { x: 7.6, y: 5.0, w: 4.95, h: 1.75 });
  s.addNotes("Highlighted row is the final agent. Explain each column briefly. Cost: under one cent per audit.");

  // =============== 9. Results 2 ===============
  s = content("Results and Discussion: Fault-wise Analysis", "Results");
  const faults = ["Label noise", "Drift", "Poisoning", "Free-rider", "Prevalence shift", "Subgroup bias", "Missing-data artefact", "Expl. instability"];
  const rec = [[1, 1, 1, 1, 1, 1], [0.87, 1, 1, 0.6, 1, 1], [1, 1, 1, 0.93, 1, 1], [1, 1, 1, 0.87, 1, 1],
               [0.8, 1, 1, 0.87, 1, 1], [0.13, 0.6, 0.93, 0.13, 1, 1], [1, 1, 1, 1, 1, 1], [0, 0.87, 1, 0, 1, 1]];
  const sub = (t) => ({ text: t, options: { bold: true, color: K.dk2, fill: { color: K.lt2 }, fontSize: 12, align: "center" } });
  const recRows = [
    [{ text: "Recall", options: { bold: true, color: "FFFFFF", fill: { color: K.dk2 }, fontSize: 13, rowspan: 2 } },
     { text: "Rule baseline", options: { bold: true, color: "FFFFFF", fill: { color: K.dk2 }, fontSize: 13, colspan: 3, align: "center" } },
     { text: "Haiku 5.5 (prompt v5)", options: { bold: true, color: "FFFFFF", fill: { color: K.dk2 }, fontSize: 13, colspan: 3, align: "center" } }],
    ["Low", "Med", "High", "Low", "Med", "High"].map(sub),
  ].concat(faults.map((f, i) => [{ text: f, options: { fontSize: 12.5, bold: true, color: K.dk1, fill: { color: "FFFFFF" } } }]
    .concat(rec[i].map((v) => ({ text: v.toFixed(2), options: { fontSize: 12.5, align: "center",
      color: v >= 0.5 ? "FFFFFF" : K.dk1, fill: { color: heatHex(v) } } })))));
  s.addTable(recRows, { x: 0.6, y: 1.4, w: 6.9, colW: [2.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8], rowH: 0.43,
    border: { type: "solid", pt: 1, color: "FFFFFF" }, valign: "middle", fontFace: "Calibri", margin: [2, 5, 2, 5], objectName: "recall-heatmap" });
  txt(s, "Share of planted clients found, 5 seeds per cell (darker = higher).", { x: 0.6, y: 6.35, w: 6.9, h: 0.3, fontSize: 11, italic: true, color: MUTED });
  const disc = [
    [fa.FaUsers, K.accent1, "Subgroup bias", "Agent finds 100% at medium severity; rules find 60%. It needs combining a subgroup query with the client screen."],
    [fa.FaMagic, K.accent2, "Prompt v5 fix", "Label noise was called poisoning in 12 of 15 cases. One rule (check local AUC first) fixed all 15."],
    [fa.FaWaveSquare, K.accent4, "Weak spots", "Faint drift (0.60 vs 0.87) and lowest severities, where the signal is below the privacy noise."],
    [fa.FaMicrochip, K.accent6, "Open models", "Qwen3 8B / 30B failed with the same tools: fixated on one state or read noise as faults."],
  ];
  for (let i = 0; i < disc.length; i++) {
    const y = 1.4 + i * 1.3;
    card(s, 7.85, y, 4.85, 1.15, "discussion-card-" + i);
    await iconCircle(s, disc[i][0], 8.0, y + 0.27, 0.6, disc[i][1], "discussion-icon-" + i);
    txt(s, disc[i][2], { x: 8.75, y: y + 0.1, w: 3.85, h: 0.32, fontSize: 15, bold: true, color: K.dk2 });
    txt(s, disc[i][3], { x: 8.75, y: y + 0.43, w: 3.85, h: 0.68, fontSize: 12, color: K.dk1 });
  }
  s.addNotes("Read the heatmap row by row. The agent's clear win is subgroup bias. The prompt change is the single biggest improvement. Lowest severities are limited by privacy noise for both auditors.");

  // =============== 10. Conclusion ===============
  pres.addSection({ title: "Closing" });
  s = content("Conclusion", "Closing", true);
  const concl = [
    [fa.FaCheckCircle, "An AI agent can audit a federated clinical model through noisy, budgeted statistics alone, without seeing patient data"],
    [fa.FaBalanceScale, "It matches a tuned rule baseline at finding faulty sites and is better at naming the fault and at subgroup bias"],
    [fa.FaCheckDouble, "The fact-checker makes reports trustworthy: 2,262 of 2,264 numbers verified; wrong numbers are caught"],
    [fa.FaCoins, "Practical cost: under one US cent per audit"],
  ];
  for (let i = 0; i < concl.length; i++) {
    const y = 1.4 + i * 1.0;
    await iconCircle(s, concl[i][0], 0.7, y, 0.62, K.accent1, "conclusion-icon-" + i);
    txt(s, concl[i][1], { x: 1.55, y: y - 0.02, w: 6.3, h: 0.9, fontSize: 16, color: "FFFFFF", valign: "middle" });
  }
  const cstats = [["0.91", "fault-type accuracy"], ["100%", "subgroup bias found (medium)"], ["< 1 cent", "cost per audit"]];
  for (let i = 0; i < cstats.length; i++) {
    const x = 0.7 + i * 2.45;
    card(s, x, 5.65, 2.25, 1.1, "conclusion-stat-" + i, "134B5A");
    txt(s, cstats[i][0], { x: x + 0.1, y: 5.72, w: 2.05, h: 0.55, fontSize: 24, bold: true, color: K.accent2, align: "center", fontFace: "Cambria" });
    txt(s, cstats[i][1], { x: x + 0.1, y: 6.28, w: 2.05, h: 0.4, fontSize: 12, color: "DDEBED", align: "center" });
  }
  card(s, 8.35, 1.4, 4.35, 5.35, "future-card", "134B5A");
  txt(s, "Future work", { x: 8.65, y: 1.6, w: 3.8, h: 0.5, fontSize: 20, bold: true, color: K.accent2, fontFace: "Cambria" });
  txt(s, bullets([
    "Formal differential-privacy guarantee for every tool",
    "Run the agent on the MLP and XGBoost federations",
    "Second dataset: eICU intensive-care records (journal version)",
    "Check the wording of reports, not only the numbers",
    "Submit the 6-page paper to an IEEE conference",
  ], 14.5, "FFFFFF"), { x: 8.65, y: 2.25, w: 3.8, h: 4.3 });
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
  await iconCircle(s, fa.FaBook, 12.05, 0.42, 0.6, K.accent1, "references-icon");
  for (let c = 0; c < 2; c++) {
    const part = refs.slice(c * 7, c * 7 + 7);
    txt(s, part.map((r, i) => ({ text: `[${c * 7 + i + 1}]  ` + r, options: { breakLine: i < part.length - 1, paraSpaceAfter: 10 } })),
      { x: 0.6 + c * 6.2, y: 1.45, w: 5.9, h: 5.35, fontSize: 13.5, color: K.dk1 });
  }
  s.addNotes("Key references. The full paper cites 29 sources.");

  // =============== 12. Outcome evidence ===============
  s = content("Outcome Specific Supporting Evidence", "Closing");
  const ev = [
    ["Working prototype", "Flask dashboard: FedAuditor report page, every claim linked to its logged tool call", "Done"],
    ["Benchmark", "FedFault: 375 federated runs with known, planted faults", "Done"],
    ["Experiments", "375 rule-based audits + 250 AI-agent audits, scored automatically", "Done"],
    ["Research paper", "6-page IEEE conference paper, 29 references, tables generated from results", "Ready"],
    ["Testing", "78 automated unit tests, all passing", "Done"],
    ["Documentation", "Step-by-step walkthrough PDF with live console output", "Done"],
  ];
  const evRows = [[hdr("Outcome"), hdr("Evidence"), hdr("Status")]].concat(ev.map((r, i) => [
    { text: r[0], options: { bold: true, fontSize: 12.5, color: K.dk1, fill: { color: i % 2 ? "FFFFFF" : K.lt2 } } },
    { text: r[1], options: { fontSize: 12, color: K.dk1, fill: { color: i % 2 ? "FFFFFF" : K.lt2 } } },
    { text: r[2], options: { bold: true, fontSize: 12.5, color: "FFFFFF", align: "center", fill: { color: r[2] === "Done" ? K.accent3 : K.accent2 } } },
  ]));
  s.addTable(evRows, { x: 0.6, y: 1.4, w: 6.9, colW: [1.75, 4.15, 1.0], rowH: 0.62, border: { type: "solid", pt: 0.5, color: "D5E4E6" },
    valign: "middle", fontFace: "Calibri", margin: [3, 6, 3, 6], objectName: "evidence-table" });
  const shotW = 4.95, shotH = shotW * 600 / 800;
  s.addImage({ path: path.join(IMG, "3_cited_row.jpg"), x: 7.8, y: 1.4, w: shotW, h: shotH, objectName: "dashboard-screenshot",
    altText: "FedAuditor dashboard: a clicked claim opens the cited tool call and highlights the logged row",
    rounding: false, shadow: shadow() });
  txt(s, "Dashboard: clicking claim c6 opens the cited tool call and highlights the exact logged row (MA, O/E ratio 0.0).",
    { x: 7.8, y: 1.5 + shotH, w: shotW, h: 0.6, fontSize: 11, italic: true, color: MUTED });
  card(s, 7.8, 5.95, shotW, 0.8, "demo-card", K.lt2);
  await iconCircle(s, fa.FaPlayCircle, 7.95, 6.06, 0.56, K.accent1, "demo-icon");
  txt(s, "Live demo: python -m fedauditor.walkthrough  (free, about 1 minute)", { x: 8.65, y: 6.05, w: 4.0, h: 0.6, fontSize: 12.5, color: K.dk2, valign: "middle" });
  s.addNotes("Show the evidence for each outcome. If time allows, run the walkthrough script or open the dashboard live.");

  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
