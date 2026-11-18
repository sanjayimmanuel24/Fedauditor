# Titus project — context for Claude

## What this is
- `Final-Year/` is a friend's final-year project, cloned from github.com/godwin1057/Final-Year.
  It is FedHealth: federated diabetes screening with logistic regression, FedAvg over 3 simulated
  age-cohort clients, PIMA data, a Flask dashboard, and a measurement layer (calibration,
  fairness, explanation margin R and P(R>0), client Shapley values).
- The guide said it is "very basic". The goal is to rebuild it into **FedAuditor**, aiming for a
  conference paper and then a journal paper.
- The proposal is a Claude Doc: https://claude.ai/artifact/DkPF7AxrvHR7cWcnSVSbaV.
  Read it only when the task needs the details.

## FedAuditor plan (decided 2026-11-14)
- An LLM agent audits a federated clinical model. It may only use aggregate tools that add noise
  and spend a privacy budget, and it never sees patient records.
- **FedFault benchmark:** 8 planted fault types (label noise, drift, poisoning, free-rider,
  prevalence shift, subgroup bias, missing-data artefact, explanation instability).
  Each is planted at 3 severities over 5 seeds, plus clean runs for measuring false alarms.
- **Fact-checker:** every claim in the agent's report must match a logged tool result.
- **Data:** BRFSS with states as clients (main); eICU (journal version, needs PhysioNet
  credentialing); PIMA and UCI kept for continuity.
- **Stack:** Flower for federation; logistic regression, XGBoost and an MLP as models.
  The existing modules become the agent's tools: reuse them, don't rewrite them.
- **Phases:** 1 data and federation, 2 benchmark, 3 agent, tools and fact-checker,
  4 experiments, 5 paper and dashboard.
- **Closest prior work** to position against: FedAgentBench, Agentic-FL, FL-MAESTRO (these
  orchestrate training; none audits) and ModelAuditor (audits a model, but not federated).

## Status
- 2026-11-14: proposal written.
- 2026-11-14: Phase 1 code done in `Final-Year/fedauditor/` (not committed):
  `brfss.py` (download/recode, state = client), `data.py` (per-client train/cal/test,
  federated mean imputation + FederatedScaler), `models.py` (numpy LR/MLP, XGBoost tree
  bagging), `federation.py` (in-process loop over Flower strategies FedAvg/FedMedian/
  FedTrimmedAvg/FedXgbBagging; `update_hook` is the seam for Phase 2 faults),
  CLIs `prepare_brfss` and `run_baseline`. Added flwr 1.43 + xgboost 3.4.
- 2026-11-14: **Phase 1 done.** BRFSS 2023 cached (`data/brfss/`, gitignored): 291k complete
  cases, 52 state clients (smallest GU 1,182), prevalence 13.6%; all recode vars found
  (no fruit/veg in 2023). Clean baseline, 20 rounds, pooled AUC fed/central: LR .821/.821,
  XGB .826/.826, MLP .816/.828 (MLP still rising). Per-state AUC .76–.87.
- 2026-11-14: **Phase 2 code done** (not committed). XGB sped up 170 s → 18 s/run: cached
  per-array margins, clients start from the global margin (`start_round`), and
  `BatchedXgbBagging` (same output as Flower's FedXgbBagging, tested). `run_federation` gained
  `data_schedule` (drift) and `record_weights`; update norms now logged for XGB too.
  `fedauditor/faults.py`: `Scenario`, `plant()` → data/schedule/hook + ground truth (clients,
  onset round, details); 3 affected states (≈5%, from clients ≥ median size, same across
  severities). Runner `python -m fedauditor.run_fedfault --dataset brfss --models lr mlp xgb
  --workers 4` → resumable `results/fedauditor/fedfault_<ds>_<model>.jsonl`. 38 tests pass.
  Seed-20 LR check: pooled AUC hides every fault except high poisoning (the paper's point);
  target clients show local AUC/ECE or update-norm signals. Hard cases: poisoning-low
  (norm unchanged, direction flipped), missing_artefact (BMI→0, PhysHlth→99; visible
  only in feature stats), subgroup_bias (visible only per subgroup). explanation_instability
  = noisy uploads. `python -m fedauditor.attribution_check` (5 seeds, 20 rounds,
  `results/fedauditor/attribution_check_brfss.json`): AUC never moves (≤.001), but global
  attributions do, only weakly. Patient top-factor agreement with clean: LR 99.9/99.4/96.1%
  (low/med/high) vs 99.85% noise floor; XGB 98.1/95.3/87.4% vs 100%. The effect fades over
  rounds because noise is scaled to the update norm, which shrinks. Open decision:
  raise severities or scale noise to the weight norm.
  Next: check full-grid results, then Phase 3 (agent, tools, fact-checker).
- 2026-11-14: **Phase 3 code done** (not committed). `audit_env.py` (`AuditRun`: re-runs a
  scenario, records per-round globals, client update vectors (XGB: tree margins on synthetic
  probes), per-round data). `tools.py`: 8 tools (overview free; global/client performance,
  client_updates norm_ratio + cos to median update, feature_summary mean/sd/p01/p99 raw units,
  subgroup_outcomes O/E + FNR, drift_by_round shift/onset/CITL, attribution_churn); Laplace
  noise c/n·m/ε, `PrivacyAccountant` (default budget 20, refuses when spent, refunds bad calls),
  call log c1, c2…; outlier screening keeps outputs small. `report.py`: schema, fact-check
  (claim = call_id + metric + value + client/round/feature/group), `score`/`aggregate`.
  `agent.py`: `ClaudeAuditor` (manual loop, claude-opus-5-5, effort high, fallbacks default,
  key from `.env` `ANTHROPIC_API_KEY`), `ScriptedAuditor`, `audit()`. CLI
  `python -m fedauditor.run_audit --fault poisoning --severity low` → `results/fedauditor/audits/`.
  52 tests pass. BRFSS poisoning-low: the 3 planted states are the only ones with mean cos < 0.
- 2026-11-15: **Phase 3 done.** Live audit (BRFSS LR poisoning-low s20, key in `.env`): agent
  flagged MD/MI/MT as poisoning, onset 1 → precision/recall/type 1.0, 7/7 claims supported,
  15 calls, ε 20/20, 8 turns, ~$0.31. Next: Phase 4 (full grid of audits + rule baseline).
- 2026-11-17: **Phase 4 done** (not committed). Full log condensed here; details live in the code.
  - **Rule baseline** `fedauditor/baseline.py` `RuleAuditor`: same tools, ε budget and report as
    the agent; fixed query plan + noise-aware robust-z thresholds, tuned on dev seed 10 only
    (eval seeds 20–24). v1 had clean FAR .6/.8 (small states) → v2; v1 kept as `*_rules_v1.jsonl`.
  - **Tools v2.2** (`tools.py`): client tables carry `flags` ("mean_cos z=-8.2": noise-adjusted
    z = (v − median)/hypot(robust scale, client Laplace SD), listed if |z| > 3); update-norm flags
    are size-adjusted (norm grows with n: corr .75); feature/subgroup outlier rows carry `z_noise`
    and need |z_noise| > 3. Pure post-processing (no extra ε, same RNG draws); rules re-run on the
    new tools gave 375/375 identical findings. Fact-checker accepts cited flags as `<stat>_z`.
  - **Prompt v5** (`agent.py` SYSTEM, shared by all LLM agents): noise ∝ 1/n, screen all clients
    first, tool↔fault map, "telling faults apart" (check local AUC before calling poisoning),
    multiple-comparisons note (|z| 3–5 is a lead), ~3 ε per screen.
  - **Agents**: `ClaudeAuditor` (fallbacks only for FALLBACK_MODELS; Haiku has none);
    `openai_agent.py` `OpenAIAuditor` for open models (Ollama/vLLM; guard rails, transcripts);
    Kaggle path: `make_kaggle_bundle.py` + `notebooks/fedauditor_kaggle_qwen.ipynb` (built by
    `notebooks/build_kaggle_notebook.py`; Kaggle auto-extracts .gz, Ollama needs zstd).
  - **Runners**: `run_audit_grid.py` (resumable, `--agents rules claude openai`, `--label`,
    `--limit`, `--max-cost`, summary re-scores from findings); `compare_agents.py` (re-scores at
    confidence cut-offs → `results/fedauditor/compare_brfss_lr.json`). Scorer: tied fault types get
    1/k credit (old tie-break depended on string hashing).
  - **Main result, LR, 125 eval scenarios** (prec / rec / type / onset err / clean FAR / facts / $):
    rules .99/.88/.87/.35/0 of 5/100%/0; Haiku 5.5 prompt v4 .98/.88/.79/.33/2 of 5/100%/1.11;
    **Haiku 5.5 prompt v5 .97/.89/.91/.24/2 of 5/100%/1.16** (label `claude-haiku-5-5-p5`).
    Haiku > rules on subgroup_bias med/high (1.0/1.0 vs .60/.93) and explanation_instability
    med; < rules on drift low and free_rider low. v5 fixed label_noise→poisoning (45/45 vs 6/45).
    Both clean FPs: GU (smallest state) at conf exactly .5; conf > .5 → FAR 0, rec .84 (post-hoc
    cut: sensitivity only). Rules on MLP .99/.89, XGB 1.0/.84, FAR 0.
  - **Open models** (story for the paper, no grid): qwen3:8b anchored on the smallest state and
    never screened; qwen3:30b-a3b (thinking) 16–29 min/audit, read noise as faults;
    qwen3-30b-a3b-instruct-2507 (2–3 min/audit) caught poisoning but kept flagging |z| 3–4 states
    despite the prompt and mis-cited numbers. Same tools/prompt with Haiku worked → model limit.
  - **Not done (budget)**: LLM audits on MLP/XGB (Haiku ≈ $1.15 each; credit left ≈ $1.61);
    Qwen full grid; second dataset. Pilots on dev seed 10 are under `*-pilot*`/`*-dev-p5` labels.
- **Phase 5 (next, paper + dashboard)**: (1) figures/tables from `compare_brfss_lr.json` and the
  grids: main table, recall heatmap fault × severity (rules vs Haiku), typing confusion, cost,
  fact-check rate, worked example audit report; (2) paper draft (conference): FedFault benchmark,
  privacy-budgeted tools, fact-checker, results, open-model failure modes, limitations (LR-only
  LLM grid, single dataset, GU false alarms); (3) dashboard: a FedAuditor page in the Flask app
  (`app.py`, `templates/`) showing an audit report with each claim linked to its tool call.
  Optional if budget allows: Haiku on MLP for a second model row.
- 2026-11-17: **Phase 5 done (draft)** (not committed). Done: (1) `python -m fedauditor.make_figures` →
  `results/fedauditor/figures/` (main_table md/tex/csv, recall_heatmap + type_confusion svg/tex/csv,
  worked_example.md = subgroup_bias-high s20, 25/25 claims); asserts every number equals
  `compare_brfss_lr.json`; no matplotlib (SVG + xcolor LaTeX). Confusion: v4 called label noise
  poisoning in 12/15 scenarios, v5 in 0/15. (3) Dashboard: `/fedauditor` page (nav link,
  `templates/fedauditor.html`, view model `fedauditor/audit_view.py`): pick any saved audit; each
  claim links to the exact logged row it cites (same match rule as the fact-checker); figures shown
  below. `.claude/launch.json` runs the app for the browser preview. Tests `test_fedauditor_phase5.py`.
  (2) Paper: `Final-Year/paper/fedauditor.tex` + `refs.bib`, IEEEtran, **6 pages** (full), 29 refs
  (the 4 agentic-FL/auditor cites checked on arXiv: 2509.23803, 2604.04895, 2608.20518, 2507.05755).
  `make_figures` also writes pooled_auc / rules_models / behaviour tables (all \input by the paper).
  Build in `paper/`: pdflatex, bibtex, pdflatex x2 (latexmk needs Perl, absent). Key facts: claims
  verified 2,262/2,264 (v4 had 9 rejected, 2 real wrong numbers); 7/8 faults move pooled AUC <= .001
  vs .008 seed spread; agent wrote "women" for Sex=1 (1 = male) -> Discussion. Left: real author block.
- 2026-11-18: **Guide walkthrough** `Final-Year/docs/walkthrough/FedAuditor_Walkthrough.pdf` (11 pages, LaTeX
  + `img/` dashboard screenshots): phases 1-5 with live console output from
  `python -m fedauditor.walkthrough` (free; data → clean model → plant poisoning-low s20 → tools/ε →
  rule audit 3/3 claims, P/R 1.0 → saved Phase 4 table; outputs in `results/fedauditor/walkthrough/`).
- 2026-11-18: **Guide-review deck** `Final-Year/docs/presentation/FedAuditor_Guide_Review.pptx` (12 slides,
  user's agenda; generator `build_deck.js`, pptxgenjs, needs NODE_PATH to a node_modules with pptxgenjs).
  Title slide has placeholder names (student/guide/institution) for the user to fill.
- 2026-11-18: **Published** to the user's own PUBLIC repo https://github.com/sanjayimmanuel24/Fedauditor
  (remote `mine`; `origin` = friend's repo, never pushed). Commit author uses the GitHub noreply email.
  Excluded: .env, venv, users.db, data/brfss (re-download with prepare_brfss), LaTeX aux files.
  `Final-Year/CLAUDE.md` is a copy of this file for the other laptop; keep them in sync when committing.
- 2026-11-18: **Dark "deep ink" redesign** (from the user's Flowstate hero prompt, option a; not committed yet).
  Theme lives in `static/css/style.css` + Tailwind tokens in `templates/base.html` (same class names, so
  layouts unchanged): base #04050c, Onest, glass cards, white pill actions, staggered reveals (hidden tab
  panels excluded). Live WebGL fluid (`static/js/fluid.js`, MIT Dobryakov engine) only on landing
  (`landing.html`, full hero) and login/signup (`_fluid_backdrop.html`); canvas must be position:absolute.
  Chart series colours validated for CVD (dashboard COLORS). `make_figures` also writes `*_dark.svg`
  (used by /fedauditor).
- 2026-11-18: **Walkthrough PDF (14 pp) + deck restyled** in the deep-ink design (not committed yet). Shared
  assets in `Final-Year/docs/design/`: `make_ink_art.py` (numpy+Pillow) -> ink_cover/ink_haze.png;
  `screens/*.jpg` captured at 2x with headless Edge over CDP (the pane's screenshots crop/lag). PDF: helvet,
  dark page, glass tcolorboxes, `*_dark.tex` heatmaps (make_figures). Deck: `build_deck.js <out> docs/design`.

## How to run (from `Final-Year/`)
- Start the app: `.\venv\Scripts\python.exe app.py` → http://127.0.0.1:5000, demo login
  demo@fedhealth.local / demo12345.
- Run the tests: `.\venv\Scripts\python.exe -m unittest discover -s tests`. All 78 pass.
- BRFSS: `python -m fedauditor.prepare_brfss --year 2023` (once), then
  `python -m fedauditor.run_baseline --dataset brfss` (also `pima`, `uci_early`).
- `.env`, `venv/` and `users.db` are gitignored. Never commit or push to the friend's repo
  unless asked.

## Working style
- The user is token-conscious. Read only the files a task needs, search with grep before
  opening whole files, and keep outputs short.
- Keep this file up to date when a phase finishes or a decision changes. Keep it short, because
  it is loaded into every session.
