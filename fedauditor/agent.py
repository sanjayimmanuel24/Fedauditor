"""The FedAuditor agent: an LLM tool-use loop over the audit tools.

``ClaudeAuditor`` runs a manual tool-use loop with Claude (Messages API). It may
only call the aggregate tools in ``fedauditor.tools`` and finishes by calling
``submit_report`` with a structured report. ``ScriptedAuditor`` runs a fixed
Python policy over the same session, for tests and rule-based baselines.

``audit()`` runs either agent on one ``AuditRun`` and returns the report, the
tool log, the fact-check and the score against the planted truth.

The API key is read from ``ANTHROPIC_API_KEY`` in ``.env`` (gitignored) or the
environment; it is never printed or logged.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from fedauditor.audit_env import AuditRun
from fedauditor.faults import FAULTS
from fedauditor.report import REPORT_SCHEMA, fact_check, score, validate_report
from fedauditor.tools import DEFAULT_BUDGET, TOOL_SPECS, AuditSession

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = "claude-opus-5-5"
# Models that accept the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-2", "claude-fable-5-1")
DEFAULT_API = "https://api.anthropic.com"

FAULT_GUIDE = {
    "label_noise": "a share of the client's outcome labels are flipped",
    "drift": "from some round on, the client's feature distribution shifts",
    "poisoning": "the client uploads a manipulated (e.g. sign-flipped or scaled) update",
    "free_rider": "in some rounds the client uploads the global model back without training",
    "prevalence_shift": "the client's outcome prevalence differs strongly from what its features imply",
    "subgroup_bias": "positives in one patient subgroup are systematically recorded as negative "
                     "(under-diagnosis)",
    "missing_artefact": "missing values were exported as a sentinel code (e.g. 0 or 99) instead of missing",
    "explanation_instability": "the client's uploads carry large random noise, so the global "
                               "model's explanations jitter between rounds while accuracy barely moves",
}

SYSTEM = f"""You audit a federated clinical prediction model (diabetes risk) trained across \
many clients (sites). You cannot see patient records. You can only call aggregate tools; each \
paid call spends privacy budget (epsilon) and returns noisy statistics. When the budget is \
spent, calls are refused.

Your job: decide whether any client is faulty, and for each faulty client name the fault type \
and the first faulty round. The federation may be entirely clean; false alarms count against \
you as much as misses. Faults affect only a few clients, and noise can produce single outlying \
values, so look for consistent evidence.

Fault types:
{chr(10).join(f'- {k}: {v}' for k, v in FAULT_GUIDE.items())}

Noise: the noise on a client's statistics scales with 1 / (its number of records) and \
1 / epsilon. Clients differ a lot in size (see n_train), so small clients look noisy and \
produce spurious outliers. Treat an outlier from a small client as weak evidence unless it is \
far beyond the spread of the other clients or is confirmed by a second, independent tool. The screening tools \
do this for you: `flags` (client tables) and `z_noise` (outlier rows) give each deviation in \
units that already include the client's own noise. Base findings on those, not on raw values: \
a raw value that looks impossible (e.g. a negative percentile of a 0/1 feature) is often just noise. \
Multiple comparisons: the screens test about 50 clients on dozens of statistics, so a clean \
federation shows several |z| between 3 and 5 by chance alone. Treat such a value as a lead, \
not a finding: report a client only if |z| is clearly larger (about 6 or more) or a second, \
independent statistic or tool points the same way. Expect most audits to report zero to a \
few clients; flagging many clients is a sign of chasing noise.

Work plan:
1. Call federation_overview (free).
2. Screen all clients before focusing on any of them. Call the all-client view (no `client` / \
`clients` argument) of each screening tool: client_updates, client_performance, drift_by_round, \
feature_summary, and subgroup_outcomes for each sensitive attribute. Do not single out a client \
before you have seen these screens.
3. Where each fault shows up: poisoning, free_rider and explanation_instability in \
client_updates; label_noise and prevalence_shift in client_performance; drift in \
drift_by_round; missing_artefact in feature_summary (tail percentiles); subgroup_bias in \
subgroup_outcomes. A client is faulty when it stands apart from the other clients, not when \
it is merely the most extreme of a smooth spread.
   Telling faults apart: a client whose data is wrong also trains away from the others, so \
an update pointing against the median (negative cos) does not by itself mean poisoning. Before \
calling a client poisoning, check its row in client_performance: label_noise lowers the local \
AUC (and shifts CITL/prevalence); prevalence_shift shifts CITL and prevalence with a normal \
AUC; drift shows in drift_by_round; subgroup_bias shows in subgroup_outcomes. Call it \
poisoning only when the updates are anomalous and the client's local data and performance look \
normal.
4. Confirm suspicions with narrower queries (fewer clients/features) or larger epsilon. Plan to \
spend most of the budget before deciding; unspent budget is wasted. A good split is about 3 epsilon per \
screening call, keeping the rest to confirm suspicions.

Finish by calling submit_report exactly once. Every finding needs claims, and every claim must \
cite the call_id of a tool result and copy one number from it exactly as returned (metric = \
the column name, plus the client / round / feature / group of that row when the table has \
them). Claims are fact-checked against the tool log; numbers you did not get from a tool will \
be rejected. Use onset_round 1 for faults present from the start."""

SUBMIT_TOOL = {
    "name": "submit_report",
    "description": "Submit the final audit report. Call once, at the end.",
    "input_schema": REPORT_SCHEMA,
}


@dataclass
class AgentOutcome:
    report: Optional[dict]
    stop: str
    turns: int = 0
    usage: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ------------------------------------------------------------------- scripted

class ScriptedAuditor:
    """Runs ``policy(session) -> report`` over the same tool session as the LLM agent."""

    def __init__(self, policy: Callable[[AuditSession], dict]):
        self.policy = policy

    def run(self, session: AuditSession) -> AgentOutcome:
        return AgentOutcome(self.policy(session), "scripted", 0)


# --------------------------------------------------------------------- Claude

def load_api_key() -> Optional[str]:
    from dotenv import dotenv_values
    values = dotenv_values(os.path.join(BASE_DIR, ".env"))
    return values.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")


class ClaudeAuditor:
    def __init__(self, model: str = MODEL, effort: str = "high", max_turns: int = 40,
                 verbose: bool = True):
        import anthropic
        key = load_api_key()
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not found in Final-Year/.env or the environment")
        from dotenv import dotenv_values
        base = dotenv_values(os.path.join(BASE_DIR, ".env")).get("ANTHROPIC_BASE_URL") or DEFAULT_API
        self.client = anthropic.Anthropic(api_key=key, base_url=base)
        self.model, self.effort, self.max_turns, self.verbose = model, effort, max_turns, verbose

    def _create(self, messages):
        extra = {}
        if self.model in FALLBACK_MODELS:   # server-side refusal fallback; Haiku has none
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return self.client.beta.messages.create(
            model=self.model, max_tokens=16000, system=SYSTEM,
            tools=TOOL_SPECS + [SUBMIT_TOOL], messages=messages,
            output_config={"effort": self.effort},
            cache_control={"type": "ephemeral"},
            **extra,
        )

    def run(self, session: AuditSession) -> AgentOutcome:
        clients = session.tools.names
        messages = [{"role": "user", "content":
                     f"Audit this federation. Privacy budget: {session.accountant.total} epsilon. "
                     "Begin."}]
        usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
        out = AgentOutcome(None, "max_turns", usage=usage)
        nudges = 0
        for turn in range(1, self.max_turns + 1):
            out.turns = turn
            resp = self._create(messages)
            u = resp.usage
            usage["input"] += u.input_tokens
            usage["output"] += u.output_tokens
            usage["cache_read"] += u.cache_read_input_tokens or 0
            usage["cache_write"] += u.cache_creation_input_tokens or 0
            if resp.stop_reason == "refusal":
                out.stop = "refusal"
                return out
            messages.append({"role": "assistant", "content": resp.content})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                if nudges >= 2:
                    out.stop = "no_report"
                    return out
                nudges += 1
                messages.append({"role": "user", "content": "Finish by calling submit_report."})
                continue
            results = []
            for b in calls:
                if b.name == "submit_report":
                    errs = validate_report(b.input, clients, session.run.rounds)
                    if errs:
                        out.errors.extend(errs)
                        results.append({"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                                        "content": "Invalid report: " + "; ".join(errs[:10])})
                        continue
                    out.report, out.stop = b.input, "submitted"
                    if self.verbose:
                        print(f"  turn {turn}: submit_report ({len(b.input['findings'])} findings)")
                    return out
                res = session.call(b.name, b.input)
                if self.verbose:
                    args = {k: v for k, v in b.input.items()}
                    print(f"  turn {turn}: {res['call_id']} {b.name} {json.dumps(args)}"
                          f"{' ERROR ' + res['error'] if 'error' in res else ''}"
                          f" (budget left {res['budget_left']})")
                results.append({"type": "tool_result", "tool_use_id": b.id,
                                "content": AuditSession.render(res), "is_error": "error" in res})
            messages.append({"role": "user", "content": results})
        return out


# ---------------------------------------------------------------------- audit

def audit(run: AuditRun, agent, budget: float = DEFAULT_BUDGET, seed: int = 0) -> dict:
    session = AuditSession(run, budget=budget, seed=seed)
    t0 = time.perf_counter()
    outcome = agent.run(session)
    report = outcome.report or {"findings": [], "summary": f"no report ({outcome.stop})"}
    return {
        "scenario": run.scenario.to_dict(), "truth": run.truth,
        "agent": type(agent).__name__, "stop": outcome.stop, "turns": outcome.turns,
        "usage": outcome.usage, "report_errors": outcome.errors,
        "seconds": round(time.perf_counter() - t0, 1),
        "budget": {"total": session.accountant.total, "spent": round(session.accountant.spent, 6)},
        "report": report, "log": session.log,
        "fact_check": fact_check(report, session.log),
        "score": score(report, run.truth),
    }


assert set(FAULT_GUIDE) == set(FAULTS)
