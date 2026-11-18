"""Phase 4: rule baseline, open-model (OpenAI-compatible) agent with a fake model, grid summary (no API)."""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import warnings
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from fedauditor import run_audit_grid
from fedauditor.agent import audit
from fedauditor.audit_env import AuditRun
from fedauditor.baseline import RuleAuditor
from fedauditor.faults import Scenario
from fedauditor.openai_agent import OpenAIAuditor, openai_tools
from fedauditor.report import score
from test_fedfault import _data


def setUpModule():
    global DATA
    warnings.filterwarnings("ignore")
    DATA = _data()


def _scenario_run(fault, severity="high", model="lr"):
    sev = "none" if fault == "clean" else severity
    return AuditRun.from_scenario(Scenario(fault, sev, rounds=6, model=model), DATA)


def _audit(fault, severity="high", model="lr"):
    return audit(_scenario_run(fault, severity, model), RuleAuditor(), seed=0)


class TestRuleBaseline(unittest.TestCase):
    def test_detects_update_faults_with_supported_claims(self):
        for fault in ("poisoning", "free_rider"):
            rec = _audit(fault)
            s = rec["score"]
            self.assertEqual(s["recall"], 1.0, fault)
            self.assertEqual(s["type_accuracy"], 1.0, fault)
            self.assertEqual(rec["fact_check"]["unsupported"], [], fault)
            self.assertLessEqual(rec["budget"]["spent"], rec["budget"]["total"])

    def test_free_rider_onset_is_dated(self):
        rec = _audit("free_rider")
        onsets = {f["client"]: f["onset_round"] for f in rec["report"]["findings"]}
        truth = rec["truth"]
        self.assertEqual(onsets[truth["clients"][0]], truth["onset_round"])

    def test_report_shape_on_every_fault(self):
        from fedauditor.faults import FAULTS
        for fault in list(FAULTS) + ["clean"]:
            rec = _audit(fault)
            self.assertEqual(rec["report_errors"], [], fault)
            self.assertEqual(rec["fact_check"]["unsupported"], [], fault)


# ------------------------------------------------------------ open-model agent

def _reply(content=None, calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"id{i}", "type": "function",
                              "function": {"name": n, "arguments": a if isinstance(a, str) else json.dumps(a)}}
                             for i, (n, a) in enumerate(calls)]
    return {"choices": [{"message": msg, "finish_reason": "tool_calls" if calls else finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def _last_tool_result(messages):
    return json.loads(next(m for m in reversed(messages) if m["role"] == "tool")["content"])


def _flag_negative_cos(messages):
    """What a competent model would report after one client_updates call."""
    res = _last_tool_result(messages)
    cols = res["clients"]["columns"]
    findings = []
    for row in res["clients"]["rows"]:
        r = dict(zip(cols, row))
        if r["mean_cos"] < 0:
            findings.append({"client": r["client"], "fault_type": "poisoning", "onset_round": 1,
                             "confidence": 0.9, "rationale": "opposing updates",
                             "claims": [{"call_id": res["call_id"], "metric": "mean_cos",
                                         "value": r["mean_cos"], "client": r["client"]}]})
    return {"findings": findings, "summary": "done"}


def _fake(script):
    """Patch for OpenAIAuditor._create: turn i answers with script[i](messages)."""
    seen = []

    def create(self, messages):
        seen.append([dict(m) for m in messages])
        return script[len(seen) - 1](messages)
    return create, seen


class TestOpenAIAuditor(unittest.TestCase):
    def test_tools_in_openai_format(self):
        tools = openai_tools()
        self.assertEqual(len(tools), 9)
        self.assertEqual(tools[-1]["function"]["name"], "submit_report")
        self.assertTrue(all(t["type"] == "function" and "parameters" in t["function"] for t in tools))

    def test_native_tool_calls_end_to_end(self):
        create, seen = _fake([
            lambda m: _reply(calls=[("federation_overview", {}), ("client_updates", {"epsilon": 2.0})]),
            lambda m: _reply(calls=[("submit_report", _flag_negative_cos(m))]),
        ])
        with mock.patch.object(OpenAIAuditor, "_create", create):
            rec = audit(_scenario_run("poisoning"), OpenAIAuditor("fake"), seed=0)
        s = rec["score"]
        self.assertEqual((rec["stop"], s["precision"], s["recall"], s["type_accuracy"]),
                         ("submitted", 1.0, 1.0, 1.0))
        self.assertEqual(rec["fact_check"]["unsupported"], [])
        self.assertEqual(rec["usage"], {"input": 200, "output": 20})
        tool_msgs = [m for m in seen[1] if m["role"] == "tool"]
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["id0", "id1"])
        self.assertEqual(seen[0][0]["role"], "system")

    def test_text_calls_bad_json_nudges_and_invalid_report(self):
        text_call = ('<think>check updates first</think>Let me look.\n<tool_call>\n'
                     '{"name": "client_updates", "arguments": {"epsilon": 2}}\n</tool_call>')
        create, seen = _fake([
            lambda m: _reply(content=text_call),                                  # Qwen text-format call
            lambda m: _reply(calls=[("client_performance", '{"epsilon": 1,')]),   # broken JSON
            lambda m: _reply(content="I think everything looks fine."),           # no tool -> nudge
            lambda m: _reply(calls=[("submit_report", {"findings": [{"client": "Atlantis"}]})]),
            lambda m: _reply(calls=[("submit_report", {"findings": [], "summary": "clean"})]),
        ])
        with mock.patch.object(OpenAIAuditor, "_create", create):
            rec = audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        self.assertEqual(rec["stop"], "submitted")
        self.assertFalse(rec["score"]["false_alarm"])
        self.assertEqual([e["tool"] for e in rec["log"]], ["client_updates"])   # the bad call never ran
        self.assertTrue(any("not valid JSON" in e for e in rec["report_errors"]))
        self.assertTrue(any("unknown client" in e for e in rec["report_errors"]))
        asst = seen[1][2]
        self.assertNotIn("<think>", asst["content"])
        self.assertNotIn("<tool_call>", asst["content"])
        self.assertEqual(asst["tool_calls"][0]["function"]["name"], "client_updates")
        self.assertIn("Error: arguments are not valid JSON", seen[2][-1]["content"])
        self.assertIn("submit_report", seen[3][-1]["content"])

    def test_gives_up_without_report_and_survives_server_errors(self):
        create, _ = _fake([lambda m: _reply(content="hmm")] * 3)
        with mock.patch.object(OpenAIAuditor, "_create", create):
            rec = audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        self.assertEqual(rec["stop"], "no_report")
        self.assertEqual(rec["report"]["findings"], [])

        def boom(self, messages):
            raise ConnectionError("server not running")
        with mock.patch.object(OpenAIAuditor, "_create", boom):
            rec = audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        self.assertTrue(rec["stop"].startswith("api_error: ConnectionError"))

    def test_http_request_against_local_fake_server(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, self.headers.get("Authorization"), body))
                out = json.dumps(_reply(calls=[("submit_report", {"findings": [], "summary": "ok"})])).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            agent = OpenAIAuditor("Qwen/Qwen3-8B-AWQ", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                  api_key="test-key")
            rec = audit(_scenario_run("clean"), agent, seed=0)
        finally:
            srv.shutdown()
        path, auth, body = received[0]
        self.assertEqual((path, auth), ("/v1/chat/completions", "Bearer test-key"))
        self.assertEqual(body["model"], "Qwen/Qwen3-8B-AWQ")
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(len(body["tools"]), 9)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(rec["stop"], "submitted")


class TestOpenAIGuardRails(unittest.TestCase):
    """Fixes for what Qwen3 via Ollama did in the first Kaggle pilot (HTTP 400s, no report)."""

    def test_history_is_server_safe(self):
        create, seen = _fake([
            lambda m: _reply(calls=[("client_updates", '{"epsilon": 2')]),       # broken JSON
            lambda m: _reply(calls=[("client_updates", {"epsilon": 2})]),        # content None + call
            lambda m: _reply(calls=[("submit_report", {"findings": [], "summary": "x"})]),
        ])
        with mock.patch.object(OpenAIAuditor, "_create", create):
            audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        for m in seen[-1]:
            self.assertIsNotNone(m.get("content"), m)
            for c in m.get("tool_calls", []):
                self.assertIsInstance(json.loads(c["function"]["arguments"]), dict)

    def test_report_written_as_text_is_accepted(self):
        report = {"findings": [], "summary": "looks clean"}
        create, _ = _fake([lambda m: _reply(content="Final report:\n```json\n" + json.dumps(report) + "\n```")])
        with mock.patch.object(OpenAIAuditor, "_create", create):
            rec = audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        self.assertEqual(rec["stop"], "submitted_text")
        self.assertFalse(rec["score"]["false_alarm"])

    def test_told_to_submit_when_budget_spent(self):
        create, seen = _fake([
            lambda m: _reply(calls=[("client_updates", {"epsilon": 20})]),
            lambda m: _reply(calls=[("submit_report", {"findings": [], "summary": "x"})]),
        ])
        with mock.patch.object(OpenAIAuditor, "_create", create):
            audit(_scenario_run("clean"), OpenAIAuditor("fake"), seed=0)
        last = seen[1][-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("privacy budget is spent", last["content"])

    def test_http_error_text_and_transcript_are_kept(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                out = b'{"error":{"message":"invalid message content type: <nil>"}}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            try:
                agent = OpenAIAuditor("qwen3:8b", base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                      transcript_dir=d)
                run = _scenario_run("clean")
                rec = audit(run, agent, seed=0)
            finally:
                srv.shutdown()
            self.assertIn("HTTP 400", rec["stop"])
            self.assertIn("invalid message content type", rec["stop"])
            path = os.path.join(d, f"{run.scenario.id}__qwen3-8b__transcript.json")
            saved = json.load(open(path))
            self.assertTrue(saved["messages"][0]["content"].endswith("/no_think"))
            self.assertIn("HTTP 400", saved["stop"])


class TestNoiseAwareScreens(unittest.TestCase):
    def test_laplace_sd_matches_the_noise_actually_added(self):
        import numpy as np
        from fedauditor.tools import _Noise, laplace_sd
        nz = _Noise(np.random.default_rng(0), epsilon=2.0, m=4)
        draws = np.array([nz(0.0, 5.0, 300) for _ in range(40000)])
        self.assertAlmostEqual(draws.std() / laplace_sd(5.0, 300, 4, 2.0), 1.0, delta=0.03)

    def test_flags_and_z_noise(self):
        from fedauditor.baseline import rows
        from fedauditor.tools import AuditSession
        run = _scenario_run("poisoning")
        s = AuditSession(run, budget=100, seed=0)
        up = {r["client"]: r["flags"] for r in rows(s.call("client_updates", {"epsilon": 3})["clients"])}
        for c in run.truth["clients"]:
            self.assertIn("mean_cos z=-", up[c])
        # a cited flag is fact-checked as metric "<stat>_z"
        from fedauditor.report import fact_check, parse_flags
        call_id = s.log[-1]["id"]
        c0 = run.truth["clients"][0]
        z = parse_flags(up[c0])["mean_cos"]
        rep = {"findings": [{"client": c0, "claims": [
            {"call_id": call_id, "metric": "mean_cos_z", "value": z, "client": c0},
            {"call_id": call_id, "metric": "mean_cos", "value": z, "client": c0}]}]}
        fc = fact_check(rep, s.log)
        self.assertEqual((fc["claims"], fc["supported"]), (2, 1))   # the z is not the mean_cos value
        perf = rows(s.call("client_performance", {"epsilon": 3})["clients"])
        self.assertEqual(perf[-1]["client"], "_pooled")
        self.assertEqual(perf[-1]["flags"], "")
        self.assertIn("flags", s.call("drift_by_round", {"epsilon": 3})["clients"]["columns"])
        fs = rows(s.call("feature_summary", {"epsilon": 3})["outliers"])
        self.assertTrue(all(r["z_noise"] is None or abs(r["z_noise"]) > 2 for r in fs))
        so = s.call("subgroup_outcomes", {"attribute": run.final_data and list(
            s.call("federation_overview")["sensitive_attributes"])[0], "epsilon": 3})
        self.assertIn("z_noise", so["outliers"]["columns"])


class TestClaudeRequest(unittest.TestCase):
    def test_fallbacks_only_for_models_that_support_them(self):
        from types import SimpleNamespace
        from fedauditor.agent import ClaudeAuditor
        sent = {}
        for model in ("claude-haiku-5-5", "claude-opus-5-5"):
            agent = object.__new__(ClaudeAuditor)
            agent.model, agent.effort = model, "high"
            agent.client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(
                create=lambda **kw: sent.__setitem__(kw["model"], kw))))
            agent._create([{"role": "user", "content": "hi"}])
        self.assertNotIn("fallbacks", sent["claude-haiku-5-5"])
        self.assertNotIn("betas", sent["claude-haiku-5-5"])
        self.assertEqual(sent["claude-opus-5-5"]["fallbacks"], "default")

    def test_pilot_label_for_claude(self):
        cfg = {"llm": "claude-haiku-5-5", "label": "claude-haiku-5-5-pilot"}
        self.assertEqual(run_audit_grid.agent_label("claude", cfg), "claude-haiku-5-5-pilot")


class TestScoreTies(unittest.TestCase):
    def test_tied_types_get_fractional_credit(self):
        truth = {"fault": "label_noise", "severity": "low", "clients": ["A", "B"], "onset_round": 1}
        rep = {"findings": [{"client": "A", "fault_type": "label_noise", "onset_round": 1},
                            {"client": "B", "fault_type": "poisoning", "onset_round": 1}]}
        self.assertEqual(score(rep, truth)["scenario_type_correct"], 0.5)
        rep["findings"].append({"client": "C", "fault_type": "label_noise", "onset_round": 1})
        self.assertEqual(score(rep, truth)["scenario_type_correct"], 1.0)
        self.assertEqual(score({"findings": []}, truth)["scenario_type_correct"], 0.0)


class TestCompareAgents(unittest.TestCase):
    def test_confidence_cutoff_removes_weak_false_alarm(self):
        from fedauditor.compare_agents import compare
        truth = {"fault": "clean", "severity": "none", "clients": [], "onset_round": None}
        line = {"id": "c1", "truth": truth, "cost_usd": 0.01, "fact_check": {"claims": 1, "supported": 1},
                "findings": [{"client": "GU", "fault_type": "drift", "onset_round": 1, "confidence": 0.4}]}
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run_audit_grid, "RES_DIR", d):
            with open(run_audit_grid.grid_path("toy", "lr", "llm"), "w") as fh:
                fh.write(json.dumps(line) + "\n")
            res = compare("toy", "lr", ["llm"], [0.0, 0.5])
        self.assertEqual(res["rows"]["llm@0"]["false_alarm_rate"], 1.0)
        self.assertEqual(res["rows"]["llm@0.5"]["false_alarm_rate"], 0.0)


class TestGrid(unittest.TestCase):
    def test_summarise_counts_and_cost(self):
        truth_f = {"fault": "poisoning", "severity": "low", "clients": ["A"], "onset_round": 1}
        truth_c = {"fault": "clean", "severity": "none", "clients": [], "onset_round": None}
        hit = {"findings": [{"client": "A", "fault_type": "poisoning", "onset_round": 1}]}
        lines = [
            {"id": "x", "score": score(hit, truth_f), "findings": hit["findings"], "truth": truth_f, "fact_check": {"claims": 2, "supported": 2},
             "calls": 5, "epsilon": 20, "cost_usd": 0.3, "stop": "submitted"},
            {"id": "y", "score": score({"findings": []}, truth_c), "findings": [], "truth": truth_c, "fact_check": {"claims": 0, "supported": 0},
             "calls": 4, "epsilon": 18, "cost_usd": 0.2, "stop": "api_error: X"},
        ]
        with tempfile.TemporaryDirectory() as d, mock.patch.object(run_audit_grid, "RES_DIR", d):
            for label in ("qwen3-8b-awq", "rules_v1"):
                with open(run_audit_grid.grid_path("toy", "lr", label), "w") as fh:
                    fh.writelines(json.dumps(x) + "\n" for x in lines)
            summ = run_audit_grid.summarise("toy")
            self.assertTrue(os.path.exists(os.path.join(d, "audit_grid_summary_toy.json")))
        self.assertEqual(list(summ), ["lr/qwen3-8b-awq"])          # v1 archives are skipped
        o = summ["lr/qwen3-8b-awq"]["overall"]
        self.assertEqual((o["recall"], o["false_alarm_rate"], o["claim_support"], o["cost_usd"]),
                         (1.0, 0.0, 1.0, 0.5))
        self.assertEqual(o["stops"], {"submitted": 1, "api_error": 1})
        self.assertEqual(summ["lr/qwen3-8b-awq"]["by_cell"]["poisoning/low"]["recall"], 1.0)

    def test_labels_and_cost(self):
        self.assertEqual(run_audit_grid.agent_label("openai", {"llm": "Qwen/Qwen3-8B-AWQ", "label": None}),
                         "qwen3-8b-awq")
        self.assertEqual(run_audit_grid.agent_label("openai", {"llm": "x", "label": "qwen14b"}), "qwen14b")
        self.assertEqual(run_audit_grid.agent_label("claude", {"llm": "claude-opus-5-5"}), "claude")
        self.assertEqual(run_audit_grid.agent_label("claude", {"llm": "claude-haiku-5-5"}), "claude-haiku-5-5")
        u = {"input": 1_000_000, "output": 1_000_000, "cache_read": 0, "cache_write": 0}
        self.assertAlmostEqual(run_audit_grid.cost_usd(u, "claude-haiku-5-5"), 0.60)
        self.assertAlmostEqual(run_audit_grid.cost_usd(u, "claude-opus-5-5"), 24.0)


if __name__ == "__main__":
    unittest.main()
