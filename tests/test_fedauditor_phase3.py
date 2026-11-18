"""Phase 3: audit tools, privacy accountant, fact-checker, scorer and agent loop (no API)."""
from __future__ import annotations

import json
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from federated_metrics import full_metrics
from fedauditor.agent import ClaudeAuditor, ScriptedAuditor, audit
from fedauditor.audit_env import AuditRun
from fedauditor.faults import Scenario
from fedauditor.report import aggregate, check_claim, fact_check, score, validate_report
from fedauditor.tools import AuditSession, BudgetExhausted, PrivacyAccountant, TOOL_NAMES
from test_fedfault import _data

BIG = 1e6   # epsilon so large the noise is negligible


def _run(fault, severity="high", rounds=6, model="lr"):
    sev = "none" if fault == "clean" else severity
    return AuditRun.from_scenario(Scenario(fault, sev, rounds=rounds, model=model), DATA)


def _rows(table):
    cols = table["columns"]
    return [dict(zip(cols, r)) for r in table["rows"]]


def setUpModule():
    global DATA
    warnings.filterwarnings("ignore")
    DATA = _data()


class TestAccountant(unittest.TestCase):
    def test_charge_refuse_refund(self):
        acc = PrivacyAccountant(2.0)
        acc.charge(1.5, "c1", "t")
        with self.assertRaises(BudgetExhausted):
            acc.charge(1.0, "c2", "t")
        acc.refund("c1")
        self.assertEqual(acc.remaining, 2.0)

    def test_session_spends_and_refuses(self):
        s = AuditSession(_run("clean"), budget=1.5)
        ov = s.call("federation_overview")
        self.assertEqual(ov["epsilon_spent"], 0.0)
        self.assertNotIn("error", s.call("client_performance", {"epsilon": 1.0}))
        bad = s.call("client_updates", {"client": "nowhere"})
        self.assertIn("error", bad)
        self.assertEqual(s.accountant.remaining, 0.5)        # invalid call refunded
        refused = s.call("global_performance", {"epsilon": 1.0})
        self.assertIn("budget", refused["error"])
        self.assertEqual([e["id"] for e in s.log], ["c1", "c2", "c3", "c4"])
        self.assertEqual([e["ok"] for e in s.log], [True, True, False, False])


class TestTools(unittest.TestCase):
    def test_noise_scales_with_epsilon(self):
        run = _run("clean")
        model = run.global_at(run.rounds)
        c = run.final_data.clients[0]
        exact = full_metrics(c.y["test"], model.predict_proba(c.X["test"]))["auc"]
        s = AuditSession(run, budget=BIG * 3)
        precise = _rows(s.call("client_performance", {"epsilon": BIG})["clients"])[0]["auc"]
        noisy = [_rows(s.call("client_performance", {"epsilon": 0.05})["clients"])[0]["auc"]
                 for _ in range(3)]
        self.assertAlmostEqual(precise, exact, places=3)
        self.assertTrue(any(abs(v - exact) > 0.01 for v in noisy))

    def test_results_are_small_aggregates(self):
        run = _run("drift")
        s = AuditSession(run, budget=100)
        n_rows = min(len(c.y["test"]) for c in run.final_data.clients)
        for name in TOOL_NAMES:
            args = {"attribute": "Sex"} if name == "subgroup_outcomes" else {}
            res = s.call(name, args)
            self.assertNotIn("error", res, name)
            for key, val in res.items():
                if isinstance(val, dict) and "rows" in val:
                    self.assertLess(len(val["rows"]), n_rows, f"{name}.{key} looks row-level")
            self.assertLess(len(json.dumps(res)), 50_000)

    def test_update_direction_flags_poisoning(self):
        run = _run("poisoning", "low")
        rows = _rows(AuditSession(run, budget=BIG).call("client_updates", {"epsilon": BIG})["clients"])
        worst = min(rows, key=lambda r: r["mean_cos"])
        self.assertEqual([worst["client"]], run.truth["clients"])
        self.assertLess(worst["mean_cos"], 0)

    def test_free_rider_zero_norm(self):
        run = _run("free_rider", "medium")
        rows = _rows(AuditSession(run, budget=BIG).call("client_updates", {"epsilon": BIG})["clients"])
        low = [r["client"] for r in rows if r["min_norm_ratio"] < 0.05]
        self.assertEqual(low, run.truth["clients"])

    def test_feature_summary_shows_sentinel(self):
        run = _run("missing_artefact", "medium")
        res = AuditSession(run, budget=BIG).call("feature_summary", {"epsilon": BIG})
        feats = {f["feature"] for d in run.truth["details"].values() for f in d["features"]}
        hits = {(r["client"], r["feature"]) for r in _rows(res["outliers"])}
        target = run.truth["clients"][0]
        self.assertTrue(any((target, f) in hits for f in feats), hits)

    def test_drift_onset(self):
        run = _run("drift", "high")
        res = AuditSession(run, budget=BIG).call("drift_by_round", {"epsilon": BIG})
        rows = {r["client"]: r for r in _rows(res["clients"])}
        target = run.truth["clients"][0]
        self.assertEqual(max(rows, key=lambda c: rows[c]["max_shift_sd"]), target)
        self.assertEqual(rows[target]["onset_round"], run.truth["onset_round"])

    def test_subgroup_bias_outlier(self):
        run = _run("subgroup_bias", "high")
        res = AuditSession(run, budget=BIG).call("subgroup_outcomes", {"attribute": "Sex", "epsilon": BIG})
        flagged = {r["client"] for r in _rows(res["outliers"]) if r["outlier_stat"] in ("oe_ratio", "outcome_rate")}
        self.assertIn(run.truth["clients"][0], flagged)

    def test_attribution_churn_rises_with_instability(self):
        def churn(fault):
            res = AuditSession(_run(fault), budget=BIG).call("attribution_churn", {"epsilon": BIG})
            return np.mean([r["rel_change_prev"] for r in _rows(res["by_round"])[2:]])
        self.assertGreater(churn("explanation_instability"), churn("clean"))


class TestFactCheckAndScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arun = _run("poisoning")
        cls.s = AuditSession(cls.arun, budget=10)
        cls.res = cls.s.call("client_updates", {"epsilon": 1.0})
        cls.s.call("client_updates", {"client": "nowhere"})
        cls.row = next(r for r in _rows(cls.res["clients"]) if r["client"] == cls.arun.truth["clients"][0])

    def _claim(self, **kw):
        base = {"call_id": "c1", "metric": "mean_cos", "value": self.row["mean_cos"],
                "client": self.row["client"]}
        return check_claim({**base, **kw}, {e["id"]: e for e in self.s.log})

    def test_claims(self):
        self.assertTrue(self._claim()["supported"])
        self.assertTrue(self._claim(value=self.row["mean_cos"] + 0.001)["supported"])
        self.assertFalse(self._claim(value=self.row["mean_cos"] + 0.5)["supported"])
        other = next(c for c in self.arun.final_data.client_names if c != self.row["client"])
        self.assertFalse(self._claim(client=other)["supported"])
        self.assertFalse(self._claim(call_id="c99")["supported"])
        self.assertFalse(self._claim(call_id="c2")["supported"])
        self.assertFalse(self._claim(metric="auc")["supported"])

    def test_score_and_validation(self):
        t = self.arun.truth
        good = {"findings": [{"client": t["clients"][0], "fault_type": "poisoning", "onset_round": 1,
                              "confidence": 0.9, "rationale": "", "claims": [
                                  {"call_id": "c1", "metric": "mean_cos", "value": self.row["mean_cos"],
                                   "client": t["clients"][0]}]}], "summary": ""}
        self.assertEqual(validate_report(good, self.arun.final_data.client_names, 6), [])
        s = score(good, t)
        self.assertEqual((s["precision"], s["recall"], s["type_accuracy"], s["onset_abs_error"]),
                         (1.0, 1.0, 1.0, 0.0))
        self.assertEqual(fact_check(good, self.s.log)["supported"], 1)
        clean_truth = {"fault": "clean", "clients": [], "onset_round": None}
        alarm = score(good, clean_truth)
        quiet = score({"findings": []}, clean_truth)
        self.assertTrue(alarm["false_alarm"])
        self.assertFalse(quiet["false_alarm"])
        agg = aggregate([s, alarm, quiet])
        self.assertEqual(agg["false_alarm_rate"], 0.5)
        self.assertEqual(agg["recall"], 1.0)
        bad = {"findings": [{"client": "X", "fault_type": "gremlins", "onset_round": 0, "claims": []}]}
        self.assertEqual(len(validate_report(bad, self.arun.final_data.client_names, 6)), 4)


def rule_policy(session: AuditSession) -> dict:
    """A tiny rule-based auditor: flag clients whose updates point against the others."""
    res = session.call("client_updates", {"epsilon": 2.0})
    findings = []
    for r in _rows(res["clients"]):
        if r["mean_cos"] < 0:
            findings.append({"client": r["client"], "fault_type": "poisoning", "onset_round": 1,
                             "confidence": 0.8, "rationale": "update opposes the others",
                             "claims": [{"call_id": res["call_id"], "metric": "mean_cos",
                                         "value": r["mean_cos"], "client": r["client"]}]})
    return {"findings": findings, "summary": f"{len(findings)} suspicious clients"}


class TestAgents(unittest.TestCase):
    def test_scripted_end_to_end(self):
        for fault in ("poisoning", "clean"):
            rec = audit(_run(fault), ScriptedAuditor(rule_policy), budget=5)
            self.assertEqual(rec["fact_check"]["unsupported"], [])
            if fault == "poisoning":
                self.assertEqual((rec["score"]["precision"], rec["score"]["recall"]), (1.0, 1.0))
            else:
                self.assertFalse(rec["score"]["false_alarm"])

    def test_claude_loop_with_fake_client(self):
        run = _run("poisoning")
        target = run.truth["clients"][0]
        seen = {}

        def block(**kw):
            return SimpleNamespace(**kw)

        def usage():
            return SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                   cache_creation_input_tokens=0)

        def fake_create(messages):
            turn = sum(m["role"] == "assistant" for m in messages)
            if turn == 0:
                return SimpleNamespace(stop_reason="tool_use", usage=usage(), content=[
                    block(type="tool_use", id="t1", name="client_updates", input={"epsilon": 1.0})])
            if turn == 1:
                res = json.loads(messages[-1]["content"][0]["content"])
                seen["row"] = next(r for r in _rows(res["clients"]) if r["client"] == target)
                bad = {"findings": [{"client": target}], "summary": ""}
                return SimpleNamespace(stop_reason="tool_use", usage=usage(), content=[
                    block(type="tool_use", id="t2", name="submit_report", input=bad)])
            self.assertTrue(messages[-1]["content"][0]["is_error"])   # invalid report bounced
            report = {"findings": [{"client": target, "fault_type": "poisoning", "onset_round": 1,
                                    "confidence": 0.9, "rationale": "", "claims": [
                                        {"call_id": "c1", "metric": "mean_cos",
                                         "value": seen["row"]["mean_cos"], "client": target}]}],
                      "summary": "one poisoner"}
            return SimpleNamespace(stop_reason="tool_use", usage=usage(), content=[
                block(type="tool_use", id="t3", name="submit_report", input=report)])

        agent = object.__new__(ClaudeAuditor)
        agent.max_turns, agent.verbose = 5, False
        agent._create = fake_create
        rec = audit(run, agent, budget=5)
        self.assertEqual(rec["stop"], "submitted")
        self.assertEqual(rec["turns"], 3)
        self.assertEqual(rec["usage"]["input"], 30)
        self.assertEqual(rec["score"]["recall"], 1.0)
        self.assertEqual(rec["fact_check"]["supported"], 1)


if __name__ == "__main__":
    unittest.main()
