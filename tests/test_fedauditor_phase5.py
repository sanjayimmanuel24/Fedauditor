"""Phase 5: audit report view (claim -> logged row) and figure helpers."""
import unittest

from fedauditor.audit_view import load_view, locate_claim
from fedauditor.make_figures import color, scenario_type_votes

ENTRY = {"id": "c3", "tool": "client_performance", "ok": True, "result": {
    "call_id": "c3", "epsilon_spent": 3.0,
    "clients": {"columns": ["client", "auc", "flags"],
                "rows": [["AK", 0.81, ""], ["MA", 0.837, "citl z=+3.7"]]}}}


class TestLocateClaim(unittest.TestCase):
    def test_finds_row_by_keys_and_value(self):
        self.assertEqual(locate_claim({"call_id": "c3", "client": "MA", "metric": "auc", "value": 0.837}, ENTRY),
                         ("clients", 1))

    def test_cited_flag(self):
        self.assertEqual(locate_claim({"call_id": "c3", "client": "MA", "metric": "citl_z", "value": 3.7}, ENTRY),
                         ("clients", 1))

    def test_wrong_value_or_failed_call(self):
        self.assertIsNone(locate_claim({"client": "MA", "metric": "auc", "value": 0.5}, ENTRY))
        self.assertIsNone(locate_claim({"client": "MA", "metric": "auc", "value": 0.837}, {**ENTRY, "ok": False}))


class TestLoadView(unittest.TestCase):
    def test_rejects_unknown_or_path_names(self):
        for bad in ("../app", "..\\app", "", "nope"):
            with self.assertRaises(KeyError):
                load_view(bad)


class TestFigureHelpers(unittest.TestCase):
    def test_type_votes_split_ties(self):
        rec = {"findings": [{"fault_type": "drift"}, {"fault_type": "poisoning"}]}
        self.assertEqual(scenario_type_votes(rec), {"drift": 0.5, "poisoning": 0.5})
        self.assertEqual(scenario_type_votes({"findings": []}), {"none": 1.0})

    def test_color_ramp_ends(self):
        self.assertEqual(color(0), "#cde2fb")
        self.assertEqual(color(1), "#0d366b")
        self.assertEqual(color(2), "#0d366b")


if __name__ == "__main__":
    unittest.main()
