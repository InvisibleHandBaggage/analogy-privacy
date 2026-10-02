import contextlib
import io
import os
import subprocess
import sys
import unittest

from analogy_privacy.leak_audit import Severity
from examples import hr_leave_policy as ex

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class ScenarioTests(unittest.TestCase):
    def test_clean_analogue_passes_everything(self):
        ev = ex.scenario_clean()
        self.assertTrue(ev["invariance"].passed)
        self.assertTrue(ev["leak"].passed)
        self.assertTrue(ev["leak"].uniqueness_assessed)
        self.assertEqual(ev["leak"].uniqueness.get("original").k, 1)  # the real case is unique in the roster
        self.assertGreaterEqual(ev["leak"].uniqueness.get("analogue").k, 5)

    def test_rare_token_leak_is_blocked(self):
        ev = ex.scenario_leaks_a_token()
        self.assertTrue(ev["invariance"].passed)
        self.assertFalse(ev["leak"].passed)
        top = [f for f in ev["leak"].direct.blocking if f.severity is Severity.HIGH]
        self.assertEqual([f.text for f in top], ["nonesuch"])

    def test_outcome_flip_is_caught_and_attributed(self):
        ev = ex.scenario_flips_outcome()
        self.assertFalse(ev["invariance"].passed)
        self.assertEqual(ev["sensitivity"]["standard_leave_policy"].flipping_fields, ("absence_months",))
        self.assertEqual(ev["sensitivity"]["standard_leave_policy"].misclassified, ())

    def test_unique_context_passes_direct_check_but_fails_uniqueness(self):
        ev = ex.scenario_unique_context()
        self.assertTrue(ev["invariance"].passed)
        self.assertTrue(ev["leak"].direct.passed)
        self.assertFalse(ev["leak"].passed)
        self.assertEqual(ev["leak"].uniqueness.get("analogue").k, 1)

    def test_mistagged_condition_is_exposed_by_second_policy(self):
        ev = ex.scenario_mistagged()
        self.assertFalse(ev["invariance"].passed)
        self.assertEqual([r.policy for r in ev["invariance"].failures], ["condition_specific_policy"])
        self.assertEqual(ev["sensitivity"]["condition_specific_policy"].misclassified, ("condition",))
        self.assertEqual(ev["sensitivity"]["standard_leave_policy"].flipping_fields, ())

    def test_condition_kept_when_decision_relevant(self):
        ev = ex.scenario_condition_kept()
        self.assertTrue(ev["invariance"].passed)
        self.assertTrue(ev["leak"].passed)
        self.assertEqual(ev["analogue"]["condition"], "cancer")
        shared = [f for f in ev["leak"].direct.findings if f.source == "condition"]
        self.assertTrue(shared and all(f.allowed for f in shared))  # visible, and allowed on purpose
        # keeping the condition narrows the anonymity set
        base = ex.scenario_clean()["leak"].uniqueness.get("analogue").k
        self.assertLess(ev["leak"].uniqueness.get("analogue").k, base)

    def test_roundtrip(self):
        rt = ex.scenario_roundtrip()
        self.assertEqual(rt["refusal"].reasons, ("adequacy sign-off is missing",))
        rec = rt["record"]
        self.assertEqual(rec.status, "cleared")
        for secret in ("Alice", "Nonesuch", "Accounting", "cancer", "E-1047"):
            self.assertNotIn(secret, rt["sent"][0])
        self.assertIn("Alice Nonesuch", rec.back_mapped_answer)
        self.assertEqual(rec.residue, ("bruce",))  # the known back-mapping miss, surfaced

    def test_roster_is_deterministic(self):
        self.assertEqual(ex.roster(), ex.roster())
        self.assertEqual(len(ex.roster()), 57)


class RunTests(unittest.TestCase):
    def test_main_runs(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ex.main()
        out = buf.getvalue()
        self.assertIn("=== 7. round trip", out)
        self.assertIn("Synthetic data only", out)

    def test_runs_as_module_and_as_script(self):
        for args in (["-m", "examples.hr_leave_policy"], [os.path.join("examples", "hr_leave_policy.py")]):
            proc = subprocess.run([sys.executable] + args, cwd=ROOT, capture_output=True, text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("=== 1. clean analogue", proc.stdout)


if __name__ == "__main__":
    unittest.main()
