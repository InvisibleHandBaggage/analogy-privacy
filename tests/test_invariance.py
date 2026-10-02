import unittest

from analogy_privacy.invariance import check_invariance, sensitivity_probe
from analogy_privacy.schema import Case, Role

ROLES = {"name": "identifying", "months": "decision_relevant", "dept": "contextual", "kind": "decision_relevant"}


def case(**kw):
    values = {"name": "Alice Nonesuch", "months": 6, "dept": "Accounting", "kind": "medical"}
    values.update(kw)
    return Case.from_mapping(values, ROLES)


def leave(c):
    if c["months"] < 6:
        return "standard"
    return "review" if c["kind"] == "medical" else "talk"


def leave_by_name_length(c):  # deliberately reads an identifying field (a mis-tagging)
    return "long" if len(c["name"]) > 12 else "short"


class InvarianceTests(unittest.TestCase):
    def test_pass_when_outcomes_equal(self):
        rep = check_invariance(case(), case(name="Bruce Placeholder", dept="Logistics"), [leave])
        self.assertTrue(rep.passed)
        self.assertEqual(rep.failures, ())
        self.assertEqual(rep.results[0].policy, "leave")

    def test_fail_when_outcome_flips(self):
        rep = check_invariance(case(), case(months=5), [leave])
        self.assertFalse(rep.passed)
        self.assertEqual(rep.failures[0].original_outcome, "review")
        self.assertEqual(rep.failures[0].analogue_outcome, "standard")
        self.assertIn("DIFFERENT", rep.summary())

    def test_all_policies_must_agree(self):
        rep = check_invariance(case(), case(name="Al"), {"leave": leave, "by_name": leave_by_name_length})
        self.assertFalse(rep.passed)
        self.assertEqual([r.policy for r in rep.failures], ["by_name"])

    def test_single_callable_accepted(self):
        self.assertTrue(check_invariance(case(), case(dept="X"), leave).passed)

    def test_custom_equivalence_single_function(self):
        def lenient(a, b):
            return {a, b} <= {"review", "standard"}

        self.assertTrue(check_invariance(case(), case(months=5), [leave], equivalence=lenient).passed)

    def test_custom_equivalence_per_policy(self):
        eq = {"leave": lambda a, b: True}
        rep = check_invariance(case(), case(months=5), {"leave": leave, "by_name": leave_by_name_length}, eq)
        self.assertTrue(rep.passed)

    def test_empty_policy_set_refused(self):
        with self.assertRaises(ValueError):
            check_invariance(case(), case(), [])

    def test_duplicate_policy_names_refused(self):
        with self.assertRaises(ValueError):
            check_invariance(case(), case(), [lambda c: 1, lambda c: 1])

    def test_policy_exception_is_a_failure_not_a_crash(self):
        def broken(c):
            raise RuntimeError("boom")

        rep = check_invariance(case(), case(), [broken])
        self.assertFalse(rep.passed)
        self.assertIn("RuntimeError", rep.failures[0].error)

    def test_equivalence_exception_is_a_failure(self):
        def bad_eq(a, b):
            raise ValueError("nope")

        rep = check_invariance(case(), case(), [leave], equivalence=bad_eq)
        self.assertFalse(rep.passed)
        self.assertIn("equivalence raised", rep.failures[0].error)


class SensitivityTests(unittest.TestCase):
    def test_flags_flipping_field_with_candidates(self):
        rep = sensitivity_probe([leave], case(), candidates={"months": [3, 12], "dept": ["Logistics"]})["leave"]
        self.assertEqual(rep.baseline, "review")
        self.assertEqual(rep.flipping_fields, ("months",))
        months = [p for p in rep.probes if p.field == "months"][0]
        self.assertEqual(months.flips, ((3, "standard"),))  # 12 does not flip
        self.assertEqual(rep.misclassified, ())

    def test_other_case_supplies_probe_values(self):
        a = case(name="Bruce Placeholder", months=5, dept="Logistics")
        rep = sensitivity_probe([leave], case(), other=a)["leave"]
        self.assertEqual(rep.flipping_fields, ("months",))
        self.assertEqual(rep.unprobed, ("kind",))  # equal in both cases: nothing to probe
        self.assertEqual(rep.inert_decision_relevant, ())
        name = [p for p in rep.probes if p.field == "name"][0]
        self.assertEqual((name.probed_values, name.flipped), (1, False))

    def test_misclassified_field_is_reported(self):
        rep = sensitivity_probe(
            {"by_name": leave_by_name_length}, case(), candidates={"name": ["Al"]}
        )["by_name"]
        self.assertEqual(rep.flipping_fields, ("name",))
        self.assertEqual(rep.misclassified, ("name",))

    def test_unprobed_fields_are_not_called_inert(self):
        rep = sensitivity_probe([leave], case(), candidates={"months": [3]})["leave"]
        self.assertIn("dept", rep.unprobed)
        self.assertIn("kind", rep.unprobed)
        self.assertNotIn("months", rep.unprobed)

    def test_inert_decision_relevant_reported_when_probed(self):
        rep = sensitivity_probe([leave], case(), candidates={"months": [12]})["leave"]
        self.assertEqual(rep.inert_decision_relevant, ("months",))

    def test_policy_error_on_perturbation_counts_as_flip(self):
        def fragile(c):
            if c["kind"] == "other":
                raise KeyError("unhandled")
            return "ok"

        rep = sensitivity_probe([fragile], case(), candidates={"kind": ["other"]})["fragile"]
        self.assertEqual(rep.flipping_fields, ("kind",))
        self.assertEqual(rep.probes[3].flips[0][1], "<error: KeyError>")

    def test_single_field_probing_misses_pure_interactions(self):
        # Documented limitation: the outcome flips only if BOTH fields change.
        def both(c):
            return c["months"] < 6 and c["kind"] == "other"

        rep = sensitivity_probe([both], case(), candidates={"months": [3], "kind": ["other"]})["both"]
        self.assertEqual(rep.flipping_fields, ())
        self.assertTrue(both(case(months=3, kind="other")))

    def test_role_recorded_on_probe(self):
        rep = sensitivity_probe([leave], case(), candidates={"dept": ["Logistics"]})["leave"]
        dept = [p for p in rep.probes if p.field == "dept"][0]
        self.assertIs(dept.role, Role.CONTEXTUAL)


if __name__ == "__main__":
    unittest.main()
