import unittest

from analogy_privacy.leak_audit import (
    Severity,
    audit_leaks,
    check_direct_leaks,
    estimate_uniqueness,
)
from analogy_privacy.schema import Case

ROLES = {
    "name": "identifying",
    "dept": "contextual",
    "quarter": "contextual",
    "months": "decision_relevant",
    "kind": "decision_relevant",
}


def case(narrative="", **kw):
    values = {"name": "Alice Nonesuch", "dept": "Accounting", "quarter": "Q3", "months": 6, "kind": "medical"}
    values.update(kw)
    return Case.from_mapping(values, ROLES, narrative)


def kinds(report, blocking=False):
    src = report.blocking if blocking else report.findings
    return sorted((f.kind, f.text) for f in src)


class DirectLeakTests(unittest.TestCase):
    def test_clean_analogue_has_no_blocking_findings(self):
        rep = check_direct_leaks(case(), case(name="Bruce Placeholder", dept="Logistics", quarter="Q1"))
        self.assertTrue(rep.passed)
        self.assertEqual(kinds(rep, blocking=True), [])

    def test_shared_decision_relevant_values_are_reported_as_allowed(self):
        rep = check_direct_leaks(case(), case(name="Bruce Placeholder", dept="Logistics", quarter="Q1"))
        shared = {f.text for f in rep.findings if f.kind == "field_value"}
        self.assertEqual(shared, {"6", "medical"})
        self.assertTrue(all(f.allowed for f in rep.findings))

    def test_decision_relevant_not_auto_allowed_when_disabled(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        rep = check_direct_leaks(case(), a, allow_decision_relevant=False)
        self.assertFalse(rep.passed)
        self.assertEqual({f.text for f in rep.blocking}, {"6", "medical"})

    def test_surname_reuse_is_high_severity(self):
        rep = check_direct_leaks(case(), case(name="Bruce Nonesuch", dept="Logistics", quarter="Q1"))
        self.assertFalse(rep.passed)
        hit = [f for f in rep.blocking if f.kind == "identifying_token"][0]
        self.assertEqual((hit.text, hit.severity, hit.source), ("nonesuch", Severity.HIGH, "name"))

    def test_whole_identifying_value_reuse(self):
        rep = check_direct_leaks(case(), case(dept="Logistics", quarter="Q1"))
        self.assertEqual(kinds(rep, blocking=True), [("field_value", "alice nonesuch")])

    def test_case_and_accent_insensitive(self):
        a = Case.from_mapping({"name": "José Quill"}, {"name": "identifying"})
        b = Case.from_mapping({"name": "JOSE QUILL"}, {"name": "identifying"})
        self.assertFalse(check_direct_leaks(a, b).passed)

    def test_identifying_token_in_narrative_of_analogue_is_caught(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1", narrative="Bruce, formerly Nonesuch, left.")
        self.assertFalse(check_direct_leaks(case(), a).passed)

    def test_allow_list_demotes_but_keeps_finding_visible(self):
        a = case(name="Bruce Placeholder", quarter="Q1")  # department Accounting kept
        blocked = check_direct_leaks(case(), a)
        self.assertEqual(kinds(blocked, blocking=True), [("field_value", "accounting")])
        ok = check_direct_leaks(case(), a, allow=["accounting"])
        self.assertTrue(ok.passed)
        self.assertIn(("field_value", "accounting"), kinds(ok))
        self.assertTrue([f for f in ok.findings if f.text == "accounting"][0].allowed)

    def test_short_contextual_value_is_not_silently_allowed(self):
        a = case(name="Bruce Placeholder", dept="Logistics")  # quarter Q3 kept
        rep = check_direct_leaks(case(), a)
        self.assertEqual(kinds(rep, blocking=True), [("field_value", "q3")])

    def test_shared_phrase_in_narrative(self):
        o = case(narrative="Alice Nonesuch was absent for 6 months after surgery at Thornbury Wing.")
        a = case(
            name="Bruce Placeholder",
            dept="Logistics",
            quarter="Q1",
            narrative="Bruce Placeholder was absent for 6 months after a fall at Thornbury Wing.",
        )
        rep = check_direct_leaks(o, a)
        phrases = [f.text for f in rep.findings if f.kind == "phrase"]
        self.assertIn("at thornbury wing", phrases)
        self.assertFalse(rep.passed)

    def test_phrase_allowed_when_all_content_tokens_allowed(self):
        o = case(narrative="Alice Nonesuch was absent for 6 months.")
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1", narrative="Bruce Placeholder was absent for 6 months.")
        self.assertFalse(check_direct_leaks(o, a).passed)
        self.assertTrue(check_direct_leaks(o, a, allow=["absent", "months"]).passed)

    def test_single_shared_word_is_low_and_does_not_fail_by_default(self):
        o = case(narrative="Alice Nonesuch attended a tribunal.")
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1", narrative="Bruce Placeholder avoided a tribunal.")
        rep = check_direct_leaks(o, a)
        self.assertEqual([f for f in rep.findings if f.kind == "phrase"], [])  # "a tribunal" has one content word
        low = [f for f in rep.findings if f.kind == "token"]
        self.assertEqual([f.text for f in low], ["tribunal"])
        self.assertEqual(low[0].severity, Severity.LOW)
        self.assertTrue(rep.passed)
        self.assertFalse(check_direct_leaks(o, a, fail_at=Severity.LOW).passed)

    def test_common_words_suppress_low_findings(self):
        o = case(narrative="Alice Nonesuch attended a tribunal.")
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1", narrative="Bruce Placeholder avoided a tribunal.")
        rep = check_direct_leaks(o, a, common_words=["tribunal"])
        self.assertEqual([f for f in rep.findings if f.kind == "token"], [])

    def test_numbers_in_identifying_fields_are_caught(self):
        o = Case.from_mapping({"emp_id": 104729}, {"emp_id": "identifying"})
        a = Case.from_mapping({"note": "see file 104729"}, {"note": "contextual"})
        self.assertFalse(check_direct_leaks(o, a).passed)


POP = (
    [{"dept": "Accounting", "quarter": "Q3", "months": 6, "kind": "medical"}]  # the original, unique
    + [{"dept": "Logistics", "quarter": "Q1", "months": 6, "kind": "medical"}] * 6
    + [{"dept": "Logistics", "quarter": "Q2", "months": 2, "kind": "personal"}] * 3
)


class UniquenessTests(unittest.TestCase):
    def test_counts_and_default_quasi_identifiers(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        rep = estimate_uniqueness(a, POP, original=case())
        f = rep.get("analogue")
        self.assertEqual(f.k, 6)
        self.assertEqual(f.quasi_identifiers, ("dept", "quarter", "months", "kind"))  # name excluded
        self.assertFalse(f.flagged)
        self.assertEqual(rep.population_size, 10)

    def test_original_scope_is_informational(self):
        rep = estimate_uniqueness(case(name="Bruce Placeholder", dept="Logistics", quarter="Q1"), POP, original=case())
        o = rep.get("original")
        self.assertEqual(o.k, 1)
        self.assertFalse(o.flagged)

    def test_unique_combination_is_flagged(self):
        a = case(name="Bruce Placeholder")  # dept, quarter kept: unique in POP
        rep = estimate_uniqueness(a, POP, original=case())
        self.assertEqual(rep.get("analogue").k, 1)
        self.assertTrue(rep.get("analogue").flagged)
        self.assertTrue(rep.get("shared_with_original").flagged)
        self.assertFalse(rep.passed)

    def test_shared_scope_uses_only_agreeing_attributes(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        s = estimate_uniqueness(a, POP, original=case()).get("shared_with_original")
        self.assertEqual(s.quasi_identifiers, ("months", "kind"))
        self.assertEqual(s.k, 7)

    def test_absent_combination_is_not_flagged_for_analogue(self):
        a = case(name="Bruce Placeholder", dept="Engineering", quarter="Q4")
        f = estimate_uniqueness(a, POP).get("analogue")
        self.assertEqual(f.k, 0)
        self.assertFalse(f.flagged)
        self.assertIn("absent", f.note)

    def test_original_not_in_population_flags_shared_scope(self):
        pop = [{"dept": "Logistics", "quarter": "Q1", "months": 2, "kind": "personal"}] * 8
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        s = estimate_uniqueness(a, pop, original=case()).get("shared_with_original")
        self.assertEqual(s.k, 0)
        self.assertTrue(s.flagged)
        self.assertIn("unreliable", s.note)

    def test_threshold_is_respected(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        self.assertFalse(estimate_uniqueness(a, POP, k_threshold=6).get("analogue").flagged)
        self.assertTrue(estimate_uniqueness(a, POP, k_threshold=7).get("analogue").flagged)

    def test_custom_matcher_generalises(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1", months=7)
        exact = estimate_uniqueness(a, POP, quasi_identifiers=["dept", "months"]).get("analogue")
        banded = estimate_uniqueness(
            a, POP, quasi_identifiers=["dept", "months"], matchers={"months": lambda pop, c: (pop >= 6) == (c >= 6)}
        ).get("analogue")
        self.assertEqual((exact.k, banded.k), (0, 6))

    def test_missing_quasi_identifier_in_population_raises(self):
        with self.assertRaises(ValueError):
            estimate_uniqueness(case(), [{"dept": "Accounting"}])  # quarter etc. absent

    def test_unknown_quasi_identifier_raises(self):
        with self.assertRaises(ValueError):
            estimate_uniqueness(case(), POP, quasi_identifiers=["salary"])

    def test_empty_population_raises(self):
        with self.assertRaises(ValueError):
            estimate_uniqueness(case(), [])

    def test_record_missing_a_field_does_not_match(self):
        pop = [{"dept": "Logistics"}, {"dept": "Logistics", "quarter": "Q1"}]
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        self.assertEqual(estimate_uniqueness(a, pop, quasi_identifiers=["dept", "quarter"]).get("analogue").k, 1)

    def test_string_matching_is_case_insensitive(self):
        pop = [{"dept": "LOGISTICS"}]
        a = case(dept="logistics")
        self.assertEqual(estimate_uniqueness(a, pop, quasi_identifiers=["dept"]).get("analogue").k, 1)


class CombinedAuditTests(unittest.TestCase):
    def test_not_assessed_without_population(self):
        a = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        rep = audit_leaks(case(), a)
        self.assertFalse(rep.uniqueness_assessed)
        self.assertTrue(rep.passed)  # direct part only
        self.assertIn("NOT ASSESSED", rep.summary())

    def test_passes_only_if_both_parts_pass(self):
        clean = case(name="Bruce Placeholder", dept="Logistics", quarter="Q1")
        self.assertTrue(audit_leaks(case(), clean, population=POP).passed)
        leaky_name = case(name="Bruce Nonesuch", dept="Logistics", quarter="Q1")
        self.assertFalse(audit_leaks(case(), leaky_name, population=POP).passed)
        unique = case(name="Bruce Placeholder")
        rep = audit_leaks(case(), unique, population=POP, allow=["accounting", "q3"])
        self.assertTrue(rep.direct.passed)
        self.assertFalse(rep.passed)
        self.assertIn("FLAGGED", rep.summary())


if __name__ == "__main__":
    unittest.main()
