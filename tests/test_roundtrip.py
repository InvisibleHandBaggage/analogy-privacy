import dataclasses
import json
import unittest

from analogy_privacy.invariance import check_invariance
from analogy_privacy.leak_audit import audit_leaks
from analogy_privacy.roundtrip import (
    ClearanceRefused,
    back_map,
    mark_cleared,
    new_record,
    release_to_cloud,
    sign_off,
)
from analogy_privacy.schema import Case

ROLES = {
    "name": "identifying",
    "dept": "contextual",
    "months": "decision_relevant",
    "condition": "identifying",
}
POP = [{"dept": "Logistics", "months": 6}] * 6 + [{"dept": "Accounting", "months": 6}]


def original():
    return Case.from_mapping(
        {"name": "Alice Nonesuch", "dept": "Accounting", "months": 6, "condition": "cancer"}, ROLES
    )


def good_analogue():
    return Case.from_mapping(
        {"name": "Bruce Placeholder", "dept": "Logistics", "months": 6, "condition": "car crash"}, ROLES
    )


def policy(c):
    return "review" if c["months"] >= 6 else "standard"


MAPPING = {"Bruce Placeholder": "Alice Nonesuch", "Logistics": "Accounting", "car crash": "cancer"}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.o = original()
        self.a = good_analogue()
        self.inv = check_invariance(self.o, self.a, [policy])
        self.leak = audit_leaks(self.o, self.a, population=POP)
        self.rec = new_record(self.o, self.a, MAPPING)

    def approved(self, rec=None):
        return sign_off(rec or self.rec, "R. Eviewer", True, timestamp="2026-01-01T00:00:00+00:00")


class RecordTests(Fixture):
    def test_fixture_is_actually_clean(self):
        self.assertTrue(self.inv.passed)
        self.assertTrue(self.leak.passed)

    def test_record_holds_digest_not_original_text(self):
        shareable = json.dumps(self.rec.to_shareable_dict())
        for secret in ("Alice", "Nonesuch", "Accounting", "cancer"):
            self.assertNotIn(secret, shareable)
        self.assertEqual(self.rec.original_digest, self.o.digest())
        self.assertFalse(self.rec.digest_keyed)

    def test_mapping_is_local_only(self):
        local = json.dumps(self.rec.to_local_dict())
        self.assertIn("Alice Nonesuch", local)  # the mapping necessarily holds original fragments
        self.assertNotIn("mapping", self.rec.to_shareable_dict())
        self.assertNotIn("back_mapped_answer", self.rec.to_shareable_dict())

    def test_signoff_note_is_local_only(self):
        rec = sign_off(self.rec, "R. Eviewer", True, note="mentions Alice", timestamp="t")
        self.assertNotIn("Alice", json.dumps(rec.to_shareable_dict()))
        self.assertEqual(rec.to_local_dict()["signoff"]["note"], "mentions Alice")

    def test_keyed_digest_recorded(self):
        rec = new_record(self.o, self.a, MAPPING, key=b"local-secret")
        self.assertTrue(rec.digest_keyed)
        self.assertEqual(rec.original_digest, self.o.digest(b"local-secret"))
        self.assertNotEqual(rec.original_digest, self.o.digest())

    def test_signoff_requires_named_reviewer(self):
        with self.assertRaises(ValueError):
            sign_off(self.rec, "  ", True)


class ClearanceTests(Fixture):
    def test_cleared_when_everything_holds(self):
        rec = mark_cleared(self.approved(), self.inv, self.leak)
        self.assertEqual(rec.status, "cleared")
        self.assertTrue(rec.uniqueness_assessed)

    def test_refused_without_signoff(self):
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(self.rec, self.inv, self.leak)
        self.assertEqual(cm.exception.reasons, ("adequacy sign-off is missing",))

    def test_refused_with_rejected_signoff(self):
        rec = sign_off(self.rec, "R. Eviewer", False)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, self.inv, self.leak)
        self.assertIn("rejected", cm.exception.reasons[0])

    def test_signoff_goes_stale_if_analogue_changes(self):
        rec = self.approved()
        changed = dataclasses.replace(rec, analogue=rec.analogue.with_values({"dept": "Support"}))
        self.assertEqual(changed.signoff_status, "stale")
        inv = check_invariance(self.o, changed.analogue, [policy])
        leak = audit_leaks(self.o, changed.analogue, population=POP)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(changed, inv, leak)
        self.assertIn("stale", " ".join(cm.exception.reasons))

    def test_refused_when_invariance_fails(self):
        bad = self.a.with_values({"months": 5})
        rec = self.approved(new_record(self.o, bad, MAPPING))
        inv = check_invariance(self.o, bad, [policy])
        leak = audit_leaks(self.o, bad, population=POP)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, inv, leak)
        self.assertIn("policy invariance failed: policy", cm.exception.reasons[0])

    def test_refused_when_leak_audit_fails(self):
        bad = self.a.with_values({"name": "Bruce Nonesuch"})
        rec = self.approved(new_record(self.o, bad, MAPPING))
        inv = check_invariance(self.o, bad, [policy])
        leak = audit_leaks(self.o, bad, population=POP)
        self.assertFalse(leak.passed)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, inv, leak)
        self.assertIn("direct leaks", " ".join(cm.exception.reasons))

    def test_refused_when_uniqueness_flagged(self):
        bad = self.a.with_values({"dept": "Accounting"})  # unique in POP
        rec = self.approved(new_record(self.o, bad, MAPPING))
        inv = check_invariance(self.o, bad, [policy])
        leak = audit_leaks(self.o, bad, population=POP, allow=["accounting"])
        self.assertTrue(leak.direct.passed)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, inv, leak)
        self.assertIn("uniqueness", " ".join(cm.exception.reasons))

    def test_uniqueness_must_be_assessed_unless_opted_out(self):
        leak = audit_leaks(self.o, self.a)  # no population
        rec = self.approved()
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, self.inv, leak)
        self.assertIn("uniqueness not assessed", cm.exception.reasons[0])
        cleared = mark_cleared(rec, self.inv, leak, require_uniqueness=False)
        self.assertEqual(cleared.status, "cleared")
        self.assertFalse(cleared.uniqueness_assessed)  # the opt-out is on the record

    def test_reports_for_other_cases_are_refused(self):
        other = self.o.with_values({"dept": "Support"})
        inv = check_invariance(other, self.a, [policy])
        leak = audit_leaks(other, self.a, population=POP)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(self.approved(), inv, leak)
        self.assertEqual(len(cm.exception.reasons), 2)
        self.assertTrue(all("not computed on this record" in r for r in cm.exception.reasons))

    def test_all_reasons_are_reported_together(self):
        bad = self.a.with_values({"months": 5, "name": "Bruce Nonesuch"})
        rec = new_record(self.o, bad, MAPPING)
        inv = check_invariance(self.o, bad, [policy])
        leak = audit_leaks(self.o, bad, population=POP)
        with self.assertRaises(ClearanceRefused) as cm:
            mark_cleared(rec, inv, leak)
        self.assertGreaterEqual(len(cm.exception.reasons), 3)  # invariance, leaks, sign-off


class ReleaseTests(Fixture):
    def cleared(self):
        return mark_cleared(self.approved(), self.inv, self.leak)

    def test_refuses_uncleared_record_and_never_calls_send(self):
        calls = []
        with self.assertRaises(ClearanceRefused):
            release_to_cloud(self.approved(), lambda t: calls.append(t) or "x")
        self.assertEqual(calls, [])

    def test_sends_only_analogue_and_back_maps(self):
        sent = []

        def cloud(text):
            sent.append(text)
            return "Bruce Placeholder in Logistics: review. Bruce is fine."

        rec = release_to_cloud(self.cleared(), cloud, question="What next?")
        text = sent[0]
        self.assertIn("fictional", text)
        self.assertIn("What next?", text)
        self.assertIn("Bruce Placeholder", text)
        for secret in ("Alice", "Nonesuch", "Accounting", "cancer"):
            self.assertNotIn(secret, text)
        self.assertEqual(rec.sent_text, text)
        self.assertEqual(rec.back_mapped_answer, "Alice Nonesuch in Accounting: review. Bruce is fine.")
        self.assertEqual(rec.residue, ("bruce",))
        self.assertEqual(rec.replacements, {"bruce placeholder": 1, "logistics": 1})

    def test_shareable_record_after_release_has_no_original_text(self):
        rec = release_to_cloud(self.cleared(), lambda t: "Bruce Placeholder: review.")
        shareable = json.dumps(rec.to_shareable_dict())
        for secret in ("Alice", "Nonesuch", "Accounting", "cancer"):
            self.assertNotIn(secret, shareable)
        self.assertIn("Alice Nonesuch", json.dumps(rec.to_local_dict()))

    def test_refuses_if_signoff_withdrawn_after_clearance(self):
        rec = sign_off(self.cleared(), "R. Eviewer", False)
        with self.assertRaises(ClearanceRefused):
            release_to_cloud(rec, lambda t: "x")


class BackMapTests(unittest.TestCase):
    def test_longest_key_first_and_no_chaining(self):
        text, counts, _ = back_map("Bruce Placeholder met Bruce.", {"Bruce Placeholder": "Alice Nonesuch", "Bruce": "Bruce2"})
        self.assertEqual(text, "Alice Nonesuch met Bruce2.")
        self.assertEqual(counts, {"bruce placeholder": 1, "bruce": 1})

    def test_chained_mappings_do_not_cascade(self):
        text, _, _ = back_map("A and B", {"A": "B", "B": "C"})
        self.assertEqual(text, "B and C")

    def test_case_insensitive_whole_word(self):
        text, _, _ = back_map("LOGISTICS, logistics, Logisticsx, xlogistics", {"Logistics": "Accounting"})
        self.assertEqual(text, "Accounting, Accounting, Logisticsx, xlogistics")

    def test_possessive(self):
        text, _, _ = back_map("Bruce Placeholder's leave", {"Bruce Placeholder": "Alice Nonesuch"})
        self.assertEqual(text, "Alice Nonesuch's leave")

    def test_residue_flags_missed_partial_references(self):
        _, _, residue = back_map("Placeholder said hello", {"Bruce Placeholder": "Alice Nonesuch"})
        self.assertEqual(residue, ("placeholder",))

    def test_empty_mapping_is_identity(self):
        self.assertEqual(back_map("text", {}), ("text", {}, ()))

    def test_keys_differing_only_by_case_are_rejected(self):
        with self.assertRaises(ValueError):
            back_map("x", {"Bob": "A", "bob": "B"})


if __name__ == "__main__":
    unittest.main()
