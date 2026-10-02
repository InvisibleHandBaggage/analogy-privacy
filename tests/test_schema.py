import json
import unittest

from analogy_privacy.schema import NARRATIVE_SOURCE, Case, CaseField, Role, project_decision_relevant, string_pairs


def make_case(**overrides):
    values = {"name": "Alice Nonesuch", "months": 6, "dept": "Accounting", "flag": True}
    values.update(overrides)
    roles = {"name": "identifying", "months": "decision_relevant", "dept": "contextual", "flag": "decision_relevant"}
    return Case.from_mapping(values, roles, narrative="Alice was away.")


class RoleTests(unittest.TestCase):
    def test_parse_accepts_enum_and_string(self):
        self.assertIs(Role.parse("identifying"), Role.IDENTIFYING)
        self.assertIs(Role.parse(Role.CONTEXTUAL), Role.CONTEXTUAL)

    def test_parse_rejects_unknown(self):
        with self.assertRaises(ValueError):
            Role.parse("secret")


class CaseTests(unittest.TestCase):
    def test_from_mapping_roles_and_values(self):
        c = make_case()
        self.assertEqual(c["months"], 6)
        self.assertIs(c.role_of("name"), Role.IDENTIFYING)
        self.assertEqual(c.names(), ("name", "months", "dept", "flag"))
        self.assertIn("dept", c)
        self.assertNotIn("salary", c)

    def test_from_mapping_requires_matching_keys(self):
        with self.assertRaises(ValueError):
            Case.from_mapping({"a": 1, "b": 2}, {"a": "identifying"})
        with self.assertRaises(ValueError):
            Case.from_mapping({"a": 1}, {"a": "identifying", "b": "contextual"})

    def test_duplicate_names_rejected(self):
        with self.assertRaises(ValueError):
            Case((CaseField("a", 1, "identifying"), CaseField("a", 2, "contextual")))

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            CaseField("a", 1, "nope")

    def test_projection_onto_decision_relevant(self):
        c = make_case()
        self.assertEqual(c.decision_view(), {"months": 6, "flag": True})
        self.assertEqual(project_decision_relevant(c), {"months": 6, "flag": True})
        self.assertEqual(c.project("identifying", Role.CONTEXTUAL), {"name": "Alice Nonesuch", "dept": "Accounting"})

    def test_with_values_is_non_destructive_and_keeps_roles(self):
        c = make_case()
        d = c.with_values({"months": 5})
        self.assertEqual(c["months"], 6)
        self.assertEqual(d["months"], 5)
        self.assertIs(d.role_of("months"), Role.DECISION_RELEVANT)
        with self.assertRaises(KeyError):
            c.with_values({"salary": 1})

    def test_with_roles(self):
        c = make_case().with_roles({"dept": "identifying"})
        self.assertIs(c.role_of("dept"), Role.IDENTIFYING)
        self.assertEqual(c["dept"], "Accounting")

    def test_digest_is_order_independent_and_value_sensitive(self):
        c = make_case()
        reordered = Case(tuple(reversed(c.fields)), c.narrative)
        self.assertEqual(c.digest(), reordered.digest())
        self.assertNotEqual(c.digest(), c.with_values({"months": 7}).digest())
        self.assertNotEqual(c.digest(), c.with_narrative("other").digest())

    def test_keyed_digest(self):
        c = make_case()
        self.assertNotEqual(c.digest(), c.digest(b"k1"))
        self.assertEqual(c.digest(b"k1"), c.digest(b"k1"))
        self.assertNotEqual(c.digest(b"k1"), c.digest(b"k2"))

    def test_digest_does_not_contain_text(self):
        self.assertNotIn("alice", make_case().digest().lower())

    def test_text_sources_skip_bool_and_none(self):
        c = make_case(flag=False)
        srcs = dict(c.text_sources())
        self.assertIn(NARRATIVE_SOURCE, srcs)
        self.assertNotIn("flag", srcs)
        self.assertEqual(srcs["months"], "6")
        c2 = Case((CaseField("x", None, "contextual"),))
        self.assertEqual(c2.text_sources(), [])

    def test_render_contains_fields(self):
        text = make_case().render()
        self.assertIn("name: Alice Nonesuch", text)
        self.assertTrue(text.startswith("Alice was away."))

    def test_to_dict_is_json_serialisable(self):
        json.dumps(make_case().to_dict())

    def test_string_pairs_only_changed_strings(self):
        a = make_case()
        b = a.with_values({"name": "Bruce Placeholder", "months": 9, "dept": "Accounting"})
        self.assertEqual(string_pairs(a, b), {"Bruce Placeholder": "Alice Nonesuch"})


if __name__ == "__main__":
    unittest.main()
