import json
import os
import subprocess
import sys
import textwrap
import unittest
import urllib.error
import urllib.request
from unittest import mock

from analogy_privacy.schema import Case, Role
from analogy_privacy.transformers import (
    MappingConflict,
    OllamaTransformer,
    RuleBasedTransformer,
    TransformError,
    Transformer,
    parse_model_output,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROLES = {
    "name": "identifying",
    "dept": "contextual",
    "months": "decision_relevant",
    "condition": "identifying",
}


def original(narrative="Alice Nonesuch of Accounting was away for 6 months. Reason: cancer."):
    return Case.from_mapping(
        {"name": "Alice Nonesuch", "dept": "Accounting", "months": 6, "condition": "cancer"}, ROLES, narrative
    )


SWAPS = {
    "name": {"Alice Nonesuch": "Bruce Placeholder"},
    "dept": {"Accounting": "Logistics"},
    "condition": {"cancer": "car crash"},
}


class RuleBasedTests(unittest.TestCase):
    def test_is_a_transformer(self):
        self.assertIsInstance(RuleBasedTransformer(SWAPS), Transformer)

    def test_swaps_fields_and_narrative_and_builds_mapping(self):
        res = RuleBasedTransformer(SWAPS).transform(original())
        a = res.analogue
        self.assertEqual((a["name"], a["dept"], a["condition"], a["months"]), ("Bruce Placeholder", "Logistics", "car crash", 6))
        self.assertEqual(a.narrative, "Bruce Placeholder of Logistics was away for 6 months. Reason: car crash.")
        self.assertEqual(
            res.mapping, {"Bruce Placeholder": "Alice Nonesuch", "Logistics": "Accounting", "car crash": "cancer"}
        )

    def test_roles_are_preserved(self):
        a = RuleBasedTransformer(SWAPS).transform(original()).analogue
        self.assertIs(a.role_of("months"), Role.DECISION_RELEVANT)
        self.assertIs(a.role_of("name"), Role.IDENTIFYING)

    def test_original_is_not_mutated(self):
        o = original()
        RuleBasedTransformer(SWAPS).transform(o)
        self.assertEqual(o["name"], "Alice Nonesuch")

    def test_unmapped_identifying_field_raises(self):
        with self.assertRaises(TransformError):
            RuleBasedTransformer({"name": {"Alice Nonesuch": "Bruce Placeholder"}}).transform(original())

    def test_unmapped_identifying_field_can_be_allowed(self):
        t = RuleBasedTransformer({"name": {"Alice Nonesuch": "Bruce Placeholder"}}, require_identifying_swaps=False)
        self.assertEqual(t.transform(original()).analogue["condition"], "cancer")

    def test_unmapped_contextual_and_decision_relevant_are_kept(self):
        t = RuleBasedTransformer({"name": {"Alice Nonesuch": "B"}, "condition": {"cancer": "fall"}})
        a = t.transform(original()).analogue
        self.assertEqual((a["dept"], a["months"]), ("Accounting", 6))

    def test_keys_match_case_insensitively(self):
        a = RuleBasedTransformer(SWAPS).transform(original().with_values({"name": "ALICE NONESUCH"})).analogue
        self.assertEqual(a["name"], "Bruce Placeholder")

    def test_non_string_swap_is_applied_but_not_in_mapping(self):
        t = RuleBasedTransformer({**SWAPS, "months": {6: 7}})
        res = t.transform(original())
        self.assertEqual(res.analogue["months"], 7)
        self.assertNotIn("7", res.mapping)

    def test_ambiguous_back_mapping_is_rejected(self):
        swaps = {"name": {"Alice Nonesuch": "Same"}, "condition": {"cancer": "same"}}
        with self.assertRaises(MappingConflict):
            RuleBasedTransformer(swaps).transform(original())

    def test_narrative_only_swaps(self):
        t = RuleBasedTransformer(SWAPS, narrative_swaps={"Alice": "Bruce"})
        a = t.transform(original("Alice Nonesuch is away. Alice returns.")).analogue
        self.assertEqual(a.narrative, "Bruce Placeholder is away. Bruce returns.")


class OllamaTests(unittest.TestCase):
    GOOD_REPLY = {
        "fields": {"name": "Bruce Placeholder", "dept": "Logistics", "months": 6, "condition": "car crash"},
        "narrative": "Bruce Placeholder of Logistics was away for 6 months. Reason: car crash.",
        "replacements": [["Alice Nonesuch", "Bruce Placeholder"]],
    }

    def fake_post(self, reply):
        calls = []

        def post(url, payload):
            calls.append((url, payload))
            return {"message": {"role": "assistant", "content": json.dumps(reply)}, "done": True}

        post.calls = calls
        return post

    def test_docstring_marks_it_untested(self):
        self.assertIn("UNTESTED", OllamaTransformer.__doc__)

    def test_refuses_non_loopback_host(self):
        with self.assertRaises(ValueError):
            OllamaTransformer("m", host="http://example.com:11434")
        with self.assertRaises(ValueError):
            OllamaTransformer("m", host="http://10.0.0.5:11434")

    def test_accepts_loopback_hosts(self):
        for host in ("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434", "http://127.0.0.2:1"):
            OllamaTransformer("m", host=host)

    def test_remote_host_needs_explicit_override(self):
        OllamaTransformer("m", host="http://example.com", allow_remote=True)

    def test_no_network_at_import_or_construction(self):
        code = textwrap.dedent(
            """
            import socket

            def boom(*a, **k):
                raise AssertionError("network used")

            socket.socket.connect = boom
            socket.getaddrinfo = boom
            import analogy_privacy
            from analogy_privacy.transformers import OllamaTransformer
            OllamaTransformer("m")
            print("ok")
            """
        )
        proc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual((proc.returncode, proc.stdout.strip()), (0, "ok"), proc.stderr)

    def test_prompt_contains_roles_values_and_notes(self):
        t = OllamaTransformer("m", policy_notes="the policy distinguishes cancer from stroke", post=self.fake_post(self.GOOD_REPLY))
        msgs = t.build_messages(original())
        self.assertEqual([m["role"] for m in msgs], ["system", "user"])
        self.assertIn("FICTIONAL ANALOGUE", msgs[0]["content"])
        self.assertIn('"decision_relevant"', msgs[1]["content"])
        self.assertIn("Alice Nonesuch", msgs[1]["content"])
        self.assertIn("cancer from stroke", msgs[1]["content"])

    def test_transform_with_injected_post(self):
        post = self.fake_post(self.GOOD_REPLY)
        res = OllamaTransformer("llama-test", post=post).transform(original())
        url, payload = post.calls[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual((payload["model"], payload["stream"], payload["format"]), ("llama-test", False, "json"))
        self.assertEqual(res.analogue["name"], "Bruce Placeholder")
        self.assertEqual(res.mapping["Bruce Placeholder"], "Alice Nonesuch")
        self.assertEqual(res.warnings, ())

    def test_unexpected_response_shape(self):
        t = OllamaTransformer("m", post=lambda u, p: {"oops": 1})
        with self.assertRaises(TransformError):
            t.transform(original())

    def test_http_post_uses_no_proxy_and_wraps_errors(self):
        t = OllamaTransformer("m")
        opener = mock.Mock()
        opener.open.side_effect = urllib.error.URLError("refused")
        with mock.patch.object(urllib.request, "build_opener", return_value=opener) as bo:
            with self.assertRaises(TransformError):
                t.transform(original())
        handler = bo.call_args[0][0]
        self.assertIsInstance(handler, urllib.request.ProxyHandler)
        self.assertEqual(handler.proxies, {})

    def test_http_post_parses_json_body(self):
        body = json.dumps({"message": {"content": json.dumps(self.GOOD_REPLY)}}).encode()
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        opener = mock.Mock()
        opener.open.return_value = resp
        with mock.patch.object(urllib.request, "build_opener", return_value=opener):
            res = OllamaTransformer("m").transform(original())
        self.assertEqual(res.analogue["dept"], "Logistics")


class ParseTests(unittest.TestCase):
    def reply(self, **changes):
        base = {
            "fields": {"name": "Bruce Placeholder", "dept": "Logistics", "months": 6, "condition": "car crash"},
            "narrative": "Bruce Placeholder of Logistics was away.",
            "replacements": [],
        }
        base.update(changes)
        return json.dumps(base)

    def test_valid_reply(self):
        res = parse_model_output(original(), self.reply())
        self.assertEqual(res.analogue["condition"], "car crash")
        self.assertEqual(res.warnings, ())

    def test_invalid_json(self):
        with self.assertRaises(TransformError):
            parse_model_output(original(), "not json")

    def test_missing_fields_object(self):
        with self.assertRaises(TransformError):
            parse_model_output(original(), json.dumps({"narrative": "x"}))

    def test_field_names_must_match_exactly(self):
        with self.assertRaises(TransformError):
            parse_model_output(original(), self.reply(fields={"name": "B"}))
        extra = {"name": "B", "dept": "L", "months": 6, "condition": "c", "salary": 1}
        with self.assertRaises(TransformError):
            parse_model_output(original(), self.reply(fields=extra))

    def test_numeric_coercion_and_type_errors(self):
        ok = {"name": "B", "dept": "L", "months": "6", "condition": "c"}
        self.assertEqual(parse_model_output(original(), self.reply(fields=ok)).analogue["months"], 6)
        bad = {"name": "B", "dept": "L", "months": "six", "condition": "c"}
        with self.assertRaises(TransformError):
            parse_model_output(original(), self.reply(fields=bad))
        notstr = {"name": 5, "dept": "L", "months": 6, "condition": "c"}
        with self.assertRaises(TransformError):
            parse_model_output(original(), self.reply(fields=notstr))

    def test_warns_when_decision_relevant_changed_or_identifying_unchanged(self):
        fields = {"name": "Alice Nonesuch", "dept": "L", "months": 5, "condition": "c"}
        res = parse_model_output(original(), self.reply(fields=fields))
        joined = " ".join(res.warnings)
        self.assertIn("decision-relevant field 'months' was changed", joined)
        self.assertIn("identifying field 'name' was left unchanged", joined)

    def test_unverifiable_replacements_are_ignored_with_warning(self):
        res = parse_model_output(original(), self.reply(replacements=[["Nobody Here", "Bruce Placeholder"], ["x"]]))
        self.assertEqual(len(res.warnings), 2)
        self.assertNotIn("Nobody Here", res.mapping.values())

    def test_malformed_replacements_ignored(self):
        res = parse_model_output(original(), self.reply(replacements="oops"))
        self.assertTrue(any("malformed" in w for w in res.warnings))

    def test_verified_replacement_enters_mapping(self):
        res = parse_model_output(original(), self.reply(replacements=[["Accounting", "Logistics"]]))
        self.assertEqual(res.mapping["Logistics"], "Accounting")


if __name__ == "__main__":
    unittest.main()
