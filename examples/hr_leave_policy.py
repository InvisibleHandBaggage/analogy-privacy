"""Toy HR leave-policy example. All data are synthetic; all names are fictional.

Run from the repository root:

    python -m examples.hr_leave_policy

Scenarios
---------
1. clean            analogue passes invariance and the leak audit.
2. leaks_a_token    analogue keeps a rare token (the surname): leak audit fails.
3. flips_outcome    analogue shifts a decision-relevant value across a threshold:
                    invariance fails; the sensitivity probe names the field.
4. unique_context   names are swapped but department and quarter are kept: the
                    direct-leak check passes (the user allow-listed them), the
                    uniqueness estimate flags k = 1.
5. mistagged        a second policy reads the specific condition; the condition
                    was tagged `identifying` and swapped: invariance fails and
                    the probe reports a mis-tagged field.
6. condition_kept   condition re-tagged `decision_relevant`: the category is
                    kept, identifiers are changed, invariance holds, and the
                    anonymity set shrinks (reported, not hidden).
7. roundtrip        sign-off gate, release to a (fake) cloud, back-mapping.

The two policies and the roster are invented for illustration. They are not
any real organisation's rules or data.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

if __package__ in (None, ""):  # allow `python examples/hr_leave_policy.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analogy_privacy import (  # noqa: E402
    Case,
    ClearanceRefused,
    RuleBasedTransformer,
    audit_leaks,
    check_invariance,
    mark_cleared,
    new_record,
    release_to_cloud,
    sensitivity_probe,
    sign_off,
)

# ---------------------------------------------------------------------------
# Toy policies (invented rules)
# ---------------------------------------------------------------------------


def standard_leave_policy(case: Case) -> str:
    """Absence of 6+ months for a medical reason goes to case review."""
    if case["absence_months"] < 6:
        return "standard_leave"
    return "case_review" if case["reason_category"] == "medical" else "manager_discussion"


def condition_specific_policy(case: Case) -> str:
    """Variant: for long medical absences the specific condition matters."""
    if case["absence_months"] < 6:
        return "standard_leave"
    if case["reason_category"] != "medical":
        return "manager_discussion"
    condition = case["condition"]
    if condition == "cancer":
        return "extended_paid_leave"
    if condition == "stroke":
        return "phased_return_plan"
    return "case_review"


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

# Roles when only `standard_leave_policy` is of interest.
ROLES = {
    "employee_name": "identifying",
    "employee_id": "identifying",
    "department": "contextual",
    "absence_quarter": "contextual",
    "absence_months": "decision_relevant",
    "reason_category": "decision_relevant",
    "condition": "identifying",
}
# Roles when `condition_specific_policy` is also of interest.
ROLES_WITH_CONDITION = dict(ROLES, condition="decision_relevant")

# Boilerplate shared by construction between original and analogue narratives.
BOILERPLATE = ("was absent for", "months last year", "reason")

# Hand-built absence log (the reference population). Cells:
# (department, quarter, months, reason_category, condition, number of records)
_CELLS = [
    ("Accounting", "Q3", 6, "medical", "cancer", 1),  # the original case itself
    ("Accounting", "Q1", 2, "personal", "n/a", 6),
    ("Accounting", "Q2", 1, "personal", "n/a", 4),
    ("Accounting", "Q4", 3, "medical", "fracture", 3),
    ("Logistics", "Q1", 6, "medical", "cancer", 5),
    ("Logistics", "Q1", 6, "medical", "stroke", 2),
    ("Logistics", "Q1", 6, "medical", "car crash", 3),
    ("Logistics", "Q3", 2, "personal", "n/a", 7),
    ("Engineering", "Q2", 6, "medical", "cancer", 4),
    ("Engineering", "Q3", 6, "medical", "stroke", 2),
    ("Engineering", "Q4", 1, "personal", "n/a", 8),
    ("Support", "Q4", 6, "medical", "cancer", 3),
    ("Support", "Q2", 2, "medical", "fracture", 5),
    ("Support", "Q1", 4, "personal", "n/a", 4),
]


def roster() -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for dept, quarter, months, reason, condition, n in _CELLS:
        for _ in range(n):
            out.append(
                {
                    "department": dept,
                    "absence_quarter": quarter,
                    "absence_months": months,
                    "reason_category": reason,
                    "condition": condition,
                }
            )
    return out


def narrative(v: Dict[str, Any]) -> str:
    return "%s of %s was absent for %d months last year. Reason: %s." % (
        v["employee_name"],
        v["department"],
        v["absence_months"],
        v["condition"],
    )


ORIGINAL_VALUES = {
    "employee_name": "Alice Nonesuch",
    "employee_id": "E-1047",
    "department": "Accounting",
    "absence_quarter": "Q3",
    "absence_months": 6,
    "reason_category": "medical",
    "condition": "cancer",
}


def original_case(roles: Dict[str, str] = ROLES) -> Case:
    return Case.from_mapping(ORIGINAL_VALUES, roles, narrative(ORIGINAL_VALUES))


def analogue_case(roles: Dict[str, str], **overrides: Any) -> Case:
    values = dict(
        ORIGINAL_VALUES,
        employee_name="Bruce Placeholder",
        employee_id="E-9021",
        department="Logistics",
        absence_quarter="Q1",
        condition="car crash",
    )
    values.update(overrides)
    return Case.from_mapping(values, roles, narrative(values))


SWAPS = {
    "employee_name": {"Alice Nonesuch": "Bruce Placeholder"},
    "employee_id": {"E-1047": "E-9021"},
    "department": {"Accounting": "Logistics"},
    "absence_quarter": {"Q3": "Q1"},
    "condition": {"cancer": "car crash"},
}

# ---------------------------------------------------------------------------
# Scenarios. Each returns the objects it produced, so tests can assert on them.
# ---------------------------------------------------------------------------


def _evaluate(original: Case, analogue: Case, policies: Any, **audit_kwargs: Any) -> Dict[str, Any]:
    return {
        "original": original,
        "analogue": analogue,
        "invariance": check_invariance(original, analogue, policies),
        "leak": audit_leaks(original, analogue, population=roster(), **audit_kwargs),
    }


def scenario_clean() -> Dict[str, Any]:
    original = original_case()
    analogue = RuleBasedTransformer(SWAPS).transform(original)
    result = _evaluate(original, analogue.analogue, [standard_leave_policy], allow=BOILERPLATE)
    result["transform"] = analogue
    return result


def scenario_leaks_a_token() -> Dict[str, Any]:
    original = original_case()
    analogue = analogue_case(ROLES, employee_name="Bruce Nonesuch")
    return _evaluate(original, analogue, [standard_leave_policy], allow=BOILERPLATE)


def scenario_flips_outcome() -> Dict[str, Any]:
    original = original_case()
    analogue = analogue_case(ROLES, absence_months=5)
    result = _evaluate(original, analogue, [standard_leave_policy], allow=BOILERPLATE)
    result["sensitivity"] = sensitivity_probe([standard_leave_policy], original, other=analogue)
    return result


def scenario_unique_context() -> Dict[str, Any]:
    original = original_case()
    analogue = analogue_case(ROLES, department="Accounting", absence_quarter="Q3")
    # A reviewer who thinks department and quarter are harmless allow-lists them:
    return _evaluate(
        original,
        analogue,
        [standard_leave_policy],
        allow=BOILERPLATE + ("accounting", "q3"),
    )


def scenario_mistagged() -> Dict[str, Any]:
    policies = [standard_leave_policy, condition_specific_policy]
    original = original_case(ROLES)  # condition wrongly tagged identifying
    analogue = RuleBasedTransformer(SWAPS).transform(original).analogue
    result = _evaluate(original, analogue, policies, allow=BOILERPLATE)
    result["sensitivity"] = sensitivity_probe(policies, original, other=analogue)
    return result


def scenario_condition_kept() -> Dict[str, Any]:
    policies = [standard_leave_policy, condition_specific_policy]
    original = original_case(ROLES_WITH_CONDITION)
    swaps = {k: v for k, v in SWAPS.items() if k != "condition"}
    analogue = RuleBasedTransformer(swaps).transform(original).analogue
    result = _evaluate(original, analogue, policies, allow=BOILERPLATE)
    result["sensitivity"] = sensitivity_probe(policies, original, other=analogue)
    return result


CANNED_CLOUD_ANSWER = (
    "Bruce Placeholder was absent for 6 months for a medical reason, so the policy sends the "
    "case to review. Logistics management should confirm the dates; Bruce should be told that "
    "the review is routine."
)
QUESTION = "Under the leave policy, what happens next for this employee, and what should the manager do?"


def scenario_roundtrip() -> Dict[str, Any]:
    ev = scenario_clean()
    record = new_record(ev["original"], ev["analogue"], ev["transform"].mapping)

    refusal = None
    try:  # no sign-off yet
        mark_cleared(record, ev["invariance"], ev["leak"])
    except ClearanceRefused as exc:
        refusal = exc

    record = sign_off(record, reviewer="J. Reviewer (HR lead)", approved=True, note="adequate for leave question")
    record = mark_cleared(record, ev["invariance"], ev["leak"])

    sent: List[str] = []

    def fake_cloud(text: str) -> str:  # stands in for the cloud call
        sent.append(text)
        return CANNED_CLOUD_ANSWER

    record = release_to_cloud(record, fake_cloud, question=QUESTION)
    return {"record": record, "refusal": refusal, "sent": sent, **ev}


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _show(title: str, ev: Dict[str, Any]) -> None:
    print("=== %s" % title)
    print("analogue narrative: %s" % ev["analogue"].narrative)
    print("invariance: %s" % ("PASS" if ev["invariance"].passed else "FAIL"))
    print(ev["invariance"].summary())
    print("leak audit: %s" % ("PASS" if ev["leak"].passed else "FAIL"))
    print(ev["leak"].summary())
    for name, rep in ev.get("sensitivity", {}).items():
        print(
            "sensitivity [%s]: flips on %s; mis-tagged: %s"
            % (name, list(rep.flipping_fields) or "none", list(rep.misclassified) or "none")
        )
    print()


def main() -> None:
    print("Synthetic data only. Roster size: %d absence records.\n" % len(roster()))
    print("Original (never leaves the machine): %s\n" % original_case().narrative)
    _show("1. clean analogue", scenario_clean())
    _show("2. analogue leaks a rare token (surname kept)", scenario_leaks_a_token())
    _show("3. analogue flips the outcome (5 months instead of 6)", scenario_flips_outcome())
    _show("4. names swapped, context kept (uniqueness)", scenario_unique_context())
    _show("5. condition mis-tagged identifying, second policy reads it", scenario_mistagged())
    _show("6. condition tagged decision-relevant and kept", scenario_condition_kept())

    print("=== 7. round trip")
    rt = scenario_roundtrip()
    print("clearance before sign-off refused: %s" % "; ".join(rt["refusal"].reasons))
    rec = rt["record"]
    print("status after sign-off: %s (sign-off: %s)" % (rec.status, rec.signoff_status))
    print("--- text sent to the cloud:\n%s" % rec.sent_text)
    print("--- cloud answer:\n%s" % rec.cloud_answer)
    print("--- back-mapped answer (local):\n%s" % rec.back_mapped_answer)
    print("residue to review by hand (analogue words left after back-mapping): %s" % list(rec.residue))
    print("shareable record keys: %s" % sorted(rec.to_shareable_dict()))
    print("local-only extra keys: %s" % sorted(set(rec.to_local_dict()) - set(rec.to_shareable_dict())))


if __name__ == "__main__":
    main()
