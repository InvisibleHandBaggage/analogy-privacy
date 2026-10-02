"""Policy-invariance check and field-sensitivity probe.

A *policy* is a callable `Case -> outcome`. The check asks, for each supplied
policy, whether the outcome on the analogue is equivalent to the outcome on
the original. It establishes nothing about policies that were not supplied.

Policy outcomes should be class labels (for example "case_review"), not text
that embeds case content: embedded content would both fail equality and leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .schema import Case, Role

Policy = Callable[[Case], Any]
Equivalence = Callable[[Any, Any], bool]
PolicySet = Union[Policy, Sequence[Policy], Mapping[str, Policy]]
EquivalenceSpec = Union[None, Equivalence, Mapping[str, Equivalence]]


def _normalise_policies(policies: PolicySet) -> List[Tuple[str, Policy]]:
    if callable(policies):
        policies = [policies]  # type: ignore[list-item]
    if isinstance(policies, Mapping):
        items = list(policies.items())
    else:
        items = [(getattr(p, "__name__", None) or "policy_%d" % i, p) for i, p in enumerate(policies)]
    if not items:
        raise ValueError("at least one policy is required (an empty set would pass vacuously)")
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("policy names must be unique (pass a {name: policy} mapping): %s" % names)
    for name, p in items:
        if not callable(p):
            raise TypeError("policy %r is not callable" % name)
    return items


def _equivalence_for(name: str, spec: EquivalenceSpec) -> Equivalence:
    if spec is None:
        return lambda a, b: a == b
    if callable(spec):
        return spec
    return spec.get(name, lambda a, b: a == b)


# ---------------------------------------------------------------------------
# Invariance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyResult:
    policy: str
    original_outcome: Any
    analogue_outcome: Any
    equivalent: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class InvarianceReport:
    results: Tuple[PolicyResult, ...]
    original_fp: str
    analogue_fp: str

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(r.equivalent and r.error is None for r in self.results)

    @property
    def failures(self) -> Tuple[PolicyResult, ...]:
        return tuple(r for r in self.results if not (r.equivalent and r.error is None))

    def summary(self) -> str:
        lines = []
        for r in self.results:
            if r.error:
                lines.append("  %-28s ERROR: %s" % (r.policy, r.error))
            else:
                mark = "same" if r.equivalent else "DIFFERENT"
                lines.append("  %-28s %s -> %s  [%s]" % (r.policy, r.original_outcome, r.analogue_outcome, mark))
        return "\n".join(lines)


def check_invariance(
    original: Case,
    analogue: Case,
    policies: PolicySet,
    equivalence: EquivalenceSpec = None,
) -> InvarianceReport:
    """Evaluate each policy on both cases and compare outcomes.

    `equivalence` is either one function applied to every policy, or a mapping
    policy-name -> function (policies not in the mapping use `==`). A policy
    or equivalence function that raises is recorded as a failure (fail closed).
    """
    results: List[PolicyResult] = []
    for name, policy in _normalise_policies(policies):
        eq = _equivalence_for(name, equivalence)
        try:
            a = policy(original)
            b = policy(analogue)
        except Exception as exc:  # noqa: BLE001 -- recorded, not swallowed
            results.append(PolicyResult(name, None, None, False, "%s: %s" % (type(exc).__name__, exc)))
            continue
        try:
            same = bool(eq(a, b))
        except Exception as exc:  # noqa: BLE001
            results.append(PolicyResult(name, a, b, False, "equivalence raised %s: %s" % (type(exc).__name__, exc)))
            continue
        results.append(PolicyResult(name, a, b, same))
    return InvarianceReport(tuple(results), original.digest(), analogue.digest())


# ---------------------------------------------------------------------------
# Sensitivity probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldProbe:
    field: str
    role: Role
    probed_values: int
    flips: Tuple[Tuple[Any, Any], ...]  # (probe value, resulting outcome)

    @property
    def flipped(self) -> bool:
        return bool(self.flips)


@dataclass(frozen=True)
class SensitivityReport:
    policy: str
    baseline: Any
    probes: Tuple[FieldProbe, ...]

    @property
    def flipping_fields(self) -> Tuple[str, ...]:
        return tuple(p.field for p in self.probes if p.flipped)

    @property
    def misclassified(self) -> Tuple[str, ...]:
        """Fields that flip the outcome but are not tagged decision_relevant."""
        return tuple(p.field for p in self.probes if p.flipped and p.role is not Role.DECISION_RELEVANT)

    @property
    def inert_decision_relevant(self) -> Tuple[str, ...]:
        """Decision-relevant fields that were probed and never flipped this policy.

        Informational: they may be read by another policy, or over-tagged.
        """
        return tuple(
            p.field for p in self.probes if p.role is Role.DECISION_RELEVANT and p.probed_values and not p.flipped
        )

    @property
    def unprobed(self) -> Tuple[str, ...]:
        """Fields for which no probe value was available (nothing can be said)."""
        return tuple(p.field for p in self.probes if p.probed_values == 0)


def sensitivity_probe(
    policies: PolicySet,
    case: Case,
    candidates: Optional[Mapping[str, Iterable[Any]]] = None,
    other: Optional[Case] = None,
    equivalence: EquivalenceSpec = None,
) -> Dict[str, SensitivityReport]:
    """Report which single-field perturbations flip each policy's outcome.

    Probe values for a field are `candidates[field]` plus, if `other` is
    given (typically the analogue), the value `other` holds for that field.
    One field is changed at a time: interactions between fields are not
    explored, and a field with no probe value is reported as unprobed rather
    than as inert. A perturbation on which the policy raises counts as a flip,
    with outcome "<error: ExceptionType>".

    Use it to check the role tags: a field tagged `identifying` or
    `contextual` that flips a policy is mis-tagged (or the swap is lossy).
    """
    candidates = candidates or {}
    reports: Dict[str, SensitivityReport] = {}
    for name, policy in _normalise_policies(policies):
        eq = _equivalence_for(name, equivalence)
        baseline = policy(case)
        probes: List[FieldProbe] = []
        for f in case.fields:
            values: List[Any] = []
            pool = list(candidates.get(f.name, ()))
            if other is not None and f.name in other:
                pool.append(other[f.name])
            for v in pool:
                if v == f.value or any(v == u for u in values):
                    continue
                values.append(v)
            flips: List[Tuple[Any, Any]] = []
            for v in values:
                try:
                    outcome = policy(case.with_values({f.name: v}))
                except Exception as exc:  # noqa: BLE001
                    flips.append((v, "<error: %s>" % type(exc).__name__))
                    continue
                if not eq(baseline, outcome):
                    flips.append((v, outcome))
            probes.append(FieldProbe(f.name, f.role, len(values), tuple(flips)))
        reports[name] = SensitivityReport(name, baseline, tuple(probes))
    return reports


__all__ = [
    "Policy",
    "PolicyResult",
    "InvarianceReport",
    "check_invariance",
    "FieldProbe",
    "SensitivityReport",
    "sensitivity_probe",
]
