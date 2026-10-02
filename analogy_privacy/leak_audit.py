"""Leak audit: direct-overlap check plus a quasi-identifier uniqueness estimate.

Two heuristics, both local, neither a proof of anonymity.

(a) Direct leaks. Tokens, shared phrases and whole field values of the
    original that reappear in the analogue (accent- and case-insensitive).
    Overlaps that are deliberately shared (decision-relevant values, boilerplate)
    are marked `allowed` rather than hidden, so they stay visible in the report.

(b) Uniqueness. Given a reference population (records the analogue could be
    matched against, for example the organisation's own log), count the records
    that match the analogue's quasi-identifier combination. A small count means
    the combination singles out few people. This is a k-anonymity-style
    estimate over the supplied population only.

Reports contain fragments of the original case. Treat them as local data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ._text import DEFAULT_STOPWORDS, contains_subsequence, fold, tokenize
from .schema import NARRATIVE_SOURCE, Case, Role


class Severity(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3


# ---------------------------------------------------------------------------
# (a) Direct leaks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakFinding:
    kind: str  # "field_value" | "identifying_token" | "phrase" | "token"
    text: str  # the overlapping string (folded); contains original content
    severity: Severity
    source: str  # original field name, or NARRATIVE_SOURCE
    allowed: bool
    reason: str


@dataclass(frozen=True)
class DirectLeakReport:
    findings: Tuple[LeakFinding, ...]
    fail_at: Severity

    @property
    def blocking(self) -> Tuple[LeakFinding, ...]:
        return tuple(f for f in self.findings if not f.allowed and f.severity >= self.fail_at)

    @property
    def passed(self) -> bool:
        return not self.blocking


def _usable(token: str, stop: Set[str], min_len: int) -> bool:
    return len(token) >= min_len and token not in stop


def check_direct_leaks(
    original: Case,
    analogue: Case,
    *,
    allow: Iterable[str] = (),
    allow_decision_relevant: bool = True,
    min_phrase_tokens: int = 2,
    stopwords: Optional[Iterable[str]] = None,
    common_words: Iterable[str] = (),
    min_token_length: int = 2,
    fail_at: Severity = Severity.MEDIUM,
) -> DirectLeakReport:
    """Find original content that reappears in the analogue.

    Severity:
      HIGH   a whole identifying field value, or any token of one, reappears.
      MEDIUM a whole contextual or decision-relevant value, or a maximal shared
             phrase from any other text. A phrase is a run of consecutive tokens
             that also occurs, in the same order, in some analogue field or the
             analogue narrative, with at least `min_phrase_tokens` content tokens.
      LOW    a shared content token from non-identifying text that is not part of
             a reported phrase and not in `common_words`.

    `common_words`: words too common to be informative. They are treated like
    stopwords for non-identifying text (never for identifying fields), so they
    suppress LOW findings and also count as ignorable inside phrases and
    whole contextual values.

    `allow`: strings whose tokens may be shared (boilerplate, deliberately kept
    categories). With `allow_decision_relevant`, tokens of decision-relevant
    field values of the original are allowed too. An overlap is `allowed` iff
    every content token in it is allowed. Allowed findings are still reported.

    `fail_at`: findings at or above this severity, not allowed, make the
    report fail.
    """
    stop = set(DEFAULT_STOPWORDS if stopwords is None else (fold(w) for w in stopwords))
    common = {fold(w) for w in common_words}
    ignorable = stop | common

    allowed_tokens: Set[str] = set()
    for phrase in allow:
        allowed_tokens.update(tokenize(phrase))
    if allow_decision_relevant:
        for f in original.fields_with_role(Role.DECISION_RELEVANT):
            if f.value is not None and not isinstance(f.value, bool):
                allowed_tokens.update(tokenize(str(f.value)))

    # Analogue side: per-source token lists (so phrases do not cross field boundaries).
    a_sources = [tokenize(text) for _, text in analogue.text_sources()]
    a_tokens: Set[str] = {t for toks in a_sources for t in toks}

    def all_allowed(tokens: Sequence[str], ignore: Set[str], by_length: bool = True) -> bool:
        """Every content token is allowed. With by_length=False, short tokens count as content."""
        return all(
            (t in allowed_tokens)
            or (t in ignore)
            or (by_length and len(t) < min_token_length)
            for t in tokens
        )

    findings: List[LeakFinding] = []
    seen: Set[Tuple[str, str, str]] = set()

    def add(kind: str, text: str, sev: Severity, source: str, allowed: bool, reason: str) -> None:
        key = (kind, text, source)
        if key not in seen:
            seen.add(key)
            findings.append(LeakFinding(kind, text, sev, source, allowed, reason))

    for source, text in original.text_sources():
        role = original.role_of(source) if source in original else None
        toks = tokenize(text)
        if not toks:
            continue

        whole_match = source != NARRATIVE_SOURCE and any(contains_subsequence(a, toks) for a in a_sources)
        if whole_match:
            phrase = " ".join(toks)
            if role is Role.IDENTIFYING:
                sev = Severity.HIGH
                ok = all_allowed(toks, stop, by_length=False)
                why = "identifying value reappears in the analogue"
            elif role is Role.DECISION_RELEVANT:
                sev = Severity.MEDIUM
                ok = allow_decision_relevant or all_allowed(toks, ignorable, by_length=False)
                why = "decision-relevant value shared (expected)" if ok else "decision-relevant value shared"
            else:
                sev = Severity.MEDIUM
                ok = all_allowed(toks, ignorable, by_length=False)
                why = "contextual value reappears in the analogue"
            add("field_value", phrase, sev, source, ok, why + (" -- allow-listed" if ok and role is not Role.DECISION_RELEVANT else ""))

        if role is Role.IDENTIFYING:
            if not whole_match:
                for t in dict.fromkeys(toks):
                    if _usable(t, stop, min_token_length) and t in a_tokens:
                        add(
                            "identifying_token",
                            t,
                            Severity.HIGH,
                            source,
                            t in allowed_tokens,
                            "token of an identifying value reappears" + (" -- allow-listed" if t in allowed_tokens else ""),
                        )
            continue

        if whole_match:
            continue  # already reported as one finding

        # Non-identifying text: maximal shared phrases, then leftover single tokens.
        covered_tokens: Set[str] = set()
        i = 0
        while i < len(toks):
            j = i
            while j < len(toks) and any(contains_subsequence(a, toks[i : j + 1]) for a in a_sources):
                j += 1
            run = toks[i:j]
            n_content = sum(1 for t in run if _usable(t, ignorable, min_token_length))
            if len(run) >= 2 and n_content >= max(1, min_phrase_tokens):
                covered_tokens.update(run)
                ok = all_allowed(run, ignorable)
                add(
                    "phrase",
                    " ".join(run),
                    Severity.MEDIUM,
                    source,
                    ok,
                    "shared phrase" + (" -- allow-listed" if ok else ""),
                )
                i = j
            else:
                i += 1
        for t in dict.fromkeys(toks):
            if t in covered_tokens or not _usable(t, ignorable, min_token_length) or t not in a_tokens:
                continue
            ok = t in allowed_tokens
            add("token", t, Severity.LOW, source, ok, "shared content word" + (" -- allow-listed" if ok else ""))

    return DirectLeakReport(tuple(findings), fail_at)


# ---------------------------------------------------------------------------
# (b) Quasi-identifier uniqueness
# ---------------------------------------------------------------------------

Matcher = Callable[[Any, Any], bool]  # (population value, case value) -> bool
Record = Mapping[str, Any]


def _norm(v: Any) -> Any:
    return fold(v).strip() if isinstance(v, str) else v


def _default_match(pop_value: Any, case_value: Any) -> bool:
    return _norm(pop_value) == _norm(case_value)


@dataclass(frozen=True)
class UniquenessFinding:
    scope: str  # "analogue" | "shared_with_original" | "original"
    quasi_identifiers: Tuple[str, ...]
    k: int
    threshold: int
    flagged: bool
    note: str


@dataclass(frozen=True)
class UniquenessReport:
    population_size: int
    findings: Tuple[UniquenessFinding, ...]

    @property
    def flagged(self) -> Tuple[UniquenessFinding, ...]:
        return tuple(f for f in self.findings if f.flagged)

    @property
    def passed(self) -> bool:
        return not self.flagged

    def get(self, scope: str) -> Optional[UniquenessFinding]:
        for f in self.findings:
            if f.scope == scope:
                return f
        return None


def _count(
    population: Sequence[Record],
    case: Case,
    qis: Sequence[str],
    matchers: Mapping[str, Matcher],
) -> int:
    n = 0
    for rec in population:
        ok = True
        for q in qis:
            if q not in rec:
                ok = False  # missing in the record: cannot be matched
                break
            if not matchers.get(q, _default_match)(rec[q], case[q]):
                ok = False
                break
        if ok:
            n += 1
    return n


def estimate_uniqueness(
    analogue: Case,
    population: Sequence[Record],
    *,
    quasi_identifiers: Optional[Sequence[str]] = None,
    original: Optional[Case] = None,
    k_threshold: int = 5,
    matchers: Optional[Mapping[str, Matcher]] = None,
) -> UniquenessReport:
    """Estimate how many population records match the analogue's quasi-identifiers.

    `quasi_identifiers` defaults to every field of the analogue not tagged
    `identifying`. Every quasi-identifier must be a key in at least one
    population record, otherwise ValueError: silently dropping one would
    inflate k and make the result look safer than it is.

    Matching is exact after case/accent folding for strings and `==` otherwise;
    pass `matchers[field] = f(population_value, case_value)` to generalise
    (bands, hierarchies). A record lacking a quasi-identifier does not match.

    Findings:
      analogue              k over all quasi-identifiers, using the analogue's values.
                            Flagged if 0 < k < k_threshold. k == 0 means the combination
                            is absent from the population: noted, not flagged, and no
                            evidence either way about the original.
      shared_with_original  (needs `original`) k over only those quasi-identifiers on
                            which analogue and original agree. These are the attributes
                            through which the original could still be singled out.
                            Flagged if k < k_threshold; k == 0 means the original is not
                            represented in the population, so the estimate is unreliable.
      original              (needs `original`) k for the original's own combination.
                            Informational, never flagged: it shows how exposed the
                            untransformed case was.
    """
    population = list(population)
    if not population:
        raise ValueError("reference population is empty")
    matchers = matchers or {}
    if quasi_identifiers is None:
        qis = [f.name for f in analogue.fields if f.role is not Role.IDENTIFYING]
    else:
        qis = list(quasi_identifiers)
    if not qis:
        raise ValueError("no quasi-identifiers to assess")
    for q in qis:
        if q not in analogue:
            raise ValueError("quasi-identifier %r is not a field of the analogue" % q)
        if original is not None and q not in original:
            raise ValueError("quasi-identifier %r is not a field of the original" % q)
        if not any(q in rec for rec in population):
            raise ValueError(
                "quasi-identifier %r does not occur in the reference population; "
                "pass quasi_identifiers explicitly or supply records that include it" % q
            )

    findings: List[UniquenessFinding] = []

    k_a = _count(population, analogue, qis, matchers)
    if k_a == 0:
        note = "combination absent from the population (no information about the original)"
    elif k_a < k_threshold:
        note = "analogue combination matches few records"
    else:
        note = "ok"
    findings.append(UniquenessFinding("analogue", tuple(qis), k_a, k_threshold, 0 < k_a < k_threshold, note))

    if original is not None:
        shared = [q for q in qis if matchers.get(q, _default_match)(original[q], analogue[q])]
        k_s = _count(population, analogue, shared, matchers)
        if k_s == 0:
            note = "original not represented in the population; estimate unreliable"
        elif k_s < k_threshold:
            note = "attributes shared with the original single out few records"
        else:
            note = "ok"
        findings.append(
            UniquenessFinding("shared_with_original", tuple(shared), k_s, k_threshold, k_s < k_threshold, note)
        )
        k_o = _count(population, original, qis, matchers)
        findings.append(
            UniquenessFinding("original", tuple(qis), k_o, k_threshold, False, "informational: exposure of the untransformed case")
        )

    return UniquenessReport(len(population), tuple(findings))


# ---------------------------------------------------------------------------
# Combined audit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakAuditReport:
    direct: DirectLeakReport
    uniqueness: Optional[UniquenessReport]
    original_fp: str
    analogue_fp: str

    @property
    def uniqueness_assessed(self) -> bool:
        return self.uniqueness is not None

    @property
    def passed(self) -> bool:
        """True iff no blocking direct finding and no flagged uniqueness finding.

        Note that an unassessed uniqueness check does not fail this property;
        see `uniqueness_assessed` (the round-trip gate requires it by default).
        """
        return self.direct.passed and (self.uniqueness is None or self.uniqueness.passed)

    def summary(self) -> str:
        lines: List[str] = []
        for f in self.direct.findings:
            tag = "allowed" if f.allowed else ("BLOCKING" if f.severity >= self.direct.fail_at else "note")
            lines.append("  direct  %-8s %-17s %-22r from %-14s %s" % (tag, f.kind, f.text, f.source, f.reason))
        if not self.direct.findings:
            lines.append("  direct  no overlaps found")
        if self.uniqueness is None:
            lines.append("  k-estimate: NOT ASSESSED (no reference population supplied)")
        else:
            for u in self.uniqueness.findings:
                tag = "FLAGGED" if u.flagged else "ok"
                lines.append(
                    "  k       %-8s %-21s k=%-3d (threshold %d) over %s -- %s"
                    % (tag, u.scope, u.k, u.threshold, list(u.quasi_identifiers), u.note)
                )
        return "\n".join(lines)


def audit_leaks(
    original: Case,
    analogue: Case,
    *,
    population: Optional[Sequence[Record]] = None,
    quasi_identifiers: Optional[Sequence[str]] = None,
    k_threshold: int = 5,
    matchers: Optional[Mapping[str, Matcher]] = None,
    allow: Iterable[str] = (),
    allow_decision_relevant: bool = True,
    fail_at: Severity = Severity.MEDIUM,
    min_phrase_tokens: int = 2,
    stopwords: Optional[Iterable[str]] = None,
    common_words: Iterable[str] = (),
    min_token_length: int = 2,
) -> LeakAuditReport:
    """Run the direct-leak check and, if `population` is given, the uniqueness estimate."""
    direct = check_direct_leaks(
        original,
        analogue,
        allow=allow,
        allow_decision_relevant=allow_decision_relevant,
        min_phrase_tokens=min_phrase_tokens,
        stopwords=stopwords,
        common_words=common_words,
        min_token_length=min_token_length,
        fail_at=fail_at,
    )
    uniq = None
    if population is not None:
        uniq = estimate_uniqueness(
            analogue,
            population,
            quasi_identifiers=quasi_identifiers,
            original=original,
            k_threshold=k_threshold,
            matchers=matchers,
        )
    return LeakAuditReport(direct, uniq, original.digest(), analogue.digest())


__all__ = [
    "Severity",
    "LeakFinding",
    "DirectLeakReport",
    "check_direct_leaks",
    "UniquenessFinding",
    "UniquenessReport",
    "estimate_uniqueness",
    "LeakAuditReport",
    "audit_leaks",
]
