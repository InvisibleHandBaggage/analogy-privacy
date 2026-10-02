"""Audit log and clearance gate for the round trip.

Flow: transform locally -> verify locally -> human signs off on adequacy ->
`mark_cleared` -> `release_to_cloud` sends the analogue and back-maps the answer.

What the gate does: `mark_cleared` raises `ClearanceRefused` unless invariance
passed, the leak audit passed (including the uniqueness estimate, unless the
caller opts out explicitly), a human sign-off is recorded for exactly this
analogue, and the reports were computed on the cases that are logged.
`release_to_cloud` refuses to send anything from a record that is not cleared.

What it cannot do: it cannot tell that the sign-off came from a human, and it
cannot stop code that bypasses it from sending text to a network.

Storage: a record holds a digest of the original, never the original text. The
mapping table (analogue string -> original string) necessarily contains
original fragments; it is excluded from `to_shareable_dict`. Keep local
records on the trusted machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ._text import DEFAULT_STOPWORDS, substitute, tokenize
from .invariance import InvarianceReport
from .leak_audit import LeakAuditReport
from .schema import Case

PROVIDER_NOTICE = (
    "Notice: every name, date, place and circumstance in the text below is fictional "
    "and does not describe any real person. Please treat it as a hypothetical case."
)


class ClearanceRefused(Exception):
    """Raised when a transform may not be cleared or an uncleared record is released."""

    def __init__(self, reasons: List[str]) -> None:
        self.reasons = tuple(reasons)
        super().__init__("; ".join(reasons))


@dataclass(frozen=True)
class SignOff:
    """A human's adequacy decision, bound to one analogue by its digest.

    The library records the flag; it cannot verify who set it.
    """

    reviewer: str
    approved: bool
    analogue_digest: str
    note: str = ""
    timestamp: str = ""


@dataclass(frozen=True)
class AuditRecord:
    original_digest: str  # HMAC-SHA-256 if a key was used, else SHA-256
    digest_keyed: bool
    analogue: Case
    mapping: Dict[str, str]  # analogue string -> original string. LOCAL ONLY.
    status: str = "pending"  # "pending" | "cleared"
    signoff: Optional[SignOff] = None
    uniqueness_assessed: Optional[bool] = None  # recorded at clearance
    cleared_analogue_digest: Optional[str] = None  # digest of the analogue the checks ran on
    sent_text: Optional[str] = None  # exactly what was passed to `send`
    cloud_answer: Optional[str] = None
    back_mapped_answer: Optional[str] = None
    replacements: Dict[str, int] = field(default_factory=dict)
    residue: Tuple[str, ...] = ()  # analogue-side words still present after back-mapping
    # In-memory binding to the original (unkeyed fingerprint). The provided
    # to_*_dict methods omit it; dataclasses.asdict() would not, so avoid that.
    # Read once, at clearance.
    original_fp: str = field(default="", repr=False, compare=False)

    @property
    def analogue_digest(self) -> str:
        return self.analogue.digest()

    @property
    def signoff_status(self) -> str:
        if self.signoff is None:
            return "missing"
        if self.signoff.analogue_digest != self.analogue_digest:
            return "stale"
        return "approved" if self.signoff.approved else "rejected"

    def to_local_dict(self) -> Dict[str, Any]:
        """Full record including the mapping and back-mapped answer. Keep local."""
        d = self.to_shareable_dict()
        d["mapping"] = dict(self.mapping)
        d["back_mapped_answer"] = self.back_mapped_answer
        d["replacements"] = dict(self.replacements)
        d["residue"] = list(self.residue)
        if self.signoff is not None:
            d["signoff"]["note"] = self.signoff.note
        return d

    def to_shareable_dict(self) -> Dict[str, Any]:
        """Record without the mapping, the back-mapped answer, or the sign-off note.

        Contains the analogue, what was sent, the cloud's answer and a digest of
        the original. It holds no fragment of the original text, provided the
        analogue passed the leak audit.
        """
        so = None
        if self.signoff is not None:
            so = {
                "reviewer": self.signoff.reviewer,
                "approved": self.signoff.approved,
                "analogue_digest": self.signoff.analogue_digest,
                "timestamp": self.signoff.timestamp,
            }
        return {
            "original_digest": self.original_digest,
            "digest_keyed": self.digest_keyed,
            "analogue": self.analogue.to_dict(),
            "status": self.status,
            "signoff_status": self.signoff_status,
            "signoff": so,
            "uniqueness_assessed": self.uniqueness_assessed,
            "sent_text": self.sent_text,
            "cloud_answer": self.cloud_answer,
        }


def new_record(
    original: Case,
    analogue: Case,
    mapping: Mapping[str, str],
    key: Optional[bytes] = None,
) -> AuditRecord:
    """Start a pending record. Stores a digest of `original`, not the original."""
    return AuditRecord(
        original_digest=original.digest(key),
        digest_keyed=key is not None,
        analogue=analogue,
        mapping=dict(mapping),
        original_fp=original.digest(),
    )


def sign_off(
    record: AuditRecord,
    reviewer: str,
    approved: bool,
    note: str = "",
    timestamp: Optional[str] = None,
) -> AuditRecord:
    """Record the human adequacy decision for this record's current analogue."""
    if not reviewer or not reviewer.strip():
        raise ValueError("a named reviewer is required")
    ts = timestamp if timestamp is not None else datetime.now(timezone.utc).isoformat(timespec="seconds")
    return replace(
        record,
        signoff=SignOff(reviewer.strip(), bool(approved), record.analogue_digest, note, ts),
    )


def mark_cleared(
    record: AuditRecord,
    invariance: InvarianceReport,
    leak: LeakAuditReport,
    *,
    require_uniqueness: bool = True,
) -> AuditRecord:
    """Return a cleared copy of `record`, or raise ClearanceRefused with all reasons.

    `require_uniqueness=False` is an explicit opt-out for use without a
    reference population; the opt-out is recorded (`uniqueness_assessed=False`).
    """
    reasons: List[str] = []
    if invariance.original_fp != record.original_fp or invariance.analogue_fp != record.analogue_digest:
        reasons.append("invariance report was not computed on this record's original and analogue")
    elif not invariance.passed:
        reasons.append("policy invariance failed: " + ", ".join(r.policy for r in invariance.failures))

    if leak.original_fp != record.original_fp or leak.analogue_fp != record.analogue_digest:
        reasons.append("leak report was not computed on this record's original and analogue")
    else:
        if not leak.direct.passed:
            reasons.append("direct leaks: %d blocking finding(s)" % len(leak.direct.blocking))
        if leak.uniqueness is not None and not leak.uniqueness.passed:
            reasons.append("uniqueness: %d flagged finding(s)" % len(leak.uniqueness.flagged))
        if leak.uniqueness is None and require_uniqueness:
            reasons.append(
                "uniqueness not assessed (supply a reference population, "
                "or pass require_uniqueness=False to record an explicit opt-out)"
            )

    status = record.signoff_status
    if status != "approved":
        reasons.append("adequacy sign-off is %s" % status)

    if reasons:
        raise ClearanceRefused(reasons)
    return replace(
        record,
        status="cleared",
        uniqueness_assessed=leak.uniqueness_assessed,
        cleared_analogue_digest=record.analogue_digest,
    )


def back_map(answer: str, mapping: Mapping[str, str]) -> Tuple[str, Dict[str, int], Tuple[str, ...]]:
    """Literal back-substitution of analogue strings by original strings.

    Returns (text, replacement counts, residue). Residue lists words from the
    mapping's analogue-side strings that still occur in the output (for
    example a first name used alone when the table holds the full name): such
    words mark places the substitution missed, to be reviewed by hand. The
    method is plain string replacement; pronouns, inflections and paraphrases
    are not handled.
    """
    text, counts = substitute(answer, mapping)
    key_tokens = set()
    value_tokens = set()
    for k, v in mapping.items():
        key_tokens.update(t for t in tokenize(k) if len(t) >= 2 and t not in DEFAULT_STOPWORDS)
        value_tokens.update(tokenize(v))
    out_tokens = set(tokenize(text))
    residue = tuple(sorted((key_tokens - value_tokens) & out_tokens))
    return text, counts, residue


def release_to_cloud(
    record: AuditRecord,
    send: Callable[[str], str],
    *,
    question: str = "",
    notice: str = PROVIDER_NOTICE,
) -> AuditRecord:
    """Send the analogue through `send` and log the back-mapped answer.

    Refuses unless the record is cleared and its sign-off is still approved for
    the current analogue. `send` is whatever calls the cloud model: it receives
    the text and returns the answer. `question` is the user's generic question;
    it is not audited, so keep case content out of it.
    """
    reasons: List[str] = []
    if record.status != "cleared":
        reasons.append("record is not cleared")
    elif record.cleared_analogue_digest != record.analogue_digest:
        reasons.append("analogue changed since clearance; rerun the checks and clear again")
    if record.signoff_status != "approved":
        reasons.append("adequacy sign-off is %s" % record.signoff_status)
    if reasons:
        raise ClearanceRefused(reasons)

    parts = [p for p in (notice, question, record.analogue.render()) if p]
    sent = "\n\n".join(parts)
    answer = send(sent)
    mapped, counts, residue = back_map(answer, record.mapping)
    return replace(
        record,
        sent_text=sent,
        cloud_answer=answer,
        back_mapped_answer=mapped,
        replacements=counts,
        residue=residue,
    )


__all__ = [
    "PROVIDER_NOTICE",
    "ClearanceRefused",
    "SignOff",
    "AuditRecord",
    "new_record",
    "sign_off",
    "mark_cleared",
    "back_map",
    "release_to_cloud",
]
