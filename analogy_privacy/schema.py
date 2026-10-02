"""Case schema: named fields, each tagged with a role.

Roles
-----
decision_relevant
    Read by at least one decision policy of interest. Preserved (or replaced
    by an outcome-equivalent value) in an analogue.
identifying
    Content that must not survive into the analogue: names, identifiers, and
    sensitive attributes that no policy of interest reads.
contextual
    Not identifying on its own, but can single someone out in combination
    (department, month of absence, tenure band). Audited as quasi-identifiers.

A field's role is a claim made by the user. The sensitivity probe in
`invariance.py` can expose some wrong claims (a field tagged `identifying`
that in fact flips a policy outcome); it cannot expose all of them.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


NARRATIVE_SOURCE = "(narrative)"  # source label used for the free-text narrative in audits


class Role(str, Enum):
    DECISION_RELEVANT = "decision_relevant"
    IDENTIFYING = "identifying"
    CONTEXTUAL = "contextual"

    @classmethod
    def parse(cls, value: Union["Role", str]) -> "Role":
        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            allowed = [r.value for r in cls]
            raise ValueError("unknown role %r; expected one of %s" % (value, allowed)) from None


@dataclass(frozen=True)
class CaseField:
    name: str
    value: Any
    role: Role

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field name must be a non-empty string")
        object.__setattr__(self, "role", Role.parse(self.role))


@dataclass(frozen=True)
class Case:
    """A case: tagged structured fields plus an optional free-text narrative.

    Policies (see `invariance.py`) are functions of the structured fields.
    The narrative is audited for leaks but is never seen by a policy, so a
    narrative that changes the meaning of the case is not detected by the
    invariance check.
    """

    fields: Tuple[CaseField, ...]
    narrative: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        names = [f.name for f in self.fields]
        if len(set(names)) != len(names):
            raise ValueError("duplicate field names: %s" % sorted(n for n in set(names) if names.count(n) > 1))

    # -- construction -----------------------------------------------------
    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        roles: Mapping[str, Union[Role, str]],
        narrative: str = "",
    ) -> "Case":
        """Build from a values mapping and a roles mapping with identical keys."""
        missing = set(values) - set(roles)
        extra = set(roles) - set(values)
        if missing or extra:
            raise ValueError(
                "values and roles must have the same keys (untagged: %s; no value: %s)"
                % (sorted(missing), sorted(extra))
            )
        return cls(tuple(CaseField(n, values[n], Role.parse(roles[n])) for n in values), narrative)

    # -- access -----------------------------------------------------------
    def names(self) -> Tuple[str, ...]:
        return tuple(f.name for f in self.fields)

    def __contains__(self, name: object) -> bool:
        return any(f.name == name for f in self.fields)

    def __getitem__(self, name: str) -> Any:
        for f in self.fields:
            if f.name == name:
                return f.value
        raise KeyError(name)

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default

    def role_of(self, name: str) -> Role:
        for f in self.fields:
            if f.name == name:
                return f.role
        raise KeyError(name)

    def fields_with_role(self, *roles: Union[Role, str]) -> Tuple[CaseField, ...]:
        wanted = {Role.parse(r) for r in roles}
        return tuple(f for f in self.fields if f.role in wanted)

    def project(self, *roles: Union[Role, str]) -> Dict[str, Any]:
        """Mapping name -> value restricted to the given roles."""
        return {f.name: f.value for f in self.fields_with_role(*roles)}

    def decision_view(self) -> Dict[str, Any]:
        """Projection onto decision-relevant fields."""
        return self.project(Role.DECISION_RELEVANT)

    # -- modification (returns new cases) --------------------------------
    def with_values(self, changes: Mapping[str, Any]) -> "Case":
        """Copy with some field values changed; roles are kept."""
        unknown = [n for n in changes if n not in self]
        if unknown:
            raise KeyError("unknown fields: %s" % sorted(unknown))
        return Case(
            tuple(replace(f, value=changes[f.name]) if f.name in changes else f for f in self.fields),
            self.narrative,
        )

    def with_narrative(self, narrative: str) -> "Case":
        return Case(self.fields, narrative)

    def with_roles(self, changes: Mapping[str, Union[Role, str]]) -> "Case":
        """Copy with some roles re-tagged; values are kept."""
        unknown = [n for n in changes if n not in self]
        if unknown:
            raise KeyError("unknown fields: %s" % sorted(unknown))
        return Case(
            tuple(replace(f, role=Role.parse(changes[f.name])) if f.name in changes else f for f in self.fields),
            self.narrative,
        )

    # -- serialisation ----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "fields": [{"name": f.name, "role": f.role.value, "value": f.value} for f in self.fields],
            "narrative": self.narrative,
        }

    def canonical_text(self) -> str:
        """Order-independent serialisation used for hashing."""
        payload = {
            "fields": {f.name: [f.role.value, f.value] for f in self.fields},
            "narrative": self.narrative,
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)

    def digest(self, key: Optional[bytes] = None) -> str:
        """SHA-256 of the canonical text, or HMAC-SHA-256 if `key` is given.

        An unkeyed hash of a low-entropy case can be reversed by guessing
        candidate cases. Use a locally held key if the digest may be shared.
        """
        data = self.canonical_text().encode("utf-8")
        if key is None:
            return hashlib.sha256(data).hexdigest()
        return hmac.new(key, data, hashlib.sha256).hexdigest()

    def render(self) -> str:
        """Plain-text rendering: narrative, then one 'name: value' line per field."""
        lines: List[str] = []
        if self.narrative:
            lines.append(self.narrative)
            lines.append("")
        lines.extend("%s: %s" % (f.name, f.value) for f in self.fields)
        return "\n".join(lines)

    def text_sources(self) -> List[Tuple[str, str]]:
        """(source, text) pairs for auditing: the narrative and each field.

        Booleans and None carry no lexical content and are skipped.
        """
        out: List[Tuple[str, str]] = []
        if self.narrative:
            out.append((NARRATIVE_SOURCE, self.narrative))
        for f in self.fields:
            if f.value is None or isinstance(f.value, bool):
                continue
            out.append((f.name, str(f.value)))
        return out


def project_decision_relevant(case: Case) -> Dict[str, Any]:
    """Functional alias for `case.decision_view()`."""
    return case.decision_view()


def string_pairs(original: Case, analogue: Case) -> Dict[str, str]:
    """Mapping analogue string -> original string for string-valued fields that differ.

    Used to build a back-mapping table. Non-string values (numbers, booleans)
    are deliberately excluded: blind textual replacement of numbers in a
    free-text answer is more likely to corrupt it than to help.
    """
    out: Dict[str, str] = {}
    for f in original.fields:
        if f.name not in analogue:
            continue
        a = analogue[f.name]
        if isinstance(f.value, str) and isinstance(a, str) and a != f.value and a and f.value:
            out[a] = f.value
    return out


__all__ = [
    "NARRATIVE_SOURCE",
    "Role",
    "CaseField",
    "Case",
    "project_decision_relevant",
    "string_pairs",
]
