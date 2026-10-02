"""Transformers: map a real case to a candidate fictional analogue.

A transformer only proposes an analogue. Nothing here establishes that the
analogue is adequate: that is the job of `invariance`, `leak_audit` and a
human sign-off.

* `Transformer`            abstract interface.
* `RuleBasedTransformer`   swap tables; deterministic; used in tests and examples.
* `OllamaTransformer`      asks a LOCAL model served by Ollama. UNTESTED, see its docstring.

No network access happens at import time.
"""

from __future__ import annotations

import abc
import ipaddress
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ._text import substitute
from .schema import Case, Role, string_pairs


class TransformError(Exception):
    """The transformer could not produce a usable analogue."""


class MappingConflict(TransformError):
    """Two different original strings were mapped to the same analogue string."""


@dataclass(frozen=True)
class TransformResult:
    analogue: Case
    mapping: Dict[str, str]  # analogue string -> original string. LOCAL ONLY.
    warnings: Tuple[str, ...] = ()


class Transformer(abc.ABC):
    """Anything with `transform(case) -> TransformResult`."""

    @abc.abstractmethod
    def transform(self, case: Case) -> TransformResult:
        raise NotImplementedError


def _add_mapping(mapping: Dict[str, str], analogue_str: str, original_str: str) -> None:
    """Record analogue -> original; refuse if one analogue string would stand for two originals.

    Originals that differ only by letter case count as the same original (the
    first spelling is kept), since back-mapping is case-insensitive anyway.
    """
    for existing, orig in mapping.items():
        if existing.lower() == analogue_str.lower():
            if orig.lower() != original_str.lower():
                raise MappingConflict(
                    "analogue string %r would stand for two different originals; "
                    "back-mapping would be ambiguous" % analogue_str
                )
            return
    mapping[analogue_str] = original_str


# ---------------------------------------------------------------------------
# Rule-based
# ---------------------------------------------------------------------------


def _key(v: Any) -> Any:
    return v.casefold() if isinstance(v, str) else v


class RuleBasedTransformer(Transformer):
    """Swap-table transformer.

    `swaps[field][original_value] = analogue_value`. String keys match
    case-insensitively. Behaviour for fields with no matching entry:

    * identifying: raise `TransformError` (identifying content must not pass
      through unchanged); set `require_identifying_swaps=False` to allow it.
    * contextual / decision-relevant: kept as they are.

    The narrative is rewritten with every string-to-string entry of every
    table plus `narrative_swaps` (whole words, case-insensitive, one pass).

    The returned mapping covers string-valued swaps only (see
    `schema.string_pairs`), plus any narrative swap that actually fired.
    """

    def __init__(
        self,
        swaps: Mapping[str, Mapping[Any, Any]],
        *,
        narrative_swaps: Optional[Mapping[str, str]] = None,
        require_identifying_swaps: bool = True,
    ) -> None:
        self._swaps: Dict[str, Dict[Any, Tuple[Any, Any]]] = {
            name: {_key(k): (k, v) for k, v in table.items()} for name, table in swaps.items()
        }
        self._narrative_swaps = dict(narrative_swaps or {})
        self._require = require_identifying_swaps

    def transform(self, case: Case) -> TransformResult:
        changes: Dict[str, Any] = {}
        for f in case.fields:
            table = self._swaps.get(f.name, {})
            try:
                hit = table.get(_key(f.value))
            except TypeError:  # unhashable value: no swap possible
                hit = None
            if hit is not None:
                changes[f.name] = hit[1]
            elif f.role is Role.IDENTIFYING and self._require:
                raise TransformError("no swap defined for identifying field %r (value not in table)" % f.name)

        analogue = case.with_values(changes)
        mapping: Dict[str, str] = {}
        for a, o in string_pairs(case, analogue).items():
            _add_mapping(mapping, a, o)

        text_table: Dict[str, str] = {}
        for table in self._swaps.values():
            for original, new in table.values():
                if isinstance(original, str) and isinstance(new, str):
                    text_table[original] = new
        text_table.update(self._narrative_swaps)

        if case.narrative:
            new_narrative, counts = substitute(case.narrative, text_table)
            analogue = analogue.with_narrative(new_narrative)
            by_lower = {k.lower(): (k, v) for k, v in text_table.items()}
            for lk in counts:
                original, new = by_lower[lk]
                _add_mapping(mapping, new, original)
        return TransformResult(analogue, mapping)


# ---------------------------------------------------------------------------
# Local model via Ollama
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a local privacy transformer. You receive one real case as JSON and must \
produce a FICTIONAL ANALOGUE of it. The analogue will be checked automatically \
and then shown to a human before anything leaves this machine.

Each field has a role:
- "decision_relevant": a downstream decision policy reads this value. Copy it \
exactly, unless the policy notes say an equivalent substitute is acceptable.
- "identifying": replace with a plausible, clearly fictional value of the same \
kind (a name for a name, a place for a place, an identifier for an identifier). \
Share no word, initial or number with the original value.
- "contextual": replace or generalise it so that the analogue does not single \
out the original person. Avoid rare combinations of attributes.

The narrative: rewrite it so it is consistent with the changed fields and \
contains none of the original names, places, dates or rare details.

Do not add facts that would change what the policies decide. Do not explain. \
Reply with exactly one JSON object and nothing else:
{"fields": {"<field name>": <new value>, ...},
 "narrative": "<rewritten narrative, or an empty string>",
 "replacements": [["<fragment of the original text>", "<fictional fragment replacing it>"], ...]}
"fields" must contain every field name from the input, exactly once. \
"replacements" lists the substitutions you made inside the narrative.
"""


def _is_loopback(host_url: str) -> bool:
    name = urllib.parse.urlparse(host_url).hostname or ""
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class OllamaTransformer(Transformer):
    """Ask a local model, served by Ollama, for a candidate analogue.

    UNTESTED against a real model or server. Prompt construction and response
    parsing are unit-tested with canned JSON; the HTTP call and the behaviour
    of any actual model are not. The request shape follows Ollama's documented
    `POST /api/chat` (non-streaming, `format: "json"`).

    The model is the weakest link: it may leave identifying content in place,
    change decision-relevant values, or invent facts. Its output is only a
    proposal and must pass the invariance check, the leak audit and a human
    sign-off. Local validation here covers structure only (same field names,
    compatible value types).

    Safety defaults: refuses a non-loopback `host` unless `allow_remote=True`
    (sending the original case to a remote host would defeat the purpose), and
    bypasses any HTTP proxy configured in the environment.

    No network call is made at import or construction, only in `transform`.
    `post` may be injected (a callable `(url, payload) -> dict`) for testing.
    """

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        *,
        policy_notes: str = "",
        timeout: float = 120.0,
        allow_remote: bool = False,
        post: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        if not allow_remote and not _is_loopback(host):
            raise ValueError("refusing non-loopback host %r (pass allow_remote=True to override)" % host)
        self.model = model
        self.url = host.rstrip("/") + "/api/chat"
        self.policy_notes = policy_notes
        self.timeout = timeout
        self._post = post or self._http_post

    # -- prompt -----------------------------------------------------------
    def build_messages(self, case: Case) -> List[Dict[str, str]]:
        payload = {
            "fields": {f.name: {"role": f.role.value, "value": f.value} for f in case.fields},
            "narrative": case.narrative,
        }
        user = "CASE:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        if self.policy_notes:
            user += "\n\nPOLICY NOTES (what the downstream decision depends on):\n" + self.policy_notes
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]

    # -- transport --------------------------------------------------------
    def _http_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        # An empty ProxyHandler ignores *_proxy environment variables, so the
        # case cannot be routed through a proxy by accident.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise TransformError("local model request failed: %s" % exc) from exc

    # -- main -------------------------------------------------------------
    def transform(self, case: Case) -> TransformResult:
        payload = {
            "model": self.model,
            "messages": self.build_messages(case),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        raw = self._post(self.url, payload)
        try:
            content = raw["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise TransformError("unexpected response shape from the local model") from exc
        return parse_model_output(case, content)


def parse_model_output(case: Case, content: str) -> TransformResult:
    """Validate a model's JSON reply against `case` and build a TransformResult."""
    try:
        obj = json.loads(content)
    except (TypeError, ValueError) as exc:
        raise TransformError("model output is not valid JSON") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("fields"), dict):
        raise TransformError("model output lacks a 'fields' object")

    new_fields = obj["fields"]
    if set(new_fields) != set(case.names()):
        raise TransformError(
            "field names differ from the input (missing: %s; extra: %s)"
            % (sorted(set(case.names()) - set(new_fields)), sorted(set(new_fields) - set(case.names())))
        )

    warnings: List[str] = []
    values: Dict[str, Any] = {}
    for f in case.fields:
        v = new_fields[f.name]
        if not isinstance(f.value, str) and f.value is not None:
            if type(v) is not type(f.value):
                try:
                    if isinstance(f.value, bool):
                        raise TypeError
                    v = type(f.value)(v)
                except (TypeError, ValueError):
                    raise TransformError(
                        "field %r: expected %s, got %r" % (f.name, type(f.value).__name__, v)
                    ) from None
        elif isinstance(f.value, str) and not isinstance(v, str):
            raise TransformError("field %r: expected a string, got %r" % (f.name, v))
        values[f.name] = v
        if f.role is Role.DECISION_RELEVANT and v != f.value:
            warnings.append("decision-relevant field %r was changed by the model" % f.name)
        if f.role is Role.IDENTIFYING and v == f.value:
            warnings.append("identifying field %r was left unchanged by the model" % f.name)

    analogue = case.with_values(values)
    narrative = obj.get("narrative", "")
    if not isinstance(narrative, str):
        raise TransformError("'narrative' must be a string")
    analogue = analogue.with_narrative(narrative)

    mapping: Dict[str, str] = {}
    for a, o in string_pairs(case, analogue).items():
        _add_mapping(mapping, a, o)

    # Narrative replacements are the model's claim; keep one only if both
    # fragments are really present in the respective texts.
    original_text = "\n".join([case.narrative] + [str(f.value) for f in case.fields]).lower()
    analogue_text = "\n".join([analogue.narrative] + [str(f.value) for f in analogue.fields]).lower()
    claimed = obj.get("replacements", [])
    if not isinstance(claimed, list):
        warnings.append("ignored a malformed 'replacements' entry")
        claimed = []
    for pair in claimed:
        if (
            isinstance(pair, (list, tuple))
            and len(pair) == 2
            and all(isinstance(x, str) and x for x in pair)
            and pair[0].lower() in original_text
            and pair[1].lower() in analogue_text
        ):
            _add_mapping(mapping, pair[1], pair[0])
        else:
            warnings.append("ignored a replacement that could not be verified locally: %r" % (pair,))
    return TransformResult(analogue, mapping, tuple(warnings))


__all__ = [
    "TransformError",
    "MappingConflict",
    "TransformResult",
    "Transformer",
    "RuleBasedTransformer",
    "OllamaTransformer",
    "parse_model_output",
    "SYSTEM_PROMPT",
]
