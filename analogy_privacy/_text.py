"""Small text helpers shared by the audit and round-trip modules (private).

Everything here is deliberately simple and deterministic: accent-folding,
tokenisation, and single-pass whole-word substitution.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Mapping, Tuple

# Function words ignored when deciding whether an overlap is "content".
# Deliberately short; users can pass their own set to the audit.
DEFAULT_STOPWORDS = frozenset(
    """
    a an and or but nor of in on at to for with without by from as into onto
    is am are was were be been being has have had having do does did done
    that this these those it its he she they his her hers their theirs him
    them who whom which what when where why how not no if then than so
    after before during about over under up down off again also only very
    can could would should will shall may might must there here per any all
    some each other such own same both more most
    """.split()
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def fold(text: str) -> str:
    """Case-fold and strip accents, so 'José' matches 'jose'."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


def tokenize(text: str) -> List[str]:
    """Folded alphanumeric tokens, in order. Apostrophes and hyphens split."""
    return _TOKEN_RE.findall(fold(text))


def contains_subsequence(haystack: List[str], needle: List[str]) -> bool:
    """True if `needle` occurs as a contiguous run inside `haystack`."""
    n = len(needle)
    if n == 0 or n > len(haystack):
        return False
    first = needle[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i : i + n] == needle:
            return True
    return False


def substitute(text: str, table: Mapping[str, str]) -> Tuple[str, Dict[str, int]]:
    """Single-pass, whole-word, case-insensitive replacement.

    Keys are matched longest-first in one regex alternation, so replacements
    are never chained (A -> B then B -> C does not happen). Returns the new
    text and a count of replacements per (lower-cased) key.

    Raises ValueError if two keys differ only by case (ambiguous).
    """
    if not table:
        return text, {}
    lowered: Dict[str, str] = {}
    for key, value in table.items():
        if not key:
            continue
        lk = key.lower()
        if lk in lowered and lowered[lk] != value:
            raise ValueError("ambiguous substitution: keys differing only by case: %r" % key)
        lowered[lk] = value
    if not lowered:
        return text, {}
    keys = sorted(lowered, key=len, reverse=True)
    pattern = re.compile(
        r"(?<!\w)(" + "|".join(re.escape(k) for k in keys) + r")(?!\w)",
        re.IGNORECASE | re.UNICODE,
    )
    counts: Dict[str, int] = {}

    def _sub(match: "re.Match[str]") -> str:
        lk = match.group(1).lower()
        counts[lk] = counts.get(lk, 0) + 1
        return lowered[lk]

    return pattern.sub(_sub, text), counts

