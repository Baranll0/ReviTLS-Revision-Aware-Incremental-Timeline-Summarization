"""Single source of truth for the lexical revision-cue lexicon (V_rev in the paper)."""

REVISION_CUES: tuple[str, ...] = (
    "revise",
    "revision",
    "correction",
    "correcting",
    "clarif",
    "retract",
    "debunk",
    "supersed",
    "rule out",
    "ruled out",
    "false alarm",
    "false report",
    "false rumor",
    "earlier report",
    "previously estimated",
    "previously reported",
    "previously stated",
    "upward revision",
    "downward revision",
    "revised upward",
    "revised downward",
)


def has_revision_cue(text: str) -> bool:
    """True if the text contains at least one lexical revision cue."""
    low = text.lower()
    return any(cue in low for cue in REVISION_CUES)


import re
from datetime import date

_MEASURE_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s+([a-z][a-z\-]{2,})", re.IGNORECASE)
_MONTHS = {
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}
_STOP_AFTER_NUMBER = {"percent", "per", "the", "and", "for", "with", "from", "that", "this", "were", "was", "are"} | _MONTHS


def measure_values(text: str) -> dict[str, set[str]]:
    """Map the noun following a number to the set of normalized numbers seen with it.

    '579 passengers' -> {'passengers': {'579'}}. Years (1900-2100) are ignored.
    This is a cheap structural signal, not a parser.
    """
    out: dict[str, set[str]] = {}
    for num, noun in _MEASURE_RE.findall(text):
        norm = num.replace(",", "").rstrip(".")
        if norm.isdigit() and 1900 <= int(norm) <= 2100:
            continue
        noun = noun.lower()
        if noun in _STOP_AFTER_NUMBER:
            continue
        out.setdefault(noun, set()).add(norm)
    return out


def has_measure_conflict(new_text: str, old_text: str) -> bool:
    """True if some measure noun appears in both texts with disjoint numbers."""
    new_m, old_m = measure_values(new_text), measure_values(old_text)
    return any(noun in old_m and new_m[noun].isdisjoint(old_m[noun]) for noun in new_m)


def has_date_conflict(new_date: date | None, old_date: date | None, max_gap_days: int) -> bool:
    if new_date is None or old_date is None:
        return False
    gap = abs((new_date - old_date).days)
    return 0 < gap <= max_gap_days
