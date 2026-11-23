"""Revision-specific evaluation: case-linked resolution, stale leakage, false retraction.

Case linkage is by article-id family: an article belongs to a case when its id equals
``case_id`` or starts with ``case_id + "_"``. Initial reports use ``{case_id}_init_*``.
"""

import re
from dataclasses import dataclass
from typing import Any

from revitls.schemas import EventState, EventStatus, TimelineEntry

POSITIVE_TYPES = {"revision", "retraction"}
NEGATIVE_TYPES = {"neg_denial", "neg_progress"}


@dataclass
class RevisionTestCase:
    """Ground-truth expectation for an event chain.

    ``case_type``: ``revision`` / ``retraction`` (positives) or ``neg_denial`` /
    ``neg_progress`` (negative controls: the original event must stay active and
    its ``expected_final_fact`` must remain in the output).
    """

    case_id: str
    expected_final_fact: str
    stale_fact_substring: str
    should_be_retracted: bool = False
    case_type: str = ""
    cue: str = "explicit"
    revision_type: str = ""

    @property
    def kind(self) -> str:
        if self.case_type:
            return self.case_type
        return "retraction" if self.should_be_retracted else "revision"

    @property
    def is_negative(self) -> bool:
        return self.kind in NEGATIVE_TYPES


def _norm(text: str) -> str:
    """Lowercase and drop thousands separators so '2,345' matches '2345'."""
    return re.sub(r"(?<=\d),(?=\d{3})", "", text.lower())


def _in_family(article_id: str, case_id: str) -> bool:
    return article_id == case_id or article_id.startswith(case_id + "_")


def _is_init(article_id: str, case_id: str) -> bool:
    return article_id == case_id or article_id.startswith(case_id + "_init")


def _state_text(s: EventState | TimelineEntry) -> str:
    date_str = s.event_date.isoformat() if s.event_date else ""
    summary = s.canonical_summary if isinstance(s, EventState) else s.summary
    return _norm(f"{date_str} {summary}")


def evaluate_cases(states: list[EventState], cases: list[RevisionTestCase]) -> list[dict[str, Any]]:
    """Per-case outcomes over the final state repository."""
    results: list[dict[str, Any]] = []
    for case in cases:
        linked = [s for s in states if any(_in_family(a, case.case_id) for a in s.supporting_article_ids)]
        active = [s for s in linked if s.status != EventStatus.RETRACTED]
        active_text = " ".join(_state_text(s) for s in active)
        stale = _norm(case.stale_fact_substring.strip())
        expected = _norm(case.expected_final_fact.strip())
        stale_present = bool(stale) and stale in active_text
        init_states = [s for s in linked if any(_is_init(a, case.case_id) for a in s.supporting_article_ids)]
        init_retracted = any(s.status == EventStatus.RETRACTED for s in init_states)

        if case.is_negative:
            fact_kept = bool(expected) and expected in active_text
            resolved = fact_kept and not init_retracted
            false_retraction = init_retracted
        elif case.should_be_retracted or case.kind == "retraction":
            resolved = bool(linked) and any(s.status == EventStatus.RETRACTED for s in linked) and not stale_present
            false_retraction = False
        else:
            resolved = bool(linked) and bool(expected) and expected in active_text and not stale_present
            false_retraction = False
        results.append(
            {
                "case_id": case.case_id,
                "kind": case.kind,
                "cue": case.cue,
                "revision_type": case.revision_type,
                "linked": bool(linked),
                "resolved": bool(resolved),
                "stale_leak": bool(stale_present),
                "false_retraction": bool(false_retraction),
                "states": [
                    {
                        "status": st.status.value,
                        "date": st.event_date.isoformat() if st.event_date else "",
                        "summary": st.canonical_summary[:300],
                    }
                    for st in linked
                ],
            }
        )
    return results


def _rate(num: int, den: int) -> float:
    return num / den if den else 0.0


def evaluate_revision_states(
    states: list[EventState], cases: list[RevisionTestCase]
) -> dict[str, float]:
    """Aggregate state-level metrics.

    RRA and SFLR are computed over positive cases only; negative controls are reported
    separately (``negative_control_pass_rate``, ``false_retraction_rate``).
    """
    if not cases:
        return {"revision_resolution_accuracy": 0.0, "stale_fact_leakage_rate": 0.0}
    per_case = evaluate_cases(states, cases)
    pos = [r for r in per_case if r["kind"] in POSITIVE_TYPES]
    neg = [r for r in per_case if r["kind"] in NEGATIVE_TYPES]
    retr = [r for r in pos if r["kind"] == "retraction"]
    return {
        "revision_resolution_accuracy": _rate(sum(r["resolved"] for r in pos), len(pos)),
        "stale_fact_leakage_rate": _rate(sum(r["stale_leak"] for r in pos), len(pos)),
        "retraction_recall": _rate(sum(r["resolved"] for r in retr), len(retr)),
        "negative_control_pass_rate": _rate(sum(r["resolved"] for r in neg), len(neg)),
        "false_retraction_rate": _rate(sum(r["false_retraction"] for r in neg), len(neg)),
    }


def evaluate_revision_timeline(
    timeline: list[TimelineEntry], cases: list[RevisionTestCase]
) -> dict[str, float]:
    """Output-level diagnostics over the emitted timeline (not the full state repository).

    ``timeline_sflr``: share of positive cases whose stale fact appears in a linked emitted entry.
    ``timeline_rra``: share of positive cases whose expected fact is emitted without the stale fact
    (retraction cases: no linked entry carries the stale fact).
    """
    pos = [c for c in cases if not c.is_negative]
    if not pos:
        return {"timeline_rra": 0.0, "timeline_sflr": 0.0}
    resolved = leaks = 0
    for case in pos:
        linked = [e for e in timeline if any(_in_family(a, case.case_id) for a in e.supporting_article_ids)]
        text = " ".join(_state_text(e) for e in linked)
        stale = _norm(case.stale_fact_substring.strip())
        expected = _norm(case.expected_final_fact.strip())
        leak = bool(stale) and stale in text
        leaks += int(leak)
        if case.kind == "retraction":
            resolved += int(not leak)
        else:
            resolved += int(bool(expected) and expected in text and not leak)
    return {"timeline_rra": resolved / len(pos), "timeline_sflr": leaks / len(pos)}
