"""Revision-specific evaluation metrics: Resolution Accuracy & Stale-Fact Leakage."""

from dataclasses import dataclass
from revitls.schemas import EventState, EventStatus


@dataclass
class RevisionTestCase:
    """Ground-truth expectation for an event that underwent factual revision or retraction."""

    case_id: str
    expected_final_fact: str
    stale_fact_substring: str
    should_be_retracted: bool = False


def evaluate_revision_states(
    states: list[EventState], cases: list[RevisionTestCase]
) -> dict[str, float]:
    """Evaluate how accurately final EventStates reflect revisions and suppress stale facts."""
    if not cases:
        return {"revision_resolution_accuracy": 0.0, "stale_fact_leakage_rate": 0.0}

    resolved_count = 0
    stale_leaks = 0
    combined_active_text = " ".join(
        f"{s.event_date.isoformat() if s.event_date else ''} {s.canonical_summary}".lower()
        for s in states
        if s.status != EventStatus.RETRACTED
    )

    for case in cases:
        stale_present = case.stale_fact_substring.lower() in combined_active_text
        if stale_present:
            stale_leaks += 1

        if case.should_be_retracted:
            if not stale_present:
                resolved_count += 1
        elif case.expected_final_fact.lower() in combined_active_text and not stale_present:
            resolved_count += 1

    total = len(cases)
    return {
        "revision_resolution_accuracy": resolved_count / total,
        "stale_fact_leakage_rate": stale_leaks / total,
    }
