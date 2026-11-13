"""4-way Revision Relation Classifier replacing binary ISSAMEEVENT."""

from dataclasses import dataclass
from typing import Optional

from revitls.llm.client import LLMClient
from revitls.schemas import EventState, ExtractedEvent, RevisionAction, RevisionType


CLASSIFIER_SYSTEM_PROMPT = (
    "You compare a newly arrived news event claim against an existing event state in a timeline.\n"
    "Classify the relation into exactly one action:\n"
    "- NEW_EVENT: Different event.\n"
    "- CORROBORATE: Same event, repeats or confirms existing facts without changing any numbers, dates, or causes.\n"
    "- UPDATE_PROGRESS: Same event, a subsequent chronological milestone that builds upon the past WITHOUT correcting, replacing, or contradicting any previous numbers, dates, causes, or claims.\n"
    "- REVISE_CONFLICT: Choose this whenever the incoming claim corrects, replaces, supersedes, contradicts, or updates a previous number (e.g. leak/casualty count), date, cause, or debunks/retracts a rumor. If any factual detail changed or was corrected, you MUST choose REVISE_CONFLICT.\n"
    "Return valid JSON only."
)


@dataclass
class ClassificationResult:
    action: RevisionAction
    revision_type: Optional[RevisionType] = None
    rationale: str = ""


class RevisionClassifier:
    """Classifies how an incoming claim relates to a candidate EventState."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def classify(self, event: ExtractedEvent, candidate: EventState) -> ClassificationResult:
        is_explicit_rev = event.is_correction_or_update or any(
            w in event.summary.lower()
            for w in ("revise", "correct", "clarif", "retract", "debunk", "supersed", "rule out", "not ")
        )
        hint = (
            "\nNOTE: The incoming claim indicates a correction, revision, or retraction. "
            "If it refers to this event, classify as REVISE_CONFLICT."
            if is_explicit_rev
            else ""
        )
        user_prompt = (
            f"Existing Event Date: {candidate.event_date}\n"
            f"Existing Summary: {candidate.canonical_summary}\n"
            f"Existing Facts: {candidate.key_facts}\n\n"
            f"Incoming Article Publish Date: {event.publish_date}\n"
            f"Incoming Claimed Event Date: {event.event_date}\n"
            f"Incoming Summary: {event.summary}\n"
            f"Incoming Facts: {event.key_facts}{hint}\n\n"
            "Return JSON with keys: action (NEW_EVENT|CORROBORATE|UPDATE_PROGRESS|REVISE_CONFLICT), "
            "revision_type (NUMERIC_FACT|DATE_CORRECTION|ENTITY_ATTRIBUTION|RETRACTION|STATE_PROGRESSION|null), "
            "rationale (short explanation)."
        )
        default_payload = {
            "action": "NEW_EVENT",
            "revision_type": None,
            "rationale": "Fallback due to JSON parse failure.",
        }
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(CLASSIFIER_SYSTEM_PROMPT, user_prompt, default_payload)
        else:
            data = self.llm.complete_json(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
        try:
            action = RevisionAction(data.get("action", "NEW_EVENT"))
        except ValueError:
            action = RevisionAction.NEW_EVENT
        try:
            rev_type = RevisionType(data["revision_type"]) if data.get("revision_type") else None
        except ValueError:
            rev_type = None
        return ClassificationResult(
            action=action,
            revision_type=rev_type,
            rationale=str(data.get("rationale", "")),
        )
