"""4-way Revision Relation Classifier replacing binary ISSAMEEVENT."""

from dataclasses import dataclass
from typing import Optional

from revitls.llm.client import LLMClient
from revitls.schemas import EventState, ExtractedEvent, RevisionAction, RevisionType


CLASSIFIER_SYSTEM_PROMPT = (
    "You compare a newly arrived news event claim against an existing event state in a timeline. "
    "Classify the relation into exactly one action:\n"
    "- NEW_EVENT: Different event.\n"
    "- CORROBORATE: Same event, repeats or confirms existing facts without new developments.\n"
    "- UPDATE_PROGRESS: Same event, adds a genuinely new subsequent development.\n"
    "- REVISE_CONFLICT: Same event, but contradicts, updates a factual number/date/actor, "
    "or retracts/debunks the existing event state.\n"
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
        user_prompt = (
            f"Existing Event Date: {candidate.event_date}\n"
            f"Existing Summary: {candidate.canonical_summary}\n"
            f"Existing Facts: {candidate.key_facts}\n\n"
            f"Incoming Article Publish Date: {event.publish_date}\n"
            f"Incoming Claimed Event Date: {event.event_date}\n"
            f"Incoming Summary: {event.summary}\n"
            f"Incoming Facts: {event.key_facts}\n\n"
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
