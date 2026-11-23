"""4-way Revision Relation Classifier replacing binary ISSAMEEVENT."""

from dataclasses import dataclass
from typing import Optional

from revitls.llm.client import LLMClient
from revitls.pipeline.cues import has_revision_cue
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
        is_explicit_rev = event.is_correction_or_update or has_revision_cue(event.summary)
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

    def classify_candidates(
        self,
        event: ExtractedEvent,
        candidates: list[EventState],
    ) -> tuple[Optional[int], ClassificationResult]:
        """Choose one matching state from a short ranked list in a single LLM call.

        Returns the selected zero-based candidate index and its relation. A ``None``
        index means that the event should create a new state. For a one-candidate
        list, preserve the original prompt and behavior exactly.
        """
        if not candidates:
            return None, ClassificationResult(action=RevisionAction.NEW_EVENT)
        if len(candidates) == 1:
            result = self.classify(event, candidates[0])
            return (0, result) if result.action != RevisionAction.NEW_EVENT else (None, result)

        candidate_block = "\n\n".join(
            f"Candidate {idx}:\n"
            f"Event Date: {candidate.event_date}\n"
            f"Summary: {candidate.canonical_summary}\n"
            f"Facts: {candidate.key_facts}"
            for idx, candidate in enumerate(candidates)
        )
        is_explicit_rev = event.is_correction_or_update or has_revision_cue(event.summary)
        hint = (
            " The incoming claim has a revision cue; if it refers to one candidate, "
            "choose REVISE_CONFLICT for that candidate."
            if is_explicit_rev
            else ""
        )
        user_prompt = (
            f"Incoming Article Publish Date: {event.publish_date}\n"
            f"Incoming Claimed Event Date: {event.event_date}\n"
            f"Incoming Summary: {event.summary}\n"
            f"Incoming Facts: {event.key_facts}\n\n"
            f"Ranked candidate states:\n{candidate_block}\n\n"
            "Select the candidate that represents the same underlying event, not merely a related topic. "
            "If none matches, use candidate_index=-1 and action=NEW_EVENT. "
            "For a selected candidate, choose exactly one action: CORROBORATE (same facts), "
            "UPDATE_PROGRESS (additive development), or REVISE_CONFLICT (correction, contradiction, "
            "supersession, or retraction). If any factual detail is corrected, use REVISE_CONFLICT."
            f"{hint}\nReturn valid JSON with candidate_index (integer), action, revision_type, and rationale."
        )
        default_payload = {
            "candidate_index": -1,
            "action": "NEW_EVENT",
            "revision_type": None,
            "rationale": "Fallback due to JSON parse failure.",
        }
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(CLASSIFIER_SYSTEM_PROMPT, user_prompt, default_payload)
        else:
            data = self.llm.complete_json(CLASSIFIER_SYSTEM_PROMPT, user_prompt)

        # Some backends and older test doubles return only the legacy action fields.
        # Fall back to the original classifier rather than silently discarding matches.
        if "candidate_index" not in data:
            for idx, candidate in enumerate(candidates):
                result = self.classify(event, candidate)
                if result.action != RevisionAction.NEW_EVENT:
                    return idx, result
            return None, ClassificationResult(action=RevisionAction.NEW_EVENT)

        try:
            selected_idx = int(data.get("candidate_index", -1))
            action = RevisionAction(data.get("action", "NEW_EVENT"))
        except (TypeError, ValueError):
            return None, ClassificationResult(action=RevisionAction.NEW_EVENT)
        try:
            rev_type = RevisionType(data["revision_type"]) if data.get("revision_type") else None
        except ValueError:
            rev_type = None
        result = ClassificationResult(
            action=action,
            revision_type=rev_type,
            rationale=str(data.get("rationale", "")),
        )
        if action == RevisionAction.NEW_EVENT or not 0 <= selected_idx < len(candidates):
            return None, ClassificationResult(
                action=RevisionAction.NEW_EVENT,
                rationale=result.rationale,
            )
        return selected_idx, result
