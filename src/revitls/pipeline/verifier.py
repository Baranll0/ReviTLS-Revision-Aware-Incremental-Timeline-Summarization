"""Triggered CHRONOS-style self-questioning & query rewriting for conflict resolution."""

from revitls.config import VerifierConfig
from revitls.llm.client import LLMClient
from revitls.schemas import Article, EventState, ExtractedEvent


QUESTIONER_SYSTEM_PROMPT = (
    "You are an investigative news editor resolving a factual conflict or information gap "
    "in a developing news timeline. Propose concise, targeted verification questions and "
    "short search queries to check official or newer reports. Return valid JSON only."
)


class TargetedVerifier:
    """Generates targeted verification queries only when conflicts or gaps arise."""

    def __init__(self, llm: LLMClient, config: VerifierConfig) -> None:
        self.llm = llm
        self.config = config

    def generate_verification_queries(
        self, state: EventState, incoming: ExtractedEvent
    ) -> list[str]:
        """Produce focused search queries to verify a disputed claim."""
        if not self.config.enable_targeted_verification:
            return []

        user_prompt = (
            f"Topic: {state.topic}\n"
            f"Current State ({state.event_date}): {state.canonical_summary}\n"
            f"Conflicting/New Claim ({incoming.event_date}, pub {incoming.publish_date}): "
            f"{incoming.summary}\n\n"
            f"Generate up to {self.config.max_verification_questions} focused search queries "
            "to verify the accurate facts/date. Return JSON: {\"queries\": [\"...\"]}."
        )
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(QUESTIONER_SYSTEM_PROMPT, user_prompt, {"queries": []})
        else:
            data = self.llm.complete_json(QUESTIONER_SYSTEM_PROMPT, user_prompt)
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
        return queries[: self.config.max_verification_questions]

    def resolve_with_evidence(
        self,
        state: EventState,
        incoming: ExtractedEvent,
        evidence_articles: list[Article],
    ) -> dict[str, object]:
        """Synthesize a verified canonical summary and event date given retrieved evidence."""
        evidence_block = "\n".join(
            f"- [{a.publish_date}] {a.title}: {a.content[:400]}"
            for a in evidence_articles[: self.config.max_evidence_docs]
        )
        user_prompt = (
            f"Topic: {state.topic}\n"
            f"Previous State ({state.event_date}): {state.canonical_summary}\n"
            f"Incoming Claim ({incoming.event_date}, pub {incoming.publish_date}): {incoming.summary}\n"
            f"Supporting Evidence:\n{evidence_block or 'None'}\n\n"
            "Decide the most accurate current truth, prioritizing official corrections and newer "
            "verified reports. Return JSON with keys: resolved_summary (string), "
            "resolved_date (YYYY-MM-DD or null), status (ACTIVE|REVISED|RETRACTED), rationale (string)."
        )
        default_payload = {
            "resolved_summary": incoming.summary,
            "resolved_date": incoming.event_date.isoformat() if incoming.event_date else None,
            "status": "REVISED",
            "rationale": "Fallback resolution.",
        }
        if hasattr(self.llm, "complete_json_safe"):
            return self.llm.complete_json_safe(QUESTIONER_SYSTEM_PROMPT, user_prompt, default_payload)
        return self.llm.complete_json(QUESTIONER_SYSTEM_PROMPT, user_prompt)
