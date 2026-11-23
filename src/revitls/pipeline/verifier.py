"""Triggered CHRONOS-style self-questioning & query rewriting for conflict resolution."""

import re

from revitls.config import VerifierConfig
from revitls.llm.client import LLMClient
from revitls.schemas import Article, EventState, ExtractedEvent


QUESTIONER_SYSTEM_PROMPT = (
    "You are an investigative news editor resolving a factual conflict, revision, or retraction "
    "in a developing news timeline.\n"
    "CRITICAL RULES FOR resolved_summary:\n"
    "1. State ONLY the verified, updated current truth. Completely purge and omit superseded numbers, "
    "outdated dates, or false theories. NEVER echo or repeat the old superseded claim (e.g. if a "
    "bridge closure was first reported as 3 days and later corrected to 10 days, write 'The bridge "
    "will be closed for 10 days', do NOT write 'extended from the earlier 3-day estimate').\n"
    "2. Set status to 'RETRACTED' ONLY if the entire event itself never happened (e.g., a false rumor where "
    "no incident occurred at all). If the event DID happen but its cause, number, or date was corrected "
    "you MUST set status to 'REVISED'.\n"
    "3. For RETRACTED, provide retraction_evidence as an exact, verbatim quote from the supplied evidence "
    "that establishes non-occurrence of the underlying event itself. A denial, apology, quotation, or "
    "retraction of a public statement is not sufficient. If no such quote exists, use status='REVISED' "
    "and retraction_evidence=null.\n"
    "4. If a date was corrected, put the exact new date in resolved_date (YYYY-MM-DD) and include the "
    "YYYY-MM-DD date in resolved_summary.\n"
    "Return valid JSON only."
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
            "Synthesize the single most accurate current truth.\n"
            "CRITICAL: Do NOT mention or repeat the old, superseded, or debunked numbers/dates in resolved_summary. "
            "Set status='RETRACTED' only when the underlying event itself is shown not to have occurred; "
            "a denial or retraction of a statement is insufficient. For RETRACTED, retraction_evidence must "
            "be an exact verbatim quote from Supporting Evidence establishing non-occurrence; otherwise set "
            "status='REVISED' and retraction_evidence=null. "
            "Return JSON: {\"resolved_summary\": \"...\", \"resolved_date\": \"YYYY-MM-DD or null\", "
            "\"status\": \"ACTIVE|REVISED|RETRACTED\", \"retraction_evidence\": \"exact quote or null\", "
            "\"rationale\": \"...\"}."
        )
        default_payload = {
            "resolved_summary": incoming.summary,
            "resolved_date": incoming.event_date.isoformat() if incoming.event_date else None,
            "status": "REVISED",
            "retraction_evidence": None,
            "rationale": "Fallback resolution.",
        }
        if hasattr(self.llm, "complete_json_safe"):
            result = self.llm.complete_json_safe(QUESTIONER_SYSTEM_PROMPT, user_prompt, default_payload)
        else:
            result = self.llm.complete_json(QUESTIONER_SYSTEM_PROMPT, user_prompt)

        if str(result.get("status", "")).upper() == "RETRACTED":
            quote = result.get("retraction_evidence")
            normalized_quote = re.sub(r"\s+", " ", str(quote or "")).strip().casefold()
            exact_quote_found = bool(normalized_quote) and any(
                normalized_quote in re.sub(r"\s+", " ", f"{article.title} {article.content}").casefold()
                for article in evidence_articles
            )
            if exact_quote_found:
                result["_retraction_evidence_validated"] = True
            else:
                # A model label without a source-grounded verbatim span cannot
                # remove an event from the final timeline.
                result["status"] = "REVISED"
                result["retraction_evidence"] = None
                result["_retraction_evidence_validated"] = False
        return result
