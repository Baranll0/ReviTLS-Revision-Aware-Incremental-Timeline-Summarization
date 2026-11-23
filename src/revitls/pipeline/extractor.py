"""Extract structured event claims, entities, and mentioned dates from an article."""

from datetime import date
from revitls.llm.client import LLMClient
from revitls.schemas import Article, ExtractedEvent


EXTRACTOR_SYSTEM_PROMPT = (
    "You are an information extraction assistant for news timeline summarization. "
    "Extract the primary event claim, its exact event date (YYYY-MM-DD if known), "
    "key named entities, core factual attributes (counts, locations, outcomes), and "
    "whether the article explicitly corrects, updates, or retracts an earlier report. "
    "Return valid JSON only."
)


SLOT_KEYS = ("who", "what", "where", "why", "how")


class EventExtractor:
    """Distills incoming articles into structured event claims."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def extract(self, article: Article) -> ExtractedEvent:
        user_prompt = (
            f"Topic: {article.topic}\n"
            f"Publish Date: {article.publish_date.isoformat()}\n"
            f"Title: {article.title}\n"
            f"Content:\n{article.content[:4000]}\n\n"
            "Return JSON with keys: event_date (YYYY-MM-DD or null), summary (1-2 sentences), "
            "entities (list of strings), key_facts (dict of attribute->value), "
            "slots (dict with optional string keys who, what, where, why, how; omit unknown), "
            "is_correction_or_update (boolean)."
        )
        default_payload = {
            "event_date": article.publish_date.isoformat(),
            "summary": article.title or article.content[:240],
            "entities": [],
            "key_facts": {},
            "slots": {},
            "is_correction_or_update": False,
        }
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(EXTRACTOR_SYSTEM_PROMPT, user_prompt, default_payload)
        else:
            data = self.llm.complete_json(EXTRACTOR_SYSTEM_PROMPT, user_prompt)
        parsed_date = None
        if data.get("event_date"):
            try:
                parsed_date = date.fromisoformat(str(data["event_date"])[:10])
            except ValueError:
                parsed_date = None

        raw_slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}
        raw_facts = data.get("key_facts") if isinstance(data.get("key_facts"), dict) else {}
        raw_entities = data.get("entities") if isinstance(data.get("entities"), list) else []
        return ExtractedEvent(
            source_article_id=article.article_id,
            publish_date=article.publish_date,
            event_date=parsed_date or article.publish_date,
            summary=str(data.get("summary", article.title)).strip(),
            entities=[str(e) for e in raw_entities],
            key_facts={str(k): str(v) for k, v in raw_facts.items()},
            slots={
                str(k).lower(): str(v).strip()
                for k, v in raw_slots.items()
                if str(k).lower() in SLOT_KEYS and str(v).strip() and str(v).strip().lower() != "null"
            },
            is_correction_or_update=bool(data.get("is_correction_or_update", False)),
        )
