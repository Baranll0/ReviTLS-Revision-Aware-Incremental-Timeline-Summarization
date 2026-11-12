"""End-to-end incremental ReviTLS pipeline orchestrator."""

from datetime import date
from typing import Optional

from revitls.config import ReviTLSConfig
from revitls.pipeline.classifier import RevisionClassifier
from revitls.pipeline.extractor import EventExtractor
from revitls.pipeline.ranker import TimelineRanker
from revitls.pipeline.state_manager import EventStateManager
from revitls.pipeline.verifier import TargetedVerifier
from revitls.protocols import EmbedderProtocol, LLMProtocol
from revitls.retrieval.memory import EventMemory
from revitls.schemas import (
    Article,
    EventState,
    EventStatus,
    ExtractedEvent,
    RevisionAction,
    TimelineEntry,
)


def _parse_iso_date(raw: object) -> Optional[date]:
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


def _select_evidence_articles(
    seen_articles: list[Article],
    queries: list[str],
    incoming: ExtractedEvent,
    max_docs: int,
) -> list[Article]:
    """Retrieve relevant already-seen articles matching verification queries or entities."""
    terms = {
        tok.lower()
        for q in queries
        for tok in q.split()
        if len(tok) >= 4
    } | {e.lower() for e in incoming.entities if len(e) >= 3}

    if not terms:
        return seen_articles[-max_docs:]

    scored: list[tuple[int, Article]] = []
    for art in seen_articles:
        haystack = f"{art.title} {art.content}".lower()
        hits = sum(1 for t in terms if t in haystack)
        if hits > 0:
            scored.append((hits, art))

    scored.sort(key=lambda item: (item[0], item[1].publish_date), reverse=True)
    return [art for _, art in scored[:max_docs]]


class ReviTLSPipeline:
    """Runs revision-aware incremental timeline summarization over a chronological stream."""

    def __init__(
        self,
        config: ReviTLSConfig,
        llm: LLMProtocol,
        embedder: EmbedderProtocol,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.memory = EventMemory(config.retrieval)
        self.extractor = EventExtractor(llm)  # type: ignore[arg-type]
        self.classifier = RevisionClassifier(llm)  # type: ignore[arg-type]
        self.verifier = TargetedVerifier(llm, config.verifier)  # type: ignore[arg-type]
        self.state_mgr = EventStateManager()
        self.ranker = TimelineRanker(config.timeline)
        self._seen_articles: list[Article] = []
        self._cluster_seq = 0

    def step(self, article: Article) -> EventState:
        """Process a single incoming article and update the evolving EventState graph."""
        self._seen_articles.append(article)
        claim = self.extractor.extract(article)
        vec = self.embedder.encode(claim.summary, normalize_embeddings=self.config.embedding.normalize)
        candidates = self.memory.search(
            vec,
            reference_date=claim.event_date,
            entities=claim.entities,
        )

        for state, _score in candidates:
            result = self.classifier.classify(claim, state)
            if result.action == RevisionAction.NEW_EVENT:
                continue

            resolved_summary: Optional[str] = None
            resolved_date: Optional[date] = None
            resolved_status: Optional[EventStatus] = None

            if result.action == RevisionAction.REVISE_CONFLICT:
                queries = self.verifier.generate_verification_queries(state, claim)
                evidence = _select_evidence_articles(
                    self._seen_articles,
                    queries,
                    claim,
                    self.config.verifier.max_evidence_docs,
                )
                resolution = self.verifier.resolve_with_evidence(state, claim, evidence)
                if resolution.get("resolved_summary"):
                    resolved_summary = str(resolution["resolved_summary"]).strip()
                resolved_date = _parse_iso_date(resolution.get("resolved_date"))
                raw_status = str(resolution.get("status", "")).upper()
                if raw_status in {s.value for s in EventStatus}:
                    resolved_status = EventStatus(raw_status)

            updated = self.state_mgr.apply_action(
                state,
                claim,
                result,
                resolved_summary=resolved_summary,
                resolved_date=resolved_date,
                resolved_status=resolved_status,
            )
            new_vec = self.embedder.encode(
                updated.canonical_summary,
                normalize_embeddings=self.config.embedding.normalize,
            )
            self.memory.add_or_update(updated, new_vec)
            return updated

        self._cluster_seq += 1
        new_state = self.state_mgr.create_state(
            f"evt_{self._cluster_seq:05d}",
            article.topic,
            claim,
        )
        self.memory.add_or_update(new_state, vec)
        return new_state

    def run_stream(
        self,
        articles: list[Article],
        num_dates: Optional[int] = None,
    ) -> tuple[list[TimelineEntry], list[EventState]]:
        """Process an article collection in publication order and return timeline + final states."""
        for article in sorted(articles, key=lambda a: (a.publish_date, a.article_id)):
            self.step(article)
        states = self.memory.all_states()
        timeline = self.ranker.build_timeline(states, num_dates=num_dates)
        return timeline, states
