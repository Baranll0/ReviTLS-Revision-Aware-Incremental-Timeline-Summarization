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


_CROSS_ENCODER_CACHE: dict[tuple[str, str], object] = {}


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
        verifier_llm: Optional[LLMProtocol] = None,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.memory = EventMemory(config.retrieval)
        active_verifier_llm = verifier_llm or llm
        self.extractor = EventExtractor(llm)  # type: ignore[arg-type]
        self.routine_classifier = RevisionClassifier(llm)  # type: ignore[arg-type]
        self.classifier = RevisionClassifier(active_verifier_llm)  # type: ignore[arg-type]
        self.verifier = TargetedVerifier(active_verifier_llm, config.verifier)  # type: ignore[arg-type]
        self.state_mgr = EventStateManager()
        self.ranker = TimelineRanker(config.timeline)
        self._seen_articles: list[Article] = []
        self._cluster_seq = 0
        self._cross_encoder = None
        if config.retrieval.cross_encoder_model:
            ce_key = (config.retrieval.cross_encoder_model, config.embedding.device)
            if ce_key not in _CROSS_ENCODER_CACHE:
                from sentence_transformers import CrossEncoder

                _CROSS_ENCODER_CACHE[ce_key] = CrossEncoder(
                    config.retrieval.cross_encoder_model,
                    device=config.embedding.device,
                )
            self._cross_encoder = _CROSS_ENCODER_CACHE[ce_key]

    def _is_revision_candidate(self, article: Article, claim: ExtractedEvent) -> bool:
        haystack = f"{article.title} {claim.summary}".lower()
        return claim.is_correction_or_update or any(
            w in haystack
            for w in (
                "revise",
                "correction",
                "correcting",
                "clarif",
                "retract",
                "debunk",
                "supersed",
                "rule out",
                "earlier report",
                "previously stated",
                "false",
            )
        )

    def step(
        self,
        article: Article,
        pre_extracted_claim: Optional[ExtractedEvent] = None,
    ) -> EventState:
        """Process a single incoming article and update the evolving EventState graph."""
        self._seen_articles.append(article)
        claim = pre_extracted_claim or self.extractor.extract(article)
        vec = self.embedder.encode(claim.summary, normalize_embeddings=self.config.embedding.normalize)
        candidates = self.memory.search(
            vec,
            reference_date=claim.event_date,
            entities=claim.entities,
        )

        is_rev_cand = self._is_revision_candidate(article, claim)
        if self.config.retrieval.strict_temporal_gate and not is_rev_cand:
            candidates = [
                (st, sc) for st, sc in candidates if st.event_date == claim.event_date
            ]

        if self._cross_encoder is not None and len(candidates) > 1:
            import numpy as np

            pairs = [(st.canonical_summary, claim.summary) for st, _ in candidates]
            ce_scores = self._cross_encoder.predict(pairs)
            art_tokens = {
                tok.strip(".,:;()\"'").lower()
                for tok in f"{article.title} {article.content[:600]}".split()
                if len(tok.strip(".,:;()\"'")) >= 4
            }
            rescored = []
            for (st, bi_sc), ce_sc in zip(candidates, ce_scores):
                st_tokens = {
                    tok.strip(".,:;()\"'").lower()
                    for tok in st.canonical_summary.split()
                    if len(tok.strip(".,:;()\"'")) >= 4
                }
                lex_overlap = len(art_tokens & st_tokens) / max(1, len(st_tokens))
                ce_prob = float(1.0 / (1.0 + np.exp(-float(ce_sc))))
                corr_bonus = 0.12 if (is_rev_cand and st.corroboration_count >= 2 and lex_overlap > 0) else 0.0
                rescored.append((st, 0.65 * bi_sc + 0.20 * ce_prob + 0.15 * lex_overlap + corr_bonus))
            rescored.sort(key=lambda item: item[1], reverse=True)
            candidates = rescored

        max_verify_cands = 3 if is_rev_cand else 1
        active_cls = self.classifier if is_rev_cand else self.routine_classifier
        for state, _score in candidates[:max_verify_cands]:
            result = active_cls.classify(claim, state)
            if result.action == RevisionAction.NEW_EVENT:
                continue
            if (
                self.config.retrieval.strict_temporal_gate
                and state.event_date != claim.event_date
                and result.action != RevisionAction.REVISE_CONFLICT
            ):
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
        ordered = sorted(articles, key=lambda a: (a.publish_date, a.article_id))
        if self.config.verifier_llm is not None:
            # Algorithm 1 Two-Phase Schedule:
            # Phase 1: 7B Extraction & Routine Same-Day Clustering (\rho_{i,j} == 0)
            claims = [self.extractor.extract(art) for art in ordered]
            rev_queue: list[tuple[Article, ExtractedEvent]] = []
            for article, claim in zip(ordered, claims):
                if self._is_revision_candidate(article, claim):
                    rev_queue.append((article, claim))
                else:
                    self.step(article, pre_extracted_claim=claim)
            # Phase 2: Cross-Encoder Reranking & 14B Verification on \mathcal{Q}_{rev} (\rho_{i,j} == 1)
            for article, claim in rev_queue:
                self.step(article, pre_extracted_claim=claim)
        else:
            for article in ordered:
                self.step(article)
        states = self.memory.all_states()
        timeline = self.ranker.build_timeline(states, num_dates=num_dates)
        return timeline, states
