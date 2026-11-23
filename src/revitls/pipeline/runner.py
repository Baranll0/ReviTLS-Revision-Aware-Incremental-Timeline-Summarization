"""End-to-end incremental ReviTLS pipeline orchestrator."""

from datetime import date
import os
import time
from typing import Optional
import numpy as np
from tqdm.auto import tqdm

from revitls.config import ReviTLSConfig
from revitls.pipeline.classifier import ClassificationResult, RevisionClassifier
from revitls.pipeline.cues import has_date_conflict, has_measure_conflict, has_revision_cue
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
        selection_mode: str = "eq6",
        disable_purge: bool = False,
        rho_mode: str = "both",
        defer_phase2: bool = True,
        purge_mode: str = "clause",
        guarded_retract: bool = False,
        disable_retract: bool = False,
        overwrite_only: bool = False,
    ) -> None:
        self.config = config
        self.embedder = embedder
        self.selection_mode = selection_mode
        self.disable_purge = disable_purge
        if rho_mode not in self._RHO_MODES:
            raise ValueError(f"Unknown rho_mode {rho_mode!r}; expected one of {sorted(self._RHO_MODES)}")
        self.rho_mode = self._RHO_MODES[rho_mode]
        self.defer_phase2 = defer_phase2
        self.memory = EventMemory(config.retrieval)
        active_verifier_llm = verifier_llm or llm
        self.extractor = EventExtractor(llm)  # type: ignore[arg-type]
        self.routine_classifier = RevisionClassifier(llm)  # type: ignore[arg-type]
        self.classifier = RevisionClassifier(active_verifier_llm)  # type: ignore[arg-type]
        self.verifier = TargetedVerifier(active_verifier_llm, config.verifier)  # type: ignore[arg-type]
        self.state_mgr = EventStateManager(
            purge_mode=purge_mode,
            guarded_retract=guarded_retract,
            disable_retract=disable_retract,
            overwrite_only=overwrite_only,
        )
        self.ranker = TimelineRanker(config.timeline)
        self._seen_articles: list[Article] = []
        self._cluster_seq = 0
        self._ce_reranked_calls = 0
        self._ce_top1_flips = 0
        self._ce_time_s = 0.0
        self._rev_top1_records: list[dict[str, str]] = []
        self._deferred_routine_log: dict[str, dict[str, object]] = {}
        self._forced_rev_ids: set[str] = set()
        self._probe_hits = 0
        self._record_deferred_routine_log = False
        self._routine_non_top1_matches = 0
        self._routine_no_match_decisions = 0
        self._routine_replay_reembeddings = 0
        self._deferred_missing_log_count = 0
        self._retraction_quote_validations = 0
        self._retraction_quote_rejections = 0
        self._diag_actions: dict[str, int] = {
            "NEW_EVENT": 0,
            "CORROBORATE": 0,
            "UPDATE_PROGRESS": 0,
            "REVISE_CONFLICT": 0,
        }
        self.last_diagnostics: dict[str, object] = {}
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

    def _is_lex_revision(self, article: Article, claim: ExtractedEvent) -> bool:
        return has_revision_cue(f"{article.title} {claim.summary}")

    def _is_revision_candidate(self, article: Article, claim: ExtractedEvent) -> bool:
        if self.rho_mode == "none":
            return False
        if self.rho_mode == "both" and article.article_id in self._forced_rev_ids:
            return True
        if self.rho_mode == "llm":
            return bool(claim.is_correction_or_update)
        if self.rho_mode == "lexicon":
            return self._is_lex_revision(article, claim)
        return bool(claim.is_correction_or_update) or self._is_lex_revision(article, claim)

    _RHO_MODES = {"both": "both", "llm": "llm", "llm_only": "llm", "lexicon": "lexicon", "lexicon_only": "lexicon", "none": "none"}

    def _implicit_probe(self, article: Article, claim: ExtractedEvent, vec: Optional[np.ndarray] = None) -> bool:
        """rho_struct: cue-free revision signal from a similar cross-date state with a conflicting value."""
        rc = self.config.retrieval
        if not rc.implicit_probe or self.rho_mode != "both":
            return False
        if vec is None:
            vec = self.embedder.encode(claim.summary, normalize_embeddings=self.config.embedding.normalize)
        claim_text = f"{claim.summary} " + " ".join(f"{v} {k}" for k, v in claim.key_facts.items())
        for state, score in self.memory.search(vec, reference_date=claim.event_date, entities=claim.entities):
            if score < rc.implicit_probe_sim:
                break
            state_text = f"{state.canonical_summary} " + " ".join(f"{v} {k}" for k, v in state.key_facts.items())
            same_day = state.event_date == claim.event_date
            if has_measure_conflict(claim_text, state_text) or (
                not same_day and has_date_conflict(claim.event_date, state.event_date, rc.implicit_probe_date_gap_days)
            ):
                self._probe_hits += 1
                return True
        return False

    def step(
        self,
        article: Article,
        pre_extracted_claim: Optional[ExtractedEvent] = None,
        probe: bool = True,
    ) -> EventState:
        """Process a single incoming article and update the evolving EventState graph."""
        self._seen_articles.append(article)
        claim = pre_extracted_claim or self.extractor.extract(article)
        vec = self.embedder.encode(claim.summary, normalize_embeddings=self.config.embedding.normalize)
        if probe and not self._is_revision_candidate(article, claim) and self._implicit_probe(article, claim, vec):
            self._forced_rev_ids.add(article.article_id)
        candidates = self.memory.search(
            vec,
            reference_date=claim.event_date,
            entities=claim.entities,
        )

        is_rev_cand = self._is_revision_candidate(article, claim)
        if self.config.retrieval.strict_temporal_gate and not is_rev_cand:
            rw = self.config.retrieval.routine_window_days
            candidates = [
                (st, sc)
                for st, sc in candidates
                if st.event_date is not None and claim.event_date is not None and abs((claim.event_date - st.event_date).days) <= rw
            ]

        if self._cross_encoder is not None and len(candidates) > 1:
            import numpy as np

            t_ce = time.perf_counter()
            pre_top1_id = candidates[0][0].cluster_id
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
                corr_bonus = self.config.retrieval.corroboration_bonus if (is_rev_cand and st.corroboration_count >= 2 and lex_overlap > 0) else 0.0
                rescored.append((st, 0.65 * bi_sc + 0.20 * ce_prob + 0.15 * lex_overlap + corr_bonus))
            rescored.sort(key=lambda item: item[1], reverse=True)
            self._ce_reranked_calls += 1
            if rescored[0][0].cluster_id != pre_top1_id:
                self._ce_top1_flips += 1
            self._ce_time_s += time.perf_counter() - t_ce
            candidates = rescored

        if is_rev_cand and candidates:
            top1_st = candidates[0][0]
            self._rev_top1_records.append(
                {
                    "article_id": article.article_id,
                    "top1_date": top1_st.event_date.isoformat() if top1_st.event_date else "",
                    "top1_summary": top1_st.canonical_summary[:120],
                    "pool_size": str(len(candidates)),
                }
            )
        if is_rev_cand:
            for state, _score in candidates[:3]:
                result = self.classifier.classify(claim, state)
                if result.action == RevisionAction.NEW_EVENT:
                    continue
                updated = self._apply_candidate_result(article, claim, state, result)
                if updated is not None:
                    return updated
        elif candidates:
            # Compare a short same-date candidate list in one call. The previous
            # routine path examined only top-1 and created a duplicate state as
            # soon as that single candidate was rejected.
            shortlist = candidates[: self.config.retrieval.routine_candidate_k]
            selected_idx, result = self.routine_classifier.classify_candidates(
                claim,
                [state for state, _score in shortlist],
            )
            if selected_idx is not None:
                if selected_idx > 0:
                    self._routine_non_top1_matches += 1
                state = shortlist[selected_idx][0]
                updated = self._apply_candidate_result(article, claim, state, result)
                if updated is not None:
                    return updated
            else:
                self._routine_no_match_decisions += 1

        self._cluster_seq += 1
        new_state = self.state_mgr.create_state(
            f"evt_{self._cluster_seq:05d}",
            article.topic,
            claim,
        )
        self.memory.add_or_update(new_state, vec)
        if self._record_deferred_routine_log and not is_rev_cand:
            self._deferred_routine_log[article.article_id] = {
                "kind": "new",
                "cluster_id": new_state.cluster_id,
                "summary_after": new_state.canonical_summary,
                "vector_after": vec,
            }
        return new_state

    def _apply_candidate_result(
        self,
        article: Article,
        claim: ExtractedEvent,
        state: EventState,
        result: ClassificationResult,
    ) -> Optional[EventState]:
        if result.action == RevisionAction.NEW_EVENT:
            return None
        if (
            self.config.retrieval.strict_temporal_gate
            and state.event_date is not None
            and claim.event_date is not None
            and abs((claim.event_date - state.event_date).days) > self.config.retrieval.routine_window_days
            and result.action != RevisionAction.REVISE_CONFLICT
        ):
            return None

        resolved_summary: Optional[str] = None
        resolved_date: Optional[date] = None
        resolved_status: Optional[EventStatus] = None
        if result.action == RevisionAction.REVISE_CONFLICT and not self.disable_purge:
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
            if "_retraction_evidence_validated" in resolution:
                if resolution["_retraction_evidence_validated"]:
                    self._retraction_quote_validations += 1
                else:
                    self._retraction_quote_rejections += 1
            if raw_status in {s.value for s in EventStatus}:
                resolved_status = EventStatus(raw_status)
        elif result.action == RevisionAction.REVISE_CONFLICT and self.disable_purge:
            result.action = RevisionAction.UPDATE_PROGRESS

        self._diag_actions[result.action.value] = self._diag_actions.get(result.action.value, 0) + 1
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
        if self._record_deferred_routine_log and not self._is_revision_candidate(article, claim):
            self._deferred_routine_log[article.article_id] = {
                "kind": "update",
                "cluster_id": state.cluster_id,
                "result": ClassificationResult(
                    action=result.action,
                    revision_type=result.revision_type,
                    rationale=result.rationale,
                ),
                "resolved_summary": resolved_summary,
                "resolved_date": resolved_date,
                "resolved_status": resolved_status,
                "summary_after": updated.canonical_summary,
                "vector_after": new_vec,
            }
        return updated

    def _replay_routine_transition(
        self,
        article: Article,
        claim: ExtractedEvent,
        record: dict[str, object],
        cluster_id_map: dict[str, str],
    ) -> EventState:
        """Replay cached Phase-1 decisions against the chronological Phase-2 state."""
        kind = record.get("kind")
        old_cluster_id = str(record.get("cluster_id", ""))
        if kind == "new":
            self._cluster_seq += 1
            cluster_id = f"evt_{self._cluster_seq:05d}"
            cluster_id_map[old_cluster_id] = cluster_id
            state = self.state_mgr.create_state(cluster_id, article.topic, claim)
            vector = record.get("vector_after")
            if not isinstance(vector, np.ndarray) or state.canonical_summary != record.get("summary_after"):
                self._routine_replay_reembeddings += 1
                vector = self.embedder.encode(
                    state.canonical_summary,
                    normalize_embeddings=self.config.embedding.normalize,
                )
            self.memory.add_or_update(state, vector)
            return state

        cluster_id = cluster_id_map.get(old_cluster_id)
        state = self.memory.get(cluster_id) if cluster_id else None
        result = record.get("result")
        if state is None or not isinstance(result, ClassificationResult):
            # Defensive recovery for a malformed/incomplete transition log.
            self._cluster_seq += 1
            cluster_id = f"evt_{self._cluster_seq:05d}"
            cluster_id_map[old_cluster_id] = cluster_id
            state = self.state_mgr.create_state(cluster_id, article.topic, claim)
        else:
            state = self.state_mgr.apply_action(
                state,
                claim,
                ClassificationResult(
                    action=result.action,
                    revision_type=result.revision_type,
                    rationale=result.rationale,
                ),
                resolved_summary=record.get("resolved_summary"),  # type: ignore[arg-type]
                resolved_date=record.get("resolved_date"),  # type: ignore[arg-type]
                resolved_status=record.get("resolved_status"),  # type: ignore[arg-type]
            )
        vector = record.get("vector_after")
        if not isinstance(vector, np.ndarray) or state.canonical_summary != record.get("summary_after"):
            self._routine_replay_reembeddings += 1
            vector = self.embedder.encode(
                state.canonical_summary,
                normalize_embeddings=self.config.embedding.normalize,
            )
        self.memory.add_or_update(state, vector)
        return state

    def run_stream(
        self,
        articles: list[Article],
        num_dates: Optional[int] = None,
    ) -> tuple[list[TimelineEntry], list[EventState]]:
        """Process an article collection in publication order and return timeline + final states."""
        self.memory = EventMemory(self.config.retrieval)
        self._seen_articles = []
        self._cluster_seq = 0
        self._diag_actions: dict[str, int] = {
            "NEW_EVENT": 0,
            "CORROBORATE": 0,
            "UPDATE_PROGRESS": 0,
            "REVISE_CONFLICT": 0,
        }
        self._ce_reranked_calls = 0
        self._ce_top1_flips = 0
        self._ce_time_s = 0.0
        self._rev_top1_records = []
        self._deferred_routine_log = {}
        self._forced_rev_ids = set()
        self._probe_hits = 0
        self._record_deferred_routine_log = False
        self._routine_non_top1_matches = 0
        self._routine_no_match_decisions = 0
        self._routine_replay_reembeddings = 0
        self._deferred_missing_log_count = 0
        self._retraction_quote_validations = 0
        self._retraction_quote_rejections = 0
        rho_total = 0
        rho_llm = 0
        rho_lex = 0
        ordered = sorted(articles, key=lambda a: (a.publish_date, a.article_id))
        show_pbar = len(ordered) >= 15 and os.environ.get("REVITLS_DISABLE_INNER_TQDM") != "1"
        per_article_latencies: list[float] = []
        phase2_reconciliation_s = 0.0
        if self.config.verifier_llm is not None and self.defer_phase2:
            # Algorithm 1 Two-Phase Schedule (Streaming Ingestion + Deferred Reconciliation):
            claims: list[ExtractedEvent] = []
            p1_bar = tqdm(ordered, desc="  Phase 1 (7B Extract)", unit="art", leave=False, disable=not show_pbar, dynamic_ncols=True)
            for art in p1_bar:
                t_art = time.perf_counter()
                claims.append(self.extractor.extract(art))
                per_article_latencies.append(time.perf_counter() - t_art)
            rev_queue: set[str] = set()
            self._record_deferred_routine_log = True
            p2_routine = tqdm(
                zip(ordered, claims),
                total=len(ordered),
                desc="  Phase 1b (Cluster)",
                unit="art",
                leave=False,
                disable=not show_pbar,
                dynamic_ncols=True,
            )
            for idx_art, (article, claim) in enumerate(p2_routine):
                t_step = time.perf_counter()
                is_lex = self._is_lex_revision(article, claim)
                is_llm = bool(claim.is_correction_or_update)
                if not self._is_revision_candidate(article, claim) and self._implicit_probe(article, claim):
                    self._forced_rev_ids.add(article.article_id)
                if self._is_revision_candidate(article, claim):
                    rho_total += 1
                    rho_llm += int(is_llm)
                    rho_lex += int(is_lex)
                    rev_queue.add(article.article_id)
                    # Keep the source article in the chronological evidence prefix,
                    # even though its expensive cross-date verification is deferred.
                    self._seen_articles.append(article)
                else:
                    self.step(article, pre_extracted_claim=claim, probe=False)
                per_article_latencies[idx_art] += time.perf_counter() - t_step
                p2_routine.set_postfix(clusters=len(self.memory._states), rho=rho_total, refresh=False)
            # Phase 2 replays the stream in publication order. Routine decisions
            # are cached from Phase 1; deferred revisions are re-retrieved and
            # verified against the state and evidence prefix available at that
            # point, so later routine articles cannot leak into an earlier update.
            self._record_deferred_routine_log = False
            if rev_queue:
                self.memory = EventMemory(self.config.retrieval)
                self._cluster_seq = 0
                self._seen_articles = []
                cluster_id_map: dict[str, str] = {}
                t_p2 = time.perf_counter()
                p2_rev = tqdm(
                    zip(ordered, claims),
                    total=len(ordered),
                    desc="  Phase 2 (Chronological Replay)",
                    unit="art",
                    leave=False,
                    disable=not show_pbar,
                    dynamic_ncols=True,
                )
                for article, claim in p2_rev:
                    if article.article_id in rev_queue:
                        self.step(article, pre_extracted_claim=claim)
                    else:
                        self._seen_articles.append(article)
                        transition = self._deferred_routine_log.get(article.article_id)
                        if transition is None:
                            # Fail closed: missing cached routine decisions become a
                            # fresh state instead of consulting a future-updated graph.
                            transition = {"kind": "new", "cluster_id": f"missing:{article.article_id}"}
                            self._deferred_missing_log_count += 1
                        self._replay_routine_transition(article, claim, transition, cluster_id_map)
                    p2_rev.set_postfix(
                        rev=self._diag_actions["REVISE_CONFLICT"],
                        clusters=len(self.memory._states),
                        refresh=False,
                    )
                phase2_reconciliation_s = time.perf_counter() - t_p2
        else:
            s_bar = tqdm(ordered, desc="  ReviTLS Online", unit="art", leave=False, disable=not show_pbar, dynamic_ncols=True)
            for article in s_bar:
                t_art = time.perf_counter()
                claim = self.extractor.extract(article)
                is_lex = self._is_lex_revision(article, claim)
                is_llm = bool(claim.is_correction_or_update)
                if self._is_revision_candidate(article, claim):
                    rho_total += 1
                    rho_llm += int(is_llm)
                    rho_lex += int(is_lex)
                self.step(article, pre_extracted_claim=claim)
                per_article_latencies.append(time.perf_counter() - t_art)
                s_bar.set_postfix(
                    clusters=len(self.memory._states),
                    rho=rho_total,
                    rev=self._diag_actions["REVISE_CONFLICT"],
                    refresh=False,
                )

        states = self.memory.all_states()
        if self.selection_mode == "corroboration":
            valid = [s for s in states if s.status != EventStatus.RETRACTED]
            top_k = num_dates or self.config.timeline.default_num_dates
            ranked = sorted(valid, key=lambda s: s.corroboration_count, reverse=True)[:top_k]
            ranked.sort(key=lambda s: s.event_date)
            timeline = [
                TimelineEntry(
                    event_date=s.event_date,
                    summary=s.canonical_summary,
                    status=s.status,
                    supporting_article_ids=list(s.supporting_article_ids),
                )
                for s in ranked
            ]
        else:
            timeline = self.ranker.build_timeline(states, num_dates=num_dates)

        revised_states = [s for s in states if s.status == EventStatus.REVISED]
        retracted_states = [s for s in states if s.status == EventStatus.RETRACTED]
        articles_by_id = {article.article_id: article for article in ordered}
        transition_records = []
        for state in states:
            for record in state.revision_history:
                if record.new_status not in {EventStatus.REVISED, EventStatus.RETRACTED}:
                    continue
                source_article = articles_by_id.get(record.source_article_id)
                transition_records.append(
                    {
                        "cluster_id": state.cluster_id,
                        "topic": state.topic,
                        "transition": record.new_status.value,
                        "event_date": record.new_date.isoformat() if record.new_date else "",
                        "publication_date": source_article.publish_date.isoformat() if source_article else "",
                        "source_article_id": record.source_article_id,
                        "source_title": source_article.title if source_article else "",
                        "previous_summary": record.previous_summary,
                        "new_summary": record.new_summary,
                        "rationale": record.rationale,
                    }
                )
        transition_records.sort(key=lambda item: (item["publication_date"], item["source_article_id"]))
        sorted_lat = sorted(per_article_latencies)
        mean_lat = sum(sorted_lat) / len(sorted_lat) if sorted_lat else 0.0
        p95_lat = sorted_lat[min(len(sorted_lat) - 1, int(0.95 * len(sorted_lat)))] if sorted_lat else 0.0
        max_lat = sorted_lat[-1] if sorted_lat else 0.0
        self.last_diagnostics = {
            "total_articles": len(ordered),
            "defer_phase2": self.defer_phase2,
            "deferred_replay_policy": (
                "chronological_cached_routine_actions" if self.config.verifier_llm is not None and self.defer_phase2 else "online"
            ),
            "deferred_routine_actions_reclassified_after_revision": False,
            "phase2_uses_future_articles": False,
            "routine_non_top1_matches": self._routine_non_top1_matches,
            "routine_no_match_decisions": self._routine_no_match_decisions,
            "routine_replay_reembeddings": self._routine_replay_reembeddings,
            "deferred_missing_routine_log_entries": self._deferred_missing_log_count,
            "retraction_quotes_validated": self._retraction_quote_validations,
            "retraction_labels_rejected_without_verbatim_evidence": self._retraction_quote_rejections,
            "mean_article_latency_s": round(mean_lat, 4),
            "p95_article_latency_s": round(p95_lat, 4),
            "max_article_latency_s": round(max_lat, 4),
            "phase2_reconciliation_s": round(phase2_reconciliation_s, 4),
            "ce_reranked_calls": self._ce_reranked_calls,
            "ce_top1_flips": self._ce_top1_flips,
            "ce_time_s": round(self._ce_time_s, 4),
            "rev_top1_records": list(self._rev_top1_records),
            "rho_triggered": rho_total,
            "rho_struct_probe_hits": self._probe_hits,
            "rho_llm_flag": rho_llm,
            "rho_lexicon_flag": rho_lex,
            "actions": dict(self._diag_actions),
            "total_states": len(states),
            "revised_count": len(revised_states),
            "retracted_count": len(retracted_states),
            "transition_records": transition_records,
            "sample_revised": [
                {
                    "date": s.event_date.isoformat() if s.event_date else "",
                    "summary": s.canonical_summary[:180],
                    "purged": [r.previous_summary for r in s.revision_history][:2],
                }
                for s in revised_states[:5]
            ],
            "sample_retracted": [
                {"date": s.event_date.isoformat() if s.event_date else "", "summary": s.canonical_summary[:180]}
                for s in retracted_states[:5]
            ],
        }
        return timeline, states
