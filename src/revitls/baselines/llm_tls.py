"""Faithful Topic LLM-TLS baseline (Hu et al., ACL 2024) on shared LLM & embedder backends."""

from collections import defaultdict
from datetime import date, timedelta
from typing import Optional
import numpy as np

from revitls.protocols import EmbedderProtocol, LLMProtocol
from revitls.schemas import Article, EventState, EventStatus, TimelineEntry


def _textrank_top_sentence(
    summaries: list[str],
    vectors: np.ndarray,
    recency_bias: float = 0.0,
) -> str:
    """Select the single most central summary in a cluster via PageRank/TextRank."""
    n = len(summaries)
    if n == 1:
        return summaries[0]

    sim = np.clip(vectors @ vectors.T, 0.0, None)
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    transition = np.divide(sim, row_sums, out=np.full_like(sim, 1.0 / n), where=row_sums > 0)

    scores = np.full(n, 1.0 / n, dtype=float)
    damping = 0.85
    for _ in range(25):
        scores = (1.0 - damping) / n + damping * (transition.T @ scores)

    if recency_bias > 0.0:
        # Linearly reward later items in chronological cluster order
        recency = np.linspace(0.0, 1.0, n, dtype=float)
        scores = (1.0 - recency_bias) * scores + recency_bias * recency * scores.max()

    return summaries[int(np.argmax(scores))]


class TopicLLMTLSBaseline:
    """Reproduces Topic LLM-TLS and simple revision-aware baseline variants."""

    def __init__(
        self,
        llm: LLMProtocol,
        embedder: EmbedderProtocol,
        top_n: int = 20,
        window_days: int = 0,
        selection_mode: str = "textrank",
        post_hoc_verifier_llm: Optional[LLMProtocol] = None,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.top_n = top_n
        self.window_days = window_days
        self.selection_mode = selection_mode
        self.post_hoc_verifier_llm = post_hoc_verifier_llm

    def _summarize_article(self, article: Article) -> tuple[date, str]:
        system = (
            "Review the news article and summarize the most significant event related to the topic "
            "as JSON: {\"event_date\": \"YYYY-MM-DD\", \"summary\": \"One-sentence summary\"}."
        )
        user = (
            f"Topic: {article.topic}\n"
            f"Publish Date: {article.publish_date.isoformat()}\n"
            f"Title: {article.title}\n"
            f"Content:\n{article.content[:3000]}"
        )
        default_sum = {"event_date": article.publish_date.isoformat(), "summary": article.title}
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(system, user, default_sum)
        else:
            data = self.llm.complete_json(system, user)
        evt_date = article.publish_date
        if data.get("event_date"):
            try:
                evt_date = date.fromisoformat(str(data["event_date"])[:10])
            except ValueError:
                pass
        return evt_date, str(data.get("summary", article.title)).strip()

    def _is_same_event(self, topic: str, d1: str, s1: str, d2: str, s2: str) -> bool:
        system = (
            "Evaluate whether two news event summaries refer to the same event. "
            "Return JSON: {\"same_event\": true/false}."
        )
        user = f"Topic: {topic}\nEvent 1 ({d1}): {s1}\nEvent 2 ({d2}): {s2}"
        if hasattr(self.llm, "complete_json_safe"):
            data = self.llm.complete_json_safe(system, user, {"same_event": False})
        else:
            data = self.llm.complete_json(system, user)
        return bool(data.get("same_event", False))

    def run_stream(
        self,
        articles: list[Article],
        num_dates: int = 20,
    ) -> tuple[list[TimelineEntry], list[EventState]]:
        """Run Topic LLM-TLS incrementally and return TimelineEntry + EventState views."""
        # Each cluster stores list of (article_id, event_date, summary, vector)
        clusters: list[list[tuple[str, date, str, np.ndarray]]] = []
        nodes_by_date: dict[date, list[tuple[int, str, np.ndarray, str]]] = defaultdict(list)

        for art in sorted(articles, key=lambda a: (a.publish_date, a.article_id)):
            evt_date, summary = self._summarize_article(art)
            vec = self.embedder.encode(summary, normalize_embeddings=True)

            # Candidate lookup: strict same-day (window_days=0) or lookback window
            candidate_nodes: list[tuple[int, str, np.ndarray, str, date]] = []
            if self.window_days <= 0:
                for cid, nsum, nvec, aid in nodes_by_date.get(evt_date, []):
                    candidate_nodes.append((cid, nsum, nvec, aid, evt_date))
            else:
                min_d = evt_date - timedelta(days=self.window_days)
                for d_key, items in nodes_by_date.items():
                    if min_d <= d_key <= evt_date:
                        for cid, nsum, nvec, aid in items:
                            candidate_nodes.append((cid, nsum, nvec, aid, d_key))

            assigned_cluster: Optional[int] = None

            if candidate_nodes:
                best_per_cluster: dict[int, tuple[float, int, str, date]] = {}
                for cid, nsum, nvec, _aid, ndate in candidate_nodes:
                    sim = float(np.dot(vec, nvec))
                    if sim >= 0.45 and (cid not in best_per_cluster or sim > best_per_cluster[cid][0]):
                        best_per_cluster[cid] = (sim, cid, nsum, ndate)
                scored = sorted(best_per_cluster.values(), key=lambda x: x[0], reverse=True)[:2]
                for _sim, cid, nsum, ndate in scored:
                    if self._is_same_event(art.topic, evt_date.isoformat(), summary, ndate.isoformat(), nsum):
                        assigned_cluster = cid
                        break

            if assigned_cluster is None:
                assigned_cluster = len(clusters)
                clusters.append([])

            clusters[assigned_cluster].append((art.article_id, evt_date, summary, vec))
            nodes_by_date[evt_date].append((assigned_cluster, summary, vec, art.article_id))

        # Rank clusters by node count and summarize
        ranked = sorted(clusters, key=len, reverse=True)[:num_dates]
        ranked.sort(key=lambda c: c[0][1])

        timeline: list[TimelineEntry] = []
        states: list[EventState] = []
        for idx, cluster in enumerate(ranked, start=1):
            sums = [item[2] for item in cluster]
            vecs = np.vstack([item[3] for item in cluster])
            if self.selection_mode == "latest":
                rep = sums[-1]
            elif self.selection_mode == "recency_textrank":
                rep = _textrank_top_sentence(sums, vecs, recency_bias=0.5)
            else:
                rep = _textrank_top_sentence(sums, vecs)
            evt_date = cluster[0][1]
            art_ids = [item[0] for item in cluster]
            status = EventStatus.REVISED if (self.window_days > 0 and len(cluster) > 1 and sums[-1] != sums[0]) else EventStatus.ACTIVE
            timeline.append(
                TimelineEntry(
                    event_date=evt_date,
                    summary=rep,
                    status=status,
                    supporting_article_ids=art_ids,
                )
            )
            states.append(
                EventState(
                    cluster_id=f"llmtls_{idx:05d}",
                    topic=articles[0].topic if articles else "",
                    event_date=evt_date,
                    canonical_summary=rep,
                    status=status,
                    supporting_article_ids=art_ids,
                    corroboration_count=len(cluster),
                    last_updated=cluster[-1][1],
                )
            )

        if self.post_hoc_verifier_llm is not None and timeline:
            timeline, states = self._apply_post_hoc_check(
                articles=articles,
                timeline=timeline,
                states=states,
                num_dates=num_dates,
            )
        return timeline, states

    def _apply_post_hoc_check(
        self,
        articles: list[Article],
        timeline: list[TimelineEntry],
        states: list[EventState],
        num_dates: int,
    ) -> tuple[list[TimelineEntry], list[EventState]]:
        """Post-hoc consistency pass over LLM-TLS timeline against later corrective articles."""
        assert self.post_hoc_verifier_llm is not None
        tl_block = "\n".join(
            f"[{i}] ({e.event_date.isoformat()}) {e.summary}" for i, e in enumerate(timeline)
        )
        # Provide chronologically last articles as potential late corrections/retractions
        sorted_arts = sorted(articles, key=lambda a: (a.publish_date, a.article_id))
        tail_arts = sorted_arts[-min(25, len(sorted_arts)) :]
        art_block = "\n".join(
            f"- ({a.publish_date.isoformat()}) {a.title}: {a.content[:400]}" for a in tail_arts
        )
        system = (
            "Review the candidate timeline against later news reports. Identify any entry that was "
            "later revised (update summary) or debunked/retracted (set status='RETRACTED'). "
            "Return JSON: {\"updates\": [{\"index\": 0, \"status\": \"ACTIVE|REVISED|RETRACTED\", \"summary\": \"...\"}]}."
        )
        user = f"Timeline:\n{tl_block}\n\nLater Reports:\n{art_block}"
        if hasattr(self.post_hoc_verifier_llm, "complete_json_safe"):
            data = self.post_hoc_verifier_llm.complete_json_safe(system, user, {"updates": []})
        else:
            data = self.post_hoc_verifier_llm.complete_json(system, user)

        updates = data.get("updates", [])
        if not isinstance(updates, list):
            return timeline, states

        by_idx: dict[int, dict] = {}
        for upd in updates:
            if isinstance(upd, dict) and isinstance(upd.get("index"), int):
                by_idx[upd["index"]] = upd

        new_tl: list[TimelineEntry] = []
        new_states: list[EventState] = []
        for idx, (entry, st) in enumerate(zip(timeline, states)):
            upd = by_idx.get(idx)
            if upd:
                st_str = str(upd.get("status", "ACTIVE")).upper()
                new_sum = str(upd.get("summary") or entry.summary).strip()
                if st_str == "RETRACTED":
                    st.status = EventStatus.RETRACTED
                    new_states.append(st)
                    continue
                if st_str == "REVISED":
                    st.status = EventStatus.REVISED
                    st.canonical_summary = new_sum
                    entry.status = EventStatus.REVISED
                    entry.summary = new_sum
            new_tl.append(entry)
            new_states.append(st)
        return new_tl[:num_dates], new_states
