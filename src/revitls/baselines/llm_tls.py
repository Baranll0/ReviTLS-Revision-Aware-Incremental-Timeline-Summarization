"""Faithful Topic LLM-TLS baseline (Hu et al., ACL 2024) on shared LLM & embedder backends."""

from collections import defaultdict
from datetime import date
from typing import Optional
import numpy as np

from revitls.protocols import EmbedderProtocol, LLMProtocol
from revitls.schemas import Article, EventState, EventStatus, TimelineEntry


def _textrank_top_sentence(summaries: list[str], vectors: np.ndarray) -> str:
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
    return summaries[int(np.argmax(scores))]


class TopicLLMTLSBaseline:
    """Reproduces Topic LLM-TLS: keyword event summary -> same-date binary clustering -> TextRank."""

    def __init__(self, llm: LLMProtocol, embedder: EmbedderProtocol, top_n: int = 20) -> None:
        self.llm = llm
        self.embedder = embedder
        self.top_n = top_n

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

            # Strict same-date heuristic (Algorithm 2 in Hu et al., 2024)
            same_day_nodes = nodes_by_date.get(evt_date, [])
            assigned_cluster: Optional[int] = None

            if same_day_nodes:
                scored = sorted(
                    (
                        (float(np.dot(vec, nvec)), cid, nsum)
                        for cid, nsum, nvec, _ in same_day_nodes
                    ),
                    key=lambda x: x[0],
                    reverse=True,
                )[: self.top_n]
                for _sim, cid, nsum in scored:
                    if self._is_same_event(art.topic, evt_date.isoformat(), summary, evt_date.isoformat(), nsum):
                        assigned_cluster = cid
                        break

            if assigned_cluster is None:
                assigned_cluster = len(clusters)
                clusters.append([])

            clusters[assigned_cluster].append((art.article_id, evt_date, summary, vec))
            nodes_by_date[evt_date].append((assigned_cluster, summary, vec, art.article_id))

        # Rank clusters by node count and summarize via TextRank
        ranked = sorted(clusters, key=len, reverse=True)[:num_dates]
        ranked.sort(key=lambda c: c[0][1])

        timeline: list[TimelineEntry] = []
        states: list[EventState] = []
        for idx, cluster in enumerate(ranked, start=1):
            sums = [item[2] for item in cluster]
            vecs = np.vstack([item[3] for item in cluster])
            rep = _textrank_top_sentence(sums, vecs)
            evt_date = cluster[0][1]
            art_ids = [item[0] for item in cluster]
            timeline.append(
                TimelineEntry(
                    event_date=evt_date,
                    summary=rep,
                    status=EventStatus.ACTIVE,
                    supporting_article_ids=art_ids,
                )
            )
            states.append(
                EventState(
                    cluster_id=f"llmtls_{idx:05d}",
                    topic=articles[0].topic if articles else "",
                    event_date=evt_date,
                    canonical_summary=rep,
                    status=EventStatus.ACTIVE,
                    supporting_article_ids=art_ids,
                    corroboration_count=len(cluster),
                    last_updated=evt_date,
                )
            )
        return timeline, states
