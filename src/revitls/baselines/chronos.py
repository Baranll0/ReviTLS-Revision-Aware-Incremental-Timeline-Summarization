"""Faithful closed-domain CHRONOS baseline (Wu et al., NAACL 2025) on shared backends."""

from datetime import date
import os
import numpy as np
from tqdm.auto import tqdm

from revitls.protocols import EmbedderProtocol, LLMProtocol
from revitls.schemas import Article, EventState, EventStatus, TimelineEntry


class ClosedDomainCHRONOSBaseline:
    """Reproduces CHRONOS closed-domain RAG: context survey -> iterative 5W1H questioning -> merge."""

    def __init__(
        self,
        llm: LLMProtocol,
        embedder: EmbedderProtocol,
        rounds: int = 2,
        top_n: int = 15,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.rounds = rounds
        self.top_n = top_n

    def _retrieve(
        self,
        query: str,
        articles: list[Article],
        art_vecs: np.ndarray,
    ) -> list[Article]:
        q_vec = self.embedder.encode(query, normalize_embeddings=True)
        sims = art_vecs @ q_vec
        top_idx = np.argsort(sims)[::-1][: self.top_n]
        return [articles[int(i)] for i in top_idx]

    def _call_json(self, system: str, user: str, default: dict) -> dict:
        if hasattr(self.llm, "complete_json_safe"):
            return self.llm.complete_json_safe(system, user, default)
        return self.llm.complete_json(system, user)

    def _generate_questions(self, topic: str, docs: list[Article], asked: list[str]) -> list[str]:
        context = "\n".join(f"[{a.publish_date}] {a.title}: {a.content[:250]}" for a in docs[:10])
        data = self._call_json(
            "Propose non-overlapping factual 5W1H questions not yet answered by the news context. "
            "Return JSON: {\"questions\": [\"...\"]}.",
            f"Topic: {topic}\nAsked: {asked}\nContext:\n{context}",
            {"questions": []},
        )
        return [str(q).strip() for q in data.get("questions", []) if str(q).strip()][:5]

    def _rewrite_query(self, question: str) -> list[str]:
        data = self._call_json(
            "Decompose the question into 2-3 focused search queries. Return JSON: {\"queries\": [\"...\"]}.",
            f"Question: {question}",
            {"queries": [question]},
        )
        return [str(q).strip() for q in data.get("queries", []) if str(q).strip()][:3]

    def _round_timeline(self, topic: str, docs: list[Article]) -> list[dict[str, str]]:
        doc_block = "\n".join(f"[{a.publish_date}] {a.title}: {a.content[:350]}" for a in docs)
        data = self._call_json(
            "Extract significant milestone events with dates. Return JSON: "
            "{\"events\": [{\"start\": \"YYYY-MM-DD\", \"summary\": \"...\"}]}.",
            f"Topic: {topic}\nArticles:\n{doc_block}",
            {"events": []},
        )
        events = data.get("events", [])
        return list(events) if isinstance(events, list) else []

    def _merge_timelines(
        self, topic: str, round_events: list[dict[str, str]], num_dates: int
    ) -> list[dict[str, str]]:
        # Follows Table 12 prompt in CHRONOS: preserve original dates, exclude later duplicates
        data = self._call_json(
            f"Merge the timelines chronologically, select top-{num_dates} events without changing "
            "the original date, and exclude duplicate events appearing in later dates. "
            "Return JSON: {\"events\": [{\"start\": \"YYYY-MM-DD\", \"summary\": \"...\"}]}.",
            f"Topic: {topic}\nEvents:\n{round_events}",
            {"events": round_events[:num_dates]},
        )
        events = data.get("events", [])
        return list(events) if isinstance(events, list) else round_events[:num_dates]

    def run_stream(
        self,
        articles: list[Article],
        num_dates: int = 20,
    ) -> tuple[list[TimelineEntry], list[EventState]]:
        """Run CHRONOS in closed-domain mode over the full topic corpus."""
        if not articles:
            return [], []

        topic = articles[0].topic
        texts = [f"{a.title} {a.content[:500]}" for a in articles]
        art_vecs = np.vstack([self.embedder.encode(t, normalize_embeddings=True) for t in texts])

        # Step 1: Context survey
        docs = self._retrieve(topic, articles, art_vecs)
        all_round_events: list[dict[str, str]] = self._round_timeline(topic, docs)
        asked: list[str] = []

        # Step 2: Iterative self-questioning rounds
        for _ in tqdm(range(self.rounds), desc="  CHRONOS Rounds", unit="rnd", leave=False, disable=len(articles) < 15 or os.environ.get("REVITLS_DISABLE_INNER_TQDM") == "1", dynamic_ncols=True):
            questions = self._generate_questions(topic, docs, asked)
            asked.extend(questions)
            round_docs: list[Article] = []
            seen_ids: set[str] = set()
            for q in questions:
                for rq in self._rewrite_query(q) or [q]:
                    for hit in self._retrieve(rq, articles, art_vecs):
                        if hit.article_id not in seen_ids:
                            seen_ids.add(hit.article_id)
                            round_docs.append(hit)
            if round_docs:
                docs = round_docs
                all_round_events.extend(self._round_timeline(topic, round_docs))

        merged = self._merge_timelines(topic, all_round_events, num_dates)

        timeline: list[TimelineEntry] = []
        states: list[EventState] = []
        for idx, ev in enumerate(merged, start=1):
            try:
                evt_date = date.fromisoformat(str(ev.get("start", ""))[:10])
            except ValueError:
                continue
            summary = str(ev.get("summary", "")).strip()
            timeline.append(TimelineEntry(event_date=evt_date, summary=summary))
            states.append(
                EventState(
                    cluster_id=f"chronos_{idx:05d}",
                    topic=topic,
                    event_date=evt_date,
                    canonical_summary=summary,
                    status=EventStatus.ACTIVE,
                    last_updated=evt_date,
                )
            )
        timeline.sort(key=lambda t: t.event_date)
        return timeline, states
