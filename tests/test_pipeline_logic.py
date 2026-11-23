"""Logic tests for ranking (Eq. 6), rho modes, the retraction quote gate, and chronological replay."""

import math
import re
from datetime import date
from typing import Any

import numpy as np
import pytest

from revitls.config import ReviTLSConfig, TimelineConfig, VerifierConfig
from revitls.pipeline.classifier import ClassificationResult
from revitls.pipeline.ranker import TimelineRanker, filled_slot_count, salience
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.pipeline.state_manager import EventStateManager
from revitls.pipeline.verifier import TargetedVerifier
from revitls.schemas import Article, EventState, EventStatus, ExtractedEvent, RevisionAction


def _st(cid: str, d: date, c: int, status=EventStatus.ACTIVE, slots=None, entities=None) -> EventState:
    return EventState(
        cluster_id=cid, topic="t", event_date=d, canonical_summary="x y z", status=status,
        entities=entities or [], slots=slots or {}, corroboration_count=c, last_updated=d,
        supporting_article_ids=[cid],
    )


class _HashEmb:
    def encode(self, sentences: Any, normalize_embeddings: bool = True) -> np.ndarray:
        def v(t: str) -> np.ndarray:
            x = np.zeros(64)
            for w in re.findall(r"[a-z]+", t.lower()):
                x[sum(map(ord, w)) % 64] += 1
            n = np.linalg.norm(x)
            return x / n if n else x

        return v(sentences) if isinstance(sentences, str) else np.vstack([v(s) for s in sentences])


class _Fixed:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def complete(self, s: str, u: str) -> str:
        return ""

    def complete_json(self, s: str, u: str) -> dict[str, Any]:
        return dict(self.payload)


class _Scripted:
    """Extracts the title as summary; articles whose title has 'Correction' REVISE their match."""

    def complete(self, s: str, u: str) -> str:
        return ""

    def complete_json_safe(self, s: str, u: str, default: dict[str, Any]) -> dict[str, Any]:
        return self.complete_json(s, u) or default

    def complete_json(self, s: str, u: str) -> dict[str, Any]:
        if "Extract the primary event claim" in s:
            title = re.search(r"Title:\s*(.+)", u).group(1).strip()
            pub = re.search(r"Publish Date:\s*(\S+)", u).group(1)
            return {
                "event_date": pub, "summary": title, "entities": ["Plant"], "key_facts": {},
                "is_correction_or_update": "Correction" in title,
            }
        if "Classify the relation" in s:
            if "candidate_index" in u:
                return {"candidate_index": -1, "action": "NEW_EVENT"}
            inc = u.split("Incoming Summary:")[-1].splitlines()[0]
            return {
                "action": "REVISE_CONFLICT" if "Correction" in inc else "NEW_EVENT",
                "revision_type": "NUMERIC_FACT", "rationale": "",
            }
        if "Generate up to" in u:
            return {"queries": ["plant"]}
        if "most accurate current truth" in u:
            return {"resolved_summary": "Plant figure is 9.", "resolved_date": None, "status": "REVISED", "retraction_evidence": None}
        return {}


def test_salience_matches_equation() -> None:
    cfg = TimelineConfig()
    s = _st("a", date(2020, 1, 1), 3, EventStatus.REVISED, {"where": "w", "why": "y"}, ["E"])
    assert filled_slot_count(s) == 5  # who(entities), what, when, where, why
    assert salience(s, cfg) == pytest.approx(0.55 * math.log(4) + 0.15 + 0.30 / 6 * 5)


def test_ranker_dedups_dates_strictly_and_excludes_retracted() -> None:
    cfg = TimelineConfig(default_num_dates=5, deduplicate_dates=True)
    states = [
        _st("a", date(2020, 1, 1), 5),
        _st("b", date(2020, 1, 1), 2),
        _st("c", date(2020, 1, 2), 1, EventStatus.RETRACTED),
        _st("d", date(2020, 1, 3), 1),
    ]
    tl = TimelineRanker(cfg).build_timeline(states)
    assert [e.event_date for e in tl] == [date(2020, 1, 1), date(2020, 1, 3)]


def test_rho_mode_validation_and_aliases() -> None:
    with pytest.raises(ValueError):
        ReviTLSPipeline(ReviTLSConfig(), llm=_Scripted(), embedder=_HashEmb(), rho_mode="llm_onyl")
    p = ReviTLSPipeline(ReviTLSConfig(), llm=_Scripted(), embedder=_HashEmb(), rho_mode="llm_only")
    assert p.rho_mode == "llm"


def test_state_manager_retracts_only_on_verified_status() -> None:
    mgr = EventStateManager()
    ev = ExtractedEvent(
        source_article_id="r", publish_date=date(2020, 1, 5), event_date=date(2020, 1, 1),
        summary="Officials debunked the rumor and said no incident occurred.",
    )
    base = ExtractedEvent(source_article_id="a", publish_date=date(2020, 1, 1), event_date=date(2020, 1, 1), summary="Rumor of incident.")
    cls = ClassificationResult(action=RevisionAction.REVISE_CONFLICT, rationale="")
    s1 = mgr.apply_action(mgr.create_state("e1", "t", base), ev, cls, resolved_summary=ev.summary, resolved_status=EventStatus.REVISED)
    assert s1.status == EventStatus.REVISED  # lexical negation alone must not retract
    s2 = mgr.apply_action(mgr.create_state("e2", "t", base), ev, cls, resolved_summary=ev.summary, resolved_status=EventStatus.RETRACTED)
    assert s2.status == EventStatus.RETRACTED


def test_retraction_without_verbatim_quote_is_downgraded() -> None:
    evidence = [Article(article_id="e", topic="t", title="Title", content="Officials confirmed that no leak occurred at the plant.", publish_date=date(2020, 1, 2))]
    st = _st("a", date(2020, 1, 1), 1)
    inc = ExtractedEvent(source_article_id="r", publish_date=date(2020, 1, 2), summary="s")
    bad = TargetedVerifier(_Fixed({"resolved_summary": "No event.", "status": "RETRACTED", "retraction_evidence": "made up quote"}), VerifierConfig())
    assert bad.resolve_with_evidence(st, inc, evidence)["status"] == "REVISED"
    good = TargetedVerifier(_Fixed({"resolved_summary": "No event.", "status": "RETRACTED", "retraction_evidence": "no leak occurred at the plant"}), VerifierConfig())
    assert good.resolve_with_evidence(st, inc, evidence)["status"] == "RETRACTED"


def test_phase2_replay_never_shows_future_articles_as_evidence() -> None:
    cfg = ReviTLSConfig()
    cfg.verifier_llm = cfg.llm
    cfg.retrieval.similarity_threshold = 0.0
    cfg.retrieval.strict_temporal_gate = True
    cfg.retrieval.cross_encoder_model = None
    cfg.embedding.device = "cpu"
    pipe = ReviTLSPipeline(cfg, llm=_Scripted(), embedder=_HashEmb(), verifier_llm=_Scripted())

    evidence_ids: list[list[str]] = []
    real = pipe.verifier.resolve_with_evidence

    def spy(state, incoming, evidence):  # type: ignore[no-untyped-def]
        evidence_ids.append([a.article_id for a in evidence])
        return real(state, incoming, evidence)

    pipe.verifier.resolve_with_evidence = spy  # type: ignore[method-assign]
    arts = [
        Article(article_id="a1", topic="t", title="Plant figure is 5", content="Plant figure is 5.", publish_date=date(2020, 1, 1)),
        Article(article_id="a2", topic="t", title="Correction: plant figure", content="Correction plant figure.", publish_date=date(2020, 1, 3)),
        Article(article_id="a3", topic="t", title="Plant later news", content="Later plant news.", publish_date=date(2020, 1, 6)),
        Article(article_id="a4", topic="t", title="Plant much later news", content="Plant much later news.", publish_date=date(2020, 1, 7)),
    ]
    pipe.run_stream(arts, num_dates=10)
    assert evidence_ids, "the correction article should have triggered verification"
    for ids in evidence_ids:
        assert "a3" not in ids and "a4" not in ids


def test_measure_and_date_conflict_signals() -> None:
    from revitls.pipeline.cues import has_date_conflict, has_measure_conflict

    assert has_measure_conflict("3,400 passengers were stranded at the port", "579 passengers were stranded at the port")
    assert not has_measure_conflict("579 passengers stranded; 3 shelters opened", "579 passengers were stranded at the port")
    assert not has_measure_conflict("Held on 5 March 2015", "Held on 7 March 2015")  # years are ignored
    assert has_date_conflict(date(2020, 1, 3), date(2020, 1, 1), 7)
    assert not has_date_conflict(date(2020, 1, 1), date(2020, 1, 1), 7)
    assert not has_date_conflict(date(2020, 3, 1), date(2020, 1, 1), 7)


class _ScriptedImplicit(_Scripted):
    def complete_json(self, s: str, u: str) -> dict[str, Any]:
        if "Classify the relation" in s and "candidate_index" not in u:
            inc = u.split("Incoming Summary:")[-1].splitlines()[0]
            return {"action": "REVISE_CONFLICT" if "now says" in inc else "NEW_EVENT", "revision_type": "NUMERIC_FACT", "rationale": ""}
        return super().complete_json(s, u)


def _implicit_run(probe_on: bool) -> int:
    cfg = ReviTLSConfig()
    cfg.verifier_llm = cfg.llm
    cfg.retrieval.similarity_threshold = 0.0
    cfg.retrieval.strict_temporal_gate = True
    cfg.retrieval.cross_encoder_model = None
    cfg.retrieval.implicit_probe = probe_on
    cfg.retrieval.implicit_probe_sim = 0.3
    cfg.embedding.device = "cpu"
    pipe = ReviTLSPipeline(cfg, llm=_ScriptedImplicit(), embedder=_HashEmb(), verifier_llm=_ScriptedImplicit())
    arts = [
        Article(article_id="a1", topic="t", title="Plant report: 579 passengers stranded", content="x", publish_date=date(2020, 1, 1)),
        Article(article_id="a2", topic="t", title="Plant report now says 3400 passengers stranded", content="x", publish_date=date(2020, 1, 20)),
    ]
    pipe.run_stream(arts, num_dates=10)
    return int(pipe.last_diagnostics["actions"]["REVISE_CONFLICT"])  # type: ignore[index]


def test_structural_probe_catches_cue_free_numeric_correction() -> None:
    assert _implicit_run(probe_on=False) == 0  # no cue, routine same-day gate: never linked
    assert _implicit_run(probe_on=True) == 1
