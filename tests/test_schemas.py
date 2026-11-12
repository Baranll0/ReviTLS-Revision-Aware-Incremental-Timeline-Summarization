"""Unit tests for ReviTLS state transitions, cross-date memory, and revision metrics."""

from datetime import date
import numpy as np

from revitls.config import RetrievalConfig, TimelineConfig
from revitls.evaluation.revision_metrics import RevisionTestCase, evaluate_revision_states
from revitls.evaluation.tls_metrics import (
    compute_align_rouge,
    compute_date_f1,
    filter_articles_by_timespan,
)
from revitls.llm.client import _extract_json_object
from revitls.pipeline.classifier import ClassificationResult
from revitls.pipeline.ranker import TimelineRanker
from revitls.pipeline.state_manager import EventStateManager
from revitls.retrieval.memory import EventMemory
from revitls.schemas import Article, EventStatus, ExtractedEvent, RevisionAction, RevisionType


def test_cross_date_revision_and_retraction_filtering() -> None:
    mgr = EventStateManager()
    memory = EventMemory(RetrievalConfig(allow_cross_date=True, similarity_threshold=0.5))

    initial_claim = ExtractedEvent(
        source_article_id="art_1",
        publish_date=date(2023, 3, 10),
        event_date=date(2023, 3, 10),
        summary="Initial reports claim 5 people were injured in the factory fire.",
        entities=["Factory Fire"],
        key_facts={"injured": "5"},
    )
    state = mgr.create_state("evt_00001", "crisis_demo", initial_claim)
    vec = np.array([1.0, 0.0, 0.0])
    memory.add_or_update(state, vec)

    # Late correction published 4 days later must still retrieve evt_00001
    hits = memory.search(np.array([0.98, 0.05, 0.0]), reference_date=date(2023, 3, 14))
    assert len(hits) == 1
    assert hits[0][0].cluster_id == "evt_00001"

    correction_claim = ExtractedEvent(
        source_article_id="art_2",
        publish_date=date(2023, 3, 14),
        event_date=date(2023, 3, 10),
        summary="Officials revised the factory fire injury count to 14.",
        entities=["Factory Fire"],
        key_facts={"injured": "14"},
        is_correction_or_update=True,
    )
    updated = mgr.apply_action(
        state,
        correction_claim,
        ClassificationResult(
            action=RevisionAction.REVISE_CONFLICT,
            revision_type=RevisionType.NUMERIC_FACT,
            rationale="Official count increased from 5 to 14.",
        ),
        resolved_summary="Officials confirmed 14 people were injured in the factory fire.",
        resolved_status=EventStatus.REVISED,
    )
    assert updated.status == EventStatus.REVISED
    assert len(updated.revision_history) == 1

    # Add a second event that gets retracted later
    rumor = ExtractedEvent(
        source_article_id="art_3",
        publish_date=date(2023, 3, 11),
        event_date=date(2023, 3, 11),
        summary="A second nearby plant reportedly exploded.",
        entities=["Nearby Plant"],
    )
    rumor_state = mgr.create_state("evt_00002", "crisis_demo", rumor)
    mgr.apply_action(
        rumor_state,
        ExtractedEvent(
            source_article_id="art_4",
            publish_date=date(2023, 3, 12),
            event_date=date(2023, 3, 11),
            summary="Authorities debunked reports of a second plant explosion.",
            entities=["Nearby Plant"],
        ),
        ClassificationResult(
            action=RevisionAction.REVISE_CONFLICT,
            revision_type=RevisionType.RETRACTION,
            rationale="Debunked by authorities.",
        ),
        resolved_summary="Authorities debunked reports of a second plant explosion.",
        resolved_status=EventStatus.RETRACTED,
    )

    ranker = TimelineRanker(TimelineConfig(default_num_dates=5, exclude_retracted=True))
    timeline = ranker.build_timeline([updated, rumor_state])
    assert len(timeline) == 1
    assert timeline[0].event_date == date(2023, 3, 10)
    assert "14" in timeline[0].summary

    rev_scores = evaluate_revision_states(
        [updated, rumor_state],
        [
            RevisionTestCase(
                case_id="c1",
                expected_final_fact="14 people were injured",
                stale_fact_substring="5 people were injured",
            ),
            RevisionTestCase(
                case_id="c2",
                expected_final_fact="",
                stale_fact_substring="second nearby plant reportedly exploded",
                should_be_retracted=True,
            ),
        ],
    )
    assert rev_scores["revision_resolution_accuracy"] == 1.0
    assert rev_scores["stale_fact_leakage_rate"] == 0.0

    date_scores = compute_date_f1(timeline, [date(2023, 3, 10)])
    assert date_scores["date_f1"] == 1.0

    ref_tl = [(date(2023, 3, 10), ["Officials confirmed 14 people were injured in the factory fire."])]
    align_scores = compute_align_rouge(timeline, ref_tl)
    assert align_scores["align_r1"] > 0.8


def test_json_fence_extraction_and_timespan_filter() -> None:
    parsed = _extract_json_object('```json\n{"action": "REVISE_CONFLICT"}\n```')
    assert parsed["action"] == "REVISE_CONFLICT"

    arts = [
        Article(article_id="1", topic="t", title="a", content="x", publish_date=date(2023, 3, 1)),
        Article(article_id="2", topic="t", title="b", content="y", publish_date=date(2023, 3, 10)),
        Article(article_id="3", topic="t", title="c", content="z", publish_date=date(2023, 4, 1)),
    ]
    filtered = filter_articles_by_timespan(arts, [(date(2023, 3, 5), ["ev"]), (date(2023, 3, 15), ["ev2"])])
    assert [a.article_id for a in filtered] == ["2"]
