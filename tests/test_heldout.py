"""Sanity tests for the held-out suite generator and its evaluator (no model needed)."""

from datetime import date

from revitls.evaluation.revision_metrics import evaluate_cases, evaluate_revision_states
from revitls.evaluation.stats import exact_mcnemar_p, holm_adjust, wilson_ci
from revitls.pipeline.cues import has_revision_cue
from revitls.revisions.heldout import HeldoutSpec, generate_heldout_suite
from revitls.schemas import EventState, EventStatus


def _state(case, articles, summary: str, status: EventStatus, event_date=date(2015, 1, 5)) -> EventState:
    ids = [a.article_id for a in articles if a.article_id == case.case_id or a.article_id.startswith(case.case_id + "_")]
    return EventState(
        cluster_id=case.case_id, topic="t", event_date=event_date, canonical_summary=summary,
        status=status, supporting_article_ids=ids, last_updated=event_date,
    )


def test_generator_is_deterministic_and_balanced() -> None:
    a1, c1 = generate_heldout_suite()
    a2, c2 = generate_heldout_suite()
    assert [x.article_id for x in a1] == [x.article_id for x in a2]
    assert len(c1) == 110
    assert sum(c.is_negative for c in c1) == 30
    ids = [a.article_id for a in a1]
    assert len(ids) == len(set(ids))
    for c in c1:
        if c.expected_final_fact and c.stale_fact_substring:
            assert c.stale_fact_substring.lower() not in c.expected_final_fact.lower()
            assert c.expected_final_fact.lower() not in c.stale_fact_substring.lower()


def test_implicit_corrections_contain_no_lexicon_cue() -> None:
    articles, cases = generate_heldout_suite()
    implicit = {c.case_id for c in cases if c.cue == "implicit"}
    for a in articles:
        if a.article_id.endswith("_rev") and a.article_id[: -len("_rev")] in implicit:
            assert not has_revision_cue(f"{a.title} {a.content}"), a.content


def test_evaluator_oracle_stale_and_overretract() -> None:
    articles, cases = generate_heldout_suite(HeldoutSpec(n_per_cell=2, n_neg_denial=3, n_neg_progress=3))

    oracle, stale, overretract = [], [], []
    for c in cases:
        if c.kind == "retraction":
            oracle.append(_state(c, articles, "No such event occurred.", EventStatus.RETRACTED))
            stale.append(_state(c, articles, f"Reports of {c.stale_fact_substring}.", EventStatus.ACTIVE))
        elif c.is_negative:
            oracle.append(_state(c, articles, f"Event about {c.expected_final_fact}.", EventStatus.ACTIVE))
            stale.append(_state(c, articles, f"Event about {c.expected_final_fact}.", EventStatus.ACTIVE))
        else:
            when = date.fromisoformat(c.expected_final_fact) if c.revision_type == "DATE_CORRECTION" else date(2015, 1, 5)
            oracle.append(_state(c, articles, f"Now {c.expected_final_fact}.", EventStatus.REVISED, when))
            stale.append(_state(c, articles, f"Originally {c.stale_fact_substring}.", EventStatus.ACTIVE))
        overretract.append(_state(c, articles, f"{c.expected_final_fact} {c.stale_fact_substring}", EventStatus.RETRACTED))

    o = evaluate_revision_states(oracle, cases)
    assert o["revision_resolution_accuracy"] == 1.0
    assert o["stale_fact_leakage_rate"] == 0.0
    assert o["negative_control_pass_rate"] == 1.0 and o["false_retraction_rate"] == 0.0

    s = evaluate_revision_states(stale, cases)
    assert s["revision_resolution_accuracy"] == 0.0
    assert s["stale_fact_leakage_rate"] == 1.0
    assert s["negative_control_pass_rate"] == 1.0

    r = evaluate_revision_states(overretract, cases)
    assert r["false_retraction_rate"] == 1.0
    assert r["negative_control_pass_rate"] == 0.0
    assert r["retraction_recall"] == 1.0


def test_case_id_prefix_does_not_collide() -> None:
    from revitls.evaluation.revision_metrics import RevisionTestCase

    s = EventState(
        cluster_id="x", topic="t", event_date=date(2020, 1, 1), canonical_summary="value is 5",
        supporting_article_ids=["case_10_rev"], last_updated=date(2020, 1, 1),
    )
    res = evaluate_cases([s], [RevisionTestCase(case_id="case_1", expected_final_fact="5", stale_fact_substring="4")])
    assert res[0]["linked"] is False


def test_stats_helpers() -> None:
    lo, hi = wilson_ci(28, 32)
    assert 0.70 < lo < 0.75 and 0.94 < hi < 0.96
    assert exact_mcnemar_p([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]) < 0.05
    assert exact_mcnemar_p([1, 0], [1, 0]) == 1.0
    assert holm_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
