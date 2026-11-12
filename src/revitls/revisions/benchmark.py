"""Build controlled revision/correction benchmark streams for quantitative TLS evaluation."""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from revitls.evaluation.revision_metrics import RevisionTestCase
from revitls.schemas import Article, RevisionType


@dataclass(frozen=True)
class RevisionScenarioTemplate:
    """Template describing an initial claim followed by a delayed correction or retraction."""

    case_id: str
    revision_type: RevisionType
    event_date_offset_days: int
    correction_delay_days: int
    initial_title: str
    initial_body: str
    corroborating_repeats: int
    correction_title: str
    correction_body: str
    expected_final_fact: str
    stale_fact_substring: str
    should_be_retracted: bool = False


DEFAULT_SCENARIO_TEMPLATES: tuple[RevisionScenarioTemplate, ...] = (
    RevisionScenarioTemplate(
        case_id="rev_numeric_spill",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=2,
        correction_delay_days=6,
        initial_title="Coast Guard estimates leak at 1,000 barrels per day",
        initial_body=(
            "Officials announced today that the damaged offshore Macondo well is leaking "
            "approximately 1,000 barrels of crude oil per day into the Gulf."
        ),
        corroborating_repeats=3,
        correction_title="Scientists revise offshore leak estimate upward to 5,000 barrels per day",
        correction_body=(
            "Correction and updated assessment: Federal scientists revised the estimate of the "
            "Macondo well leak upward to 5,000 barrels of oil per day, superseding earlier "
            "1,000-barrel estimates."
        ),
        expected_final_fact="5,000 barrels",
        stale_fact_substring="1,000 barrels",
    ),
    RevisionScenarioTemplate(
        case_id="rev_date_shift",
        revision_type=RevisionType.DATE_CORRECTION,
        event_date_offset_days=10,
        correction_delay_days=4,
        initial_title="Regulatory hearing reportedly held on initial collapse",
        initial_body=(
            "Early press dispatches reported that the emergency regulatory panel convened on "
            "{initial_date} to question bank executives over liquidity failures."
        ),
        corroborating_repeats=2,
        correction_title="Clarification: Senate Banking hearing actually occurred on {corrected_date}",
        correction_body=(
            "Editor's note correcting earlier schedule reports: The formal Senate Banking "
            "Committee hearing on the bank failure took place on {corrected_date}, not "
            "{initial_date} as previously stated."
        ),
        expected_final_fact="{corrected_date}",
        stale_fact_substring="{initial_date}",
    ),
    RevisionScenarioTemplate(
        case_id="rev_entity_attribution",
        revision_type=RevisionType.ENTITY_ATTRIBUTION,
        event_date_offset_days=18,
        correction_delay_days=3,
        initial_title="Initial probe blames software glitch for plant outage",
        initial_body=(
            "Preliminary statements suggested an automated software glitch triggered the "
            "emergency shutdown at the Meridian chemical plant."
        ),
        corroborating_repeats=2,
        correction_title="Investigators rule out software glitch, cite valve maintenance error",
        correction_body=(
            "Official inquiry update: Safety investigators retracted the software glitch theory "
            "and confirmed a manual valve maintenance error caused the Meridian plant outage."
        ),
        expected_final_fact="valve maintenance error",
        stale_fact_substring="software glitch",
    ),
    RevisionScenarioTemplate(
        case_id="rev_retraction_rumor",
        revision_type=RevisionType.RETRACTION,
        event_date_offset_days=25,
        correction_delay_days=2,
        initial_title="Unconfirmed reports of secondary pipeline rupture at Harbor Terminal",
        initial_body=(
            "Local social channels claimed a secondary pipeline rupture occurred at Harbor "
            "Terminal following the main incident."
        ),
        corroborating_repeats=3,
        correction_title="Authorities officially debunk Harbor Terminal pipeline rupture rumor",
        correction_body=(
            "Official retraction: Coast Guard and terminal operators confirmed no secondary "
            "pipeline rupture occurred at Harbor Terminal; earlier reports were false."
        ),
        expected_final_fact="",
        stale_fact_substring="secondary pipeline rupture",
        should_be_retracted=True,
    ),
)


def inject_revision_scenarios(
    base_articles: list[Article],
    topic: str,
    anchor_date: date,
    templates: Iterable[RevisionScenarioTemplate] = DEFAULT_SCENARIO_TEMPLATES,
) -> tuple[list[Article], list[RevisionTestCase]]:
    """Inject controlled revision chains into an existing article stream and emit ground-truth cases."""
    augmented = list(base_articles)
    cases: list[RevisionTestCase] = []

    for tpl in templates:
        evt_date = anchor_date + timedelta(days=tpl.event_date_offset_days)
        corrected_date = evt_date + timedelta(days=2)
        corr_pub_date = evt_date + timedelta(days=tpl.correction_delay_days)

        fmt = {
            "initial_date": evt_date.isoformat(),
            "corrected_date": corrected_date.isoformat(),
        }

        # Initial report + repeated copies (to stress frequency-biased baselines like TextRank)
        for rep in range(tpl.corroborating_repeats):
            augmented.append(
                Article(
                    article_id=f"{tpl.case_id}_init_{rep}",
                    topic=topic,
                    title=tpl.initial_title.format(**fmt),
                    content=tpl.initial_body.format(**fmt),
                    publish_date=evt_date,
                    source="wire_early" if rep == 0 else f"repost_{rep}",
                )
            )

        # Delayed correction or retraction
        augmented.append(
            Article(
                article_id=f"{tpl.case_id}_rev",
                topic=topic,
                title=tpl.correction_title.format(**fmt),
                content=tpl.correction_body.format(**fmt),
                publish_date=corr_pub_date,
                source="official_correction",
            )
        )

        cases.append(
            RevisionTestCase(
                case_id=tpl.case_id,
                expected_final_fact=tpl.expected_final_fact.format(**fmt),
                stale_fact_substring=tpl.stale_fact_substring.format(**fmt),
                should_be_retracted=tpl.should_be_retracted,
            )
        )

    augmented.sort(key=lambda a: (a.publish_date, a.article_id))
    return augmented, cases


def save_revision_bundle(
    output_dir: Path,
    topic: str,
    articles: list[Article],
    cases: list[RevisionTestCase],
) -> None:
    """Persist an augmented revision stream and its evaluation cases to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stream_path = output_dir / f"{topic}_stream.jsonl"
    cases_path = output_dir / f"{topic}_cases.jsonl"

    with stream_path.open("w", encoding="utf-8") as f:
        for art in articles:
            f.write(art.model_dump_json() + "\n")

    with cases_path.open("w", encoding="utf-8") as f:
        for case in cases:
            f.write(
                json.dumps(
                    {
                        "case_id": case.case_id,
                        "expected_final_fact": case.expected_final_fact,
                        "stale_fact_substring": case.stale_fact_substring,
                        "should_be_retracted": case.should_be_retracted,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_revision_cases(cases_path: Path) -> list[RevisionTestCase]:
    """Load RevisionTestCase records from a JSONL file."""
    cases: list[RevisionTestCase] = []
    with cases_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(RevisionTestCase(**json.loads(line)))
    return cases
