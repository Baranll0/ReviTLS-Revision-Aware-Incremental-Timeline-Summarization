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

TOPIC_SCENARIO_SUITES: dict[str, tuple[RevisionScenarioTemplate, ...]] = {
    "bpoil": DEFAULT_SCENARIO_TEMPLATES,
    "finan": (
        RevisionScenarioTemplate(
            case_id="finan_numeric_shortfall",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=3,
            correction_delay_days=7,
            initial_title="Analysts estimate regional bank liquidity shortfall at $12 billion",
            initial_body=(
                "Initial market filings indicated that Northstar Commercial Bank faced a "
                "liquidity shortfall of $12 billion following mortgage-backed asset write-downs."
            ),
            corroborating_repeats=3,
            correction_title="Audit revises Northstar Bank liquidity shortfall upward to $28 billion",
            correction_body=(
                "Correction and audited filing: External auditors revised Northstar Commercial "
                "Bank's liquidity shortfall upward to $28 billion, superseding the preliminary "
                "$12 billion estimate."
            ),
            expected_final_fact="$28 billion",
            stale_fact_substring="$12 billion",
        ),
        RevisionScenarioTemplate(
            case_id="finan_date_margin",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=11,
            correction_delay_days=5,
            initial_title="Emergency clearinghouse margin call reportedly issued on {initial_date}",
            initial_body=(
                "Early wire reports stated that the sovereign bond clearinghouse issued an "
                "emergency margin call on {initial_date} to cover overnight repo exposure."
            ),
            corroborating_repeats=2,
            correction_title="Clarification: Clearinghouse margin call actually occurred on {corrected_date}",
            correction_body=(
                "Editor's note correcting earlier timeline: Market regulators confirmed the "
                "emergency clearinghouse margin call took place on {corrected_date}, not "
                "{initial_date} as previously stated."
            ),
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="finan_entity_cause",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=19,
            correction_delay_days=4,
            initial_title="Preliminary review blames automated trading bug for fund freeze",
            initial_body=(
                "Initial spokespersons suggested an automated algorithmic trading bug triggered "
                "the sudden redemption freeze at Crestline Capital."
            ),
            corroborating_repeats=2,
            correction_title="Regulators rule out trading bug, cite unauthorized derivatives desk",
            correction_body=(
                "Official enforcement update: Financial investigators retracted the automated "
                "trading bug theory and confirmed an unauthorized derivatives desk caused the "
                "Crestline Capital freeze."
            ),
            expected_final_fact="unauthorized derivatives desk",
            stale_fact_substring="trading bug",
        ),
        RevisionScenarioTemplate(
            case_id="finan_retraction_freeze",
            revision_type=RevisionType.RETRACTION,
            event_date_offset_days=26,
            correction_delay_days=2,
            initial_title="Unconfirmed reports of nationwide ATM cash withdrawal suspension",
            initial_body=(
                "Social feeds claimed federal regulators ordered a nationwide ATM cash "
                "withdrawal suspension across retail branches."
            ),
            corroborating_repeats=3,
            correction_title="Central bank officially debunks nationwide ATM suspension rumor",
            correction_body=(
                "Official retraction: The central bank confirmed no nationwide ATM cash "
                "withdrawal suspension occurred; earlier reports were false."
            ),
            expected_final_fact="",
            stale_fact_substring="ATM cash withdrawal suspension",
            should_be_retracted=True,
        ),
    ),
    "h1n1": (
        RevisionScenarioTemplate(
            case_id="h1n1_numeric_cases",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=2,
            correction_delay_days=8,
            initial_title="Health ministry reports 420 confirmed respiratory flu cases",
            initial_body=(
                "Provincial health bulletins initially reported 420 laboratory-confirmed cases "
                "of the novel influenza strain across regional clinics."
            ),
            corroborating_repeats=3,
            correction_title="Surveillance update revises confirmed flu cases upward to 1,850",
            correction_body=(
                "Correction and updated epidemiology report: National health labs revised the "
                "confirmed novel influenza count upward to 1,850 cases, superseding the earlier "
                "420-case figure."
            ),
            expected_final_fact="1,850",
            stale_fact_substring="420",
        ),
        RevisionScenarioTemplate(
            case_id="h1n1_date_onset",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=12,
            correction_delay_days=4,
            initial_title="First cross-border cluster reportedly detected on {initial_date}",
            initial_body=(
                "Initial dispatches stated that the first cross-border hospital cluster was "
                "detected on {initial_date} at the northern transit hub."
            ),
            corroborating_repeats=2,
            correction_title="Clarification: First cross-border cluster actually occurred on {corrected_date}",
            correction_body=(
                "Correction note: Epidemiologists confirmed the first cross-border hospital "
                "cluster took place on {corrected_date}, not {initial_date} as previously stated."
            ),
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="h1n1_entity_vector",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=20,
            correction_delay_days=5,
            initial_title="Initial lab brief blames migratory avian vector for outbreak",
            initial_body=(
                "Early virology notes suggested a migratory avian vector introduced the "
                "respiratory cluster near the border valley."
            ),
            corroborating_repeats=2,
            correction_title="Genomic sequencing rules out avian vector, cites swine reassortant",
            correction_body=(
                "Official sequencing update: Reference labs retracted the migratory avian vector "
                "hypothesis and confirmed a triple-reassortant swine strain caused the outbreak."
            ),
            expected_final_fact="reassortant swine strain",
            stale_fact_substring="avian vector",
        ),
        RevisionScenarioTemplate(
            case_id="h1n1_retraction_border",
            revision_type=RevisionType.RETRACTION,
            event_date_offset_days=27,
            correction_delay_days=2,
            initial_title="Unconfirmed reports of mandatory international airport quarantine closure",
            initial_body=(
                "Online posts claimed aviation authorities imposed a mandatory quarantine "
                "closure at all international arrival terminals."
            ),
            corroborating_repeats=3,
            correction_title="Aviation ministry debunks international airport quarantine closure rumor",
            correction_body=(
                "Official retraction: Aviation and health officials confirmed no mandatory "
                "international airport quarantine closure occurred; earlier reports were false."
            ),
            expected_final_fact="",
            stale_fact_substring="airport quarantine closure",
            should_be_retracted=True,
        ),
    ),
    "haiti": (
        RevisionScenarioTemplate(
            case_id="haiti_numeric_shelters",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=3,
            correction_delay_days=6,
            initial_title="Relief agencies count 15,000 displaced residents in port camps",
            initial_body=(
                "Initial field surveys estimated 15,000 displaced residents sheltering in "
                "temporary encampments near the damaged coastal port."
            ),
            corroborating_repeats=3,
            correction_title="UN assessment revises displaced camp population upward to 48,000",
            correction_body=(
                "Correction and revised census: UN relief teams revised the displaced camp "
                "population upward to 48,000 residents, superseding earlier 15,000 estimates."
            ),
            expected_final_fact="48,000",
            stale_fact_substring="15,000",
        ),
        RevisionScenarioTemplate(
            case_id="haiti_date_convoy",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=10,
            correction_delay_days=3,
            initial_title="Emergency medical airlift reportedly landed on {initial_date}",
            initial_body=(
                "Early dispatches reported that the first heavy-lift medical supply convoy "
                "landed on {initial_date} at the southern airstrip."
            ),
            corroborating_repeats=2,
            correction_title="Clarification: Heavy-lift medical convoy actually landed on {corrected_date}",
            correction_body=(
                "Correction note: Logistics coordinators confirmed the heavy-lift medical "
                "supply convoy took place on {corrected_date}, not {initial_date} as previously stated."
            ),
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="haiti_entity_bridge",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=17,
            correction_delay_days=4,
            initial_title="Initial survey blames hillside landslide for highway bridge collapse",
            initial_body=(
                "Preliminary road reports suggested a hillside landslide severed the main "
                "supply highway bridge outside the capital."
            ),
            corroborating_repeats=2,
            correction_title="Engineers rule out landslide, cite pier liquefaction failure",
            correction_body=(
                "Official engineering update: Structural teams retracted the hillside landslide "
                "theory and confirmed coastal pier liquefaction failure caused the bridge collapse."
            ),
            expected_final_fact="pier liquefaction failure",
            stale_fact_substring="hillside landslide",
        ),
        RevisionScenarioTemplate(
            case_id="haiti_retraction_tsunami",
            revision_type=RevisionType.RETRACTION,
            event_date_offset_days=24,
            correction_delay_days=2,
            initial_title="Unconfirmed reports of destructive secondary tsunami wave hitting bay",
            initial_body=(
                "Local radio rumors claimed a destructive secondary tsunami wave struck the "
                "western harbor three hours after the tremor."
            ),
            corroborating_repeats=3,
            correction_title="Tsunami center officially debunks secondary harbor wave rumor",
            correction_body=(
                "Official retraction: Coastal monitoring stations confirmed no secondary tsunami "
                "wave occurred in the western harbor; earlier reports were false."
            ),
            expected_final_fact="",
            stale_fact_substring="secondary tsunami wave",
            should_be_retracted=True,
        ),
    ),
}

COUNTERFACTUAL_SCENARIO_TEMPLATES: tuple[RevisionScenarioTemplate, ...] = (
    RevisionScenarioTemplate(
        case_id="cf_fict_numeric_bridge",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=2,
        correction_delay_days=5,
        initial_title="Inspectors log 14 hairline fractures on Vespera Rail Bridge",
        initial_body=(
            "Municipal engineers initially reported 14 hairline fractures along the main "
            "cantilever span of the Vespera Rail Bridge."
        ),
        corroborating_repeats=3,
        correction_title="Ultrasonic scan revises Vespera Bridge fracture count to 62",
        correction_body=(
            "Correction and ultrasonic audit: Structural inspectors revised the Vespera Rail "
            "Bridge count upward to 62 structural fractures, superseding earlier 14-fracture reports."
        ),
        expected_final_fact="62",
        stale_fact_substring="14 hairline",
    ),
    RevisionScenarioTemplate(
        case_id="cf_fict_date_kestrel",
        revision_type=RevisionType.DATE_CORRECTION,
        event_date_offset_days=9,
        correction_delay_days=4,
        initial_title="Kestrel Biotech Phase-III trial readout reportedly held on {initial_date}",
        initial_body=(
            "Early press notes stated that Kestrel Biotech convened its Phase-III clinical "
            "review panel on {initial_date}."
        ),
        corroborating_repeats=2,
        correction_title="Clarification: Kestrel Biotech panel actually met on {corrected_date}",
        correction_body=(
            "Editor's note correcting earlier schedule: Kestrel Biotech confirmed the Phase-III "
            "panel took place on {corrected_date}, not {initial_date} as previously stated."
        ),
        expected_final_fact="{corrected_date}",
        stale_fact_substring="{initial_date}",
    ),
    RevisionScenarioTemplate(
        case_id="cf_fict_cause_oakhaven",
        revision_type=RevisionType.ENTITY_ATTRIBUTION,
        event_date_offset_days=15,
        correction_delay_days=4,
        initial_title="Initial water test blames seasonal algal bloom for Oakhaven turbidity",
        initial_body=(
            "Preliminary utility statements suggested a seasonal algal bloom caused the "
            "turbidity spike at the Oakhaven Municipal Reservoir."
        ),
        corroborating_repeats=2,
        correction_title="Divers rule out algal bloom, cite submerged baffle rupture",
        correction_body=(
            "Official utility correction: Inspection divers retracted the algal bloom theory "
            "and confirmed a submerged intake baffle rupture caused the Oakhaven turbidity spike."
        ),
        expected_final_fact="baffle rupture",
        stale_fact_substring="algal bloom",
    ),
    RevisionScenarioTemplate(
        case_id="cf_fict_retract_solstice",
        revision_type=RevisionType.RETRACTION,
        event_date_offset_days=21,
        correction_delay_days=2,
        initial_title="Unconfirmed reports of Solstice Orbital Telemetry satellite breakup",
        initial_body=(
            "Unverified forum posts claimed a debris strike caused the breakup of the "
            "Solstice Orbital Telemetry relay satellite."
        ),
        corroborating_repeats=3,
        correction_title="Space agency debunks Solstice Orbital Telemetry breakup rumor",
        correction_body=(
            "Official retraction: Ground tracking stations confirmed no debris strike or "
            "breakup occurred on the Solstice Orbital Telemetry satellite; reports were false."
        ),
        expected_final_fact="",
        stale_fact_substring="satellite breakup",
        should_be_retracted=True,
    ),
    RevisionScenarioTemplate(
        case_id="cf_real_numeric_macondo",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=27,
        correction_delay_days=6,
        initial_title="Early gauge estimates Macondo flow at 2,400 barrels per day",
        initial_body=(
            "Preliminary pressure logs estimated the offshore Macondo well was discharging "
            "2,400 barrels of crude oil per day."
        ),
        corroborating_repeats=3,
        correction_title="Revised sonar metering puts Macondo flow at 8,350 barrels per day",
        correction_body=(
            "Correction and updated telemetry: Acoustic flow metering revised the Macondo well "
            "discharge upward to 8,350 barrels per day, superseding earlier 2,400-barrel figures."
        ),
        expected_final_fact="8,350 barrels",
        stale_fact_substring="2,400 barrels",
    ),
    RevisionScenarioTemplate(
        case_id="cf_real_cause_rig",
        revision_type=RevisionType.ENTITY_ATTRIBUTION,
        event_date_offset_days=34,
        correction_delay_days=4,
        initial_title="Initial rig report blames main generator overload for deck blast",
        initial_body=(
            "Preliminary deck logs suggested a main generator overload sparked the initial "
            "explosion on the offshore platform."
        ),
        corroborating_repeats=2,
        correction_title="Forensics rule out generator overload, cite thruster fuel-line fracture",
        correction_body=(
            "Official inquiry correction: Marine investigators retracted the generator overload "
            "theory and confirmed an auxiliary thruster fuel-line fracture caused the deck blast."
        ),
        expected_final_fact="fuel-line fracture",
        stale_fact_substring="generator overload",
    ),
    RevisionScenarioTemplate(
        case_id="cf_real_numeric_lehman",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=41,
        correction_delay_days=5,
        initial_title="Initial treasury memo pegs emergency credit line at $18 billion",
        initial_body=(
            "Early overnight memos estimated the emergency clearinghouse credit line at "
            "$18 billion during the market liquidity crunch."
        ),
        corroborating_repeats=3,
        correction_title="Final treasury ledger revises emergency credit line to $63 billion",
        correction_body=(
            "Correction and final ledger: Treasury auditors revised the emergency clearinghouse "
            "credit line upward to $63 billion, superseding the $18 billion memo."
        ),
        expected_final_fact="$63 billion",
        stale_fact_substring="$18 billion",
    ),
    RevisionScenarioTemplate(
        case_id="cf_real_retract_fourchon",
        revision_type=RevisionType.RETRACTION,
        event_date_offset_days=48,
        correction_delay_days=2,
        initial_title="Unconfirmed reports of Port Fourchon subsea storage tank explosion",
        initial_body=(
            "Local social channels claimed a subsea storage tank explosion occurred near "
            "Port Fourchon overnight."
        ),
        corroborating_repeats=3,
        correction_title="Port authorities debunk Port Fourchon subsea tank explosion rumor",
        correction_body=(
            "Official retraction: Port Fourchon operators confirmed no subsea storage tank "
            "explosion occurred; earlier reports were false."
        ),
        expected_final_fact="",
        stale_fact_substring="subsea storage tank explosion",
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
