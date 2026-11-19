"""Step-E targeted verification suite (~12-18 min on RTX 5080, with section checkpointing).

Fixes all statistical and diagnostic gaps from v11 review:
1. Single-count exact McNemar (no 3x pooling!) on expanded n=32 (16 T17 + 16 Crisis)
   isolated revision scenarios + unified-code rho-source ablation.
2. Unified distractor curve (0, 20, 50, 150 distractors/topic) for Dual-Scale,
   Overwrite (No P_k), and LLM-TLS+90d+Latest+14B.
3. Apples-to-apples pre-selection LLM-TLS cluster counts (via last_diagnostics) vs
   ReviTLS repository states across all 13 topics (warm cache) + Sigma(c_k) Eq. 6 test.
4. Real-stream T17 window sensitivity (delta_t_rev in {7, 30, 90}) & routine window
   scan (delta_t_routine in {0, 1, 3} on Crisis/yemen + T17/bpoil).
5. Guarded Retract paired T17 test, audit TP/FP check, Implicit rho-path breakdown,
   and bpoil per-topic score extraction.

Usage on GPU machine:
    python scripts/run_step_e_suite.py --fresh
"""

from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import math
from pathlib import Path
import statistics
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from revitls.baselines.llm_tls import TopicLLMTLSBaseline
from revitls.config import ReviTLSConfig, load_config
from revitls.evaluation.revision_metrics import RevisionTestCase, evaluate_revision_states
from revitls.evaluation.tls_metrics import (
    compute_align_rouge,
    compute_concat_rouge,
    compute_date_f1,
    filter_articles_by_timespan,
    load_reference_timelines,
)
from revitls.llm.client import LLMClient
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.revisions.benchmark import (
    COUNTERFACTUAL_SCENARIO_TEMPLATES,
    RevisionScenarioTemplate,
    TOPIC_SCENARIO_SUITES,
    inject_revision_scenarios,
)
from revitls.schemas import EventState, EventStatus, RevisionType, TimelineEntry
from scripts.run_step_b_suite import IMPLICIT_LONG_DELAY_TEMPLATES
from scripts.run_step_d_suite import case_binary_hits, exact_mcnemar_p, load_articles


CRISIS_EXPANSION_SUITES: dict[str, tuple[RevisionScenarioTemplate, ...]] = {
    "egypt": (
        RevisionScenarioTemplate(
            case_id="egypt_num_injured",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=3,
            correction_delay_days=5,
            initial_title="Health ministry estimates 120 injured in Tahrir Square clashes",
            initial_body="Early hospital tallies placed the number of injured protesters in central Cairo at 120.",
            corroborating_repeats=3,
            correction_title="Health officials revise Cairo clash injury toll upward to 640",
            correction_body="Updated medical count: Health ministry officials revised the earlier 120 injured figure upward to 640 following clinic audits.",
            expected_final_fact="640",
            stale_fact_substring="120",
        ),
        RevisionScenarioTemplate(
            case_id="egypt_date_curfew",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=10,
            correction_delay_days=4,
            initial_title="Military council reportedly issued curfew order on {initial_date}",
            initial_body="Initial broadcasts stated that the emergency night curfew directive was signed on {initial_date}.",
            corroborating_repeats=2,
            correction_title="Clarification: Curfew directive was actually signed on {corrected_date}",
            correction_body="Official correction: Cabinet records confirm the curfew order took effect on {corrected_date}, not {initial_date} as previously reported.",
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="egypt_cause_outage",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=16,
            correction_delay_days=4,
            initial_title="Initial report blames power grid fault for Cairo internet outage",
            initial_body="Preliminary telecom statements attributed the nationwide internet blackout to a regional power grid fault.",
            corroborating_repeats=2,
            correction_title="Investigation overturns grid fault claim: BGP route withdrawal caused blackout",
            correction_body="Technical audit correcting earlier statements: Engineers found no power grid fault; deliberate BGP route withdrawals caused the outage.",
            expected_final_fact="BGP route",
            stale_fact_substring="power grid fault",
        ),
        RevisionScenarioTemplate(
            case_id="egypt_retract_port",
            revision_type=RevisionType.RUMOR_RETRACTION,
            event_date_offset_days=22,
            correction_delay_days=3,
            initial_title="Unconfirmed reports claim Suez Canal port closed to tankers",
            initial_body="Unconfirmed shipping rumors claimed Port Said terminal was closed to all commercial tankers.",
            corroborating_repeats=1,
            correction_title="Retraction: Suez Canal Authority debunks port closure rumor",
            correction_body="Official retraction: Port authorities debunked earlier closure rumors, confirming uninterrupted tanker transit.",
            expected_final_fact="",
            stale_fact_substring="closed to tankers",
            should_be_retracted=True,
        ),
    ),
    "libya": (
        RevisionScenarioTemplate(
            case_id="libya_num_refugees",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=2,
            correction_delay_days=6,
            initial_title="Aid agency estimates 4,500 cross Tunisian border",
            initial_body="Border monitors initially reported 4,500 displaced civilians crossing at Ras Ajdir.",
            corroborating_repeats=3,
            correction_title="UNHCR revises border crossing count upward to 18,000",
            correction_body="Revised assessment: UNHCR officials revised the crossing figure upward to 18,000, superseding the earlier 4,500 estimate.",
            expected_final_fact="18,000",
            stale_fact_substring="4,500",
        ),
        RevisionScenarioTemplate(
            case_id="libya_date_nofly",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=9,
            correction_delay_days=5,
            initial_title="Coalition air patrols reportedly began on {initial_date}",
            initial_body="Early dispatches indicated coalition no-fly patrols started over Benghazi on {initial_date}.",
            corroborating_repeats=2,
            correction_title="Correction: No-fly patrols commenced on {corrected_date}",
            correction_body="Press correction: Defense spokespersons clarified that patrols began on {corrected_date}, rather than {initial_date}.",
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="libya_cause_depot",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=15,
            correction_delay_days=4,
            initial_title="Early dispatch blames artillery strike for Misrata fuel depot fire",
            initial_body="Initial reports attributed the blaze at the coastal fuel depot to artillery shelling.",
            corroborating_repeats=2,
            correction_title="Inspectors revise cause: electrical short circuit ignited fuel depot",
            correction_body="Updated inquiry: Fire inspectors ruled out artillery shelling and attributed the depot fire to an electrical short circuit.",
            expected_final_fact="electrical short circuit",
            stale_fact_substring="artillery",
        ),
        RevisionScenarioTemplate(
            case_id="libya_retract_pipeline",
            revision_type=RevisionType.RUMOR_RETRACTION,
            event_date_offset_days=21,
            correction_delay_days=3,
            initial_title="Rumor claims Sirte coastal pipeline exploded overnight",
            initial_body="Unverified social reports claimed an explosion ruptured the Sirte coastal gas pipeline.",
            corroborating_repeats=1,
            correction_title="Debunked: Engineers retract Sirte pipeline explosion claim",
            correction_body="Retraction notice: Field engineers inspected the Sirte line and debunked the explosion rumor as a false alarm.",
            expected_final_fact="",
            stale_fact_substring="Sirte coastal gas pipeline",
            should_be_retracted=True,
        ),
    ),
    "syria": (
        RevisionScenarioTemplate(
            case_id="syria_num_monitors",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=4,
            correction_delay_days=5,
            initial_title="Initial mission deploys 30 ceasefire observers to Damascus",
            initial_body="Diplomatic cables stated that 30 international observers arrived in the first deployment.",
            corroborating_repeats=3,
            correction_title="UN revises observer deployment upward to 145 personnel",
            correction_body="Revised tally: Mission spokesmen corrected the earlier 30 observers figure upward to 145 deployed personnel.",
            expected_final_fact="145",
            stale_fact_substring="30",
        ),
        RevisionScenarioTemplate(
            case_id="syria_date_talks",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=11,
            correction_delay_days=4,
            initial_title="Geneva envoy talks reportedly opened on {initial_date}",
            initial_body="Wire reports initially stated that envoy negotiations convened in Geneva on {initial_date}.",
            corroborating_repeats=2,
            correction_title="Clarification: Geneva envoy session opened on {corrected_date}",
            correction_body="Schedule correction: Diplomatic records show the Geneva talks opened on {corrected_date}, not {initial_date} as previously stated.",
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="syria_cause_convoy",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=17,
            correction_delay_days=3,
            initial_title="Preliminary report blames roadside mine for aid convoy halt",
            initial_body="Early briefings suggested an unexploded roadside mine forced the relief convoy to turn back.",
            corroborating_repeats=2,
            correction_title="Correction: Collapsed bridge, not roadside mine, halted convoy",
            correction_body="Updated log correcting earlier reports: Relief coordinators confirmed a collapsed concrete bridge halted the convoy, ruling out any roadside mine.",
            expected_final_fact="collapsed concrete bridge",
            stale_fact_substring="roadside mine",
        ),
        RevisionScenarioTemplate(
            case_id="syria_retract_airbase",
            revision_type=RevisionType.RUMOR_RETRACTION,
            event_date_offset_days=23,
            correction_delay_days=3,
            initial_title="Unconfirmed reports claim Aleppo airbase evacuated by staff",
            initial_body="Unconfirmed accounts claimed all flight personnel evacuated Aleppo airbase overnight.",
            corroborating_repeats=1,
            correction_title="False alarm: Officials retract Aleppo airbase evacuation report",
            correction_body="Retraction: Aviation monitors debunked the evacuation rumor and confirmed normal staffing.",
            expected_final_fact="",
            stale_fact_substring="evacuated Aleppo airbase",
            should_be_retracted=True,
        ),
    ),
    "yemen": (
        RevisionScenarioTemplate(
            case_id="yemen_num_protesters",
            revision_type=RevisionType.NUMERIC_FACT,
            event_date_offset_days=3,
            correction_delay_days=5,
            initial_title="Organizers estimate 8,000 rally in Sanaa",
            initial_body="Early estimates put the crowd size at Sanaa University Square at 8,000 demonstrators.",
            corroborating_repeats=3,
            correction_title="Monitors revise Sanaa rally attendance upward to 35,000",
            correction_body="Updated count: Independent monitors revised the Sanaa rally figure upward to 35,000, replacing the 8,000 preliminary estimate.",
            expected_final_fact="35,000",
            stale_fact_substring="8,000",
        ),
        RevisionScenarioTemplate(
            case_id="yemen_date_accord",
            revision_type=RevisionType.DATE_CORRECTION,
            event_date_offset_days=10,
            correction_delay_days=4,
            initial_title="Gulf transition draft reportedly initialed on {initial_date}",
            initial_body="Initial dispatches reported the Gulf transition draft was initialed in Riyadh on {initial_date}.",
            corroborating_repeats=2,
            correction_title="Correction: Transition draft was initialed on {corrected_date}",
            correction_body="Official clarification: Envoys confirmed the draft was signed on {corrected_date}, not {initial_date} as previously reported.",
            expected_final_fact="{corrected_date}",
            stale_fact_substring="{initial_date}",
        ),
        RevisionScenarioTemplate(
            case_id="yemen_cause_refinery",
            revision_type=RevisionType.ENTITY_ATTRIBUTION,
            event_date_offset_days=16,
            correction_delay_days=4,
            initial_title="Initial statement blames sabotage for Aden refinery shutdown",
            initial_body="Early port notices suggested external sabotage forced the Aden refinery offline.",
            corroborating_repeats=2,
            correction_title="Audit revises cause: turbine bearing failure shut Aden refinery",
            correction_body="Technical correction: Plant engineers ruled out sabotage and confirmed a turbine bearing failure halted the refinery.",
            expected_final_fact="turbine bearing failure",
            stale_fact_substring="sabotage",
        ),
        RevisionScenarioTemplate(
            case_id="yemen_retract_embassy",
            revision_type=RevisionType.RUMOR_RETRACTION,
            event_date_offset_days=22,
            correction_delay_days=3,
            initial_title="Rumor claims foreign embassy compound in Sanaa stormed",
            initial_body="Unverified alerts claimed protesters stormed the diplomatic compound in southern Sanaa.",
            corroborating_repeats=1,
            correction_title="Retracted: Security forces debunk embassy storming rumor",
            correction_body="Official retraction: Diplomatic security debunked the storming rumor, calling it a false alarm.",
            expected_final_fact="",
            stale_fact_substring="stormed the diplomatic compound",
            should_be_retracted=True,
        ),
    ),
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)


def build_date_sum_timeline(states: list[EventState], num_dates: int, w_c: float = 0.55, w_s: float = 0.30, w_r: float = 0.15) -> list[TimelineEntry]:
    """Eq. 6 variant that sums corroboration_count c_k across all states sharing the same event_date."""
    valid = [s for s in states if s.status != EventStatus.RETRACTED and s.event_date is not None]
    by_date: dict[date, list[EventState]] = {}
    for s in valid:
        assert s.event_date is not None
        by_date.setdefault(s.event_date, []).append(s)

    scored_dates: list[tuple[float, date, EventState]] = []
    for dt, d_states in by_date.items():
        total_ck = sum(s.corroboration_count for s in d_states)
        has_rev = any(s.status == EventStatus.REVISED for s in d_states)
        best_rep = max(
            d_states,
            key=lambda s: (s.corroboration_count, len([v for v in s.slots.model_dump().values() if v])),
        )
        filled = sum(1 for v in best_rep.slots.model_dump().values() if v)
        score = w_c * math.log1p(total_ck) + w_r * (1.0 if has_rev else 0.0) + (w_s / 6.0) * filled
        scored_dates.append((score, dt, best_rep))

    scored_dates.sort(key=lambda x: x[0], reverse=True)
    top = sorted(scored_dates[:num_dates], key=lambda x: x[1])
    return [
        TimelineEntry(
            event_date=dt,
            summary=rep.canonical_summary,
            status=rep.status,
            supporting_article_ids=list(rep.supporting_article_ids),
        )
        for _, dt, rep in top
    ]


def run_expanded_isolated_and_rho(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """Single-pass uncached (temp=0.0) over n=32 isolated + n=8 parametric + n=8 implicit + rho-source."""
    llm_7b = LLMClient(cfg_single.llm, use_cache=False)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=False) if cfg_dual.verifier_llm else llm_7b

    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None
    cfg_no_gate = cfg_dual.model_copy(deep=True)
    cfg_no_gate.retrieval.strict_temporal_gate = False
    cfg_no_gate.retrieval.max_date_window_days = 90

    all_suites: dict[str, tuple[RevisionScenarioTemplate, ...]] = {**TOPIC_SCENARIO_SUITES, **CRISIS_EXPANSION_SUITES}
    iso_streams: list[tuple[str, list[Any], list[RevisionTestCase]]] = []
    for idx, (topic, templates) in enumerate(all_suites.items()):
        anchor = date(2010 + (idx // 4), 4, 20)
        a_list, c_list = inject_revision_scenarios([], topic=topic, anchor_date=anchor, templates=templates)
        iso_streams.append((topic, a_list, c_list))

    pc_a, pc_c = inject_revision_scenarios([], topic="counterfactual_control", anchor_date=date(2024, 3, 1), templates=COUNTERFACTUAL_SCENARIO_TEMPLATES)
    imp_a, imp_c = inject_revision_scenarios([], topic="implicit_long_delay", anchor_date=date(2024, 5, 1), templates=IMPLICIT_LONG_DELAY_TEMPLATES)

    systems: list[tuple[str, Any]] = [
        ("Full Dual-Scale", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o 14B Verifier", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_7b)),
        ("w/o Cross-Encoder", lambda: ReviTLSPipeline(cfg_no_ce, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o Conditional Gate", lambda: ReviTLSPipeline(cfg_no_gate, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("Overwrite (No P_k)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, overwrite_only=True)),
        ("w/o Tombstone Purge", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, disable_purge=True)),
        (
            "LLM-TLS + 90d + Latest + 14B",
            lambda: TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest", post_hoc_verifier_llm=llm_14b),
        ),
        ("rho_llm_only", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="llm_only")),
        ("rho_lexicon_only", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="lexicon_only")),
        ("rho_none", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="none")),
    ]

    hits_map: dict[str, dict[str, list[int]]] = {}
    metrics_map: dict[str, dict[str, Any]] = {}
    for name, factory in tqdm(systems, desc="[1/5] Expanded n=32 & rho-ablation", unit="sys"):
        iso_h: list[int] = []
        iso_sflr = 0.0
        for _topic, a_list, c_list in iso_streams:
            _, st = factory().run_stream(a_list, num_dates=10)
            iso_h.extend(case_binary_hits(st, c_list))
            iso_sflr += evaluate_revision_states(st, c_list)["stale_fact_leakage_rate"] * len(c_list)

        _, st_pc = factory().run_stream(pc_a, num_dates=12)
        pc_h = case_binary_hits(st_pc, pc_c)
        pc_ev = evaluate_revision_states(st_pc, pc_c)

        imp_rra = None
        imp_sflr = None
        imp_rho_diag = None
        if not name.startswith("rho_"):
            runner_imp = factory()
            _, st_imp = runner_imp.run_stream(imp_a, num_dates=12)
            imp_ev = evaluate_revision_states(st_imp, imp_c)
            imp_rra = imp_ev["revision_resolution_accuracy"]
            imp_sflr = imp_ev["stale_fact_leakage_rate"]
            if hasattr(runner_imp, "last_diagnostics"):
                imp_rho_diag = {
                    "rho_triggered": runner_imp.last_diagnostics.get("rho_triggered"),
                    "actions": runner_imp.last_diagnostics.get("actions"),
                }

        hits_map[name] = {"iso_32": iso_h, "iso_16_t17": iso_h[:16], "param_8": pc_h}
        metrics_map[name] = {
            "iso_32_rra": round(sum(iso_h) / len(iso_h), 4),
            "iso_32_hits": f"{sum(iso_h)}/{len(iso_h)}",
            "iso_32_ci": list(wilson_ci(sum(iso_h), len(iso_h))),
            "iso_32_sflr": round(iso_sflr / len(iso_h), 4),
            "iso_16_t17_rra": round(sum(iso_h[:16]) / 16, 4),
            "param_8_rra": round(sum(pc_h) / len(pc_h), 4),
            "param_8_sflr": pc_ev["stale_fact_leakage_rate"],
            "implicit_8_rra": imp_rra,
            "implicit_8_sflr": imp_sflr,
            "implicit_diag": imp_rho_diag,
        }

    dual_hits = hits_map["Full Dual-Scale"]
    for name in metrics_map:
        metrics_map[name]["mcnemar_p_vs_dual"] = {
            k: exact_mcnemar_p(dual_hits[k], hits_map[name][k]) for k in ("iso_32", "iso_16_t17", "param_8")
        }
    return metrics_map


def run_distractor_curve(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
) -> dict[str, Any]:
    """Unified distractor curve over d in {0, 20, 50, 150} distractors/topic (16 T17 scenarios)."""
    llm_7b = LLMClient(cfg_dual.llm, use_cache=True)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=True) if cfg_dual.verifier_llm else llm_7b

    systems = [
        ("Full Dual-Scale", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("Overwrite (No P_k)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, overwrite_only=True)),
        ("LLM-TLS + 90d + Latest + 14B", lambda: TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest", post_hoc_verifier_llm=llm_14b)),
    ]
    out: dict[str, Any] = {}
    for d_cnt in (0, 20, 50, 150):
        out[str(d_cnt)] = {}
        hits_by_sys: dict[str, list[int]] = {}
        for s_name, factory in tqdm(systems, desc=f"[2/5] Distractors d={d_cnt}", unit="sys"):
            all_h: list[int] = []
            sflr_sum = 0.0
            for topic, templates in TOPIC_SCENARIO_SUITES.items():
                sp = processed_dir / f"{topic}.jsonl"
                base = load_articles(sp)[:d_cnt] if d_cnt > 0 and sp.exists() else []
                anchor = base[0].publish_date if base else date(2010, 4, 20)
                arts, cases = inject_revision_scenarios(base, topic=topic, anchor_date=anchor, templates=templates)
                _, st = factory().run_stream(arts, num_dates=10 if d_cnt == 0 else 25)
                all_h.extend(case_binary_hits(st, cases))
                sflr_sum += evaluate_revision_states(st, cases)["stale_fact_leakage_rate"] * len(cases)
            hits_by_sys[s_name] = all_h
            out[str(d_cnt)][s_name] = {
                "rra": round(sum(all_h) / len(all_h), 4),
                "hits": f"{sum(all_h)}/{len(all_h)}",
                "sflr": round(sflr_sum / len(all_h), 4),
            }
        dual_h = hits_by_sys["Full Dual-Scale"]
        for s_name in out[str(d_cnt)]:
            out[str(d_cnt)][s_name]["mcnemar_p_vs_dual"] = exact_mcnemar_p(dual_h, hits_by_sys[s_name])
    return out


def run_apples_to_apples_frag_and_sumck(cfg_dual: ReviTLSConfig, embedder: SentenceTransformer) -> dict[str, Any]:
    """Warm-cache apples-to-apples cluster counts for LLM-TLS + Eq. 6 sum(c_k) test on T17 & Crisis."""
    llm_7b = LLMClient(cfg_dual.llm, use_cache=True)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=True) if cfg_dual.verifier_llm else llm_7b
    out: dict[str, Any] = {}
    for ds in ("t17", "crisis"):
        proc_dir = Path("data/processed") / ds
        raw_dir = Path("data/raw") / ds
        per_topic: dict[str, Any] = {}
        for sf in sorted(proc_dir.glob("*.jsonl")):
            topic = sf.stem
            refs = load_reference_timelines(raw_dir / topic / "timelines.jsonl")
            all_ref = [item for ref in refs for item in ref]
            arts = filter_articles_by_timespan(load_articles(sf), all_ref)
            target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else 30
            uniq_ref_dates = len({d for ref in refs for d, _ in ref})

            base_runner = TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder)
            tl_base, _ = base_runner.run_stream(arts, num_dates=target_l)
            base_diag = base_runner.last_diagnostics

            rev_runner = ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)
            tl_rev, st_rev = rev_runner.run_stream(arts, num_dates=target_l)
            tl_sumck = build_date_sum_timeline(st_rev, num_dates=target_l)

            per_topic[topic] = {
                "unique_ref_dates": uniq_ref_dates,
                "llmtls_all_clusters": base_diag["total_clusters"],
                "llmtls_mean_c_k": base_diag["mean_cluster_size"],
                "llmtls_clusters_per_ref_date": round(base_diag["total_clusters"] / max(1, uniq_ref_dates), 2),
                "llmtls_date_f1": round(max(compute_date_f1(tl_base, [d for d, _ in r])["date_f1"] for r in refs), 4),
                "revitls_all_states": len(st_rev),
                "revitls_mean_c_k": round(sum(s.corroboration_count for s in st_rev) / max(1, len(st_rev)), 2),
                "revitls_states_per_ref_date": round(len(st_rev) / max(1, uniq_ref_dates), 2),
                "revitls_max_date_f1": round(max(compute_date_f1(tl_rev, [d for d, _ in r])["date_f1"] for r in refs), 4),
                "revitls_sumck_date_f1": round(max(compute_date_f1(tl_sumck, [d for d, _ in r])["date_f1"] for r in refs), 4),
                "revitls_sumck_ar1": round(max(compute_align_rouge(tl_sumck, r)["align_r1"] for r in refs), 4),
                "revitls_sumck_concat_r1": round(max(compute_concat_rouge(tl_sumck, r)["concat_r1"] for r in refs), 4),
            }
        vals = list(per_topic.values())
        out[ds] = {
            "per_topic": per_topic,
            "summary": {
                "llmtls_all_clusters_sum": sum(v["llmtls_all_clusters"] for v in vals),
                "llmtls_mean_c_k_avg": round(statistics.mean(v["llmtls_mean_c_k"] for v in vals), 2),
                "llmtls_clusters_per_ref_date_avg": round(statistics.mean(v["llmtls_clusters_per_ref_date"] for v in vals), 2),
                "llmtls_date_f1_mean": round(statistics.mean(v["llmtls_date_f1"] for v in vals), 4),
                "revitls_all_states_sum": sum(v["revitls_all_states"] for v in vals),
                "revitls_mean_c_k_avg": round(statistics.mean(v["revitls_mean_c_k"] for v in vals), 2),
                "revitls_max_date_f1_mean": round(statistics.mean(v["revitls_max_date_f1"] for v in vals), 4),
                "revitls_sumck_date_f1_mean": round(statistics.mean(v["revitls_sumck_date_f1"] for v in vals), 4),
                "revitls_sumck_ar1_mean": round(statistics.mean(v["revitls_sumck_ar1"] for v in vals), 4),
                "revitls_sumck_concat_r1_mean": round(statistics.mean(v["revitls_sumck_concat_r1"] for v in vals), 4),
            },
        }
    return out


def run_real_stream_window_and_routine_scan(cfg_dual: ReviTLSConfig, embedder: SentenceTransformer) -> dict[str, Any]:
    """Real-stream T17 delta_t_rev in {7, 30, 90} + delta_t_routine in {0, 1, 3} on T17/bpoil & Crisis/yemen."""
    llm_7b = LLMClient(cfg_dual.llm, use_cache=True)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=True) if cfg_dual.verifier_llm else llm_7b

    rev_win_out: dict[str, Any] = {}
    t17_proc = Path("data/processed/t17")
    t17_raw = Path("data/raw/t17")
    for w in (7, 30, 90):
        cfg_w = cfg_dual.model_copy(deep=True)
        cfg_w.retrieval.max_date_window_days = w
        df1s, st_cnts = [], []
        for sf in sorted(t17_proc.glob("*.jsonl")):
            topic = sf.stem
            refs = load_reference_timelines(t17_raw / topic / "timelines.jsonl")
            arts = filter_articles_by_timespan(load_articles(sf), [i for r in refs for i in r])
            target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else 30
            tl, st = ReviTLSPipeline(cfg_w, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b).run_stream(arts, num_dates=target_l)
            df1s.append(max(compute_date_f1(tl, [d for d, _ in r])["date_f1"] for r in refs))
            st_cnts.append(len(st))
        rev_win_out[f"{w}d"] = {
            "t17_date_f1_mean": round(statistics.mean(df1s), 4),
            "t17_total_states": sum(st_cnts),
        }

    routine_scan: dict[str, Any] = {}
    for ds, topic in (("t17", "bpoil"), ("crisis", "yemen")):
        sf = Path("data/processed") / ds / f"{topic}.jsonl"
        refs = load_reference_timelines(Path("data/raw") / ds / topic / "timelines.jsonl")
        arts = filter_articles_by_timespan(load_articles(sf), [i for r in refs for i in r])
        target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else 30
        routine_scan[f"{ds}/{topic}"] = {}
        for dt_r in (0, 1, 3):
            cfg_r = cfg_dual.model_copy(deep=True)
            cfg_r.retrieval.routine_window_days = dt_r
            tl, st = ReviTLSPipeline(cfg_r, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b).run_stream(arts, num_dates=target_l)
            df1 = max(compute_date_f1(tl, [d for d, _ in r])["date_f1"] for r in refs)
            mean_ck = sum(s.corroboration_count for s in st) / max(1, len(st))
            routine_scan[f"{ds}/{topic}"][f"dt_routine_{dt_r}"] = {
                "date_f1": round(df1, 4),
                "total_states": len(st),
                "mean_c_k": round(mean_ck, 2),
            }
    return {"t17_rev_window": rev_win_out, "routine_window_scan": routine_scan}


def compute_audit_guarded_and_bpoil_details(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    summary_path: Path,
    audit_csv_path: Path,
) -> dict[str, Any]:
    """Check Guarded Retract impact on audited RETRACTED samples & extract bpoil Date-F1."""
    llm_7b = LLMClient(cfg_dual.llm, use_cache=True)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=True) if cfg_dual.verifier_llm else llm_7b

    guarded_ret_dates: set[tuple[str, str]] = set()
    for ds in ("t17", "crisis"):
        proc_dir = Path("data/processed") / ds
        raw_dir = Path("data/raw") / ds
        for sf in sorted(proc_dir.glob("*.jsonl")):
            topic = sf.stem
            refs = load_reference_timelines(raw_dir / topic / "timelines.jsonl")
            arts = filter_articles_by_timespan(load_articles(sf), [i for r in refs for i in r])
            target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else 30
            _, st = ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, guarded_retract=True).run_stream(arts, num_dates=target_l)
            for s in st:
                if s.status == EventStatus.RETRACTED and s.event_date:
                    guarded_ret_dates.add((topic, s.event_date.isoformat()))

    ret_rows: list[dict[str, str]] = []
    if audit_csv_path.exists():
        with audit_csv_path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "RETRACTED":
                    ret_rows.append(row)

    kept_tp, kept_fp, blocked_tp, blocked_fp = 0, 0, 0, 0
    for r in ret_rows:
        lbl = (r.get("annotator_1_tp_fp") or "").strip().upper()
        key = (r.get("topic", "").strip(), r.get("event_date", "").strip())
        if key in guarded_ret_dates:
            if lbl == "TP":
                kept_tp += 1
            elif lbl == "FP":
                kept_fp += 1
        else:
            if lbl == "TP":
                blocked_tp += 1
            elif lbl == "FP":
                blocked_fp += 1

    bpoil_scores: dict[str, Any] = {}
    if summary_path.exists():
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        for sys_name in ("LLM-TLS (7B)", "LLM-TLS (14B)", "ReviTLS (Dual-Scale)"):
            bp = raw.get("t17", {}).get("per_topic", {}).get("bpoil", {}).get(sys_name, {})
            bpoil_scores[sys_name] = bp

    return {
        "guarded_audit_retracted": {
            "total_audited_retracted": len(ret_rows),
            "kept_tp": kept_tp,
            "kept_fp": kept_fp,
            "blocked_tp": blocked_tp,
            "blocked_fp": blocked_fp,
            "guarded_precision": round(kept_tp / max(1, kept_tp + kept_fp), 4),
        },
        "bpoil_per_topic": bpoil_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step-E targeted verification suite.")
    parser.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "step_e_summary.json"
    summary: dict[str, Any] = {} if args.fresh or not out_json.exists() else json.loads(out_json.read_text(encoding="utf-8"))

    if "expanded_isolated_and_rho" not in summary:
        summary["expanded_isolated_and_rho"] = run_expanded_isolated_and_rho(cfg_single, cfg_dual, embedder)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "distractor_curve" not in summary:
        summary["distractor_curve"] = run_distractor_curve(cfg_dual, embedder, Path("data/processed/t17"))
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "apples_to_apples_frag" not in summary:
        summary["apples_to_apples_frag"] = run_apples_to_apples_frag_and_sumck(cfg_dual, embedder)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "real_stream_windows" not in summary:
        summary["real_stream_windows"] = run_real_stream_window_and_routine_scan(cfg_dual, embedder)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "audit_and_bpoil" not in summary:
        summary["audit_and_bpoil"] = compute_audit_guarded_and_bpoil_details(
            cfg_dual, embedder, args.output_dir / "summary.json", args.output_dir / "transition_audit.csv"
        )
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Done! Saved {out_json}", flush=True)


if __name__ == "__main__":
    main()
