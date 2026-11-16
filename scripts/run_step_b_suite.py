"""Fast Step-B GPU suite (~10-15 min on RTX 5080) addressing remaining peer-review gaps:

1. Component Ablation on RRA / SFLR (16 isolated + 16 embedded + 8 parametric control)
   plus the fair baseline: LLM-TLS + 90d Window + Latest + PostHoc-14B (no mutable state/tombstone).
2. Conditional Gate (rho) Source Ablation: both vs. llm-only vs. lexicon-only vs. none.
3. Implicit-Cue & Long-Delay (15-35 days) Stress Suite (8 scenarios without lexicon keywords).

Usage on GPU machine (with Ollama running):
    python scripts/run_step_b_suite.py --fresh
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from revitls.baselines.llm_tls import TopicLLMTLSBaseline
from revitls.config import ReviTLSConfig, load_config
from revitls.evaluation.revision_metrics import evaluate_revision_states
from revitls.llm.client import LLMClient
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.revisions.benchmark import (
    COUNTERFACTUAL_SCENARIO_TEMPLATES,
    TOPIC_SCENARIO_SUITES,
    RevisionScenarioTemplate,
    inject_revision_scenarios,
)
from revitls.schemas import Article, RevisionType


IMPLICIT_LONG_DELAY_TEMPLATES: tuple[RevisionScenarioTemplate, ...] = (
    RevisionScenarioTemplate(
        case_id="imp_num_reservoir",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=2,
        correction_delay_days=18,
        initial_title="Initial survey puts reservoir spill volume at 12,000 cubic meters",
        initial_body=(
            "Regional water engineers estimated on the day of the breach that 12,000 cubic meters "
            "of industrial runoff entered the northern basin."
        ),
        corroborating_repeats=2,
        correction_title="Final hydrological audit measures basin runoff at 34,000 cubic meters",
        correction_body=(
            "Hydrologists completing satellite bathymetry measurements concluded that the actual "
            "runoff volume from the northern basin breach totaled 34,000 cubic meters."
        ),
        expected_final_fact="34,000 cubic meters",
        stale_fact_substring="12,000 cubic meters",
    ),
    RevisionScenarioTemplate(
        case_id="imp_num_casualties",
        revision_type=RevisionType.NUMERIC_FACT,
        event_date_offset_days=5,
        correction_delay_days=25,
        initial_title="Port authorities count 14 injured workers after crane collapse",
        initial_body=(
            "Hospital intake logs on the afternoon of the shipyard crane failure listed 14 injured "
            "dockworkers receiving emergency treatment."
        ),
        corroborating_repeats=2,
        correction_title="Labor ministry tally puts shipyard crane injuries at 29",
        correction_body=(
            "Consolidated clinic and subcontractor records released this week show 29 dockworkers "
            "sustained injuries during the shipyard crane collapse."
        ),
        expected_final_fact="29 dockworkers",
        stale_fact_substring="14 injured",
    ),
    RevisionScenarioTemplate(
        case_id="imp_causal_grid",
        revision_type=RevisionType.ENTITY_ATTRIBUTION,
        event_date_offset_days=3,
        correction_delay_days=15,
        initial_title="Operators point to lightning strike for substation blackout",
        initial_body=(
            "Utility dispatchers attributed the regional substation outage to a severe lightning "
            "strike along the western transmission corridor."
        ),
        corroborating_repeats=2,
        correction_title="Forensic teardown traces substation blackout to corroded relay switch",
        correction_body=(
            "Independent grid inspectors found no storm damage on the towers and determined that "
            "a corroded relay switch inside transformer bay three caused the substation blackout."
        ),
        expected_final_fact="corroded relay switch",
        stale_fact_substring="lightning strike",
    ),
    RevisionScenarioTemplate(
        case_id="imp_causal_cargo",
        revision_type=RevisionType.ENTITY_ATTRIBUTION,
        event_date_offset_days=4,
        correction_delay_days=30,
        initial_title="Coast patrol suspects fuel line leak in freighter engine fire",
        initial_body=(
            "Early harbour briefings pointed toward a ruptured diesel fuel line as the ignition "
            "source aboard the cargo vessel MV Horizon."
        ),
        corroborating_repeats=2,
        correction_title="Marine board links MV Horizon fire to lithium battery cargo",
        correction_body=(
            "Metallurgical testing of the lower hold showed intact fuel pipes and established that "
            "thermal runaway in undocumented lithium battery pallets ignited the MV Horizon fire."
        ),
        expected_final_fact="lithium battery",
        stale_fact_substring="diesel fuel line",
    ),
    RevisionScenarioTemplate(
        case_id="imp_date_quake",
        revision_type=RevisionType.DATE_CORRECTION,
        event_date_offset_days=6,
        correction_delay_days=21,
        initial_title="Seismic fault rupture logged on {initial_date} near coastal ridge",
        initial_body=(
            "Preliminary borehole readings placed the initial deep fault slip along the coastal "
            "ridge on {initial_date}."
        ),
        corroborating_repeats=2,
        correction_title="Ocean-bottom sensors date coastal fault slip to {corrected_date}",
        correction_body=(
            "Synchronized ocean-bottom seismometer logs place the onset of the coastal ridge "
            "fault rupture on {corrected_date} rather than {initial_date}."
        ),
        expected_final_fact="{corrected_date}",
        stale_fact_substring="{initial_date}",
    ),
    RevisionScenarioTemplate(
        case_id="imp_date_treaty",
        revision_type=RevisionType.DATE_CORRECTION,
        event_date_offset_days=8,
        correction_delay_days=35,
        initial_title="Trade delegates sign tariff protocol on {initial_date}",
        initial_body=(
            "Press summaries from Geneva indicated that trade ministers finalized the tariff "
            "protocol signature on {initial_date}."
        ),
        corroborating_repeats=2,
        correction_title="Archival registry records tariff protocol signing on {corrected_date}",
        correction_body=(
            "The official Geneva secretariat deposit shows that ministers signed the final "
            "tariff protocol on {corrected_date}, not {initial_date}."
        ),
        expected_final_fact="{corrected_date}",
        stale_fact_substring="{initial_date}",
    ),
    RevisionScenarioTemplate(
        case_id="imp_retract_bridge",
        revision_type=RevisionType.RUMOR_RETRACTION,
        event_date_offset_days=10,
        correction_delay_days=16,
        initial_title="Motorists report structural crack on East River suspension bridge",
        initial_body=(
            "Commuter calls to local radio described a visible steel girder crack on the "
            "East River suspension bridge following morning tremors."
        ),
        corroborating_repeats=2,
        correction_title="Ultrasonic scan finds East River bridge girders completely intact",
        correction_body=(
            "State structural engineers completed a full ultrasonic inspection of the East River "
            "bridge and confirmed no girder cracks or structural damage exist; the roadway never fractured."
        ),
        expected_final_fact="",
        stale_fact_substring="girder crack",
        should_be_retracted=True,
    ),
    RevisionScenarioTemplate(
        case_id="imp_retract_vaccine",
        revision_type=RevisionType.RUMOR_RETRACTION,
        event_date_offset_days=12,
        correction_delay_days=28,
        initial_title="Clinics halt batch K-19 over suspected cold-chain spoilage",
        initial_body=(
            "Regional clinics briefly quarantined vaccine batch K-19 amid fears that a warehouse "
            "freezer failure had spoiled the entire shipment."
        ),
        corroborating_repeats=2,
        correction_title="Thermometer logs confirm batch K-19 remained within cold-chain limits",
        correction_body=(
            "Independent laboratory potency assays and continuous warehouse telemetry showed that "
            "batch K-19 never warmed above threshold and no spoilage occurred."
        ),
        expected_final_fact="",
        stale_fact_substring="cold-chain spoilage",
        should_be_retracted=True,
    ),
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return [round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)]


def load_articles(stream_path: Path) -> list[Article]:
    with stream_path.open("r", encoding="utf-8") as f:
        return [Article.model_validate_json(line) for line in f if line.strip()]


def build_ablation_factories(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> list[tuple[str, Any]]:
    llm_7b = LLMClient(cfg_single.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b

    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None

    cfg_no_gate = cfg_dual.model_copy(deep=True)
    cfg_no_gate.retrieval.strict_temporal_gate = False
    cfg_no_gate.retrieval.max_date_window_days = 90

    return [
        ("Full Dual-Scale", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o 14B Verifier", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_7b)),
        ("w/o Cross-Encoder", lambda: ReviTLSPipeline(cfg_no_ce, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o Conditional Gate", lambda: ReviTLSPipeline(cfg_no_gate, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o Tombstone Purge", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, disable_purge=True)),
        (
            "LLM-TLS + 90d + Latest + 14B",
            lambda: TopicLLMTLSBaseline(
                llm=llm_7b,
                embedder=embedder,
                window_days=90,
                selection_mode="latest",
                post_hoc_verifier_llm=llm_14b,
            ),
        ),
    ]


def build_rho_factories(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> list[tuple[str, Any]]:
    llm_7b = LLMClient(cfg_dual.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    return [
        ("rho_both (LLM or Lex)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="both")),
        ("rho_llm_only", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="llm")),
        ("rho_lexicon_only", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="lexicon")),
        ("rho_none (Always Same-Day)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="none")),
    ]


def eval_revision_factories(
    factories: list[tuple[str, Any]],
    processed_dir: Path,
    include_embedded: bool = True,
) -> dict[str, Any]:
    out: dict[str, dict[str, Any]] = {"isolated": {}, "embedded": {}, "parametric": {}}
    # 1. Multi-topic 16 scenarios
    for topic, templates in tqdm(TOPIC_SCENARIO_SUITES.items(), desc="  Topics (16-suite)", unit="topic", leave=False):
        stream_path = processed_dir / f"{topic}.jsonl"
        if not stream_path.exists():
            continue
        base_articles = load_articles(stream_path)[:150]
        anchor = base_articles[0].publish_date if base_articles else date(2010, 4, 20)
        iso_arts, iso_cases = inject_revision_scenarios([], topic=topic, anchor_date=anchor, templates=templates)
        emb_arts, emb_cases = inject_revision_scenarios(base_articles, topic=topic, anchor_date=anchor, templates=templates)

        for name, factory in factories:
            _, iso_states = factory().run_stream(iso_arts, num_dates=10)
            iso_sc = evaluate_revision_states(iso_states, iso_cases)
            out["isolated"].setdefault(name, {})[topic] = {
                "rra": iso_sc["revision_resolution_accuracy"],
                "sflr": iso_sc["stale_fact_leakage_rate"],
            }
            if include_embedded:
                _, emb_states = factory().run_stream(emb_arts, num_dates=25)
                emb_sc = evaluate_revision_states(emb_states, emb_cases)
                out["embedded"].setdefault(name, {})[topic] = {
                    "rra": emb_sc["revision_resolution_accuracy"],
                    "sflr": emb_sc["stale_fact_leakage_rate"],
                }

    # 2. Parametric Control 8 scenarios
    pc_arts, pc_cases = inject_revision_scenarios(
        [], topic="counterfactual_control", anchor_date=date(2024, 3, 1), templates=COUNTERFACTUAL_SCENARIO_TEMPLATES
    )
    for name, factory in tqdm(factories, desc="  Parametric (8-suite)", unit="sys", leave=False):
        _, pc_states = factory().run_stream(pc_arts, num_dates=12)
        pc_sc = evaluate_revision_states(pc_states, pc_cases)
        n_pc = len(pc_cases)
        rra_k = round(pc_sc["revision_resolution_accuracy"] * n_pc)
        sflr_k = round(pc_sc["stale_fact_leakage_rate"] * n_pc)
        out["parametric"][name] = {
            "rra": round(rra_k / max(1, n_pc), 4),
            "rra_ci95": wilson_ci(rra_k, n_pc),
            "sflr": round(sflr_k / max(1, n_pc), 4),
            "sflr_ci95": wilson_ci(sflr_k, n_pc),
        }

    # Macro summaries with 95% Wilson CIs
    macro: dict[str, dict[str, Any]] = {}
    for mode in ("isolated", "embedded"):
        if not out[mode]:
            continue
        macro[mode] = {}
        for name, topic_map in out[mode].items():
            n_tot = len(topic_map) * 4
            rra_k = round(sum(v["rra"] * 4 for v in topic_map.values()))
            sflr_k = round(sum(v["sflr"] * 4 for v in topic_map.values()))
            macro[mode][name] = {
                "rra": round(rra_k / max(1, n_tot), 4),
                "rra_ci95": wilson_ci(rra_k, n_tot),
                "sflr": round(sflr_k / max(1, n_tot), 4),
                "sflr_ci95": wilson_ci(sflr_k, n_tot),
            }
    out["macro_with_ci"] = macro
    return out


def run_implicit_long_delay(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    llm_7b = LLMClient(cfg_single.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    systems: list[tuple[str, Any]] = [
        ("LLM-TLS (7B)", TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder)),
        (
            "LLM-TLS + 90d + Latest + 14B",
            TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest", post_hoc_verifier_llm=llm_14b),
        ),
        ("ReviTLS (rho_llm_only)", ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="llm")),
        ("ReviTLS (Full Dual-Scale)", ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, rho_mode="both")),
    ]
    arts, cases = inject_revision_scenarios(
        [], topic="implicit_long_delay", anchor_date=date(2024, 5, 1), templates=IMPLICIT_LONG_DELAY_TEMPLATES
    )
    n_cases = len(cases)
    results: dict[str, Any] = {"num_cases": n_cases, "delay_range_days": [15, 35], "systems": {}}
    for name, runner in tqdm(systems, desc="  Implicit/Long-Delay", unit="sys", leave=False):
        _, states = runner.run_stream(arts, num_dates=12)
        sc = evaluate_revision_states(states, cases)
        rra_k = round(sc["revision_resolution_accuracy"] * n_cases)
        sflr_k = round(sc["stale_fact_leakage_rate"] * n_cases)
        results["systems"][name] = {
            "rra": round(rra_k / max(1, n_cases), 4),
            "rra_ci95": wilson_ci(rra_k, n_cases),
            "sflr": round(sflr_k / max(1, n_cases), 4),
            "sflr_ci95": wilson_ci(sflr_k, n_cases),
        }
    return results


def write_markdown_report(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = ["# Step-B Experimental Report (RRA/SFLR Ablation, Rho Source, Implicit/Long-Delay)\n"]
    ab = summary.get("component_ablation_rra", {})
    if ab:
        lines.append("## 1. Component Ablation on Revision & Parametric Suites\n")
        lines.append("| Variant | Isolated RRA [95% CI] | Isolated SFLR | Embedded RRA [95% CI] | Embedded SFLR | Parametric RRA [95% CI] | Parametric SFLR |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        macro = ab.get("macro_with_ci", {})
        pc = ab.get("parametric", {})
        for name in macro.get("isolated", {}).keys():
            iso = macro["isolated"][name]
            emb = macro.get("embedded", {}).get(name, {})
            p = pc.get(name, {})
            lines.append(
                f"| **{name}** | {iso['rra']:.3f} {iso['rra_ci95']} | {iso['sflr']:.3f} | "
                f"{emb.get('rra', 0):.3f} {emb.get('rra_ci95', [])} | {emb.get('sflr', 0):.3f} | "
                f"{p.get('rra', 0):.3f} {p.get('rra_ci95', [])} | {p.get('sflr', 0):.3f} |"
            )
        lines.append("")

    rho = summary.get("rho_ablation", {})
    if rho:
        lines.append("## 2. Conditional Gate (rho) Source Ablation\n")
        lines.append("| Rho Mode | Isolated RRA [95% CI] | Isolated SFLR | Parametric RRA [95% CI] | Parametric SFLR |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        macro = rho.get("macro_with_ci", {})
        pc = rho.get("parametric", {})
        for name in macro.get("isolated", {}).keys():
            iso = macro["isolated"][name]
            p = pc.get(name, {})
            lines.append(
                f"| **{name}** | {iso['rra']:.3f} {iso['rra_ci95']} | {iso['sflr']:.3f} | "
                f"{p.get('rra', 0):.3f} {p.get('rra_ci95', [])} | {p.get('sflr', 0):.3f} |"
            )
        lines.append("")

    imp = summary.get("implicit_long_delay", {})
    if imp:
        lines.append("## 3. Implicit-Cue & Long-Delay (15-35 Days, 8 Scenarios)\n")
        lines.append("| System | RRA [95% CI] | SFLR [95% CI] |")
        lines.append("| :--- | :---: | :---: |")
        for name, v in imp.get("systems", {}).items():
            lines.append(f"| **{name}** | {v['rra']:.3f} {v['rra_ci95']} | {v['sflr']:.3f} {v['sflr_ci95']} |")
        lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast Step-B ablation & implicit-cue suite.")
    parser.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "step_b_summary.json"
    out_md = args.output_dir / "step_b_report.md"

    if out_json.exists() and not args.fresh:
        summary: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
        print(f"[Resume] Loaded {out_json}", flush=True)
    else:
        summary = {}

    t17_proc = Path("data/processed/t17")
    if "component_ablation_rra" not in summary or args.fresh:
        print("[1/3] Running Component Ablation on RRA/SFLR + Fair Gate+14B Baseline...", flush=True)
        ab_factories = build_ablation_factories(cfg_single, cfg_dual, embedder)
        summary["component_ablation_rra"] = eval_revision_factories(ab_factories, t17_proc, include_embedded=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "rho_ablation" not in summary or args.fresh:
        print("[2/3] Running Conditional Gate (rho) Source Ablation...", flush=True)
        rho_factories = build_rho_factories(cfg_dual, embedder)
        summary["rho_ablation"] = eval_revision_factories(rho_factories, t17_proc, include_embedded=False)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "implicit_long_delay" not in summary or args.fresh:
        print("[3/3] Running 8 Implicit-Cue & Long-Delay (15-35d) Scenarios...", flush=True)
        summary["implicit_long_delay"] = run_implicit_long_delay(cfg_single, cfg_dual, embedder)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown_report(summary, out_md)
    print(f"Done! Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
