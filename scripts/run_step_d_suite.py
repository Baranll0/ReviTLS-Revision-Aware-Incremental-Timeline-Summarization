"""Step-D comprehensive verification suite (~15-20 min on RTX 5080).

Covers:
1. Uncached 3-repeat revision ablation (use_cache=False) with mean +/- std & exact McNemar test,
   including Overwrite (No P_k) baseline.
2. True uncached latency (Online Immediate vs Deferred Phase-2, use_cache=False) + Cross-Encoder Recall@1.
3. Crisis & T17 fragmentation diagnosis (LLM-TLS vs ReviTLS state counts, c_k, nodes/ref_date) +
   isolated Guarded Retract + Disable Retract across all 13 topics (warm cache).
4. bpoil-excluded T17 (n=8) and Revision (12 cases) breakdown from summary.json.

Usage on GPU machine (with Ollama running):
    python scripts/run_step_d_suite.py --fresh
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import statistics
import time
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
    TOPIC_SCENARIO_SUITES,
    inject_revision_scenarios,
)
from revitls.schemas import Article
from scripts.run_step_b_suite import IMPLICIT_LONG_DELAY_TEMPLATES


def exact_mcnemar_p(a_hits: list[int], b_hits: list[int]) -> float | None:
    """Two-sided exact binomial McNemar p-value over paired binary outcomes."""
    if len(a_hits) != len(b_hits) or not a_hits:
        return None
    b = sum(1 for x, y in zip(a_hits, b_hits) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(a_hits, b_hits) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    prob = sum(math.comb(n, i) * (0.5**n) for i in range(0, k + 1))
    return round(min(1.0, 2.0 * prob), 4)


def load_articles(stream_path: Path) -> list[Article]:
    with stream_path.open("r", encoding="utf-8") as f:
        return [Article.model_validate_json(line) for line in f if line.strip()]


def case_binary_hits(states: list, cases: list[RevisionTestCase]) -> list[int]:
    return [1 if evaluate_revision_states(states, [c])["revision_resolution_accuracy"] == 1.0 else 0 for c in cases]


def build_uncached_factories(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> list[tuple[str, Any]]:
    llm_7b = LLMClient(cfg_single.llm, use_cache=False)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=False) if cfg_dual.verifier_llm else llm_7b

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
        ("Overwrite (No P_k)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, overwrite_only=True)),
        ("w/o Tombstone Purge", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, disable_purge=True)),
        ("Guarded Retract Only", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, guarded_retract=True)),
        (
            "LLM-TLS + 90d + Latest + 14B",
            lambda: TopicLLMTLSBaseline(
                llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest", post_hoc_verifier_llm=llm_14b
            ),
        ),
    ]


def run_repeated_revisions(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
    repeats: int = 3,
) -> dict[str, Any]:
    factories = build_uncached_factories(cfg_single, cfg_dual, embedder)
    # Preload streams once
    topic_data: list[tuple[str, list[Article], list[RevisionTestCase], list[Article], list[RevisionTestCase]]] = []
    for topic, templates in TOPIC_SCENARIO_SUITES.items():
        sp = processed_dir / f"{topic}.jsonl"
        if not sp.exists():
            continue
        base = load_articles(sp)[:150]
        anchor = base[0].publish_date if base else date(2010, 4, 20)
        iso_a, iso_c = inject_revision_scenarios([], topic=topic, anchor_date=anchor, templates=templates)
        emb_a, emb_c = inject_revision_scenarios(base, topic=topic, anchor_date=anchor, templates=templates)
        topic_data.append((topic, iso_a, iso_c, emb_a, emb_c))

    pc_a, pc_c = inject_revision_scenarios(
        [], topic="counterfactual_control", anchor_date=date(2024, 3, 1), templates=COUNTERFACTUAL_SCENARIO_TEMPLATES
    )
    imp_a, imp_c = inject_revision_scenarios(
        [], topic="implicit_long_delay", anchor_date=date(2024, 5, 1), templates=IMPLICIT_LONG_DELAY_TEMPLATES
    )

    suites = ("isolated_16", "embedded_16", "parametric_8", "implicit_8")
    raw_runs: dict[str, dict[str, list[float]]] = {name: {f"{s}_rra": [] for s in suites} for name, _ in factories}
    for name in raw_runs:
        for s in suites:
            raw_runs[name][f"{s}_sflr"] = []
    pooled_hits: dict[str, dict[str, list[int]]] = {name: {s: [] for s in suites} for name, _ in factories}

    for rep in tqdm(range(repeats), desc="  Repeated Uncached Runs", unit="rep"):
        for name, factory in tqdm(factories, desc=f"    Rep {rep+1}/{repeats}", unit="sys", leave=False):
            # 1. Isolated & Embedded 16
            iso_hits: list[int] = []
            emb_hits: list[int] = []
            iso_sflr = 0.0
            emb_sflr = 0.0
            for _topic, iso_a, iso_c, emb_a, emb_c in topic_data:
                _, s_iso = factory().run_stream(iso_a, num_dates=10)
                iso_hits.extend(case_binary_hits(s_iso, iso_c))
                iso_sflr += evaluate_revision_states(s_iso, iso_c)["stale_fact_leakage_rate"] * 4
                _, s_emb = factory().run_stream(emb_a, num_dates=25)
                emb_hits.extend(case_binary_hits(s_emb, emb_c))
                emb_sflr += evaluate_revision_states(s_emb, emb_c)["stale_fact_leakage_rate"] * 4
            raw_runs[name]["isolated_16_rra"].append(sum(iso_hits) / max(1, len(iso_hits)))
            raw_runs[name]["isolated_16_sflr"].append(iso_sflr / max(1, len(iso_hits)))
            raw_runs[name]["embedded_16_rra"].append(sum(emb_hits) / max(1, len(emb_hits)))
            raw_runs[name]["embedded_16_sflr"].append(emb_sflr / max(1, len(emb_hits)))
            pooled_hits[name]["isolated_16"].extend(iso_hits)
            pooled_hits[name]["embedded_16"].extend(emb_hits)

            # 2. Parametric 8
            _, s_pc = factory().run_stream(pc_a, num_dates=12)
            pc_h = case_binary_hits(s_pc, pc_c)
            raw_runs[name]["parametric_8_rra"].append(sum(pc_h) / len(pc_h))
            raw_runs[name]["parametric_8_sflr"].append(evaluate_revision_states(s_pc, pc_c)["stale_fact_leakage_rate"])
            pooled_hits[name]["parametric_8"].extend(pc_h)

            # 3. Implicit 8
            _, s_imp = factory().run_stream(imp_a, num_dates=12)
            imp_h = case_binary_hits(s_imp, imp_c)
            raw_runs[name]["implicit_8_rra"].append(sum(imp_h) / len(imp_h))
            raw_runs[name]["implicit_8_sflr"].append(evaluate_revision_states(s_imp, imp_c)["stale_fact_leakage_rate"])
            pooled_hits[name]["implicit_8"].extend(imp_h)

    summary: dict[str, Any] = {}
    dual_hits = pooled_hits["Full Dual-Scale"]
    for name, metrics in raw_runs.items():
        entry: dict[str, Any] = {}
        for k, vals in metrics.items():
            entry[k] = {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
                "runs": [round(v, 4) for v in vals],
            }
        entry["mcnemar_p_vs_dual"] = {
            s: exact_mcnemar_p(dual_hits[s], pooled_hits[name][s]) for s in suites
        }
        summary[name] = entry
    return summary


def run_true_latency_and_ce_recall(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
) -> dict[str, Any]:
    llm_7b = LLMClient(cfg_dual.llm, use_cache=False)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=False) if cfg_dual.verifier_llm else llm_7b
    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None

    configs = [
        ("Deferred Phase-2 (Uncached)", cfg_dual, True),
        ("Online Immediate (Uncached)", cfg_dual, False),
        ("Deferred w/o CE (Uncached)", cfg_no_ce, True),
    ]
    out: dict[str, Any] = {}
    for label, cfg, defer in configs:
        rra_hits = 0
        total_cases = 0
        total_time = 0.0
        mean_lats: list[float] = []
        p95_lats: list[float] = []
        p2_times: list[float] = []
        ce_calls = 0
        ce_flips = 0
        top1_date_hits = 0
        top1_total = 0

        for topic, templates in tqdm(TOPIC_SCENARIO_SUITES.items(), desc=f"  {label}", unit="topic", leave=False):
            sp = processed_dir / f"{topic}.jsonl"
            if not sp.exists():
                continue
            base = load_articles(sp)[:150]
            anchor = base[0].publish_date if base else date(2010, 4, 20)
            emb_a, emb_c = inject_revision_scenarios(base, topic=topic, anchor_date=anchor, templates=templates)
            # Map revision article_id -> expected event_date (anchor + offset)
            target_dates: dict[str, str] = {}
            for t in templates:
                from datetime import timedelta

                evt_dt = (anchor + timedelta(days=t.event_date_offset_days)).isoformat()
                target_dates[f"{t.case_id}_corr"] = evt_dt

            runner = ReviTLSPipeline(cfg, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, defer_phase2=defer)
            t0 = time.perf_counter()
            _, states = runner.run_stream(emb_a, num_dates=25)
            total_time += time.perf_counter() - t0
            rra_hits += sum(case_binary_hits(states, emb_c))
            total_cases += len(emb_c)
            diag = runner.last_diagnostics
            mean_lats.append(float(diag.get("mean_article_latency_s", 0.0)))
            p95_lats.append(float(diag.get("p95_article_latency_s", 0.0)))
            p2_times.append(float(diag.get("phase2_reconciliation_s", 0.0)))
            ce_calls += int(diag.get("ce_reranked_calls", 0))
            ce_flips += int(diag.get("ce_top1_flips", 0))
            for rec in diag.get("rev_top1_records", []):  # type: ignore[union-attr]
                aid = str(rec.get("article_id", ""))
                for k_pref, exp_dt in target_dates.items():
                    if aid.startswith(k_pref):
                        top1_total += 1
                        if rec.get("top1_date") == exp_dt:
                            top1_date_hits += 1

        out[label] = {
            "rra": round(rra_hits / max(1, total_cases), 4),
            "total_time_s": round(total_time, 2),
            "mean_article_latency_s": round(sum(mean_lats) / max(1, len(mean_lats)), 4),
            "p95_article_latency_s": round(sum(p95_lats) / max(1, len(p95_lats)), 4),
            "total_phase2_s": round(sum(p2_times), 2),
            "ce_reranked_calls": ce_calls,
            "ce_top1_flips": ce_flips,
            "rev_recall_at_1": round(top1_date_hits / max(1, top1_total), 4) if top1_total else None,
            "rev_top1_hits": top1_date_hits,
            "rev_top1_total": top1_total,
        }
    return out


def run_fragmentation_and_retract_ablation(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """Use warm deterministic cache (use_cache=True) to extract LLM-TLS vs ReviTLS state counts & retract ablations."""
    llm_7b = LLMClient(cfg_dual.llm, use_cache=True)
    llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=True) if cfg_dual.verifier_llm else llm_7b

    systems: list[tuple[str, Any]] = [
        ("LLM-TLS (7B)", lambda: TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder)),
        ("ReviTLS (Dual-Scale)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("ReviTLS (Guarded Retract)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, guarded_retract=True)),
        ("ReviTLS (Disable Retract)", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, disable_retract=True)),
    ]

    out: dict[str, Any] = {"t17": {}, "crisis": {}}
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
            unique_ref_dates = len({d for ref in refs for d, _ in ref})
            per_topic[topic] = {"num_articles": len(arts), "unique_ref_dates": unique_ref_dates, "systems": {}}
            for s_name, factory in tqdm(systems, desc=f"  [{ds}/{topic}]", unit="sys", leave=False):
                tl, states = factory().run_stream(arts, num_dates=target_l)
                df1 = max(compute_date_f1(tl, [d for d, _ in ref])["date_f1"] for ref in refs)
                ar1 = max(compute_align_rouge(tl, ref)["align_r1"] for ref in refs)
                cr1 = max(compute_concat_rouge(tl, ref)["concat_r1"] for ref in refs)
                mean_ck = sum(s.corroboration_count for s in states) / max(1, len(states))
                ret_cnt = sum(1 for s in states if s.status.value == "RETRACTED")
                per_topic[topic]["systems"][s_name] = {
                    "date_f1": round(df1, 4),
                    "ar1": round(ar1, 4),
                    "concat_r1": round(cr1, 4),
                    "total_states": len(states),
                    "mean_c_k": round(mean_ck, 2),
                    "nodes_per_ref_date": round(len(states) / max(1, unique_ref_dates), 2),
                    "retracted_count": ret_cnt,
                }
        # Aggregate
        agg: dict[str, Any] = {}
        for s_name, _ in systems:
            t_list = list(per_topic.values())
            agg[s_name] = {
                "date_f1_mean": round(statistics.mean(t["systems"][s_name]["date_f1"] for t in t_list), 4),
                "ar1_mean": round(statistics.mean(t["systems"][s_name]["ar1"] for t in t_list), 4),
                "concat_r1_mean": round(statistics.mean(t["systems"][s_name]["concat_r1"] for t in t_list), 4),
                "total_states_sum": sum(t["systems"][s_name]["total_states"] for t in t_list),
                "mean_c_k_avg": round(statistics.mean(t["systems"][s_name]["mean_c_k"] for t in t_list), 2),
                "nodes_per_ref_date_avg": round(statistics.mean(t["systems"][s_name]["nodes_per_ref_date"] for t in t_list), 2),
                "retracted_sum": sum(t["systems"][s_name]["retracted_count"] for t in t_list),
            }
        out[ds] = {"per_topic": per_topic, "summary": agg}
    return out


def compute_bpoil_excluded_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    t17_pt = data.get("datasets", {}).get("t17", {}).get("per_topic", {})
    topics_no_bpoil = [t for t in sorted(t17_pt.keys()) if t != "bpoil"]
    out: dict[str, Any] = {"t17_n8": {}, "revision_n12": {}}
    if topics_no_bpoil:
        for sys_name in ("LLM-TLS (7B)", "LLM-TLS (14B)", "ReviTLS (Dual-Scale)"):
            vals_df = [float(t17_pt[t][sys_name]["date_f1"]) for t in topics_no_bpoil]
            vals_ar = [float(t17_pt[t][sys_name]["ar1"]) for t in topics_no_bpoil]
            vals_cr = [float(t17_pt[t][sys_name]["concat_r1"]) for t in topics_no_bpoil]
            out["t17_n8"][sys_name] = {
                "date_f1": f"{statistics.mean(vals_df):.3f} +/- {statistics.stdev(vals_df):.3f}",
                "ar1": f"{statistics.mean(vals_ar):.3f} +/- {statistics.stdev(vals_ar):.3f}",
                "concat_r1": f"{statistics.mean(vals_cr):.3f} +/- {statistics.stdev(vals_cr):.3f}",
            }
    for mode in ("isolated", "embedded"):
        m_map = data.get("revision_suite", {}).get(mode, {})
        topics_rev = [t for t in ("finan", "h1n1", "haiti") if t in m_map]
        if topics_rev:
            dual_rra = sum(m_map[t]["ReviTLS (Dual-Scale)"]["rra"] * 4 for t in topics_rev) / (len(topics_rev) * 4)
            out["revision_n12"][mode] = round(dual_rra, 4)
    return out


def write_markdown_report(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = ["# Step-D Experimental Report (Multi-Run Stats, True Latency, Fragmentation)\n"]
    rep = summary.get("repeated_revisions", {})
    if rep:
        lines.append("## 1. 3-Run Uncached Revision Ablation (Mean +/- Std & McNemar p vs Dual-Scale)\n")
        lines.append("| System | Isolated-16 RRA | McNemar p | Embedded-16 RRA | McNemar p | Parametric-8 RRA | Implicit-8 RRA |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for name, v in rep.items():
            mc = v.get("mcnemar_p_vs_dual", {})
            lines.append(
                f"| **{name}** | {v['isolated_16_rra']['mean']:.3f}±{v['isolated_16_rra']['std']:.3f} | {mc.get('isolated_16')} | "
                f"{v['embedded_16_rra']['mean']:.3f}±{v['embedded_16_rra']['std']:.3f} | {mc.get('embedded_16')} | "
                f"{v['parametric_8_rra']['mean']:.3f}±{v['parametric_8_rra']['std']:.3f} | "
                f"{v['implicit_8_rra']['mean']:.3f}±{v['implicit_8_rra']['std']:.3f} |"
            )
        lines.append("")

    lat = summary.get("true_latency_and_ce", {})
    if lat:
        lines.append("## 2. True Uncached Latency & Cross-Encoder Recall@1 (16 Embedded)\n")
        lines.append("| Mode | RRA | Total (s) | Mean Art. Lat (s) | P95 Lat (s) | CE Recall@1 | CE Calls/Flips |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k, v in lat.items():
            lines.append(
                f"| **{k}** | {v['rra']:.3f} | {v['total_time_s']:.1f} | {v['mean_article_latency_s']:.3f} | "
                f"{v['p95_article_latency_s']:.3f} | {v['rev_recall_at_1']} ({v['rev_top1_hits']}/{v['rev_top1_total']}) | "
                f"{v['ce_reranked_calls']}/{v['ce_top1_flips']} |"
            )
        lines.append("")

    frag = summary.get("fragmentation_and_retract", {})
    if frag:
        lines.append("## 3. T17 & Crisis Fragmentation + Guarded/Disable Retract\n")
        for ds in ("t17", "crisis"):
            s_map = frag.get(ds, {}).get("summary", {})
            if not s_map:
                continue
            lines.append(f"### {ds.upper()}\n")
            lines.append("| System | Date-F1 | AR-1 | Concat-R1 | Total States | Mean c_k | Nodes/RefDate | Retracted |")
            lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for k, v in s_map.items():
                lines.append(
                    f"| **{k}** | {v['date_f1_mean']:.3f} | {v['ar1_mean']:.3f} | {v['concat_r1_mean']:.3f} | "
                    f"{v['total_states_sum']} | {v['mean_c_k_avg']} | {v['nodes_per_ref_date_avg']} | {v['retracted_sum']} |"
                )
            lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Step-D multi-run & fragmentation suite.")
    parser.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "step_d_summary.json"
    out_md = args.output_dir / "step_d_report.md"

    if out_json.exists() and not args.fresh:
        summary: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
    else:
        summary = {}

    t17_proc = Path("data/processed/t17")
    if "repeated_revisions" not in summary or args.fresh:
        print(f"[1/4] Running {args.repeats}-repeat uncached revision ablation + Overwrite...", flush=True)
        summary["repeated_revisions"] = run_repeated_revisions(cfg_single, cfg_dual, embedder, t17_proc, repeats=args.repeats)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "true_latency_and_ce" not in summary or args.fresh:
        print("[2/4] Measuring true uncached latency (Online vs Deferred) & CE Recall@1...", flush=True)
        summary["true_latency_and_ce"] = run_true_latency_and_ce_recall(cfg_dual, embedder, t17_proc)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "fragmentation_and_retract" not in summary or args.fresh:
        print("[3/4] Running T17+Crisis fragmentation diagnosis & Guarded/Disable Retract...", flush=True)
        summary["fragmentation_and_retract"] = run_fragmentation_and_retract_ablation(cfg_dual, embedder)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["bpoil_excluded"] = compute_bpoil_excluded_summary(args.output_dir / "summary.json")
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(summary, out_md)
    print(f"Done! Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
