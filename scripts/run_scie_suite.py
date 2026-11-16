"""One-command SCI-E experimental suite for ReviTLS.

Runs:
1. Full Multi-Topic TLS evaluation on T17 (9 topics) and Crisis (4 topics) with mean +/- std & paired tests.
2. Multi-Topic Revision Suite (16 scenarios across bpoil, finan, h1n1, haiti) in both
   (a) isolated and (b) stream-embedded modes.
3. Parametric Memory / Data Contamination Control (8 fictional & counterfactual scenarios)
   including a pre-correction Temporal Blindness audit.

Usage on GPU machine (with Ollama running):
    python scripts/run_scie_suite.py --datasets t17 crisis
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

from revitls.baselines.chronos import ClosedDomainCHRONOSBaseline
from revitls.baselines.llm_tls import TopicLLMTLSBaseline
from revitls.config import ReviTLSConfig, load_config
from revitls.evaluation.revision_metrics import evaluate_revision_states
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


def load_articles(stream_path: Path) -> list[Article]:
    with stream_path.open("r", encoding="utf-8") as f:
        return [Article.model_validate_json(line) for line in f if line.strip()]


def sample_topic_articles(
    articles: list[Article],
    refs: list,
    max_articles: int = 80,
) -> list[Article]:
    """Select a reference-date-stratified chronological stream of up to max_articles."""
    if max_articles <= 0 or len(articles) <= max_articles:
        return articles

    ref_dates = {d for ref in refs for d, _ in ref}
    by_date: dict[date, list[Article]] = {}
    for art in articles:
        by_date.setdefault(art.publish_date, []).append(art)

    selected_ids: set[str] = set()
    selected: list[Article] = []

    # Step 1: Ensure every ground-truth reference date present in the stream is covered
    for d in sorted(ref_dates):
        for art in by_date.get(d, [])[:2]:
            if len(selected) >= max_articles:
                break
            if art.article_id not in selected_ids:
                selected_ids.add(art.article_id)
                selected.append(art)

    # Step 2: Fill remaining budget uniformly across the chronological stream
    remaining = [a for a in articles if a.article_id not in selected_ids]
    needed = max_articles - len(selected)
    if needed > 0 and remaining:
        if len(remaining) <= needed:
            selected.extend(remaining)
        else:
            step = len(remaining) / needed
            for i in range(needed):
                selected.append(remaining[min(len(remaining) - 1, int(i * step))])

    selected.sort(key=lambda a: (a.publish_date, a.article_id))
    return selected


def wilson_ci(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% Wilson score confidence interval for binomial proportions."""
    if n <= 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)) / denom
    return [round(max(0.0, center - half), 3), round(min(1.0, center + half), 3)]


def paired_stats(a: list[float], b: list[float]) -> dict[str, float | None]:
    """Compute paired t-test p-value, Wilcoxon p-value, and Hedges' g effect size."""
    if len(a) != len(b) or len(a) < 2:
        return {"t_p": None, "wilcoxon_p": None, "hedges_g": None}
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0.0:
        t_p = 1.0 if mean_d == 0.0 else 0.0
        g_val = 0.0
    else:
        t_stat = abs(mean_d) / (std_d / math.sqrt(n))
        j_corr = 1.0 - (3.0 / (4.0 * n - 1.0)) if n > 1 else 1.0
        g_val = (mean_d / std_d) * j_corr
        try:
            from scipy import stats  # type: ignore

            t_p = float(stats.ttest_rel(a, b).pvalue)
        except Exception:
            t_p = float(math.erfc(t_stat / math.sqrt(2.0)))

    try:
        from scipy import stats  # type: ignore

        non_zero = [d for d in diffs if abs(d) > 1e-12]
        w_p = float(stats.wilcoxon(non_zero).pvalue) if len(non_zero) >= 1 else 1.0
    except Exception:
        w_p = None
    return {"t_p": t_p, "wilcoxon_p": w_p, "hedges_g": round(g_val, 3)}


def apply_holm_bonferroni(raw_pvals: list[tuple[str, str, float | None]]) -> dict[tuple[str, str], float | None]:
    """Adjust a family of p-values using Holm-Bonferroni step-down correction."""
    valid = [(k1, k2, p) for k1, k2, p in raw_pvals if p is not None]
    valid.sort(key=lambda x: x[2])  # type: ignore[arg-type]
    m = len(valid)
    adjusted: dict[tuple[str, str], float | None] = {(k1, k2): None for k1, k2, _ in raw_pvals}
    running_max = 0.0
    for idx, (k1, k2, p) in enumerate(valid):
        adj = min(1.0, p * (m - idx))  # type: ignore[operator]
        running_max = max(running_max, adj)
        adjusted[(k1, k2)] = running_max
    return adjusted


def build_systems(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    include_extended: bool = True,
) -> list[tuple[str, Any]]:
    llm_7b = LLMClient(cfg_single.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    systems: list[tuple[str, Any]] = [
        ("LLM-TLS (7B)", TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder)),
        ("CHRONOS (7B)", ClosedDomainCHRONOSBaseline(llm=llm_7b, embedder=embedder)),
        ("ReviTLS (Single-Scale)", ReviTLSPipeline(cfg_single, llm=llm_7b, embedder=embedder)),
        ("ReviTLS (Dual-Scale)", ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
    ]
    if include_extended:
        systems.extend(
            [
                ("LLM-TLS (14B)", TopicLLMTLSBaseline(llm=llm_14b, embedder=embedder)),
                ("CHRONOS (14B)", ClosedDomainCHRONOSBaseline(llm=llm_14b, embedder=embedder)),
                ("LLM-TLS + 90d + Latest", TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest")),
                ("LLM-TLS (Recency-TR)", TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, selection_mode="recency_textrank")),
                ("LLM-TLS + PostHoc-14B", TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder, post_hoc_verifier_llm=llm_14b)),
                ("ReviTLS (TextRank Sel.)", ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, selection_mode="textrank")),
            ]
        )
    return systems


def eval_topic_stream(
    articles: list[Article],
    refs: list,
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    label: str = "topic",
) -> dict[str, dict[str, Any]]:
    target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else cfg_dual.timeline.default_num_dates
    results: dict[str, dict[str, Any]] = {}
    systems = build_systems(cfg_single, cfg_dual, embedder)
    sys_bar = tqdm(systems, desc=f"  [{label}] Systems", unit="sys", leave=False, dynamic_ncols=True)
    for name, runner in sys_bar:
        sys_bar.set_postfix(model=name, arts=len(articles), refresh=True)
        if hasattr(runner, "llm") and hasattr(runner.llm, "virtual_elapsed_bonus"):
            runner.llm.virtual_elapsed_bonus = 0.0
        if hasattr(runner, "extractor") and hasattr(runner.extractor.llm, "virtual_elapsed_bonus"):
            runner.extractor.llm.virtual_elapsed_bonus = 0.0
        if hasattr(runner, "verifier") and hasattr(runner.verifier.llm, "virtual_elapsed_bonus"):
            runner.verifier.llm.virtual_elapsed_bonus = 0.0

        t0 = time.perf_counter()
        timeline, _states = runner.run_stream(articles, num_dates=target_l)
        bonus = 0.0
        if hasattr(runner, "llm") and hasattr(runner.llm, "virtual_elapsed_bonus"):
            bonus += runner.llm.virtual_elapsed_bonus
        elif hasattr(runner, "extractor") and hasattr(runner.extractor.llm, "virtual_elapsed_bonus"):
            bonus += runner.extractor.llm.virtual_elapsed_bonus
            if (
                hasattr(runner, "verifier")
                and runner.verifier.llm is not runner.extractor.llm
                and hasattr(runner.verifier.llm, "virtual_elapsed_bonus")
            ):
                bonus += runner.verifier.llm.virtual_elapsed_bonus
        elapsed = (time.perf_counter() - t0) + bonus

        date_f1 = max(compute_date_f1(timeline, [d for d, _ in ref])["date_f1"] for ref in refs)
        aligns = [compute_align_rouge(timeline, ref) for ref in refs]
        concats = [compute_concat_rouge(timeline, ref) for ref in refs]
        results[name] = {
            "date_f1": date_f1,
            "ar1": max(x["align_r1"] for x in aligns),
            "ar2": max(x["align_r2"] for x in aligns),
            "concat_r1": max(x["concat_r1"] for x in concats),
            "concat_r2": max(x["concat_r2"] for x in concats),
            "time_s": elapsed,
            "num_articles": len(articles),
        }
        if hasattr(runner, "last_diagnostics") and runner.last_diagnostics:
            results[name]["diagnostics"] = runner.last_diagnostics
        tqdm.write(
            f"  [{label}] {name:<24} | Date-F1={date_f1:.3f} | Concat-R1={results[name]['concat_r1']:.3f} "
            f"| AR-1={results[name]['ar1']:.3f} | Time={elapsed:.1f}s"
        )
    return results


def summarize_dataset(per_topic: dict[str, dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    topics = list(per_topic.keys())
    systems = list(per_topic[topics[0]].keys())
    metrics = ("ar1", "ar2", "concat_r1", "concat_r2", "date_f1", "time_s")
    summary: dict[str, dict[str, Any]] = {}
    for sys_name in systems:
        summary[sys_name] = {}
        for m in metrics:
            vals = [float(per_topic[t][sys_name][m]) for t in topics]
            summary[sys_name][m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    # Paired significance: ReviTLS (Dual-Scale) vs baselines with Holm-Bonferroni, Wilcoxon, and Hedges' g
    if "ReviTLS (Dual-Scale)" in systems:
        raw_stats: dict[str, dict[str, dict[str, float | None]]] = {}
        raw_p_list: list[tuple[str, str, float | None]] = []
        bases = [
            b
            for b in (
                "LLM-TLS (7B)",
                "LLM-TLS (14B)",
                "CHRONOS (7B)",
                "CHRONOS (14B)",
                "ReviTLS (Single-Scale)",
                "ReviTLS (TextRank Sel.)",
                "LLM-TLS + 90d + Latest",
                "LLM-TLS + PostHoc-14B",
            )
            if b in systems
        ]
        for base in bases:
            raw_stats[base] = {}
            for m in ("concat_r1", "date_f1", "ar1"):
                st = paired_stats(
                    [float(per_topic[t]["ReviTLS (Dual-Scale)"][m]) for t in topics],
                    [float(per_topic[t][base][m]) for t in topics],
                )
                raw_stats[base][m] = st
                raw_p_list.append((base, m, st["t_p"]))

        adj_map = apply_holm_bonferroni(raw_p_list)
        for base in bases:
            for m in ("concat_r1", "date_f1", "ar1"):
                raw_stats[base][m]["holm_p"] = adj_map.get((base, m))
        summary["_paired_stats_vs_dual"] = raw_stats
    return summary


def run_revision_suite(
    processed_dir: Path,
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    cached_topic_streams: dict[str, list[Article]] | None = None,
    max_articles: int = 80,
) -> dict[str, Any]:
    """Evaluate 16 multi-topic scenarios in both isolated and stream-embedded modes."""
    isolated_res: dict[str, dict[str, dict[str, float]]] = {}
    embedded_res: dict[str, dict[str, dict[str, float]]] = {}

    for topic, templates in TOPIC_SCENARIO_SUITES.items():
        stream_path = processed_dir / f"{topic}.jsonl"
        if not stream_path.exists():
            continue
        if cached_topic_streams and topic in cached_topic_streams:
            base_articles = cached_topic_streams[topic][: (max_articles if max_articles > 0 else 150)]
        else:
            raw_arts = load_articles(stream_path)
            base_articles = raw_arts[:max_articles] if max_articles > 0 else raw_arts[:150]
        anchor = base_articles[0].publish_date if base_articles else date(2010, 4, 20)

        # (a) Isolated revision suite (only the revision chains, no background noise)
        iso_articles, iso_cases = inject_revision_scenarios([], topic=topic, anchor_date=anchor, templates=templates)
        isolated_res[topic] = {}
        for name, runner in build_systems(cfg_single, cfg_dual, embedder):
            _, states = runner.run_stream(iso_articles, num_dates=10)
            sc = evaluate_revision_states(states, iso_cases)
            isolated_res[topic][name] = {
                "rra": sc["revision_resolution_accuracy"],
                "sflr": sc["stale_fact_leakage_rate"],
            }

        # (b) Embedded in noisy real stream
        emb_articles, emb_cases = inject_revision_scenarios(
            base_articles, topic=topic, anchor_date=anchor, templates=templates
        )
        embedded_res[topic] = {}
        for name, runner in build_systems(cfg_single, cfg_dual, embedder):
            _, states = runner.run_stream(emb_articles, num_dates=25)
            sc = evaluate_revision_states(states, emb_cases)
            embedded_res[topic][name] = {
                "rra": sc["revision_resolution_accuracy"],
                "sflr": sc["stale_fact_leakage_rate"],
            }

    # Compute macro totals + 95% Wilson CIs
    macro: dict[str, dict[str, Any]] = {"isolated": {}, "embedded": {}}
    for mode_name, mode_res in (("isolated", isolated_res), ("embedded", embedded_res)):
        if not mode_res:
            continue
        sys_names = list(next(iter(mode_res.values())).keys())
        for s_name in sys_names:
            n_total = len(mode_res) * 4
            rra_k = round(sum(mode_res[t][s_name]["rra"] * 4 for t in mode_res))
            sflr_k = round(sum(mode_res[t][s_name]["sflr"] * 4 for t in mode_res))
            macro[mode_name][s_name] = {
                "rra": round(rra_k / max(1, n_total), 4),
                "rra_ci95": wilson_ci(rra_k, n_total),
                "sflr": round(sflr_k / max(1, n_total), 4),
                "sflr_ci95": wilson_ci(sflr_k, n_total),
            }

    return {"isolated": isolated_res, "embedded": embedded_res, "macro_with_ci": macro}


def run_parametric_memory_control(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """Madde 4: Fictional + Counterfactual revision suite & pre-correction Temporal Blindness audit."""
    anchor = date(2024, 3, 1)
    articles, cases = inject_revision_scenarios(
        [], topic="counterfactual_control", anchor_date=anchor, templates=COUNTERFACTUAL_SCENARIO_TEMPLATES
    )

    # 1. Temporal Blindness Audit: run only pre-correction articles (*_init_*)
    pre_articles = [a for a in articles if "_init_" in a.article_id]
    dual_cfg = cfg_dual
    llm = LLMClient(dual_cfg.llm)
    verifier_llm = LLMClient(dual_cfg.verifier_llm) if dual_cfg.verifier_llm else None
    blind_runner = ReviTLSPipeline(dual_cfg, llm=llm, embedder=embedder, verifier_llm=verifier_llm)
    _, pre_states = blind_runner.run_stream(pre_articles, num_dates=12)
    pre_text = " ".join(s.canonical_summary.lower() for s in pre_states)
    premature_leaks = sum(
        1 for c in cases if c.expected_final_fact and c.expected_final_fact.lower() in pre_text
    )

    # 2. Full counterfactual evaluation across systems
    post_results: dict[str, dict[str, Any]] = {}
    for name, runner in build_systems(cfg_single, cfg_dual, embedder):
        _, states = runner.run_stream(articles, num_dates=12)
        sc = evaluate_revision_states(states, cases)
        n_cases = len(cases)
        rra_k = round(sc["revision_resolution_accuracy"] * n_cases)
        sflr_k = round(sc["stale_fact_leakage_rate"] * n_cases)
        post_results[name] = {
            "rra": sc["revision_resolution_accuracy"],
            "rra_ci95": wilson_ci(rra_k, n_cases),
            "sflr": sc["stale_fact_leakage_rate"],
            "sflr_ci95": wilson_ci(sflr_k, n_cases),
        }

    return {
        "num_cases": len(cases),
        "premature_leak_count": premature_leaks,
        "premature_leak_rate": premature_leaks / max(1, len(cases)),
        "systems": post_results,
    }


def run_t17_component_ablation(
    t17_streams: dict[str, list[Article]],
    raw_t17_dir: Path,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, Any]:
    """Run genuine component ablations across all T17 topics (9 topics)."""
    llm_7b = LLMClient(cfg_dual.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b

    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None

    cfg_no_gate = cfg_dual.model_copy(deep=True)
    cfg_no_gate.retrieval.strict_temporal_gate = False
    cfg_no_gate.retrieval.max_date_window_days = 90

    variants: list[tuple[str, Any]] = [
        ("Full Dual-Scale", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o 14B Verifier", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_7b)),
        ("w/o Cross-Encoder", lambda: ReviTLSPipeline(cfg_no_ce, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o Conditional Gate", lambda: ReviTLSPipeline(cfg_no_gate, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)),
        ("w/o Tombstone Purge", lambda: ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, disable_purge=True)),
    ]

    per_topic: dict[str, dict[str, dict[str, float]]] = {}
    ab_bar = tqdm(sorted(t17_streams.items()), desc="T17 Ablation", unit="topic", dynamic_ncols=True)
    for topic, articles in ab_bar:
        ab_bar.set_postfix(topic=topic, arts=len(articles), refresh=True)
        refs = load_reference_timelines(raw_t17_dir / topic / "timelines.jsonl")
        target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else cfg_dual.timeline.default_num_dates
        per_topic[topic] = {}
        for v_name, factory in tqdm(variants, desc=f"  [{topic}] Variants", unit="var", leave=False, dynamic_ncols=True):
            runner = factory()
            if hasattr(runner, "extractor") and hasattr(runner.extractor.llm, "virtual_elapsed_bonus"):
                runner.extractor.llm.virtual_elapsed_bonus = 0.0
            if hasattr(runner, "verifier") and hasattr(runner.verifier.llm, "virtual_elapsed_bonus"):
                runner.verifier.llm.virtual_elapsed_bonus = 0.0
            t0 = time.perf_counter()
            timeline, _ = runner.run_stream(articles, num_dates=target_l)
            bonus = 0.0
            if hasattr(runner, "extractor") and hasattr(runner.extractor.llm, "virtual_elapsed_bonus"):
                bonus += runner.extractor.llm.virtual_elapsed_bonus
                if (
                    hasattr(runner, "verifier")
                    and runner.verifier.llm is not runner.extractor.llm
                    and hasattr(runner.verifier.llm, "virtual_elapsed_bonus")
                ):
                    bonus += runner.verifier.llm.virtual_elapsed_bonus
            elapsed = (time.perf_counter() - t0) + bonus
            date_f1 = max(compute_date_f1(timeline, [d for d, _ in ref])["date_f1"] for ref in refs)
            aligns = [compute_align_rouge(timeline, ref) for ref in refs]
            concats = [compute_concat_rouge(timeline, ref) for ref in refs]
            per_topic[topic][v_name] = {
                "date_f1": date_f1,
                "ar1": max(x["align_r1"] for x in aligns),
                "concat_r1": max(x["concat_r1"] for x in concats),
                "time_s": elapsed,
            }
            tqdm.write(
                f"  [Ablation/{topic}] {v_name:<22} | Date-F1={date_f1:.3f} | Concat-R1={per_topic[topic][v_name]['concat_r1']:.3f} | Time={elapsed:.1f}s"
            )

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for v_name, _ in variants:
        summary[v_name] = {}
        for m in ("date_f1", "ar1", "concat_r1", "time_s"):
            vals = [per_topic[t][v_name][m] for t in per_topic]
            summary[v_name][m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    return {"per_topic": per_topic, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SCI-E multi-topic + counterfactual suite.")
    parser.add_argument("--datasets", nargs="+", default=["t17", "crisis"], choices=["t17", "crisis"])
    parser.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--max-articles", type=int, default=0, help="Max reference-stratified articles per topic (0 = full corpus)")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing summary.json and start from scratch")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    args = parser.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "summary.json"
    if out_json.exists() and not args.fresh:
        try:
            all_output: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"[Resume] Loaded existing checkpoint from {out_json}", flush=True)
        except Exception:
            all_output = {"datasets": {}, "revision_suite": {}, "parametric_control": {}}
    else:
        all_output = {"datasets": {}, "revision_suite": {}, "parametric_control": {}}
    all_output.setdefault("datasets", {})
    t17_sampled_streams: dict[str, list[Article]] = {}

    topic_tasks: list[tuple[str, Path, Path, Path]] = []
    for ds in args.datasets:
        proc_dir = Path("data/processed") / ds
        raw_dir = Path("data/raw") / ds
        if not proc_dir.exists() or not any(proc_dir.glob("*.jsonl")):
            from scripts.prepare_datasets import convert_dataset_root  # type: ignore

            convert_dataset_root(raw_dir, proc_dir)
        for stream_file in sorted(proc_dir.glob("*.jsonl")):
            topic_tasks.append((ds, proc_dir, raw_dir, stream_file))

    overall_bar = tqdm(topic_tasks, desc="Overall Topics", unit="topic", dynamic_ncols=True)
    for ds, _proc_dir, raw_dir, stream_file in overall_bar:
        topic = stream_file.stem
        per_topic: dict[str, dict[str, dict[str, Any]]] = all_output["datasets"].get(ds, {}).get("per_topic", {})
        tl_file = raw_dir / topic / "timelines.jsonl"
        refs = load_reference_timelines(tl_file)
        articles = load_articles(stream_file)
        all_ref_entries = [item for ref in refs for item in ref]
        articles = filter_articles_by_timespan(articles, all_ref_entries)
        articles = sample_topic_articles(articles, refs, max_articles=args.max_articles)
        if ds == "t17":
            t17_sampled_streams[topic] = articles
        overall_bar.set_postfix(topic=f"{ds}/{topic}", arts=len(articles), refresh=True)
        if topic in per_topic and "ReviTLS (Dual-Scale)" in per_topic[topic] and "LLM-TLS (14B)" in per_topic[topic]:
            tqdm.write(f"[{ds}/{topic}] Skipping completed topic (checkpoint found).")
            continue
        tqdm.write(f"\n[{ds}/{topic}] Evaluating {len(articles)} articles across 10 systems...")
        per_topic[topic] = eval_topic_stream(articles, refs, cfg_single, cfg_dual, embedder, label=f"{ds}/{topic}")
        dual_sc = per_topic[topic]["ReviTLS (Dual-Scale)"]
        tqdm.write(
            f"  => [{ds}/{topic}] DONE | Dual-Scale: Date-F1={dual_sc['date_f1']:.3f}, "
            f"Concat-R1={dual_sc['concat_r1']:.3f}, AR-1={dual_sc['ar1']:.3f}"
        )
        all_output["datasets"][ds] = {
            "per_topic": per_topic,
            "summary": summarize_dataset(per_topic),
        }
        out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")

    t17_proc = Path("data/processed/t17")
    if t17_proc.exists():
        if all_output.get("revision_suite") and not args.fresh:
            print("[Revision Suite] Skipping completed revision_suite (checkpoint found).", flush=True)
        else:
            print("[Revision Suite] Running 16 multi-topic scenarios (isolated + embedded)...", flush=True)
            all_output["revision_suite"] = run_revision_suite(
                t17_proc,
                cfg_single,
                cfg_dual,
                embedder,
                cached_topic_streams=t17_sampled_streams,
                max_articles=args.max_articles,
            )
            out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")

    if all_output.get("parametric_control") and not args.fresh:
        print("[Parametric Memory Control] Skipping completed parametric_control (checkpoint found).", flush=True)
    else:
        print("[Parametric Memory Control] Running 8 counterfactual/fictional scenarios...", flush=True)
        all_output["parametric_control"] = run_parametric_memory_control(cfg_single, cfg_dual, embedder)
        out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")

    if t17_sampled_streams:
        print("[T17 Component Ablation] Running 5 variants across all 9 T17 topics...", flush=True)
        all_output["t17_ablation"] = run_t17_component_ablation(
            t17_sampled_streams,
            Path("data/raw/t17"),
            cfg_dual,
            embedder,
        )

    out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")
    print(f"All SCI-E results written to {out_json}", flush=True)


if __name__ == "__main__":
    main()
