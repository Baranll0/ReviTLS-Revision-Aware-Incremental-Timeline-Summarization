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


def paired_t_pvalue(a: list[float], b: list[float]) -> float | None:
    """Compute two-sided paired t-test p-value without hard dependency on scipy."""
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0.0:
        return 1.0 if mean_d == 0.0 else 0.0
    t_stat = abs(mean_d) / (std_d / math.sqrt(len(diffs)))
    try:
        from scipy import stats  # type: ignore

        return float(stats.ttest_rel(a, b).pvalue)
    except Exception:
        # Normal approximation fallback for quick summary when scipy is absent
        return float(math.erfc(t_stat / math.sqrt(2.0)))


def build_systems(
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> list[tuple[str, Any]]:
    llm = LLMClient(cfg_single.llm)
    verifier_llm = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else None
    return [
        ("LLM-TLS", TopicLLMTLSBaseline(llm=llm, embedder=embedder)),
        ("CHRONOS", ClosedDomainCHRONOSBaseline(llm=llm, embedder=embedder)),
        ("ReviTLS (Single-Scale)", ReviTLSPipeline(cfg_single, llm=llm, embedder=embedder)),
        ("ReviTLS (Dual-Scale)", ReviTLSPipeline(cfg_dual, llm=llm, embedder=embedder, verifier_llm=verifier_llm)),
    ]


def eval_topic_stream(
    articles: list[Article],
    refs: list,
    cfg_single: ReviTLSConfig,
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
) -> dict[str, dict[str, float]]:
    target_l = max(1, round(sum(len(r) for r in refs) / len(refs))) if refs else cfg_dual.timeline.default_num_dates
    results: dict[str, dict[str, float]] = {}
    for name, runner in build_systems(cfg_single, cfg_dual, embedder):
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
        }
    return results


def summarize_dataset(per_topic: dict[str, dict[str, dict[str, float]]]) -> dict[str, dict[str, Any]]:
    topics = list(per_topic.keys())
    systems = list(per_topic[topics[0]].keys())
    metrics = ("ar1", "ar2", "concat_r1", "concat_r2", "date_f1", "time_s")
    summary: dict[str, dict[str, Any]] = {}
    for sys_name in systems:
        summary[sys_name] = {}
        for m in metrics:
            vals = [per_topic[t][sys_name][m] for t in topics]
            summary[sys_name][m] = {
                "mean": statistics.mean(vals),
                "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            }
    # Paired significance: ReviTLS (Dual-Scale) vs LLM-TLS & CHRONOS
    if "ReviTLS (Dual-Scale)" in systems:
        pvals: dict[str, dict[str, float | None]] = {}
        for base in ("LLM-TLS", "CHRONOS", "ReviTLS (Single-Scale)"):
            if base not in systems:
                continue
            pvals[base] = {
                m: paired_t_pvalue(
                    [per_topic[t]["ReviTLS (Dual-Scale)"][m] for t in topics],
                    [per_topic[t][base][m] for t in topics],
                )
                for m in ("concat_r1", "date_f1", "ar1")
            }
        summary["_paired_pvalues_vs_dual"] = pvals
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
            base_articles = cached_topic_streams[topic]
        else:
            base_articles = load_articles(stream_path)[:max_articles]
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

    return {"isolated": isolated_res, "embedded": embedded_res}


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
    post_results: dict[str, dict[str, float]] = {}
    for name, runner in build_systems(cfg_single, cfg_dual, embedder):
        _, states = runner.run_stream(articles, num_dates=12)
        sc = evaluate_revision_states(states, cases)
        post_results[name] = {
            "rra": sc["revision_resolution_accuracy"],
            "sflr": sc["stale_fact_leakage_rate"],
        }

    return {
        "num_cases": len(cases),
        "premature_leak_count": premature_leaks,
        "premature_leak_rate": premature_leaks / max(1, len(cases)),
        "systems": post_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full SCI-E multi-topic + counterfactual suite.")
    parser.add_argument("--datasets", nargs="+", default=["t17", "crisis"], choices=["t17", "crisis"])
    parser.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--max-articles", type=int, default=50, help="Max reference-stratified articles per topic")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    args = parser.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "summary.json"
    all_output: dict[str, Any] = {"datasets": {}, "revision_suite": {}, "parametric_control": {}}
    t17_sampled_streams: dict[str, list[Article]] = {}

    for ds in args.datasets:
        proc_dir = Path("data/processed") / ds
        raw_dir = Path("data/raw") / ds
        if not proc_dir.exists() or not any(proc_dir.glob("*.jsonl")):
            from scripts.prepare_datasets import convert_dataset_root  # type: ignore

            convert_dataset_root(raw_dir, proc_dir)

        per_topic: dict[str, dict[str, dict[str, float]]] = {}
        for stream_file in sorted(proc_dir.glob("*.jsonl")):
            topic = stream_file.stem
            tl_file = raw_dir / topic / "timelines.jsonl"
            refs = load_reference_timelines(tl_file)
            articles = load_articles(stream_file)
            all_ref_entries = [item for ref in refs for item in ref]
            articles = filter_articles_by_timespan(articles, all_ref_entries)
            articles = sample_topic_articles(articles, refs, max_articles=args.max_articles)
            if ds == "t17":
                t17_sampled_streams[topic] = articles
            print(f"[{ds}/{topic}] Evaluating {len(articles)} articles...", flush=True)
            per_topic[topic] = eval_topic_stream(articles, refs, cfg_single, cfg_dual, embedder)
            dual_sc = per_topic[topic]["ReviTLS (Dual-Scale)"]
            print(
                f"  -> [{ds}/{topic}] Dual-Scale: Date-F1={dual_sc['date_f1']:.3f}, "
                f"Concat-R1={dual_sc['concat_r1']:.3f}, AR-1={dual_sc['ar1']:.3f}",
                flush=True,
            )
            all_output["datasets"][ds] = {
                "per_topic": per_topic,
                "summary": summarize_dataset(per_topic),
            }
            out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")

    t17_proc = Path("data/processed/t17")
    if t17_proc.exists():
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

    print("[Parametric Memory Control] Running 8 counterfactual/fictional scenarios...", flush=True)
    all_output["parametric_control"] = run_parametric_memory_control(cfg_single, cfg_dual, embedder)

    out_json.write_text(json.dumps(all_output, indent=2), encoding="utf-8")
    print(f"All SCI-E results written to {out_json}", flush=True)


if __name__ == "__main__":
    main()
