"""Fast Step-C GPU suite (~10-15 min on RTX 5080) covering remaining reviewer items:

1. Online Immediate (defer_phase2=False) vs. Deferred (defer_phase2=True) on 16 embedded scenarios:
   measures RRA, SFLR, mean/p95/max per-article latency, total time, and Cross-Encoder flips.
2. Safe Purge & Guarded Retract (purge_mode="claim_span", guarded_retract=True) vs. Default:
   tested on T17/egypt (checking false retraction of 2011-02-11, Date-F1, Concat-R1) and 16 embedded.
3. 90-day Window Sensitivity (7 / 30 / 60 / 90 / 180 days) on 16 embedded + 8 implicit/long-delay (15-35d).

Usage on GPU machine (with Ollama running):
    python scripts/run_step_c_suite.py --fresh
"""

from __future__ import annotations

import argparse
from datetime import date
import json
import math
from pathlib import Path
import time
from typing import Any

from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

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
from revitls.revisions.benchmark import TOPIC_SCENARIO_SUITES, inject_revision_scenarios
from revitls.schemas import Article
from scripts.run_step_b_suite import IMPLICIT_LONG_DELAY_TEMPLATES


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


def run_online_vs_deferred(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
) -> dict[str, Any]:
    llm_7b = LLMClient(cfg_dual.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    modes = [
        ("Deferred Phase-2 (Default)", True),
        ("Online Immediate Phase-2", False),
    ]
    out: dict[str, Any] = {}
    for label, defer in modes:
        rra_sum = 0.0
        sflr_sum = 0.0
        n_topics = 0
        total_time = 0.0
        mean_lats: list[float] = []
        p95_lats: list[float] = []
        ce_calls = 0
        ce_flips = 0
        for topic, templates in tqdm(TOPIC_SCENARIO_SUITES.items(), desc=f"  {label}", unit="topic", leave=False):
            stream_path = processed_dir / f"{topic}.jsonl"
            if not stream_path.exists():
                continue
            base_arts = load_articles(stream_path)[:150]
            anchor = base_arts[0].publish_date if base_arts else date(2010, 4, 20)
            emb_arts, emb_cases = inject_revision_scenarios(base_arts, topic=topic, anchor_date=anchor, templates=templates)
            runner = ReviTLSPipeline(cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, defer_phase2=defer)
            t0 = time.perf_counter()
            _, states = runner.run_stream(emb_arts, num_dates=25)
            total_time += time.perf_counter() - t0
            sc = evaluate_revision_states(states, emb_cases)
            rra_sum += sc["revision_resolution_accuracy"] * 4
            sflr_sum += sc["stale_fact_leakage_rate"] * 4
            n_topics += 1
            diag = runner.last_diagnostics
            mean_lats.append(float(diag.get("mean_article_latency_s", 0.0)))
            p95_lats.append(float(diag.get("p95_article_latency_s", 0.0)))
            ce_calls += int(diag.get("ce_reranked_calls", 0))
            ce_flips += int(diag.get("ce_top1_flips", 0))
        n_tot = max(1, n_topics * 4)
        rra_k = round(rra_sum)
        sflr_k = round(sflr_sum)
        out[label] = {
            "rra": round(rra_k / n_tot, 4),
            "rra_ci95": wilson_ci(rra_k, n_tot),
            "sflr": round(sflr_k / n_tot, 4),
            "total_time_s": round(total_time, 2),
            "mean_article_latency_s": round(sum(mean_lats) / max(1, len(mean_lats)), 4),
            "p95_article_latency_s": round(sum(p95_lats) / max(1, len(p95_lats)), 4),
            "ce_reranked_calls": ce_calls,
            "ce_top1_flips": ce_flips,
        }
    return out


def run_safe_purge_and_retract(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
    raw_t17_dir: Path,
) -> dict[str, Any]:
    llm_7b = LLMClient(cfg_dual.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    variants = [
        ("Default (Clause + Unchecked Retract)", "clause", False),
        ("Safe (Claim-Span + Guarded Retract)", "claim_span", True),
    ]
    out: dict[str, Any] = {"egypt_t17": {}, "embedded_16": {}}

    # 1. T17/egypt natural stream check
    egypt_path = processed_dir / "egypt.jsonl"
    tl_path = raw_t17_dir / "egypt" / "timelines.jsonl"
    if egypt_path.exists() and tl_path.exists():
        refs = load_reference_timelines(tl_path)
        all_ref = [item for ref in refs for item in ref]
        arts = filter_articles_by_timespan(load_articles(egypt_path), all_ref)
        target_l = max(1, round(sum(len(r) for r in refs) / len(refs)))
        for v_name, p_mode, g_ret in variants:
            runner = ReviTLSPipeline(
                cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, purge_mode=p_mode, guarded_retract=g_ret
            )
            tl, states = runner.run_stream(arts, num_dates=target_l)
            df1 = max(compute_date_f1(tl, [d for d, _ in ref])["date_f1"] for ref in refs)
            cr1 = max(compute_concat_rouge(tl, ref)["concat_r1"] for ref in refs)
            ar1 = max(compute_align_rouge(tl, ref)["align_r1"] for ref in refs)
            ret_dates = [s.event_date.isoformat() for s in states if s.status.value == "RETRACTED" and s.event_date]
            out["egypt_t17"][v_name] = {
                "date_f1": round(df1, 4),
                "concat_r1": round(cr1, 4),
                "ar1": round(ar1, 4),
                "retracted_count": len(ret_dates),
                "retracted_2011_02_11": "2011-02-11" in ret_dates,
            }

    # 2. 16 embedded revision suite check
    for v_name, p_mode, g_ret in variants:
        rra_sum = 0.0
        sflr_sum = 0.0
        n_topics = 0
        for topic, templates in TOPIC_SCENARIO_SUITES.items():
            stream_path = processed_dir / f"{topic}.jsonl"
            if not stream_path.exists():
                continue
            base_arts = load_articles(stream_path)[:150]
            anchor = base_arts[0].publish_date if base_arts else date(2010, 4, 20)
            emb_arts, emb_cases = inject_revision_scenarios(base_arts, topic=topic, anchor_date=anchor, templates=templates)
            runner = ReviTLSPipeline(
                cfg_dual, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b, purge_mode=p_mode, guarded_retract=g_ret
            )
            _, states = runner.run_stream(emb_arts, num_dates=25)
            sc = evaluate_revision_states(states, emb_cases)
            rra_sum += sc["revision_resolution_accuracy"] * 4
            sflr_sum += sc["stale_fact_leakage_rate"] * 4
            n_topics += 1
        n_tot = max(1, n_topics * 4)
        rra_k = round(rra_sum)
        out["embedded_16"][v_name] = {
            "rra": round(rra_k / n_tot, 4),
            "rra_ci95": wilson_ci(rra_k, n_tot),
            "sflr": round(round(sflr_sum) / n_tot, 4),
        }
    return out


def run_window_sensitivity(
    cfg_dual: ReviTLSConfig,
    embedder: SentenceTransformer,
    processed_dir: Path,
) -> dict[str, Any]:
    llm_7b = LLMClient(cfg_dual.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    windows = [7, 30, 60, 90, 180]
    imp_arts, imp_cases = inject_revision_scenarios(
        [], topic="implicit_long_delay", anchor_date=date(2024, 5, 1), templates=IMPLICIT_LONG_DELAY_TEMPLATES
    )
    out: dict[str, Any] = {}
    for w in tqdm(windows, desc="  Window Sensitivity", unit="win"):
        cfg_w = cfg_dual.model_copy(deep=True)
        cfg_w.retrieval.max_date_window_days = w

        # 16 embedded
        rra_sum = 0.0
        sflr_sum = 0.0
        n_topics = 0
        for topic, templates in TOPIC_SCENARIO_SUITES.items():
            stream_path = processed_dir / f"{topic}.jsonl"
            if not stream_path.exists():
                continue
            base_arts = load_articles(stream_path)[:150]
            anchor = base_arts[0].publish_date if base_arts else date(2010, 4, 20)
            emb_arts, emb_cases = inject_revision_scenarios(base_arts, topic=topic, anchor_date=anchor, templates=templates)
            runner = ReviTLSPipeline(cfg_w, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)
            _, states = runner.run_stream(emb_arts, num_dates=25)
            sc = evaluate_revision_states(states, emb_cases)
            rra_sum += sc["revision_resolution_accuracy"] * 4
            sflr_sum += sc["stale_fact_leakage_rate"] * 4
            n_topics += 1
        n_tot = max(1, n_topics * 4)

        # 8 implicit/long-delay (15-35d)
        imp_runner = ReviTLSPipeline(cfg_w, llm=llm_7b, embedder=embedder, verifier_llm=llm_14b)
        _, imp_states = imp_runner.run_stream(imp_arts, num_dates=12)
        imp_sc = evaluate_revision_states(imp_states, imp_cases)

        out[f"{w}d"] = {
            "embedded_16_rra": round(round(rra_sum) / n_tot, 4),
            "embedded_16_sflr": round(round(sflr_sum) / n_tot, 4),
            "implicit_15_35d_rra": round(imp_sc["revision_resolution_accuracy"], 4),
            "implicit_15_35d_sflr": round(imp_sc["stale_fact_leakage_rate"], 4),
        }
    return out


def write_markdown_report(summary: dict[str, Any], out_md: Path) -> None:
    lines: list[str] = ["# Step-C Experimental Report (Online Latency, Safe Purge, Window Sensitivity)\n"]
    ovd = summary.get("online_vs_deferred", {})
    if ovd:
        lines.append("## 1. Online Immediate vs. Deferred Phase-2 Reconciliation (16 Embedded)\n")
        lines.append("| Mode | RRA [95% CI] | SFLR | Total (s) | Mean Art. Latency (s) | P95 Latency (s) | CE Calls / Flips |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k, v in ovd.items():
            lines.append(
                f"| **{k}** | {v['rra']:.3f} {v['rra_ci95']} | {v['sflr']:.3f} | {v['total_time_s']:.1f} | "
                f"{v['mean_article_latency_s']:.3f} | {v['p95_article_latency_s']:.3f} | {v['ce_reranked_calls']} / {v['ce_top1_flips']} |"
            )
        lines.append("")

    spr = summary.get("safe_purge_retract", {})
    if spr:
        lines.append("## 2. Claim-Span Purge & Guarded Retraction\n")
        lines.append("| Variant | T17/egypt Date-F1 | Concat-R1 | Retracted | 2011-02-11 Retracted? | Embedded-16 RRA | Embedded-16 SFLR |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        for k in spr.get("egypt_t17", {}).keys():
            eg = spr["egypt_t17"][k]
            em = spr.get("embedded_16", {}).get(k, {})
            lines.append(
                f"| **{k}** | {eg['date_f1']:.3f} | {eg['concat_r1']:.3f} | {eg['retracted_count']} | "
                f"{eg['retracted_2011_02_11']} | {em.get('rra', 0):.3f} | {em.get('sflr', 0):.3f} |"
            )
        lines.append("")

    ws = summary.get("window_sensitivity", {})
    if ws:
        lines.append("## 3. Lookback Window Sensitivity (7 / 30 / 60 / 90 / 180 Days)\n")
        lines.append("| Window | Embedded-16 RRA | Embedded-16 SFLR | Implicit 15-35d RRA | Implicit 15-35d SFLR |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for w, v in ws.items():
            lines.append(
                f"| **{w}** | {v['embedded_16_rra']:.3f} | {v['embedded_16_sflr']:.3f} | "
                f"{v['implicit_15_35d_rra']:.3f} | {v['implicit_15_35d_sflr']:.3f} |"
            )
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fast Step-C GPU experiments.")
    parser.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scie_results"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "step_c_summary.json"
    out_md = args.output_dir / "step_c_report.md"

    if out_json.exists() and not args.fresh:
        summary: dict[str, Any] = json.loads(out_json.read_text(encoding="utf-8"))
        print(f"[Resume] Loaded {out_json}", flush=True)
    else:
        summary = {}

    t17_proc = Path("data/processed/t17")
    raw_t17 = Path("data/raw/t17")

    if "online_vs_deferred" not in summary or args.fresh:
        print("[1/3] Running Online Immediate vs. Deferred Phase-2 + Cross-Encoder flips...", flush=True)
        summary["online_vs_deferred"] = run_online_vs_deferred(cfg_dual, embedder, t17_proc)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "safe_purge_retract" not in summary or args.fresh:
        print("[2/3] Running Claim-Span Purge & Guarded Retraction on T17/egypt + Embedded-16...", flush=True)
        summary["safe_purge_retract"] = run_safe_purge_and_retract(cfg_dual, embedder, t17_proc, raw_t17)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if "window_sensitivity" not in summary or args.fresh:
        print("[3/3] Running Window Sensitivity (7/30/60/90/180d)...", flush=True)
        summary["window_sensitivity"] = run_window_sensitivity(cfg_dual, embedder, t17_proc)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_markdown_report(summary, out_md)
    print(f"Done! Saved {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
