"""Full-corpus T17 / Crisis evaluation with per-reference scores and per-system resume.

Differences from the legacy run_scie_suite.py main loop:
  * scores are stored per reference timeline; the primary aggregate is the MEAN over references
    (the max over references used previously is kept only as ``*_max`` for traceability);
  * checkpointing is per (dataset, topic, system), so a crash or Ctrl-C loses at most one run;
  * systems can be selected, and the T17-only component ablations are optional.

Output layout is compatible with scripts/export_transition_audit.py:
    {"meta": {...}, "datasets": {ds: {"per_topic": {topic: {system: {...}}}}}}

Example (RTX 5080, Ollama serving the 7B and 14B models):
    python scripts/run_fullcorpus.py --datasets t17 crisis --output outputs/final/summary.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentence_transformers import SentenceTransformer  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from revitls.baselines.chronos import ClosedDomainCHRONOSBaseline  # noqa: E402
from revitls.baselines.llm_tls import TopicLLMTLSBaseline  # noqa: E402
from revitls.config import ReviTLSConfig, load_config  # noqa: E402
from revitls.evaluation.tls_metrics import (  # noqa: E402
    compute_align_rouge,
    compute_concat_rouge,
    compute_date_f1,
    filter_articles_by_timespan,
    load_reference_timelines,
)
from revitls.llm.client import LLMClient  # noqa: E402
from revitls.pipeline.runner import ReviTLSPipeline  # noqa: E402
from revitls.schemas import Article  # noqa: E402

MAIN_SYSTEMS = [
    "LLM-TLS (7B)",
    "LLM-TLS (14B)",
    "CHRONOS (7B)",
    "CHRONOS (14B)",
    "LLM-TLS + 90d + Latest",
    "LLM-TLS (Recency-TR)",
    "LLM-TLS + PostHoc-14B",
    "ReviTLS (Single-Scale)",
    "ReviTLS (Corroboration Sel.)",
    "ReviTLS (Dual-Scale)",
]
ABLATION_SYSTEMS = [
    "ReviTLS (w/o 14B Verifier)",
    "ReviTLS (w/o Cross-Encoder)",
    "ReviTLS (w/o Conditional Gate)",
    "ReviTLS (w/o Structural Probe)",
    "ReviTLS (w/o Tombstone Purge)",
    "ReviTLS (Guarded Retract)",
    "ReviTLS (Disable Retract)",
]
METRICS = ("date_f1", "ar1", "ar2", "concat_r1", "concat_r2")


def system_factories(
    cfg_single: ReviTLSConfig, cfg_dual: ReviTLSConfig, embedder: Any, llm_7b: Any, llm_14b: Any
) -> dict[str, Callable[[], Any]]:
    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None
    cfg_no_gate = cfg_dual.model_copy(deep=True)
    cfg_no_gate.retrieval.strict_temporal_gate = False
    cfg_no_probe = cfg_dual.model_copy(deep=True)
    cfg_no_probe.retrieval.implicit_probe = False

    def dual(cfg: ReviTLSConfig = cfg_dual, verifier: Any = llm_14b, **kw: Any) -> Callable[[], Any]:
        return lambda: ReviTLSPipeline(cfg, llm=llm_7b, embedder=embedder, verifier_llm=verifier, **kw)

    return {
        "LLM-TLS (7B)": lambda: TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder),
        "LLM-TLS (14B)": lambda: TopicLLMTLSBaseline(llm=llm_14b, embedder=embedder),
        "CHRONOS (7B)": lambda: ClosedDomainCHRONOSBaseline(llm=llm_7b, embedder=embedder),
        "CHRONOS (14B)": lambda: ClosedDomainCHRONOSBaseline(llm=llm_14b, embedder=embedder),
        "LLM-TLS + 90d + Latest": lambda: TopicLLMTLSBaseline(
            llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest"
        ),
        "LLM-TLS (Recency-TR)": lambda: TopicLLMTLSBaseline(
            llm=llm_7b, embedder=embedder, selection_mode="recency_textrank"
        ),
        "LLM-TLS + PostHoc-14B": lambda: TopicLLMTLSBaseline(
            llm=llm_7b, embedder=embedder, post_hoc_verifier_llm=llm_14b
        ),
        "ReviTLS (Single-Scale)": lambda: ReviTLSPipeline(cfg_single, llm=llm_7b, embedder=embedder),
        "ReviTLS (Corroboration Sel.)": dual(selection_mode="corroboration"),
        "ReviTLS (Dual-Scale)": dual(),
        "ReviTLS (w/o 14B Verifier)": dual(verifier=llm_7b),
        "ReviTLS (w/o Cross-Encoder)": dual(cfg_no_ce),
        "ReviTLS (w/o Conditional Gate)": dual(cfg_no_gate),
        "ReviTLS (w/o Structural Probe)": dual(cfg_no_probe),
        "ReviTLS (w/o Tombstone Purge)": dual(disable_purge=True),
        "ReviTLS (Guarded Retract)": dual(guarded_retract=True),
        "ReviTLS (Disable Retract)": dual(disable_retract=True),
    }


def load_articles(path: Path) -> list[Article]:
    with path.open("r", encoding="utf-8") as f:
        return [Article.model_validate_json(line) for line in f if line.strip()]


def score_timeline(timeline: list[Any], refs: list[Any]) -> dict[str, Any]:
    per_ref: dict[str, list[float]] = {m: [] for m in METRICS}
    for ref in refs:
        al = compute_align_rouge(timeline, ref)
        co = compute_concat_rouge(timeline, ref)
        per_ref["date_f1"].append(compute_date_f1(timeline, [d for d, _ in ref])["date_f1"])
        per_ref["ar1"].append(al["align_r1"])
        per_ref["ar2"].append(al["align_r2"])
        per_ref["concat_r1"].append(co["concat_r1"])
        per_ref["concat_r2"].append(co["concat_r2"])
    out: dict[str, Any] = {m: statistics.fmean(v) for m, v in per_ref.items()}
    out.update({f"{m}_max": max(v) for m, v in per_ref.items()})
    out["per_ref"] = per_ref
    return out


def slim_diagnostics(diag: dict[str, Any]) -> dict[str, Any]:
    drop = {"rev_top1_records"}
    return {k: v for k, v in diag.items() if k not in drop}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["t17", "crisis"], choices=["t17", "crisis"])
    ap.add_argument("--single-config", type=Path, default=Path("configs/default.yaml"))
    ap.add_argument("--dual-config", type=Path, default=Path("configs/hybrid_revitls.yaml"))
    ap.add_argument("--output", type=Path, default=Path("outputs/final/summary.json"))
    ap.add_argument("--systems", default="", help="comma-separated subset (default: MAIN_SYSTEMS)")
    ap.add_argument("--ablations", action="store_true", help="add component ablations (T17 only)")
    ap.add_argument("--topics", default="", help="comma-separated topic filter, e.g. bpoil,egypt")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    cfg_single = load_config(args.single_config)
    cfg_dual = load_config(args.dual_config)
    embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)
    llm_7b = LLMClient(cfg_single.llm)
    llm_14b = LLMClient(cfg_dual.verifier_llm) if cfg_dual.verifier_llm else llm_7b
    factories = system_factories(cfg_single, cfg_dual, embedder, llm_7b, llm_14b)

    selected = [s.strip() for s in args.systems.split(",") if s.strip()] or list(MAIN_SYSTEMS)
    abl = list(ABLATION_SYSTEMS) if args.ablations else []
    topic_filter = {t.strip() for t in args.topics.split(",") if t.strip()}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {"datasets": {}}
    if args.output.exists() and not args.fresh:
        data = json.loads(args.output.read_text(encoding="utf-8"))
    data["meta"] = {
        "llm": cfg_single.llm.model,
        "verifier_llm": cfg_dual.verifier_llm.model if cfg_dual.verifier_llm else None,
        "embedding": cfg_dual.embedding.model_name,
        "cross_encoder": cfg_dual.retrieval.cross_encoder_model,
        "aggregation": "mean over reference timelines (per-reference lists stored under per_ref)",
        "ar_cr_implementation": "revitls.evaluation.tls_metrics (reimplementation; not tilse)",
    }

    tasks: list[tuple[str, Path, Path]] = []
    for ds in args.datasets:
        proc, raw = Path("data/processed") / ds, Path("data/raw") / ds
        for f in sorted(proc.glob("*.jsonl")):
            if not topic_filter or f.stem in topic_filter:
                tasks.append((ds, f, raw))
    # Small topics first so partial results appear early.
    tasks.sort(key=lambda t: t[1].stat().st_size)

    for ds, stream_file, raw in tqdm(tasks, desc="topics", unit="topic"):
        topic = stream_file.stem
        refs = load_reference_timelines(raw / topic / "timelines.jsonl")
        articles = filter_articles_by_timespan(load_articles(stream_file), [x for r in refs for x in r])
        target_l = max(1, round(sum(len(r) for r in refs) / len(refs)))
        per_topic = data["datasets"].setdefault(ds, {"per_topic": {}})["per_topic"].setdefault(topic, {})
        names = selected + (abl if ds == "t17" else [])
        for name in names:
            if name in per_topic:
                continue
            if name not in factories:
                raise SystemExit(f"Unknown system {name!r}. Known: {sorted(factories)}")
            for llm in (llm_7b, llm_14b):
                if hasattr(llm, "virtual_elapsed_bonus"):
                    llm.virtual_elapsed_bonus = 0.0
            runner = factories[name]()
            t0 = time.perf_counter()
            timeline, final_states = runner.run_stream(articles, num_dates=target_l)
            elapsed = time.perf_counter() - t0 + sum(
                getattr(llm, "virtual_elapsed_bonus", 0.0) for llm in {id(llm_7b): llm_7b, id(llm_14b): llm_14b}.values()
            )
            res = score_timeline(timeline, refs)
            ref_dates = {d for r in refs for d, _ in r}
            res.update(
                {
                    "time_s": elapsed,
                    "num_articles": len(articles),
                    "target_l": target_l,
                    "n_states": len(final_states),
                    "mean_corroboration": statistics.fmean([st.corroboration_count for st in final_states]) if final_states else 0.0,
                    "n_ref_dates": len(ref_dates),
                    "n_retracted": sum(st.status.value == "RETRACTED" for st in final_states),
                    "n_revised": sum(st.status.value == "REVISED" for st in final_states),
                }
            )
            diag = getattr(runner, "last_diagnostics", None)
            if diag:
                res["diagnostics"] = slim_diagnostics(diag)
            per_topic[name] = res
            args.output.write_text(json.dumps(data, indent=1), encoding="utf-8")
            tqdm.write(
                f"[{ds}/{topic}] {name:<30} Date-F1={res['date_f1']:.3f} AR-1={res['ar1']:.3f} "
                f"Concat-R1={res['concat_r1']:.3f} t={elapsed:.0f}s"
            )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
