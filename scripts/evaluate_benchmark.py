"""Side-by-side benchmark evaluator for ReviTLS, Topic LLM-TLS, and Closed-Domain CHRONOS."""

import argparse
import json
import time
from pathlib import Path
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
from revitls.revisions.benchmark import load_revision_cases
from revitls.schemas import Article


def load_articles(stream_path: Path) -> list[Article]:
    with stream_path.open("r", encoding="utf-8") as f:
        return [Article.model_validate_json(line) for line in f if line.strip()]


def format_markdown_table(rows: list[dict[str, object]]) -> str:
    header = (
        "| System | Date-F1 | Align-R1 | Align-R2 | Concat-R1 | Concat-R2 | RRA (↑) | SFLR (↓) | Time (s) |"
    )
    sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['system']} | {r['date_f1']:.3f} | {r['align_r1']:.3f} | {r['align_r2']:.3f} | "
            f"{r['concat_r1']:.3f} | {r['concat_r2']:.3f} | {r['rra']:.3f} | {r['sflr']:.3f} | {r['seconds']:.1f} |"
        )
    return "\n".join(lines)


def evaluate_single_run(
    articles: list[Article],
    cfg: ReviTLSConfig,
    llm: LLMClient,
    embedder: SentenceTransformer,
    refs: list,
    cases: list,
    num_dates: int | None,
) -> list[dict[str, object]]:
    target_l = num_dates
    if target_l is None and refs:
        target_l = max(1, round(sum(len(r) for r in refs) / len(refs)))
    target_l = target_l or cfg.timeline.default_num_dates

    runners = [
        ("LLM-TLS (Topic)", TopicLLMTLSBaseline(llm=llm, embedder=embedder)),
        ("CHRONOS (Closed)", ClosedDomainCHRONOSBaseline(llm=llm, embedder=embedder)),
        ("ReviTLS (Ours)", ReviTLSPipeline(cfg, llm=llm, embedder=embedder)),
    ]

    rows: list[dict[str, object]] = []
    for name, runner in runners:
        t0 = time.perf_counter()
        timeline, states = runner.run_stream(articles, num_dates=target_l)
        elapsed = time.perf_counter() - t0

        date_f1 = align_r1 = align_r2 = concat_r1 = concat_r2 = 0.0
        if refs:
            date_f1 = max(compute_date_f1(timeline, [d for d, _ in ref])["date_f1"] for ref in refs)
            aligns = [compute_align_rouge(timeline, ref) for ref in refs]
            align_r1 = max(x["align_r1"] for x in aligns)
            align_r2 = max(x["align_r2"] for x in aligns)
            concats = [compute_concat_rouge(timeline, ref) for ref in refs]
            concat_r1 = max(x["concat_r1"] for x in concats)
            concat_r2 = max(x["concat_r2"] for x in concats)

        rev_scores = evaluate_revision_states(states, cases) if cases else {
            "revision_resolution_accuracy": 0.0,
            "stale_fact_leakage_rate": 0.0,
        }

        rows.append(
            {
                "system": name,
                "date_f1": date_f1,
                "align_r1": align_r1,
                "align_r2": align_r2,
                "concat_r1": concat_r1,
                "concat_r2": concat_r2,
                "rra": rev_scores["revision_resolution_accuracy"],
                "sflr": rev_scores["stale_fact_leakage_rate"],
                "seconds": elapsed,
            }
        )
    return rows


def aggregate_rows(all_topic_rows: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    """Macro-average metrics across all evaluated topics."""
    systems = [r["system"] for r in all_topic_rows[0]]
    metrics = ("date_f1", "align_r1", "align_r2", "concat_r1", "concat_r2", "rra", "sflr", "seconds")
    agg: list[dict[str, object]] = []
    for idx, sys_name in enumerate(systems):
        entry: dict[str, object] = {"system": sys_name}
        for m in metrics:
            entry[m] = sum(float(topic_rows[idx][m]) for topic_rows in all_topic_rows) / len(all_topic_rows)
        agg.append(entry)
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ReviTLS vs LLM-TLS & CHRONOS.")
    parser.add_argument("--stream", type=Path, default=None, help="Single topic article JSONL")
    parser.add_argument("--ref-timelines", type=Path, default=None, help="Single topic timelines.jsonl")
    parser.add_argument("--revision-cases", type=Path, default=None, help="Optional *_cases.jsonl")
    parser.add_argument("--processed-dir", type=Path, default=None, help="Batch mode: data/processed/t17")
    parser.add_argument("--raw-dir", type=Path, default=None, help="Batch mode: data/raw/t17")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--num-dates", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = LLMClient(cfg.llm)
    embedder = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)

    if args.processed_dir and args.raw_dir:
        all_topic_rows: list[list[dict[str, object]]] = []
        per_topic: dict[str, list[dict[str, object]]] = {}
        for stream_file in sorted(args.processed_dir.glob("*.jsonl")):
            topic = stream_file.stem
            tl_file = args.raw_dir / topic / "timelines.jsonl"
            refs = load_reference_timelines(tl_file) if tl_file.exists() else []
            articles = load_articles(stream_file)
            if refs:
                # Filter to union of reference timeline spans to match news-tls protocol
                all_ref_entries = [item for ref in refs for item in ref]
                articles = filter_articles_by_timespan(articles, all_ref_entries)
            print(f"\n=== Evaluating topic: {topic} ({len(articles)} articles) ===")
            rows = evaluate_single_run(articles, cfg, llm, embedder, refs, [], args.num_dates)
            print(format_markdown_table(rows))
            all_topic_rows.append(rows)
            per_topic[topic] = rows

        macro = aggregate_rows(all_topic_rows)
        print("\n=== Macro-Average Across All Topics ===")
        print(format_markdown_table(macro))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps({"macro": macro, "per_topic": per_topic}, indent=2),
                encoding="utf-8",
            )
        return

    if not args.stream:
        raise ValueError("Provide either --stream (single topic) or --processed-dir + --raw-dir (batch).")

    articles = load_articles(args.stream)
    refs = load_reference_timelines(args.ref_timelines) if args.ref_timelines else []
    cases = load_revision_cases(args.revision_cases) if args.revision_cases else []
    rows = evaluate_single_run(articles, cfg, llm, embedder, refs, cases, args.num_dates)
    print(format_markdown_table(rows))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
