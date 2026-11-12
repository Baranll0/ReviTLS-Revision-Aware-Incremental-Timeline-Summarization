"""CLI entrypoint to run the ReviTLS incremental pipeline on a chronological JSONL stream."""

import argparse
from pathlib import Path
from sentence_transformers import SentenceTransformer

from revitls.config import load_config
from revitls.llm.client import LLMClient
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.schemas import Article


def load_articles(stream_path: Path) -> list[Article]:
    articles: list[Article] = []
    with stream_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                articles.append(Article.model_validate_json(line))
    articles.sort(key=lambda a: (a.publish_date, a.article_id))
    return articles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ReviTLS incremental timeline summarization.")
    parser.add_argument("--stream", type=Path, required=True, help="Chronological article JSONL")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, required=True, help="Output timeline JSONL")
    parser.add_argument("--num-dates", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    llm = LLMClient(cfg.llm)
    embedder = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)

    pipeline = ReviTLSPipeline(cfg, llm=llm, embedder=embedder)
    timeline, states = pipeline.run_stream(load_articles(args.stream), num_dates=args.num_dates)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for entry in timeline:
            f.write(entry.model_dump_json() + "\n")

    revised = sum(1 for s in states if s.revision_history)
    print(f"Generated {len(timeline)} timeline entries ({revised} revised states) -> {args.output}")


if __name__ == "__main__":
    main()
