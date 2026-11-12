"""Convert news-tls (T17 / Crisis) topic folders into unified chronological JSONL streams."""

import argparse
import gzip
import json
from pathlib import Path
from typing import IO


def _open_jsonl(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def find_articles_file(topic_dir: Path) -> Path:
    """Locate the article JSONL(.gz) file inside a news-tls topic directory."""
    candidates = [
        topic_dir / "articles.preprocessed.jsonl.gz",
        topic_dir / "articles.jsonl.gz",
        topic_dir / "articles.preprocessed.jsonl",
        topic_dir / "articles.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No articles JSONL(.gz) file found in {topic_dir}. "
        "Run scripts/download_datasets.ps1 first."
    )


def convert_news_tls_topic(topic_dir: Path, output_path: Path) -> int:
    """Normalize a single news-tls topic folder into a chronologically sorted Article stream."""
    articles_file = find_articles_file(topic_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    with _open_jsonl(articles_file) as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            raw = json.loads(line)
            pub_date = str(raw.get("time", raw.get("date", "")))[:10]
            if not pub_date:
                continue
            title = (raw.get("title") or "").strip()
            text = (raw.get("text") or raw.get("content") or "").strip()
            if not title and text:
                title = text.split("\n", 1)[0][:120].strip()
            records.append(
                {
                    "article_id": str(raw.get("id", f"{topic_dir.name}_{idx}")),
                    "topic": topic_dir.name,
                    "title": title,
                    "content": text,
                    "publish_date": pub_date,
                }
            )

    records.sort(key=lambda r: (r["publish_date"], r["article_id"]))
    with output_path.open("w", encoding="utf-8") as out:
        for rec in records:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(records)


def convert_dataset_root(raw_dataset_dir: Path, processed_dir: Path) -> dict[str, int]:
    """Convert all topic subdirectories under a dataset folder (e.g. data/raw/t17)."""
    stats: dict[str, int] = {}
    for topic_dir in sorted(p for p in raw_dataset_dir.iterdir() if p.is_dir()):
        out_file = processed_dir / f"{topic_dir.name}.jsonl"
        stats[topic_dir.name] = convert_news_tls_topic(topic_dir, out_file)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TLS corpora into chronological streams.")
    parser.add_argument("--source", type=Path, required=True, help="Dataset dir or single topic dir")
    parser.add_argument("--output", type=Path, required=True, help="Output dir or JSONL file")
    args = parser.parse_args()

    if any(
        (args.source / name).exists()
        for name in ("articles.preprocessed.jsonl.gz", "articles.jsonl.gz", "articles.jsonl")
    ):
        n = convert_news_tls_topic(args.source, args.output)
        print(f"Wrote {n} articles -> {args.output}")
    else:
        stats = convert_dataset_root(args.source, args.output)
        for topic, n in stats.items():
            print(f"  {topic}: {n} articles")


if __name__ == "__main__":
    main()
