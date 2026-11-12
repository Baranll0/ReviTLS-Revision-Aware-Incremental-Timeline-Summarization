"""Generate controlled revision stress-test bundles from processed TLS article streams."""

import argparse
from datetime import date
from pathlib import Path

from revitls.revisions.benchmark import inject_revision_scenarios, save_revision_bundle
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
    parser = argparse.ArgumentParser(description="Inject controlled revision scenarios into a stream.")
    parser.add_argument("--stream", type=Path, required=True, help="Processed topic JSONL stream")
    parser.add_argument("--output-dir", type=Path, default=Path("data/revisions"))
    parser.add_argument("--max-base-articles", type=int, default=150)
    args = parser.parse_args()

    base = load_articles(args.stream)[: args.max_base_articles]
    if not base:
        raise ValueError(f"No articles found in {args.stream}")

    topic = base[0].topic
    anchor = base[0].publish_date if isinstance(base[0].publish_date, date) else date.today()
    augmented, cases = inject_revision_scenarios(base, topic=topic, anchor_date=anchor)
    save_revision_bundle(args.output_dir, topic=topic, articles=augmented, cases=cases)
    print(
        f"Saved revision benchmark for '{topic}': {len(augmented)} articles, "
        f"{len(cases)} revision test cases -> {args.output_dir}"
    )


if __name__ == "__main__":
    main()
