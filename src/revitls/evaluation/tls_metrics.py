"""Standard TLS evaluation metrics (Date-F1, Concat ROUGE-N, Align ROUGE-N, and timespan filtering)."""

import json
from datetime import date, timedelta
from pathlib import Path
from rouge_score import rouge_scorer

from revitls.schemas import Article, TimelineEntry


def load_reference_timelines(timelines_jsonl: Path) -> list[list[tuple[date, list[str]]]]:
    """Load news-tls / Open-TLS timelines.jsonl into parsed (date, sentences) lists."""
    parsed: list[list[tuple[date, list[str]]]] = []
    with timelines_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_tl = json.loads(line)
            items: list[tuple[date, list[str]]] = []
            for raw_date, sents in raw_tl:
                items.append((date.fromisoformat(str(raw_date)[:10]), [str(s) for s in sents]))
            parsed.append(items)
    return parsed


def filter_articles_by_timespan(
    articles: list[Article],
    reference: list[tuple[date, list[str]]],
    extension_days: int = 0,
) -> list[Article]:
    """Filter articles to the reference timeline's [start, end] date window (news-tls protocol)."""
    if not reference:
        return articles
    ref_dates = [d for d, _ in reference]
    start = min(ref_dates) - timedelta(days=extension_days)
    end = max(ref_dates) + timedelta(days=extension_days)
    return [a for a in articles if start <= a.publish_date <= end]


def compute_date_f1(predicted: list[TimelineEntry], reference_dates: list[date]) -> dict[str, float]:
    """Compute precision, recall, and F1 over timeline dates."""
    pred_dates = {entry.event_date for entry in predicted}
    ref_dates = set(reference_dates)

    if not pred_dates and not ref_dates:
        return {"precision": 1.0, "recall": 1.0, "date_f1": 1.0}
    if not pred_dates or not ref_dates:
        return {"precision": 0.0, "recall": 0.0, "date_f1": 0.0}

    overlap = len(pred_dates & ref_dates)
    precision = overlap / len(pred_dates)
    recall = overlap / len(ref_dates)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "date_f1": f1}


def compute_concat_rouge(
    predicted: list[TimelineEntry],
    reference: list[tuple[date, list[str]]],
) -> dict[str, float]:
    """Compute Concat ROUGE-1 and ROUGE-2 F1 scores."""
    pred_text = " ".join(entry.summary for entry in predicted).strip()
    ref_text = " ".join(sent for _, sents in reference for sent in sents).strip()
    if not pred_text or not ref_text:
        return {"concat_r1": 0.0, "concat_r2": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    scores = scorer.score(ref_text, pred_text)
    return {
        "concat_r1": float(scores["rouge1"].fmeasure),
        "concat_r2": float(scores["rouge2"].fmeasure),
    }


def compute_align_rouge(
    predicted: list[TimelineEntry],
    reference: list[tuple[date, list[str]]],
) -> dict[str, float]:
    """Compute Alignment-based ROUGE-1 and ROUGE-2 F1 (date-proximity + content similarity)."""
    if not predicted or not reference:
        return {"align_r1": 0.0, "align_r2": 0.0}

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

    # Build pairwise alignment candidates weighted by temporal proximity 1 / (1 + |delta_days|)
    pairs: list[tuple[float, int, int, float, float]] = []
    for i, pred in enumerate(predicted):
        for j, (ref_date, ref_sents) in enumerate(reference):
            ref_text = " ".join(ref_sents).strip()
            if not pred.summary.strip() or not ref_text:
                continue
            r_scores = scorer.score(ref_text, pred.summary)
            r1 = float(r_scores["rouge1"].fmeasure)
            r2 = float(r_scores["rouge2"].fmeasure)
            day_dist = abs((pred.event_date - ref_date).days)
            prox_weight = 1.0 / (1.0 + day_dist)
            pairs.append((prox_weight * (r1 + 0.5 * r2), i, j, r1 * prox_weight, r2 * prox_weight))

    pairs.sort(key=lambda item: item[0], reverse=True)
    used_pred: set[int] = set()
    used_ref: set[int] = set()
    sum_r1 = 0.0
    sum_r2 = 0.0

    for _score, i, j, wr1, wr2 in pairs:
        if i in used_pred or j in used_ref:
            continue
        used_pred.add(i)
        used_ref.add(j)
        sum_r1 += wr1
        sum_r2 += wr2

    # Harmonic mean over |predicted| (precision-like) and |reference| (recall-like)
    def _f1(total_score: float) -> float:
        p = total_score / len(predicted)
        r = total_score / len(reference)
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {"align_r1": _f1(sum_r1), "align_r2": _f1(sum_r2)}
