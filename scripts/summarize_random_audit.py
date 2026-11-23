"""Summarize the annotated stratified random transition audit.

Input: outputs/final/transition_audit_random.csv with annotator_1_tp_fp / annotator_2_tp_fp
filled independently (TP/FP). Optional column ``adjudicated_tp_fp`` resolves disagreements and
``UNCERTAIN`` items (excluded from the estimate and counted separately).

Estimator: stratified (dataset x topic x transition) design-weighted precision with the usual
stratified variance including the finite-population correction; 95% normal-approximation interval,
clipped to [0, 1]. Strata with a single sampled unit use the conservative p(1-p) <= 0.25.

    python scripts/summarize_random_audit.py --csv outputs/final/transition_audit_random.csv
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def label_of(row: dict[str, str]) -> str | None:
    a1 = (row.get("annotator_1_tp_fp") or "").strip().upper()
    a2 = (row.get("annotator_2_tp_fp") or "").strip().upper()
    adj = (row.get("adjudicated_tp_fp") or "").strip().upper()
    if "UNCERTAIN" in (row.get("notes") or "").upper() and not adj:
        return None
    if a1 in {"TP", "FP"} and a1 == a2:
        return a1
    return adj if adj in {"TP", "FP"} else None


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    p1 = sum(a == "TP" for a, _ in pairs) / n
    p2 = sum(b == "TP" for _, b in pairs) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else None


def weighted_precision(rows: list[dict[str, str]]) -> dict[str, float | int | None]:
    strata: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    pop: dict[tuple[str, str, str], int] = {}
    for r in rows:
        lab = label_of(r)
        key = (r["dataset"], r["topic"], r["transition"])
        pop[key] = int(r["population_stratum_n"])
        if lab is not None:
            strata[key].append(lab)
    total_n = sum(pop[k] for k in strata)
    if total_n == 0:
        return {"precision": None, "ci_lo": None, "ci_hi": None, "n_sampled": 0, "n_population": 0}
    est = var = 0.0
    n_sampled = 0
    for k, labs in strata.items():
        n_h, big_n = len(labs), pop[k]
        p_h = sum(l == "TP" for l in labs) / n_h
        w = big_n / total_n
        est += w * p_h
        s2 = p_h * (1 - p_h) * n_h / (n_h - 1) if n_h > 1 else 0.25
        var += w * w * (1 - min(1.0, n_h / big_n)) * s2 / n_h
        n_sampled += n_h
    se = math.sqrt(var)
    return {
        "precision": round(est, 4),
        "ci_lo": round(max(0.0, est - 1.96 * se), 4),
        "ci_hi": round(min(1.0, est + 1.96 * se), 4),
        "n_sampled": n_sampled,
        "n_population": total_n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("outputs/final/transition_audit_random.csv"))
    ap.add_argument("--out", type=Path, default=Path("outputs/final/audit_summary.json"))
    args = ap.parse_args()
    with args.csv.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    pairs = [
        ((r.get("annotator_1_tp_fp") or "").strip().upper(), (r.get("annotator_2_tp_fp") or "").strip().upper())
        for r in rows
    ]
    pairs = [p for p in pairs if p[0] in {"TP", "FP"} and p[1] in {"TP", "FP"}]
    out = {
        "n_rows": len(rows),
        "n_double_labelled": len(pairs),
        "raw_agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
        "cohens_kappa": None if cohen_kappa(pairs) is None else round(cohen_kappa(pairs), 4),
        "n_uncertain_or_unresolved": sum(label_of(r) is None for r in rows),
        "overall": weighted_precision(rows),
        "revised": weighted_precision([r for r in rows if r["transition"] == "REVISED"]),
        "retracted": weighted_precision([r for r in rows if r["transition"] == "RETRACTED"]),
        "unweighted_sample": {
            t: {
                "tp": sum(label_of(r) == "TP" for r in rows if r["transition"] == t),
                "n": sum(label_of(r) is not None for r in rows if r["transition"] == t),
            }
            for t in ("REVISED", "RETRACTED")
        },
        "note": "Precision among emitted transitions only; recall is not estimated by this audit.",
    }
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
