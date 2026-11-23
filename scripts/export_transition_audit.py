"""Export natural-stream REVISED and RETRACTED transitions from summary.json for 2-annotator audit,
and compute Cohen's kappa + precision / false-retraction rates once annotated.

No GPU required.

Usage:
    python scripts/export_transition_audit.py                 # Export CSV + MD
    python scripts/export_transition_audit.py --export-random-audit # New stratified sample; writes separate files
    python scripts/export_transition_audit.py --compute-kappa # Compute Cohen's kappa & precision
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

from revitls.evaluation.tls_metrics import load_reference_timelines


def _refuse_annotated_overwrite(out_csv: Path, force: bool = False) -> None:
    if not out_csv.exists() or force:
        return
    with out_csv.open("r", encoding="utf-8", newline="") as f:
        existing = list(csv.DictReader(f))
    has_annotations = any(
        (row.get("annotator_1_tp_fp") or "").strip()
        or (row.get("annotator_2_tp_fp") or "").strip()
        or (row.get("notes") or "").strip()
        for row in existing
    )
    if has_annotations:
        raise FileExistsError(
            f"{out_csv} contains human annotations and will not be overwritten. "
            "Back it up and pass --force only if replacing it is intentional."
        )


def export_random_audit(
    summary_path: Path,
    out_csv: Path,
    out_md: Path,
    per_stratum: int = 5,
    seed: int = 20261119,
) -> None:
    """Draw a reproducible stratified sample from complete transition diagnostics.

    Strata are dataset × topic × transition type. This audits the precision of
    emitted transitions; it does not estimate missed-transition recall.
    """
    _refuse_annotated_overwrite(out_csv)
    data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))
    strata: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for ds in ("t17", "crisis"):
        per_topic = data.get("datasets", {}).get(ds, {}).get("per_topic", {})
        for topic, systems in per_topic.items():
            diagnostics = systems.get("ReviTLS (Dual-Scale)", {}).get("diagnostics", {})
            records = diagnostics.get("transition_records")
            if records is None:
                raise ValueError(
                    f"{ds}/{topic} has no transition_records. This summary predates the current pipeline; "
                    "run the finalized evaluation before drawing the audit sample."
                )
            for record in records:
                kind = str(record.get("transition", "")).upper()
                if kind not in {"REVISED", "RETRACTED"}:
                    continue
                strata.setdefault((ds, str(topic), kind), []).append(record)

    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for (ds, topic, kind), population in sorted(strata.items()):
        n_population = len(population)
        n_sample = min(per_stratum, n_population)
        selected = rng.sample(population, n_sample)
        inclusion_probability = n_sample / n_population
        for record in selected:
            rows.append({
                "dataset": ds,
                "topic": topic,
                "transition": kind,
                "event_date": str(record.get("event_date", "")),
                "publication_date": str(record.get("publication_date", "")),
                "source_article_id": str(record.get("source_article_id", "")),
                "source_title": str(record.get("source_title", "")).replace("\n", " "),
                "previous_summary": str(record.get("previous_summary", "")).replace("\n", " "),
                "new_summary": str(record.get("new_summary", "")).replace("\n", " "),
                "population_stratum_n": str(n_population),
                "sample_stratum_n": str(n_sample),
                "inclusion_probability": f"{inclusion_probability:.8f}",
                "design_weight": f"{1.0 / inclusion_probability:.4f}",
                "annotator_1_tp_fp": "",
                "annotator_2_tp_fp": "",
                "notes": "",
            })

    if not rows:
        raise ValueError("No REVISED/RETRACTED transition records found in the summary.")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    md = [
        "# Stratified Random Natural-Stream Transition Audit",
        "",
        f"- Seed: `{seed}`; requested sample per dataset × topic × transition stratum: `{per_stratum}`.",
        f"- Exported records: **{len(rows)}** across **{len(strata)}** non-empty strata.",
        "- Annotate `annotator_1_tp_fp` and `annotator_2_tp_fp` as `TP` or `FP` independently after checking the source article and prior context; the system rationale is intentionally withheld.",
        "- Use `design_weight` for population-level precision estimates because strata have unequal sampling fractions.",
        "- This sample estimates precision of emitted transitions only; recall requires reviewing source articles that the system did not label as transitions.",
        "- Source article text is not copied into this file; use `source_article_id` to locate it in the licensed corpus.",
        "- Annotation rubric: `docs/transition_audit_protocol.md`.",
        "",
        f"CSV: `{out_csv}`",
    ]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Exported stratified random audit: {len(rows)} records, {len(strata)} strata -> {out_csv}")


def compute_kappa(csv_path: Path, out_json: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    paired: list[tuple[str, str, dict[str, str]]] = []
    for r in rows:
        a1 = (r.get("annotator_1_tp_fp") or "").strip().upper()
        a2 = (r.get("annotator_2_tp_fp") or "").strip().upper()
        if a1 in {"TP", "FP"} and a2 in {"TP", "FP"}:
            paired.append((a1, a2, r))

    n = len(paired)
    if n == 0:
        print("No paired TP/FP annotations found yet in annotator_1_tp_fp / annotator_2_tp_fp.", flush=True)
        return

    agree = sum(1 for a1, a2, _ in paired if a1 == a2)
    po = agree / n
    p1_tp = sum(1 for a1, _, _ in paired if a1 == "TP") / n
    p2_tp = sum(1 for _, a2, _ in paired if a2 == "TP") / n
    pe = p1_tp * p2_tp + (1.0 - p1_tp) * (1.0 - p2_tp)
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 1.0

    def stats_for(subset: list[tuple[str, str, dict[str, str]]]) -> dict[str, Any]:
        if not subset:
            return {"n": 0, "consensus_tp": 0, "consensus_fp": 0, "precision": None, "fp_rate": None}
        # Conservative consensus: TP only if both annotators mark TP (or majority/a1 on ties reported separately)
        both_tp = sum(1 for a1, a2, _ in subset if a1 == "TP" and a2 == "TP")
        any_fp = len(subset) - both_tp
        return {
            "n": len(subset),
            "consensus_tp": both_tp,
            "disagree_or_fp": any_fp,
            "strict_precision": round(both_tp / len(subset), 4),
            "strict_fp_rate": round(any_fp / len(subset), 4),
            "a1_precision": round(sum(1 for a1, _, _ in subset if a1 == "TP") / len(subset), 4),
            "a2_precision": round(sum(1 for _, a2, _ in subset if a2 == "TP") / len(subset), 4),
        }

    retracted = [x for x in paired if x[2].get("transition") == "RETRACTED"]
    revised = [x for x in paired if x[2].get("transition") == "REVISED"]

    res = {
        "total_annotated": n,
        "raw_agreement": round(po, 4),
        "cohens_kappa": round(kappa, 4),
        "overall": stats_for(paired),
        "revised": stats_for(revised),
        "retracted": stats_for(retracted),
        "t17_retracted": stats_for([x for x in retracted if x[2].get("dataset") == "t17"]),
        "crisis_retracted": stats_for([x for x in retracted if x[2].get("dataset") == "crisis"]),
    }
    out_json.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)


def export_audit(
    summary_path: Path,
    raw_t17_dir: Path,
    out_csv: Path,
    out_md: Path,
    force: bool = False,
) -> None:
    _refuse_annotated_overwrite(out_csv, force=force)

    data: dict[str, Any] = json.loads(summary_path.read_text(encoding="utf-8"))

    rows: list[dict[str, str]] = []
    t17_retracted_total = 0
    t17_retracted_ref_overlap = 0

    for ds in ("t17", "crisis"):
        per_topic = data.get("datasets", {}).get(ds, {}).get("per_topic", {})
        for topic, sys_map in sorted(per_topic.items()):
            dual = sys_map.get("ReviTLS (Dual-Scale)", {})
            diag = dual.get("diagnostics", {})
            ref_dates: set[str] = set()
            if ds == "t17":
                tl_path = raw_t17_dir / topic / "timelines.jsonl"
                if tl_path.exists():
                    refs = load_reference_timelines(tl_path)
                    ref_dates = {d.isoformat() for ref in refs for d, _ in ref}

            for item in diag.get("sample_retracted", []):
                dt = str(item.get("date", ""))
                in_ref = "yes" if (ds == "t17" and dt in ref_dates) else ("no" if ds == "t17" else "n/a")
                if ds == "t17":
                    t17_retracted_total += 1
                    if dt in ref_dates:
                        t17_retracted_ref_overlap += 1
                rows.append(
                    {
                        "dataset": ds,
                        "topic": topic,
                        "transition": "RETRACTED",
                        "date": dt,
                        "in_t17_ref_dates": in_ref,
                        "summary": str(item.get("summary", "")).replace("\n", " "),
                        "purged_previous": "",
                        "annotator_1_tp_fp": "",
                        "annotator_2_tp_fp": "",
                        "notes": "",
                    }
                )

            for item in diag.get("sample_revised", []):
                dt = str(item.get("date", ""))
                purged = " | ".join(str(p).replace("\n", " ") for p in item.get("purged", []))
                rows.append(
                    {
                        "dataset": ds,
                        "topic": topic,
                        "transition": "REVISED",
                        "date": dt,
                        "in_t17_ref_dates": "yes" if (ds == "t17" and dt in ref_dates) else ("no" if ds == "t17" else "n/a"),
                        "summary": str(item.get("summary", "")).replace("\n", " "),
                        "purged_previous": purged,
                        "annotator_1_tp_fp": "",
                        "annotator_2_tp_fp": "",
                        "notes": "",
                    }
                )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "topic",
        "transition",
        "date",
        "in_t17_ref_dates",
        "summary",
        "purged_previous",
        "annotator_1_tp_fp",
        "annotator_2_tp_fp",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Natural-Stream Transition Audit Sheet (2-Annotator Protocol)",
        "",
        f"- Total exported transition samples: **{len(rows)}**",
        f"- T17 sampled `RETRACTED` dates overlapping ground-truth reference dates: **{t17_retracted_ref_overlap} / {t17_retracted_total}**",
        f"- CSV for independent annotation (`annotator_1_tp_fp`, `annotator_2_tp_fp` as `TP` or `FP`): `{out_csv}`",
        "",
        "| Dataset | Topic | Type | Date | In Ref? | Summary | Purged Previous | A1 | A2 |",
        "| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['topic']} | {r['transition']} | {r['date']} | {r['in_t17_ref_dates']} | "
            f"{r['summary'][:100]} | {r['purged_previous'][:80]} | | |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        f"Exported {len(rows)} transitions to {out_csv} and {out_md}. "
        f"T17 retracted-date ref overlap: {t17_retracted_ref_overlap}/{t17_retracted_total}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export transition audit or compute Cohen's kappa.")
    parser.add_argument("--compute-kappa", action="store_true")
    parser.add_argument("--export-random-audit", action="store_true")
    parser.add_argument("--audit-per-stratum", type=int, default=5)
    parser.add_argument("--audit-seed", type=int, default=20261119)
    parser.add_argument("--summary", type=Path, default=Path("outputs/final/summary.json"), help="full-corpus summary with transition_records")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/final"))
    parser.add_argument("--force", action="store_true", help="Allow replacing a legacy audit CSV with human annotations.")
    args = parser.parse_args()

    if args.audit_per_stratum < 1:
        parser.error("--audit-per-stratum must be at least 1")

    out_csv = Path("outputs/scie_results/transition_audit.csv")
    if args.compute_kappa:
        compute_kappa(out_csv, Path("outputs/scie_results/audit_kappa_summary.json"))
    elif args.export_random_audit:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        export_random_audit(
            args.summary,
            args.out_dir / "transition_audit_random.csv",
            args.out_dir / "transition_audit_random.md",
            per_stratum=args.audit_per_stratum,
            seed=args.audit_seed,
        )
    else:
        export_audit(
            Path("outputs/scie_results/summary.json"),
            Path("data/raw/t17"),
            out_csv,
            Path("outputs/scie_results/transition_audit.md"),
            force=args.force,
        )


if __name__ == "__main__":
    main()
