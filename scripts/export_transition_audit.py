"""Export natural-stream REVISED and RETRACTED transitions from summary.json for 2-annotator audit,
and compute Cohen's kappa + precision / false-retraction rates once annotated.

No GPU required.

Usage:
    python scripts/export_transition_audit.py                 # Export CSV + MD
    python scripts/export_transition_audit.py --compute-kappa # Compute Cohen's kappa & precision
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from revitls.evaluation.tls_metrics import load_reference_timelines


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


def export_audit(summary_path: Path, raw_t17_dir: Path, out_csv: Path, out_md: Path) -> None:
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
    args = parser.parse_args()

    out_csv = Path("outputs/scie_results/transition_audit.csv")
    if args.compute_kappa:
        compute_kappa(out_csv, Path("outputs/scie_results/audit_kappa_summary.json"))
    else:
        export_audit(
            Path("outputs/scie_results/summary.json"),
            Path("data/raw/t17"),
            out_csv,
            Path("outputs/scie_results/transition_audit.md"),
        )


if __name__ == "__main__":
    main()
