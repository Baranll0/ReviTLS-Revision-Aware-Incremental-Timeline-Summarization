"""Export natural-stream REVISED and RETRACTED transitions from summary.json for human audit.

Also computes how many sampled T17 RETRACTED dates overlap with ground-truth reference dates.
No GPU required.

Usage:
    python scripts/export_transition_audit.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from revitls.evaluation.tls_metrics import load_reference_timelines


def main() -> None:
    summary_path = Path("outputs/scie_results/summary.json")
    raw_t17_dir = Path("data/raw/t17")
    out_csv = Path("outputs/scie_results/transition_audit.csv")
    out_md = Path("outputs/scie_results/transition_audit.md")

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
                        "human_verdict_tp_or_fp": "",
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
                        "human_verdict_tp_or_fp": "",
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
        "human_verdict_tp_or_fp",
        "notes",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    md_lines = [
        "# Natural-Stream Transition Audit Sheet",
        "",
        f"- Total exported transition samples: **{len(rows)}**",
        f"- T17 sampled `RETRACTED` dates overlapping ground-truth reference dates: **{t17_retracted_ref_overlap} / {t17_retracted_total}**",
        f"- CSV for annotation: `{out_csv}`",
        "",
        "| Dataset | Topic | Type | Date | In Ref? | Summary | Purged Previous | Verdict (TP/FP) |",
        "| :--- | :--- | :--- | :--- | :---: | :--- | :--- | :---: |",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['dataset']} | {r['topic']} | {r['transition']} | {r['date']} | {r['in_t17_ref_dates']} | "
            f"{r['summary'][:100]} | {r['purged_previous'][:80]} | |"
        )
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(
        f"Exported {len(rows)} transitions to {out_csv} and {out_md}. "
        f"T17 retracted-date ref overlap: {t17_retracted_ref_overlap}/{t17_retracted_total}",
        flush=True,
    )


if __name__ == "__main__":
    main()
