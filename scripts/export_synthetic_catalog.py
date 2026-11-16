"""Export full catalog of all 32 synthetic scenarios (16 multi-topic + 8 parametric + 8 implicit/long-delay).

Generates a transparent markdown/JSON appendix documenting source topic, anchor/delays,
initial claim, corrective evidence, expected final fact, forbidden stale substring, and target state.

Usage:
    python scripts/export_synthetic_catalog.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from revitls.revisions.benchmark import COUNTERFACTUAL_SCENARIO_TEMPLATES, TOPIC_SCENARIO_SUITES
from scripts.run_step_b_suite import IMPLICIT_LONG_DELAY_TEMPLATES


def _template_to_dict(suite_name: str, topic: str, t: Any) -> dict[str, Any]:
    return {
        "suite": suite_name,
        "topic": topic,
        "case_id": t.case_id,
        "revision_type": str(t.revision_type.value),
        "event_offset_days": t.event_date_offset_days,
        "correction_delay_days": t.correction_delay_days,
        "target_status": "RETRACTED" if t.should_be_retracted else "REVISED",
        "initial_title": t.initial_title,
        "initial_body": t.initial_body,
        "corroborating_repeats": t.corroborating_repeats,
        "correction_title": t.correction_title,
        "correction_body": t.correction_body,
        "expected_final_fact": t.expected_final_fact,
        "stale_fact_substring": t.stale_fact_substring,
    }


def main() -> None:
    out_json = Path("outputs/scie_results/synthetic_suite_catalog.json")
    out_md = Path("outputs/scie_results/synthetic_suite_catalog.md")

    rows: list[dict[str, Any]] = []
    for topic, templates in TOPIC_SCENARIO_SUITES.items():
        for t in templates:
            rows.append(_template_to_dict("16-Scenario Multi-Topic Suite", topic, t))
    for t in COUNTERFACTUAL_SCENARIO_TEMPLATES:
        rows.append(_template_to_dict("8-Scenario Parametric Control", "counterfactual_control", t))
    for t in IMPLICIT_LONG_DELAY_TEMPLATES:
        rows.append(_template_to_dict("8-Scenario Implicit & Long-Delay", "implicit_long_delay", t))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    md: list[str] = [
        "# Synthetic Revision Benchmark Catalog (32 Scenarios)",
        "",
        f"Total scenarios documented: **{len(rows)}**",
        "",
        "| Suite | Topic | Case ID | Type | Delay (d) | Target Status | Expected Final Fact | Forbidden Stale Substring |",
        "| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |",
    ]
    for r in rows:
        md.append(
            f"| {r['suite']} | `{r['topic']}` | `{r['case_id']}` | `{r['revision_type']}` | "
            f"{r['correction_delay_days']} | `{r['target_status']}` | "
            f"{r['expected_final_fact'] or '*(excluded)*'} | `{r['stale_fact_substring']}` |"
        )
    md.append("")
    md.append("## Full Scenario Texts (Initial Claim vs. Corrective Evidence)")
    md.append("")
    for i, r in enumerate(rows, 1):
        md.append(f"### {i}. [{r['suite']}] `{r['topic']}` / `{r['case_id']}` (`{r['target_status']}`, $\\Delta t={r['correction_delay_days']}$d)")
        md.append(f"- **Initial ({r['corroborating_repeats']} repeats):** *{r['initial_title']}* — {r['initial_body']}")
        md.append(f"- **Correction:** *{r['correction_title']}* — {r['correction_body']}")
        md.append(f"- **Evaluation Rule:** Target status `{r['target_status']}`, expected=`{r['expected_final_fact']}`, forbidden=`{r['stale_fact_substring']}`\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"Exported {len(rows)} scenarios to {out_json} and {out_md}", flush=True)


if __name__ == "__main__":
    main()
