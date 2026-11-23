"""Generate LaTeX tables and number macros for the paper from result JSON files.

Every number in the paper's results sections is produced here, so nothing is copied by hand.
Missing inputs produce "pending" placeholders and the paper still compiles.

    python scripts/make_paper_tables.py \
        --summary outputs/final/summary.json --heldout outputs/final/heldout.json --out paper/generated
"""

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from revitls.evaluation.stats import holm_adjust, wilson_ci  # noqa: E402

PENDING = r"\textit{pending}"
MAIN_ORDER = [
    "LLM-TLS (7B)", "LLM-TLS (14B)", "LLM-TLS + 90d + Latest", "LLM-TLS (Recency-TR)", "LLM-TLS + PostHoc-14B",
    "CHRONOS (7B)", "CHRONOS (14B)", "ReviTLS (Single-Scale)", "ReviTLS (Corroboration Sel.)", "ReviTLS (Dual-Scale)",
]
ABLATION_ORDER = [
    "ReviTLS (Dual-Scale)", "ReviTLS (w/o 14B Verifier)", "ReviTLS (w/o Cross-Encoder)", "ReviTLS (w/o Conditional Gate)",
    "ReviTLS (w/o Structural Probe)", "ReviTLS (w/o Tombstone Purge)", "ReviTLS (Guarded Retract)", "ReviTLS (Disable Retract)",
]
HELDOUT_ORDER = [
    "ReviTLS (full)", "w/o 14B verifier", "w/o cross-encoder", "w/o conditional gate", "w/o structural probe",
    "overwrite only (no P_k)", "w/o tombstone purge", "rho = LLM flag only", "rho = lexicon only", "rho = none",
    "LLM-TLS (7B)", "LLM-TLS + 90d + latest + 14B",
]
PRETTY = {"CHRONOS (7B)": "CHRONOS (7B, impl.-limited)", "CHRONOS (14B)": "CHRONOS (14B, impl.-limited)"}
GROUPS = [
    ("revision/explicit", "Revision, explicit cue"), ("revision/implicit", "Revision, implicit cue"),
    ("retraction/explicit", "Retraction, explicit cue"), ("retraction/implicit", "Retraction, implicit cue"),
    ("neg_denial/explicit", "Neg.\\ control: denial"), ("neg_progress/explicit", "Neg.\\ control: follow-up"),
]


def esc(s: str) -> str:
    return s.replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def f3(x: float | None) -> str:
    return PENDING if x is None else f"{x:.3f}"


def write_macro(lines: list[str], name: str, value: str) -> None:
    lines.append(rf"\newcommand{{\{name}}}{{{value}}}")


def mean_std(vals: list[float]) -> tuple[float, float]:
    return statistics.fmean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def corpus_table(summary: dict[str, Any], ds: str, systems: list[str], metrics: list[tuple[str, str, bool]]) -> tuple[list[str], dict[str, dict[str, tuple[float, float]]]]:
    per_topic = summary["datasets"][ds]["per_topic"]
    topics = sorted(per_topic)
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for sys_name in systems:
        if all(sys_name in per_topic[t] for t in topics):
            stats[sys_name] = {m: mean_std([per_topic[t][sys_name][m] for t in topics]) for m, _, _ in metrics}
    best: dict[str, float] = {}
    second: dict[str, float] = {}
    for m, _, higher in metrics:
        vals = sorted({round(v[m][0], 6) for v in stats.values()}, reverse=higher)
        if vals:
            best[m] = vals[0]
            second[m] = vals[1] if len(vals) > 1 else vals[0]
    rows = []
    for sys_name in systems:
        if sys_name not in stats:
            continue
        cells = []
        for m, _, higher in metrics:
            mu, sd = stats[sys_name][m]
            txt = f"{mu:.3f}" if m != "time_s" else f"{mu:.1f}"
            sdtxt = f"{{\\tiny$\\pm${sd:.3f}}}" if m != "time_s" else ""
            r = round(mu, 6)
            if r == best[m]:
                txt = rf"\textbf{{{txt}}}"
            elif r == second[m]:
                txt = rf"\underline{{{txt}}}"
            cells.append(txt + sdtxt)
        rows.append(f"  & {esc(PRETTY.get(sys_name, sys_name))} & " + " & ".join(cells) + r" \\")
    return rows, stats


def main_table(summary: dict[str, Any] | None) -> str:
    head = (
        "\\begin{tabular}{@{}llcccccc@{}}\n\\toprule\n"
        "\\textbf{Corpus} & \\textbf{Method} & \\textbf{AR-1} $\\uparrow$ & \\textbf{AR-2} $\\uparrow$ & "
        "\\textbf{Concat-R1} $\\uparrow$ & \\textbf{Concat-R2} $\\uparrow$ & \\textbf{Date-F1} $\\uparrow$ & \\textbf{Time (s)} $\\downarrow$ \\\\\n\\midrule\n"
    )
    if summary is None:
        return head + f"\\multicolumn{{8}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    metrics = [("ar1", "AR-1", True), ("ar2", "AR-2", True), ("concat_r1", "C-R1", True), ("concat_r2", "C-R2", True), ("date_f1", "Date-F1", True), ("time_s", "Time", False)]
    body = []
    for ds, label in (("t17", "T17"), ("crisis", "Crisis")):
        if ds not in summary.get("datasets", {}):
            continue
        rows, _ = corpus_table(summary, ds, MAIN_ORDER, metrics)
        n = len(summary["datasets"][ds]["per_topic"])
        if rows:
            rows[0] = rf"\multirow{{{len(rows)}}}{{*}}{{\textbf{{{label}}} ($n={n}$)}}" + rows[0][1:]
            body += rows + [r"\midrule"]
    if body:
        body.pop()
    return head + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def paired_table(summary: dict[str, Any] | None) -> str:
    head = "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n\\textbf{Corpus} & \\textbf{Dual-Scale vs.} & \\textbf{Metric} & $\\Delta$ & \\textbf{Wilcoxon $p$} & \\textbf{Holm $p$} \\\\\n\\midrule\n"
    if summary is None:
        return head + f"\\multicolumn{{6}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    try:
        from scipy import stats
    except ImportError:
        return head + "\\multicolumn{6}{c}{scipy not installed} \\\\\n\\bottomrule\n\\end{tabular}\n"
    rows = []
    for ds, label in (("t17", "T17"), ("crisis", "Crisis")):
        per_topic = summary.get("datasets", {}).get(ds, {}).get("per_topic", {})
        topics = sorted(per_topic)
        if not topics or not all("ReviTLS (Dual-Scale)" in per_topic[t] for t in topics):
            continue
        entries = []
        for base in ("LLM-TLS (7B)", "LLM-TLS (14B)", "CHRONOS (7B)", "ReviTLS (Single-Scale)"):
            if not all(base in per_topic[t] for t in topics):
                continue
            for m, mlabel in (("date_f1", "Date-F1"), ("concat_r1", "Concat-R1")):
                a = [per_topic[t]["ReviTLS (Dual-Scale)"][m] for t in topics]
                b = [per_topic[t][base][m] for t in topics]
                diffs = [x - y for x, y in zip(a, b)]
                nz = [d for d in diffs if abs(d) > 1e-12]
                p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 and len(topics) >= 5 else float("nan")
                entries.append((label, base, mlabel, statistics.fmean(diffs), p))
        adj = holm_adjust([e[4] for e in entries])
        for (label_, base, mlabel, d, p), a_ in zip(entries, adj):
            ptxt = "--" if math.isnan(p) else f"{p:.3f}"
            atxt = "--" if math.isnan(a_) else f"{a_:.3f}"
            rows.append(f"{label_} & {esc(base)} & {mlabel} & {d:+.3f} & {ptxt} & {atxt} \\\\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def ablation_table(summary: dict[str, Any] | None) -> str:
    head = "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n\\textbf{Variant} & \\textbf{Date-F1} & \\textbf{AR-1} & \\textbf{Concat-R1} & \\textbf{Time (s)} \\\\\n\\midrule\n"
    t17 = (summary or {}).get("datasets", {}).get("t17", {}).get("per_topic")
    if not t17:
        return head + f"\\multicolumn{{5}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    topics = sorted(t17)
    rows = []
    for name in ABLATION_ORDER:
        if not all(name in t17[t] for t in topics):
            continue
        cells = []
        for m in ("date_f1", "ar1", "concat_r1", "time_s"):
            mu, sd = mean_std([t17[t][name][m] for t in topics])
            cells.append(f"{mu:.1f}" if m == "time_s" else f"{mu:.3f}{{\\tiny$\\pm${sd:.3f}}}")
        rows.append(f"{esc(name.replace('ReviTLS (', '').rstrip(')'))} & " + " & ".join(cells) + r" \\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def frag_table(summary: dict[str, Any] | None) -> str:
    head = "\\begin{tabular}{@{}llcccc@{}}\n\\toprule\n\\textbf{Corpus} & \\textbf{System} & \\textbf{Date-F1} & \\textbf{States} & \\textbf{Mean $c_k$} & \\textbf{States / ref.\\ date} \\\\\n\\midrule\n"
    if summary is None:
        return head + f"\\multicolumn{{6}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    rows = []
    for ds, label in (("t17", "T17"), ("crisis", "Crisis")):
        per_topic = summary.get("datasets", {}).get(ds, {}).get("per_topic", {})
        topics = sorted(per_topic)
        for name in ("LLM-TLS (7B)", "ReviTLS (Single-Scale)", "ReviTLS (Dual-Scale)"):
            if not topics or not all(name in per_topic[t] and "n_states" in per_topic[t][name] for t in topics):
                continue
            n_states = sum(per_topic[t][name]["n_states"] for t in topics)
            ref_dates = sum(per_topic[t][name]["n_ref_dates"] for t in topics)
            ck = statistics.fmean([per_topic[t][name]["mean_corroboration"] for t in topics])
            f1 = statistics.fmean([per_topic[t][name]["date_f1"] for t in topics])
            rows.append(f"{label} & {esc(name)} & {f1:.3f} & {n_states:,} & {ck:.2f} & {n_states / max(1, ref_dates):.2f} \\\\")
    if not rows:
        return head + f"\\multicolumn{{6}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def nobpoil_table(summary: dict[str, Any] | None) -> str:
    head = (
        "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n\\textbf{System} & \\textbf{Date-F1} & \\textbf{AR-1} & \\textbf{Concat-R1} \\\\\n\\midrule\n"
    )
    t17 = (summary or {}).get("datasets", {}).get("t17", {}).get("per_topic")
    if not t17:
        return head + f"\\multicolumn{{4}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    topics = sorted(t for t in t17 if t != "bpoil")
    rows = []
    for name in ("LLM-TLS (7B)", "LLM-TLS (14B)", "ReviTLS (Single-Scale)", "ReviTLS (Dual-Scale)"):
        if topics and all(name in t17[t] for t in topics):
            vals = [statistics.fmean([t17[t][name][m] for t in topics]) for m in ("date_f1", "ar1", "concat_r1")]
            rows.append(f"{esc(name)} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def heldout_table(held: dict[str, Any] | None) -> str:
    head = (
        "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n"
        "\\textbf{System} & \\textbf{Pos.\\ RRA} [95\\% CI] & \\textbf{SFLR} $\\downarrow$ & \\textbf{Neg.\\ pass} $\\uparrow$ & \\textbf{False retr.} $\\downarrow$ & \\textbf{Timeline RRA} \\\\\n\\midrule\n"
    )
    if held is None:
        return head + f"\\multicolumn{{6}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    rows = []
    for name in HELDOUT_ORDER:
        s = held["systems"].get(name)
        if not s:
            continue
        sm = s["summary"]
        pr, sf, npass, fr = sm["positive_resolved"], sm["positive_stale_leak"], sm["negative_pass"], sm["negative_false_retraction"]
        rows.append(
            f"{esc(name)} & {pr['rate']:.3f} [{pr['wilson95'][0]:.2f}, {pr['wilson95'][1]:.2f}] ({pr['k']}/{pr['n']}) & "
            f"{sf['rate']:.3f} & {npass['rate']:.3f} ({npass['k']}/{npass['n']}) & {fr['rate']:.3f} & {s['timeline_metrics']['timeline_rra']:.3f} \\\\"
        )
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def breakdown_table(held: dict[str, Any] | None) -> str:
    cols = ["ReviTLS (full)", "w/o structural probe", "w/o 14B verifier", "LLM-TLS + 90d + latest + 14B", "LLM-TLS (7B)"]
    short = ["Full", "w/o probe", "w/o 14B", "LLM-TLS+90d+14B", "LLM-TLS (7B)"]
    head = "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}\n\\toprule\n\\textbf{Case group} & " + " & ".join(f"\\textbf{{{c}}}" for c in short) + " \\\\\n\\midrule\n"
    if held is None:
        return head + f"\\multicolumn{{{len(cols) + 1}}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    rows = []
    for key, label in GROUPS:
        cells = []
        for c in cols:
            g = held["systems"].get(c, {}).get("summary", {}).get("by_group_resolved", {}).get(key)
            cells.append(f"{g['k']}/{g['n']}" if g else "--")
        rows.append(f"{label} & " + " & ".join(cells) + r" \\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def heldout_paired_table(held: dict[str, Any] | None) -> str:
    head = "\\begin{tabular}{@{}lccc@{}}\n\\toprule\n\\textbf{Variant vs.\\ full} & $\\Delta$ pass rate [95\\% CI] & McNemar $p$ & Holm $p$ \\\\\n\\midrule\n"
    pv = (held or {}).get("paired_vs_full")
    if not pv:
        return head + f"\\multicolumn{{4}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    rows = []
    for name in HELDOUT_ORDER:
        if name in pv:
            r = pv[name]
            d, lo, hi = r["diff_mean_ci95"]
            rows.append(f"{esc(name)} & {-d:+.3f} [{-hi:+.3f}, {-lo:+.3f}] & {r['mcnemar_p']:.4f} & {r['holm_p']:.4f} \\\\")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def macros(summary: dict[str, Any] | None, held: dict[str, Any] | None) -> str:
    L: list[str] = [r"% Auto-generated by scripts/make_paper_tables.py -- do not edit."]
    full = (held or {}).get("systems", {}).get("ReviTLS (full)")
    base = (held or {}).get("systems", {}).get("LLM-TLS + 90d + latest + 14B")
    base7 = (held or {}).get("systems", {}).get("LLM-TLS (7B)")
    meta = (held or {}).get("meta", {})
    write_macro(L, "HeldN", str(meta.get("n_cases", PENDING)))
    write_macro(L, "HeldPos", str(meta.get("n_positive", PENDING)))
    write_macro(L, "HeldNeg", str(meta.get("n_negative", PENDING)))
    write_macro(L, "HeldArticles", str(meta.get("n_articles", PENDING)))
    for tag, s in (("Full", full), ("BaseNinety", base), ("BaseSeven", base7)):
        if s:
            sm = s["summary"]
            pr = sm["positive_resolved"]
            write_macro(L, f"Held{tag}RRA", f"{pr['rate']:.3f}")
            write_macro(L, f"Held{tag}K", str(pr["k"]))
            write_macro(L, f"Held{tag}Lo", f"{pr['wilson95'][0]:.3f}")
            write_macro(L, f"Held{tag}Hi", f"{pr['wilson95'][1]:.3f}")
            write_macro(L, f"Held{tag}SFLR", f"{sm['positive_stale_leak']['rate']:.3f}")
            write_macro(L, f"Held{tag}NegPass", f"{sm['negative_pass']['rate']:.3f}")
            write_macro(L, f"Held{tag}FalseRetr", f"{sm['negative_false_retraction']['rate']:.3f}")
            g = sm["by_group_resolved"]
            for key, mac in (("revision/explicit", "RevExp"), ("revision/implicit", "RevImp"), ("retraction/explicit", "RetExp"), ("retraction/implicit", "RetImp")):
                if key in g:
                    write_macro(L, f"Held{tag}{mac}", f"{g[key]['k']}/{g[key]['n']}")
        else:
            for suffix in ("RRA", "K", "Lo", "Hi", "SFLR", "NegPass", "FalseRetr", "RevExp", "RevImp", "RetExp", "RetImp"):
                write_macro(L, f"Held{tag}{suffix}", PENDING)
    if summary:
        for ds, tag in (("t17", "T"), ("crisis", "C")):
            per_topic = summary.get("datasets", {}).get(ds, {}).get("per_topic", {})
            topics = sorted(per_topic)
            for sys_name, mac in (("ReviTLS (Dual-Scale)", "Dual"), ("LLM-TLS (7B)", "LSeven"), ("LLM-TLS (14B)", "LFourteen")):
                if topics and all(sys_name in per_topic[t] for t in topics):
                    write_macro(L, f"DateF{tag}{mac}", f"{statistics.fmean([per_topic[t][sys_name]['date_f1'] for t in topics]):.3f}")
                    write_macro(L, f"Time{tag}{mac}", f"{statistics.fmean([per_topic[t][sys_name]['time_s'] for t in topics]):.0f}")
                else:
                    write_macro(L, f"DateF{tag}{mac}", PENDING)
                    write_macro(L, f"Time{tag}{mac}", PENDING)
    else:
        for tag in ("T", "C"):
            for mac in ("Dual", "LSeven", "LFourteen"):
                write_macro(L, f"DateF{tag}{mac}", PENDING)
                write_macro(L, f"Time{tag}{mac}", PENDING)
    return "\n".join(L) + "\n"


def audit_macros(audit: dict[str, Any] | None) -> str:
    L: list[str] = []
    keys = (("Rev", "revised"), ("Ret", "retracted"), ("All", "overall"))
    for tag, key in keys:
        a = (audit or {}).get(key)
        if a and a.get("precision") is not None:
            write_macro(L, f"Audit{tag}Prec", f"{100 * a['precision']:.1f}")
            write_macro(L, f"Audit{tag}Lo", f"{100 * a['ci_lo']:.1f}")
            write_macro(L, f"Audit{tag}Hi", f"{100 * a['ci_hi']:.1f}")
            write_macro(L, f"Audit{tag}N", str(a["n_sampled"]))
            write_macro(L, f"Audit{tag}Pop", f"{a['n_population']:,}")
        else:
            for suf in ("Prec", "Lo", "Hi", "N", "Pop"):
                write_macro(L, f"Audit{tag}{suf}", PENDING)
    write_macro(L, "AuditKappa", f"{audit['cohens_kappa']:.2f}" if audit and audit.get("cohens_kappa") is not None else PENDING)
    write_macro(L, "AuditAgree", f"{100 * audit['raw_agreement']:.1f}" if audit and audit.get("raw_agreement") is not None else PENDING)
    return "\n".join(L) + "\n"


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return esc(text if len(text) <= n else text[: n - 1].rstrip() + "...")


def case_study_table(held: dict[str, Any] | None) -> str:
    head = (
        "\\begin{tabular}{@{}p{2.0cm}p{4.6cm}p{4.9cm}p{4.9cm}@{}}\n\\toprule\n"
        "\\textbf{Case} & \\textbf{Initial report $\\rightarrow$ later article} & \\textbf{ReviTLS (full): final linked state(s)} & \\textbf{LLM-TLS (7B): final linked state(s)} \\\\\n\\midrule\n"
    )
    if held is None or "ReviTLS (full)" not in held.get("systems", {}) or "LLM-TLS (7B)" not in held.get("systems", {}):
        return head + f"\\multicolumn{{4}}{{c}}{{{PENDING}}} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    from revitls.revisions.heldout import HeldoutSpec, generate_heldout_suite

    articles, cases = generate_heldout_suite(HeldoutSpec(n_per_cell=held["meta"].get("n_per_cell", 10), seed=held["meta"]["seed"]))
    by_id = {a.article_id: a for a in articles}
    full = {r["case_id"]: r for r in held["systems"]["ReviTLS (full)"]["per_case"]}
    base = {r["case_id"]: r for r in held["systems"]["LLM-TLS (7B)"]["per_case"]}
    picks = []
    for kind, cue in (("revision", "explicit"), ("revision", "implicit"), ("retraction", "explicit"), ("neg_denial", "explicit")):
        c = next((c for c in cases if c.kind == kind and c.cue == cue and c.case_id in full), None)
        if c:
            picks.append(c)

    def fmt_states(r: dict[str, Any]) -> str:
        mark = r"\checkmark" if r["resolved"] else r"$\times$"
        if not r["states"]:
            return f"{mark}\\; (no linked state)"
        return f"{mark}\\; " + " / ".join(f"[{s['status']}] {s['date']}: {_clip(s['summary'], 150)}" for s in r["states"][:2])

    rows = []
    for c in picks:
        init = by_id[f"{c.case_id}_init_0"]
        rev = by_id[f"{c.case_id}_rev"]
        label = esc(f"{c.kind}, {c.cue}" if not c.is_negative else c.kind)
        rows.append(
            f"{label} & {_clip(init.content, 120)} $\\rightarrow$ ({rev.publish_date}) {_clip(rev.content, 150)} & {fmt_states(full[c.case_id])} & {fmt_states(base[c.case_id])} \\\\"
        )
        rows.append("\\addlinespace")
    return head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path, default=Path("outputs/final/summary.json"))
    ap.add_argument("--heldout", type=Path, default=Path("outputs/final/heldout.json"))
    ap.add_argument("--audit", type=Path, default=Path("outputs/final/audit_summary.json"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    args = ap.parse_args()
    summary, held, audit = load(args.summary), load(args.heldout), load(args.audit)
    args.out.mkdir(parents=True, exist_ok=True)
    files = {
        "macros.tex": macros(summary, held) + audit_macros(audit),
        "tab_case_study.tex": case_study_table(held),
        "tab_main.tex": main_table(summary),
        "tab_paired.tex": paired_table(summary),
        "tab_ablation_t17.tex": ablation_table(summary),
        "tab_frag.tex": frag_table(summary),
        "tab_nobpoil.tex": nobpoil_table(summary),
        "tab_heldout.tex": heldout_table(held),
        "tab_heldout_breakdown.tex": breakdown_table(held),
        "tab_heldout_paired.tex": heldout_paired_table(held),
    }
    for name, text in files.items():
        (args.out / name).write_text(text, encoding="utf-8")
    print(f"Wrote {len(files)} files to {args.out} (summary={'yes' if summary else 'no'}, heldout={'yes' if held else 'no'}, audit={'yes' if audit else 'no'})")


if __name__ == "__main__":
    main()
