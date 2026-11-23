"""Run the held-out revision suite over ReviTLS variants and baselines.

Usage (GPU machine, Ollama serving the 7B and 14B models):
    python scripts/run_heldout_suite.py --output outputs/final/heldout.json

Plumbing check without any model (stub LLM + hashing embedder; numbers are meaningless):
    python scripts/run_heldout_suite.py --dry-run --limit-cases 12 --output outputs/final/heldout_dry.json

The script checkpoints after every system and resumes from an existing output file.
"""

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from revitls.baselines.llm_tls import TopicLLMTLSBaseline  # noqa: E402
from revitls.config import ReviTLSConfig, load_config  # noqa: E402
from revitls.evaluation.revision_metrics import (  # noqa: E402
    evaluate_cases,
    evaluate_revision_states,
    evaluate_revision_timeline,
)
from revitls.evaluation.stats import exact_mcnemar_p, holm_adjust, paired_bootstrap_diff, wilson_ci  # noqa: E402
from revitls.pipeline.runner import ReviTLSPipeline  # noqa: E402
from revitls.revisions.heldout import HeldoutSpec, generate_heldout_suite  # noqa: E402


class StubLLM:
    """Offline stand-in: every call falls back to the pipeline's deterministic default payload."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return ""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {}

    def complete_json_safe(self, system_prompt: str, user_prompt: str, default: dict[str, Any]) -> dict[str, Any]:
        return dict(default)


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for the dry run."""

    DIM = 256

    def encode(self, sentences: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        def vec(text: str) -> np.ndarray:
            v = np.zeros(self.DIM)
            for tok in text.lower().split():
                v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.DIM] += 1.0
            n = np.linalg.norm(v)
            return v / n if (normalize_embeddings and n > 0) else v

        return vec(sentences) if isinstance(sentences, str) else np.vstack([vec(s) for s in sentences])


def build_systems(
    cfg_dual: ReviTLSConfig, llm_7b: Any, llm_14b: Any, embedder: Any
) -> list[tuple[str, Callable[[], Any]]]:
    cfg_no_ce = cfg_dual.model_copy(deep=True)
    cfg_no_ce.retrieval.cross_encoder_model = None
    cfg_no_gate = cfg_dual.model_copy(deep=True)
    cfg_no_gate.retrieval.strict_temporal_gate = False
    cfg_no_probe = cfg_dual.model_copy(deep=True)
    cfg_no_probe.retrieval.implicit_probe = False

    def revitls(cfg: ReviTLSConfig = cfg_dual, verifier: Any = llm_14b, **kw: Any) -> Callable[[], Any]:
        return lambda: ReviTLSPipeline(cfg, llm=llm_7b, embedder=embedder, verifier_llm=verifier, **kw)

    return [
        ("ReviTLS (full)", revitls()),
        ("w/o 14B verifier", revitls(verifier=llm_7b)),
        ("w/o cross-encoder", revitls(cfg_no_ce)),
        ("w/o conditional gate", revitls(cfg_no_gate)),
        ("w/o structural probe", revitls(cfg_no_probe)),
        ("overwrite only (no P_k)", revitls(overwrite_only=True)),
        ("w/o tombstone purge", revitls(disable_purge=True)),
        ("rho = LLM flag only", revitls(rho_mode="llm")),
        ("rho = lexicon only", revitls(rho_mode="lexicon")),
        ("rho = none", revitls(rho_mode="none")),
        ("LLM-TLS (7B)", lambda: TopicLLMTLSBaseline(llm=llm_7b, embedder=embedder)),
        (
            "LLM-TLS + 90d + latest + 14B",
            lambda: TopicLLMTLSBaseline(
                llm=llm_7b, embedder=embedder, window_days=90, selection_mode="latest", post_hoc_verifier_llm=llm_14b
            ),
        ),
    ]


def _rate_block(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    k = sum(int(r[key]) for r in rows)
    lo, hi = wilson_ci(k, n)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "wilson95": [round(lo, 4), round(hi, 4)]}


def summarize(per_case: list[dict[str, Any]]) -> dict[str, Any]:
    pos = [r for r in per_case if r["kind"] in {"revision", "retraction"}]
    neg = [r for r in per_case if r["kind"] in {"neg_denial", "neg_progress"}]
    out: dict[str, Any] = {
        "positive_resolved": _rate_block(pos, "resolved"),
        "positive_stale_leak": _rate_block(pos, "stale_leak"),
        "negative_pass": _rate_block(neg, "resolved"),
        "negative_false_retraction": _rate_block(neg, "false_retraction"),
    }
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in per_case:
        groups[f"{r['kind']}/{r['cue']}"].append(r)
        if r["revision_type"]:
            groups[f"type:{r['revision_type']}/{r['cue']}"].append(r)
    out["by_group_resolved"] = {g: _rate_block(rs, "resolved") for g, rs in sorted(groups.items())}
    return out


def run(args: argparse.Namespace) -> None:
    spec = HeldoutSpec(n_per_cell=args.n_per_cell, seed=args.seed)
    articles, cases = generate_heldout_suite(spec)
    if args.limit_cases:
        keep = {c.case_id for c in cases[: args.limit_cases]}
        cases = [c for c in cases if c.case_id in keep]
        articles = [
            a for a in articles
            if any(a.article_id == k or a.article_id.startswith(k + "_") or a.article_id.startswith("bg_" + k + "_") for k in keep)
        ]

    cfg_dual = load_config(args.config)
    cfg_dual.timeline.deduplicate_dates = False  # timeline-level metrics should not depend on date collisions
    if args.dry_run:
        cfg_dual.retrieval.cross_encoder_model = None
        cfg_dual.embedding.device = "cpu"
        llm_7b = llm_14b = StubLLM()
        embedder: Any = HashEmbedder()
    else:
        from sentence_transformers import SentenceTransformer

        from revitls.llm.client import LLMClient

        llm_7b = LLMClient(cfg_dual.llm, use_cache=False)
        llm_14b = LLMClient(cfg_dual.verifier_llm, use_cache=False) if cfg_dual.verifier_llm else llm_7b
        embedder = SentenceTransformer(cfg_dual.embedding.model_name, device=cfg_dual.embedding.device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {}
    if out_path.exists() and not args.fresh:
        result = json.loads(out_path.read_text(encoding="utf-8"))
    result["meta"] = {
        "suite": "heldout_v1",
        "seed": args.seed,
        "n_per_cell": args.n_per_cell,
        "n_articles": len(articles),
        "n_cases": len(cases),
        "n_positive": sum(not c.is_negative for c in cases),
        "n_negative": sum(c.is_negative for c in cases),
        "config": args.config,
        "dry_run": args.dry_run,
        "llm": cfg_dual.llm.model,
        "verifier_llm": cfg_dual.verifier_llm.model if cfg_dual.verifier_llm else None,
    }
    systems_out: dict[str, Any] = result.setdefault("systems", {})

    wanted = set(args.systems.split(",")) if args.systems else None
    for name, factory in build_systems(cfg_dual, llm_7b, llm_14b, embedder):
        if (wanted and name not in wanted) or name in systems_out:
            continue
        t0 = time.perf_counter()
        runner = factory()
        timeline, states = runner.run_stream(list(articles), num_dates=10**6)
        elapsed = time.perf_counter() - t0
        per_case = evaluate_cases(states, cases)
        diag = getattr(runner, "last_diagnostics", {}) or {}
        systems_out[name] = {
            "seconds": round(elapsed, 1),
            "n_states": len(states),
            "state_metrics": evaluate_revision_states(states, cases),
            "timeline_metrics": evaluate_revision_timeline(timeline, cases),
            "summary": summarize(per_case),
            "per_case": per_case,
            "diagnostics": {
                k: diag.get(k)
                for k in (
                    "actions", "rho_triggered", "rho_llm_flag", "rho_lexicon_flag", "retracted_count", "revised_count",
                    "retraction_quotes_validated", "retraction_labels_rejected_without_verbatim_evidence",
                    "routine_non_top1_matches", "routine_no_match_decisions",
                )
            },
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[{name}] {elapsed:.0f}s  pos_resolved={systems_out[name]['summary']['positive_resolved']['rate']}", flush=True)

    # Paired comparisons against the full system (positives and negatives pooled as pass/fail).
    full = systems_out.get("ReviTLS (full)")
    if full:
        ref = [int(r["resolved"]) for r in full["per_case"]]
        names, pvals = [], []
        for name, s in systems_out.items():
            if name == "ReviTLS (full)":
                continue
            other = [int(r["resolved"]) for r in s["per_case"]]
            names.append(name)
            pvals.append(exact_mcnemar_p(ref, other))
        adj = holm_adjust(pvals)
        result["paired_vs_full"] = {
            n: {
                "mcnemar_p": round(p, 5),
                "holm_p": round(a, 5),
                "diff_mean_ci95": [round(v, 4) for v in paired_bootstrap_diff(ref, [int(r["resolved"]) for r in systems_out[n]["per_case"]])],
            }
            for n, p, a in zip(names, pvals, adj)
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/hybrid_revitls.yaml")
    ap.add_argument("--output", default="outputs/final/heldout.json")
    ap.add_argument("--n-per-cell", type=int, default=10)
    ap.add_argument("--seed", type=int, default=HeldoutSpec.seed)
    ap.add_argument("--systems", default="", help="comma-separated subset of system names")
    ap.add_argument("--limit-cases", type=int, default=0, help="smoke-test: use only the first N cases")
    ap.add_argument("--dry-run", action="store_true", help="stub LLM + hashing embedder, no GPU/Ollama needed")
    ap.add_argument("--fresh", action="store_true", help="ignore an existing output file")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
