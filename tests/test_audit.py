"""Sanity checks for the audit summarizer."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from summarize_random_audit import cohen_kappa, label_of, weighted_precision  # noqa: E402


def _row(ds, topic, tr, n_pop, a1, a2, adj=""):
    return {
        "dataset": ds, "topic": topic, "transition": tr, "population_stratum_n": str(n_pop),
        "annotator_1_tp_fp": a1, "annotator_2_tp_fp": a2, "adjudicated_tp_fp": adj, "notes": "",
    }


def test_weighted_precision_uses_population_weights() -> None:
    rows = [_row("t17", "a", "REVISED", 90, "TP", "TP") for _ in range(5)]
    rows += [_row("t17", "b", "REVISED", 10, "FP", "FP") for _ in range(5)]
    est = weighted_precision(rows)
    assert est["precision"] == 0.9  # 0.9*1.0 + 0.1*0.0, not the unweighted 0.5
    assert est["n_sampled"] == 10


def test_disagreement_needs_adjudication_and_kappa() -> None:
    assert label_of(_row("d", "t", "REVISED", 5, "TP", "FP")) is None
    assert label_of(_row("d", "t", "REVISED", 5, "TP", "FP", "FP")) == "FP"
    assert cohen_kappa([("TP", "TP"), ("FP", "FP"), ("TP", "FP"), ("FP", "FP")]) is not None
    assert cohen_kappa([("TP", "TP")] * 3) is None  # degenerate: pe == 1
