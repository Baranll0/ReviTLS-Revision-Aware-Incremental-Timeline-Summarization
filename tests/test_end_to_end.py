"""Offline end-to-end tests comparing Topic LLM-TLS vs ReviTLS on revision scenarios."""

from datetime import date
import re
from typing import Any
import numpy as np

from revitls.baselines.llm_tls import TopicLLMTLSBaseline
from revitls.config import ReviTLSConfig
from revitls.evaluation.revision_metrics import evaluate_revision_states
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.revisions.benchmark import inject_revision_scenarios


class DeterministicBagEmbedder:
    """Lightweight deterministic embedder mapping key scenario tokens into fixed dimensions."""

    VOCAB = ("macondo", "barrel", "oil", "bank", "hearing", "senate", "meridian", "valve", "glitch", "harbor", "pipeline")

    def encode(self, sentences: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        def _vec(text: str) -> np.ndarray:
            low = text.lower()
            v = np.array([1.0 if tok in low else 0.0 for tok in self.VOCAB], dtype=float)
            if v.sum() == 0:
                v[0] = 1.0
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                if norm > 0:
                    v = v / norm
            return v

        if isinstance(sentences, str):
            return _vec(sentences)
        return np.vstack([_vec(s) for s in sentences])


class DeterministicRevisionMockLLM:
    """Rule-based mock LLM simulating extraction, 4-way revision classification, and verification."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return ""

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        # 1. EventExtractor / TopicLLMTLS summarize
        if "Extract the primary event claim" in system_prompt or "most significant event" in system_prompt:
            pub_match = re.search(r"Publish Date:\s*(\d{4}-\d{2}-\d{2})", user_prompt)
            pub_date = pub_match.group(1) if pub_match else "2010-04-20"
            title_match = re.search(r"Title:\s*(.+)", user_prompt)
            content_match = re.search(r"Content:\n(.+)", user_prompt, flags=re.DOTALL)
            summary = content_match.group(1).strip().splitlines()[0] if content_match else (
                title_match.group(1).strip() if title_match else "Event"
            )
            evt_date = pub_date
            if "Meridian" in user_prompt or "valve" in user_prompt:
                entities = ["Meridian Plant"]
            elif "Harbor Terminal" in user_prompt:
                entities = ["Harbor Terminal"]
            elif "Macondo" in user_prompt or "barrels" in user_prompt:
                entities = ["Macondo Well"]
            else:
                entities = ["Senate Banking"]
                dates = re.findall(r"\d{4}-\d{2}-\d{2}", user_prompt)
                if len(dates) >= 2:
                    evt_date = dates[-1]
            return {
                "event_date": evt_date,
                "summary": summary,
                "entities": entities,
                "key_facts": {},
                "is_correction_or_update": "Correction" in user_prompt or "retraction" in user_prompt.lower(),
            }

        # 2. TopicLLMTLS binary ISSAMEEVENT
        if "refer to the same event" in system_prompt:
            s1, s2 = user_prompt.lower(), user_prompt.lower()
            same = ("macondo" in s1 and s1.count("macondo") >= 2) or (
                "harbor" in s1 and s1.count("harbor") >= 2
            )
            return {"same_event": same}

        # 3. ReviTLS 4-way RevisionClassifier
        if "Classify the relation into exactly one action" in system_prompt:
            low = user_prompt.lower()
            inc_line = low.split("incoming summary:", 1)[-1].splitlines()[0]
            is_corr = any(
                w in inc_line
                for w in ("revised", "correction", "retracted", "retraction", "debunked", "rule out", "correcting")
            )
            if not is_corr:
                if (
                    low.count("macondo") >= 2
                    or low.count("harbor") >= 2
                    or low.count("meridian") >= 2
                    or low.count("senate") >= 2
                    or low.count("regulatory panel") >= 2
                ):
                    return {"action": "CORROBORATE", "revision_type": None, "rationale": "Repeat report."}
                return {"action": "NEW_EVENT", "revision_type": None, "rationale": "Distinct event."}

            if "harbor terminal" in inc_line:
                return {"action": "REVISE_CONFLICT", "revision_type": "RETRACTION", "rationale": "Debunked rumor."}
            if "senate" in inc_line:
                return {"action": "REVISE_CONFLICT", "revision_type": "DATE_CORRECTION", "rationale": "Date corrected."}
            if "meridian" in inc_line:
                return {"action": "REVISE_CONFLICT", "revision_type": "ENTITY_ATTRIBUTION", "rationale": "Cause updated."}
            return {"action": "REVISE_CONFLICT", "revision_type": "NUMERIC_FACT", "rationale": "Numeric revision."}

        # 4. TargetedVerifier query generation
        if "Generate up to" in user_prompt:
            return {"queries": ["official correction report"]}

        # 5. TargetedVerifier evidence resolution
        if "most accurate current truth" in user_prompt:
            header = user_prompt.split("Supporting Evidence:")[0]
            if "harbor terminal" in header.lower():
                return {
                    "resolved_summary": "Authorities confirmed no pipeline incident occurred at Harbor Terminal.",
                    "resolved_date": None,
                    "status": "RETRACTED",
                    "rationale": "Official retraction.",
                }
            if "5,000 barrels" in header:
                return {
                    "resolved_summary": "Federal scientists revised the Macondo well leak to 5,000 barrels per day.",
                    "resolved_date": None,
                    "status": "REVISED",
                    "rationale": "Supersedes 1,000-barrel estimate.",
                }
            if "valve maintenance error" in header:
                return {
                    "resolved_summary": "Safety investigators confirmed a manual valve maintenance error caused the Meridian outage.",
                    "resolved_date": None,
                    "status": "REVISED",
                    "rationale": "Supersedes software glitch theory.",
                }
            inc_claim_line = header.split("Incoming Claim", 1)[-1]
            dates = re.findall(r"took place on (\d{4}-\d{2}-\d{2})", inc_claim_line) or re.findall(r"\d{4}-\d{2}-\d{2}", inc_claim_line)
            corrected = dates[0] if dates else "2010-05-02"
            return {
                "resolved_summary": f"Senate Banking Committee hearing occurred on {corrected}.",
                "resolved_date": corrected,
                "status": "REVISED",
                "rationale": "Corrected hearing date.",
            }

        raise AssertionError(f"Unhandled mock prompt: {system_prompt}")


def test_revitls_outperforms_llmtls_on_revision_benchmark() -> None:
    articles, cases = inject_revision_scenarios([], topic="demo_crisis", anchor_date=date(2010, 4, 20))
    llm = DeterministicRevisionMockLLM()
    embedder = DeterministicBagEmbedder()

    llmtls = TopicLLMTLSBaseline(llm=llm, embedder=embedder)
    _, llmtls_states = llmtls.run_stream(articles, num_dates=10)
    llmtls_scores = evaluate_revision_states(llmtls_states, cases)

    cfg = ReviTLSConfig()
    cfg.embedding.normalize = True
    revitls = ReviTLSPipeline(cfg, llm=llm, embedder=embedder)
    timeline, revitls_states = revitls.run_stream(articles, num_dates=10)
    revitls_scores = evaluate_revision_states(revitls_states, cases)

    assert revitls_scores["revision_resolution_accuracy"] == 1.0
    assert revitls_scores["stale_fact_leakage_rate"] == 0.0
    assert llmtls_scores["stale_fact_leakage_rate"] > revitls_scores["stale_fact_leakage_rate"]
    # Retracted Harbor Terminal rumor must not appear in emitted timeline
    assert all("harbor terminal" not in entry.summary.lower() for entry in timeline)
