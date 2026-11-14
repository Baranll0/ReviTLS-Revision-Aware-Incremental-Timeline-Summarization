"""Interactive Streamlit Demo & Research Presentation Dashboard for ReviTLS."""

import json
from datetime import date
from pathlib import Path
import pandas as pd
import streamlit as st

from revitls.config import load_config
from revitls.llm.client import LLMClient
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.revisions.benchmark import inject_revision_scenarios
from revitls.schemas import Article, EventState, EventStatus

st.set_page_config(
    page_title="ReviTLS: Revision-Aware Incremental TLS",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PUBLISHED_LITERATURE = {
    "Crisis Dataset (Closed-Domain TLS)": pd.DataFrame(
        [
            {"System": "CLUST (ACL 2020)", "Date-F1": 0.226, "Align-R1": 0.061, "Align-R2": 0.013, "Source": "Hu et al., 2024"},
            {"System": "EGC (EMNLP 2021)", "Date-F1": 0.291, "Align-R1": 0.079, "Align-R2": 0.015, "Source": "Hu et al., 2024"},
            {"System": "LLM-TLS (ACL 2024 - LLaMA2-13B)", "Date-F1": 0.329, "Align-R1": 0.112, "Align-R2": 0.032, "Source": "Hu et al., 2024"},
            {"System": "CHRONOS (NAACL 2025 - Qwen2.5-72B)", "Date-F1": 0.323, "Align-R1": 0.108, "Align-R2": 0.045, "Source": "Wu et al., 2025"},
        ]
    ),
    "T17 Dataset (Closed-Domain TLS)": pd.DataFrame(
        [
            {"System": "CLUST (ACL 2020)", "Date-F1": 0.407, "Align-R1": 0.082, "Align-R2": 0.020, "Source": "Hu et al., 2024"},
            {"System": "EGC (EMNLP 2021)", "Date-F1": 0.550, "Align-R1": 0.103, "Align-R2": 0.024, "Source": "Hu et al., 2024"},
            {"System": "LLM-TLS (ACL 2024 - LLaMA2-13B)", "Date-F1": 0.528, "Align-R1": 0.118, "Align-R2": 0.036, "Source": "Hu et al., 2024"},
            {"System": "CHRONOS (NAACL 2025 - Qwen2.5-72B)", "Date-F1": 0.522, "Align-R1": 0.116, "Align-R2": 0.042, "Source": "Wu et al., 2025"},
        ]
    ),
}

PARADIGM_COMPARISON = pd.DataFrame(
    [
        {
            "Feature": "Execution Mode",
            "LLM-TLS (ACL 2024)": "Streaming / Incremental",
            "CHRONOS (NAACL 2025)": "Static / Batch RAG rounds",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "Streaming / Dual-Stage Incremental",
        },
        {
            "Feature": "Memory & Event Nodes",
            "LLM-TLS (ACL 2024)": "Static immutable nodes",
            "CHRONOS (NAACL 2025)": "Stateless round timelines",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "Mutable EventState + Audit Log",
        },
        {
            "Feature": "Candidate Retrieval & Gating",
            "LLM-TLS (ACL 2024)": "Strict same-date heuristic",
            "CHRONOS (NAACL 2025)": "Global BM25 / Elastic chunks",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "Strict Same-Day Gate + 90d Revision Gate + Cross-Encoder",
        },
        {
            "Feature": "Update / Conflict Logic",
            "LLM-TLS (ACL 2024)": "Binary ISSAMEEVENT + TextRank",
            "CHRONOS (NAACL 2025)": "Table 12 Blind deduplication",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "4-Way Classification + 14B Targeted Verifier",
        },
        {
            "Feature": "Superseded Facts",
            "LLM-TLS (ACL 2024)": "Retained via TextRank popularity",
            "CHRONOS (NAACL 2025)": "Retained if repeated in rounds",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "Explicitly purged on resolution",
        },
        {
            "Feature": "False Rumors & Retractions",
            "LLM-TLS (ACL 2024)": "Cannot retract",
            "CHRONOS (NAACL 2025)": "Cannot retract",
            "ReviTLS & ReviTLS-Hybrid (Ours)": "Marked RETRACTED and filtered",
        },
    ]
)

MULTI_MODEL_ABLATION = pd.DataFrame(
    [
        {
            "Backbone / Architecture": "Qwen2.5-7B-Instruct",
            "Attention / Paradigm": "Dense GQA (7B)",
            "Date-F1": 0.4375,
            "Align-R1": 0.0704,
            "Align-R2": 0.0105,
            "Concat-R1": 0.5397,
            "Concat-R2": 0.1951,
            "RRA (↑)": 0.7500,
            "SFLR (↓)": 0.0000,
            "Time (s)": 619.1,
        },
        {
            "Backbone / Architecture": "Llama-3.1-8B-Instruct",
            "Attention / Paradigm": "Dense GQA (8B)",
            "Date-F1": 0.2222,
            "Align-R1": 0.0501,
            "Align-R2": 0.0091,
            "Concat-R1": 0.3116,
            "Concat-R2": 0.0487,
            "RRA (↑)": 0.5000,
            "SFLR (↓)": 0.2500,
            "Time (s)": 444.1,
        },
        {
            "Backbone / Architecture": "Gemma-3-12B",
            "Attention / Paradigm": "5:1 Local-Global Sliding Attention (12B)",
            "Date-F1": 0.1176,
            "Align-R1": 0.0438,
            "Align-R2": 0.0161,
            "Concat-R1": 0.3874,
            "Concat-R2": 0.0758,
            "RRA (↑)": 0.2500,
            "SFLR (↓)": 0.5000,
            "Time (s)": 1067.7,
        },
        {
            "Backbone / Architecture": "Qwen2.5-14B-Instruct",
            "Attention / Paradigm": "Dense GQA (14B)",
            "Date-F1": 0.3158,
            "Align-R1": 0.0385,
            "Align-R2": 0.0056,
            "Concat-R1": 0.4427,
            "Concat-R2": 0.1651,
            "RRA (↑)": 0.7500,
            "SFLR (↓)": 0.0000,
            "Time (s)": 706.3,
        },
        {
            "Backbone / Architecture": "ReviTLS-Hybrid (7B+14B+CE+Gate)",
            "Attention / Paradigm": "Dual-LLM + Cross-Encoder + Temporal Gate",
            "Date-F1": 0.5714,
            "Align-R1": 0.0808,
            "Align-R2": 0.0155,
            "Concat-R1": 0.5594,
            "Concat-R2": 0.1874,
            "RRA (↑)": 0.7500,
            "SFLR (↓)": 0.2500,
            "Time (s)": 488.5,
        },
    ]
)


def _status_badge(status: EventStatus) -> str:
    if status == EventStatus.REVISED:
        return "🟡 REVISED"
    if status == EventStatus.RETRACTED:
        return "🔴 RETRACTED"
    return "🟢 ACTIVE"


def _render_states(states: list[EventState], show_retracted: bool) -> None:
    ordered = sorted(states, key=lambda s: (s.event_date or date.min, s.cluster_id))
    for st_obj in ordered:
        if st_obj.status == EventStatus.RETRACTED and not show_retracted:
            continue

        header = (
            f"{_status_badge(st_obj.status)} | **{st_obj.event_date}** — "
            f"{st_obj.canonical_summary}"
        )
        with st.expander(header, expanded=(st_obj.status != EventStatus.ACTIVE)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cluster ID", st_obj.cluster_id)
            c2.metric("Corroborations", st_obj.corroboration_count)
            c3.metric("Revisions", len(st_obj.revision_history))
            c4.metric("Status", st_obj.status.value)

            if st_obj.entities:
                st.write("**Entities:**", ", ".join(f"`{e}`" for e in st_obj.entities))

            if st_obj.revision_history:
                st.markdown("#### 📜 Revision Audit Trail")
                for idx, rec in enumerate(st_obj.revision_history, start=1):
                    type_str = f" / {rec.revision_type.value}" if rec.revision_type else ""
                    st.warning(
                        f"**Revision #{idx}: `{rec.action.value}{type_str}`** "
                        f"(Triggered by Article `{rec.source_article_id}`)\n\n"
                        f"- ❌ **Superseded / Old ({rec.previous_date}):** {rec.previous_summary}\n"
                        f"- ✅ **Current Resolved ({rec.new_date}):** {rec.new_summary}\n"
                        f"- 💡 **Rationale:** {rec.rationale or 'N/A'}"
                    )


def main() -> None:
    st.title("⏱️ ReviTLS: Revision-Aware Incremental Timeline Summarization")
    st.markdown(
        "**Research Demo & Benchmark Dashboard** — Dual-Scale (`7B+14B`), Cross-Attention Reranking, "
        "and Temporal Gating for Evolving Developments, Corrections, and Retractions."
    )

    tabs = st.tabs(
        [
            "🏛️ Architecture & Hybrid Design",
            "📊 Experimental Results & Multi-Model Ablation",
            "🔍 Timeline Stream & Revision Inspector",
            "📚 Published Literature Benchmarks",
        ]
    )

    # ---------------- TAB 1: ARCHITECTURE ----------------
    with tabs[0]:
        st.header("ReviTLS-Hybrid System Architecture")
        st.markdown(
            """
            **ReviTLS-Hybrid** bridges **LLM-TLS** (*Hu et al., ACL 2024*) and **CHRONOS** (*Wu et al., NAACL 2025*)
            while solving the *Cross-Date Over-Merging Bottleneck* of large LLMs via:
            1. **Phase 1 (`Qwen2.5-7B` Extractor):** Fast, high-precision structured event and entity extraction.
            2. **Strict Same-Day Temporal Gate + 90-Day Revision Window:** Restricts standard progression merges to the same date while opening the 90-day cross-date window exclusively for detected revision/retraction candidates.
            3. **Cross-Attention Reranker (`ms-marco-MiniLM-L-6-v2`) + Lexical Overlap Gate:** Rescores `[Existing State ⊕ Incoming Claim]` pairs.
            4. **Phase 2 (`Qwen2.5-14B` Classifier & Targeted Verifier):** Resolves conflicts, purges superseded facts, and enforces `RETRACTED` states without VRAM model-swap thrashing.
            """
        )

        st.markdown(
            """
            ```mermaid
            flowchart TD
                A["Incoming Article a_t (Chronological Stream)"] --> B["Phase 1: Qwen2.5-7B EventExtractor"]
                B --> C["Dense Bi-Encoder (bge-base) + 90-Day EventMemory"]
                C --> G1{"Revision / Correction Candidate?"}
                G1 -- "No (Standard News)" --> G2["Strict Same-Day Temporal Gate (d_cand == d_claim)"]
                G1 -- "Yes (Revision/Retraction)" --> G3["90-Day Cross-Date Candidate Pool"]
                G2 --> CE["Cross-Attention Reranker (MiniLM-L-6-v2) + Lexical Gate"]
                G3 --> CE
                CE --> F["Phase 2: Qwen2.5-14B 4-Way Revision Classifier"]
                F -- NEW_EVENT --> E["Create New EventState (ACTIVE)"]
                F -- CORROBORATE --> G["Increment Corroboration Count"]
                F -- UPDATE_PROGRESS --> H["Same-Day Milestone Progression"]
                F -- REVISE_CONFLICT --> I["14B Targeted Verifier (CHRONOS 5W1H Self-Questioning)"]
                I --> K["Purge Superseded Facts & Update EventDate"]
                K --> L{"Debunked Rumor?"}
                L -- Yes --> M["Set Status: RETRACTED"]
                L -- No --> N["Set Status: REVISED + Audit Trail"]
                E --> O["Date-Deduplicated Milestone Ranker (Filter RETRACTED)"]
                G --> O
                H --> O
                M --> O
                N --> O
                O --> P["Final Milestone Timeline T_topic"]
            ```
            """
        )

        st.subheader("Architectural Comparison Against Baseline Papers")
        st.dataframe(PARADIGM_COMPARISON, use_container_width=True, hide_index=True)

    # ---------------- TAB 2: BENCHMARK RESULTS ----------------
    with tabs[1]:
        st.header("Measured Experimental Benchmark Results (RTX 5080 16GB)")

        primary_df = pd.DataFrame(
            [
                {
                    "System": "LLM-TLS (Topic Baseline)",
                    "Date-F1": 0.6000,
                    "Align-R1": 0.0858,
                    "Align-R2": 0.0189,
                    "Concat-R1": 0.4098,
                    "Concat-R2": 0.1524,
                    "Revision Accuracy (RRA ↑)": 0.0000,
                    "Stale Fact Leakage (SFLR ↓)": 1.0000,
                    "Time (s)": 99.65,
                },
                {
                    "System": "CHRONOS (Closed Baseline)",
                    "Date-F1": 0.1905,
                    "Align-R1": 0.0262,
                    "Align-R2": 0.0061,
                    "Concat-R1": 0.2315,
                    "Concat-R2": 0.0361,
                    "Revision Accuracy (RRA ↑)": 0.2500,
                    "Stale Fact Leakage (SFLR ↓)": 0.2500,
                    "Time (s)": 19.61,
                },
                {
                    "System": "ReviTLS (Qwen2.5-7B)",
                    "Date-F1": 0.4375,
                    "Align-R1": 0.0704,
                    "Align-R2": 0.0105,
                    "Concat-R1": 0.5397,
                    "Concat-R2": 0.1951,
                    "Revision Accuracy (RRA ↑)": 0.7500,
                    "Stale Fact Leakage (SFLR ↓)": 0.0000,
                    "Time (s)": 619.13,
                },
                {
                    "System": "ReviTLS-Hybrid (7B+14B+CE+Gate)",
                    "Date-F1": 0.5714,
                    "Align-R1": 0.0808,
                    "Align-R2": 0.0155,
                    "Concat-R1": 0.5594,
                    "Concat-R2": 0.1874,
                    "Revision Accuracy (RRA ↑)": 0.7500,
                    "Stale Fact Leakage (SFLR ↓)": 0.2500,
                    "Time (s)": 488.50,
                },
            ]
        )

        st.subheader("1. Primary Comparison: `ReviTLS-Hybrid` & `ReviTLS-7B` vs Baselines (`T17/bpoil`)")
        st.dataframe(
            primary_df.style.highlight_max(
                subset=["Date-F1", "Align-R1", "Concat-R1", "Concat-R2", "Revision Accuracy (RRA ↑)"],
                color="#d4edda",
            ).highlight_min(subset=["Stale Fact Leakage (SFLR ↓)"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("#### 🎯 Revision Handling: RRA (↑) vs SFLR (↓)")
            st.bar_chart(primary_df.set_index("System")[["Revision Accuracy (RRA ↑)", "Stale Fact Leakage (SFLR ↓)"]])
        with chart_col2:
            st.markdown("#### 📅 Temporal & Content Quality: Date-F1 & Concat-R1")
            st.bar_chart(primary_df.set_index("System")[["Date-F1", "Concat-R1", "Concat-R2"]])

        st.subheader("2. 5-Way Multi-Model & Architectural Ablation Study")
        st.caption(
            "Comparison across Qwen2.5-7B, Llama-3.1-8B, Gemma-3-12B, Qwen2.5-14B, and Dual-Scale ReviTLS-Hybrid."
        )
        st.dataframe(
            MULTI_MODEL_ABLATION.style.highlight_max(
                subset=["Date-F1", "Align-R1", "Align-R2", "Concat-R1", "Concat-R2", "RRA (↑)"],
                color="#d4edda",
            ).highlight_min(subset=["SFLR (↓)", "Time (s)"], color="#d4edda"),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("3. Inspect Raw Benchmark JSON Files (`outputs/*.json`)")
        json_files = sorted(str(p) for p in Path("outputs").glob("*.json")) if Path("outputs").exists() else []
        if json_files:
            selected_json = st.selectbox("Select raw output file:", json_files)
            raw_rows = json.loads(Path(selected_json).read_text(encoding="utf-8"))
            st.dataframe(pd.DataFrame(raw_rows), use_container_width=True, hide_index=True)

    # ---------------- TAB 3: STREAM INSPECTOR ----------------
    with tabs[2]:
        st.header("Interactive Timeline Stream & EventState Inspector")

        with st.sidebar:
            st.header("Inspector Configuration")
            mode = st.radio(
                "Execution Mode",
                ["Offline Demo (Deterministic Mock)", "Live GPU (Ollama / vLLM)"],
            )
            config_path = st.selectbox(
                "Config YAML Path",
                [
                    "configs/hybrid_revitls.yaml",
                    "configs/default.yaml",
                    "configs/llama3_8b.yaml",
                    "configs/qwen_14b.yaml",
                    "configs/gemma3_12b.yaml",
                ],
            )
            show_retracted = st.checkbox("Display RETRACTED events in list", value=True)
            max_articles = st.slider("Max articles to ingest", min_value=5, max_value=200, value=25)

        stream_options = sorted(
            [str(p) for p in Path("data/processed").rglob("*.jsonl")]
            + [str(p) for p in Path("data/revisions").glob("*_stream.jsonl")]
        )
        selected_stream = st.selectbox(
            "Select article stream to inspect:",
            ["<Built-in 4-Type Revision Demo>"] + stream_options,
        )

        if st.button("▶️ Run / Re-run Timeline Pipeline", type="primary"):
            if selected_stream == "<Built-in 4-Type Revision Demo>":
                articles, _cases = inject_revision_scenarios(
                    [], topic="demo_crisis", anchor_date=date(2010, 4, 20)
                )
            else:
                with Path(selected_stream).open("r", encoding="utf-8") as f:
                    articles = [Article.model_validate_json(line) for line in f if line.strip()]
                articles.sort(key=lambda a: (a.publish_date, a.article_id))

            articles = articles[:max_articles]

            if mode == "Offline Demo (Deterministic Mock)":
                from tests.test_end_to_end import DeterministicBagEmbedder, DeterministicRevisionMockLLM

                cfg = load_config(config_path)
                cfg.retrieval.cross_encoder_model = None
                pipeline = ReviTLSPipeline(
                    cfg,
                    llm=DeterministicRevisionMockLLM(),
                    embedder=DeterministicBagEmbedder(),
                )
            else:
                from sentence_transformers import SentenceTransformer

                cfg = load_config(config_path)
                llm = LLMClient(cfg.llm)
                v_llm = LLMClient(cfg.verifier_llm) if cfg.verifier_llm else None
                embedder = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)
                pipeline = ReviTLSPipeline(cfg, llm=llm, embedder=embedder, verifier_llm=v_llm)

            prog = st.progress(0.0, text="Processing stream...")
            for idx, art in enumerate(articles, start=1):
                pipeline.step(art)
                prog.progress(idx / len(articles), text=f"Processed article {idx}/{len(articles)} ({art.publish_date})")

            states = pipeline.memory.all_states()
            rev_count = sum(1 for s in states if s.status == EventStatus.REVISED)
            ret_count = sum(1 for s in states if s.status == EventStatus.RETRACTED)
            st.success(
                f"Pipeline Complete: {len(states)} event clusters | "
                f"{rev_count} revised events | {ret_count} retracted rumors"
            )
            _render_states(states, show_retracted=show_retracted)

    # ---------------- TAB 4: LITERATURE BENCHMARKS ----------------
    with tabs[3]:
        st.header("Published Baselines from ACL 2024 & NAACL 2025 Literature")
        for dataset_name, df_pub in PUBLISHED_LITERATURE.items():
            st.subheader(dataset_name)
            st.dataframe(df_pub, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
