"""Interactive Streamlit Demo & Inspection Dashboard for ReviTLS."""

import json
from datetime import date
from pathlib import Path
import streamlit as st

from revitls.config import load_config
from revitls.llm.client import LLMClient
from revitls.pipeline.runner import ReviTLSPipeline
from revitls.revisions.benchmark import inject_revision_scenarios
from revitls.schemas import Article, EventState, EventStatus

st.set_page_config(page_title="ReviTLS — Revision-Aware TLS", layout="wide")


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
            c1, c2, c3 = st.columns(3)
            c1.metric("Cluster ID", st_obj.cluster_id)
            c2.metric("Corroborations", st_obj.corroboration_count)
            c3.metric("Revisions", len(st_obj.revision_history))

            if st_obj.revision_history:
                st.markdown("**Revision Audit Log:**")
                for idx, rec in enumerate(st_obj.revision_history, start=1):
                    st.info(
                        f"**#{idx} [{rec.action.value}"
                        f"{f' / {rec.revision_type.value}' if rec.revision_type else ''}]** "
                        f"(Article `{rec.source_article_id}`)\n\n"
                        f"- **Before ({rec.previous_date}):** {rec.previous_summary}\n"
                        f"- **After ({rec.new_date}):** {rec.new_summary}\n"
                        f"- **Rationale:** {rec.rationale or 'N/A'}"
                    )


def main() -> None:
    st.title("ReviTLS: Revision-Aware Incremental Timeline Summarization")
    st.caption("Interactive timeline evolution, contradiction resolution, and benchmark inspector")

    tab_live, tab_bench = st.tabs(["1. Incremental Stream & Revisions", "2. Benchmark Results Inspector"])

    with tab_live:
        with st.sidebar:
            st.header("Pipeline Settings")
            mode = st.radio(
                "Execution Mode",
                ["Live Local LLM (Ollama/vLLM)", "Offline Demo (No GPU Required)"],
            )
            config_path = st.text_input("Config YAML", value="configs/default.yaml")
            show_retracted = st.checkbox("Show RETRACTED rumors in inspector", value=True)
            max_articles = st.slider("Max articles to process", min_value=5, max_value=200, value=25)

        stream_options = sorted(
            [str(p) for p in Path("data/processed").rglob("*.jsonl")]
            + [str(p) for p in Path("data/revisions").glob("*_stream.jsonl")]
        )
        selected_stream = st.selectbox(
            "Select article stream (or use built-in synthetic revision demo)",
            ["<Built-in 4-Type Revision Demo>"] + stream_options,
        )

        if st.button("Run Incremental Timeline", type="primary"):
            if selected_stream == "<Built-in 4-Type Revision Demo>":
                articles, _cases = inject_revision_scenarios(
                    [], topic="demo_crisis", anchor_date=date(2010, 4, 20)
                )
            else:
                with Path(selected_stream).open("r", encoding="utf-8") as f:
                    articles = [Article.model_validate_json(line) for line in f if line.strip()]
                articles.sort(key=lambda a: (a.publish_date, a.article_id))

            articles = articles[:max_articles]

            if mode == "Offline Demo (No GPU Required)":
                from tests.test_end_to_end import DeterministicBagEmbedder, DeterministicRevisionMockLLM

                cfg = load_config(config_path)
                pipeline = ReviTLSPipeline(
                    cfg,
                    llm=DeterministicRevisionMockLLM(),
                    embedder=DeterministicBagEmbedder(),
                )
            else:
                from sentence_transformers import SentenceTransformer

                cfg = load_config(config_path)
                llm = LLMClient(cfg.llm)
                embedder = SentenceTransformer(cfg.embedding.model_name, device=cfg.embedding.device)
                pipeline = ReviTLSPipeline(cfg, llm=llm, embedder=embedder)

            progress = st.progress(0.0, text="Processing chronological stream...")
            for idx, art in enumerate(articles, start=1):
                pipeline.step(art)
                progress.progress(idx / len(articles), text=f"Processed {idx}/{len(articles)}: {art.publish_date}")

            states = pipeline.memory.all_states()
            st.success(
                f"Done! {len(states)} event clusters "
                f"({sum(1 for s in states if s.status == EventStatus.REVISED)} revised, "
                f"{sum(1 for s in states if s.status == EventStatus.RETRACTED)} retracted)."
            )
            _render_states(states, show_retracted=show_retracted)

    with tab_bench:
        output_files = sorted(str(p) for p in Path("outputs").glob("*.json")) if Path("outputs").exists() else []
        if not output_files:
            st.info("No saved benchmark JSON files found under `outputs/` yet. Run `scripts/evaluate_benchmark.py` first.")
        else:
            chosen = st.selectbox("Select benchmark output JSON", output_files)
            payload = json.loads(Path(chosen).read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "macro" in payload:
                st.subheader("Macro-Average Comparison")
                st.dataframe(payload["macro"], use_container_width=True)
                st.subheader("Per-Topic Breakdown")
                for topic, rows in payload.get("per_topic", {}).items():
                    with st.expander(f"Topic: {topic}"):
                        st.dataframe(rows, use_container_width=True)
            else:
                st.dataframe(payload, use_container_width=True)


if __name__ == "__main__":
    main()
