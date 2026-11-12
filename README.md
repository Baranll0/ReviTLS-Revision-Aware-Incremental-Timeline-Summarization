# ReviTLS: Revision-Aware Incremental Timeline Summarization

**ReviTLS** is an incremental, revision-aware Topic Timeline Summarization (TLS) framework built to handle **new developments, numeric/date corrections, entity re-attributions, and rumor retractions** in evolving news streams.

It synthesizes and extends two LLM-based TLS paradigms:
- **LLM-TLS** (*Hu et al., ACL 2024*): Incremental streaming event graph construction.
- **CHRONOS** (*Wu et al., NAACL 2025*): Iterative 5W1H self-questioning and query rewriting, adapted here as a **targeted conflict-verification module**.

---

## Project Structure

```text
revision-aware-tls/
├── app/
│   └── streamlit_app.py             # Interactive demo & benchmark inspector UI
├── configs/
│   └── default.yaml                 # Default config tuned for RTX 5070 (12 GB VRAM)
├── data/                            # Raw & processed corpora (ignored by git)
├── docs/
│   └── architecture.md              # Technical design & gap analysis
├── scripts/
│   ├── download_datasets.ps1        # Downloads T17, Crisis, and Open-TLS
│   ├── prepare_datasets.py          # Normalizes raw .jsonl.gz into chronological streams
│   ├── build_revision_benchmark.py  # Injects controlled revision/retraction test cases
│   ├── run_pipeline.py              # Single-stream ReviTLS CLI runner
│   └── evaluate_benchmark.py        # Side-by-side LLM-TLS vs CHRONOS vs ReviTLS evaluator
├── src/revitls/                     # Core package
│   ├── baselines/                   # Fair closed-domain LLM-TLS & CHRONOS adapters
│   ├── evaluation/                  # Date-F1, Concat-ROUGE, Align-ROUGE, RRA, SFLR
│   ├── llm/                         # Resilient OpenAI-compatible client (retry + fallback)
│   ├── pipeline/                    # Extractor, 4-way Classifier, Verifier, StateManager, Ranker
│   ├── retrieval/                   # Cross-date semantic & entity-aware EventMemory
│   └── revisions/                   # Revision stress-test generator
└── tests/                           # Offline unit & end-to-end tests (no GPU needed)
```

---

## RTX 5070 (12 GB VRAM) Setup Guide

> **Note for GitHub:** `.gitignore` already excludes `data/raw/`, `data/processed/`, `data/revisions/`, and `outputs/`, so only code and configs are pushed. On the RTX 5070 machine, you will clone the repo and run `scripts/download_datasets.ps1`.

### 1. Clone & Create Python Environment

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd revision-aware-tls

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

pip install --upgrade pip
pip install -e ".[dev,gpu,ui]"
```

*(Optional, for official `tilse` scoring if needed: `pip install -e ".[eval]"`)*

### 2. Verify Offline Tests First (Sanity Check)

```bash
pytest
```

All unit and end-to-end mock tests should pass immediately without needing a GPU.

### 3. Start Local LLM Server (Ollama on RTX 5070)

Install [Ollama](https://ollama.com/) and pull the recommended 4-bit 7B model (fits comfortably in 12 GB VRAM alongside the `BAAI/bge-base-en-v1.5` embedder):

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
ollama serve
```

Check `configs/default.yaml` to make sure `llm.base_url` (`http://localhost:11434/v1`) and `llm.model` (`qwen2.5:7b-instruct-q4_K_M`) match your running server.

> **14B Ablation Tip:** If you want to test `qwen2.5:14b-instruct-q4_K_M` on a subset, set `embedding.device: "cpu"` in `configs/default.yaml` so the 14B model gets full 12 GB VRAM headroom.

### 4. Download & Prepare Benchmark Datasets

On Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
```

Normalize **T17** (9 topics) and **Crisis** (4 topics) into chronological `.jsonl` streams:

```bash
python scripts/prepare_datasets.py --source data/raw/t17 --output data/processed/t17
python scripts/prepare_datasets.py --source data/raw/crisis --output data/processed/crisis
```

---

## Running Experiments

### A. Quick Single-Topic Run + Revision Stress-Test (`T17/bpoil`)

1. Generate the controlled revision benchmark (numeric, date, entity attribution, and retraction cases):
   ```bash
   python scripts/build_revision_benchmark.py --stream data/processed/t17/bpoil.jsonl --output-dir data/revisions
   ```
2. Evaluate **LLM-TLS**, **CHRONOS**, and **ReviTLS** side-by-side:
   ```bash
   python scripts/evaluate_benchmark.py \
     --stream data/revisions/bpoil_stream.jsonl \
     --ref-timelines data/raw/t17/bpoil/timelines.jsonl \
     --revision-cases data/revisions/bpoil_cases.jsonl \
     --output-json outputs/bpoil_rev_eval.json
   ```

### B. Full Dataset Batch Evaluation (`T17` & `Crisis`)

Run all topics in a dataset (automatically filters articles to reference timeline spans and uses `l = avg(len(ref_dates))`):

```bash
# Full T17 (9 topics)
python scripts/evaluate_benchmark.py \
  --processed-dir data/processed/t17 \
  --raw-dir data/raw/t17 \
  --output-json outputs/t17_batch_eval.json

# Full Crisis (4 topics)
python scripts/evaluate_benchmark.py \
  --processed-dir data/processed/crisis \
  --raw-dir data/raw/crisis \
  --output-json outputs/crisis_batch_eval.json
```

### C. Run ReviTLS Alone on Any Stream

```bash
python scripts/run_pipeline.py \
  --stream data/processed/t17/bpoil.jsonl \
  --output outputs/bpoil_timeline.jsonl
```

---

## Interactive Presentation & Inspection UI

Launch the Streamlit dashboard to step through incremental updates, inspect `REVISED` / `RETRACTED` audit logs, or view saved benchmark tables:

```bash
streamlit run app/streamlit_app.py
```
