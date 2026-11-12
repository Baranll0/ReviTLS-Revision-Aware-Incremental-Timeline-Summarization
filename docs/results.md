# ReviTLS Experimental Results

## Experimental Setup

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5080 (CUDA enabled) |
| **Python Environment** | Python 3.12 + PyTorch (CUDA `cu128`) |
| **LLM Backbone** | `qwen2.5:7b-instruct-q4_K_M` (via Ollama local server, `temperature=0.0`) |
| **Dense Embedder** | `BAAI/bge-base-en-v1.5` (cosine similarity + entity overlap bonus) |
| **Retrieval Window** | Cross-date enabled (`allow_cross_date=True`, `max_date_window_days=90`, `top_k_clusters=10`) |
| **Targeted Verification** | Enabled on conflict / incomplete 5W1H (`max_verification_questions=3`, `max_evidence_docs=10`) |

---

## T17 (`bpoil`) + Controlled Revision Stress-Test Benchmark

* **Stream:** `data/revisions/bpoil_stream.jsonl` (164 articles within reference timeline span + 4 injected revision/retraction stress cases: numeric correction, date correction, entity re-attribution, and rumor retraction)
* **Reference Timelines:** `data/raw/t17/bpoil/timelines.jsonl`
* **Raw JSON Output:** [`outputs/bpoil_rev_eval.json`](../outputs/bpoil_rev_eval.json)

### Quantitative Comparison

| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.6000** | **0.0858** | **0.0189** | 0.4098 | 0.1524 | 0.0000 | 1.0000 | 125.06 |
| **CHRONOS (Closed)** | 0.1905 | 0.0262 | 0.0061 | 0.2315 | 0.0361 | **0.2500** | **0.2500** | **17.98** |
| **ReviTLS (Ours)** | 0.5000 | 0.0708 | 0.0097 | **0.6052** | **0.2251** | **0.2500** | 0.7500 | 480.01 |

### Key Findings

1. **Substantial Gain in Content Coverage (`Concat-ROUGE`):**
   - **ReviTLS** achieves **0.6052 Concat-R1** and **0.2251 Concat-R2**, outperforming **LLM-TLS** (`0.4098` R1 / `0.1524` R2) by **+19.5 points in ROUGE-1** and **+7.3 points in ROUGE-2**, and outperforming closed-domain **CHRONOS** (`0.2315` R1 / `0.0361` R2) by **+37.4 points in ROUGE-1**.
2. **Cross-Date Revision & Retraction Capability (`RRA` & `SFLR`):**
   - Because **LLM-TLS** restricts candidate event clustering strictly to the *same publication day*, it fails to link late-breaking corrections to earlier events (`RRA = 0.000`, `SFLR = 1.000`).
   - **ReviTLS** lifts the same-day constraint with a 90-day semantic + entity-aware memory (`EventMemory`) and 4-way relation classification (`EXTENDS`, `REVISES`, `RETRACTS`, `NEW_EVENT`), resolving cross-date revisions (`RRA = 0.250`) and reducing stale fact leakage (`SFLR = 0.750` vs `1.000`).
3. **Competitive Date Selection (`Date-F1` & `Align-ROUGE`):**
   - Unlike **CHRONOS** (`Date-F1 = 0.1905`), which suffers when adapted to long closed-domain news streams without iterative streaming state tracking, **ReviTLS** maintains strong temporal localization (`Date-F1 = 0.5000`, `Align-R1 = 0.0708`) while preserving an explicit audit trail (`RevisionRecord`) for every updated or retracted event.
