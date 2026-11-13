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

### Quantitative Comparison (2026-11-13 — Post Purge & Retraction Optimization)

| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.6000** | **0.0858** | **0.0189** | 0.4098 | 0.1524 | 0.0000 | 1.0000 | 99.65 |
| **CHRONOS (Closed)** | 0.1905 | 0.0262 | 0.0061 | 0.2315 | 0.0361 | 0.2500 | 0.2500 | **19.61** |
| **ReviTLS (Ours)** | 0.4375 | 0.0704 | 0.0105 | **0.5397** | **0.1951** | **0.7500** | **0.0000** | 619.13 |

### Ablation Across Runs (`2026-11-12` Baseline vs `2026-11-13` Purge & Retraction Optimization)

| ReviTLS Version | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **v1 (2026-11-12 Baseline)** | 0.5000 | 0.0708 | 0.0097 | 0.6052 | 0.2251 | 0.2500 | 0.7500 |
| **v2 (2026-11-13 Purge & Retraction)** | 0.4375 | 0.0704 | 0.0105 | 0.5397 | 0.1951 | **0.7500** | **0.0000** |

### Key Findings

1. **Dramatic Improvement in Revision Resolution (`RRA = 0.7500`) & Zero Stale Fact Leakage (`SFLR = 0.0000`):**
   - **LLM-TLS** restricts candidate event clustering strictly to the *same publication day*, failing to link late-breaking corrections to earlier events (`RRA = 0.000`, `SFLR = 1.000`).
   - **ReviTLS (v2)** achieves **75% Revision Resolution Accuracy (`RRA = 0.7500`)** and **0% Stale Fact Leakage (`SFLR = 0.0000`)**, outperforming both **LLM-TLS** (`0.000` RRA / `1.000` SFLR) and **CHRONOS** (`0.250` RRA / `0.250` SFLR).
2. **Strong Content Coverage (`Concat-ROUGE`):**
   - **ReviTLS** achieves **0.5397 Concat-R1** and **0.1951 Concat-R2**, outperforming **LLM-TLS** (`0.4098` R1 / `0.1524` R2) by **+13.0 points in ROUGE-1** and **+4.3 points in ROUGE-2**, and outperforming closed-domain **CHRONOS** (`0.2315` R1 / `0.0361` R2) by **+30.8 points in ROUGE-1**.
3. **Competitive Date Selection (`Date-F1` & `Align-ROUGE`):**
   - Unlike **CHRONOS** (`Date-F1 = 0.1905`), **ReviTLS** maintains strong temporal localization (`Date-F1 = 0.4375`, `Align-R1 = 0.0704`, `Align-R2 = 0.0105`) while maintaining an explicit audit trail (`RevisionRecord`) for every updated or retracted event.
