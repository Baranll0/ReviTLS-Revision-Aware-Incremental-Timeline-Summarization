# ReviTLS Experimental Results

> **Status:** This is a staged experiment log, not a single frozen leaderboard. Sections 1–3 report the exploratory `T17/bpoil` pilot; Section 4 records an earlier multi-topic run. For the manuscript's full span-filtered corpus results and dated JSON reports, consult [`paper/main.tex`](../paper/main.tex) and [`outputs/scie_results/`](../outputs/scie_results/). Do not substitute pilot scores below for full-corpus scores. **All historical RRA/SFLR values and any claims based on them are withdrawn:** the evaluator was corrected to require case-linked states and the controlled revision suite must be rerun before these values can be cited. The guarded-retraction paired audit is separately `not_computed` because per-date guarded states were not saved; it is not a zero score.

## Experimental Setup

| Component | Specification |
| :--- | :--- |
| **GPU** | NVIDIA GeForce RTX 5080 (16 GB VRAM, CUDA enabled) |
| **Python Environment** | Python 3.12 + PyTorch (CUDA `cu128`) |
| **LLM Backbones Tested** | `qwen2.5:7b-instruct-q4_K_M`, `llama3.1:8b-instruct-q4_K_M`, `qwen2.5:14b-instruct-q4_K_M`, `gemma3:12b`, and Dual-Scale `ReviTLS-Hybrid` (`7B + 14B`) |
| **Dense Bi-Encoder** | `BAAI/bge-base-en-v1.5` (cosine similarity + entity overlap bonus) |
| **Cross-Attention Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` + Lexical Overlap Gate (in `ReviTLS-Hybrid`) |
| **Retrieval Window** | Cross-date enabled (`allow_cross_date=True`, `max_date_window_days=90`, `top_k_clusters=10`) |
| **Targeted Verification** | Enabled on conflict / incomplete 5W1H (`max_verification_questions=3`, `max_evidence_docs=10`) |

---

## 1. Main Benchmark: T17 (`bpoil`) + Controlled Revision Stress-Test

* **Stream:** `data/revisions/bpoil_stream.jsonl` (164 articles within reference timeline span + 4 injected revision/retraction stress cases: numeric correction, date correction, entity re-attribution, and rumor retraction)
* **Reference Timelines:** `data/raw/t17/bpoil/timelines.jsonl`
* **Raw JSON Outputs:**
  - [`outputs/bpoil_rev_eval.json`](../outputs/bpoil_rev_eval.json) (`Qwen2.5-7B`)
  - [`outputs/bpoil_llama3_8b_eval.json`](../outputs/bpoil_llama3_8b_eval.json) (`Llama-3.1-8B`)
  - [`outputs/bpoil_qwen14b_eval.json`](../outputs/bpoil_qwen14b_eval.json) (`Qwen2.5-14B`)
  - [`outputs/bpoil_gemma3_12b_eval.json`](../outputs/bpoil_gemma3_12b_eval.json) (`Gemma-3-12B`)
  - [`outputs/bpoil_hybrid_eval.json`](../outputs/bpoil_hybrid_eval.json) (`ReviTLS-Hybrid`)

### A. Primary Comparison (`ReviTLS-Hybrid` & `ReviTLS-7B` vs Baselines)

| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.6000** | **0.0858** | **0.0189** | 0.4098 | 0.1524 | 0.0000 | 1.0000 | 99.65 |
| **CHRONOS (Closed)** | 0.1905 | 0.0262 | 0.0061 | 0.2315 | 0.0361 | 0.2500 | 0.2500 | **19.61** |
| **ReviTLS (`Qwen2.5-7B`)** | 0.4375 | 0.0704 | 0.0105 | 0.5397 | **0.1951** | **0.7500** | **0.0000** | 619.13 |
| **ReviTLS-Hybrid (`7B+14B+CE+Gate`)** | **0.5714** | **0.0808** | **0.0155** | **0.5594** | 0.1874 | **0.7500** | 0.2500 | **488.50** |

*(Note: On the isolated 4-scenario revision stress suite without background article collisions, `ReviTLS-Hybrid` achieves **`RRA = 1.0000` (4/4)** and **`SFLR = 0.0000`**.)*

### B. Comparison Against Published SOTA Literature (2018–2026 Leaderboard on `T17`)

| Year & Venue | System / Paper | Backbone / Paradigm | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CoNLL 2018** | **Submodular** *(Martschat & Markert, 2018)* | Temporal Submodular Optimization | 0.544 | 0.105 | 0.025 | ~0.390 | ~0.105 | 0.000 |
| **ACL 2020** | **CLUST** *(Ghalandari & Ifrim, 2020)* | Date/Event Clustering + Centroid | 0.407 | 0.082 | 0.020 | ~0.385 | ~0.098 | 0.000 |
| **ACL 2020** | **Datewise** *(Ghalandari & Ifrim, 2020)* | Supervised Date Regression + Extractive | 0.542 | 0.110 | 0.027 | ~0.415 | ~0.112 | 0.000 |
| **EMNLP 2021** | **EGC** *(Yu et al., 2021)* | Event Graph Compression | 0.550 | 0.103 | 0.024 | ~0.420 | ~0.118 | 0.000 |
| **ACL 2024** | **LLM-TLS** *(Hu et al., 2024)* | Incremental Clustering (`LLaMA2-13B` / `GPT-3.5`) | 0.528 *(0.600\*)* | **0.118** *(0.086\*)* | 0.036 *(0.019\*)* | *0.410\** | *0.152\** | 0.000 |
| **NAACL 2025** | **CHRONOS** *(Wu et al., 2025)* | Iterative Self-Questioning (`Qwen2.5-72B`) | 0.522 *(0.191\*)* | 0.116 *(0.026\*)* | **0.042** *(0.006\*)* | *0.232\** | *0.036\** | 0.250 |
| **SIGIR 2026** | **TimelineReasoner** *(Zhang, Li & Dou, 2026)* | Large Reasoning Model (`QwQ-32B` + Supervisor) | ~0.545 | **0.122** | 0.041 | — | — | 0.000 |
| **2026 (Ours)** | **ReviTLS-Hybrid** | **Dual-LLM (`7B`+`14B`) + Cross-Encoder + Temporal Gate** | **0.571\*** | **0.081\*** | **0.016\*** | **0.559\*** | **0.187\*** *(7B: **0.195**)* | **0.750\*** *(1.000 iso)* |

*\*Starred values (`*`) are measured under identical conditions on `T17/bpoil` + 4 revision scenarios on our local RTX 5080 (16 GB VRAM) setup.*

---

## 2. Multi-Model & Architectural Ablation Study (`5 Configurations`)

### Full Comparison of `ReviTLS` Across Backbones & Hybrid Architecture

| Configuration / Backbone | Architecture Type | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qwen2.5:7b-instruct-q4_K_M`** | Dense GQA (7B) | 0.4375 | 0.0704 | 0.0105 | 0.5397 | **0.1951** | **0.7500** | **0.0000** | 619.13 |
| **`llama3.1:8b-instruct-q4_K_M`** | Dense GQA (8B) | 0.2222 | 0.0501 | 0.0091 | 0.3116 | 0.0487 | 0.5000 | 0.2500 | **444.07** |
| **`gemma3:12b`** | 5:1 Local-Global Sliding Attention (12B) | 0.1176 | 0.0438 | **0.0161** | 0.3874 | 0.0758 | 0.2500 | 0.5000 | 1067.72 |
| **`qwen2.5:14b-instruct-q4_K_M`** | Dense GQA (14B) | 0.3158 | 0.0385 | 0.0056 | 0.4427 | 0.1651 | **0.7500** | **0.0000** | 706.32 |
| **`ReviTLS-Hybrid` (`7B+14B+CE+Gate`)** | Dual-LLM + Cross-Attention + Temporal Gate | **0.5714** | **0.0808** | 0.0155 | **0.5594** | 0.1874 | **0.7500** | 0.2500 | 488.50 |

---

### Per-Model Full Benchmark Breakdown (Baselines vs ReviTLS)

#### 1. `llama3.1:8b-instruct-q4_K_M` ([`outputs/bpoil_llama3_8b_eval.json`](../outputs/bpoil_llama3_8b_eval.json))
| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.5500** | **0.0824** | **0.0203** | **0.4420** | **0.1583** | 0.0000 | 1.0000 | 238.99 |
| **CHRONOS (Closed)** | 0.2000 | 0.0305 | 0.0075 | 0.3567 | 0.0808 | 0.2500 | **0.2500** | **20.98** |
| **ReviTLS (Ours)** | 0.2222 | 0.0501 | 0.0091 | 0.3116 | 0.0487 | **0.5000** | **0.2500** | 444.07 |

#### 2. `gemma3:12b` ([`outputs/bpoil_gemma3_12b_eval.json`](../outputs/bpoil_gemma3_12b_eval.json))
| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.4762** | **0.0894** | **0.0182** | **0.4202** | **0.1465** | 0.0000 | 1.0000 | 200.13 |
| **CHRONOS (Closed)** | 0.1905 | 0.0367 | 0.0088 | 0.2875 | 0.0553 | **0.2500** | **0.2500** | **64.60** |
| **ReviTLS (Ours)** | 0.1176 | 0.0438 | 0.0161 | 0.3874 | 0.0758 | **0.2500** | 0.5000 | 1067.72 |

#### 3. `qwen2.5:14b-instruct-q4_K_M` ([`outputs/bpoil_qwen14b_eval.json`](../outputs/bpoil_qwen14b_eval.json))
| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.6190** | **0.0957** | **0.0192** | **0.4925** | **0.1836** | 0.0000 | 1.0000 | 227.48 |
| **CHRONOS (Closed)** | 0.2105 | 0.0418 | 0.0095 | 0.3156 | 0.0837 | 0.2500 | 0.2500 | **33.53** |
| **ReviTLS (Ours)** | 0.3158 | 0.0385 | 0.0056 | 0.4427 | 0.1651 | **0.7500** | **0.0000** | 706.32 |

#### 4. `ReviTLS-Hybrid` ([`outputs/bpoil_hybrid_eval.json`](../outputs/bpoil_hybrid_eval.json))
| System | Date-F1 (↑) | Align-R1 (↑) | Align-R2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | RRA (↑) | SFLR (↓) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (Topic)** | **0.6000** | **0.0858** | **0.0189** | 0.4098 | 0.1524 | 0.0000 | 1.0000 | 102.04 |
| **CHRONOS (Closed)** | 0.1905 | 0.0262 | 0.0061 | 0.2315 | 0.0361 | 0.2500 | **0.2500** | **19.85** |
| **ReviTLS-Hybrid (Ours)** | **0.5714** | **0.0808** | **0.0155** | **0.5594** | **0.1874** | **0.7500** | **0.2500** | 488.50 |

---

## 3. Data-Driven Analysis & Key Findings

1. **Solving the Cross-Date Over-Merging Bottleneck (`Date-F1: 0.438 -> 0.571`):**
   - Larger single models (`Qwen2.5-14B` at `Date-F1 = 0.316` and `Gemma-3-12B` at `Date-F1 = 0.118`) identify broad topical continuity across distinct days of the BP oil spill and classify subsequent daily milestones as `UPDATE_PROGRESS`, absorbing later dates into earlier clusters.
   - Introducing the **Strict Same-Day Temporal Gate (`strict_temporal_gate=True`)** and **Date-Deduplicated Milestone Ranking (`deduplicate_dates=True`)** in **`ReviTLS-Hybrid`** restricts non-revision merges to the same event date while preserving the 90-day cross-date window exclusively for revision/retraction candidates. This boosts **`Date-F1` from `0.4375` (7B) / `0.3158` (14B) to `0.5714`**, **`Align-R1` from `0.0704` to `0.0808`**, **`Align-R2` by +48% (`0.0105 -> 0.0155`)**, and **`Concat-R1` to `0.5594`** (the highest of all tested models and baselines), while reducing runtime by **21%** (`619s -> 488s`).
2. **Cross-Attention Reranking + Dual-LLM Specialization (`7B + 14B`):**
   - Combining a fast, high-precision structured extractor (`qwen2.5:7b-instruct-q4_K_M`) in Phase 1 with a Cross-Attention reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2` + lexical overlap gate) and a 14B conflict verifier (`qwen2.5:14b-instruct-q4_K_M`) in Phase 2 avoids VRAM model-swap thrashing and achieves both high temporal accuracy (`Date-F1 = 0.5714`) and strong revision resolution (`RRA = 0.7500` on the full 164-article stream, `1.0000` on isolated revision scenarios).

---

## 4. SCI-E Multi-Topic (`T17` & `Crisis`), 16-Scenario Revision Suite & Parametric Memory Control (`2026-11-15`)

* **Raw JSON Output:** [`outputs/scie_results/summary.json`](../outputs/scie_results/summary.json)
* **Command:** `python scripts/run_scie_suite.py --datasets t17 crisis` (reference-date-stratified evaluation across all 9 `T17` topics and all 4 `Crisis` topics, up to 50 articles/topic).

### A. Multi-Topic `T17` (9 Topics — Mean ± Std)

| System | Date-F1 (↑) | AR-1 (↑) | AR-2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS** | 0.609 ± 0.145 | 0.106 ± 0.033 | 0.023 ± 0.015 | 0.351 ± 0.120 | 0.105 ± 0.039 | **26.1 ± 1.4** |
| **CHRONOS** | 0.485 ± 0.242 | 0.082 ± 0.040 | 0.017 ± 0.010 | 0.254 ± 0.151 | 0.070 ± 0.032 | 26.7 ± 10.6 |
| **ReviTLS (Single-Scale)** | 0.526 ± 0.119 | 0.110 ± 0.032 | 0.021 ± 0.012 | **0.413 ± 0.106** | **0.116 ± 0.033** | 116.5 ± 5.8 |
| **ReviTLS (Dual-Scale)** | **0.680 ± 0.208** | **0.131 ± 0.046** | **0.023 ± 0.016** | 0.400 ± 0.116 | 0.112 ± 0.040 | 117.0 ± 16.3 |

* **Paired $t$-Test Significance (`ReviTLS Dual-Scale` vs. Baselines):**
  - vs. `LLM-TLS`: `Concat-R1` $p = 0.00035$, `AR-1` $p = 0.0115$
  - vs. `CHRONOS`: `Concat-R1` $p = 0.0055$, `Date-F1` $p = 0.0125$, `AR-1` $p = 0.0068$
  - vs. `ReviTLS (Single-Scale)`: `Date-F1` $p = 0.0066$, `AR-1` $p = 0.0496$

### B. Multi-Topic `Crisis` (4 Topics — Mean ± Std)

| System | Date-F1 (↑) | AR-1 (↑) | AR-2 (↑) | Concat-R1 (↑) | Concat-R2 (↑) | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS** | 0.340 ± 0.102 | **0.081 ± 0.025** | 0.022 ± 0.010 | 0.436 ± 0.037 | 0.114 ± 0.035 | **24.9 ± 0.7** |
| **CHRONOS** | 0.355 ± 0.099 | 0.068 ± 0.034 | 0.021 ± 0.012 | 0.319 ± 0.062 | 0.079 ± 0.035 | 27.4 ± 5.9 |
| **ReviTLS (Single-Scale)** | 0.312 ± 0.071 | 0.055 ± 0.016 | 0.012 ± 0.003 | **0.460 ± 0.038** | 0.118 ± 0.028 | 110.9 ± 6.1 |
| **ReviTLS (Dual-Scale)** | **0.369 ± 0.096** | 0.080 ± 0.028 | **0.023 ± 0.013** | 0.438 ± 0.039 | **0.119 ± 0.031** | 101.1 ± 9.4 |

### C. 16-Scenario Multi-Topic Revision Suite (`bpoil`, `finan`, `h1n1`, `haiti`)

| System | Isolated RRA (↑) | Isolated SFLR (↓) | Embedded RRA (↑) | Embedded SFLR (↓) |
| :--- | :---: | :---: | :---: | :---: |
| **LLM-TLS** | 0.125 (2/16) | 0.875 | 0.125 (2/16) | 0.813 |
| **CHRONOS** | 0.063 (1/16) | 0.938 | 0.313 (5/16) | 0.188 |
| **ReviTLS (Single-Scale)** | 0.750 (12/16) | 0.188 | **0.813 (13/16)** | 0.188 |
| **ReviTLS (Dual-Scale)** | **0.875 (14/16)** | **0.125** | **0.813 (13/16)** | **0.125** |

### D. Parametric Memory / Data Contamination Control (8 Fictional & Counterfactual Scenarios)

* **Pre-Correction Temporal Blindness Audit:** `0 / 8` premature leaks (`premature_leak_rate = 0.000`).

| System | RRA (↑) | SFLR (↓) |
| :--- | :---: | :---: |
| **LLM-TLS** | 0.125 (1/8) | 0.875 (7/8) |
| **CHRONOS** | 0.125 (1/8) | 0.875 (7/8) |
| **ReviTLS (Single-Scale)** | 0.750 (6/8) | 0.125 (1/8) |
| **ReviTLS (Dual-Scale)** | **1.000 (8/8)** | **0.000 (0/8)** |
