# Step-D Experimental Report (Multi-Run Stats, True Latency, Fragmentation)

> **Statistical note:** Treat this report's three-repeat outcomes as descriptive only. The McNemar $p$-values pool repeated runs of the same deterministic scenarios and are not valid independent-sample inference; they are withdrawn for manuscript claims. Use the single-count expanded paired analysis ($n=32$) in `step_e_summary.json` and the Step-E manuscript table instead.

## 1. 3-Run Uncached Revision Ablation (Mean +/- Std & McNemar p vs Dual-Scale)

| System | Isolated-16 RRA | McNemar p | Embedded-16 RRA | McNemar p | Parametric-8 RRA | Implicit-8 RRA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dual-Scale** | 0.875±0.000 | 1.0 | 1.000±0.000 | 1.0 | 1.000±0.000 | 0.375±0.000 |
| **w/o 14B Verifier** | 0.688±0.000 | 0.0352 | 0.688±0.000 | 0.0001 | 0.500±0.000 | 0.375±0.000 |
| **w/o Cross-Encoder** | 0.875±0.000 | 1.0 | 0.938±0.000 | 0.25 | 1.000±0.000 | 0.375±0.000 |
| **w/o Conditional Gate** | 0.938±0.000 | 0.25 | 0.938±0.000 | 0.25 | 1.000±0.000 | 0.875±0.000 |
| **Overwrite (No P_k)** | 0.812±0.000 | 0.5078 | 0.812±0.000 | 0.0039 | 0.875±0.000 | 0.375±0.000 |
| **LLM-TLS + 90d + Latest + 14B** | 0.312±0.000 | 0.0 | 0.312±0.000 | 0.0 | 0.250±0.000 | 0.250±0.000 |

## 2. True Uncached Latency & Cross-Encoder Recall@1 (16 Embedded)

| Mode | RRA | Total (s) | Mean Art. Lat (s) | P95 Lat (s) | CE Recall@1 | CE Calls/Flips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deferred Phase-2 (Uncached)** | 0.875 | 564.1 | 1.798 | 2.824 | None (0/0) | 89/20 |
| **Online Immediate (Uncached)** | 0.875 | 643.9 | 2.515 | 7.216 | None (0/0) | 103/17 |
| **Deferred w/o CE (Uncached)** | 0.875 | 566.4 | 1.803 | 2.890 | None (0/0) | 0/0 |

## 3. T17 & Crisis Fragmentation + Guarded/Disable Retract

### T17

| System | Date-F1 | AR-1 | Concat-R1 | Total States | Mean c_k | Nodes/RefDate | Retracted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (7B)** | 0.542 | 0.111 | 0.402 | 399 | 5.25 | 0.75 | 0 |
| **ReviTLS (Dual-Scale)** | 0.544 | 0.120 | 0.446 | 2886 | 1.17 | 5.82 | 12 |
| **ReviTLS (Guarded Retract)** | 0.556 | 0.121 | 0.445 | 2886 | 1.17 | 5.82 | 9 |
| **ReviTLS (Disable Retract)** | 0.552 | 0.120 | 0.446 | 2886 | 1.17 | 5.82 | 0 |

### CRISIS

| System | Date-F1 | AR-1 | Concat-R1 | Total States | Mean c_k | Nodes/RefDate | Retracted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LLM-TLS (7B)** | 0.320 | 0.099 | 0.428 | 73 | 21.54 | 0.17 | 0 |
| **ReviTLS (Dual-Scale)** | 0.241 | 0.075 | 0.487 | 10504 | 1.18 | 24.62 | 33 |
| **ReviTLS (Guarded Retract)** | 0.241 | 0.075 | 0.487 | 10504 | 1.18 | 24.62 | 28 |
| **ReviTLS (Disable Retract)** | 0.241 | 0.075 | 0.487 | 10504 | 1.18 | 24.62 | 0 |
