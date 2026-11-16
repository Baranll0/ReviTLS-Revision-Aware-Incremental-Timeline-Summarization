# Step-C Experimental Report (Online Latency, Safe Purge, Window Sensitivity)

## 1. Online Immediate vs. Deferred Phase-2 Reconciliation (16 Embedded)

| Mode | RRA [95% CI] | SFLR | Total (s) | Mean Art. Latency (s) | P95 Latency (s) | CE Calls / Flips |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Deferred Phase-2 (Default)** | 0.938 [0.717, 0.989] | 0.062 | 7.5 | 0.010 | 0.016 | 215 / 65 |
| **Online Immediate Phase-2** | 0.875 [0.64, 0.965] | 0.062 | 385.5 | 0.588 | 3.047 | 211 / 64 |

## 2. Claim-Span Purge & Guarded Retraction

| Variant | T17/egypt Date-F1 | Concat-R1 | Retracted | 2011-02-11 Retracted? | Embedded-16 RRA | Embedded-16 SFLR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Default (Clause + Unchecked Retract)** | 0.900 | 0.472 | 1 | True | 0.938 | 0.062 |
| **Safe (Claim-Span + Guarded Retract)** | 0.900 | 0.452 | 0 | False | 0.688 | 0.312 |

## 3. Lookback Window Sensitivity (7 / 30 / 60 / 90 / 180 Days)

| Window | Embedded-16 RRA | Embedded-16 SFLR | Implicit 15-35d RRA | Implicit 15-35d SFLR |
| :--- | :---: | :---: | :---: | :---: |
| **7d** | 0.812 | 0.188 | 0.375 | 0.625 |
| **30d** | 0.938 | 0.062 | 0.375 | 0.625 |
| **60d** | 0.938 | 0.062 | 0.375 | 0.625 |
| **90d** | 0.938 | 0.062 | 0.375 | 0.625 |
| **180d** | 0.938 | 0.062 | 0.375 | 0.625 |
