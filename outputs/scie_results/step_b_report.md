# Step-B Experimental Report (RRA/SFLR Ablation, Rho Source, Implicit/Long-Delay)

## 1. Component Ablation on Revision & Parametric Suites

| Variant | Isolated RRA [95% CI] | Isolated SFLR | Embedded RRA [95% CI] | Embedded SFLR | Parametric RRA [95% CI] | Parametric SFLR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Dual-Scale** | 0.938 [0.717, 0.989] | 0.062 | 0.938 [0.717, 0.989] | 0.062 | 1.000 [0.676, 1.0] | 0.000 |
| **w/o 14B Verifier** | 0.812 [0.57, 0.934] | 0.125 | 0.562 [0.332, 0.769] | 0.375 | 0.625 [0.306, 0.863] | 0.250 |
| **w/o Cross-Encoder** | 0.938 [0.717, 0.989] | 0.062 | 0.875 [0.64, 0.965] | 0.125 | 1.000 [0.676, 1.0] | 0.000 |
| **w/o Conditional Gate** | 0.938 [0.717, 0.989] | 0.062 | 0.875 [0.64, 0.965] | 0.062 | 1.000 [0.676, 1.0] | 0.000 |
| **w/o Tombstone Purge** | 0.250 [0.102, 0.495] | 0.688 | 0.250 [0.102, 0.495] | 0.688 | 0.250 [0.071, 0.591] | 0.750 |
| **LLM-TLS + 90d + Latest + 14B** | 0.312 [0.142, 0.556] | 0.688 | 0.125 [0.035, 0.36] | 0.875 | 0.375 [0.137, 0.694] | 0.500 |

## 2. Conditional Gate (rho) Source Ablation

| Rho Mode | Isolated RRA [95% CI] | Isolated SFLR | Parametric RRA [95% CI] | Parametric SFLR |
| :--- | :---: | :---: | :---: | :---: |
| **rho_both (LLM or Lex)** | 0.938 [0.717, 0.989] | 0.062 | 1.000 [0.676, 1.0] | 0.000 |
| **rho_llm_only** | 0.938 [0.717, 0.989] | 0.062 | 1.000 [0.676, 1.0] | 0.000 |
| **rho_lexicon_only** | 0.875 [0.64, 0.965] | 0.125 | 1.000 [0.676, 1.0] | 0.000 |
| **rho_none (Always Same-Day)** | 0.062 [0.011, 0.283] | 0.938 | 0.125 [0.022, 0.471] | 0.875 |

## 3. Implicit-Cue & Long-Delay (15-35 Days, 8 Scenarios)

| System | RRA [95% CI] | SFLR [95% CI] |
| :--- | :---: | :---: |
| **LLM-TLS (7B)** | 0.125 [0.022, 0.471] | 0.750 [0.409, 0.929] |
| **LLM-TLS + 90d + Latest + 14B** | 0.125 [0.022, 0.471] | 0.750 [0.409, 0.929] |
| **ReviTLS (rho_llm_only)** | 0.375 [0.137, 0.694] | 0.625 [0.306, 0.863] |
| **ReviTLS (Full Dual-Scale)** | 0.375 [0.137, 0.694] | 0.625 [0.306, 0.863] |
