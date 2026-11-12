# Datasets Guide

This project benchmarks on the intersection of **LLM-TLS** and **CHRONOS**, plus a controlled **Revision-Stress Benchmark**.

## 1. Closed-Domain Benchmarks: `T17` & `Crisis`
Both LLM-TLS (ACL 2024) and CHRONOS (NAACL 2025) evaluate on **T17** and **Crisis**:
- **T17**: 9 topics (19 timelines), avg. ~508 articles/topic, avg. 212 days.
- **Crisis**: 4 topics (22 timelines), avg. ~2310 articles/topic, avg. 343 days.

**How to obtain:**
Clone the official `news-tls` repository (Ghalandari & Ifrim, ACL 2020):
```bash
git clone https://github.com/complementizer/news-tls.git
```
Copy/link the `T17` and `Crisis` directories into `data/raw/t17` and `data/raw/crisis`, then run `scripts/prepare_datasets.py` to convert articles into chronologically ordered streams.

## 2. Open-Domain / Modern News Benchmark: `Open-TLS`
Introduced by **CHRONOS** (NAACL 2025):
- 50 journalist-authored timelines (mostly post-2020) across Politics, Society, Economy, Sports, and Technology.

**How to obtain:**
Clone the official CHRONOS repository:
```bash
git clone https://github.com/Alibaba-NLP/CHRONOS.git
```
Ground-truth timelines reside under `data/open/{NEWS_KEYWORD}/timelines.jsonl`.

## 3. Controlled Revision-Aware Benchmark (`data/revisions/`)
Standard TLS benchmarks do not explicitly label contradictions, numeric corrections, date shifts, or retractions. To rigorously measure revision handling for publication, `data/revisions/` holds:
1. **Natural Revisions** extracted from evolving news articles in `Crisis` / `T17`.
2. **Controlled Counterfactual-then-Corrected Streams** across 4 revision types:
   - `NUMERIC_FACT_CORRECTION` (e.g., casualty/financial/vote updates)
   - `DATE_CORRECTION` (early rumored date vs. confirmed date)
   - `ENTITY_ATTRIBUTION_CORRECTION` (misattributed suspect/cause corrected later)
   - `RETRACTION` (early false claim later debunked)
