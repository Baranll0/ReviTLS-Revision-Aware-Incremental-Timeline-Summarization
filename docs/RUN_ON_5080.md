# Final run on the RTX 5080 machine

Everything below produces the numbers that the paper reads from `paper/generated/` (via `scripts/make_paper_tables.py`). Nothing is copied by hand.

**Freeze rule.** The held-out suite is only a held-out test if the system does not change after it is run. Once `heldout` starts, do not edit code, prompts, or configs. If something has to change, rerun every stage and say so in the paper.

## 0. One-time setup

```powershell
git clone https://github.com/Baranll0/ReviTLS-Revision-Aware-Incremental-Timeline-Summarization
cd ReviTLS-Revision-Aware-Incremental-Timeline-Summarization
git checkout main

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev,gpu,eval]"

ollama pull qwen2.5:7b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct-q4_K_M

# T17 and Crisis corpora (needed for the t17 / crisis stages)
powershell -ExecutionPolicy Bypass -File scripts\download_datasets.ps1
python scripts/prepare_datasets.py --source data/raw/t17 --output data/processed/t17
python scripts/prepare_datasets.py --source data/raw/crisis --output data/processed/crisis
```

Optional speed-up: if the old machine state still has `outputs\.llm_deterministic_cache.jsonl`, copy it into `outputs\`. Baseline prompts did not change, so their calls are served from the cache (their runtime is still charged at the original latency). ReviTLS prompts changed, so those calls run fresh.

## 1. Smoke test (about 1 minute)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage check
```

This runs the unit tests, lists the Ollama models, and does a stub-LLM dry run of the held-out pipeline. Fix anything that fails before continuing.

## 2. Held-out revision suite (run this first)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage heldout
```

110 chains, about 600 articles, 12 systems, resumable. Budget roughly 20-40 minutes per ReviTLS variant (several hours in total). For a first look you can run only the key systems and finish the rest later:

```powershell
python scriptsun_heldout_suite.py --systems "ReviTLS (full),LLM-TLS (7B),LLM-TLS + 90d + latest + 14B,w/o structural probe"
```

The script skips systems that are already in the JSON, so the full run later continues from there. Output: `outputs/final/heldout.json` (per-case results for every system). This is the most informative result, so commit and push it as soon as it finishes:

```powershell
git add outputs/final paper/generated
git commit -m "results: held-out revision suite"
git push
```

## 3. T17 full corpus with ablations

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage t17
```

9 topics, 10 main systems plus 7 ablations. Resumable per (topic, system). Output: `outputs/final/summary.json`.

## 4. Crisis full corpus (long; run overnight)

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage crisis
```

4 topics, 13,937 articles. The previous implementation needed about 2 hours per topic for ReviTLS and about 1.3 hours for the 14B LLM-TLS baseline; expect more now because routine claims compare up to three candidates. Topics run smallest-first and the JSON is saved after every system, so partial results are usable. You can stop and restart at any time.

## 5. Natural-stream audit

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_final_5080.ps1 -Stage audit
```

This writes `outputs/final/transition_audit_random.csv` (stratified random sample, seed 20261123). Two people annotate it independently as described in `docs/transition_audit_protocol.md`, filling `annotator_1_tp_fp` and `annotator_2_tp_fp` with `TP`/`FP`. Use an extra `adjudicated_tp_fp` column for disagreements. Then:

```powershell
python scripts\summarize_random_audit.py
python scripts\make_paper_tables.py
```

The second annotator must not see the first annotator's labels.

## 6. Send the results back

```powershell
git add outputs/final paper/generated
git commit -m "results: full corpus, audit"
git push
```

Do not commit `outputs/.llm_deterministic_cache.jsonl` or the raw corpora (both are git-ignored).

## What each file feeds

| File | Feeds |
| --- | --- |
| `outputs/final/heldout.json` | Tables `heldout`, `heldout_breakdown`, `heldout_paired`, `case_study`; abstract and Section VI-B numbers |
| `outputs/final/summary.json` | Tables `main`, `paired`, `ablation_t17`, `frag`, `nobpoil`; transition audit sampling |
| `outputs/final/audit_summary.json` | Audit numbers in the abstract and Section VI-C |
| `outputs/final/env.json` | Exact commit, package versions, GPU, Ollama model digests |
