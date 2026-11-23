# ReviTLS Architecture

## 1. Positioning

| Aspect | LLM-TLS (ACL 2024) | CHRONOS (NAACL 2025) | ReviTLS |
| :--- | :--- | :--- | :--- |
| Mode | Incremental stream | Batch per query | Incremental stream, deferred reconciliation |
| Memory | Static event nodes | Per-round timelines | Mutable `EventState` + append-only revision log |
| Date linking | Same-date only | Round merging | Conditional dual horizon: same-day (routine) / 90 days (revision candidates) |
| Update policy | Binary `ISSAMEEVENT` | Blind merge | Four-way relation: `NEW_EVENT`, `CORROBORATE`, `UPDATE_PROGRESS`, `REVISE_CONFLICT` |
| Verification | None | Unconditional multi-round 5W1H | Triggered, only for revision candidates |
| Retraction | No | No | Explicit `RETRACTED` state, accepted only with a verbatim source quote |

## 2. Pipeline

1. **Extraction (7B).** `pipeline/extractor.py` returns summary, event date, entities, key facts, `slots` (who/what/where/why/how), and an LLM revision flag.
2. **Revision candidate** (`runner._is_revision_candidate`): LLM flag OR lexicon (`pipeline/cues.py`, single source of truth) OR structural probe (`runner._implicit_probe`, enabled by `retrieval.implicit_probe`). The probe needs a cross-date state with bi-encoder score >= `implicit_probe_sim` and either a measure-noun/number conflict or an event date 1-7 days away. It only creates a candidate; the classifier decides.
3. **Phase 1.** Claims are extracted; non-candidates are matched in publication order against up to `routine_candidate_k` same-day shortlist states in one call (`classifier.classify_candidates`) and the decision is cached. Candidates are queued and stay in the evidence prefix.
4. **Phase 2 (chronological replay).** The repository is reset; the stream is replayed in publication order. Cached routine decisions are re-applied in place. Each candidate is re-retrieved over the 90-day horizon from the state as of its publication time, reranked (bi-encoder + cross-encoder + lexical overlap + corroboration bonus), and tried against the top 3 candidates with the 14B classifier. Across dates only `REVISE_CONFLICT` is accepted.
5. **Verification** (`pipeline/verifier.py`). Up to 3 queries, up to 10 evidence articles from the seen prefix, 14B returns resolved summary/date/status. `RETRACTED` is downgraded to `REVISED` unless `retraction_evidence` is a verbatim (whitespace/case-normalized) span of a supplied evidence article. Lexical negation never retracts (`state_manager`).
6. **State update** (`pipeline/state_manager.py`). The previous summary/date go to `revision_history`; the active summary is replaced by the resolved summary after a clause-level regex purge of trailing comparatives.
7. **Timeline** (`pipeline/ranker.py`). Retracted states are dropped; Eq. (6) salience `w_c log(1+c) + w_r [REVISED] + (w_s/6) * filled_slots`, one state per date, top-l, chronological.

**Known limitation.** Routine decisions are cached from the Phase-1 graph and are not re-evaluated after a Phase-2 revision changes a state. Replay is chronological and leak-free (tested in `tests/test_pipeline_logic.py`) but is not full event-sourced recomputation. Revisions are visible only at stream end.

## 3. Ablation switches

| Variant | Switch |
| :--- | :--- |
| w/o 14B verifier | `verifier_llm=<7B client>` |
| w/o cross-encoder | `retrieval.cross_encoder_model = null` |
| w/o conditional gate | `retrieval.strict_temporal_gate = false` |
| w/o structural probe | `retrieval.implicit_probe = false` |
| w/o purge (append) | `disable_purge=True` |
| overwrite only | `overwrite_only=True` |
| rho sources | `rho_mode in {both, llm, lexicon, none}` (unknown values raise) |
| corroboration-only ranking | `selection_mode="corroboration"` |
| guarded / disabled retraction | `guarded_retract=True` / `disable_retract=True` |

## 4. Evaluation design

- **Standard TLS** (`scripts/run_fullcorpus.py`): T17 and Crisis, date-span filtered, target length = mean reference length, **mean over reference timelines**, topic mean +/- std.
- **Held-out revision suite** (`revisions/heldout.py`, `scripts/run_heldout_suite.py`): generated, fictional entities, four positive case types x explicit/implicit cue, plus negative controls (denial of a side allegation / remark retraction; non-contradicting clarification). Metrics in `evaluation/revision_metrics.py`. Run once after freezing the system.
- **Natural-stream audit** (`scripts/export_transition_audit.py --export-random-audit`, `scripts/summarize_random_audit.py`): stratified random sample of emitted transitions, two independent annotators, design-weighted precision. Protocol: `transition_audit_protocol.md`. Precision only; no recall.
- `revisions/benchmark.py` holds the hand-written templates the system was developed against. Do not report them as held-out.
