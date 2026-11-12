# ReviTLS Architecture & Research Positioning

## 1. Gap Analysis of Baseline Papers

| Aspect | LLM-TLS (ACL 2024) | CHRONOS (NAACL 2025) | **ReviTLS (Ours)** |
| :--- | :--- | :--- | :--- |
| **Execution Mode** | Incremental stream | Static / batch per query | **Incremental stream** |
| **Memory Representation** | Static event nodes + graph edges | Stateless per-round timelines | **Mutable `EventState` graph + revision log** |
| **Date Linking** | Strict same-date heuristic (Topic TLS) | Round merging (`keep original date`) | **Cross-date semantic + entity + temporal-mention linking** |
| **Update / Conflict Policy** | Binary `ISSAMEEVENT` + TextRank | Blind merge (`exclude later duplicates`) | **4-way relation (`NEW`, `CORROBORATE`, `UPDATE`, `REVISE_CONFLICT`)** |
| **Evidence Verification** | None (single-pass) | Unconditional multi-round 5W1H | **Triggered targeted self-questioning on conflicts/gaps** |
| **Retracted Rumors** | Retained if frequent | Retained if frequent across rounds | **Explicitly retracted / filtered or annotated** |

## 2. Incremental Pipeline Flow

For each arriving article $a_t$ (sorted by publication timestamp):
1. **Claim Extraction (`extractor.py`)**: Extract structured event claims $(d_{\text{event}}, \text{entities}, \text{summary}, \text{key\_facts})$.
2. **Cross-Date Retrieval (`memory.py`)**: Query active `EventState` clusters using dense embeddings + entity overlap without enforcing $d_{\text{event}} = d_{\text{cluster}}$.
3. **Revision Classification (`classifier.py`)**: Classify $(e_t, C_k)$ as `NEW_EVENT`, `CORROBORATE`, `UPDATE_PROGRESS`, or `REVISE_CONFLICT`.
4. **Triggered Verification (`verifier.py`)**: If `REVISE_CONFLICT` (or missing critical 5W1H slots), generate targeted verification queries (adapted from CHRONOS Questioner + Rewriter) and retrieve authoritative/newer corroborating articles from the seen stream (or external index).
5. **State Update (`state_manager.py`)**: Update `canonical_summary`, `event_date` (if corrected), `status` (`ACTIVE`, `REVISED`, `RETRACTED`), and append to `revision_history`.
6. **Milestone Ranking (`ranker.py`)**: Exclude `RETRACTED` clusters, rank remaining clusters by corroborated prominence, and emit the chronological timeline of length $l$.

## 3. Evaluation Plan
1. **Standard Topic TLS (`Crisis`, `T17`):** Compare directly against published `LLM-TLS` and `CHRONOS` tables using `tilse` (`Align-R1`, `Align-R2`, `Date-F1`) + wall-clock runtime.
2. **Revision-Stress Benchmark:** Measure:
   - **Revision Resolution Accuracy (RRA):** % of injected/real revisions whose final `EventState` reflects the corrected fact/date.
   - **Stale-Fact Leakage Rate (SFLR):** % of final timeline summaries still containing superseded/retracted claims (lower is better).
   - **Retraction Precision/Recall:** Ability to suppress debunked events.
