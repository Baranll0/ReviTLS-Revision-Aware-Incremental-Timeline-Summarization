# Stratified Random Natural-Stream Transition Audit

- Seed: `20261123`; requested sample per dataset × topic × transition stratum: `5`.
- Exported records: **65** across **13** non-empty strata.
- Annotate `annotator_1_tp_fp` and `annotator_2_tp_fp` as `TP` or `FP` independently after checking the source article and prior context; the system rationale is intentionally withheld.
- Use `design_weight` for population-level precision estimates because strata have unequal sampling fractions.
- This sample estimates precision of emitted transitions only; recall requires reviewing source articles that the system did not label as transitions.
- Source article text is not copied into this file; use `source_article_id` to locate it in the licensed corpus.
- Annotation rubric: `docs/transition_audit_protocol.md`.

CSV: `outputs\final\transition_audit_random.csv`
