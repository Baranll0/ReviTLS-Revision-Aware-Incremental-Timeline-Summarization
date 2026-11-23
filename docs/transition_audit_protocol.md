# Natural-Stream Transition Audit Protocol

## Purpose and estimand

This audit estimates the precision of system-emitted `REVISED` and `RETRACTED` transitions in the evaluated corpus. It does not measure recall: articles that the system failed to identify as transitions are absent from the transition population and require a separate source-article sampling design.

The random audit is stratified by dataset × topic × transition type. Annotators must record both labels independently. Use `design_weight` when estimating population precision across the sampled strata; unweighted pooled percentages describe the audit sample, not the transition population.

## Unit and evidence reviewed

The unit is one emitted transition record, identified by `source_article_id`. Locate the article in the corresponding licensed source corpus. Read the full source article and the earlier context needed to understand the state before this publication. Judge whether the transition is supported at the time it was published; do not use later articles to retroactively justify it. The CSV intentionally omits the model rationale to reduce anchoring.

## Label definitions

- **`REVISED` = TP:** The source article supports a material correction, replacement, supersession, or changed fact/date/cause for the same underlying event, and the recorded transition links the correct prior and updated states.
- **`REVISED` = FP:** The article is unrelated, merely additive/corroborative, does not support the asserted change, or the prior/new state linkage is materially wrong.
- **`RETRACTED` = TP:** Evidence available at that point supports that the underlying event itself did not occur. A report, theory, cause, estimate, or public statement being withdrawn is not by itself enough if the underlying event occurred.
- **`RETRACTED` = FP:** The event occurred but a detail or explanation was corrected; the source only reports a denial, apology, retraction of a statement, or unsupported claim of non-occurrence; or the cited source does not support retracting this event.

If the article is unavailable, ambiguous, or lacks enough prior context, annotate `UNCERTAIN` in `notes` and leave the TP/FP field blank until adjudication. Do not force uncertain cases into FP.

## Annotation and reporting

1. Freeze the evaluated code, corpus manifest, and generated `summary.json` before sampling.
2. Generate the reproducible sample with `python scripts/export_transition_audit.py --export-random-audit` and retain its seed, stratum counts, inclusion probabilities, and weights.
3. Have two annotators label the sample independently before discussion. Keep the original CSV unchanged; resolve disagreements in a separate adjudication record.
4. Report sample size, population size, weighted precision by transition type and overall, uncertainty intervals that respect the stratified design, raw agreement, and Cohen's kappa with its limitations. Also report the disagreement/adjudication process.
5. Do not describe this audit as measuring recall or as a corpus-wide guarantee. If the sample is too small for useful intervals, report that limitation and increase the predeclared sample size rather than implying precision.
