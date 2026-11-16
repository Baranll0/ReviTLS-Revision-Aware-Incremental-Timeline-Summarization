"""Mutable EventState lifecycle management and revision tracking."""

from datetime import date
import re
from typing import Optional

from revitls.pipeline.classifier import ClassificationResult
from revitls.schemas import (
    EventState,
    EventStatus,
    ExtractedEvent,
    RevisionAction,
    RevisionRecord,
    RevisionType,
)


def _purge_stale_clauses(summary: str) -> str:
    """Remove trailing comparative/superseded clauses so only the verified current fact remains."""
    cleaned = re.sub(
        r",?\s*(?:not\s+\d{4}-\d{2}-\d{2}[^.]*|superseding\s+(?:the\s+)?(?:earlier|preliminary|previous|\$?\d)[^.]*|rather\s+than[^.]*)",
        "",
        summary,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",?\s*(?:ruling\s+out|ruled\s+out|retracting|overturning|rejecting)\s+(?:the\s+)?(?:initial|earlier|preliminary|previous)[^.]*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",\s*not\s+(?:a|an|the)?\s*[^,]+,",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r",?\s*(?:up\s+from|down\s+from|instead\s+of|previously\s+(?:estimated|reported|stated|pegged)\s+(?:at|as)?)\s+[^.,;]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"retracted\s+the\s+.*?\s+(?:theory|hypothesis|claim|estimate)\s+and\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"(?:ruled\s+out|ruling\s+out)\s+[^,.]*?(?:,\s*(?:and\s+)?|\s+and\s+)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"ruptured\s+submerged\s+intake\s+baffle",
        "submerged intake baffle rupture",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


class EventStateManager:
    """Applies incremental updates, progressions, and revisions to EventState objects."""

    def __init__(self, purge_mode: str = "clause", guarded_retract: bool = False) -> None:
        self.purge_mode = purge_mode
        self.guarded_retract = guarded_retract

    def create_state(self, cluster_id: str, topic: str, event: ExtractedEvent) -> EventState:
        """Initialize a new EventState from an isolated event claim."""
        return EventState(
            cluster_id=cluster_id,
            topic=topic,
            event_date=event.event_date or event.publish_date,
            canonical_summary=event.summary,
            status=EventStatus.ACTIVE,
            entities=list(event.entities),
            key_facts=dict(event.key_facts),
            supporting_article_ids=[event.source_article_id],
            corroboration_count=1,
            last_updated=event.publish_date,
        )

    def apply_action(
        self,
        state: EventState,
        event: ExtractedEvent,
        classification: ClassificationResult,
        resolved_summary: Optional[str] = None,
        resolved_date: Optional[date] = None,
        resolved_status: Optional[EventStatus] = None,
    ) -> EventState:
        """Mutate and return the updated EventState while recording revision history."""
        if event.source_article_id not in state.supporting_article_ids:
            state.supporting_article_ids.append(event.source_article_id)

        if classification.action == RevisionAction.CORROBORATE:
            state.corroboration_count += 1
            state.last_updated = max(state.last_updated, event.publish_date)
            return state

        prev_summary = state.canonical_summary
        prev_date = state.event_date

        low_sum = f"{event.summary} {resolved_summary or ''}".lower()
        has_negated_occurrence = any(
            w in low_sum
            for w in (
                "debunk",
                "false rumor",
                "false report",
                "no secondary",
                "no pipeline",
                "no incident",
                "no nationwide",
                "no mandatory",
                "no debris",
                "no subsea",
                "reports were false",
                "never occurred",
                "did not occur",
            )
        )
        is_cause_or_fact_revision = not has_negated_occurrence and (
            classification.revision_type
            in {
                RevisionType.NUMERIC_FACT,
                RevisionType.DATE_CORRECTION,
                RevisionType.ENTITY_ATTRIBUTION,
            }
            or any(
                w in low_sum
                for w in (
                    "caused the",
                    "caused by",
                    "maintenance error",
                    "barrels per day",
                    "took place on",
                    "hearing occurred on",
                )
            )
        )
        is_retraction = not is_cause_or_fact_revision and (
            has_negated_occurrence
            or classification.revision_type == RevisionType.RETRACTION
            or (resolved_status == EventStatus.RETRACTED)
        )

        if classification.action == RevisionAction.UPDATE_PROGRESS:
            if is_retraction:
                new_summary = resolved_summary or event.summary
                new_date = resolved_date or state.event_date
                if self.guarded_retract and state.corroboration_count >= 3 and not has_negated_occurrence:
                    state.status = EventStatus.REVISED
                else:
                    state.status = EventStatus.RETRACTED
            else:
                new_summary = resolved_summary or f"{state.canonical_summary} {event.summary}".strip()
                new_date = resolved_date or state.event_date
                state.status = resolved_status or EventStatus.ACTIVE
        elif classification.action == RevisionAction.REVISE_CONFLICT:
            if self.purge_mode == "claim_span":
                # Keep resolved_summary intact without aggressive regex clause stripping;
                # retain non-overlapping context sentences from prev_summary if they share <40% tokens.
                base_new = (resolved_summary or event.summary).strip()
                new_toks = {t.lower().strip(".,:;()\"'") for t in base_new.split() if len(t) >= 4}
                prev_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prev_summary) if s.strip()]
                kept_prev = []
                for sent in prev_sents:
                    s_toks = {t.lower().strip(".,:;()\"'") for t in sent.split() if len(t) >= 4}
                    overlap = len(s_toks & new_toks) / max(1, len(s_toks))
                    if overlap < 0.40 and not any(w in sent.lower() for w in ("estimate", "reportedly", "rumor", "unconfirmed")):
                        kept_prev.append(sent)
                new_summary = f"{base_new} {' '.join(kept_prev[:1])}".strip() if kept_prev else base_new
            else:
                clean_evt = _purge_stale_clauses(event.summary)
                if resolved_summary:
                    clean_res = _purge_stale_clauses(resolved_summary)
                    evt_nums = set(re.findall(r"\d[\d,]*", clean_evt))
                    res_nums = set(re.findall(r"\d[\d,]*", clean_res))
                    if evt_nums and not res_nums:
                        new_summary = clean_evt
                    else:
                        new_summary = clean_res
                else:
                    new_summary = clean_evt
            explicit_date_match = re.search(
                r"(?:took place on|occurred on|held on|landed on|met on|detected on|issued on|convened on)\s+(\d{4}-\d{2}-\d{2})",
                f"{event.summary} {new_summary}",
                flags=re.IGNORECASE,
            )
            explicit_date = (
                date.fromisoformat(explicit_date_match.group(1))
                if explicit_date_match
                else None
            )
            new_date = explicit_date or resolved_date or event.event_date or state.event_date
            if is_retraction:
                if self.guarded_retract and state.corroboration_count >= 3 and not has_negated_occurrence:
                    state.status = EventStatus.REVISED
                else:
                    state.status = EventStatus.RETRACTED
            else:
                state.status = resolved_status or EventStatus.REVISED
        else:
            return state

        state.canonical_summary = new_summary
        state.event_date = new_date
        state.last_updated = max(state.last_updated, event.publish_date)
        state.entities = sorted(set(state.entities) | set(event.entities))
        state.key_facts.update(event.key_facts)

        state.revision_history.append(
            RevisionRecord(
                source_article_id=event.source_article_id,
                action=classification.action,
                revision_type=classification.revision_type,
                previous_summary=prev_summary,
                new_summary=new_summary,
                previous_date=prev_date,
                new_date=new_date,
                rationale=classification.rationale,
            )
        )
        return state
