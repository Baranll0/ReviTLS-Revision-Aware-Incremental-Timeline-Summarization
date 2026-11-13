"""Mutable EventState lifecycle management and revision tracking."""

from datetime import date
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


class EventStateManager:
    """Applies incremental updates, progressions, and revisions to EventState objects."""

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

        is_retraction = (
            classification.revision_type == RevisionType.RETRACTION
            or (resolved_status == EventStatus.RETRACTED)
            or any(
                w in event.summary.lower()
                for w in ("debunk", "false rumor", "false report", "no secondary", "reports were false")
            )
        )

        if classification.action == RevisionAction.UPDATE_PROGRESS:
            if is_retraction:
                new_summary = resolved_summary or event.summary
                new_date = resolved_date or state.event_date
                state.status = EventStatus.RETRACTED
            else:
                new_summary = resolved_summary or f"{state.canonical_summary} {event.summary}".strip()
                new_date = resolved_date or state.event_date
                state.status = resolved_status or EventStatus.ACTIVE
        elif classification.action == RevisionAction.REVISE_CONFLICT:
            new_summary = resolved_summary or event.summary
            new_date = resolved_date or event.event_date or state.event_date
            state.status = EventStatus.RETRACTED if is_retraction else (resolved_status or EventStatus.REVISED)
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
