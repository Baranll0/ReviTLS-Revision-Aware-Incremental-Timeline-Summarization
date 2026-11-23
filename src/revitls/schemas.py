"""Core data schemas for Revision-Aware Incremental Timeline Summarization."""

from datetime import date, datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RevisionAction(str, Enum):
    """Operational relation between an incoming event claim and an existing EventState."""

    NEW_EVENT = "NEW_EVENT"
    CORROBORATE = "CORROBORATE"
    UPDATE_PROGRESS = "UPDATE_PROGRESS"
    REVISE_CONFLICT = "REVISE_CONFLICT"


class EventStatus(str, Enum):
    """Lifecycle status of an EventState node in the evolving timeline graph."""

    ACTIVE = "ACTIVE"
    REVISED = "REVISED"
    RETRACTED = "RETRACTED"


class RevisionType(str, Enum):
    """Fine-grained taxonomy for factual revisions."""

    NUMERIC_FACT = "NUMERIC_FACT"
    DATE_CORRECTION = "DATE_CORRECTION"
    ENTITY_ATTRIBUTION = "ENTITY_ATTRIBUTION"
    RETRACTION = "RETRACTION"
    STATE_PROGRESSION = "STATE_PROGRESSION"


class Article(BaseModel):
    """Single document arriving in a chronological news stream."""

    article_id: str
    topic: str
    title: str
    content: str
    publish_date: date
    source: Optional[str] = None
    url: Optional[str] = None


class ExtractedEvent(BaseModel):
    """Structured event claim distilled from an incoming article."""

    source_article_id: str
    publish_date: date
    event_date: Optional[date] = None
    summary: str
    entities: list[str] = Field(default_factory=list)
    key_facts: dict[str, str] = Field(default_factory=dict)
    slots: dict[str, str] = Field(default_factory=dict)
    is_correction_or_update: bool = False


class RevisionRecord(BaseModel):
    """Audit log entry capturing why and how an EventState changed."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_article_id: str
    action: RevisionAction
    revision_type: Optional[RevisionType] = None
    previous_summary: str
    new_summary: str
    previous_date: Optional[date] = None
    new_date: Optional[date] = None
    new_status: EventStatus = EventStatus.ACTIVE
    rationale: str = ""


class EventState(BaseModel):
    """Mutable event cluster state replacing static LLM-TLS graph nodes."""

    cluster_id: str
    topic: str
    event_date: Optional[date]
    canonical_summary: str
    status: EventStatus = EventStatus.ACTIVE
    entities: list[str] = Field(default_factory=list)
    key_facts: dict[str, str] = Field(default_factory=dict)
    slots: dict[str, str] = Field(default_factory=dict)
    supporting_article_ids: list[str] = Field(default_factory=list)
    corroboration_count: int = 1
    last_updated: date
    revision_history: list[RevisionRecord] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    """Final timeline item emitted for evaluation or display."""

    event_date: date
    summary: str
    status: EventStatus = EventStatus.ACTIVE
    supporting_article_ids: list[str] = Field(default_factory=list)
