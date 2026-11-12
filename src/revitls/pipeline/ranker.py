"""Revision-aware milestone ranking and timeline emission."""

from revitls.config import TimelineConfig
from revitls.schemas import EventState, EventStatus, TimelineEntry


class TimelineRanker:
    """Selects top-l milestone events while filtering retracted rumors."""

    def __init__(self, config: TimelineConfig) -> None:
        self.config = config

    def build_timeline(
        self, states: list[EventState], num_dates: int | None = None
    ) -> list[TimelineEntry]:
        target_l = num_dates or self.config.default_num_dates
        candidates = [
            s
            for s in states
            if s.event_date is not None
            and not (self.config.exclude_retracted and s.status == EventStatus.RETRACTED)
        ]

        # Prioritize corroborated and verified/revised events over uncorroborated rumors
        def prominence_score(state: EventState) -> tuple[int, int]:
            verified_bonus = 1 if state.status == EventStatus.REVISED else 0
            return (state.corroboration_count + verified_bonus, len(state.supporting_article_ids))

        top_states = sorted(candidates, key=prominence_score, reverse=True)[:target_l]
        top_states.sort(key=lambda s: s.event_date)  # chronological order

        return [
            TimelineEntry(
                event_date=s.event_date,  # type: ignore[arg-type]
                summary=s.canonical_summary,
                status=s.status,
                supporting_article_ids=list(s.supporting_article_ids),
            )
            for s in top_states
        ]
