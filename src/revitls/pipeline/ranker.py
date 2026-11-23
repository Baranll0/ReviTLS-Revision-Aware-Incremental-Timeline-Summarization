"""Revision-aware milestone ranking and timeline emission."""

import math

from revitls.config import TimelineConfig
from revitls.schemas import EventState, EventStatus, TimelineEntry


def filled_slot_count(state: EventState) -> int:
    """Number of populated 5W1H slots (who, what, when, where, why, how), in [0, 6]."""
    filled = 0
    filled += int(bool(state.entities) or bool(state.slots.get("who")))
    filled += int(bool(state.canonical_summary.strip()) or bool(state.slots.get("what")))
    filled += int(state.event_date is not None)
    filled += sum(int(bool(state.slots.get(k))) for k in ("where", "why", "how"))
    return filled


def salience(state: EventState, config: TimelineConfig) -> float:
    """Eq. (6): w_c*log(1+c) + w_r*1[REVISED] + (w_s/6)*|filled 5W1H slots|."""
    return (
        config.w_corroboration * math.log1p(state.corroboration_count)
        + config.w_revised * float(state.status == EventStatus.REVISED)
        + config.w_slots / 6.0 * filled_slot_count(state)
    )


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
        ranked = sorted(
            candidates,
            key=lambda s: (salience(s, self.config), len(s.supporting_article_ids)),
            reverse=True,
        )
        if self.config.deduplicate_dates:
            seen_dates: set[object] = set()
            unique: list[EventState] = []
            for st in ranked:
                if st.event_date not in seen_dates:
                    seen_dates.add(st.event_date)
                    unique.append(st)
            top_states = unique[:target_l]
        else:
            top_states = ranked[:target_l]
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
