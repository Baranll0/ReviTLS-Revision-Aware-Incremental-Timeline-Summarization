"""Cross-date semantic and entity-aware EventState memory."""

from datetime import date
from typing import Optional
import numpy as np

from revitls.config import RetrievalConfig
from revitls.schemas import EventState


class EventMemory:
    """In-memory vector + metadata index for evolving EventState clusters."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self._states: dict[str, EventState] = {}
        self._vectors: dict[str, np.ndarray] = {}

    def add_or_update(self, state: EventState, vector: np.ndarray) -> None:
        """Register a new EventState or refresh its embedding after revision."""
        norm = np.linalg.norm(vector)
        unit_vec = vector / norm if norm > 0 else vector
        self._states[state.cluster_id] = state
        self._vectors[state.cluster_id] = unit_vec

    def get(self, cluster_id: str) -> Optional[EventState]:
        return self._states.get(cluster_id)

    def all_states(self) -> list[EventState]:
        return list(self._states.values())

    def search(
        self,
        query_vector: np.ndarray,
        reference_date: Optional[date] = None,
        entities: Optional[list[str]] = None,
    ) -> list[tuple[EventState, float]]:
        """Retrieve top-k candidate EventStates, optionally allowing cross-date matching."""
        if not self._states:
            return []

        q_norm = np.linalg.norm(query_vector)
        q_unit = query_vector / q_norm if q_norm > 0 else query_vector
        entity_set = {e.lower() for e in (entities or [])}

        scored: list[tuple[EventState, float]] = []
        for cid, state in self._states.items():
            if not self.config.allow_cross_date and reference_date and state.event_date:
                if state.event_date != reference_date:
                    continue
            elif reference_date and state.event_date:
                day_diff = abs((reference_date - state.event_date).days)
                if day_diff > self.config.max_date_window_days:
                    continue

            cos_sim = float(np.dot(q_unit, self._vectors[cid]))
            state_entities = {e.lower() for e in state.entities}
            entity_bonus = 0.05 * len(entity_set & state_entities)
            score = cos_sim + entity_bonus

            if score >= self.config.similarity_threshold:
                scored.append((state, score))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.config.top_k_clusters]
