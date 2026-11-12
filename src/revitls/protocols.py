"""Structural protocols for LLM and embedding backends (enables seamless GPU & offline testing)."""

from typing import Any, Protocol
import numpy as np


class LLMProtocol(Protocol):
    """Minimal chat-completion interface shared by LLMClient and deterministic test doubles."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class EmbedderProtocol(Protocol):
    """Minimal embedding interface matching SentenceTransformer.encode."""

    def encode(self, sentences: str | list[str], normalize_embeddings: bool = True) -> np.ndarray:
        ...
