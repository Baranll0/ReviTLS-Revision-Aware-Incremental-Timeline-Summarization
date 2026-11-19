"""Typed configuration loader for ReviTLS."""

from pathlib import Path
import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: float = 0.0
    max_tokens: int = 1024


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    device: str = "cuda"
    normalize: bool = True


class RetrievalConfig(BaseModel):
    top_k_clusters: int = 10
    allow_cross_date: bool = True
    max_date_window_days: int = 90
    routine_window_days: int = 0
    similarity_threshold: float = 0.45
    strict_temporal_gate: bool = False
    cross_encoder_model: str | None = None


class VerifierConfig(BaseModel):
    enable_targeted_verification: bool = True
    max_verification_questions: int = 3
    max_evidence_docs: int = 10


class TimelineConfig(BaseModel):
    default_num_dates: int = 20
    exclude_retracted: bool = True
    deduplicate_dates: bool = False


class ReviTLSConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    verifier_llm: LLMConfig | None = None
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    verifier: VerifierConfig = Field(default_factory=VerifierConfig)
    timeline: TimelineConfig = Field(default_factory=TimelineConfig)


def load_config(path: str | Path) -> ReviTLSConfig:
    """Load configuration from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ReviTLSConfig.model_validate(raw)
