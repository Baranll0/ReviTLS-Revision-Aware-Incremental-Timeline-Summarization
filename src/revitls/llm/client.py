"""OpenAI-compatible LLM client supporting local (Ollama/vLLM) and remote endpoints."""

import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any
from openai import OpenAI

from revitls.config import LLMConfig

logger = logging.getLogger(__name__)

_DETERMINISTIC_CACHE: dict[str, tuple[str, float]] = {}
_CACHE_FILE = Path("outputs/.llm_deterministic_cache.jsonl")
_CACHE_LOADED = False


def _ensure_cache_loaded() -> None:
    global _CACHE_LOADED
    if _CACHE_LOADED:
        return
    _CACHE_LOADED = True
    if _CACHE_FILE.exists():
        try:
            with _CACHE_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        _DETERMINISTIC_CACHE[rec["k"]] = (rec["v"], float(rec.get("dt", 1.2)))
        except Exception:
            pass


def _extract_json_object(raw: str) -> dict[str, Any]:
    """Strip markdown fences if present and parse the first JSON object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE | re.MULTILINE)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Expected JSON object in LLM output, got: {raw[:200]}")
    return json.loads(match.group(0))


class LLMClient:
    """Thin wrapper around an OpenAI-compatible chat completion endpoint."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = OpenAI(base_url=config.base_url, api_key=config.api_key)
        self.virtual_elapsed_bonus: float = 0.0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion for the given prompts."""
        _ensure_cache_loaded()
        cache_key = hashlib.sha256(
            f"{self.config.model}\0{system_prompt}\0{user_prompt}".encode("utf-8")
        ).hexdigest()
        if self.config.temperature == 0.0 and cache_key in _DETERMINISTIC_CACHE:
            cached_text, orig_dt = _DETERMINISTIC_CACHE[cache_key]
            self.virtual_elapsed_bonus += orig_dt
            return cached_text

        t0 = time.perf_counter()
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        dt = time.perf_counter() - t0
        out = (response.choices[0].message.content or "").strip()
        if self.config.temperature == 0.0:
            _DETERMINISTIC_CACHE[cache_key] = (out, dt)
            try:
                _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
                with _CACHE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": cache_key, "v": out, "dt": round(dt, 4)}) + "\n")
            except Exception:
                pass
        return out

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object response with automatic retry on malformed output."""
        prompt = user_prompt
        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            raw = self.complete(system_prompt, prompt)
            try:
                return _extract_json_object(raw)
            except (ValueError, json.JSONDecodeError) as err:
                last_err = err
                if attempt < max_retries:
                    prompt = (
                        f"{user_prompt}\n\n"
                        "IMPORTANT: Your previous response was not valid JSON. "
                        "Reply with ONLY a single valid JSON object and no extra text."
                    )
        raise ValueError(f"Failed to parse JSON after {max_retries + 1} attempts: {last_err}")

    def complete_json_safe(
        self,
        system_prompt: str,
        user_prompt: str,
        default: dict[str, Any],
        max_retries: int = 2,
    ) -> dict[str, Any]:
        """Parse JSON with retries, returning a safe fallback dict instead of crashing long streams."""
        try:
            return self.complete_json(system_prompt, user_prompt, max_retries=max_retries)
        except Exception as err:
            logger.warning("LLM JSON fallback triggered: %s", err)
            return dict(default)
