"""OpenAI-compatible LLM client supporting local (Ollama/vLLM) and remote endpoints."""

import json
import logging
import re
from typing import Any
from openai import OpenAI

from revitls.config import LLMConfig

logger = logging.getLogger(__name__)


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

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a text completion for the given prompts."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

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
