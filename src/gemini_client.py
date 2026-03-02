"""Wrapper around Google GenAI SDK for text generation and embeddings."""

from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from src.config import settings
from src.cost_tracker import tracker

logger = logging.getLogger(__name__)

# Pricing per 1M tokens (USD)
_PRICING: dict[str, dict[str, float]] = {
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-embedding-001": {"input": 0.006, "output": 0.0},
}

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.google_api_key)
    return _client


def generate(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    json_mode: bool = False,
) -> str:
    """Generate text with cost tracking."""
    model = model or settings.model_fast
    tracker.check_budget()

    client = _get_client()
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    if system:
        config.system_instruction = system
    if json_mode:
        config.response_mime_type = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    input_tokens = response.usage_metadata.prompt_token_count or 0
    output_tokens = response.usage_metadata.candidates_token_count or 0
    tracker.record(model, input_tokens, output_tokens)

    return response.text or ""


def generate_json(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Generate and parse JSON output."""
    raw = generate(
        prompt,
        model=model,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Failed to parse JSON, attempting extraction: %s", raw[:200])
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
        raise


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Get embeddings with cost tracking."""
    model = model or settings.model_embedding
    tracker.check_budget()

    client = _get_client()
    response = client.models.embed_content(
        model=model,
        contents=texts,
    )

    total_tokens = sum(len(t.split()) * 1.3 for t in texts)  # rough estimate
    tracker.record(model, int(total_tokens), 0)

    return [e.values for e in response.embeddings]
