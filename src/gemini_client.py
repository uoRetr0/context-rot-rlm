"""Wrapper around Google GenAI SDK for text generation and embeddings."""

from __future__ import annotations

import json
import logging
import re
import time
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
_last_call_time: float = 0.0
_MIN_CALL_INTERVAL: float = 1.0  # slower pacing reduces short-window rate-limit spikes


def _extract_string_field(raw: str, field: str) -> str | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    match = re.search(pattern, raw, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            repaired = (
                candidate
                .replace("\\'", "'")
                .replace('\\"', '"')
                .replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\r", "\r")
                .replace("\\\\", "\\")
            )
            return repaired
    return None


def _extract_number_field(raw: str, field: str) -> float | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw)
    if match:
        return float(match.group(1))
    return None


def _extract_bool_field(raw: str, field: str) -> bool | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*(true|false)'
    match = re.search(pattern, raw, flags=re.IGNORECASE)
    if match:
        return match.group(1).lower() == "true"
    return None


def _extract_array_field(raw: str, field: str) -> list[Any] | None:
    pattern = rf'"{re.escape(field)}"\s*:\s*(\[[^\]]*\])'
    match = re.search(pattern, raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def _repair_json_object(raw: str) -> dict[str, Any]:
    """Best-effort recovery for partially malformed JSON responses."""
    repaired: dict[str, Any] = {}

    for field in ("answer", "reasoning", "thought", "code"):
        value = _extract_string_field(raw, field)
        if value is not None:
            repaired[field] = value

    confidence = _extract_number_field(raw, "confidence")
    if confidence is not None:
        repaired["confidence"] = confidence

    requires_multi_hop = _extract_bool_field(raw, "requires_multi_hop")
    if requires_multi_hop is not None:
        repaired["requires_multi_hop"] = requires_multi_hop

    for field in ("evidence_used", "sub_questions", "search_queries"):
        value = _extract_array_field(raw, field)
        if value is not None:
            repaired[field] = value

    if repaired:
        return repaired

    raise json.JSONDecodeError("Unable to repair JSON response", raw, 0)


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

    global _last_call_time
    max_retries = 6
    for attempt in range(max_retries):
        # Rate limit: wait if calling too fast
        elapsed = time.time() - _last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            _last_call_time = time.time()
            break
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                wait = min(2 ** attempt + 1, 60)
                logger.warning("API error, retrying in %ds (attempt %d/%d): %s", wait, attempt + 1, max_retries, err_str[:80])
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

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
            candidate = raw[start:end]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                raw = candidate
        return _repair_json_object(raw)


def embed(texts: list[str], *, model: str | None = None) -> list[list[float]]:
    """Get embeddings with cost tracking."""
    model = model or settings.model_embedding
    tracker.check_budget()

    client = _get_client()
    max_retries = 6
    for attempt in range(max_retries):
        try:
            response = client.models.embed_content(
                model=model,
                contents=texts,
            )
            break
        except Exception as e:
            err_str = str(e)
            if any(k in err_str for k in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE")):
                wait = min(2 ** attempt + 1, 60)
                logger.warning("Embed API error, retrying in %ds (attempt %d/%d)", wait, attempt + 1, max_retries)
                time.sleep(wait)
                if attempt == max_retries - 1:
                    raise
            else:
                raise

    total_tokens = sum(len(t.split()) * 1.3 for t in texts)  # rough estimate
    tracker.record(model, int(total_tokens), 0)

    return [e.values for e in response.embeddings]
