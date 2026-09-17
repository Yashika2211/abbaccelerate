"""Claude access with a deterministic fallback — PROJECT_BRIEF.md §9.

Two rules govern everything here:

* **The LLM writes prose; it never produces numbers.** Narration prompts receive a
  JSON payload of already-computed values and are told to use only those.
* **An LLM outage must never break the product.** With no API key, a failed call,
  or a response that does not validate, every function here falls back to
  deterministic rules or templates and records that it did so.

The fallback is not a degraded mode we hope nobody notices — the run timeline says
plainly which path was taken.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from kairos.config import get_settings

log = logging.getLogger(__name__)

PROMPTS = Path(__file__).parent / "prompts"
MAX_ATTEMPTS = 2

T = TypeVar("T", bound=BaseModel)


@dataclass
class LLMOutcome:
    """What came back, and honestly how."""

    value: Any
    used_llm: bool
    note: str

    @property
    def ok(self) -> bool:
        return self.value is not None


def load_prompt(name: str) -> str:
    path = PROMPTS / f"{name}.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def available() -> bool:
    return get_settings().llm_enabled


def _client():
    from anthropic import Anthropic

    return Anthropic(api_key=get_settings().anthropic_api_key)


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in fences despite being asked not to."""
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def structured(
    system_prompt: str,
    payload: dict[str, Any],
    schema: type[T],
    max_tokens: int = 2000,
) -> LLMOutcome:
    """Ask for JSON matching ``schema``. Retries once on validation failure.

    Returns an outcome rather than raising: the caller always has a rules-based
    path, and a run that silently degrades is better than a run that dies.
    """
    if not available():
        return LLMOutcome(None, False, "No ANTHROPIC_API_KEY set; used deterministic rules.")

    settings = get_settings()
    instruction = (
        f"{system_prompt}\n\n## Required JSON schema\n\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}"
    )
    last_error = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            user_content = json.dumps(payload, indent=2, default=str)
            if last_error:
                user_content += (
                    f"\n\nYour previous response failed validation with: {last_error}\n"
                    f"Return corrected JSON only."
                )
            response = _client().messages.create(
                model=settings.kairos_llm_model,
                max_tokens=max_tokens,
                system=instruction,
                messages=[{"role": "user", "content": user_content}],
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            return LLMOutcome(
                schema.model_validate_json(_extract_json(text)),
                True,
                f"Claude ({settings.kairos_llm_model}) responded on attempt {attempt}.",
            )
        except ValidationError as exc:
            last_error = str(exc)[:500]
            log.warning("LLM response failed schema validation (attempt %s)", attempt)
        except Exception as exc:  # noqa: BLE001 - never break a run over the LLM
            log.warning("LLM call failed: %s", exc)
            return LLMOutcome(None, False, f"LLM call failed ({type(exc).__name__}); used rules.")

    return LLMOutcome(None, False, "LLM response never matched the schema; used rules.")


def prose(
    system_prompt: str,
    payload: dict[str, Any],
    fallback: str,
    max_tokens: int = 400,
) -> LLMOutcome:
    """Ask for a short piece of grounded prose, falling back to a template."""
    if not available():
        return LLMOutcome(fallback, False, "No ANTHROPIC_API_KEY set; used template narration.")
    try:
        settings = get_settings()
        response = _client().messages.create(
            model=settings.kairos_llm_model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(payload, indent=2, default=str)}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return (
            LLMOutcome(text, True, "Narrated by Claude from computed values.")
            if text
            else LLMOutcome(fallback, False, "Empty LLM response; used template narration.")
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("LLM narration failed: %s", exc)
        return LLMOutcome(fallback, False, f"LLM unavailable ({type(exc).__name__}); used template.")
