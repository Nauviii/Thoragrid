"""Thin wrapper around Groq API for LLM Call 1 and 2."""

import time

from groq import BadRequestError, Groq, RateLimitError
from config.settings import settings

_client = Groq(api_key=settings.groq_api_key)



def _is_json_validation_failure(exc: BadRequestError) -> bool:
    """Whether a 400 is Groq reporting that the generation did not satisfy the JSON schema."""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body.get("error", {}).get("code") == "json_validate_failed"
    return "json_validate_failed" in str(exc)


def call_groq(
    system_prompt: str,
    user_prompt: str,
    schema: dict | None = None,
    schema_name: str | None = None,
    max_tokens: int | None = None,
    max_retries: int = 3,
) -> str:
    
    kwargs = {
        "model":       settings.groq_model,
        "temperature": settings.llm_temperature,
        "max_tokens":  max_tokens or settings.llm_max_tokens,
        "reasoning_effort": settings.llm_reasoning_effort,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        }

    for attempt in range(max_retries + 1):
        try:
            response = _client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except RateLimitError:
            if attempt == max_retries:
                raise
            time.sleep(2 ** attempt)
        except BadRequestError as exc:
            if _is_json_validation_failure(exc) and attempt < max_retries:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise