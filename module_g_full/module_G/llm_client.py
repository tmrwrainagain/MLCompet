"""LLM client for methodological assessment.

Wraps Gemini with OpenAI fallback — same strategy as Module A's llm.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from .config import GEMINI_API_KEY, OPENAI_API_KEY, MODEL_FAST

_RETRYABLE = ("503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "OVERLOADED")
_NOT_FOUND = ("NOT_FOUND", "404")


def _is_retryable(e: Exception) -> bool:
    return any(t in str(e).upper() for t in _RETRYABLE)


def _is_not_found(e: Exception) -> bool:
    return any(t in str(e).upper() for t in _NOT_FOUND)


def _gemini(prompt: str, model: str) -> str:
    from google import genai
    client = genai.Client(api_key=GEMINI_API_KEY)
    for attempt in range(1, 3):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (getattr(resp, "text", "") or "").strip()
        except Exception as exc:
            if _is_retryable(exc) and attempt < 2:
                time.sleep(3 * attempt)
                continue
            raise


def _openai(prompt: str) -> str:
    import openai as _openai_lib
    key = OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = _openai_lib.OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=60,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_text(prompt: str, model: str = MODEL_FAST) -> str:
    """Try Gemini models in order, fallback to OpenAI."""
    models = [model, "gemini-2.0-flash", "gemini-2.0-flash-001"]
    last_err: Exception | None = None
    for m in models:
        try:
            return _gemini(prompt, m)
        except Exception as exc:
            last_err = exc
            if _is_retryable(exc) or _is_not_found(exc):
                continue
            raise
    try:
        result = _openai(prompt)
        print("  [LLM-G] Used OpenAI gpt-4o-mini.")
        return result
    except Exception as exc:
        raise RuntimeError("All LLM providers failed") from exc


def extract_json_object(text: str) -> dict:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return {}
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return {}


def extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return []
