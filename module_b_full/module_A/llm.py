"""
Shared LLM client helpers.

Fallback order:
  1. Gemini 3.0 Flash family
  2. Gemini 2.5 Flash
  3. Gemini 1.5 Flash family
  4. OpenAI gpt-4o
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from google import genai

_CONFIG_PATH = Path(__file__).with_name("config.py")
_spec = importlib.util.spec_from_file_location("module_a_runtime_config", _CONFIG_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load Module A config from {_CONFIG_PATH}")
_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_config)

GEMINI_API_KEY = _config.GEMINI_API_KEY
MODEL_FAST = _config.MODEL_FAST

# Project-requested order:
# Gemini 3.0 -> Gemini 2.5 Flash -> Gemini 1.5 Flash -> OpenAI.
# Some families expose different concrete IDs, so we try aliases in-family.
_MODEL_FAMILIES = {
    "gemini-2.5-flash": [
        "gemini-2.5-flash",
    ],
    "gemini-3-flash-preview": [
        "gemini-3-flash-preview",
    ],
    "gemini-2.0-flash": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-flash-latest",
    ],
}
_DEFAULT_MODEL_ORDER = [
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.0-flash",
]
_RETRIES_PER_MODEL = 2  # 2 attempts per model for transient errors
_RETRY_DELAY_SEC = 3.0

_OVERLOAD_MESSAGE = (
    "Сервера моделей сейчас перегружены или недоступны. "
    "Проблема не на стороне кода, а на стороне моделей. "
    "Попробуйте немного позже или снова."
)

_RETRYABLE_TOKENS = ("503", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "429", "OVERLOADED", "MODEL_OVERLOADED")
_NOT_FOUND_TOKENS = ("NOT_FOUND", "404")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(token in msg for token in _RETRYABLE_TOKENS)


def _is_not_found(exc: Exception) -> bool:
    msg = str(exc).upper()
    return any(token in msg for token in _NOT_FOUND_TOKENS)


_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _openai_generate(prompt: str) -> str:
    try:
        import openai as _openai
    except ImportError as exc:
        raise RuntimeError("openai package not installed") from exc

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = _openai.OpenAI(api_key=openai_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        timeout=60,
    )
    return (resp.choices[0].message.content or "").strip()


def _openai_multimodal(contents: Any) -> str:
    if isinstance(contents, str):
        return _openai_generate(contents)
    if isinstance(contents, list):
        text_parts = []
        for part in contents:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        return _openai_generate("\n".join(text_parts)) if text_parts else ""
    return ""


def _gemini_generate(prompt_or_contents: Any, model: str) -> str:
    def _call() -> str:
        resp = get_client().models.generate_content(model=model, contents=prompt_or_contents)
        return (getattr(resp, "text", "") or "").strip()

    last_err: Exception | None = None
    for attempt in range(1, _RETRIES_PER_MODEL + 1):
        try:
            return _call()
        except Exception as exc:
            last_err = exc
            if _is_retryable(exc) and attempt < _RETRIES_PER_MODEL:
                time.sleep(_RETRY_DELAY_SEC * attempt)
                continue
            raise
    raise last_err  # pragma: no cover


def _build_model_order(primary_model: str) -> list[str]:
    order = list(_DEFAULT_MODEL_ORDER)
    if primary_model in order:
        order.remove(primary_model)
    order.insert(0, primary_model)
    return order


def _try_openai(prompt_or_contents: Any, multimodal: bool) -> str:
    if multimodal:
        return _openai_multimodal(prompt_or_contents)
    return _openai_generate(str(prompt_or_contents))


def _generate_with_fallback(prompt_or_contents: Any, primary_model: str, multimodal: bool = False) -> str:
    chain = _build_model_order(primary_model)
    last_err: Exception | None = None
    primary_failed = False

    for family_name in chain:
        candidates = _MODEL_FAMILIES.get(family_name, [family_name])
        family_err: Exception | None = None

        for model_id in candidates:
            try:
                result = _gemini_generate(prompt_or_contents, model_id)
                if family_name != primary_model or model_id != primary_model:
                    print(f"  [LLM] Used fallback model: {model_id}")
                return result
            except Exception as exc:
                family_err = exc
                last_err = exc

                if _is_not_found(exc):
                    print(f"  [LLM] {model_id} not found / invalid, trying next alias...")
                    continue

                if _is_retryable(exc):
                    print(f"  [LLM] {model_id} unavailable (503), trying next model...")
                    break

                raise

        if family_err and (_is_retryable(family_err) or _is_not_found(family_err)):
            if not primary_failed:
                primary_failed = True
                # Try OpenAI immediately after primary model fails
                openai_key = os.environ.get("OPENAI_API_KEY", "")
                if openai_key:
                    print("  [LLM] Primary Gemini unavailable, trying OpenAI...")
                    try:
                        result = _try_openai(prompt_or_contents, multimodal)
                        print("  [LLM] Used OpenAI gpt-4o-mini.")
                        return result
                    except Exception as exc:
                        print(f"  [LLM] OpenAI failed: {exc}, continuing with other Gemini models...")
                        last_err = exc
            continue

    print("  [LLM] All Gemini models failed, trying OpenAI...")
    try:
        result = _try_openai(prompt_or_contents, multimodal)
        print("  [LLM] Used OpenAI as fallback.")
        return result
    except Exception as exc:
        last_err = exc
        print(f"  [LLM] OpenAI also failed: {exc}")

    print(f"\n  !! {_OVERLOAD_MESSAGE}\n")
    raise RuntimeError(_OVERLOAD_MESSAGE) from last_err


def generate_text(prompt: str, model: str = MODEL_FAST) -> str:
    return _generate_with_fallback(prompt, model, multimodal=False)


def generate_multimodal(contents: Any, model: str = MODEL_FAST) -> str:
    return _generate_with_fallback(contents, model, multimodal=True)


def upload_file(path: Path, mime_type: str | None = None):
    config = {"mime_type": mime_type} if mime_type else None
    last_err: Exception | None = None
    for attempt in range(1, _RETRIES_PER_MODEL + 1):
        try:
            return get_client().files.upload(file=str(path), config=config)
        except Exception as exc:
            last_err = exc
            if _is_retryable(exc) and attempt < _RETRIES_PER_MODEL:
                time.sleep(_RETRY_DELAY_SEC * attempt)
                continue
            raise
    raise last_err  # pragma: no cover


def get_file(name: str):
    return get_client().files.get(name=name)


def wait_for_file(name: str, attempts: int = 30, delay_sec: float = 2.0):
    for _ in range(attempts):
        file_obj = get_file(name)
        state = getattr(getattr(file_obj, "state", None), "name", "")
        if state != "PROCESSING":
            return file_obj
        time.sleep(delay_sec)
    return get_file(name)


def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {}


def extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text or "", re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []
