from __future__ import annotations

import os
from typing import Dict, List

import httpx
import requests
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

load_dotenv()


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Environment variable {name} is required but not set.")
    return value


def _get_env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _truncate(text: str, limit: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


# -----------------------------------------------------------------------------
# Config (all from .env)
# -----------------------------------------------------------------------------

# Backend selector: "openai" | "compatible"
LLM_BACKEND = _get_env("LLM_BACKEND", "compatible")

COMPATIBLE_URL = _get_env("COMPATIBLE_URL", "https://api.bltcy.ai/v1/chat/completions")
COMPATIBLE_MODEL = _get_env("COMPATIBLE_MODEL", "gpt-4o-mini")

OPENAI_MODEL = _get_env("OPENAI_MODEL", "gpt-4o-mini")


# -----------------------------------------------------------------------------
# Compatible API (OpenAI-compatible relay)
# -----------------------------------------------------------------------------
def call_compatible_api(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.5,
    max_tokens: int = 200,
) -> str:
    api_key = _require_env("COMPATIBLE_KEY")
    _model = model or COMPATIBLE_MODEL

    payload = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(COMPATIBLE_URL, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            print(
                "[Compatible usage] "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, "status_code", None)
        body_text = getattr(e.response, "text", "") if e.response is not None else ""
        print("Compatible API call failed:")
        print(f"  url: {COMPATIBLE_URL}")
        print(f"  status: {status_code}")
        print(f"  model: {_model}")
        print(f"  response: {_truncate(body_text)}")
        raise


# -----------------------------------------------------------------------------
# OpenAI official SDK
# -----------------------------------------------------------------------------
def _build_openai_client(api_key: str, timeout_s: float = 60.0) -> OpenAI:
    http_client = httpx.Client(
        timeout=httpx.Timeout(timeout_s, connect=10.0),
        limits=httpx.Limits(max_connections=50, max_keepalive_connections=0),
        http2=True,
    )
    return OpenAI(api_key=api_key, http_client=http_client)


_RETRYABLE = (
    httpx.ReadTimeout,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
    httpx.ConnectError,
)


@retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_RETRYABLE),
)
def call_openai_api(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.2,
    max_tokens: int = 200,
) -> str:
    api_key = _require_env("OPENAI_KEY")
    _model = model or OPENAI_MODEL
    client = _build_openai_client(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI API request failed:", repr(e))
        raise


# -----------------------------------------------------------------------------
# Unified entry point — reads LLM_BACKEND from .env
# -----------------------------------------------------------------------------
def call_llm_api(
    messages: List[Dict[str, str]],
    model: str = None,
    temperature: float = 0.5,
    max_tokens: int = 200,
) -> str:
    """
    Unified LLM chat call. Routes to the backend configured in .env:
      LLM_BACKEND=compatible  →  call_compatible_api  (OpenAI-compatible relay)
      LLM_BACKEND=openai      →  call_openai_api      (OpenAI official SDK)
    """
    if LLM_BACKEND == "openai":
        return call_openai_api(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        return call_compatible_api(messages, model=model, temperature=temperature, max_tokens=max_tokens)
