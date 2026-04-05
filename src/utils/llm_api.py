from __future__ import annotations

import os
from typing import Dict, List

import httpx
import requests
from openai import OpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential


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
# Config (all keys from environment variables)
# -----------------------------------------------------------------------------

COMPATIBLE_URL = _get_env("COMPATIBLE_URL", "https://api.bltcy.ai/v1/chat/completions")
COMPATIBLE_MODEL = _get_env("COMPATIBLE_MODEL", "gpt-4o-mini")
COMPATIBLE_KEY = _get_env("COMPATIBLE_KEY")
OPENAI_URL = _get_env("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = _get_env("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_KEY = _get_env("OPENAI_KEY")
SILICONFLOW_URL = _get_env("SILICONFLOW_CHAT_URL", "https://api.siliconflow.cn/v1/chat/completions")
SILICONFLOW_MODEL = _get_env("SILICONFLOW_CHAT_MODEL", "Qwen/Qwen3-32B")
SILICONFLOW_KEY = _get_env("SILICONFLOW_KEY")


# -----------------------------------------------------------------------------
# SiliconFlow
# -----------------------------------------------------------------------------
def call_siliconflow_api(
    messages: List[Dict[str, str]],
    model: str = SILICONFLOW_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 300,
    timeout: float = 60.0,
) -> str:
    api_key = _require_env(SILICONFLOW_KEY)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(SILICONFLOW_URL, json=payload, headers=headers, timeout=timeout)
        if response.status_code != 200:
            try:
                error_detail = response.json()
                print(f"SiliconFlow API error {response.status_code}: {error_detail}")
            except Exception:
                print(f"SiliconFlow API error {response.status_code}: {response.text[:500]}")
            response.raise_for_status()

        result = response.json()
        usage = result.get("usage") if isinstance(result, dict) else None
        if isinstance(usage, dict):
            print(
                "[SiliconFlow usage] "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"total={usage.get('total_tokens')}"
            )
        return result["choices"][0]["message"]["content"].strip()

    except requests.exceptions.RequestException as e:
        print(f"SiliconFlow API call failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            try:
                print(f"Error details: {e.response.json()}")
            except Exception:
                print(f"Error response text: {e.response.text[:500]}")
        raise


# -----------------------------------------------------------------------------
# Plato-compatible
# -----------------------------------------------------------------------------
def call_compatible_api(
    messages: List[Dict[str, str]],
    model: str = COMPATIBLE_MODEL,
    temperature: float = 0.5,
    max_tokens: int = 200,
) -> str:
    api_key = _require_env(COMPATIBLE_KEY)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "stream": False,
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
        print(f"  model: {model}")
        print(f"  payload: {_truncate(payload)}")
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
    model: str = OPENAI_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 200,
) -> str:
    api_key = _require_env(OPENAI_KEY)
    client = _build_openai_client(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print("OpenAI API request failed:", repr(e))
        raise
