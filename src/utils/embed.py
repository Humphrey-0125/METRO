import os
import time
import json
import requests
from typing import List, Optional

# ---------------- Config ----------------
EMBED_API_URL = os.getenv("SILICONFLOW_EMBED_URL", "https://api.siliconflow.cn/v1/embeddings")
EMBED_MODEL_NAME = os.getenv("SILICONFLOW_EMBED_MODEL", "BAAI/bge-large-en-v1.5")

EMBED_API_KEY = os.getenv("SILICONFLOW_KEY", "")

MAX_EMBED_TEXT_LENGTH = 800
CHUNK_OVERLAP = 50
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# backoff 参数：更可控
RETRY_BASE_DELAY = 1.0
RETRY_BACKOFF_FACTOR = 1.8


class Embedder:
    def __init__(self, model_name: str = None, api_key: str = None, api_url: str = None, debug: bool = True):
        self.model_name = (model_name or EMBED_MODEL_NAME).strip()
        self.api_key = (api_key or EMBED_API_KEY).strip()
        self.api_url = (api_url or EMBED_API_URL).strip()
        self.debug = bool(debug)

        if not self.api_key:
            raise ValueError(
                "❌ SiliconFlow API key is missing. Please set env SILICONFLOW_API_KEY "
                "or pass api_key explicitly."
            )
        if not (self.api_url and self.api_url.startswith("http")):
            raise ValueError(f"Invalid EMBED_API_URL: {self.api_url}")

        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "persuasion-agent/1.0 (embedding-debug)",
        })

        if self.debug:
            key_head = self.api_key[:6] + "..." if self.api_key else "EMPTY"
            print(f"[EMBED][INIT] api_url={self.api_url}")
            print(f"[EMBED][INIT] model={self.model_name}")
            print(f"[EMBED][INIT] api_key_head={key_head}")

    def _split_text(self, text: str) -> List[str]:
        if not text:
            return [""]

        lines = text.split("\n")
        chunks = []
        current = ""

        for ln in lines:
            ln = ln.strip()
            while len(ln) > MAX_EMBED_TEXT_LENGTH:
                head = ln[:MAX_EMBED_TEXT_LENGTH]
                ln = ln[MAX_EMBED_TEXT_LENGTH:]
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(head)

            add_len = (1 if current else 0) + len(ln)
            if len(current) + add_len <= MAX_EMBED_TEXT_LENGTH:
                current = (current + "\n" + ln) if current else ln
            else:
                if current:
                    chunks.append(current)
                current = ln

        if current:
            chunks.append(current)

        if len(chunks) == 1 and len(chunks[0]) > MAX_EMBED_TEXT_LENGTH:
            fixed = []
            s = chunks[0]
            start = 0
            step = MAX_EMBED_TEXT_LENGTH - CHUNK_OVERLAP if CHUNK_OVERLAP < MAX_EMBED_TEXT_LENGTH else MAX_EMBED_TEXT_LENGTH
            while start < len(s):
                fixed.append(s[start:start + MAX_EMBED_TEXT_LENGTH])
                start += step
            return fixed

        return chunks

    def _debug_http(self, resp: requests.Response, payload: dict):
        """打印足够的信息定位 403/401，但不泄露敏感内容。"""
        try:
            text = resp.text
        except Exception:
            text = "<no resp.text>"
        if text and len(text) > 1200:
            text = text[:1200] + "...(truncated)"

        # 只显示 input 的长度，不显示内容
        input_len = None
        if isinstance(payload, dict) and "input" in payload and isinstance(payload["input"], str):
            input_len = len(payload["input"])

        print(f"[EMBED][HTTP] status={resp.status_code}")
        print(f"[EMBED][HTTP] url={resp.url}")
        print(f"[EMBED][HTTP] model={payload.get('model')}")
        print(f"[EMBED][HTTP] input_len={input_len}")
        print(f"[EMBED][HTTP] resp_body={text}")

    def _encode_single(self, text: str) -> List[float]:
        payload = {"model": self.model_name, "input": text}
        last_err = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.post(self.api_url, json=payload, timeout=REQUEST_TIMEOUT)

                # ✅ 先在这里做诊断输出，再 raise
                if resp.status_code in (401, 403):
                    self._debug_http(resp, payload)
                    # 这类错误重试也没意义，直接抛出
                    resp.raise_for_status()

                # 其他错误：也打印一下 body，方便定位（比如 429/5xx）
                if resp.status_code >= 400:
                    self._debug_http(resp, payload)
                    resp.raise_for_status()

                result = resp.json()
                data = result.get("data")
                if not data or not isinstance(data, list) or "embedding" not in data[0]:
                    raise ValueError(f"Unexpected response format: {result}")

                emb = data[0]["embedding"]
                if not isinstance(emb, list):
                    raise ValueError("Embedding is not a list")
                return emb

            except Exception as e:
                last_err = e
                if attempt < MAX_RETRIES:
                    sleep_s = RETRY_BASE_DELAY * (RETRY_BACKOFF_FACTOR ** (attempt - 1))
                    if self.debug:
                        print(f"[EMBED][RETRY] attempt={attempt}/{MAX_RETRIES} err={type(e).__name__}: {e} sleep={sleep_s:.2f}s")
                    time.sleep(sleep_s)
                else:
                    raise last_err

        raise last_err

    def encode(self, text: str) -> List[float]:
        if text is None:
            text = ""
        text = text.replace(" | ", "\n").strip()

        chunks = self._split_text(text)

        if len(chunks) == 1:
            vec = self._encode_single(chunks[0])
            return self._l2_normalize(vec)

        dim = None
        acc = None
        count = 0
        for ch in chunks:
            vec = self._encode_single(ch)
            if dim is None:
                dim = len(vec)
                acc = [0.0] * dim
            if len(vec) != dim:
                continue
            for i in range(dim):
                acc[i] += vec[i]
            count += 1

        if not acc or count == 0:
            raise ValueError("No embeddings generated")

        merged = [x / count for x in acc]
        return self._l2_normalize(merged)

    @staticmethod
    def _l2_normalize(vec: List[float]) -> List[float]:
        norm = sum(x * x for x in vec) ** 0.5
        if norm > 0:
            return [x / norm for x in vec]
        return vec