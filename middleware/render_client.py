"""Client for vLLM's render endpoint (`/v1/chat/completions/render`).

vLLM exposes a GPU-less render path that runs its OWN preprocessing —
`apply_chat_template` (with tools) + tokenization — and returns the canonical
``token_ids`` it would feed the model, without generating. Using it makes vLLM
the single source of truth for prompt construction, so the middleware no longer
has to load a tokenizer or replicate the chat template (which is where
template/tokenizer drift bugs come from).

Endpoint: ``POST {base_url}/chat/completions/render`` where ``base_url`` is the
backend's OpenAI base (e.g. ``http://localhost:8000/v1``). Response is a
``GenerateRequest`` whose ``token_ids`` is the rendered+tokenized prompt
(including the assistant generation prompt at the tail).
"""
from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

# Fan-out bursts to a single vLLM exhaust its TCP accept backlog (Errno 111) or emit transient 5xx; retry with backoff instead of surfacing a 500.
_RETRY_STATUS = {502, 503, 504}
_RETRY_BACKOFF = (0.1, 0.25, 0.5, 1.0)  # seconds; len = max retries after the first try

# Per-request backend HTTP timeout. RedKnot decode is ~4x slower (per-token SDPA,
# no cuda-graph), so deep-context generations blow past the old 60s default and
# ReadTimeout -> retry-storm -> stalled agents. Env-configurable; 600s default.
_DEFAULT_TIMEOUT = float(os.environ.get("MIDDLEWARE_RENDER_TIMEOUT", "600"))


class RenderClient:
    def __init__(self, base_url: str, model: str, http_client: httpx.Client, timeout: float = _DEFAULT_TIMEOUT):
        # base_url is the OpenAI base, e.g. http://localhost:8000/v1
        self._base_url = base_url.rstrip("/")
        self._url = self._base_url + "/chat/completions/render"
        self._model = model
        self._http = http_client
        self._timeout = timeout

    def _post(self, url: str, body: Dict[str, Any]) -> httpx.Response:
        """POST, retrying transport errors and 502/503/504 with backoff; 4xx and final transient errors re-raise."""
        last_exc: Optional[Exception] = None
        for attempt in range(len(_RETRY_BACKOFF) + 1):
            try:
                resp = self._http.post(url, json=body, timeout=self._timeout)
                if resp.status_code in _RETRY_STATUS:
                    resp.raise_for_status()
                return resp
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                # Only retry transient classes; a 4xx HTTPStatusError is not retried.
                if isinstance(exc, httpx.HTTPStatusError) and \
                        exc.response.status_code not in _RETRY_STATUS:
                    raise
                last_exc = exc
                if attempt < len(_RETRY_BACKOFF):
                    time.sleep(_RETRY_BACKOFF[attempt])
        raise last_exc  # exhausted retries on a transient error

    def fetch_block_size(self) -> Optional[int]:
        """Read the engine's resolved KV ``block_size`` from vLLM's ``/metrics``
        (the ``vllm:cache_config_info{... block_size="N" ...}`` gauge). Returns
        ``None`` if metrics is unreachable or the field is absent.
        """
        # /metrics is served at the server root, not under /v1.
        root = self._base_url[:-3] if self._base_url.endswith("/v1") else self._base_url
        try:
            resp = self._http.get(root.rstrip("/") + "/metrics", timeout=self._timeout)
            resp.raise_for_status()
        except Exception:
            return None
        m = re.search(r'vllm:cache_config_info\{[^}]*\bblock_size="(\d+)"', resp.text)
        return int(m.group(1)) if m else None

    def parse_output(
        self,
        text: str,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Postprocess raw model output via vLLM's ``/v1/chat/completions/parse``.

        Runs the server's configured reasoning + tool-call parsers (the same
        ones ``/v1/chat/completions`` uses) on text we generated through raw
        ``/v1/completions``. Returns
        ``{reasoning_content, content, tool_calls, tools_called}``.

        Raises on transport/HTTP error — the endpoint is required (no local
        parser fallback).
        """
        body: Dict[str, Any] = {"model": self._model, "text": text}
        if tools:
            body["tools"] = tools
        resp = self._post(self._base_url + "/chat/completions/parse", body)
        resp.raise_for_status()
        return resp.json()

    def render_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Render messages (+ optional tools) into canonical token_ids via vLLM.

        ``add_generation_prompt=False`` is used to find per-message token
        boundaries (incremental prefixes); ``True`` produces the real prompt.

        Raises on transport/HTTP error or if the response lacks token_ids — the
        caller should treat a render failure as a hard error (no silent
        fallback) so prompt construction stays consistent with the backend.
        """
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if tools:
            body["tools"] = tools

        resp = self._post(self._url, body)
        resp.raise_for_status()
        data = resp.json()

        token_ids = data.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(
                f"render endpoint returned no token_ids (keys={list(data)[:8]})"
            )
        return token_ids
