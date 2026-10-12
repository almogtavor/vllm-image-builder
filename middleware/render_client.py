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

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Fan-out bursts to a single vLLM exhaust its TCP accept backlog (Errno 111) or emit transient 5xx; retry with backoff instead of surfacing a 500.
_RETRY_STATUS = {502, 503, 504}
_RETRY_BACKOFF = (0.1, 0.25, 0.5, 1.0)  # seconds; len = max retries after the first try

# An engine crash restarts the backend pod, which takes minutes. The ladder above
# rides out a burst, not an outage: it gives up after ~1.85s, the caller sees a
# 500, and every in-flight agent session dies of a fault it could have waited
# out -- losing a whole benchmark cell. Past the ladder, a backend that is
# *absent* (as opposed to slow) is therefore polled until it answers.
_DOWN_WAIT = float(os.environ.get("MIDDLEWARE_BACKEND_DOWN_WAIT", "420"))
_DOWN_POLL = float(os.environ.get("MIDDLEWARE_BACKEND_DOWN_POLL", "5"))


def backend_is_down(exc: BaseException) -> bool:
    """True when ``exc`` says the backend is absent, not merely slow.

    A read timeout does not qualify: the request was accepted, so re-sending it
    only adds load to a backend that is already struggling.
    """
    if getattr(exc, "status_code", None) == 503:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 503:
        return True
    seen = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, (httpx.ConnectError, httpx.ConnectTimeout)):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def riding_out_restart(fn, what: str = "backend call"):
    """Run ``fn()``, re-running it while the backend is down, then re-raise.

    Wraps a whole call rather than a single HTTP request so a stream that dies
    mid-generation is re-issued from scratch, which is what the caller wants: an
    agent asking for a completion does not care that the first attempt was cut
    off by a pod restart.
    """
    deadline = time.monotonic() + _DOWN_WAIT
    while True:
        try:
            return fn()
        except BaseException as exc:
            if not backend_is_down(exc) or time.monotonic() >= deadline:
                raise
            logger.warning("%s: backend down (%s), retrying in %.0fs",
                           what, type(exc).__name__, _DOWN_POLL)
            time.sleep(_DOWN_POLL)

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
        """POST, retrying transient errors with backoff, then polling out a restart; 4xx re-raises."""
        deadline = time.monotonic() + _DOWN_WAIT
        attempt = 0
        while True:
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
                if attempt < len(_RETRY_BACKOFF):
                    time.sleep(_RETRY_BACKOFF[attempt])
                    attempt += 1
                    continue
                # Ladder exhausted. A burst would have cleared by now, so a still
                # unreachable backend is restarting: wait for it rather than
                # killing the session.
                if not backend_is_down(exc) or time.monotonic() >= deadline:
                    raise
                logger.warning("render %s: backend down (%s), retrying in %.0fs",
                               url.rsplit("/", 1)[-1], type(exc).__name__, _DOWN_POLL)
                time.sleep(_DOWN_POLL)

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
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[int]:
        """Render messages (+ optional tools) into canonical token_ids via vLLM.

        ``add_generation_prompt=False`` is used to find per-message token
        boundaries (incremental prefixes); ``True`` produces the real prompt.

        ``chat_template_kwargs`` is forwarded to the template so callers control
        template-level behaviour (e.g. ``reasoning_primer=false`` to suppress the
        chain-of-thought preamble on single-turn tasks). Without forwarding it, the
        prompt the middleware builds would differ from what the caller asked for.

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
        if chat_template_kwargs:
            body["chat_template_kwargs"] = chat_template_kwargs

        resp = self._post(self._url, body)
        resp.raise_for_status()
        data = resp.json()

        token_ids = data.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(
                f"render endpoint returned no token_ids (keys={list(data)[:8]})"
            )
        return token_ids
