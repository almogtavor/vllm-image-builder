"""SGLang backend client for the modular middleware — a drop-in analogue of the
vLLM-fork ``RenderClient`` (middleware/render_client.py) so the existing PIC
pipeline (``build_pic_prompt`` / ``PromptProcessor``) works UNCHANGED against an
SGLang server. Shared by qcfuse and redknot (both are SGLang-based).

Why this exists: SGLang has none of the vLLM fork's token-level endpoints
(``/chat/completions/render``, ``/chat/completions/parse``, the vLLM ``/metrics``
KV-config gauge). So we provide the same three methods the middleware calls on a
RenderClient, implemented against SGLang:

  * ``render_chat``  -> local HF tokenizer ``apply_chat_template`` (returns token_ids)
  * ``fetch_block_size`` -> SGLang page size via ``/get_server_info`` (usually 1)
  * ``parse_output`` -> tool-call detection over the raw text (native SGLang parsers)

Plus one SGLang-specific extra the middleware needs because SGLang's OpenAI
``/v1/completions`` does NOT accept pre-tokenized ``input_ids`` the way vLLM does:

  * ``generate(token_ids, sampling)`` -> SGLang native ``POST /generate`` with
    ``{"input_ids": [...], "sampling_params": {...}}`` (its io_struct accepts input_ids).

The whole point (vs the old passthrough shortcut): real TOKENS flow through the
middleware — it tokenizes, the span policy runs on token_ids, and SGLang decodes
from those exact ids. No message-level proxy.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx


class SglangClient:
    """RenderClient-compatible client backed by an SGLang server + local tokenizer."""

    def __init__(
        self,
        base_url: str,
        model: str,
        http_client: httpx.Client,
        # RedKnot decode is ~4x slower (per-token SDPA, no cuda-graph); a 4096-tok
        # gen at the effective ~10 tok/s under concurrency exceeds the old 300s cap
        # -> ReadTimeout -> retry storm. Env-configurable; 600s default.
        timeout: float = float(os.environ.get("MIDDLEWARE_RENDER_TIMEOUT", "600")),
        tokenizer_path: Optional[str] = None,
        tool_call_parser: str = "qwen25",
    ) -> None:
        # base_url is the OpenAI-style root (…/v1); SGLang's native routes live at
        # the SERVER root (/generate, /get_server_info), not under /v1.
        self._base_url = base_url.rstrip("/")
        self._root = self._base_url[:-3] if self._base_url.endswith("/v1") else self._base_url
        self._root = self._root.rstrip("/")
        self._model = model
        self._http = http_client
        self._timeout = timeout
        self._parser_name = tool_call_parser
        # Local tokenizer: SGLang can't render for us, so we replicate the chat
        # template + tokenization here (same model repo the server loaded).
        # Prefer a baked-in path (SGLANG_TOKENIZER_PATH, set in the sglang mw image)
        # so the pod loads it offline — no HF download, no /tmp cache permission crash.
        from transformers import AutoTokenizer  # lazy: heavy import
        src = tokenizer_path or os.environ.get("SGLANG_TOKENIZER_PATH") or model
        self._tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        self._sep_ids = None  # lazily-encoded <|blendsep|> token ids (QCFuse blend)

    # -- RenderClient-compatible surface -------------------------------------

    def render_chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """Render messages (+tools) to token_ids via the local HF chat template.

        Mirrors RenderClient.render_chat exactly (same signature + return type), so
        build_pic_prompt's incremental-prefix boundary probing works unchanged.
        """
        # DeepSeek-V4 has no jinja chat template: it ships a python encoder
        # (vendored encoding_dsv4.py, from the RedKnot fork's serving path).
        # drop_thinking=False keeps prior-turn reasoning -> renders stay
        # append-only (prefix-STABLE), which the PIC span build requires.
        if os.environ.get("SGLANG_CHAT_ENCODING") == "dsv4":
            import encoding_dsv4
            msgs = list(messages)
            if tools and msgs:
                # dsv4 renders tools via render_tools on the message stream
                msgs[0] = {**msgs[0], "tools": encoding_dsv4.tools_from_openai_format(tools)}
            text = encoding_dsv4.encode_messages(
                msgs, thinking_mode="thinking", drop_thinking=False
            )
            ids = self._tok(text, add_special_tokens=False)["input_ids"]
            if not isinstance(ids, list) or not ids:
                raise ValueError("dsv4 encoder produced no token_ids")
            return ids
        # Qwen3.5's template does `tool_call.arguments|items` (expects a dict), but
        # OpenAI-wire tool_calls carry `arguments` as a JSON *string*. Coerce to dict
        # so the template renders (else `Can only get item pairs from a mapping`).
        messages = self._normalize_tool_call_args(messages)
        # enable_thinking=True: keep Qwen3 reasoning ON (the model reasons in
        # <think>..</think> before acting). SGLANG_ENABLE_THINKING can force it off.
        think = os.environ.get("SGLANG_ENABLE_THINKING", "1") != "0"
        ids = self._tok.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            enable_thinking=think,
        )
        # Qwen3's template returns a BatchEncoding (dict-like) when tools are present,
        # a plain list otherwise. Normalise to a flat list of ints.
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if ids and isinstance(ids[0], list):  # batched -> take the single row
            ids = ids[0]
        if not isinstance(ids, list) or not ids:
            raise ValueError("local tokenizer produced no token_ids")
        return ids

    def fetch_block_size(self) -> Optional[int]:
        """Always 1: SGLang needs NO span padding. Its radix cache is token-granular
        and RedKnot's SegPagedAttention maps head segments via virtual-page
        indirection — block alignment is a vLLM-fork (block-hash prefix cache)
        requirement only. Pinning 1 makes pad_tokens a no-op by construction; a
        page_size>1 backend would otherwise make us splice pad tokens INTO the
        /generate input_ids (prompt pollution, not cache alignment)."""
        return 1

    def parse_output(
        self, text: str, tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Detect tool calls in the raw model text, returning the SAME dict shape
        the middleware expects from the fork /parse:
        {reasoning_content, content, tool_calls, tools_called}.

        Qwen3 emits Hermes-style tool calls: ``<tool_call>{"name":..,"arguments":..}
        </tool_call>`` (possibly several). We parse that format directly — no heavy
        sglang dependency in the middleware image; just a regex + json.
        """
        # DeepSeek-V4 output uses DSML blocks, not Hermes tags: parse with the
        # vendored encoder (returns OpenAI-format tool_calls; malformed -> no calls).
        if os.environ.get("SGLANG_CHAT_ENCODING") == "dsv4":
            import encoding_dsv4
            try:
                m = encoding_dsv4.parse_message_from_completion_text(
                    text, thinking_mode="thinking"
                )
                calls = m.get("tool_calls") or []
                return {
                    "reasoning_content": m.get("reasoning_content") or None,
                    "content": m.get("content") or "",
                    "tool_calls": calls,
                    "tools_called": bool(calls),
                }
            except ValueError:
                return {"reasoning_content": None, "content": text,
                        "tool_calls": [], "tools_called": False}
        reasoning, body = self._split_think(text)
        calls = self._parse_qwen_tool_calls(body)
        if not calls:
            return {"reasoning_content": reasoning, "content": body or text,
                    "tool_calls": None, "tools_called": False}
        # strip the <tool_call> blocks out of the visible content (both formats
        # wrap in <tool_call>...</tool_call>, so _QC_BLOCK_RE covers each)
        content = self._QC_BLOCK_RE.sub("", body).strip() or None
        return {"reasoning_content": reasoning, "content": content,
                "tool_calls": calls, "tools_called": True}

    # -- SGLang-specific: pre-tokenized generation ---------------------------

    def generate(
        self,
        token_ids: List[int],
        sampling_params: Dict[str, Any],
        span_starts: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """SGLang native POST /generate with pre-tokenized input_ids. Returns
        {text, completion_tokens, finish_reason, cached_tokens} — cached_tokens is
        SGLang's KV-reuse count (meta_info.cached_tokens), the ground-truth signal
        that a warmed span's KV was reused rather than re-prefilled.

        When span_starts is given AND QCFuse blend is enabled (QCFUSE_RATIO env),
        inject the <|blendsep|> token at each span boundary and pass the blend kwargs
        so QCFuse computes each span INDEPENDENTLY in a single prefill (block-diagonal
        via the separator) and selectively recomputes for the query — no warmup call."""
        input_ids, blend = self._maybe_blend(token_ids, span_starts)
        body = {
            "input_ids": input_ids,
            "sampling_params": sampling_params,
            "stream": False,
            # keep special tokens in the returned text so parse_output can see the
            # Qwen <tool_call> markers (SGLang /generate top-level flag, not sampling).
            "return_text_in_logprobs": False,
            "skip_special_tokens": False,
            **blend,
        }
        r = self._http.post(self._root + "/generate", json=body, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        # SGLang returns a dict (single) with text + meta_info.
        if isinstance(data, list):
            data = data[0]
        meta = data.get("meta_info", {}) or {}
        fr = meta.get("finish_reason")
        return {
            "text": data.get("text", ""),
            "completion_tokens": meta.get("completion_tokens")
            or meta.get("output_len") or 0,
            "cached_tokens": meta.get("cached_tokens") or 0,
            "prompt_tokens": meta.get("prompt_tokens") or len(token_ids),
            "finish_reason": (fr.get("type") if isinstance(fr, dict) else fr) or "stop",
        }

    # QCFuse blend config (env-driven; ratio=rho, the recompute fraction / legolink-K analog)
    _BLEND_SEP = "<|blendsep|>"

    def _maybe_blend(self, token_ids, span_starts):
        """If blend is on and spans exist, splice the <|blendsep|> token at each span
        boundary and return (new_input_ids, blend_kwargs). Else return (token_ids, {})."""
        ratio = os.environ.get("QCFUSE_RATIO")
        if not ratio or not span_starts:
            return token_ids, {}
        if self._sep_ids is None:
            self._sep_ids = self._tok.encode(self._BLEND_SEP, add_special_tokens=False)
        if not self._sep_ids:
            return token_ids, {}
        # insert sep tokens at each span start (descending so earlier indices stay valid)
        ids = list(token_ids)
        for s in sorted({int(x) for x in span_starts if 0 < int(x) < len(ids)}, reverse=True):
            ids[s:s] = self._sep_ids
        crit = os.environ.get("QCFUSE_CRITICAL_LAYERS")
        blend = {
            "blend_style": "DO_BLEND_FINISH",
            "separator": self._BLEND_SEP,
            "is_contextblend": True,
            "context_cache_source": "query",
            "ratio": float(ratio),
            "digest_ratio": float(os.environ.get("QCFUSE_DIGEST_RATIO", "0.1")),
        }
        if crit:
            blend["critical_layers"] = [int(x) for x in crit.split(",") if x.strip()]
        return ids, blend

    def warm(self, token_ids: List[int]) -> Dict[str, Any]:
        """Prefill token_ids with 0 new tokens so SGLang caches their KV — primes a
        span for reuse. Returns the generate() dict (cached_tokens tells how much of
        THIS prefill was itself already cached)."""
        return self.generate(token_ids, {"max_new_tokens": 0, "temperature": 0})

    # -- RedKnot: build offline segments per span, then query reusing them ----
    # RedKnot can't collapse to one prefill (its online forward splices offline
    # segments that must already exist), so this is N builds + 1 query — the build
    # IS the independent-span compute; the query applies the head-class sparse reuse.

    def build_segment(self, token_ids: List[int], segment_id: str) -> Dict[str, Any]:
        """Dense-prefill this span and snapshot its KV as RedKnot offline segment
        <segment_id> (via the __RKBUILD__ sentinel). Returns the generate() dict."""
        body = {
            "input_ids": token_ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "stream": False,
            "redknot_offline_segments": ["__RKBUILD__:%s" % segment_id],
        }
        r = self._http.post(self._root + "/generate", json=body, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        return {"segment_id": segment_id, "meta_info": data.get("meta_info", {})}

    def generate_redknot(
        self,
        token_ids: List[int],
        sampling_params: Dict[str, Any],
        segment_ids: List[str],
    ) -> Dict[str, Any]:
        """Query reusing pre-built offline segments: the RedKnot backend splices their
        KV and applies the per-head-class sparse mask. segment_ids name the builds."""
        body = {
            "input_ids": token_ids,
            "sampling_params": sampling_params,
            "stream": False,
            "return_text_in_logprobs": False,
            "skip_special_tokens": False,
            "redknot_offline_segments": list(segment_ids),
        }
        r = self._http.post(self._root + "/generate", json=body, timeout=self._timeout)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            data = data[0]
        meta = data.get("meta_info", {}) or {}
        fr = meta.get("finish_reason")
        # Backend receipt of what was ACTUALLY spliced (vs asked): the fork's
        # redknot_backend records it per query and the output streamer surfaces
        # it via cached_tokens_details {redknot_requested/spliced/missing}.
        details = meta.get("cached_tokens_details") or {}
        receipt = (
            {k: details[k] for k in
             ("redknot_requested", "redknot_spliced", "redknot_missing")
             if k in details}
            or None
        )
        return {
            "text": data.get("text", ""),
            "completion_tokens": meta.get("completion_tokens")
            or meta.get("output_len") or 0,
            "cached_tokens": meta.get("cached_tokens") or 0,
            "prompt_tokens": meta.get("prompt_tokens") or len(token_ids),
            "finish_reason": (fr.get("type") if isinstance(fr, dict) else fr) or "stop",
            "redknot_segments": list(segment_ids),
            "redknot_receipt": receipt,
        }

    # -- internals ------------------------------------------------------------

    # Qwen3 / Hermes tool-call block: <tool_call>{...json...}</tool_call>
    _TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    # Qwen3.5 / qwen3_coder nested XML: <tool_call><function=NAME>
    # <parameter=P>\nVAL\n</parameter>...</function></tool_call>
    _QC_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    _QC_FUNC_RE = re.compile(r"<function=(.*?)>(.*?)</function>", re.DOTALL)
    _QC_PARAM_RE = re.compile(
        r"<parameter=(.*?)>\n?(.*?)\n?</parameter>", re.DOTALL)

    @staticmethod
    def _normalize_tool_call_args(messages):
        """Return a copy of messages where every assistant tool_call's `arguments`
        is a dict (JSON-string args -> parsed). Qwen3.5's template iterates them
        with `|items`, so a string arg crashes the render during span probing."""
        out = []
        for m in messages:
            tcs = m.get("tool_calls")
            if not tcs:
                out.append(m)
                continue
            new_tcs = []
            for tc in tcs:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                    tc = {**tc, "function": {**fn, "arguments": args}}
                new_tcs.append(tc)
            out.append({**m, "tool_calls": new_tcs})
        return out

    def _split_think(self, text: str):
        """Pull out <think>...</think> reasoning (Qwen3 thinking mode); return
        (reasoning_or_None, remaining_text)."""
        m = self._THINK_RE.search(text)
        if not m:
            return None, text
        reasoning = m.group(1).strip()
        return (reasoning or None), self._THINK_RE.sub("", text, count=1).strip()

    def _parse_qwen_tool_calls(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """Extract tool calls into OpenAI tool_call dicts. SGLANG_TOOL_PARSER selects
        the format: qwen3_coder = Qwen3.5's nested <function=..><parameter=..> XML;
        default = Hermes-style <tool_call>{json}</tool_call>."""
        if os.environ.get("SGLANG_TOOL_PARSER") == "qwen3_coder":
            return self._parse_qwen3_coder_tool_calls(text)
        calls = []
        for i, m in enumerate(self._TOOLCALL_RE.finditer(text)):
            try:
                obj = json.loads(m.group(1))
            except Exception:
                continue
            args = obj.get("arguments", {})
            calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": obj.get("name", ""),
                    # OpenAI expects arguments as a JSON string
                    "arguments": args if isinstance(args, str) else json.dumps(args),
                },
            })
        return calls or None

    def _parse_qwen3_coder_tool_calls(
        self, text: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse Qwen3.5 nested tool calls: each <tool_call> holds a
        <function=NAME> with <parameter=P>\\nVALUE\\n</parameter> children.
        Mirrors the fork's Qwen3CoderDetector; args become a JSON-string."""
        calls = []
        idx = 0
        for block in self._QC_BLOCK_RE.finditer(text):
            for fm in self._QC_FUNC_RE.finditer(block.group(1)):
                name = fm.group(1).strip()
                args = {p.group(1).strip(): p.group(2)
                        for p in self._QC_PARAM_RE.finditer(fm.group(2))}
                calls.append({
                    "id": f"call_{idx}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                })
                idx += 1
        return calls or None
