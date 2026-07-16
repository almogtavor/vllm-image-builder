"""
Middleware for OpenAI-compatible API with modular processing.
"""

import hashlib
import logging
import os
import subprocess
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal

import json
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from config_models import MiddlewareConfig
from processors import PromptProcessor
from logger import MiddlewareLogger
import request_trace
import span_policy
import warmup


# Request Models
class ChatMessage(BaseModel):
    """Model for a single chat message."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message author"
    )
    content: Optional[str] = Field(default=None, description="The content of the message")
    name: Optional[str] = Field(default=None, description="Function/tool name (tool role)")
    tool_call_id: Optional[str] = Field(default=None, description="ID this message responds to (tool role)")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Past tool calls from this assistant turn"
    )


class ChatCompletionRequest(BaseModel):
    """Model for chat completion requests."""

    model: str = Field(..., description="ID of the model to use")
    messages: List[ChatMessage] = Field(
        ..., description="List of messages comprising the conversation"
    )
    max_tokens: Optional[int] = Field(
        default=4096,
        description="Maximum number of tokens to generate. Default is generous so "
        "clients that omit it aren't silently truncated mid-reasoning (verbose "
        "reasoning models easily exceed a small cap).",
        ge=1,
    )
    temperature: Optional[float] = Field(
        default=0.0, description="Sampling temperature between 0 and 2", ge=0.0, le=2.0
    )
    seed: Optional[int] = Field(
        default=None, description="Random seed for deterministic generation"
    )
    tools: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tools the model may call (OpenAI format)"
    )
    # Anti-repetition sampling. Without forwarding these, the backend degenerates into
    # token repetition (gemma-4 / vLLM #40080) - the cause of middleware-mode loops.
    top_p: Optional[float] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    min_p: Optional[float] = Field(default=None)
    repetition_penalty: Optional[float] = Field(default=None)
    frequency_penalty: Optional[float] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_stream(
    stream,
) -> tuple[str, Optional[str], Optional[int], Optional[int]]:
    """
    Process streaming response from the backend.

    Returns:
        Tuple of (full_text, finish_reason, completion_tokens, cached_tokens).
        Both token counts come from the backend's usage chunk (requires
        stream_options={"include_usage": True}). cached_tokens is the KV
        prefix-cache reuse (usage.prompt_tokens_details.cached_tokens),
        defaulting to 0 when the backend reports details but no count; None if
        the backend didn't report it at all.
    """
    full_text = ""
    finish_reason = None
    completion_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None

    for chunk in stream:
        # The final usage chunk (include_usage) carries token counts and no choices.
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            if getattr(usage, "completion_tokens", None) is not None:
                completion_tokens = usage.completion_tokens
            details = getattr(usage, "prompt_tokens_details", None)
            if details is not None:
                ct = getattr(details, "cached_tokens", None)
                cached_tokens = ct if ct is not None else 0

        if chunk.choices:
            choice = chunk.choices[0]
            if choice.text:
                full_text += choice.text
            if choice.finish_reason:
                finish_reason = choice.finish_reason

    return full_text, finish_reason, completion_tokens, cached_tokens


def calculate_usage_info(
    full_prompt: List[int],
    completion_tokens: Optional[int],
) -> Dict[str, Any]:
    """Build usage info from the prompt token count and the backend's
    completion_tokens (no local tokenizer)."""
    prompt_tokens = len(full_prompt)
    ct = completion_tokens or 0
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": ct,
        "total_tokens": prompt_tokens + ct,
    }


def warm_sglang_spans(client, prepared, request_id) -> List[Dict[str, Any]]:
    """Prefill each PIC span's token prefix through SGLang so its KV is cached, and
    return a per-span reuse check. For span i we warm token_ids[:span_end_i] with 0
    new tokens; SGLang reports cached_tokens = how much of that prefix was ALREADY
    cached (from an earlier warm of the same conversation/prefix). reused=True means
    the span's KV came from cache, not a re-prefill — the proof of reuse the run needs.
    """
    toks = prepared.token_ids
    checks = []
    for i, ch in enumerate(prepared.chunks):
        if ch.get("kind") != "span":
            continue
        end = ch.get("end", 0)
        if end <= 0 or end > len(toks):
            continue
        try:
            r = client.warm(toks[:end])
        except Exception as e:
            logger.warning("Request %s: span %d warm failed: %s", request_id, i, e)
            continue
        cached = int(r.get("cached_tokens") or 0)
        span_start = ch.get("start", 0)
        # a span is "reused" if the cache already covered into this span's own tokens
        reused = cached >= end - 1  # whole prefix incl. the span was cached
        checks.append({
            "span_index": i, "span_start": span_start, "span_end": end,
            "prefix_len": end, "cached_tokens": cached, "reused": reused,
        })
        logger.info(
            "Request %s: span %d [%d:%d] warm -> cached_tokens=%d reused=%s",
            request_id, i, span_start, end, cached, reused,
        )
    return checks


# Segment ids we have already built this process lifetime. RedKnot offline segments are
# position-INDEPENDENT (built at [0,L), spliced anywhere via RoPE realign), so a span with
# identical tokens is the SAME segment across requests -> build once, re-splice cheaply.
# Without this, request-scoped ids rebuilt every span every request: a 30-span task rebuilt
# span1 ~30x (triangular 465 builds for 30 unique spans) -> made RedKnot-ON too slow to
# finish the sweep. Content-hash keying is RedKnot's intended design (see the paper's
# offline-segment cache). Bounded to cap memory.
_built_segments: "OrderedDict[str, None]" = OrderedDict()
_BUILT_SEG_CAP = 8192


def _segment_id(token_slice: List[int]) -> str:
    """Content-addressed id: identical span tokens -> identical id (cross-request reuse)."""
    h = hashlib.sha1()
    h.update(b",".join(str(t).encode() for t in token_slice))
    return "seg-%s" % h.hexdigest()[:20]


def build_redknot_segments(client, prepared, request_id) -> List[str]:
    """RedKnot: dense-prefill each PIC span independently and snapshot its KV as an
    offline segment (the __RKBUILD__ sentinel). Returns the ordered segment ids so the
    query request can name them for head-class sparse reuse. This IS RedKnot's
    'compute the span independently' step; the query then applies the sparse mask.

    Segments are content-hashed so an identical span builds ONCE and is re-spliced on every
    later request (RedKnot offline KV is position-independent), instead of rebuilding per
    request.
    """
    toks = prepared.token_ids
    seg_ids: List[str] = []
    prev = 0
    for i, ch in enumerate(prepared.chunks):
        if ch.get("kind") != "span":
            continue
        end = ch.get("end", 0)
        if end <= prev or end > len(toks):
            continue
        span_toks = toks[prev:end]
        sid = _segment_id(span_toks)
        if sid in _built_segments:
            seg_ids.append(sid)  # already built (same tokens seen before) -> re-splice, no rebuild
            _built_segments.move_to_end(sid)
            logger.info("Request %s: RedKnot reuse cached segment %s [%d:%d]",
                        request_id, sid, prev, end)
            prev = end
            continue
        try:
            client.build_segment(span_toks, sid)  # this span's tokens only
            seg_ids.append(sid)
            _built_segments[sid] = None
            while len(_built_segments) > _BUILT_SEG_CAP:
                _built_segments.popitem(last=False)
            logger.info("Request %s: RedKnot built segment %s [%d:%d]",
                        request_id, sid, prev, end)
        except Exception as e:
            logger.warning("Request %s: RedKnot build %s failed: %s",
                           request_id, sid, e)
        prev = end
    return seg_ids


def _now_iso() -> str:
    """UTC timestamp for trace records, e.g. 2026-06-09T09:36:12.481Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _git_sha() -> Optional[str]:
    """Best-effort short git sha for the run manifest (None outside a checkout)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _trace_chunks(prepared_prompt, warmup_result) -> List[dict]:
    """Shape the BuiltPrompt chunks (+ per-chunk warmup info) for the trace.
    None (prompt build failed) -> no chunks."""
    if prepared_prompt is None:
        return []
    cross = set(prepared_prompt.cross_span_starts)
    per_chunk = warmup_result.per_chunk if warmup_result else {}
    out = []
    for idx, c in enumerate(prepared_prompt.chunks):
        tc = dict(c)
        if c["kind"] == "span":
            tc["span_key"] = request_trace.span_key(
                prepared_prompt.token_ids[c["start"]:c["end"]])
            tc["is_cross_end"] = c["end"] in cross
        if idx in per_chunk:
            tc["warmup"] = per_chunk[idx]
        out.append(tc)
    return out


def coerce_tool_call_args(
    tool_calls: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Retype tool-call arguments that the backend's /parse returned as strings.

    WHY: vLLM's minimax_m2 tool parser stringifies EVERY argument value, so an argument the
    tool declares as an array comes back as `"view_range": "[150, 220]"` (a str holding a
    list). The agent's schema validation then rejects the call -- "Input should be a valid
    list" -- and the session dies after ~3 steps. On MiniMax this sank every middleware mode
    to ~15% solved while the two no-middleware modes sat at ~90%; it looked like a KV-cache
    result and was really a parser bug. The fork's own /v1/chat/completions is unaffected --
    only the /parse endpoint the middleware depends on.

    Coercion is driven by the DECLARED schema (we get `tools` on the request), never by
    guessing: only params the tool declares non-string are touched, and only when the string
    actually parses as that type. A genuine string argument is left exactly as it was.
    """
    if not tool_calls or not tools:
        return tool_calls

    NON_STRING = {"array", "object", "integer", "number", "boolean"}

    def declared_types(spec):
        """All json types a param may hold. Handles the flat `{"type": "array"}` form AND the
        Pydantic `Optional[...]` form exgentic emits: `{"anyOf": [{"type":"array",...},
        {"type":"null"}]}` -- there is NO top-level `type` there, which is exactly what made an
        earlier version of this skip `view_range` and leave it a string on MiniMax."""
        if not isinstance(spec, dict):
            return set()
        out = set()
        t = spec.get("type")
        if isinstance(t, str):
            out.add(t)
        elif isinstance(t, list):
            out.update(x for x in t if isinstance(x, str))
        for sub in (spec.get("anyOf") or spec.get("oneOf") or spec.get("allOf") or []):
            out |= declared_types(sub)
        return out

    # {tool_name: {param: set_of_declared_json_types}}
    schema: Dict[str, Dict[str, Any]] = {}
    for t in tools:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        if not name or not isinstance(props, dict):
            continue
        schema[name] = {p: declared_types(spec) for p, spec in props.items()}
    for tc in tool_calls:
        fn = (tc or {}).get("function") or {}
        want = schema.get(fn.get("name"))
        raw = fn.get("arguments")
        if not want or not isinstance(raw, str):
            continue
        try:
            args = json.loads(raw)
        except (ValueError, TypeError):
            continue
        if not isinstance(args, dict):
            continue
        changed = False
        for key, val in list(args.items()):
            types = want.get(key) or set()
            # coerce only if the param can be non-string and is NOT allowed to be a string
            # (a str|array union means the model may legitimately mean the string form).
            if not isinstance(val, str) or not (types & NON_STRING) or "string" in types:
                continue
            try:
                parsed_val = json.loads(val)
            except (ValueError, TypeError):
                continue  # not JSON after all -> the model really meant a string
            args[key] = parsed_val
            changed = True
        if changed:
            fn["arguments"] = json.dumps(args)
    return tool_calls


def build_response(
    request_id: str,
    full_text: str,
    finish_reason: Optional[str],
    usage_info: Optional[dict],
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Build the final response object.

    Args:
        request_id: Unique request identifier
        full_text: Generated text (post-parser content; may be None if all output was tool calls)
        finish_reason: Reason for completion (can be None)
        usage_info: Token usage dict
        reasoning_content: Reasoning span extracted from the raw output, if any
        tool_calls: Structured tool_calls list, if any

    Returns:
        Response dictionary in OpenAI format
    """
    message: Dict[str, Any] = {"role": "assistant", "content": full_text}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = tool_calls
        # OpenAI clients flip on finish_reason="tool_calls"; tools_called wins.
        finish_reason = "tool_calls"

    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "usage": usage_info,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
    }


def create_app(
    config_path: str | None = None,
) -> FastAPI:
    """Create FastAPI application with middleware."""

    # Load configuration - prefer env var, then argument, then default
    config_path = (
        config_path
        or os.environ.get("MIDDLEWARE_CONFIG_PATH")
        or "middleware_config.yaml"
    )
    config = MiddlewareConfig.from_yaml(config_path)

    # Apply env var overrides for backend config
    if "VLLM_ADDRESS" in os.environ:
        config.backend.base_url = os.environ["VLLM_ADDRESS"]
    if "MODEL_NAME" in os.environ:
        config.backend.model = os.environ["MODEL_NAME"]
        config.model.model_id = os.environ["MODEL_NAME"]
    if os.environ.get("BACKEND_KIND"):
        config.backend.kind = os.environ["BACKEND_KIND"]

    # Validate and sync model configuration
    if not config.backend.model and not config.model.model_id:
        logger.error(
            "Model not defined. Please set MODEL_NAME environment variable or "
            "configure backend.model or model.model_id in the config file."
        )
        raise ValueError(
            "Model not defined. Please set MODEL_NAME environment variable or "
            "configure backend.model or model.model_id in the config file."
        )
    elif config.backend.model and not config.model.model_id:
        config.model.model_id = config.backend.model
    elif config.model.model_id and not config.backend.model:
        config.backend.model = config.model.model_id

    # Pad token id used to align per-message chunks to block boundaries.
    if "VLLM_V1_SPANS_PAD_TOKEN" in os.environ:
        config.processing.padding.pad_token_id = int(
            os.environ["VLLM_V1_SPANS_PAD_TOKEN"]
        )

    if "MIDDLEWARE_WARMUP_ENABLED" in os.environ:
        config.processing.warmup.enabled = (
            os.environ["MIDDLEWARE_WARMUP_ENABLED"].lower() in ("1", "true", "yes")
        )
    if "MIDDLEWARE_WARMUP_CACHE_SIZE" in os.environ:
        config.processing.warmup.cache_size = int(
            os.environ["MIDDLEWARE_WARMUP_CACHE_SIZE"]
        )

    # Per-mode span-policy override; "full" sets "none" (no spans). See span_policy.py.
    if os.environ.get("MIDDLEWARE_SPAN_POLICY"):
        config.processing.span_policy = os.environ["MIDDLEWARE_SPAN_POLICY"]

    # Min span token length: a unit is a span only if longer ("atleast1000" sets 1000).
    if os.environ.get("MIDDLEWARE_MIN_SPAN_TOKENS"):
        config.processing.min_span_tokens = int(
            os.environ["MIDDLEWARE_MIN_SPAN_TOKENS"]
        )

    # SPAN_MODE selects the KV-reuse behavior. "naive" reuses span KV but sends NO
    # cross_span_starts - vLLM then stitches reused spans back in without the
    # in-context cross-correction, which is the naive (position-uncorrected) baseline
    # the spans/legolink work is measured against. spans/legolink keep cross markers.
    span_mode = os.environ.get("SPAN_MODE", "spans").lower()

    app = FastAPI(title="Middleware")

    # Create httpx client with SSL verification disabled (like curl -k)
    http_client = httpx.Client(verify=False)

    # Create OpenAI client with validated config and custom http client
    openai_client = OpenAI(
        base_url=config.backend.base_url,
        api_key=config.backend.api_key,
        timeout=config.backend.timeout,
        http_client=http_client,
    )

    # SGLang backends have none of the vLLM fork's token endpoints (/render, /parse,
    # /metrics). We inject an interface-compatible SglangClient (local HF tokenizer +
    # native /generate + native tool parser) so the SAME PIC pipeline tokenizes and
    # drives SGLang with real token_ids — no passthrough shortcut.
    is_sglang = config.backend.kind == "sglang"
    sglang_client = None
    if is_sglang:
        import sys
        sys.path.insert(0, "/workspace/sglang")  # committed shared dir (see repo)
        from sglang_client import SglangClient
        sglang_client = SglangClient(
            base_url=config.backend.base_url,
            model=config.backend.model or config.model.model_id,
            http_client=http_client,
        )

    # Initialize processor and logger (SGLang injects its client into the processor).
    processor = PromptProcessor(config, render_client=sglang_client)
    middleware_logger = MiddlewareLogger(config)

    # Per-worker LRU of already-warmed spans (skip re-warming; 0 disables).
    cache_size = config.processing.warmup.cache_size
    span_cache = warmup.SpanCache(cache_size) if cache_size > 0 else None

    # Policy that decides which messages are reusable PIC spans.
    policy = span_policy.get_span_policy(config.processing.span_policy)

    # Per-run trace directory (one per process; pid disambiguates workers) and
    # its manifest. Traces are gated on logging.responses.enabled; writing the
    # manifest is best-effort — logging must never take the service down.
    started = datetime.now(timezone.utc)
    run_id = request_trace.run_dir_name(started, os.getpid())
    run_dir = str(Path(config.logging.output_dir) / run_id)
    if config.logging.responses.enabled:
        try:
            request_trace.write_manifest(run_dir, {
                "run_id": run_id,
                "started_at": started.isoformat(timespec="seconds"),
                "pid": os.getpid(),
                "model": config.model.model_id,
                "backend_base_url": config.backend.base_url,
                "block_size": processor.block_size,
                "pad_token_id": config.processing.padding.pad_token_id,
                "span_policy": config.processing.span_policy,
                "warmup": {"enabled": config.processing.warmup.enabled,
                           "cache_size": config.processing.warmup.cache_size},
                "git_sha": _git_sha(),
            })
        except Exception as e:
            logger.warning("run manifest write failed (continuing): %s", e)

    # Store in app state
    app.state.config = config
    app.state.client = openai_client
    app.state.span_cache = span_cache
    app.state.span_policy = policy
    app.state.processor = processor
    app.state.logger = middleware_logger
    app.state.run_id = run_id
    app.state.run_dir = run_dir
    app.state.is_sglang = is_sglang
    app.state.sglang_client = sglang_client

    app.state.span_mode = span_mode

    @app.on_event("startup")
    async def startup_event():
        """Log startup information."""
        logger.info("Middleware started with config: %s", config_path)
        logger.info("Backend URL: %s", config.backend.base_url)
        logger.info("Model: %s", config.model.model_id)

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        return {
            "object": "list",
            "data": [
                {
                    "id": config.model.model_id,
                    "object": "model",
                }
            ],
        }

    @app.get("/v1/metrics")
    async def get_metrics():
        """Get performance metrics."""
        return app.state.logger.metrics.get_metrics()

    @app.delete("/v1/metrics")
    async def clear_metrics():
        """Clear performance metrics."""
        count = app.state.logger.metrics.clear_metrics()
        return {"message": f"Cleared {count} metrics"}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
        """Handle chat completion requests."""
        model = request.model
        # Pydantic -> dicts; exclude_none drops unset optional fields. The render
        # endpoint normalizes tool-call args / tool content itself, so no local pass.
        messages = [msg.model_dump(exclude_none=True) for msg in request.messages]
        max_tokens = request.max_tokens
        temperature = request.temperature
        seed = request.seed
        tools = request.tools
        # Anti-repetition sampling to forward to the backend /completions.
        sampling_top = {k: v for k, v in (("top_p", request.top_p),
            ("frequency_penalty", request.frequency_penalty),
            ("presence_penalty", request.presence_penalty)) if v is not None}
        sampling_xb = {k: v for k, v in (("top_k", request.top_k),
            ("min_p", request.min_p),
            ("repetition_penalty", request.repetition_penalty)) if v is not None}

        # QUEST: follow every tool-output run with a short review query so a
        # span never ends the prompt - the gap recompute always has a real
        # post-span query to score prefix blocks against. Deterministic per
        # conversation prefix, so prefix caching is unaffected.
        if app.state.span_mode.startswith("quest"):
            injected: list[dict] = []
            for i, m in enumerate(messages):
                injected.append(m)
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if m.get("role") == "tool" and (nxt is None or nxt.get("role") != "tool"):
                    injected.append(
                        {"role": "user", "content": "Review the tool output."}
                    )
            messages = injected

        # The active span policy decides which messages are reusable PIC spans.
        pic = app.state.span_policy.select_spans(messages)

        # Generate request ID
        request_id = app.state.logger.get_next_request_id()
        app.state.logger.log_request_start(request_id)

        t_request = time.perf_counter()
        render_ms = warmup_ms = final_ms = parse_ms = 0.0
        prepared_prompt = None
        warmup_result = None

        def write_request_trace(status, final, error=None, parse_calls=0):
            """Best-effort: assemble + write the per-request trace (never raises).

            Defined before any backend phase so failures in render/warmup/
            generate/parse all leave an error trace (with whatever was built
            by then), not just stream errors.
            """
            if not config.logging.responses.enabled:
                return
            try:
                warm_calls = warmup_result.calls if warmup_result else 0
                trace = request_trace.build_trace(
                    request_id=request_id,
                    run_id=app.state.run_id,
                    timestamp=_now_iso(),
                    status=status,
                    model=model,
                    params={"max_tokens": max_tokens, "temperature": temperature,
                            "seed": seed},
                    messages=messages,
                    tools=tools,
                    chunks=_trace_chunks(prepared_prompt, warmup_result),
                    final=final,
                    timing={
                        "render_ms": round(render_ms, 1),
                        "warmup_ms": round(warmup_ms, 1),
                        "final_round_trip_ms": round(final_ms, 1),
                        "parse_ms": round(parse_ms, 1),
                        "wall_clock_ms": round((time.perf_counter() - t_request) * 1000, 1),
                        "backend_calls": {
                            # actual render round trips (boundary cache skips
                            # prefixes verified on earlier turns)
                            "render": (prepared_prompt.render_calls
                                       if prepared_prompt else 0),
                            "warmup": warm_calls,
                            "final": 1,
                            "parse": parse_calls,
                            "total": ((prepared_prompt.render_calls
                                       if prepared_prompt else 0)
                                      + warm_calls + 1 + parse_calls),
                        },
                    },
                    error=error,
                )
                request_trace.write_trace(app.state.run_dir, request_id, trace)
            except Exception as te:
                logger.warning(
                    "Request %s: trace write failed (continuing): %s", request_id, te
                )

        logger.debug(
            "Request %s: Processing prompt with %d messages%s, PIC=%s",
            request_id,
            len(messages),
            f" and {len(tools)} tools" if tools else "",
            sorted(pic),
        )

        # Process prompt through pipeline (render endpoint tokenizes; middleware pads).
        t0 = time.perf_counter()
        try:
            prepared_prompt = app.state.processor.process_prompt(messages, tools=tools, pic=pic)
        except Exception as e:
            render_ms = (time.perf_counter() - t0) * 1000
            logger.error("Request %s: prompt build failed: %s", request_id, e)
            write_request_trace(
                "error", None,
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise
        render_ms = (time.perf_counter() - t0) * 1000
        full_prompt = prepared_prompt.token_ids
        span_starts = prepared_prompt.span_starts
        cross_span_starts = prepared_prompt.cross_span_starts

        logger.debug(
            "Request %s: Prompt processed, %d tokens, %d spans",
            request_id, len(full_prompt), len(span_starts),
        )

        # Cap the completion so prompt + output fits the backend context window.
        # Injected review turns (and span padding) are invisible to the agent's
        # own truncation math, so near-cap prompts would otherwise 400 with
        # "maximum context length is N tokens".
        ctx_limit = int(os.environ.get("BACKEND_MAX_MODEL_LEN", "92160"))
        if max_tokens is not None:
            max_tokens = max(16, min(max_tokens, ctx_limit - len(full_prompt)))

        # Use backend model if configured, otherwise use request model
        backend_model = config.backend.model if config.backend.model else model

        # Warmup: prefill each span before the real request so KV reuse is primed.
        span_reuse = []  # per-span reuse check, written to the trace (both vLLM + sglang)
        redknot_segments: List[str] = []  # built offline segment ids (RedKnot on)
        redknot_receipt: Optional[Dict[str, Any]] = None  # backend splice receipt
        # RedKnot-ON must NEVER silently degrade to a plain full prefill: a swallowed
        # build error or an empty span set would make the query fall through to plain
        # generate() and be recorded as a "RedKnot" result while head-class recompute
        # never ran. That is catastrophic (invalidates the whole comparison), so we fail
        # loud instead. redknot_on gates the strict path; n_span_chunks is how many
        # reusable PIC spans the policy actually selected for THIS request.
        redknot_on = app.state.is_sglang and os.environ.get("REDKNOT_ENABLED") == "1"
        n_span_chunks = sum(
            1 for ch in prepared_prompt.chunks if ch.get("kind") == "span"
        )
        if config.processing.warmup.enabled:
            t0 = time.perf_counter()
            try:
                if app.state.is_sglang and os.environ.get("QCFUSE_RATIO"):
                    # QCFuse blend computes each span independently DURING the single
                    # prefill (via <|blendsep|> injection in generate()) — no separate
                    # warmup call. Skip the warmup trip; the blend is on-the-fly.
                    pass
                elif redknot_on:
                    # RedKnot: build one offline segment per span (dense prefill +
                    # snapshot); the query then splices them with the head-class mask.
                    redknot_segments = build_redknot_segments(
                        app.state.sglang_client, prepared_prompt, request_id
                    )
                elif app.state.is_sglang:
                    # Prefill each span's token prefix through SGLang /generate so its KV
                    # is cached; the method (RedKnot head-reuse) then operates on warmed
                    # segments. Record cached_tokens per span as proof.
                    span_reuse = warm_sglang_spans(
                        app.state.sglang_client, prepared_prompt, request_id
                    )
                else:
                    warmup_result = warmup.warm_pic(
                        app.state.client, prepared_prompt, backend_model, temperature, seed,
                        cache=app.state.span_cache,
                    )
            except Exception as e:
                # A swallowed RedKnot build error is exactly the silent-failure mode we
                # refuse to allow: re-raise so the request 500s instead of quietly
                # serving a plain prefill counted as RedKnot.
                if redknot_on:
                    raise RuntimeError(
                        "RedKnot segment build FAILED for request %s (%d span(s)): %s "
                        "— no plain-prefill fallback under REDKNOT_ENABLED=1"
                        % (request_id, n_span_chunks, e)
                    ) from e
                logger.warning(
                    "Request %s: warmup failed (continuing): %s", request_id, e
                )
            warmup_ms = (time.perf_counter() - t0) * 1000

        # HARD INVARIANT (no fallback): under RedKnot-ON the head-class recompute MUST
        # run whenever there are reusable spans. If spans were selected but no segments
        # got built (warmup disabled, empty build, swallowed error), the query below
        # would silently do a plain full prefill with NO recompute yet still be tallied
        # as RedKnot. Refuse to serve it — a visible failure is correct; a fake
        # RedKnot number is not.
        if redknot_on and n_span_chunks > 0 and not redknot_segments:
            raise RuntimeError(
                "RedKnot recompute DID NOT RUN for request %s: %d PIC span(s) selected "
                "but 0 segments built → query would do a plain full prefill. No silent "
                "fallback under REDKNOT_ENABLED=1."
                % (request_id, n_span_chunks)
            )

        # Main inference with streaming
        try:
            # Build completion parameters. include_usage gives us
            # completion_tokens from the backend (no local tokenizer).
            completion_params = {
                "model": backend_model,
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
                **sampling_top,
            }
            # Add seed if provided
            if seed is not None:
                completion_params["seed"] = seed

            # Keep special tokens in the raw /completions text so /parse can see
            # gemma's <|tool_call> (and <|channel> reasoning) markers; otherwise
            # they're stripped and tool_calls come back empty.
            extra_body: dict = {"skip_special_tokens": False, **sampling_xb}
            # Pass block-aligned span starts (and cross/recompute points) to vLLM
            # — drives PIC span reuse and the cross-stitch between spans. In "naive"
            # mode we deliberately withhold cross_span_starts: spans are reused but
            # not cross-corrected, which is the naive baseline (vs spans/legolink).
            if span_starts:
                xargs = {"span_starts": span_starts}
                if cross_span_starts and app.state.span_mode != "naive":
                    xargs["cross_span_starts"] = cross_span_starts
                extra_body["vllm_xargs"] = xargs
            completion_params["extra_body"] = extra_body

            t0 = time.perf_counter()
            if app.state.is_sglang:
                # SGLang path: drive its native /generate with the pre-tokenized
                # PIC prompt (input_ids). Same token_ids the span policy built, so
                # KV reuse is on the real tokens — not a message-level proxy.
                # Only SGLang-supported SamplingParams keys (no seed/skip_special_tokens
                # here — those go elsewhere; unknown keys make SGLang 500).
                sp = {"max_new_tokens": max_tokens, "temperature": temperature}
                for k in ("top_p", "top_k", "min_p", "frequency_penalty",
                          "presence_penalty", "repetition_penalty"):
                    v = getattr(request, k, None)
                    if v is not None:
                        sp[k] = v
                if redknot_segments:
                    # RedKnot query: reuse the just-built offline segments (head-class
                    # sparse mask splices their KV) instead of a plain full prefill.
                    gen = app.state.sglang_client.generate_redknot(
                        full_prompt, sp, redknot_segments
                    )
                    # RECEIPT ENFORCEMENT (no fallback): the backend reports what it
                    # ACTUALLY spliced (meta_info.cached_tokens_details). Asked !=
                    # spliced means some span ranges got a dense prefill with NO
                    # reuse — refuse the response instead of tallying it as RedKnot.
                    redknot_receipt = gen.get("redknot_receipt")
                    if not redknot_receipt:
                        raise RuntimeError(
                            "RedKnot query for request %s returned NO splice receipt "
                            "(%d segment(s) requested) — backend predates receipts or "
                            "dropped the plan; cannot prove reuse, refusing."
                            % (request_id, len(redknot_segments))
                        )
                    if redknot_receipt.get("redknot_spliced") != len(redknot_segments):
                        raise RuntimeError(
                            "RedKnot spliced %s/%d segment(s) for request %s "
                            "(missing: %s) — partial reuse is a silent-degradation "
                            "path, refusing."
                            % (redknot_receipt.get("redknot_spliced"),
                               len(redknot_segments), request_id,
                               redknot_receipt.get("redknot_missing"))
                        )
                    logger.info(
                        "Request %s: RedKnot receipt: spliced %s/%d segment(s)",
                        request_id, redknot_receipt.get("redknot_spliced"),
                        len(redknot_segments),
                    )
                else:
                    # No segments here means: not RedKnot, or a genuine no-span request.
                    # Guard the silent-fallback door shut — the invariant above already
                    # forbids redknot_on + spans + no-segments, so reaching this branch
                    # under RedKnot with spans present is a bug that must never ship.
                    if redknot_on and n_span_chunks > 0:
                        raise RuntimeError(
                            "RedKnot plain-prefill fallback reached for request %s "
                            "with %d span(s) — must never run under REDKNOT_ENABLED=1"
                            % (request_id, n_span_chunks)
                        )
                    # Pass span boundaries so QCFuse (if enabled) splices <|blendsep|>
                    # and computes each span independently in ONE prefill (no warmup).
                    gen = app.state.sglang_client.generate(
                        full_prompt, sp, span_starts=span_starts
                    )
                full_text = gen["text"]
                finish_reason = gen["finish_reason"]
                completion_tokens = gen["completion_tokens"]
                # cached_tokens on the REAL request = KV reused from the warmed spans.
                cached_tokens = gen.get("cached_tokens")
            else:
                stream = app.state.client.completions.create(**completion_params)
                # Process stream
                full_text, finish_reason, completion_tokens, cached_tokens = (
                    process_stream(stream)
                )
            final_ms = (time.perf_counter() - t0) * 1000
            if cached_tokens is None:
                logger.info(
                    "Request %s: real call: prompt_tokens=%d, cached_tokens missing",
                    request_id, len(full_prompt),
                )
            else:
                logger.info(
                    "Request %s: real call: prompt_tokens=%d, cached_tokens=%d",
                    request_id, len(full_prompt), cached_tokens,
                )
        except Exception as e:
            logger.error(
                "Request %s: Error streaming to client: %s",
                request_id,
                str(e),
                exc_info=True,
            )
            # Trace the failed request too (no final chunk; error recorded).
            write_request_trace(
                "error", None,
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise

        # Self-heal: if the real request's cache hits are well below what warmup
        # primed, a span we trusted wasn't cached (eviction) — drop those keys
        # and re-warm so the next request is primed. No-op if the backend doesn't
        # report cached_tokens. vLLM-only (uses /completions warmup). Best-effort.
        if config.processing.warmup.enabled and not app.state.is_sglang:
            try:
                warmup.revalidate(
                    app.state.client, prepared_prompt, cached_tokens,
                    warmup_result, backend_model, temperature, seed,
                    cache=app.state.span_cache,
                )
            except Exception as e:
                logger.warning(
                    "Request %s: warmup revalidate failed (continuing): %s",
                    request_id, e,
                )

        # Postprocess the raw stream text via vLLM's /v1/chat/completions/parse
        # (the server's own reasoning + tool-call parsers). No local parsing.
        raw_text = full_text
        t0 = time.perf_counter()
        try:
            parsed = app.state.processor.render_client.parse_output(raw_text, tools=tools)
        except Exception as e:
            parse_ms = (time.perf_counter() - t0) * 1000
            logger.error("Request %s: parse failed: %s", request_id, e)
            write_request_trace(
                "error", None,
                error={"type": type(e).__name__, "message": str(e)},
            )
            raise
        parse_ms = (time.perf_counter() - t0) * 1000
        reasoning_content = parsed.get("reasoning_content")
        full_text = parsed.get("content")
        # If parsing yielded no content (e.g. truncated before </think> closed),
        # fall back to the raw stream so the message isn't empty.
        if full_text is None and reasoning_content is None:
            full_text = raw_text

        tool_calls_payload: Optional[List[Dict[str, Any]]] = None
        if parsed.get("tools_called") and parsed.get("tool_calls"):
            tool_calls_payload = coerce_tool_call_args(parsed["tool_calls"], tools)

        # Usage: prompt_tokens from the rendered token list, completion_tokens
        # from the backend's usage chunk (no local tokenizer).
        usage_info = calculate_usage_info(full_prompt, completion_tokens)

        # Metrics (the /v1/metrics endpoint); the per-request trace below
        # replaces the old response_req_*.json files.
        app.state.logger.metrics.log_request_complete(request_id, usage_info)

        # Per-request trace: the chunk-centric record (input -> chunks with
        # per-chunk warmup -> final + output -> timing). Best-effort.
        final_finish = "tool_calls" if tool_calls_payload else finish_reason
        # Reuse check (both backends): per-span cached_tokens from warmup + the
        # request-level cached_tokens. n_spans_reused / n_spans lets the results tell
        # at a glance whether span KV reuse actually happened for this request.
        n_reused = sum(1 for s in span_reuse if s.get("reused"))
        reuse_check = {
            "backend": config.backend.kind,
            "n_spans": len(span_reuse) or len(redknot_segments),
            "n_spans_reused": n_reused,
            "request_cached_tokens": cached_tokens,
            "spans": span_reuse,
        }
        if redknot_segments:
            # RedKnot proof: the backend RECEIPT of what was actually spliced
            # (enforced == requested above), not just what we asked for.
            reuse_check["method"] = "redknot"
            reuse_check["redknot_segments"] = redknot_segments
            reuse_check["redknot_receipt"] = redknot_receipt
            reuse_check["n_spans_reused"] = (
                redknot_receipt or {}
            ).get("redknot_spliced", 0)
        write_request_trace(
            "success",
            {
                "span_starts_sent": span_starts,
                "cross_span_starts_sent": cross_span_starts,
                "reuse_check": reuse_check,
                "round_trip_ms": round(final_ms, 1),
                "usage": {
                    "prompt_tokens": usage_info["prompt_tokens"],
                    "cached_tokens": cached_tokens,
                    "completion_tokens": usage_info["completion_tokens"],
                    "total_tokens": usage_info["total_tokens"],
                },
                "finish_reason": final_finish,
                "output": {
                    "content": full_text,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls_payload,
                },
            },
            parse_calls=1,
        )

        # Build and return response
        return build_response(
            request_id,
            full_text,
            finish_reason,
            usage_info,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls_payload,
        )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    host = app.state.config.server.host
    port = app.state.config.server.port

    logger.info("Starting server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port)
