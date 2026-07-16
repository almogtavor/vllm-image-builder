"""
Render-based prompt processor.

vLLM's render endpoint (`/v1/chat/completions/render`) owns tokenization and
chat-template rendering. The middleware's ONLY token work is padding: chunks are
padded so each span starts on a KV block boundary. No local tokenizer is loaded.

A request may declare which messages are PIC (reusable spans) — see
``build_pic_prompt``. When none are declared, every message is its own span
(backward-compatible with the per-message behavior).
"""
import hashlib
import json
import logging as _lg
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import httpx

from config_models import MiddlewareConfig
from lru import LRUCache
from render_client import RenderClient


@dataclass
class BuiltPrompt:
    """Result of building a prompt for the backend.

    - ``token_ids``: the padded prompt sent to vLLM.
    - ``span_starts``: block-aligned start of each reusable (PIC) span.
    - ``cross_span_starts``: block-aligned end of each span where in-context
      (non-span) content resumes — the recompute/stitch points.
    - ``chunks``: ordered layout, each ``{"kind": "span"|"non", "start", "end",
      "source_messages", "content_tokens"}`` — drives warmup and the request trace
      (content_tokens is the unpadded length; end-start includes padding).
    - ``render_calls``: render-endpoint round trips this build actually made.
    """
    token_ids: List[int] = field(default_factory=list)
    span_starts: List[int] = field(default_factory=list)
    cross_span_starts: List[int] = field(default_factory=list)
    chunks: List[dict] = field(default_factory=list)
    render_calls: int = 0


def _digest(data: str) -> str:
    return hashlib.sha1(data.encode()).hexdigest()


def _tok_digest(tokens: List[int]) -> str:
    return _digest(",".join(map(str, tokens)))


class BoundaryCache(LRUCache):
    """Bounded LRU of unit-boundary offsets, keyed by message-prefix content.

    Prefix boundaries are immutable: for an identical ``messages[:b]`` (+tools),
    the rendered prefix — and therefore its end offset in any full render it is
    a prefix of — never changes. That is exactly the prefix-stability invariant
    the builder already enforces, so boundaries computed on earlier turns of a
    growing conversation can be reused instead of re-rendered (the dominant
    middleware cost: one render round trip per unit per request, each rendering
    the whole prefix server-side).

    Entries store ``(offset, token_digest)``; on reuse the digest is checked
    against the *current* full render's slice, so a hit is only honored when
    ``full[:offset]`` is byte-identical to the previously verified prefix — a
    mismatch (template/tokenizer change, collision) falls back to a real render.
    """

    def __init__(self, maxsize: int = 4096):
        super().__init__(maxsize)

    @staticmethod
    def key(messages: List[dict], tools: Optional[List[dict]]) -> str:
        return _digest(json.dumps([messages, tools], sort_keys=True, ensure_ascii=False))


def pad_tokens(tokens: List[int], pad_token_id: int, block_size: int) -> List[int]:
    """Pad tokens to a block_size multiple by inserting padding before the last token."""
    if not tokens or len(tokens) % block_size == 0:
        return tokens
    pad_len = block_size - len(tokens) % block_size
    return tokens[:-1] + [pad_token_id] * pad_len + tokens[-1:]


_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def _is_tool_result(msg: dict) -> bool:
    """True if a message is a tool result — either ``role: tool`` or a
    ``role: user`` message wrapping ``<tool_response>…</tool_response>`` (the
    Qwen3 template renders both identically, as a user tool_response block).
    Tool results after an assistant tool-call turn form one unit with it."""
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "user":
        c = msg.get("content")
        return isinstance(c, str) and c.lstrip().startswith("<tool_response>")
    return False


def _is_call_reply(msg: dict) -> bool:
    """True if a message is the reply to a preceding assistant tool-call — a
    tool result (see :func:`_is_tool_result`) OR a bare ``role: user`` message.

    The agent can return a tool call's outcome as an unwrapped ``role: user``
    turn (e.g. a dispatch error ``"Error: Sending ..."`` with no
    ``<tool_response>`` wrapper). Positionally it is still the assistant turn's
    reply, and Qwen3 renders it as the turn's continuation, so it must merge into
    the same unit — otherwise the cut lands after a bare assistant tool-call turn,
    which the template renders differently as last-vs-middle → prefix drift.

    Used ONLY to extend the assistant+tool-call merge, never for the standalone
    tool-run coalesce, so a leading real user query is never mis-merged."""
    return _is_tool_result(msg) or msg.get("role") == "user"


def _strip_reasoning(messages: List[dict]) -> List[dict]:
    """Strip ``<think>…</think>`` reasoning from assistant history so incremental
    prefix renders match the full render.

    Qwen3's chat template keeps an assistant turn's ``<think>`` block only when the
    turn is AFTER the last real-user query (``loop.index0 > last_query_index``) and
    strips it otherwise — a CONTEXT-DEPENDENT decision. So the same assistant turn
    renders with its think block in a short prefix (where it is the last turn) but
    without it in the full render (where a later user/tool turn follows) → the
    prefix is not a token-prefix of the full render → ``_unit_boundaries`` raises
    "render is not prefix-stable". We remove the think blocks up front so every
    render (prefix and full) is identical regardless of what follows. The backend's
    own template strips them the same way, so generation is unaffected; the final
    (in-flight) assistant turn has no persisted think to strip.
    """
    out: List[dict] = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "assistant" and isinstance(c, str) and ("<think>" in c or "</think>" in c):
            c = _THINK_RE.sub("", c)  # strip closed <think>…</think> blocks
            if "</think>" in c:  # bare trailing closer with no opener
                c = c.split("</think>")[-1].lstrip("\n")
            if "<think>" in c:
                # unclosed opener (generation cut off mid-think, e.g. a dispatch
                # error truncated the turn) — drop from the opener to end so the
                # render can't depend on a partial think block.
                c = c.split("<think>")[0].rstrip("\n")
            m = {**m, "content": c}
        out.append(m)
    return out


def _message_units(messages: List[dict]) -> List[Tuple[int, int]]:
    """Split messages into boundary units: each message is its own unit, except
    a maximal run of consecutive ``tool`` messages, which is one unit.

    Chat templates (e.g. MiniMax) render adjacent tool results inside one
    shared container block — rendering a conversation cut between two tool
    messages closes the container, so that cut is structurally never a token
    prefix of the full render. Units are exactly the positions where an
    incremental render is prefix-stable.
    """
    units: List[Tuple[int, int]] = []
    i, n = 0, len(messages)
    while i < n:
        j = i + 1
        if _is_tool_result(messages[i]):
            while j < n and _is_tool_result(messages[j]):
                j += 1
        elif messages[i].get("role") == "assistant" \
                and j < n and _is_call_reply(messages[j]):
            # Qwen3 renders an assistant turn differently when it is the LAST
            # message vs. followed by another turn — think-block retention depends
            # on `loop.index0 > last_query_index`, and the trailing im_end/seam
            # shifts — so a cut right after ANY assistant turn is not prefix-stable.
            # Merge the assistant turn with its following reply run (tool results
            # and/or the bare-user dispatch errors the agent injects) into one unit,
            # whether or not we could parse a tool call out of the assistant turn
            # (an error can truncate the turn before its <tool_call> is emitted).
            # The reply may be `role: tool`, a wrapped-user `<tool_response>`, or a
            # bare `role: user` — all continue the same unit.
            while j < n and _is_call_reply(messages[j]):
                j += 1
        units.append((i, j))
        i = j
    return units


def _unit_boundaries(
    render_client: RenderClient,
    messages: List[dict],
    tools: Optional[List[dict]],
    units: List[Tuple[int, int]],
    cache: Optional[BoundaryCache] = None,
) -> Tuple[List[int], List[int], int]:
    """Render the full prompt and find each unit's end offset within it.

    Boundaries come from incremental render calls (``add_generation_prompt=False``)
    at unit ends only — a cut inside a unit (between adjacent tool messages) is
    never requested because the template makes it non-prefix-stable. Each prefix
    must be an exact slice of the full render (no BPE seam drift) — verified
    here, fail fast otherwise.

    With a :class:`BoundaryCache`, boundaries already verified on earlier turns
    of the conversation are reused (digest-checked against this request's full
    render) instead of re-rendered, so a growing conversation costs one full
    render plus prefix renders for the new units only.

    Returns ``(full, unit_end, render_calls)`` where ``unit_end[k]`` is the end
    offset of unit k in the unpadded full render (``unit_end[-1] == len(full)``,
    carrying the gen prompt) and ``render_calls`` counts actual round trips.
    """
    full = render_client.render_chat(messages, tools=tools, add_generation_prompt=True)
    calls = 1

    unit_end: List[int] = []
    for _, b in units[:-1]:
        key = BoundaryCache.key(messages[:b], tools) if cache is not None else None
        if cache is not None:
            hit = cache.get(key)
            if hit is not None:
                offset, tdig = hit
                if offset <= len(full) and _tok_digest(full[:offset]) == tdig:
                    unit_end.append(offset)
                    continue
                # digest mismatch (template change / stale entry) -> real render
        prefix = render_client.render_chat(
            messages[:b], tools=tools, add_generation_prompt=False
        )
        calls += 1
        if full[: len(prefix)] != prefix:
            # Log the message shapes at the failing boundary so the exact template
            # case can be identified and fixed in _message_units (never swallow it).
            shape = [
                (m.get("role"),
                 bool(m.get("tool_calls")),
                 ("<tool_call>" in m["content"]) if isinstance(m.get("content"), str) else False,
                 (m["content"].lstrip()[:14]) if isinstance(m.get("content"), str) else None)
                for m in messages[:min(b + 1, len(messages))]
            ]
            _lg.getLogger("middleware").error(
                "prefix-drift after msg %d; shapes (role, has_tool_calls, has_<tool_call>, head)=%s",
                b - 1, shape)
            raise ValueError(
                f"render is not prefix-stable after message {b - 1}: "
                f"render(messages[:{b}]) is not a prefix of the full render"
            )
        if cache is not None:
            cache.put(key, (len(prefix), _tok_digest(prefix)))
        unit_end.append(len(prefix))
    unit_end.append(len(full))

    if any(e > ne for e, ne in zip(unit_end, unit_end[1:])):
        raise ValueError(f"unit render boundaries are not monotonic: {unit_end}")
    return full, unit_end, calls


def build_pic_prompt(
    render_client: RenderClient,
    messages: List[dict],
    tools: Optional[List[dict]],
    pad_token_id: int,
    block_size: int,
    pic: Optional[set] = None,
    boundary_cache: Optional[BoundaryCache] = None,
    min_span_tokens: int = 0,
) -> BuiltPrompt:
    """Build the prompt, treating ``pic`` message indices as reusable spans.

    Messages are first split into boundary units (see :func:`_message_units`):
    each message alone, except a run of consecutive tool messages, which the
    template renders as one container block and therefore must stay one unit.
    A unit is a span only if ALL its messages are PIC — a mixed tool run stays
    in-context rather than caching volatile content inside a reusable span.
    Maximal runs of non-span units are in-context (non-span) chunks. Chunks
    are padded to a block multiple at every boundary except the last (the
    tail), so every ``span_start`` and ``cross_span_start`` lands on a KV
    block boundary.

    ``pic=None`` ⇒ every unit is a span (per-message behavior, with adjacent
    tool messages forming one span). ``pic=set()`` ⇒ no spans (the whole
    prompt is one in-context chunk).

    Returns a :class:`BuiltPrompt`.
    """
    if not messages:
        return BuiltPrompt()

    # Strip historical <think> reasoning first so prefix and full renders agree
    # (Qwen3 keeps think only for post-last-query turns -> context-dependent drift).
    messages = _strip_reasoning(messages)

    n = len(messages)
    pic_set = set(range(n)) if pic is None else {i for i in pic if 0 <= i < n}

    units = _message_units(messages)
    full, unit_end, render_calls = _unit_boundaries(
        render_client, messages, tools, units, cache=boundary_cache)
    unit_start = [0] + unit_end[:-1]
    # A unit is a span when every SPAN-ELIGIBLE message in it is PIC (and >=1 is),
    # plus the min_span_tokens length gate. Span-eligible = a message the policy can
    # select (a tool result). _message_units merges the leading assistant+tool_calls
    # message into the unit for render-stability; that assistant is a carrier, never
    # PIC, so requiring ALL messages to be PIC makes every assistant+tool unit
    # non-span -> 0 segments built (the RedKnot-ON bug). Require instead: no eligible
    # message is left unselected, and at least one eligible message is selected.
    def _carrier(m):
        return messages[m].get("role") == "assistant"
    unit_is_span = [
        any(m in pic_set for m in range(a, b))
        and all((m in pic_set) or _carrier(m) for m in range(a, b))
        and (unit_end[i] - unit_start[i] > min_span_tokens)
        for i, (a, b) in enumerate(units)
    ]

    # Group units into chunks: each span unit is its own span chunk; a maximal
    # run of non-span units is one non-span chunk.
    groups: List[Tuple[str, int, int]] = []  # (kind, first_unit, last_unit_exclusive)
    u = 0
    while u < len(units):
        if unit_is_span[u]:
            groups.append(("span", u, u + 1))
            u += 1
        else:
            v = u
            while v < len(units) and not unit_is_span[v]:
                v += 1
            groups.append(("non", u, v))
            u = v

    out: List[int] = []
    chunks: List[dict] = []
    for gi, (kind, ua, ub) in enumerate(groups):
        seg = full[unit_start[ua]:unit_end[ub - 1]]  # raw tokens for units [ua, ub)
        raw_len = len(seg)
        start = len(out)
        if gi != len(groups) - 1:  # pad every chunk except the last (tail) to align the next
            seg = pad_tokens(seg, pad_token_id, block_size)
        out.extend(seg)
        chunks.append({"kind": kind, "start": start, "end": len(out),
                       "source_messages": list(range(units[ua][0], units[ub - 1][1])),
                       "content_tokens": raw_len})

    span_starts = [c["start"] for c in chunks if c["kind"] == "span"]
    span_start_set = set(span_starts)
    # cross at the end of each span chunk where in-context content resumes; skip
    # the last chunk (no following content) and skip positions that are also a
    # span start (adjacent PIC spans — a block can't be both fan-in and recompute).
    cross_span_starts = [
        c["end"]
        for gi, c in enumerate(chunks)
        if c["kind"] == "span"
        and gi != len(chunks) - 1
        and c["end"] not in span_start_set
    ]
    return BuiltPrompt(out, span_starts, cross_span_starts, chunks, render_calls)


class PromptProcessor:
    """Render-only prompt processor: render endpoint tokenizes, middleware pads."""

    def __init__(self, config: MiddlewareConfig, render_client=None):
        # Construction-only: render endpoint tokenizes, middleware pads. Warmup
        # lives in warmup.py (it's a generation-side concern, not construction).
        self.config = config
        proc_cfg = config.processing
        self.pad_token_id = proc_cfg.padding.pad_token_id
        # Per-worker reuse of already-verified unit boundaries: a growing
        # conversation re-renders only its new units, not every prefix.
        self.boundary_cache = BoundaryCache()

        # render_client is injectable so SGLang backends can pass an interface-
        # compatible SglangClient (sglang/sglang_client.py) — same render_chat /
        # fetch_block_size / parse_output surface — without changing build_pic_prompt.
        self.render_client = render_client or RenderClient(
            base_url=config.backend.base_url,
            model=config.backend.model or config.model.model_id,
            http_client=httpx.Client(verify=False),
        )

        # block_size comes from vLLM's resolved KV config (/metrics), not config:
        # padding must align to the real block size or PIC span_starts fail
        # vLLM's alignment check, so fail fast rather than guess.
        self.block_size = self.render_client.fetch_block_size()
        if not self.block_size:
            raise ValueError(
                "Could not read block_size from vLLM /metrics "
                f"(vllm:cache_config_info); is {config.backend.base_url} reachable?"
            )

    def process_prompt(
        self,
        messages: List[dict],
        tools: Optional[List[dict]] = None,
        pic: Optional[set] = None,
    ) -> BuiltPrompt:
        """Render the prompt via vLLM and pad PIC spans to block alignment.

        ``pic`` is the set of message indices the client declared as reusable
        spans (None ⇒ every message is a span). Returns a :class:`BuiltPrompt`.
        """
        return build_pic_prompt(
            self.render_client,
            messages,
            tools,
            self.pad_token_id,
            self.block_size,
            pic,
            boundary_cache=self.boundary_cache,
            min_span_tokens=self.config.processing.min_span_tokens,
        )
