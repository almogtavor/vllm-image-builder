"""Span warmup: prime PIC span reuse before the real request.

Separate from prompt construction (processors.py): the processor *builds* the
prompt via the render endpoint, whereas warmup is a generation-side cache
priming step using the OpenAI completions client.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from lru import LRUCache
from request_trace import fmt_key

logger = logging.getLogger(__name__)

# A real request whose prefix-cache hits fall below this fraction of what warmup
# primed is treated as "made no sense" (a span we trusted wasn't actually cached
# — eviction, or a broken warmup assumption) → invalidate + re-warm.
_REVALIDATE_MIN_RATIO = 0.5


@dataclass
class WarmupResult:
    """What a warmup pass primed, so the real request's cache hits can be
    sanity-checked — plus per-chunk info for the request trace."""
    primed_keys: List[int] = field(default_factory=list)   # span content keys believed cached
    expected_cached_tokens: int = 0                        # tokens the real request should hit
    per_chunk: Dict[int, dict] = field(default_factory=dict)  # chunk index -> trace warmup dict
    calls: int = 0                                         # backend warm calls actually made


class SpanCache(LRUCache):
    """Bounded LRU set of span keys this worker has already warmed.

    Optimistic and per-worker: a key being present means "we warmed this span
    and assume vLLM still has it cached". vLLM may have evicted it under memory
    pressure, in which case we skip a warmup that would have helped — never a
    correctness bug, only a missed optimization. Keys are content hashes of the
    (block-aligned) span tokens; because vLLM hashes spans prefix-independently,
    the same span content maps to the same key across requests.

    ``discard`` (so the next request re-warms a key) comes from
    :class:`LRUCache`.
    """

    def seen(self, key: int) -> bool:
        """Return True if key is cached, refreshing its recency (LRU touch)."""
        return self.get(key) is not None

    def add(self, key: int) -> Optional[int]:
        """Insert key; return the LRU key evicted to make room (or None)."""
        return self.put(key, True)


def _response_cached_tokens(resp) -> Optional[int]:
    """Prefix-cache hit count from a completion response's usage, or None.

    Reads ``usage.prompt_tokens_details.cached_tokens``; returns 0 when the
    backend reports details but no count, None when it reports nothing.
    """
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    if details is None:
        return None
    ct = getattr(details, "cached_tokens", None)
    return ct if ct is not None else 0


def _warm_one(client, tokens, span_starts, cross_span_starts, model,
              temperature, seed, cache, label) -> Optional[dict]:
    """Issue one max_tokens=1 prefill, with content-keyed dedup + logging.

    Returns trace info for the call — ``{key, decision, backend_call, cache_op,
    evicted_key}`` — or None for an empty span.
    """
    if not tokens:
        return None
    key = hash(tuple(tokens))  # content key; vLLM hashes spans prefix-independently
    if cache is not None and cache.seen(key):
        logger.info("warmup %s: tokens=%d, cached (skipped)", label, len(tokens))
        return {"key": key, "decision": "skipped_local_cache",
                "backend_call": None, "cache_op": None, "evicted_key": None}
    params = {"model": model, "prompt": tokens, "max_tokens": 1, "temperature": temperature}
    if seed is not None:
        params["seed"] = seed
    xargs = {}
    if span_starts:
        xargs["span_starts"] = span_starts
    if cross_span_starts:
        xargs["cross_span_starts"] = cross_span_starts
    if xargs:
        params["extra_body"] = {"vllm_xargs": xargs}
    t0 = time.perf_counter()
    resp = client.completions.create(**params)
    ms = (time.perf_counter() - t0) * 1000
    cached = _response_cached_tokens(resp)
    logger.info("warmup %s: tokens=%d, cached_tokens=%s",
                label, len(tokens), "missing" if cached is None else cached)
    evicted = cache.add(key) if cache is not None else None  # record only after a successful warm
    return {"key": key, "decision": "warmed",
            "backend_call": {
                "cached_tokens": cached,
                "real_compute_tokens": len(tokens) - cached if cached is not None else None,
                "round_trip_ms": round(ms, 1),
            },
            "cache_op": "added" if cache is not None else None,
            "evicted_key": fmt_key(evicted) if evicted is not None else None}


def warm_pic(
    client,
    built,
    model: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    cache: Optional[SpanCache] = None,
) -> WarmupResult:
    """Warm a request's spans before the real call, given a ``BuiltPrompt``.

    Two phases (cf. the PIC warmup design):

      1. **Each span independently** — prefill every PIC span's own tokens. Span
         blocks hash prefix-independently, so warming a span alone populates the
         exact cache entry it has inside the full prompt; reusable across requests.
      2. **Cumulative per in-context region** — for each non-span chunk that is
         followed by a span, prefill the whole prefix up to that next span start,
         carrying the spans/cross within it. The earlier spans come back as cache
         hits; only the new in-context (cross-stitch) blocks are real compute.
         Contiguous spans (nothing in-context between them) produce no cumulative
         pass — there's nothing new to warm.

    The real request is unaffected — it always sends the full span_starts /
    cross_span_starts. ``cache`` dedups by content so a growing conversation only
    warms genuinely-new spans/regions.

    Returns a :class:`WarmupResult` describing the spans primed (their keys and
    total tokens) so the real request's cache hits can be sanity-checked, plus
    ``per_chunk`` trace info (chunk index → what warmup did to it).
    """
    result = WarmupResult()
    if client is None:
        return result
    token_ids = built.token_ids
    chunks = built.chunks
    n_spans = sum(1 for c in chunks if c["kind"] == "span")
    if not n_spans:
        return result  # no spans → nothing to pre-seed (one in-context block)

    # Phase 1: each span alone (bare prompt; at offset 0 the first block is a
    # natural span-start, so its hash matches the in-context span-start block).
    k = 0
    for idx, c in enumerate(chunks):
        if c["kind"] != "span":
            continue
        k += 1
        toks = token_ids[c["start"]:c["end"]]
        info = _warm_one(client, toks, None, None, model,
                         temperature, seed, cache, f"span {k}/{n_spans}")
        if info is None:
            continue
        result.primed_keys.append(info.pop("key"))
        result.expected_cached_tokens += len(toks)
        if info["decision"] == "warmed":
            result.calls += 1
        result.per_chunk[idx] = {"kind": "per_pic", **info}

    # Phase 2: cumulative prefix for each non-span chunk that precedes a span.
    span_starts = built.span_starts
    cross = built.cross_span_starts
    for i, c in enumerate(chunks):
        if c["kind"] != "non" or i + 1 >= len(chunks) or chunks[i + 1]["kind"] != "span":
            continue
        s = chunks[i + 1]["start"]
        sub_spans = [p for p in span_starts if p < s]
        sub_cross = [p for p in cross if p < s]
        info = _warm_one(client, token_ids[:s], sub_spans, sub_cross,
                         model, temperature, seed, cache, f"cumulative→{s}")
        if info is None:
            continue
        info.pop("key")
        info.pop("evicted_key")  # cumulative prefixes aren't tracked as spans
        if info["decision"] == "warmed":
            result.calls += 1
        result.per_chunk[i] = {"kind": "cumulative", "label": f"cumulative→{s}",
                               "covers_chunks": list(range(i + 1)),
                               "xargs": {"span_starts": sub_spans,
                                         "cross_span_starts": sub_cross},
                               **info}
    return result


def revalidate(
    client,
    built,
    cached_tokens: Optional[int],
    result: Optional[WarmupResult],
    model: str,
    temperature: float = 0.0,
    seed: Optional[int] = None,
    cache: Optional[SpanCache] = None,
) -> bool:
    """Self-heal the warmup cache when the real request's hits don't add up.

    After the real request, ``cached_tokens`` is its prefix-cache reuse. If it's
    far below what warmup primed (``< _REVALIDATE_MIN_RATIO`` of
    ``expected_cached_tokens``), a span we trusted wasn't actually cached — vLLM
    evicted it, or the warmup assumption broke. We forget those span keys and
    re-warm immediately (the real request just re-populated vLLM, so the re-warm
    is mostly cache hits) so subsequent requests are primed again.

    No-op when ``cached_tokens`` is unavailable (backend doesn't report it) or
    nothing was primed. Returns True if a re-warm was triggered.
    """
    if cached_tokens is None or result is None or result.expected_cached_tokens <= 0:
        return False
    if cached_tokens >= result.expected_cached_tokens * _REVALIDATE_MIN_RATIO:
        return False  # hits made sense
    logger.warning(
        "warmup cache stale: cached_tokens=%d well below expected ~%d "
        "(span evicted or warmup ineffective); invalidating + re-warming",
        cached_tokens, result.expected_cached_tokens,
    )
    if cache is not None:
        for key in result.primed_keys:
            cache.discard(key)
    warm_pic(client, built, model, temperature, seed, cache=cache)
    return True
