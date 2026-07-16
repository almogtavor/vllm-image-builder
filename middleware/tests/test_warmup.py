"""Tests for span warmup (warmup.py)."""

from unittest.mock import Mock

from warmup import warm_pic, revalidate, _response_cached_tokens, SpanCache
from processors import BuiltPrompt


def _client():
    c = Mock()
    c.completions.create.return_value = Mock(usage=Mock(prompt_tokens_details=None))
    return c


def _two_span_built():
    return BuiltPrompt(
        token_ids=list(range(64)),
        span_starts=[0, 32],
        cross_span_starts=[],
        chunks=[
            {"kind": "span", "start": 0, "end": 32},
            {"kind": "span", "start": 32, "end": 64},
        ],
    )


def test_warm_pic_independent_spans_then_cumulative():
    """Interleaved: warm each span alone, plus a cumulative pass per non-span
    chunk that precedes a span (here: the leading non-span before the span)."""
    client = _client()
    built = BuiltPrompt(
        token_ids=list(range(96)),
        span_starts=[32],
        cross_span_starts=[64],
        chunks=[
            {"kind": "non", "start": 0, "end": 32},
            {"kind": "span", "start": 32, "end": 64},
            {"kind": "non", "start": 64, "end": 96},   # trailing tail, no following span
        ],
    )
    warm_pic(client, built, model="m")

    prompts = [c.kwargs["prompt"] for c in client.completions.create.call_args_list]
    assert prompts == [list(range(32, 64)),    # phase 1: the span alone
                       list(range(0, 32))]     # phase 2: cumulative up to the span start
    assert all(c.kwargs["max_tokens"] == 1 for c in client.completions.create.call_args_list)


def test_warm_pic_contiguous_spans_no_cumulative():
    """Back-to-back spans (nothing in-context between them) → no cumulative pass."""
    client = _client()
    built = BuiltPrompt(
        token_ids=list(range(48)),
        span_starts=[0, 16],
        cross_span_starts=[],
        chunks=[
            {"kind": "span", "start": 0, "end": 16},
            {"kind": "span", "start": 16, "end": 32},
            {"kind": "non", "start": 32, "end": 48},   # trailing tail
        ],
    )
    warm_pic(client, built, model="m")

    prompts = [c.kwargs["prompt"] for c in client.completions.create.call_args_list]
    assert prompts == [list(range(0, 16)), list(range(16, 32))]  # two spans, no cumulative


def test_warm_pic_dedup_across_requests():
    client = _client()
    cache = SpanCache(maxsize=64)
    built = BuiltPrompt(
        token_ids=list(range(64)),
        span_starts=[32],
        cross_span_starts=[],
        chunks=[
            {"kind": "non", "start": 0, "end": 32},
            {"kind": "span", "start": 32, "end": 64},   # last chunk = span (tail)
        ],
    )
    warm_pic(client, built, model="m", cache=cache)
    assert client.completions.create.call_count == 2   # span + leading-non cumulative

    client.completions.create.reset_mock()
    warm_pic(client, built, model="m", cache=cache)     # identical → all cached
    client.completions.create.assert_not_called()


def test_warm_pic_noop_without_client_or_spans():
    built = BuiltPrompt(token_ids=list(range(16)), span_starts=[], cross_span_starts=[],
                        chunks=[{"kind": "non", "start": 0, "end": 16}])
    warm_pic(None, built, model="m")           # no client → no raise
    c = _client()
    warm_pic(c, built, model="m")              # no spans → nothing to warm
    c.completions.create.assert_not_called()


def test_warm_pic_returns_result_with_expected_tokens():
    client = _client()
    result = warm_pic(client, _two_span_built(), model="m")
    assert result.expected_cached_tokens == 64    # 32 + 32 span tokens
    assert len(result.primed_keys) == 2           # one key per span


def test_span_cache_lru():
    c = SpanCache(maxsize=2)
    assert not c.seen(1)
    c.add(1); c.add(2)            # order: [1, 2]
    assert c.seen(1)              # touch 1 -> [2, 1]; 2 is now LRU
    c.add(3)                      # [2, 1, 3] over capacity -> evict LRU (2) -> [1, 3]
    assert c.seen(1) and c.seen(3)
    assert not c.seen(2)          # evicted
    assert len(c) == 2


def test_span_cache_discard():
    c = SpanCache(maxsize=8)
    c.add(1); c.add(2)
    c.discard(1)
    assert not c.seen(1) and c.seen(2)
    c.discard(999)  # absent key -> no-op


def test_revalidate_noop_when_missing_or_sufficient():
    client = _client()
    built = _two_span_built()
    result = warm_pic(client, built, model="m")
    client.completions.create.reset_mock()

    assert revalidate(client, built, None, result, model="m") is False   # missing
    assert revalidate(client, built, 40, result, model="m") is False     # >= 32 (50% of 64)
    client.completions.create.assert_not_called()


def test_revalidate_rewarms_when_nonsensical():
    client = _client()
    cache = SpanCache(maxsize=64)
    built = _two_span_built()
    result = warm_pic(client, built, model="m", cache=cache)   # primes 2 spans
    assert len(cache) == 2
    client.completions.create.reset_mock()

    triggered = revalidate(client, built, 0, result, model="m", cache=cache)
    assert triggered is True
    assert client.completions.create.call_count == 2   # re-warmed both spans
    assert len(cache) == 2


def test_response_cached_tokens():
    assert _response_cached_tokens(Mock(usage=Mock(prompt_tokens_details=None))) is None
    assert _response_cached_tokens(Mock(usage=None)) is None
    hit = Mock(usage=Mock(prompt_tokens_details=Mock(cached_tokens=48)))
    assert _response_cached_tokens(hit) == 48
    no_count = Mock(usage=Mock(prompt_tokens_details=Mock(cached_tokens=None)))
    assert _response_cached_tokens(no_count) == 0
