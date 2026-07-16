"""Tests for the render-based prompt path.

The middleware builds the prompt via vLLM's render endpoint (token boundaries
from incremental render calls) and ONLY pads each per-message chunk so the next
chunk starts on a block boundary. No local tokenizer.
"""

from unittest.mock import Mock, patch

import pytest

from config_models import MiddlewareConfig


def _config() -> dict:
    return {
        "server": {"host": "0.0.0.0", "port": 8080},
        "backend": {"base_url": "http://localhost:8000/v1", "api_key": "vllm", "model": "test-model"},
        "model": {"model_id": "test-model", "assistant_placeholder": "<|assistant|>"},
        "processing": {
            "tokenization": {"enabled": True, "add_special_tokens": False},
            "padding": {"enabled": True, "pad_token_id": 60},
            "delimiter_splitting": {"enabled": True, "strategy": "per_message"},
            "span_mode": {"enabled": True, "mode": "spans", "plus_token_id": 43, "recompute_token_id": 64},
            "warmup": {"enabled": True},
        },
        "logging": {"output_dir": "/tmp", "metrics": {"enabled": False}, "responses": {"enabled": False}},
    }


@patch("processors.RenderClient")
def test_process_prompt_pads_per_message_chunks_to_block(mock_render_cls):
    from processors import PromptProcessor

    # 2 messages. block_size=16, pad_token_id=60.
    # full (agp=True) = 20 tokens; message 0 ends at offset 5 (prefix render).
    # => chunk0 = full[0:5] (5) -> padded to 16; chunk1 = full[5:20] (15, last, unpadded).
    full = list(range(100, 120))
    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        return full[:5]  # prefix for messages[:1]
    render_instance = Mock()
    render_instance.render_chat.side_effect = render_chat
    render_instance.fetch_block_size.return_value = 16  # block_size sourced from vLLM
    mock_render_cls.return_value = render_instance

    cfg = MiddlewareConfig(**_config())
    proc = PromptProcessor(cfg)

    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]
    built = proc.process_prompt(msgs, tools=tools)   # pic=None -> every message is a span
    out, span_starts = built.token_ids, built.span_starts

    # chunk0 (5 tokens) padded to 16, chunk1 (15) unpadded => 31
    assert len(out) == 31
    assert out[16:] == full[5:20]               # last chunk starts at block boundary 16
    assert out[:16].count(60) == 11             # pad tokens inserted into chunk0
    # span_starts: chunk0 at 0, chunk1 at 16 (block-aligned)
    assert span_starts == [0, 16]
    assert all(s % 16 == 0 for s in span_starts)
    assert built.cross_span_starts == []        # adjacent all-PIC spans -> no cross
    # render was given the tools
    assert render_instance.render_chat.call_args_list[0].kwargs.get("tools") == tools


@patch("processors.RenderClient")
def test_process_prompt_raises_without_engine_block_size(mock_render_cls):
    """block_size comes from vLLM; if /metrics can't provide it, fail fast."""
    from processors import PromptProcessor

    render_instance = Mock()
    render_instance.fetch_block_size.return_value = None
    mock_render_cls.return_value = render_instance

    with pytest.raises(ValueError, match="block_size"):
        PromptProcessor(MiddlewareConfig(**_config()))


def test_build_pic_prompt_all_spans_alignment():
    """pic=None -> every message is a span; each non-last chunk padded to block-align."""
    from processors import build_pic_prompt

    full = list(range(1000, 1040))  # 40 tokens
    # 3 messages; prefixes: msg0 ends @7, msg1 ends @18, last boundary = 40
    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        return {1: full[:7], 2: full[:18]}[len(messages)]
    rc = Mock(); rc.render_chat.side_effect = render_chat

    msgs = [{"role": "a", "content": "x"}, {"role": "b", "content": "y"}, {"role": "c", "content": "z"}]
    built = build_pic_prompt(rc, msgs, None, pad_token_id=60, block_size=16)
    out = built.token_ids

    # chunk0: 7 -> 16, chunk1: 11 -> 16, chunk2: 22 (last, unpadded) => 54
    assert len(out) == 54
    assert out[0:16] == full[0:6] + [60] * 9 + [full[6]]
    assert out[16:32] == full[7:17] + [60] * 5 + [full[17]]
    assert out[32:] == full[18:40]              # last chunk starts exactly at offset 32
    assert built.span_starts == [0, 16, 32]     # every message a span
    assert all(s % 16 == 0 for s in built.span_starts)
    assert built.cross_span_starts == []        # adjacent all-PIC spans -> dropped


def test_build_pic_prompt_interleaved():
    """pic={1} of 3 messages -> non | span | non (tail); cross at the span end."""
    from processors import build_pic_prompt

    full = list(range(2000, 2048))  # 48 tokens, 16 per message (already block-aligned)
    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        return {1: full[:16], 2: full[:32]}[len(messages)]
    rc = Mock(); rc.render_chat.side_effect = render_chat

    msgs = [{"role": "a", "content": "x"}, {"role": "b", "content": "y"}, {"role": "c", "content": "z"}]
    built = build_pic_prompt(rc, msgs, None, 60, 16, pic={1})

    kinds = [(c["kind"], c["start"], c["end"]) for c in built.chunks]
    assert kinds == [("non", 0, 16), ("span", 16, 32), ("non", 32, 48)]
    assert built.span_starts == [16]            # only m1
    assert built.cross_span_starts == [32]      # span end, where in-context content resumes
    assert built.token_ids == full              # no padding needed (already 16-aligned)


def test_build_pic_prompt_groups_adjacent_tool_messages():
    """An assistant tool_calls turn merges with its following tool-result run into
    ONE unit (the assistant->tool exchange is atomic; Qwen3 renders the assistant
    turn's trailing seam differently when a cut lands right after it), and adjacent
    tool messages share one container. So user, assistant, tool, tool -> [0] [1,4)."""
    from processors import build_pic_prompt

    # messages: user, assistant, tool, tool -> units [0] [1,4)
    # fake template: 8 tok/message, the two tool results share one container
    # (16 tok); rendering a cut INSIDE the run closes the container -> NOT a
    # prefix of full.
    full = list(range(3000, 3032))  # 32 tokens

    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        k = len(messages)
        if k == 1:
            return full[:16]        # user unit end (stable)
        return full[:20] + [9999]   # any cut past the user (assistant/tool) -> unstable

    rc = Mock(); rc.render_chat.side_effect = render_chat
    msgs = [{"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "tool_call_id": "1", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "content": "r2"}]
    built = build_pic_prompt(rc, msgs, None, pad_token_id=60, block_size=16)

    # pic=None -> every UNIT a span: [user] [assistant,tool,tool]
    kinds = [(c["kind"], c["start"], c["end"]) for c in built.chunks]
    assert kinds == [("span", 0, 16), ("span", 16, 32)]
    assert built.span_starts == [0, 16]
    assert built.token_ids[16:] == full[16:]    # assistant+tool run is one intact slice
    # neither the assistant cut (messages[:2]) nor the intra-tool cut (messages[:3])
    # may be requested -> only the user boundary (k=1) is probed
    prefix_lens = [len(c.args[0]) if c.args else len(c.kwargs["messages"])
                   for c in rc.render_chat.call_args_list
                   if not c.kwargs.get("add_generation_prompt", True)]
    assert 2 not in prefix_lens and 3 not in prefix_lens


def test_build_pic_prompt_mixed_pic_tool_run_stays_incontext():
    """A tool run is a span only if ALL its messages are PIC; a half-selected
    run falls back to in-context (never cache volatile content in a span)."""
    from processors import build_pic_prompt

    full = list(range(4000, 4032))  # u=8, a=8, tool-container=16

    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        return {1: full[:16]}[len(messages)]   # only the user boundary is probed

    rc = Mock(); rc.render_chat.side_effect = render_chat
    msgs = [{"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
            {"role": "tool", "tool_call_id": "1", "content": "r1"},
            {"role": "tool", "tool_call_id": "2", "content": "r2"}]

    # the assistant+tool run is ONE unit; a partial PIC selection can't span it -> non
    built = build_pic_prompt(rc, msgs, None, 60, 16, pic={2})
    assert [c["kind"] for c in built.chunks] == ["non"]
    assert built.span_starts == []

    # the whole assistant+tool unit selected -> it is one span
    rc.render_chat.side_effect = render_chat
    built = build_pic_prompt(rc, msgs, None, 60, 16, pic={1, 2, 3})
    assert [c["kind"] for c in built.chunks] == ["non", "span"]
    assert built.span_starts == [16]
    c = built.chunks[1]                                       # run intact at the tail
    assert (c["kind"], c["start"], c["end"]) == ("span", 16, 32)


def test_boundary_cache_reuses_verified_prefixes():
    """A growing conversation re-renders only its NEW units: cached boundary
    offsets are reused after a digest check against the new full render."""
    from processors import BoundaryCache, build_pic_prompt

    cache = BoundaryCache()
    full1 = list(range(0, 32))            # 3 messages, unit-0 ends at token 16
    full2 = list(range(0, 48))            # +2 messages; old tokens unchanged

    # A tool run trails the assistant so an assistant+tool turn is one unit (an
    # assistant merges with its following reply run — see _message_units).
    # msgs1 units = [(0,1),(1,3)] -> one boundary render at messages[:1].
    # msgs2 units = [(0,1),(1,3),(3,5)] -> boundaries at messages[:1] (cached) and
    # messages[:3] (new).
    msgs1 = [{"role": "user", "content": "a"},
             {"role": "assistant", "content": "b", "tool_calls": [{"id": "c1"}]},
             {"role": "tool", "content": "r1"}]
    msgs2 = msgs1 + [{"role": "assistant", "content": "d", "tool_calls": [{"id": "c2"}]},
                     {"role": "tool", "content": "r2"}]

    def rc1(messages, tools=None, add_generation_prompt=True):
        return full1 if add_generation_prompt else {1: full1[:16]}[len(messages)]

    def rc2(messages, tools=None, add_generation_prompt=True):
        return full2 if add_generation_prompt else {1: full2[:16], 3: full2[:32]}[len(messages)]

    r1 = Mock(); r1.render_chat.side_effect = rc1
    built1 = build_pic_prompt(r1, msgs1, None, 60, 16, boundary_cache=cache)
    assert built1.render_calls == 2                 # full + boundary@1 (cold)

    r2 = Mock(); r2.render_chat.side_effect = rc2
    built2 = build_pic_prompt(r2, msgs2, None, 60, 16, boundary_cache=cache)
    assert built2.render_calls == 2                 # full + boundary@3; @1 from cache
    requested = [len(c.kwargs.get("messages", c.args[0] if c.args else []))
                 for c in r2.render_chat.call_args_list
                 if not c.kwargs.get("add_generation_prompt", True)]
    assert requested == [3]                         # messages[:1] never re-rendered
    assert built2.span_starts == [0, 16, 32]        # boundaries identical to uncached


def test_boundary_cache_digest_mismatch_falls_back_to_render():
    """A stale cached offset (token digest no longer matches the full render)
    is ignored and the prefix is re-rendered + re-verified."""
    from processors import BoundaryCache, _tok_digest, build_pic_prompt

    cache = BoundaryCache()
    full = list(range(0, 32))

    def rc(messages, tools=None, add_generation_prompt=True):
        return full if add_generation_prompt else {1: full[:16]}[len(messages)]

    msgs = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    # poison: right key, wrong digest (as if the template changed)
    cache.put(BoundaryCache.key(msgs[:1], None), (16, _tok_digest([9, 9, 9])))

    r = Mock(); r.render_chat.side_effect = rc
    built = build_pic_prompt(r, msgs, None, 60, 16, boundary_cache=cache)
    assert built.render_calls == 2                  # fell back to a real render
    assert built.span_starts == [0, 16]
    # cache healed: rebuilt entry now hits
    r2 = Mock(); r2.render_chat.side_effect = rc
    built2 = build_pic_prompt(r2, msgs, None, 60, 16, boundary_cache=cache)
    assert built2.render_calls == 1                 # full render only


def test_build_pic_prompt_raises_on_prefix_drift():
    """If a prefix render isn't an exact prefix of the full render, FAIL FAST — a
    silent no-span fallback would let a request that should splice run with zero
    span reuse and look successful, masking a broken boundary. Spans must work;
    an unstable boundary is a bug to fix in _message_units, not to swallow."""
    from processors import build_pic_prompt

    full = list(range(1000, 1020))

    def render_chat(messages, tools=None, add_generation_prompt=True):
        if add_generation_prompt:
            return full
        return [9999, 9998]  # not a prefix of full

    rc = Mock(); rc.render_chat.side_effect = render_chat
    msgs = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    with pytest.raises(ValueError, match="prefix-stable"):
        build_pic_prompt(rc, msgs, None, pad_token_id=60, block_size=16)


def test_build_pic_prompt_empty():
    from processors import build_pic_prompt
    built = build_pic_prompt(Mock(), [], None, 60, 16)
    assert built.token_ids == [] and built.span_starts == [] and built.chunks == []


class TestRenderClient:
    def _http_returning(self, payload):
        http = Mock()
        resp = Mock()
        resp.json.return_value = payload
        resp.raise_for_status = Mock()
        http.post.return_value = resp
        return http

    def test_posts_to_render_endpoint_with_agp(self):
        from render_client import RenderClient

        http = self._http_returning({"token_ids": [1, 2, 3]})
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        out = rc.render_chat([{"role": "user", "content": "hi"}], tools=None)

        assert out == [1, 2, 3]
        args, kwargs = http.post.call_args
        assert args[0] == "http://localhost:8000/v1/chat/completions/render"
        assert kwargs["json"]["model"] == "m"
        assert kwargs["json"]["messages"] == [{"role": "user", "content": "hi"}]
        assert kwargs["json"]["add_generation_prompt"] is True   # default
        assert "tools" not in kwargs["json"]

    def test_add_generation_prompt_false_forwarded(self):
        from render_client import RenderClient

        http = self._http_returning({"token_ids": [1]})
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        rc.render_chat([{"role": "user", "content": "x"}], add_generation_prompt=False)
        assert http.post.call_args.kwargs["json"]["add_generation_prompt"] is False

    def test_includes_tools_when_present(self):
        from render_client import RenderClient

        http = self._http_returning({"token_ids": [9]})
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        tools = [{"type": "function", "function": {"name": "bash"}}]
        rc.render_chat([{"role": "user", "content": "x"}], tools=tools)
        assert http.post.call_args.kwargs["json"]["tools"] == tools

    def test_raises_on_empty_token_ids(self):
        from render_client import RenderClient

        http = self._http_returning({"token_ids": []})
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        with pytest.raises(ValueError, match="no token_ids"):
            rc.render_chat([{"role": "user", "content": "x"}])

    def test_parse_output_posts_to_parse_endpoint(self):
        from render_client import RenderClient

        payload = {
            "reasoning_content": "thinking",
            "content": "",
            "tool_calls": [{"function": {"name": "bash", "arguments": "{}"}}],
            "tools_called": True,
        }
        http = self._http_returning(payload)
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        out = rc.parse_output("raw text", tools=[{"type": "function", "function": {"name": "bash"}}])

        assert out == payload
        args, kwargs = http.post.call_args
        assert args[0] == "http://localhost:8000/v1/chat/completions/parse"
        assert kwargs["json"]["text"] == "raw text"
        assert kwargs["json"]["model"] == "m"
        assert kwargs["json"]["tools"][0]["function"]["name"] == "bash"

    def test_parse_output_omits_tools_when_none(self):
        from render_client import RenderClient

        http = self._http_returning({"content": "hi", "tools_called": False})
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        rc.parse_output("hi")
        assert "tools" not in http.post.call_args.kwargs["json"]

    def _http_get_returning(self, text):
        http = Mock()
        resp = Mock()
        resp.text = text
        resp.raise_for_status = Mock()
        http.get.return_value = resp
        return http

    def test_fetch_block_size_parses_metrics(self):
        from render_client import RenderClient

        metrics = (
            '# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n'
            'vllm:cache_config_info{block_size="16",cache_dtype="auto",'
            'enable_prefix_caching="True"} 1.0\n'
        )
        http = self._http_get_returning(metrics)
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        assert rc.fetch_block_size() == 16
        # /metrics is at the server root, not under /v1
        assert http.get.call_args.args[0] == "http://localhost:8000/metrics"

    def test_fetch_block_size_none_when_absent(self):
        from render_client import RenderClient

        http = self._http_get_returning("vllm:num_requests_running 0.0\n")
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        assert rc.fetch_block_size() is None

    def test_fetch_block_size_none_on_error(self):
        from render_client import RenderClient

        http = Mock()
        http.get.side_effect = Exception("connection refused")
        rc = RenderClient(base_url="http://localhost:8000/v1", model="m", http_client=http)
        assert rc.fetch_block_size() is None
