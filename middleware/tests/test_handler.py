"""Offline tests for the /v1/chat/completions handler and middleware.py helpers.

The whole vLLM backend is mocked (render token_ids, the generate stream, and
/parse), so these run in CI with no GPU and no network — they exercise the glue
that wires render -> pad -> warmup -> generate -> parse -> response together.
"""
from unittest.mock import Mock, patch

import pytest
import yaml
from fastapi.testclient import TestClient


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _write_config(tmp_path, warmup=False, responses=False):
    cfg = {
        "server": {"host": "0.0.0.0", "port": 8080},
        "backend": {"base_url": "http://vllm/v1", "api_key": "k", "timeout": 30,
                    "model": "test-model"},
        "model": {"model_id": "test-model"},
        "processing": {"padding": {"pad_token_id": 60}, "warmup": {"enabled": warmup}},
        "logging": {
            "output_dir": str(tmp_path),
            "metrics": {"enabled": False},
            "responses": {"enabled": responses},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _chunk(text=None, finish=None, usage=None):
    c = Mock()
    if usage is not None:
        c.usage = usage
        c.choices = []
    else:
        c.usage = None
        choice = Mock()
        choice.text = text
        choice.finish_reason = finish
        c.choices = [choice]
    return c


def _fake_stream(text="Hello", finish="stop", completion_tokens=2, cached=4096):
    # cached defaults high so warmup tests don't trip the revalidate self-heal;
    # revalidate-specific tests pass a low cached explicitly.
    usage = Mock(completion_tokens=completion_tokens,
                 prompt_tokens_details=Mock(cached_tokens=cached))
    return iter([_chunk(text=text), _chunk(finish=finish), _chunk(usage=usage)])


def _render_chat(messages, tools=None, add_generation_prompt=True):
    """Prefix-stable fake render: full = range(40); each prefix is range(0, 18*k)."""
    if add_generation_prompt:
        return list(range(40))
    return list(range(18 * len(messages)))  # exact prefix of range(40)


def _build_app(tmp_path, *, warmup, parse_result, mock_render_cls, mock_openai,
               stream=None, responses=False):
    render_inst = Mock()
    render_inst.fetch_block_size.return_value = 16
    render_inst.render_chat.side_effect = _render_chat
    render_inst.parse_output.return_value = parse_result
    mock_render_cls.return_value = render_inst

    openai_inst = Mock()

    def create_side_effect(**kwargs):
        if kwargs.get("stream"):
            return stream if stream is not None else _fake_stream()
        # warmup call (non-stream, max_tokens=1)
        return Mock(usage=Mock(prompt_tokens_details=Mock(cached_tokens=0)))

    openai_inst.completions.create.side_effect = create_side_effect
    mock_openai.return_value = openai_inst

    from middleware import create_app
    app = create_app(config_path=_write_config(tmp_path, warmup=warmup,
                                               responses=responses))
    return app, openai_inst, render_inst


# --------------------------------------------------------------------------- #
# handler integration
# --------------------------------------------------------------------------- #
@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_plain_completion_happy_path(mock_render_cls, mock_openai, tmp_path):
    parsed = {"reasoning_content": None, "content": "Hello",
              "tool_calls": None, "tools_called": False}
    app, openai_inst, _ = _build_app(
        tmp_path, warmup=False, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
    )
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    })
    assert r.status_code == 200
    body = r.json()
    msg = body["choices"][0]["message"]
    assert msg["content"] == "Hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    # usage: prompt_tokens = rendered length (40), completion_tokens from stream
    assert body["usage"]["prompt_tokens"] == 40
    assert body["usage"]["completion_tokens"] == 2
    # the real generate call carried the prompt token ids
    real = [c for c in openai_inst.completions.create.call_args_list if c.kwargs.get("stream")]
    assert real and real[0].kwargs["prompt"] == list(range(40))


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_span_starts_forwarded(mock_render_cls, mock_openai, tmp_path):
    parsed = {"content": "ok", "reasoning_content": None, "tool_calls": None, "tools_called": False}
    app, openai_inst, _ = _build_app(
        tmp_path, warmup=False, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
    )
    client = TestClient(app)
    # two messages -> span_starts = [0, 32]; must reach vllm_xargs
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    })
    assert r.status_code == 200
    real = [c for c in openai_inst.completions.create.call_args_list if c.kwargs.get("stream")][0]
    assert real.kwargs["extra_body"] == {"vllm_xargs": {"span_starts": [0, 32]}}


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_tool_call_response(mock_render_cls, mock_openai, tmp_path):
    tool_calls = [{"id": "x", "type": "function",
                   "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]
    parsed = {"reasoning_content": "thinking", "content": None,
              "tool_calls": tool_calls, "tools_called": True}
    app, _, _ = _build_app(
        tmp_path, warmup=False, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
        stream=_fake_stream(text="<tool>", finish="stop"),
    )
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "user", "content": "list files"}],
        "tools": [{"type": "function", "function": {"name": "bash"}}],
    })
    assert r.status_code == 200
    msg = r.json()["choices"][0]["message"]
    assert msg["tool_calls"] == tool_calls
    assert msg["reasoning_content"] == "thinking"
    # tool_calls flip finish_reason regardless of the stream's finish
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_warmup_fires_per_span_when_enabled(mock_render_cls, mock_openai, tmp_path):
    parsed = {"content": "ok", "reasoning_content": None, "tool_calls": None, "tools_called": False}
    app, openai_inst, _ = _build_app(
        tmp_path, warmup=True, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
    )
    client = TestClient(app)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    })
    assert r.status_code == 200
    warmups = [c for c in openai_inst.completions.create.call_args_list
               if not c.kwargs.get("stream")]
    assert len(warmups) == 2  # one per span
    assert all(w.kwargs["max_tokens"] == 1 for w in warmups)


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_warmup_dedups_across_requests(mock_render_cls, mock_openai, tmp_path):
    parsed = {"content": "ok", "reasoning_content": None, "tool_calls": None, "tools_called": False}
    app, openai_inst, _ = _build_app(
        tmp_path, warmup=True, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
    )
    client = TestClient(app)
    payload = {"model": "test-model",
               "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}

    client.post("/v1/chat/completions", json=payload)
    warmups_1 = [c for c in openai_inst.completions.create.call_args_list if not c.kwargs.get("stream")]
    assert len(warmups_1) == 2  # cold: both spans warmed

    openai_inst.completions.create.reset_mock()
    client.post("/v1/chat/completions", json=payload)  # identical -> spans already cached
    warmups_2 = [c for c in openai_inst.completions.create.call_args_list if not c.kwargs.get("stream")]
    assert len(warmups_2) == 0


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_sparse_span_policy_drives_spans(mock_render_cls, mock_openai, tmp_path):
    """A policy selecting only some messages → sparse span_starts + cross, and
    warmup runs span + cumulative for the in-context region before the span."""
    from span_policy import SpanPolicy

    class OnlyMessageOne(SpanPolicy):
        def select_spans(self, messages):
            return {1}

    parsed = {"content": "ok", "reasoning_content": None, "tool_calls": None, "tools_called": False}
    app, openai_inst, _ = _build_app(
        tmp_path, warmup=True, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
    )
    app.state.span_policy = OnlyMessageOne()  # swap in a sparse policy
    client = TestClient(app)
    # 3 messages, only m1 is a span → non[m0] | span[m1] | non[m2] (tail)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "a"},
                     {"role": "user", "content": "b"},
                     {"role": "assistant", "content": "c"}]})
    assert r.status_code == 200
    real = [c for c in openai_inst.completions.create.call_args_list if c.kwargs.get("stream")][0]
    assert real.kwargs["extra_body"] == {"vllm_xargs": {"span_starts": [32], "cross_span_starts": [64]}}
    warmups = [c for c in openai_inst.completions.create.call_args_list if not c.kwargs.get("stream")]
    assert len(warmups) == 2          # span m1 (phase 1) + cumulative up to it (phase 2)


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_trace_file_written(mock_render_cls, mock_openai, tmp_path):
    """e2e (mocked backend): one request writes run_<ts>_<pid>/{index.json,
    req_000000.json} with the chunk-centric trace — per_pic + cumulative
    warmup folded into chunks, final chunk with usage + output, timing."""
    import json

    from span_policy import SpanPolicy

    class OnlyMessageOne(SpanPolicy):
        def select_spans(self, messages):
            return {1}

    parsed = {"content": "ok", "reasoning_content": "thought", "tool_calls": None,
              "tools_called": False}
    app, _, _ = _build_app(
        tmp_path, warmup=True, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai,
        stream=_fake_stream(cached=32), responses=True,
    )
    app.state.span_policy = OnlyMessageOne()
    client = TestClient(app)
    # 3 messages, only m1 a span → chunks: non[m0] | span[m1] | non[m2 tail] | final
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "a"},
                     {"role": "user", "content": "b"},
                     {"role": "assistant", "content": "c"}]})
    assert r.status_code == 200

    run_dirs = list(tmp_path.glob("run_*"))
    assert len(run_dirs) == 1
    manifest = json.loads((run_dirs[0] / "index.json").read_text())
    assert manifest["block_size"] == 16
    assert manifest["span_policy"] == "all_messages"   # config value (instance swapped in-test)

    trace = json.loads((run_dirs[0] / "req_000000.json").read_text())
    assert trace["status"] == "success"
    assert [c["kind"] for c in trace["chunks"]] == ["non", "span", "non", "final"]

    non0, span1, tail = trace["chunks"][0], trace["chunks"][1], trace["chunks"][2]
    # non[m0]: 18 raw tokens padded to 32; warmed cumulatively (covers itself only).
    assert (non0["content_tokens"], non0["padded_tokens"], non0["pad_added"]) == (18, 32, 14)
    assert non0["warmup"]["kind"] == "cumulative"
    assert non0["warmup"]["covers_chunks"] == [0]
    assert non0["content_preview"].startswith("<system>")
    # span[m1]: per_pic warmed, block-aligned, its end is a cross point.
    assert span1["token_range"] == [32, 64]
    assert span1["is_cross_end"] is True and span1["span_key"].startswith("0x")
    assert span1["warmup"]["kind"] == "per_pic"
    assert span1["warmup"]["decision"] == "warmed"
    # tail: unpadded, not warmed.
    assert tail["pad_added"] == 0 and "warmup" not in tail

    fin = trace["chunks"][-1]
    assert fin["span_starts_sent"] == [32] and fin["cross_span_starts_sent"] == [64]
    assert fin["usage"]["cached_tokens"] == 32
    assert fin["usage"]["cache_hit_rate"] == round(32 / 68, 3)
    assert fin["output"] == {"content": "ok", "reasoning_content": "thought",
                             "tool_calls": None}
    assert trace["timing"]["backend_calls"] == {
        "render": 3, "warmup": 2, "final": 1, "parse": 1, "total": 7}


@patch("middleware.OpenAI")
@patch("processors.RenderClient")
def test_error_trace_on_render_failure(mock_render_cls, mock_openai, tmp_path):
    """A prompt-build failure (e.g. render prefix-instability) returns 500 AND
    leaves an error trace — failed requests must not vanish from the trace dir."""
    import json

    parsed = {"content": "ok", "reasoning_content": None, "tool_calls": None,
              "tools_called": False}
    app, _, render_inst = _build_app(
        tmp_path, warmup=False, parse_result=parsed,
        mock_render_cls=mock_render_cls, mock_openai=mock_openai, responses=True,
    )

    # A hard render failure (the backend /render endpoint erroring) must 500 AND
    # leave an error trace. (Prefix-instability itself no longer 500s — it falls
    # back to no-spans — so we trigger a genuine render error here instead.)
    def failing_render(messages, tools=None, add_generation_prompt=True):
        raise ValueError("render endpoint returned no token_ids")

    render_inst.render_chat.side_effect = failing_render
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "s"},
                     {"role": "user", "content": "u"}]})
    assert r.status_code == 500

    run_dirs = list(tmp_path.glob("run_*"))
    trace = json.loads((run_dirs[0] / "req_000000.json").read_text())
    assert trace["status"] == "error"
    assert trace["error"]["type"] == "ValueError"
    assert "token_ids" in trace["error"]["message"]
    assert trace["chunks"] == []       # failed before any chunk was built
    assert trace["input"]["messages"][0]["content"] == "s"  # input still captured


# --------------------------------------------------------------------------- #
# pure-function units (now importable: no module-level app)
# --------------------------------------------------------------------------- #
def test_process_stream_extracts_text_and_tokens():
    from middleware import process_stream
    text, finish, completion, cached = process_stream(
        _fake_stream(text="Hi", finish="stop", completion_tokens=3, cached=7),
    )
    assert text == "Hi"
    assert finish == "stop"
    assert completion == 3
    assert cached == 7


def test_calculate_usage_info():
    from middleware import calculate_usage_info
    u = calculate_usage_info([1, 2, 3, 4], completion_tokens=5)
    assert u == {"prompt_tokens": 4, "completion_tokens": 5,
                 "total_tokens": 9}


def test_build_response_tool_calls_force_finish_reason():
    from middleware import build_response
    tc = [{"function": {"name": "bash", "arguments": "{}"}}]
    resp = build_response("req_0", "", finish_reason="stop", usage_info={},
                          reasoning_content="why", tool_calls=tc)
    msg = resp["choices"][0]["message"]
    assert msg["tool_calls"] == tc
    assert msg["reasoning_content"] == "why"
    assert resp["choices"][0]["finish_reason"] == "tool_calls"  # flipped from "stop"


def test_build_response_plain():
    from middleware import build_response
    resp = build_response("req_1", "hello", finish_reason="stop", usage_info={})
    msg = resp["choices"][0]["message"]
    assert msg == {"role": "assistant", "content": "hello"}
    assert resp["choices"][0]["finish_reason"] == "stop"
