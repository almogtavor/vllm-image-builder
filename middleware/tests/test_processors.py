"""Unit tests for the (render-only) processor module."""

from processors import pad_tokens


class TestPaddingFunction:
    """Test pad_tokens function."""

    def test_padding_to_block_size(self):
        """Pads to a block-size multiple by inserting pad before the last token."""
        pad_token_id = 999
        tokens = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = pad_tokens(tokens, pad_token_id, block_size=16)

        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9] + [999] * 6 + [10]
        assert result == expected
        assert len(result) == 16

    def test_no_padding_when_already_aligned(self):
        pad_token_id = 999
        tokens = list(range(1, 17))  # 16 tokens
        result = pad_tokens(tokens, pad_token_id, block_size=16)
        assert result == tokens
        assert len(result) == 16

    def test_empty_tokens_unchanged(self):
        assert pad_tokens([], 999, block_size=16) == []


class TestStripReasoning:
    """`_strip_reasoning` removes historical <think> blocks so Qwen3 prefix
    renders stay stable (the template keeps think only for post-last-query turns,
    a context-dependent choice that otherwise breaks _unit_boundaries)."""

    def test_strips_think_block_from_assistant(self):
        from processors import _strip_reasoning
        msgs = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "<think>reasoning</think>\n\nthe answer"},
        ]
        out = _strip_reasoning(msgs)
        assert out[1]["content"] == "the answer"
        assert out[0] is msgs[0]  # non-assistant untouched (identity)

    def test_bare_trailing_marker(self):
        from processors import _strip_reasoning
        msgs = [{"role": "assistant", "content": "loose reasoning</think>\n\nanswer"}]
        assert _strip_reasoning(msgs)[0]["content"] == "answer"

    def test_no_think_untouched(self):
        from processors import _strip_reasoning
        msgs = [{"role": "assistant", "content": "just an answer"}]
        out = _strip_reasoning(msgs)
        assert out[0]["content"] == "just an answer"

    def test_unclosed_think_dropped_to_end(self):
        # A dispatch error can truncate the turn mid-think: an opener with no
        # closer. The partial think must be dropped from the opener to end, else
        # the render depends on the partial block (case E prefix drift).
        from processors import _strip_reasoning
        msgs = [{"role": "assistant", "content": "prefix text\n<think>\nOkay, let me lo"}]
        assert _strip_reasoning(msgs)[0]["content"] == "prefix text"

    def test_unclosed_think_only(self):
        from processors import _strip_reasoning
        msgs = [{"role": "assistant", "content": "<think>\nOkay, let me lo"}]
        assert _strip_reasoning(msgs)[0]["content"] == ""

    def test_preserves_tool_calls_and_other_keys(self):
        from processors import _strip_reasoning
        msgs = [{"role": "assistant", "content": "<think>x</think>\n\ngo",
                 "tool_calls": [{"id": "c1"}]}]
        out = _strip_reasoning(msgs)
        assert out[0]["content"] == "go"
        assert out[0]["tool_calls"] == [{"id": "c1"}]

    def test_tool_and_user_roles_ignored(self):
        from processors import _strip_reasoning
        msgs = [
            {"role": "tool", "content": "<think>not stripped</think>x"},
            {"role": "user", "content": "<think>not stripped</think>y"},
        ]
        out = _strip_reasoning(msgs)
        assert "not stripped" in out[0]["content"]
        assert "not stripped" in out[1]["content"]


class TestMessageUnitsMergeAssistantTool:
    """Assistant+tool_calls merges with its following tool run so cuts never land
    right after a bare assistant (which Qwen3 renders differently as last-vs-middle)."""

    def test_assistant_toolcall_merges_with_tool_run(self):
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "go", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "r1", "tool_call_id": "c1"},
            {"role": "assistant", "content": "go2", "tool_calls": [{"id": "c2"}]},
            {"role": "tool", "content": "r2", "tool_call_id": "c2"},
        ]
        # (0,1) user | (1,3) asst+tool | (3,5) asst+tool
        assert _message_units(msgs) == [(0, 1), (1, 3), (3, 5)]

    def test_plain_tool_run_stays_one_unit(self):
        from processors import _message_units
        msgs = [
            {"role": "tool", "content": "a"},
            {"role": "tool", "content": "b"},
        ]
        assert _message_units(msgs) == [(0, 2)]

    def test_content_toolcall_merges_with_bare_user_error_reply(self):
        # Case D (live): the agent returns a tool call's outcome as an UNWRAPPED
        # role:user turn (a dispatch error, no <tool_response>). It must still
        # merge into the assistant tool-call turn's unit — else the cut lands
        # after a bare assistant tool-call and Qwen3 drifts (last-vs-middle).
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "Resolve the given issue..."},
            {"role": "assistant", "content": "<tool_call>\n{\"name\":\"bash\"}"},
            {"role": "user", "content": "Error: Sending the request failed"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\":\"bash\"}"},
            {"role": "user", "content": "Error: again"},
        ]
        # (0,1) user | (1,3) asst+error-reply | (3,5) asst+error-reply
        assert _message_units(msgs) == [(0, 1), (1, 3), (3, 5)]

    def test_bare_user_alone_not_a_tool_run(self):
        # A standalone/leading real user query is NOT a call reply — never merged.
        from processors import _message_units
        assert _message_units([{"role": "user", "content": "just a question"}]) == [(0, 1)]

    def test_assistant_without_toolcall_merges_with_error_reply(self):
        # Case E (live): a dispatch error truncated the assistant turn BEFORE any
        # <tool_call> was emitted (an unclosed <think> and no parseable call), and
        # the error came back as a bare role:user turn. The merge must not depend
        # on detecting a tool call — ANY assistant followed by a reply is one unit,
        # else the cut after the bare assistant drifts (Qwen3 last-vs-middle).
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "Resolve the given issue..."},
            {"role": "assistant", "content": "<think>\nOkay, let me look"},
            {"role": "user", "content": "Error: Sending the request failed"},
        ]
        assert _message_units(msgs) == [(0, 1), (1, 3)]

    def test_trailing_assistant_answer_is_own_unit(self):
        # A final assistant answer with nothing after it has no reply to merge and
        # stays its own unit (guards the generalized merge from over-reaching).
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "go", "tool_calls": [{"id": "c"}]},
            {"role": "tool", "content": "r"},
            {"role": "assistant", "content": "final answer"},
        ]
        assert _message_units(msgs) == [(0, 1), (1, 3), (3, 4)]


class TestHasToolCall:
    """Assistant tool calls are merged whether structured (tool_calls field) or
    echoed back as <tool_call> text in content (Hermes/qwen25) — both must merge
    with the following tool run or the cut after the assistant is unstable."""

    def test_structured_tool_calls(self):
        from processors import _has_tool_call
        assert _has_tool_call({"role": "assistant", "tool_calls": [{"id": "c"}]})

    def test_content_embedded_tool_call(self):
        from processors import _has_tool_call
        assert _has_tool_call({"role": "assistant",
                               "content": "<tool_call>\n{\"name\":\"read\"}\n</tool_call>"})

    def test_plain_assistant_no_tool_call(self):
        from processors import _has_tool_call
        assert not _has_tool_call({"role": "assistant", "content": "just an answer"})

    def test_content_embedded_merges_with_tool_run(self):
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "<tool_call>\n{\"name\":\"r\"}\n</tool_call>"},
            {"role": "tool", "content": "r1"},
        ]
        # asst(1) has a content-embedded tool call -> merges with tool(2) -> (1,3)
        assert _message_units(msgs) == [(0, 1), (1, 3)]


class TestIsToolResult:
    """Tool results are recognized whether role:tool or role:user wrapping
    <tool_response> (Qwen3 renders both as a user tool_response block)."""

    def test_role_tool(self):
        from processors import _is_tool_result
        assert _is_tool_result({"role": "tool", "content": "x"})

    def test_role_user_tool_response_wrapper(self):
        from processors import _is_tool_result
        assert _is_tool_result({"role": "user", "content": "<tool_response>\ndata\n</tool_response>"})

    def test_plain_user_not_tool_result(self):
        from processors import _is_tool_result
        assert not _is_tool_result({"role": "user", "content": "a real question"})

    def test_assistant_not_tool_result(self):
        from processors import _is_tool_result
        assert not _is_tool_result({"role": "assistant", "content": "x"})

    def test_tool_as_user_wrapper_merges(self):
        from processors import _message_units
        msgs = [
            {"role": "user", "content": "t"},
            {"role": "assistant", "content": "<tool_call>\n{}\n</tool_call>"},
            {"role": "user", "content": "<tool_response>\ndata\n</tool_response>"},
        ]
        # asst(1) tool-call merges with the user-wrapped tool_response(2) -> (1,3)
        assert _message_units(msgs) == [(0, 1), (1, 3)]
