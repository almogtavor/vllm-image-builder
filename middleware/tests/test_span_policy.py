"""Tests for the span-selection policy (span_policy.py)."""

import logging

import pytest

from span_policy import (
    SpanPolicy,
    AllMessagesSpanPolicy,
    ToolFileReadSpanPolicy,
    AllToolResponsesSpanPolicy,
    get_span_policy,
)


def _bash_call(call_id, command):
    return {"role": "assistant", "tool_calls": [
        {"id": call_id, "type": "function",
         "function": {"name": "bash", "arguments": '{"command": "%s"}' % command}}]}


def _tool_result(call_id, content="..."):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def test_all_messages_policy_selects_every_index():
    policy = AllMessagesSpanPolicy()
    assert policy.select_spans([{}, {}, {}]) == {0, 1, 2}
    assert policy.select_spans([]) == set()


def test_get_span_policy_returns_registered_instance():
    policy = get_span_policy("all_messages")
    assert isinstance(policy, AllMessagesSpanPolicy)
    assert isinstance(policy, SpanPolicy)


def test_get_span_policy_unknown_raises():
    with pytest.raises(ValueError, match="unknown span_policy"):
        get_span_policy("does_not_exist")


def test_span_policy_is_abstract():
    with pytest.raises(TypeError):
        SpanPolicy()  # cannot instantiate an abstract policy


# --------------------------------------------------------------------------- #
# ToolFileReadSpanPolicy
# --------------------------------------------------------------------------- #
def test_tool_file_read_marks_code_read_response():
    """The tool-response message for `cat utils.py` is selected (not the call)."""
    policy = ToolFileReadSpanPolicy()
    msgs = [
        {"role": "user", "content": "show utils"},
        _bash_call("c1", "cat src/utils.py"),     # index 1: the invoking call
        _tool_result("c1", "def f(): ..."),        # index 2: the file output -> PIC
    ]
    assert policy.select_spans(msgs) == {2}


def test_tool_file_read_various_commands_and_exts():
    policy = ToolFileReadSpanPolicy()
    msgs = [
        _bash_call("a", "grep -n TODO app/server.ts"), _tool_result("a"),   # idx 1 -> PIC
        _bash_call("b", "sed -n '1,40p' main.go"),      _tool_result("b"),   # idx 3 -> PIC
        _bash_call("c", "head -50 config.yaml"),        _tool_result("c"),   # idx 5 -> PIC
    ]
    assert policy.select_spans(msgs) == {1, 3, 5}


def test_tool_file_read_ignores_non_reads_and_non_code():
    policy = ToolFileReadSpanPolicy()
    msgs = [
        _bash_call("a", "ls -la /tmp"),          _tool_result("a"),   # not a read
        _bash_call("b", "cat README"),           _tool_result("b"),   # no code extension
        _bash_call("c", "python script.py"),     _tool_result("c"),   # runs, not reads
        _bash_call("d", "echo hi > out.py"),     _tool_result("d"),   # writes, not reads
    ]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_requires_matching_call():
    policy = ToolFileReadSpanPolicy()
    # tool result with no preceding bash call (orphan id) -> not selected
    assert policy.select_spans([_tool_result("missing", "def f(): ...")]) == set()


def test_tool_file_read_namespaced_bash_name():
    policy = ToolFileReadSpanPolicy()
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"id": "x", "type": "function",
             "function": {"name": "mcp__environment__bash",
                          "arguments": {"command": "cat a.py"}}}]},   # dict args, namespaced name
        _tool_result("x"),
    ]
    assert policy.select_spans(msgs) == {1}


def test_get_span_policy_tool_file_read():
    assert isinstance(get_span_policy("tool_file_read"), ToolFileReadSpanPolicy)


# --------------------------------------------------------------------------- #
# AllToolResponsesSpanPolicy
# --------------------------------------------------------------------------- #
def test_tool_responses_marks_every_tool_message():
    """Every role='tool' message is a span, regardless of what produced it."""
    policy = AllToolResponsesSpanPolicy()
    msgs = [
        {"role": "system", "content": "sys"},
        _bash_call("a", "ls /tmp"),          # assistant tool_call -> not a span
        _tool_result("a", "files..."),        # idx 2 -> span
        {"role": "user", "content": "ok"},
        _bash_call("b", "echo hi > x.py"),     # idx 4 -> not a span
        _tool_result("b", "done"),             # idx 5 -> span
    ]
    assert policy.select_spans(msgs) == {2, 5}


def test_tool_responses_ignores_command_and_content():
    """Unlike tool_file_read, it spans tool outputs even for non-read commands."""
    policy = AllToolResponsesSpanPolicy()
    msgs = [_bash_call("a", "rm -rf /tmp/x"), _tool_result("a", "removed")]
    assert policy.select_spans(msgs) == {1}      # spanned despite a non-read command


def test_tool_responses_none_when_no_tools():
    policy = AllToolResponsesSpanPolicy()
    assert policy.select_spans([{"role": "user", "content": "hi"},
                                {"role": "assistant", "content": "hello"}]) == set()


def test_get_span_policy_tool_responses():
    assert isinstance(get_span_policy("tool_responses"), AllToolResponsesSpanPolicy)
# ToolFileReadSpanPolicy — rejecter regression tests and positive regressions
# Added after multi-lens review tightened the policy (heredoc/tee anchors,
# _EXEC_RE command-position anchor, _GREP_LIST_RE l-anywhere-in-cluster fix).
# --------------------------------------------------------------------------- #
def test_tool_file_read_rejects_stdout_redirect_to_code_file():
    """Pin _WRITE_RE's `> file.py` redirect branch — a `cat input.py > out.py` has both a read verb and a code ext, but the output is the *written* file (unstable, generated). Locks the redirect-to-code-file"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('a', 'cat input.py > out.py'), _tool_result('a')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_cat_heredoc_write():
    """Pin _WRITE_RE's heredoc (`<<`) branch on a realistic `cat > file.py << 'EOF' ... EOF` shape. Uses dict arguments to avoid quoting hell in the JSON-string helper. Without this, dropping the heredoc arm"""
    policy = ToolFileReadSpanPolicy()
    msgs = [
        {'role': 'assistant', 'tool_calls': [
            {'id': 'h', 'type': 'function',
             'function': {'name': 'bash',
                          'arguments': {'command': "cat > file.py << 'EOF'\nprint(1)\nEOF"}}}]},
        _tool_result('h'),
    ]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_tee_write_to_code_file():
    """Pin _WRITE_RE's `tee` branch when tee targets a code file (`cat src.py | tee dst.py`). The output is mirrored to a writable sink; treat the read as polluted. Locks that the tighter command-position + """
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('t', 'cat src.py | tee dst.py'), _tool_result('t')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_keeps_tee_to_log_file():
    """Positive regression for the _WRITE_RE `tee` fix: `cat file.py | tee build.log` is a legit read whose output is duplicated to an *untracked-ext* log. Previously the bare `\btee\b` rejected this; the new anchor keeps it. (.txt joined _CODE_EXTS, so the sink is .log now.)"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('t', 'cat file.py | tee build.log'), _tool_result('t')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_rejects_tee_write_to_txt_file():
    """txt is a tracked extension now, so mirroring output into a .txt via tee is a write to tracked content — rejected, same as tee to a .py."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('t', 'cat file.py | tee notes.txt'), _tool_result('t')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_keeps_txt_and_csv_reads():
    """txt/csv reads qualify as spans (review follow-up: txt, csv added to _CODE_EXTS)."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('a', 'cat notes.txt'), _tool_result('a'),
            _bash_call('b', 'head -50 data.csv'), _tool_result('b')]
    assert policy.select_spans(msgs) == {1, 3}


def test_tool_file_read_keeps_grep_for_left_shift_literal():
    """Positive regression for the _WRITE_RE `<<` fix: `grep '<<=' op.py` searches for left-shift assignment in a code file. The literal `<<` inside the quoted pattern previously triggered the heredoc reject"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', "grep '<<=' op.py"), _tool_result('g')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_rejects_sed_inplace_short_flag():
    """Pin _SED_INPLACE_RE on the textbook `sed -i 's/a/b/' file.py` form. Without this, breaking or dropping _SED_INPLACE_RE would silently mark in-place edits (which mutate the file) as cacheable PIC spans"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('s', "sed -i 's/a/b/' file.py"), _tool_result('s')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_sed_long_in_place_flag():
    """Pin _SED_INPLACE_RE's `--in-place` long-form alternation independently of `-i`. Without this, the long-flag branch can silently regress while the short-flag test still passes."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('s', "sed --in-place 's/a/b/' file.py"), _tool_result('s')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_keeps_sed_dash_n_read():
    """Positive regression: `sed -n '1,40p' file.py` is a read, not in-place. Pins that the `-n` flag does not trigger _SED_INPLACE_RE (a future loosening of `i` to `[in]` would break this)."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('s', "sed -n '1,40p' file.py"), _tool_result('s')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_rejects_pytest_piped_to_head():
    """Pin _EXEC_RE on `pytest test_foo.py | head -50` — head is a read verb + .py is a code ext, but the upstream is dynamic test output, not stable file content. Highest-volume failure mode in agentic SWE-"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('p', 'pytest test_foo.py | head -50'), _tool_result('p')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_python_dash_m_pytest_piped():
    """Pin _EXEC_RE's `python3?` alternation independently of `pytest`. Without this, the `python|python3` branch can silently regress while the pytest case still passes."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('p', 'python -m pytest test.py | tail -n 20'), _tool_result('p')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_manage_py_runner_piped():
    """Pin the `manage.py` alternation of _EXEC_RE on the canonical Django shape `python manage.py test app.tests | head -50`. Each alternation in _EXEC_RE is an independent risk surface."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('m', 'python manage.py test app.tests | head -50'), _tool_result('m')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_node_runner_piped():
    """Pin the `node` alternation of _EXEC_RE on `node script.js | head`. Locks the JS-runner branch independently."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('n', 'node script.js | head'), _tool_result('n')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_keeps_grep_for_runner_name_as_search_term():
    """Positive regression for the _EXEC_RE command-position anchor: `grep nox conf.py` is a legit read whose search term happens to be a runner name. Previously the un-anchored _EXEC_RE rejected it; the new"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep nox conf.py'), _tool_result('g')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_keeps_head_of_backup_with_runner_name_in_path():
    """Positive regression for the _EXEC_RE command-position anchor: `head -20 manage.py.bak` is a legit read of a backup file whose path contains `manage.py` as a substring. Previously the un-anchored regex"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('h', 'head -20 manage.py.bak'), _tool_result('h')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_keeps_grep_for_node_as_search_term():
    """Positive regression for _EXEC_RE: `grep node src/server.js` searches for the literal symbol `node` inside a JS file. Without the command-position anchor, the runner-name substring would falsely reject"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep node src/server.js'), _tool_result('g')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_rejects_grep_list_short_flag():
    """Pin _GREP_LIST_RE on the textbook `grep -l TODO app/server.py` — output is a filename list, not stable file content."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep -l TODO app/server.py'), _tool_result('g')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_grep_list_recursive_combined_flags():
    """Pin the _GREP_LIST_RE fix: `grep -rln TODO src/app.py` has `l` in the middle of a flag cluster (with `n` after). The previous `l\b` anchor missed this; the new `l[A-Za-z]*\b` catches `-rln`, `-lr`, `-"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep -rln TODO src/app.py'), _tool_result('g')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_grep_files_with_matches_long_flag():
    """Pin _GREP_LIST_RE's `--files-with-matches` long-flag alternation independently of `-l`. Without this, the long-flag branch can regress while the short-flag test passes."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep --files-with-matches TODO app/server.py'), _tool_result('g')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_rejects_find_xargs_grep_l_include_py():
    """Pin _GREP_LIST_RE on the SWE-bench-realistic `find . | xargs grep -l "x" --include="*.py"` shape — confirms `[^|]*` segment-scope still anchors `grep` correctly across pipelines. Uses dict arguments t"""
    policy = ToolFileReadSpanPolicy()
    msgs = [
        {'role': 'assistant', 'tool_calls': [
            {'id': 'x', 'type': 'function',
             'function': {'name': 'bash',
                          'arguments': {'command': 'find . | xargs grep -l "x" --include="*.py"'}}}]},
        _tool_result('x'),
    ]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_keeps_grep_context_flag_A():
    """Positive regression for _GREP_LIST_RE precision: `grep -A 5 def file.py` uses the `-A` context flag (no `l`). Asserts that flag clusters without `l` do not trigger the listing rejecter."""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('g', 'grep -A 5 def file.py'), _tool_result('g')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_keeps_grep_recursive_include_py():
    """Positive regression for the SWE-bench top-frequency `grep -r "x" /repo --include="*.py"` content-grep shape. Uses dict arguments to avoid quoting `"` through the JSON-string helper. Confirms `--includ"""
    policy = ToolFileReadSpanPolicy()
    msgs = [
        {'role': 'assistant', 'tool_calls': [
            {'id': 'g', 'type': 'function',
             'function': {'name': 'bash',
                          'arguments': {'command': 'grep -r "x" /repo --include="*.py"'}}}]},
        _tool_result('g'),
    ]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_keeps_cat_with_stderr_redirect():
    """Positive regression: `cat file.py 2>/dev/null` includes a `>` redirect but to `/dev/null` (not a code-ext path). Pins that the load-bearing `\.<ext>` anchor on _WRITE_RE keeps stderr redirects from fa"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('a', 'cat file.py 2>/dev/null'), _tool_result('a')]
    assert policy.select_spans(msgs) == {1}


def test_tool_file_read_does_not_treat_shell_read_as_a_read_verb():
    """Pin the deliberate omission of `read` from _READ_RE: `read line < file.py` is a shell builtin assigning to a variable, not a file-content read. Locks the precision choice so a future revert that re-ad"""
    policy = ToolFileReadSpanPolicy()
    msgs = [_bash_call('r', 'read line < file.py'), _tool_result('r')]
    assert policy.select_spans(msgs) == set()


def test_tool_file_read_logs_per_hit_and_summary_only_when_nonempty(caplog):
    """Pin the logging contract: per-hit INFO line is emitted for each caught span and includes the `cmd[:120]` truncation; the summary INFO line fires only when spans is non-empty. Verifies truncation by pa"""
    policy = ToolFileReadSpanPolicy()
    import logging
    long_cmd = 'cat ' + ('a' * 200) + '.py'
    msgs = [_bash_call('c1', long_cmd), _tool_result('c1')]
    with caplog.at_level(logging.INFO, logger='span_policy'):
        assert policy.select_spans(msgs) == {1}
    records = [r.getMessage() for r in caplog.records]
    assert any('caught msg[1] as PIC span' in m for m in records)
    assert any('selected 1/2 messages as PIC spans' in m for m in records)
    assert not any(long_cmd in m for m in records)  # cmd[:120] truncation


def test_tool_file_read_summary_log_suppressed_when_no_spans(caplog):
    """Pin that the summary log line is suppressed entirely when no spans are selected (avoids `selected 0/M` noise on every chat-only / warmup request). Construct a no-match command and assert no INFO recor"""
    policy = ToolFileReadSpanPolicy()
    import logging
    msgs = [_bash_call('z', 'ls -la /tmp'), _tool_result('z')]
    with caplog.at_level(logging.INFO, logger='span_policy'):
        assert policy.select_spans(msgs) == set()
    records = [r.getMessage() for r in caplog.records]
    assert not any('selected' in m for m in records)
