"""Mock test for request_trace: build a chunk-centric trace from synthetic
inputs (no live vLLM) and verify the shape + derived fields + file writing.

Runnable two ways:
    uv run pytest tests/test_request_trace.py
    python tests/test_request_trace.py          # stdlib only
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # middleware/
import request_trace as rt


def _mock_inputs():
    """The tool_file_read scenario: 7 messages → 3 in-context + 2 PIC spans, both warmup phases."""
    messages = [
        {"role": "system", "content": "You are a coding agent in a sandboxed repo. Use bash to inspect files."},
        {"role": "user", "content": "Refactor the retry logic in src/client.py to use exponential backoff."},
        {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "cat src/client.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "bash",
         "content": "import time\nimport httpx\n\nclass ApiClient: ..."},
        {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "cat src/retry.py"}'}}]},
        {"role": "tool", "tool_call_id": "c2", "name": "bash",
         "content": "def with_retry(fn, retries=3, base=0.5): ..."},
        {"role": "user", "content": "Now apply exponential backoff and show the tests."},
    ]
    chunks = [
        {"kind": "non", "start": 0, "end": 896, "source_messages": [0, 1, 2], "content_tokens": 870,
         "warmup": {"kind": "cumulative", "label": "cumulative→896", "covers_chunks": [0],
                    "xargs": {"span_starts": [], "cross_span_starts": []}, "decision": "warmed",
                    "backend_call": {"cached_tokens": 0, "real_compute_tokens": 896, "round_trip_ms": 78.5},
                    "cache_op": "added"}},
        {"kind": "span", "start": 896, "end": 1600, "source_messages": [3], "content_tokens": 690,
         "is_cross_end": True, "span_key": "0x21bd9f",
         "warmup": {"kind": "per_pic", "decision": "warmed",
                    "backend_call": {"cached_tokens": 0, "real_compute_tokens": 704, "round_trip_ms": 61.0},
                    "cache_op": "added", "evicted_key": None}},
        {"kind": "non", "start": 1600, "end": 1664, "source_messages": [4], "content_tokens": 30,
         "warmup": {"kind": "cumulative", "label": "cumulative→1664", "covers_chunks": [0, 1, 2],
                    "xargs": {"span_starts": [896], "cross_span_starts": [1600]}, "decision": "warmed",
                    "backend_call": {"cached_tokens": 1600, "real_compute_tokens": 64, "round_trip_ms": 33.2},
                    "cache_op": "added"}},
        {"kind": "span", "start": 1664, "end": 1984, "source_messages": [5], "content_tokens": 300,
         "is_cross_end": True, "span_key": "0xa3f7d1",
         "warmup": {"kind": "per_pic", "decision": "warmed",
                    "backend_call": {"cached_tokens": 0, "real_compute_tokens": 320, "round_trip_ms": 38.4},
                    "cache_op": "added", "evicted_key": None}},
        {"kind": "non", "start": 1984, "end": 2032, "source_messages": [6], "content_tokens": 48},  # tail, no warmup
    ]
    final = {
        "span_starts_sent": [896, 1664], "cross_span_starts_sent": [1600, 1984], "round_trip_ms": 2208.6,
        "usage": {"prompt_tokens": 2032, "cached_tokens": 1984, "completion_tokens": 264, "total_tokens": 2296},
        "finish_reason": "tool_calls",
        "output": {"content": "I'll add exponential backoff to both…",
                   "reasoning_content": "client.py uses a fixed 1s sleep; retry.py has no backoff…",
                   "tool_calls": [{"id": "c3", "type": "function", "function": {
                       "name": "write_file", "arguments": '{"path": "src/client.py", "content": "…"}'}}]},
    }
    timing = {"render_ms": 372.4, "warmup_ms": 211.1, "final_round_trip_ms": 2208.6,
              "parse_ms": 24.8, "wall_clock_ms": 2840.9,
              "backend_calls": {"render": 7, "warmup": 4, "final": 1, "parse": 1, "total": 13}}
    return dict(request_id="req_000003", run_id="run_20260609T093455Z_84213",
                timestamp="2026-06-09T09:36:12.481Z", status="success", model="MiniMaxAI/MiniMax-M2.7",
                params={"max_tokens": 4096, "temperature": 0.0, "seed": None},
                messages=messages, tools=[{"type": "function", "function": {"name": "bash"}}],
                chunks=chunks, final=final, timing=timing)


def test_trace_structure():
    t = rt.build_trace(**_mock_inputs())

    assert set(t) == {"request_id", "run_id", "timestamp", "status", "input", "chunks", "timing"}
    assert [c["kind"] for c in t["chunks"]] == ["non", "span", "non", "span", "non", "final"]
    assert "token_ids" not in json.dumps(t)   # removed everywhere

    c0, c1, c4 = t["chunks"][0], t["chunks"][1], t["chunks"][4]
    assert (c0["padded_tokens"], c0["pad_added"]) == (896, 26)
    assert c0["warmup"]["kind"] == "cumulative"
    assert c1["token_range"] == [896, 1600] and (c1["padded_tokens"], c1["pad_added"]) == (704, 14)
    assert c1["span_key"] == "0x21bd9f" and c1["warmup"]["kind"] == "per_pic"
    assert c1["content_preview"].startswith("<tool>")     # readable text, not ids
    assert "warmup" not in c4                              # unwarmed tail chunk

    fin = t["chunks"][-1]
    assert fin["kind"] == "final"
    assert fin["usage"]["cache_hit_rate"] == round(1984 / 2032, 3)   # 0.976, derived
    assert set(fin["output"]) == {"content", "reasoning_content", "tool_calls"}
    assert "span_start" not in fin                        # final isn't a span


def test_write_files():
    t = rt.build_trace(**_mock_inputs())
    run_dir = tempfile.mkdtemp()
    tp = rt.write_trace(run_dir, t["request_id"], t)
    mp = rt.write_manifest(run_dir, {"run_id": t["run_id"], "model": "MiniMaxAI/MiniMax-M2.7",
                                     "span_policy": "tool_file_read", "block_size": 64, "git_sha": "69b9b74"})
    assert Path(tp).name == "req_000003.json" and Path(mp).name == "index.json"
    assert json.loads(Path(tp).read_text()) == t          # round-trips exactly


def test_edge_cases():
    inp = _mock_inputs()

    # Backend didn't report cached_tokens -> no hit rate, no crash.
    inp["final"]["usage"]["cached_tokens"] = None
    t = rt.build_trace(**inp)
    assert "cache_hit_rate" not in t["chunks"][-1]["usage"]

    # Span skipped via local SpanCache -> backend_call null, passes through.
    inp = _mock_inputs()
    inp["chunks"][1]["warmup"] = {"kind": "per_pic", "decision": "skipped_local_cache",
                                  "backend_call": None, "cache_op": None, "evicted_key": None}
    t = rt.build_trace(**inp)
    assert t["chunks"][1]["warmup"]["decision"] == "skipped_local_cache"
    assert t["chunks"][1]["warmup"]["backend_call"] is None

    # Errored request: no final chunk, error recorded, prompt chunks kept.
    inp = _mock_inputs()
    inp.update(status="error", final=None,
               error={"type": "APIConnectionError", "message": "backend unreachable"})
    t = rt.build_trace(**inp)
    assert t["status"] == "error" and t["error"]["type"] == "APIConnectionError"
    assert [c["kind"] for c in t["chunks"]] == ["non", "span", "non", "span", "non"]  # no final


if __name__ == "__main__":
    test_trace_structure()
    print("test_trace_structure: OK")
    test_write_files()
    print("test_write_files:     OK")
    test_edge_cases()
    print("test_edge_cases:      OK")
    # also write a sample where it's easy to eyeball
    t = rt.build_trace(**_mock_inputs())
    out = rt.write_trace("/tmp/trace_sample", t["request_id"], t)
    print("\nchunk timeline:")
    for c in t["chunks"]:
        if c["kind"] == "final":
            tag = f"cached={c['usage']['cached_tokens']}/{c['usage']['prompt_tokens']} ({c['usage']['cache_hit_rate']})"
        else:
            tag = (c.get("warmup") or {}).get("kind", "(not warmed)")
        print(f"  {c['kind']:5}  range {str(c.get('token_range','-')):14}  {tag}")
    print(f"\nsample trace: {out}")
    print("ALL TESTS PASSED")
