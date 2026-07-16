"""Per-request trace logging.

Assembles the chunk-centric trace for one request and writes it as a single
JSON file (``<request_id>.json``) under a per-run directory, alongside a run
manifest (``index.json``).

Pure stdlib, no backend calls: the caller passes the data it already has
(messages, the built prompt's chunks, per-chunk warmup, final usage/output),
and this module only shapes and writes it. That keeps it trivially testable
with mock inputs.
"""
import json
from datetime import datetime
from pathlib import Path

__all__ = ["run_dir_name", "fmt_key", "span_key", "build_trace", "write_trace",
           "write_manifest"]


def run_dir_name(started_at: datetime, pid: int) -> str:
    """Directory name for one middleware process: ``run_<UTC>_<pid>``."""
    return f"run_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{pid}"


def fmt_key(key: int) -> str:
    """Hex-format a span content key for the trace."""
    return f"0x{key & 0xFFFFFFFFFFFFFFFF:x}"


def span_key(tokens) -> str:
    """Content key for a span's tokens — same hash warmup dedups on, so the
    same span content maps to the same key across requests. (Int-tuple hashes
    are unsalted, so keys are stable across processes too.)"""
    return fmt_key(hash(tuple(tokens)))


def _preview(messages, idxs, limit=90):
    """A short readable preview of the messages a chunk covers."""
    parts = []
    for i in idxs:
        m = messages[i]
        role = m.get("role", "?")
        if m.get("content"):
            parts.append(f"<{role}> {m['content']}")
        elif m.get("tool_calls"):
            names = ", ".join((tc.get("function") or {}).get("name", "?") for tc in m["tool_calls"])
            parts.append(f"<{role}> tool_call: {names}")
        else:
            parts.append(f"<{role}>")
    text = "  ".join(parts).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def _chunk_record(c, messages):
    """Shape one prompt chunk (kind 'span' | 'non') into its trace record."""
    padded = c["end"] - c["start"]
    rec = {
        "kind": c["kind"],
        "source_messages": c["source_messages"],
        "token_range": [c["start"], c["end"]],
        "content_tokens": c["content_tokens"],
        "padded_tokens": padded,
        "pad_added": padded - c["content_tokens"],
        "content_preview": _preview(messages, c["source_messages"]),
    }
    if c["kind"] == "span":
        rec["span_start"] = c["start"]
        rec["is_cross_end"] = c.get("is_cross_end", False)
        rec["span_key"] = c["span_key"]
    if c.get("warmup") is not None:        # in-context tail chunks aren't warmed
        rec["warmup"] = c["warmup"]
    return rec


def build_trace(*, request_id, run_id, timestamp, status, model, params,
                messages, tools, chunks, final, timing, error=None):
    """Build the chunk-centric trace dict.

    chunks : prompt chunks — dicts with kind/start/end/source_messages/
             content_tokens, plus span_key/is_cross_end for spans and an
             optional pre-shaped ``warmup`` dict (the warmup phase that
             produced it; absent for unwarmed tail chunks).
    final  : the real request + parsed response, appended as a ``kind:"final"``
             chunk. ``cache_hit_rate`` is derived here (skipped when the
             backend didn't report cached_tokens). None for an errored
             request — the trace then has no final chunk.
    error  : optional ``{type, message}`` recorded when status == "error".
    """
    records = [_chunk_record(c, messages) for c in chunks]

    if final is not None:
        usage = dict(final["usage"])
        cached, prompt = usage.get("cached_tokens"), usage.get("prompt_tokens")
        if cached is not None and prompt:
            usage["cache_hit_rate"] = round(cached / prompt, 3)
        records.append({
            "kind": "final",
            "span_starts_sent": final["span_starts_sent"],
            "cross_span_starts_sent": final["cross_span_starts_sent"],
            "round_trip_ms": final["round_trip_ms"],
            "usage": usage,
            "finish_reason": final["finish_reason"],
            "output": final["output"],
        })

    return {
        "request_id": request_id,
        "run_id": run_id,
        "timestamp": timestamp,
        "status": status,
        **({"error": error} if error is not None else {}),
        "input": {
            "model": model,
            "max_tokens": params.get("max_tokens"),
            "temperature": params.get("temperature"),
            "seed": params.get("seed"),
            "messages": messages,
            "tools": tools,
        },
        "chunks": records,
        "timing": timing,
    }


def write_trace(run_dir, request_id, trace) -> str:
    """Write ``<request_id>.json`` under ``run_dir``; return the path."""
    p = Path(run_dir) / f"{request_id}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trace, indent=2, ensure_ascii=False))
    return str(p)


def write_manifest(run_dir, manifest) -> str:
    """Write the run manifest to ``index.json`` (once per run); return the path."""
    p = Path(run_dir) / "index.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return str(p)
