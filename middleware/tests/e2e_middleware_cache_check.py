"""End-to-end: drive a growing multi-turn conversation at the *deployed middleware*
(chat/completions in -> middleware splits per message, warms up PIC chunks, forwards
to the co-located vLLM -> reply back) and watch the vLLM prefix-cache counters to
confirm earlier PIC chunks are reused as the conversation progresses.

Assumes two local port-forwards are already up:
    oc port-forward -n llm-d-pic <pod> 18080:8080   # middleware
    oc port-forward -n llm-d-pic <pod> 18001:8000   # vLLM (for /metrics)
"""
import re
import sys
import time

import requests

MW = "http://localhost:18080"     # middleware (chat completions)
VLLM = "http://localhost:18001"   # vLLM (metrics)
MODEL = "MiniMaxAI/MiniMax-M2.7"
TURNS = 8
GEN_TOKENS = 32                   # small real reply so the conversation can grow naturally

USER_PROMPTS = [
    "Hi! In one short sentence, what is a prefix cache in an LLM server?",
    "Got it. Now, in one sentence, why does padding chunks to a block size help?",
    "Thanks. What's the difference between the '+' and '@' span separators? One line.",
    "And what does 'position-independent caching' add on top of normal prefix caching?",
    "Briefly: what is a 'PIC chunk' in this middleware?",
    "Why does the middleware do a warmup pass before the real request?",
    "What's one limitation of per-message splitting with the MiniMax chat template?",
    "Summarize this whole conversation in one sentence.",
]


def metrics():
    txt = requests.get(f"{VLLM}/metrics", timeout=10).text
    def g(name):
        m = re.search(rf'^{re.escape(name)}\{{[^}}]*}}\s+([0-9.eE+]+)', txt, re.M)
        return float(m.group(1)) if m else None
    return g("vllm:prefix_cache_queries_total"), g("vllm:prefix_cache_hits_total")


def chat(messages):
    r = requests.post(f"{MW}/v1/chat/completions",
                      json={"model": MODEL, "messages": messages,
                            "max_tokens": GEN_TOKENS, "temperature": 0.0},
                      timeout=600)
    r.raise_for_status()
    return r.json()


def main():
    # sanity
    try:
        m = requests.get(f"{MW}/v1/models", timeout=10)
        print("middleware /v1/models:", m.status_code, m.text[:200])
    except Exception as e:
        print("middleware /v1/models not reachable:", e)
    q0, h0 = metrics()
    print(f"vLLM prefix-cache counters at start: queries={q0:.0f} hits={h0:.0f}\n")

    messages = []
    print(f"{'turn':>4} {'msgs':>4} | {'Δqueries (tok)':>14} {'Δhits (tok)':>12} {'hit%':>6} | reply (first 70 chars)")
    print("-" * 110)
    prev_q, prev_h = metrics()
    for i in range(TURNS):
        messages.append({"role": "user", "content": USER_PROMPTS[i % len(USER_PROMPTS)]})
        before = metrics()
        t0 = time.time()
        resp = chat(messages)
        dt = time.time() - t0
        after = metrics()
        reply = resp["choices"][0]["message"]["content"] or ""
        usage = resp.get("usage", {})
        messages.append({"role": "assistant", "content": reply})
        dq = after[0] - before[0]
        dh = after[1] - before[1]
        hp = f"{100*dh/dq:5.1f}" if dq > 0 else "  n/a"
        r1 = reply.replace("\n", " ")[:70]
        print(f"{i+1:>4} {len(messages):>4} | {dq:>14.0f} {dh:>12.0f} {hp:>6} | "
              f"[{dt:4.1f}s, pt={usage.get('prompt_tokens','?')}] {r1}")
    print()
    qf, hf = metrics()
    print(f"vLLM prefix-cache counters at end:   queries={qf:.0f} hits={hf:.0f}  "
          f"(over the run: {qf-q0:.0f} tokens queried, {hf-h0:.0f} hit -> {100*(hf-h0)/max(1,qf-q0):.1f}%)")
    print("\nExpect: Δhits per turn climbs as the conversation grows (earlier PIC chunks reused),")
    print("while the miss part (Δqueries - Δhits) stays ~one new turn's worth.")


if __name__ == "__main__":
    main()
