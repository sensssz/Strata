#!/usr/bin/env python3
"""Post-upgrade bench: decode + prefill, matching the v0.2.4.2 baseline method.
Decode: fixed short prompt (fib/memo), max_tokens 160. Run 1 = warm-up; report run 2.
Prefill: ~16K-token prompt of RANDOM UNIQUE pseudo-words + fresh nonce per run, max_tokens 8,
calibrated via one probe against usage.prompt_tokens (never char math).
Writes bench/results/post-<tag>-{decode,prefill}-runN.json and prints a summary line per run.
"""
import json, random, string, sys, time, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "4243"
TAG = sys.argv[2] if len(sys.argv) > 2 else "post"
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = sys.argv[5] if len(sys.argv) > 5 else "qwen3.8-flash-next-abliterated-iq3_s"
DEC_PROMPT = "Write a Python function fib(n) with memo, then explain its complexity in 3 sentences."

def req(messages, max_tokens):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": c} for c in messages],
                       "max_tokens": max_tokens}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=1200) as f:
        return json.load(f)

def pseudo_words(n, rng):
    cons, vow = "bdghjklmnprstv", "aeiouy"
    return " ".join(rng.choice(cons) + rng.choice(vow) + (rng.choice(cons) if rng.random() < 0.4 else "")
                    + (rng.choice(string.digits) if rng.random() < 0.05 else "") for _ in range(n))

def fill_for(target_words, rng):
    return (f"Read this log nonce-{''.join(rng.choices(string.ascii_lowercase, k=8))} and answer OK: "
            + pseudo_words(target_words, rng))

def run_decode(n_runs):
    for i in range(1, n_runs + 1):
        d = req([DEC_PROMPT], 160)
        out = f"bench/results/{LABEL}-{TAG}-decode-run{i}.json"
        json.dump(d, open(out, "w"))
        t = d["timings"]
        print(f"decode run{i}: {t['predicted_per_second']:.1f} tok/s acc {t['draft_n_accepted']}/{t['draft_n']}", flush=True)

def run_prefill(target_tokens, n_runs=2):
    rng = random.Random(time.time_ns())
    words = int(target_tokens * 2.4)  # ~2.13 chars/token pseudo-words -> start high, calibrate down
    d = req([fill_for(words, random.Random(time.time_ns()))], 1)  # probe
    got = d["usage"]["prompt_tokens"]
    words = int(words * target_tokens / got)
    print(f"calib: probe {got} tok -> refill {words} words", flush=True)
    for i in range(1, n_runs + 1):
        r2 = random.Random(time.time_ns() + i)
        d = req([fill_for(words, r2)], 8)
        det = d.get("usage", {}).get("prompt_tokens_details", {})
        assert det.get("cached_tokens", 0) == 0, f"prefix cache fake! cached={det}"
        out = f"bench/results/{LABEL}-{TAG}-prefill-run{i}.json"
        json.dump(d, open(out, "w"))
        t = d["timings"]
        print(f"prefill run{i}: {t['prompt_per_second']:.1f} tok/s @ {d['usage']['prompt_tokens']} tok cached {det.get('cached_tokens')}", flush=True)

if __name__ == "__main__":
    what = sys.argv[4] if len(sys.argv) > 4 else "both"
    if what in ("both", "decode"):
        run_decode(2)
    if what in ("both", "prefill"):
        run_prefill(16000)
