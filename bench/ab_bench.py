#!/usr/bin/env python3
"""A/B bench for the mtp-lend rebuild: decode + prefill via the OpenAI API.

Usage: ab_bench.py <label>
Saves JSON to /home/sens/Strata/bench/results/<label>.json
"""
import json, sys, time, urllib.request, hashlib

PORT = 4243
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "Qwen Flash Next IQ3_S"
OUT = f"/home/sens/Strata/bench/results/{sys.argv[1]}.json"

def chat(messages, max_tokens, extra=None):
    payload = {"model": MODEL, "messages": messages,
               "max_tokens": max_tokens, "stream": False}
    if extra:
        payload.update(extra)
    body = json.dumps(payload).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.load(r)
    wall = time.time() - t0
    t = data.get("timings", {})
    msg = (data.get("choices") or [{}])[0].get("message", {})
    text = (msg.get("content") or "") + "\x00" + (msg.get("reasoning_content") or "")
    return {
        "wall_s": round(wall, 1),
        "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "prompt_tokens": t.get("prompt_n"),
        "predicted_tokens": t.get("predicted_n"),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_per_second": t.get("predicted_per_second"),
        "draft_n": t.get("draft_n"),
        "draft_n_accepted": t.get("draft_n_accepted"),
        "prompt_ms": t.get("prompt_ms"),
        "predicted_ms": t.get("predicted_ms"),
        "timings_keys": sorted(t.keys()),
    }

# --- decode bench: force a full 512-token generation (thinking + answer), 3 runs
decode_prompt = ("Explain how a lighthouse works, from the optics to the keeper's routine. "
                 "Write as much detail as you can and do not stop early.")
decode = []
for i in range(3):
    r = chat([{"role": "user", "content": decode_prompt}], 512)
    print(f"decode run {i+1}: {r['predicted_per_second']} tok/s "
          f"(gen {r['predicted_tokens']} tok, wall {r['wall_s']}s)")
    decode.append(r)

# --- prefill bench: ~20K-token prompt, small generation
para = ("The lighthouse keeper climbed the spiral stairs each evening, logging the weather, "
        "the tide, and the position of every vessel he could see. Fog rolled in from the "
        "northwest most afternoons in autumn, and the beam swept the water in slow circles. "
        "He kept a notebook of every ship that passed, noting name, heading, and speed. "
        "In winter the gales made the lamp house shudder, and he would coil the rope tighter "
        "around the railing before going below to check the oil reserves. ")
# target ~20K tokens at ~3.08 chars/token -> ~61K chars
n = 300
big = (para * n)[:61000]
prefill = []
for i in range(2):
    # unique token per run: an identical prompt would hit the prompt cache and skew the number
    r = chat([{"role": "user", "content": f"[run {i}] " + big + "\nIn one word: is this text about a lighthouse?"}], 16)
    print(f"prefill run {i+1}: {r['prompt_per_second']} tok/s on {r['prompt_tokens']} prompt tokens "
          f"(prompt_ms={r['prompt_ms']})")
    prefill.append(r)

# --- greedy parity probe: temperature 0, fixed prompt, text hash for cross-arm diff
probe = chat([{"role": "user", "content": "Explain what a checksum is, in exactly two short sentences."}],
             96, {"temperature": 0})
print(f"parity probe: {probe['predicted_tokens']} tok, sha {probe['text_sha256']}")

result = {
    "label": sys.argv[1],
    "ts": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    "model": MODEL,
    "port": PORT,
    "parity_probe": probe,
    "decode_runs": decode,
    "prefill_runs": prefill,
}
import os
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print(f"saved -> {OUT}")
