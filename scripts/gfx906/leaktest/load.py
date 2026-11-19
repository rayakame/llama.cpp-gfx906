#!/usr/bin/env python3
"""load.py <url> <count>: chat requests of changing length, so ggml captures its graphs again for each one.

Prints "<number> <unix time>" after each request.
"""
import json
import random
import sys
import time
import urllib.request

url, count = sys.argv[1], int(sys.argv[2])
rng = random.Random(15)
words = ["lamp", "river", "stone", "garden", "letter", "window", "harbour", "candle", "field", "bridge", "winter", "clock"]
for i in range(count):
    text = " ".join(rng.choice(words) for _ in range(rng.randint(40, 900)))
    body = {"messages": [{"role": "user", "content": f"Request {i}. Count the words in: {text}"}],
            "max_tokens": rng.randint(16, 64), "temperature": 0.7, "seed": i, "ignore_eos": True, "cache_prompt": False}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    print(i, round(time.time(), 1), flush=True)
