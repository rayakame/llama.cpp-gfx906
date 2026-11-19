#!/usr/bin/env python3
"""client.py <url> <out.json>: three requests one after another, then four at once, all at temperature 0.

Writes the server's own timings per request and prints one summary line.
"""
import json
import sys
import threading
import time
import urllib.request

ONE = [
    ("fixed", "List the first twelve prime numbers, then explain in three sentences why there are infinitely many "
              "primes. Use plain text.", 300),
    ("prose", "Write a detailed explanation of how a canal lock works, about 350 words.", 400),
    ("code", "Write a Python module with five small, documented functions for date arithmetic.", 400),
]
FOUR = [
    "Write a detailed explanation of how a canal lock works, about 350 words.",
    "Write a Python module with five small, documented functions for date arithmetic.",
    "Explain how a refrigerator moves heat, step by step, about 300 words.",
    "Write a Bash script that rotates log files in a directory, with comments.",
]


def ask(url, name, text, max_tokens):
    body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens, "temperature": 0, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json", "Connection": "close"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        reply = json.load(r)
    t = reply.get("timings", {})
    return {"request": name, "wall_s": round(time.time() - t0, 2), "predicted_n": t.get("predicted_n"),
            "predicted_per_second": t.get("predicted_per_second"), "prompt_per_second": t.get("prompt_per_second"),
            "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
            "text": reply["choices"][0]["message"]["content"]}


def main():
    url, out = sys.argv[1], sys.argv[2]
    one = [ask(url, name, text, n) for name, text, n in ONE]

    four = [None] * len(FOUR)

    def run(i):
        four[i] = ask(url, f"four{i}", FOUR[i], 400)

    t0 = time.time()
    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(FOUR))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0
    if any(r is None for r in four):
        sys.exit("a request of the four failed")
    total = sum(r["predicted_n"] for r in four) / wall

    with open(out, "w") as f:
        json.dump({"one": one, "four": four, "four_wall_s": round(wall, 2), "four_total_per_second": total}, f, indent=1)
    print(" ".join(f"{r['request']} {r['predicted_per_second']:.1f}" for r in one) + f", four at once {total:.1f} t/s")  # noqa: NP100


if __name__ == "__main__":
    main()
