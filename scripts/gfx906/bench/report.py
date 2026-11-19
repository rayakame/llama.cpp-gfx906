#!/usr/bin/env python3
"""report.py <outdir>: the tables of a bench.sh run as Markdown, the first label against the others.

Runs with a flag (a stop at a temperature limit, another power cap, a failed start) are left out and listed.
"""
import glob
import json
import os
import statistics
import sys

TESTS = [("pp4", 4, 0), ("pp8", 8, 0), ("pp16", 16, 0), ("pp512", 512, 0), ("pp2048", 2048, 0), ("tg128", 0, 128)]


def runs(out, q, label):
    prefixes = {p.rsplit(".", 2)[0] for p in glob.glob(os.path.join(out, q, f"{label}.*.*.samples"))}
    for run in sorted(prefixes):
        yield run, os.path.exists(run + ".flags") and os.path.getsize(run + ".flags") > 0


def bench_samples(run):
    found = {}
    if not os.path.exists(run + ".bench.jsonl"):
        return found
    for line in open(run + ".bench.jsonl"):
        if not line.startswith("{"):
            continue
        r = json.loads(line)
        for name, p, n in TESTS:
            if r["n_prompt"] == p and r["n_gen"] == n:
                found.setdefault(name, []).extend(r["samples_ts"])
    return found


def server(run):
    if not os.path.exists(run + ".server.json"):
        return None
    d = json.load(open(run + ".server.json"))
    return [r["predicted_per_second"] for r in d["one"]], d["four_total_per_second"]


def lead(ours, others):
    best = max((o for o in others if o), default=None)
    if not ours or not best:
        return ""
    return f"{(ours / best - 1) * 100:+.0f} %"


def fmt(v):
    return f"{v:.1f}" if v else "-"


def main():
    out = sys.argv[1]
    meta = json.load(open(os.path.join(out, "meta.json")))
    labels = [i["label"] for i in meta["images"]]
    files = sorted(d for d in os.listdir(out) if os.path.isdir(os.path.join(out, d)))
    left_out = []
    one, four, bench = {}, {}, {}
    for q in files:
        for label in labels:
            singles, totals, samples = [], [], {}
            for run, flagged in runs(out, q, label):
                if flagged:
                    left_out.append(f"{os.path.relpath(run, out)}: " + " ".join(sorted(set(open(run + ".flags").read().split("\n")) - {""})))
                    continue
                s = server(run)
                if s:
                    singles.extend(s[0])
                    totals.append(s[1])
                for name, v in bench_samples(run).items():
                    samples.setdefault(name, []).extend(v)
            one[q, label] = statistics.median(singles) if singles else None
            four[q, label] = statistics.median(totals) if totals else None
            for name, v in samples.items():
                bench[q, label, name] = statistics.median(v)

    w = sys.stdout.write
    base, others = labels[0], labels[1:]
    cap = f"power cap fixed at {meta['cap']} W" if meta["cap"] else "power cap not fixed"
    w(f"Measured {meta['date'][:10]}, {cap}, {meta['reps']} repetitions, both passes (A B C C B A), median.\n\n")
    for i in meta["images"]:
        w(f"- `{i['label']}`: `{i['image']}` ({i['version'] or 'no version'})\n")
    w("\n### Writing with MTP, tokens/s\n\nOne user: fixed, prose and code at temperature 0. Four users: four requests at once, total.\n\n")
    w("| File | " + " | ".join(f"{label}, 1 user" for label in labels) + f" | {base} ahead | "
      + " | ".join(f"{label}, 4 users" for label in labels) + f" | {base} ahead |\n")
    w("| --- " * (2 * len(labels) + 3) + "|\n")
    for q in files:
        cells = [fmt(one[q, label]) for label in labels] + [lead(one[q, base], [one[q, o] for o in others])]
        cells += [fmt(four[q, label]) for label in labels] + [lead(four[q, base], [four[q, o] for o in others])]
        w(f"| {q} | " + " | ".join(cells) + " |\n")
    w("\n### llama-bench, tokens/s\n\npp4, pp8 and pp16 are the shapes of the MTP verify pass for one, two and four users.\n\n")
    w("| File | Test | " + " | ".join(labels) + f" | {base} ahead |\n")
    w("| --- " * (len(labels) + 3) + "|\n")
    for q in files:
        for name, _, _ in TESTS:
            vals = [bench.get((q, label, name)) for label in labels]
            w(f"| {q} | {name} | " + " | ".join(fmt(v) for v in vals) + f" | {lead(vals[0], vals[1:])} |\n")
    if left_out:
        w("\nLeft out:\n\n" + "".join(f"- {r}\n" for r in left_out))


if __name__ == "__main__":
    main()
