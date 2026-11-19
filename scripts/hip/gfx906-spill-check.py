#!/usr/bin/env python3
# Fails when a gfx906 kernel spills more VGPRs than the baseline lists; --update rewrites the baseline from the log.

import os
import re
import sys

FUNC = re.compile(r"^\S+:\d+:\d+:(?: remark:)? Function Name: (\S+)")
SPILL = re.compile(r"^\S+:\d+:\d+:(?: remark:)?\s+VGPRs Spill: (\d+)")


def parse_log(path):
    spills = {}
    name = None
    with open(path, errors="replace") as f:
        for line in f:
            m = FUNC.match(line)
            if m:
                name = m.group(1)
                continue
            m = SPILL.match(line)
            if m and name is not None:
                spills[name] = max(spills.get(name, 0), int(m.group(1)))
    return spills


def read_baseline(path):
    known = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and not line.startswith("#"):
                    known[parts[1]] = int(parts[0])
    return known


def main():
    if len(sys.argv) < 3:
        print("usage: gfx906-spill-check.py <metrics log> <baseline> [--update]", file=sys.stderr)  # noqa: NP100
        sys.exit(2)
    log, baseline = sys.argv[1], sys.argv[2]
    all_kernels = parse_log(log)
    if not all_kernels:
        print(f"no kernel resource remarks in {log}", file=sys.stderr)  # noqa: NP100
        sys.exit(2)
    spills = {n: s for n, s in all_kernels.items() if s > 0}

    if "--update" in sys.argv[3:]:
        with open(baseline, "w") as f:
            f.write("# VGPR spills of gfx906 kernels (ROCm 10.1, max-ilp) that are known and accepted: <spills> <kernel>\n")
            for name in sorted(spills):
                f.write(f"{spills[name]} {name}\n")
        print(f"{len(spills)} of {len(all_kernels)} kernels spill; written to {baseline}")  # noqa: NP100
        return

    known = read_baseline(baseline)
    new = {n: s for n, s in spills.items() if s > known.get(n, 0)}
    gone = sorted(n for n in known if n not in spills)
    for name in sorted(spills):
        mark = "NEW" if name in new else "known"
        print(f"{spills[name]:5d} {mark:5s} {name}")  # noqa: NP100
    for name in gone:
        print(f"    0 gone  {name}")  # noqa: NP100
    print(f"{len(spills)} of {len(all_kernels)} kernels spill, {len(new)} new or worse, {len(gone)} no longer spill")  # noqa: NP100
    sys.exit(1 if new else 0)


if __name__ == "__main__":
    main()
