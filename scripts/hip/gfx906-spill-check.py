#!/usr/bin/env python3
# Fails when a gfx906 kernel spills more VGPRs than the baseline lists; --update rewrites the baseline from the log.

import importlib.util
import os
import sys

spec = importlib.util.spec_from_file_location("vgpr_check", os.path.join(os.path.dirname(__file__), "gcn-cdna-vgpr-check.py"))
vgpr_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vgpr_check)


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
    spills = {name: d["spill"] for name, d in vgpr_check.parse_log_file(log).items() if d["spill"] > 0}

    if "--update" in sys.argv[3:]:
        with open(baseline, "w") as f:
            f.write("# VGPR spills of gfx906 kernels (ROCm 10.1, max-ilp) that are known and accepted: <spills> <kernel>\n")
            for name in sorted(spills):
                f.write(f"{spills[name]} {name}\n")
        print(f"{len(spills)} spilling kernels written to {baseline}")  # noqa: NP100
        return

    known = read_baseline(baseline)
    new = {n: s for n, s in spills.items() if s > known.get(n, 0)}
    gone = sorted(n for n in known if n not in spills)
    for name in sorted(spills):
        mark = "NEW" if name in new else "known"
        print(f"{spills[name]:5d} {mark:5s} {name}")  # noqa: NP100
    for name in gone:
        print(f"    0 gone  {name}")  # noqa: NP100
    print(f"{len(spills)} kernels spill, {len(new)} new or worse, {len(gone)} no longer spill")  # noqa: NP100
    sys.exit(1 if new else 0)


if __name__ == "__main__":
    main()
