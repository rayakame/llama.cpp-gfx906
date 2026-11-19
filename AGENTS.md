# Instructions for llama.cpp-gfx906

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with one goal: the best speed llama.cpp can reach for Qwen3.8 27B on one AMD Instinct MI50 (gfx906, 32 GB, ROCm 10.1), for as many quantizations as possible. It is not a place to prepare upstream contributions.

---

## Rules for this fork

- Never push to ggml-org/llama.cpp, and never open issues, PRs, comments or reviews there. `gh` defaults to this repository; keep it that way.
- All work is on branch `gfx906`, one commit per change on top of an upstream tag. `UPSTREAM_REF` in `.devops/gfx906.Dockerfile` names the tag. `master` mirrors upstream and gets no commits.
- Moving to a newer upstream tag is a rebase of `gfx906`. Drop commits upstream has merged; re-read the code a commit touches when it no longer applies.
- Upstream's CI workflows are deleted on `gfx906`, so the fork runs only its own `gfx906-*.yml`. When a rebase brings back a changed workflow file, delete it again.
- Commit messages say what changed, why, and what it measured on the MI50. No `Co-authored-by`, `Assisted-by` or other tool lines.
- Every new kernel path or host behavior gets an environment variable that turns it off, named in its commit message and in `docs/gfx906.md`.
- A change lands only when all of these hold:
    - `test-backend-ops` passes against the CPU for every op the change touches, at the shapes Qwen3.8 27B uses.
    - Answers at temperature 0 stay identical, or the difference is explained (summation order) and the quality checks do not move.
    - `llama-bench` pp4, pp8, pp16, tg128, pp512 and pp2048, run A/B/B/A against the previous build in one session, shows no regression in any of them. pp4 and pp16 are the MTP verify pass of one and of four requests.
    - For server-side changes: the server with `--spec-type draft-mtp --spec-draft-n-max 3`, one request and four at once.
- Card limits for every test: stop at 95 °C junction or 85 °C memory.

## Releasing

1. Tag `gfx906-v<x.y.z>` on `gfx906`; `gfx906-image.yml` publishes `ghcr.io/rayakame/llama.cpp-gfx906:<UPSTREAM_REF>-v<x.y.z>`.
2. Run `gfx906 compare images` (workflow_dispatch): upstream's newest release and mx-llama.cpp's newest commit, built from the same Dockerfile, as `compare-upstream-<tag>` and `compare-mx-<commit>`.
3. `scripts/gfx906/leaktest/leaktest.sh` on the release image passes.
4. `scripts/gfx906/bench/bench.sh` measures the release against both compare images with the card's power cap held at 205 W and its fans at full speed (`CAP=205`); `report.py` makes the tables for the release notes and the README.

## Building and testing

```sh
# the image the fork publishes; TESTS=ON adds test-backend-ops
docker build -f .devops/gfx906.Dockerfile --build-arg TESTS=ON -t llama-gfx906 .

# correctness of one op against the CPU, then its speed
docker run --rm --device /dev/kfd --device /dev/dri --entrypoint /app/test-backend-ops llama-gfx906 test -b ROCm0 -o MUL_MAT
docker run --rm --device /dev/kfd --device /dev/dri --entrypoint /app/test-backend-ops llama-gfx906 perf -b ROCm0 -o MUL_MAT
```

## Code and commit standards

- Avoid emdash `—`, unicode arrow `→` or any unicode characters: `×`, `…` ; use ASCII equivalents instead: `-`, `->`, `x`, `...`
- Code comments:
    - Keep code comments concise (usually 1-2 lines)
    - Avoid redundant or excessive inline commentary
    - Avoid hard-wrapping it to a fixed column width - that hurts readability
    - Use ASD-STE100 Simplified Technical English, simple wordings
- Prefer reusing existing infrastructure over introducing new components. Keep changes close to the code they tune, so a rebase onto a newer upstream stays small.
- Do NOT split a line into multiple lines mid-sentence, do NOT try to force the line to fit a fixed number of characters
- Before writing any code, read all relevant files and understand the existing patterns - your changes must blend in with the surrounding codebase.
- Write code first, then add comments only where they are needed.
- Llama.cpp has a dedicated Jinja engine in `common/jinja`; it does not use Minja.

## Useful resources

- [What this fork changes, its switches and measurements](docs/gfx906.md)
- [Build documentation](docs/build.md)
- [Server usage documentation](tools/server/README.md)
- [Server development documentation](tools/server/README-dev.md)
- [Code review guidelines](skills/code-review/SKILL.md) - upstream's per-area rules, useful for any change here
- [Slice model](skills/slice-model/SKILL.md) - build a small slice of a big HF model to test conversion and inference
