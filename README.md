# llama.cpp-gfx906

**One goal: the fastest Qwen3.8 27B an AMD Instinct MI50 can run, for every quantization that fits.**

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for one card and one model: the MI50 32 GB (gfx906, the chip of the MI60, Radeon VII and Radeon Pro VII as well) and Qwen3.8 27B with its built-in MTP draft head. Every change here was written for that pair and measured on the card, and nothing lands that makes a quantization or a workload slower.

Against upstream's newest release, v1.0.0, on the same card at the same 205 W, v0.6.0 is faster on all six files tested with four users at once (+31 % to +65 %) and with one user on five of them (up to +77 % on Q4_1; level on Q8_0). [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) is still ahead with one user on Q8_0, Q4_K_M and UD-Q4_K_XL; the Q8_0 and Q4_K kernels are the next work.

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Quantizations](#quantizations)
- [What is changed](#what-is-changed)
- [How changes are tested](#how-changes-are-tested)
- [Upstream and credits](#upstream-and-credits)

## Results

Every release is measured against upstream's newest release and against [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), a llama.cpp fork tuned for gfx906, all three built from the same Dockerfile with the same ROCm and flags, on the same card in one session. The order is A B C C B A per file, every run starts from a 50 °C chip, and the card's power cap is held at 205 W with its fans at full speed for the whole run, so that only the source differs.

v0.6.0 against upstream v1.0.0 and mx-llama.cpp at commit 557a6068f, tokens/s, the best of each group in bold:

| File | 1 user: fork | upstream | mx | 4 users: fork | upstream | mx |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_1 | **70.0** | 39.6 | 39.9 | **92.0** | 58.0 | 46.2 |
| Q4_K_M | 40.8 | 35.9 | **43.1** | **82.9** | 50.3 | 16.0 |
| Q8_0 | 35.7 | 35.6 | **43.1** | **64.3** | 49.2 | 55.3 |
| UD-IQ3_XXS | **28.3** | 25.8 | 27.0 | **58.5** | 37.5 | 43.1 |
| UD-Q4_K_XL | 39.4 | 37.4 | **41.0** | **71.1** | 46.9 | 17.2 |
| UD-Q6_K | **42.8** | 37.3 | 42.0 | **72.1** | 52.6 | 23.6 |

One user is the speed of a single chat, the median over three prompts (a short answer, prose and code) in both passes; four users is the total of four requests at once. Both with MTP (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3`), 32k context, four slots. The full tables, with `llama-bench` prefill and the verify-pass shapes for every file, are in [docs/benchmarks/v0.6.0.md](docs/benchmarks/v0.6.0.md).

## Quick start

The image `ghcr.io/rayakame/llama.cpp-gfx906` runs `llama-server` and needs only the card's device files; the host needs the in-kernel `amdgpu` driver and nothing from ROCm.

```sh
docker run -d --device /dev/kfd --device /dev/dri -v /path/to/models:/models -p 8080:8080 \
  ghcr.io/rayakame/llama.cpp-gfx906:b12230-v0.6.0 \
  -m /models/Qwen3.8-27B-Q4_1.gguf -ngl 99 -fa on -lm dio \
  -c 131072 -np 4 --kv-unified -b 2048 -ub 2048 \
  --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3 \
  --jinja --host 0.0.0.0 --port 8080
```

| Flag | Why |
| --- | --- |
| `--spec-type draft-mtp --spec-draft-n-max 3` | Qwen3.8 drafts three tokens with its own MTP head and the card verifies them in one pass. The largest single gain. |
| `--spec-draft-p-min 0.3` | drafts only tokens the head is fairly sure of: fewer wasted verify columns |
| `-lm dio` | reads the model with direct I/O instead of mmap, which has been reported to hang on gfx906 setups |
| `-np 4 --kv-unified` | four requests share one context and are verified together |
| `-b 2048 -ub 2048` | the prefill batch the kernels are tuned for |

The same with Docker Compose:

```yaml
services:
  llama:
    image: ghcr.io/rayakame/llama.cpp-gfx906:b12230-v0.6.0
    command: [-m, /models/Qwen3.8-27B-Q4_1.gguf, -ngl, "99", -fa, "on", -lm, dio, -c, "131072", -np, "4",
              --kv-unified, -b, "2048", -ub, "2048", --spec-type, draft-mtp, --spec-draft-n-max, "3",
              --spec-draft-p-min, "0.3", --jinja, --host, 0.0.0.0, --port, "8080"]
    devices: [/dev/kfd, /dev/dri]
    volumes: [/path/to/models:/models:ro]
    ports: ["8080:8080"]
```

## Quantizations

A GGUF file mixes weight types, and the speed of each type is its own kernel work. What the fork has for each type today:

| Weight type | Carries most of | Work in this fork |
| --- | --- | --- |
| Q4_1, Q4_0 | Q4_1, Q4_0 | multi-column mat-vec for the verify pass; Q4_1 matrix kernel with 2 rows per lane |
| Q5_K, Q6_K | the UD-Q5 and UD-Q6 files, the output head of most files | multi-column mat-vec for the verify pass |
| Q4_K, Q8_0, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, Q3_K | Q4_K_M, Q8_0 and the UD-Q3/UD-Q4 files | upstream's kernels with the GCN tile configs and stream-k; their own mat-vec work is next |

Every file gains from the parts that are not weight kernels: the Gated DeltaNet kernels, the fused gate, the graph cache and the host-side changes. The fastest file on this card is Q4_1, and the [results](#results) show where each tested file stands.

## What is changed

The branch `gfx906` is upstream tag `b12230` plus one commit per change; [docs/gfx906.md](docs/gfx906.md) lists every commit with its source and its off switch. In short:

- **The MTP verify pass.** With MTP every step multiplies 2 to 8 token columns at once, a shape upstream treats as either a mat-vec or a small matrix product. Here a mat-vec kernel keeps each column's activations in registers and lets the row groups of a wave share them, so activations are read once per group of rows instead of once per row (Q4_0, Q4_1, Q5_K, Q6_K).
- **Prefill.** The Q4_1 matrix kernel computes 2 rows per lane, so the lanes of a wave no longer all read the same activations: 45 % fewer shared-memory reads per dot product. GCN tile configs and stream-k for small batches, from upstream PRs.
- **Gated DeltaNet.** A column-per-thread recurrence kernel; the gate (two small projections, softplus, sigmoid) in one launch; the recurrent state updated in place; the conv state's store computes its slot order itself.
- **Host side.** Graph executables are re-instantiated instead of updated, which stops ROCm's per-request memory growth; graphs are cached per shape, so a changing draft length replays; an exact top-k for the sampler and the draft; drafts of several requests padded to one length, so they verify together.

Every new path has an environment variable that turns it off ([list](docs/gfx906.md#the-changes)).

## How changes are tested

- `test-backend-ops` against the CPU for every op a change touches, at the shapes Qwen3.8 27B uses.
- Answers at temperature 0 compared with the build before; a change in summation order is explained, not waved through.
- `llama-bench` pp4, pp8, pp16 (the verify pass of one, two and four requests), tg128, pp512 and pp2048, A/B/B/A against the previous build in one session.
- The server with MTP: one request and four at once.
- Before a release: a real workload of document filing and tool calls with its scores compared, the [memory leak test](scripts/gfx906/leaktest/leaktest.sh), and the [release benchmark](scripts/gfx906/bench/bench.sh) against upstream and mx-llama.cpp.
- CI compiles every kernel for gfx906 and fails when one starts spilling registers ([`gfx906-check`](.github/workflows/gfx906-check.yml)).

## Upstream and credits

This fork follows upstream by rebasing `gfx906` onto newer tags; changes upstream merges are dropped from the branch.

- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT; the upstream PRs carried here are named in [docs/gfx906.md](docs/gfx906.md).
- [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) for ROCm with gfx906 and the build flags.
- [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) for the column-thread Gated DeltaNet kernel.
- [JeanP00l/mi50-qwen3.8-fast](https://github.com/JeanP00l/mi50-qwen3.8-fast) for the idea of a multi-column Q4 mat-vec for the verify pass.
- [exabit-io](https://github.com/exabit-io/llama.cpp-gfx906-tuning) for the register limit of the head-256 attention kernel.
- The gfx906 community on Discord, for years of findings about this card.

Upstream's README, for everything that is not specific to this fork: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp#readme).

License: MIT, as upstream ([LICENSE](LICENSE)).
