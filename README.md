# llama.cpp-gfx906

**One goal: the fastest Qwen3.8 27B an AMD Instinct MI50 can run, for every quantization that fits.**

This is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for one card and one model: the MI50 32 GB (gfx906, the chip of the MI60, Radeon VII and Radeon Pro VII as well) and Qwen3.8 27B with its built-in MTP draft head. Every change here was written for that pair and measured on the card, and nothing lands that makes a quantization or a workload slower.

Against upstream's newest release, v1.0.0, on the same card at the same 205 W, v0.7.0 writes faster on all six files tested: +53 % to +102 % with one user and +105 % to +123 % with four users at once. It is ahead of [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp) on every file as well, from +15 % with one user on Q8_0 up; mx-llama.cpp keeps only the prefill of long prompts on Q8_0 (pp512 and pp2048, 3 to 8 % faster).

## Contents

- [Results](#results)
- [Quick start](#quick-start)
- [Quantizations](#quantizations)
- [What is changed](#what-is-changed)
- [How changes are tested](#how-changes-are-tested)
- [Upstream and credits](#upstream-and-credits)

## Results

Every release is measured against upstream's newest release and against [mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp), a llama.cpp fork tuned for gfx906, all three built from the same Dockerfile with the same ROCm and flags, on the same card in one session. The order is A B C C B A per file, every run starts from a 50 °C chip, and the card's power cap is held at 205 W with its fans at full speed for the whole run, so that only the source differs.

v0.7.0 against upstream v1.0.0 and mx-llama.cpp at commit 557a6068f, tokens/s, the best of each group in bold:

| File | 1 user: fork | upstream | mx | 4 users: fork | upstream | mx |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_1 | **76.2** | 37.7 | 37.2 | **122.5** | 55.8 | 45.2 |
| Q4_K_M | **55.1** | 35.3 | 42.9 | **109.5** | 49.1 | 16.1 |
| Q8_0 | **49.6** | 32.4 | 43.3 | **92.7** | 45.2 | 55.3 |
| UD-IQ3_XXS | **45.9** | 26.6 | 27.0 | **79.5** | 37.0 | 43.1 |
| UD-Q4_K_XL | **60.7** | 35.8 | 39.4 | **102.6** | 46.2 | 17.0 |
| UD-Q6_K | **58.6** | 37.3 | 41.7 | **107.2** | 52.3 | 23.5 |

One user is the speed of a single chat: three prompts (a short answer, prose and code), each the median of both passes, and the mean of the three; four users is the total of four requests at once. Both with MTP (`--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.3`), 32k context, four slots. The full tables, with `llama-bench` prefill and the verify-pass shapes for every file, are in [docs/benchmarks/v0.7.0.md](docs/benchmarks/v0.7.0.md); earlier releases are in [docs/benchmarks](docs/benchmarks/).

## Quick start

The image `ghcr.io/rayakame/llama.cpp-gfx906` runs `llama-server` and needs only the card's device files; the host needs the in-kernel `amdgpu` driver and nothing from ROCm.

```sh
docker run -d --device /dev/kfd --device /dev/dri -v /path/to/models:/models -p 8080:8080 \
  ghcr.io/rayakame/llama.cpp-gfx906:b12230-v0.7.0 \
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
    image: ghcr.io/rayakame/llama.cpp-gfx906:b12230-v0.7.0
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
| Q4_1, Q4_0 | Q4_1, Q4_0 | row-group mat-vec for 2..8 columns and a wide one for 9..16, up/gate/GLU in one mat-vec; MMQ lane grids, for Q4_1 also the fused gate/up MMQ |
| Q8_0 | Q8_0, the Gated DeltaNet gate projections of the UD files | weights in planes for aligned 16-byte loads; mat-vec up to 8 columns; MMQ lane grid with an unrolled, prefetched K loop |
| Q6_K | UD-Q6_K, the output head of most files | row-group mat-vec with 4 rows per group; the output head in planes; MMQ lane grid |
| Q4_K, Q5_K | Q4_K_M, UD-Q4_K_XL, UD-Q5 | row-group mat-vec, 16-byte weight loads, K-quant MMQ loaders with 16-byte y reads, a mat-vec/MMQ switch per shape |
| Q2_K, Q3_K, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, IQ1_M | UD-IQ3_XXS and the other UD-Q2/UD-Q3 files | row-group mat-vec for 2..8 columns, IQ3 grids in LDS, wider MMQ tiles, MMQ for IQ1_M instead of rocBLAS |

Every file gains from the parts that are not weight kernels: flash attention, the Gated DeltaNet kernels, the fusions, the graph cache and the host-side changes. The fastest file on this card is Q4_1, and the [results](#results) show where each tested file stands.

## What is changed

The branch `gfx906` is upstream tag `b12230` plus one commit per change; [docs/gfx906.md](docs/gfx906.md) lists every commit with its source and its off switch. In short:

- **The MTP verify pass.** With MTP every step multiplies 2 to 16 token columns at once, a shape upstream treats as either a mat-vec or a small matrix product. Here mat-vec kernels keep each column's activations in registers and let the row groups of a wave share them, for Q4_0, Q4_1, Q8_0, the K-quants and most IQ types up to 8 columns and for Q4_0 and Q4_1 up to 16; a switch per type and shape picks the mat-vec or the matrix kernel.
- **Prefill.** Matrix kernels with several rows and columns per lane (Q4_0, Q4_1, Q6_K and the Q8_0 layout), register prefetch, stream-k by resident blocks, Q8_0 and Q6_K weights rearranged into planes at load for aligned loads, and the quantized input reused across the projections that share it.
- **Flash attention.** Tile configs for 1 to 8 tokens, K/V prefetch into registers, fully masked tiles skipped.
- **Gated DeltaNet.** A column-thread recurrence kernel; the conv step with its q/k norms and the gate in one launch each; the recurrent states read and updated in place in the cache.
- **Fusions.** Residual add with RMS norm, RMS norm with rope and the cache write, gate/up/GLU.
- **Host side.** Graph executables are re-instantiated instead of updated, which stops ROCm's per-request memory growth; graphs are cached per shape; an exact top-k for the sampler and the draft, with the top logits found on the card; drafts of several requests padded to one length, so they verify together; the MTP draft head over the first 100256 vocabulary rows.

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
