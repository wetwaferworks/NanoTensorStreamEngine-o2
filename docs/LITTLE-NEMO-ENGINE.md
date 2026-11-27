# The Little Nemo engine

Little Nemo is a pure C inference engine for NVIDIA's Nemotron-Nano-9B-v2, built
alongside NTSE. It exists for one question, in JD's formulation: **we do not get
to keep the original weights at these pack rates, so the packed model has to
prove that its output is still the same reasoning.** Little Nemo is the
raw-weight reference the packed model will be measured against, and the
conversation front end JD asked for: talk to the model in real time, watch it
think, and see the machine work while it does. Source: `littlenemo/`. Runtime is C
and nothing else; the one Python piece is the GUI ([GUI.md](GUI.md)), which
renders and relays and does no model work. Implementation: Elfie.

## The model

Nemotron-Nano-9B-v2 is a 56-layer hybrid: **27 Mamba-2 layers, 25 MLP layers, 4
attention layers** (at indices 14, 21, 30 and 39), hidden width 4480, vocabulary
131072 with an untied `lm_head`, epsilon 1e-5 everywhere, BOS 1, EOS 12, PAD 0.
Every block has the same shape: RMSNorm, one mixer, residual add. The exact math
the engine reproduces is in
[littlenemo/BLOCK-MATH-SPEC.md](../littlenemo/BLOCK-MATH-SPEC.md), derived from
NVIDIA's reference implementation and verified against hand-computed fixtures.

### Mamba-2 forward (27 layers)

- `in_proj` 4480 to 22656, no bias, split into gate `z` (10240), conv input `u`
  (12288) and raw `dt` (128).
- Causal conv1d, window 4, then SiLU; `u` splits into `x` (10240), `B` (1024) and
  `C` (1024), per group: 8 groups of 128 broadcast over 128 heads.
- SSM decode step per head (128 heads, head dim 80, state 128): `dt =
  softplus(dt_raw + dt_bias)`, `A = -exp(A_log)`, `state *= exp(dt*A)`, `state +=
  dt*B*x`, `y = C . state`, `y += D*x`. There is no clamp on `dt` for this
  checkpoint (a correction to the architecture study; 0.001 and 0.1 are
  initialization constants, not an inference clamp).
- Gated RMSNorm applied after the gate (`norm_before_gate = False`), 8 groups of
  1280, then `out_proj` 10240 to 4480.

### MLP and attention forward

- MLP (25 layers): `down(relu2(up(x)))`, up 4480 to 15680, relu2 = `relu(v)^2`.
- Attention (4 layers): q 4480 to 5120 (40 heads x 128), k and v to 1024 (8 KV
  heads x 128), repeated 5 times to the q heads, no position embeddings of any kind
  (NoPE), causal, scale `1/sqrt(128)`, KV cache in bf16 in the reference design.

## What is verified

| Item | Evidence |
|---|---|
| Checkpoint inspector and loader | tensor inventory matches the architecture study; fingerprints: layer 0 `A_log` FNV-1a `f73e4acba693588f`, layer 21 `q_proj` RMS 0.017538, FNV-1a `66a6639137e13b79` |
| Mamba-2 conv, SSM step, gated norm | 19/19 hand-computed checks |
| GQA attention decode | 19/19, including a closed-form proof of the head-to-KV-group mapping |
| Byte-level BPE tokenizer over `tokenizer.json` | 27/27 fixture checks including malformed-input refusals; the checkpoint's own pre-tokenizer (a Split step with the exact pattern, refused loudly if it differs) and `ignore_merges`; a seventy-case battery matches the reference token for token; `test_tokenizer_split` 5/5 (the two-space prompt `  a` encodes as `[1032, 1261]`) |
| Assembled forward | 3 tokens through the first 15 real layers (all three block types): hidden state finite, RMS 0.520647, bit-exact across a state reset, token-sensitive; 7/7 |
| Hardening | span and overflow guards at index time, 64-bit file seeks throughout (a 32-bit `long` would misread the 4.9 GB shards on Windows), geometry validation at the kernel doors, decode error channels, NaN that propagates; five crafted hostile checkpoints refused cleanly; silent under ASan and UBSan; a bf16 NaN canonicalization bug caught by review and fixed with a regression check |
| Whole-geometry validation | READY is printed only for a checkpoint that passes: every tensor the steps will fetch present, correctly shaped and typed; each shard's true file size is captured at open and a tensor whose span ends past it is refused (no READY for a truncated download) |
| Coherent state commitment | every step preflights its tensors before the first layer's state moves; a failure mid-commit invalidates the state until reset; the refused step is bit-identical to never attempting it |
| SIMD | the AVX2 kernel lives in its own translation unit (`src/nn_simd.c`, the only file compiled with `-mavx2`); entry-counter checks prove dispatch; synthetic 7/7, with the checkpoint 14/14; kernel drift against scalar exactly zero; multi-lane bit-identical; the scalar fallback contains no VEX instructions |
| Layer-major prefill | `test_prefill`: bit-identical to token-by-token stepping (chunk boundary exercised) |

## Layer-major prefill

`nemo_model_prefill(model, tokens, n, hidden_out)` runs up to `NEMO_PREFILL_CHUNK`
tokens through the stack one **layer** at a time: each layer's weights are fetched
once and applied to every token in order before moving to the next layer, instead of
re-fetching all 56 layers per token. Per-token math and the per-layer state order
are unchanged, so the result is bit-identical to repeated `nemo_model_step_hidden`
calls (the `test_prefill` suite checks exactly that, in two chunks to exercise the
boundary). This matters because on this machine the engine is bound by reading the
weights.

## The binaries

| Binary | Source | Purpose |
|---|---|---|
| `nemo-engine` | `src/engine.c` | the reference engine: protocol v2, greedy streaming generation, prompts encoded as is |
| `engine_gui` | `src/engine_gui.c` | the GUI adapter: a copy of `engine.c` with a thin `--chat` mode, `--think`, layer-major prefill and a few real v3 lines. `engine.c` itself is unchanged and stays the reference build |
| `nemo-inspect` | `src/nemo.c` | the checkpoint inspector (`nemo inspect <shard.safetensors>`: every tensor's name, dtype, shape, element count and byte span) |

`build.bat` (or `build.sh`) from `littlenemo/` builds `nemo-engine`, `nemo-inspect`
and the test suites into `bin/`. `engine_gui` is built by hand from the same
objects:

```
cd littlenemo
build.bat
gcc -O2 -std=c99 -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64 -static -o bin\engine_gui.exe src\engine_gui.c obj\forward.o obj\attn.o obj\mamba.o obj\nn.o obj\nn_simd.o obj\loader.o obj\json.o obj\pool.o obj\tokenizer.o -lm -lpsapi
```

### Engine flags

```
nemo-engine <checkpoint_dir> <tokenizer.json> [options]
engine_gui  <checkpoint_dir> <tokenizer.json> [options]
```

| Flag | Meaning |
|---|---|
| `--mode minimum\|medium\|max\|ultra` | pool lane count from the queried core count: minimum = 1 lane, medium = half the cores, max = all but one (the INI default), ultra = every core; an unknown mode is a startup error |
| `--threads N` | exact lane count; the READY mode then reads `custom` |
| `--simd` / `--no-simd` | opt into the AVX2 projection kernels (scalar is the golden reference and the default); reached only through a runtime CPUID check |
| `--max-new N` | cap generated tokens per turn |
| `--max-pos N` | cap the context |
| `--layers N` | truncate the stack (plumbing tests only, not a model output) |
| `--serve-stdio` | the default transport, named explicitly |
| `--serve PORT` | accept one TCP client (1 to 65535) and speak the same protocol over the socket; POSIX only for now (the Windows input shims cannot poll a socket) |
| `--chat` (`engine_gui`) | wrap each PROMPT in the checkpoint's own chat template (`tokenizer_config.json`, Nemotron-H: `<SPECIAL_10>System`, `<SPECIAL_11>User` / `Assistant`, `<SPECIAL_12>`) and keep the conversation in the model state across turns |
| `--think` (`engine_gui`) | reasoning on by default; per turn `/think` and `/no_think` in the user text override it; the default is no-think (answers only) |

The CUDA build of the same adapter (`nemo-engine-cuda`, [ACCEL-CUDA.md](ACCEL-CUDA.md)) accepts every flag above plus `--cuda-pack`, `--accel`, `--expand`, `--full-path`, `--host-threads`, `--tok-overlap`, `--lm-head`, `--prefill-chunk` and the `--dtc-*` options, which the CPU engines reject as unknown arguments (the chat window passes them only to an engine whose file name contains `cuda`).

`engine_gui` computes on the CPU (no GPU lane is linked), reports `accel=cpu` in
READY, measures `PREFILL` and the added `STATS` keys with the same monotonic clock,
reports `MEM` as peak working set, and splits `<think>`/`</think>` into `THINK`
and `TOK` streams. Nothing is estimated beyond the existing, labeled `gbs`.

### The protocol

The engine speaks a line protocol (v2, plus the backward-compatible v3
extensions) on stdin and stdout, or over TCP. Every message type is documented in
[littlenemo/PROTOCOL.md](../littlenemo/PROTOCOL.md) and indexed in [GUI.md](GUI.md).

### The engine pool

The engine pool is the zero-mutex pool from `nq2-handshake.h`: it splits each large
projection across worker lanes while the next layer is prepared, with results
bit-identical to the single-lane path ([ZERO-MUTEX.md](ZERO-MUTEX.md)).

## Honest status

- The full checkpoint is downloaded, validated, and the engine prints READY on it.
  In the 2026-11-26 bench session the real engine held a chat and a 10-question quiz
  scored 10 of 10 (the exact prompts, grading rules and runner are in [LITTLE-NEMO-QUIZ.md](LITTLE-NEMO-QUIZ.md)). A tokenizer-based evaluation over a fixed passage (teacher-forced top-1/top-5, first divergence, perplexity, and decoded-text agreement) is in [LITTLE-NEMO-TOKENEVAL.md](LITTLE-NEMO-TOKENEVAL.md). The earlier
  raw-scaffold battery in [LITTLE-NEMO.md](../LITTLE-NEMO.md) recorded France and
  Japan passed.
- It runs the raw bf16 weights on the CPU at about **0.07 tokens/s**: each token
  reads the whole raw checkpoint (roughly 18 GB for 9 billion parameters) from disk
  on a machine with 16 GB of RAM. Layer-major prefill helps the prompt, not
  generation.
- Nothing has yet shown that compression keeps the model's intelligence. When the
  NTSE-o2 packed weight source plugs into the same fetch points, the behavioral
  battery runs the same prompts three ways (raw, packed, packed with dictionary
  compute) and scores token-for-token agreement, first divergence and final-answer
  agreement, hash-chained per PCHIE discipline. See [ROADMAP.md](ROADMAP.md).
- A Little Nemo accel adapter exists (`accel/littlenemo/`); it is PCIe-bound
  ([ACCEL.md](ACCEL.md)).
- A CUDA lane runs the compressed pack resident on the GPU at about 20 to 22 tokens/s
  (DTC; 16 to 17 expanded) and scored **7 of 10** on the same quiz the raw CPU engine scored 10 of 10 on;
  the loss comes from the pack, not the GPU. See [ACCEL-CUDA.md](ACCEL-CUDA.md).
- Little Nemo is text-only; image and video attachments are refused by the GUI before
  anything is sent.

## Reviews this engine has been through

The 2026-11-24 reviews by Gizmo and Archie (whole-geometry validation, coherent
state commitment, and the false SIMD verification), and Codex (Astra 6.2) on
2026-11-25 (the tokenizer's pre-tokenizer, an instruction-set-clean scalar
fallback, committing a capped turn's final token with the hidden-only step, and
shard-size capture). Details are in [LITTLE-NEMO.md](../LITTLE-NEMO.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| the engine exits before READY, naming a tensor | whole-geometry validation failed: the checkpoint is incomplete or the wrong model |
| READY never appears and the window shows an error | check `checkpoint` and `tokenizer` paths in `littlenemo.ini`; the engine prints the reason on stderr |
| answers wrapped in nothing, or the model rambles | you are running `nemo-engine` (no chat template); use `engine_gui --chat` |
| `unknown option` | flags go after the two positional arguments; `--serve PORT` needs a value |
| the first reply takes a long time | prefill and generation read the whole checkpoint from disk; expect about 0.07 tokens/s |
| `--serve` fails on Windows | POSIX only for now; use the default stdio transport |