# Inference and streaming tools

Every tool here is pure C99, takes only a C compiler and the C library to build,
and prints what it did. All flags were checked against the sources in `tools/`
(`nq2-infer.c`, `nq2-infer-lab.c`, `nq2-infer-triforce.c`, `nq2-run.c`,
`nq2-streamtest.c`, `nq2-logitcmp.c`). Build with `build.bat` or `sh build.sh`;
the threaded tools (`nq2-infer`, `nq2-infer-lab`, `nq2-infer-triforce`,
`nq2-infer-triforce-dbg`, `nq2-streamtest`) need a POSIX-thread MinGW on Windows.

**What the engines run.** The three engines run NVIDIA Nemotron-Labs-Sigma-3B,
a continuous-diffusion language model, either from its raw safetensors or from a
pack. Sigma ships no tokenizer in the tested download: token IDs in, token IDs
out.

## The three engines

All three take the same run-shaping flags.

| Flag | Default | Meaning |
|---|---|---|
| `--dir DIR` | required | model directory; every `.safetensors` in it, sorted so shard order matches the quantizer's packing order |
| `--packed STREAM BOOK` | off (raw) | run from a pack instead of the raw shards |
| `--ids 1,25,882` | required | prompt token IDs |
| `--tokens N` | 64 | maximum new tokens |
| `--steps N` | 32 | diffusion steps (at least 1) |
| `--block N` | 32 | diffusion block size (1 to 4096; larger is clamped with a warning) |
| `--sampler ddim\|ddpm` | ddim | sampler |
| `--schedule log_snr\|edm_sigma` | log_snr | noise schedule |
| `--guidance F` | 3.0 | guidance scale |
| `--score-temp F` | 0.3 | score temperature |
| `--seed N` | 1234 | seed (deterministic: the same seed reproduces the same tokens) |
| `--threads N` | 0 (auto) | worker threads |

An unknown or incomplete option is a named refusal with exit 1.

### `nq2-infer`: the plain engine

The reference engine: the verified path the pinned anchors were taken on.

```
nq2-infer --dir <sigma-dir> --packed sigma-full-b256-v2run.nq2 sigma-full-b256-v2run.nq2book --ids 1 --tokens 1 --steps 1
```

### `nq2-infer-lab`: the instrumented engine

Everything `nq2-infer` does, plus the instruments used for the fidelity studies:

| Flag | Meaning |
|---|---|
| `--forensics` | per-ordinal stall forensics (misses only) |
| `--logits-dump FILE` | write the first denoise evaluation's conditional logits (a `LOG1` block) |
| `--logits-dump-all FILE` | append one `LOG1` block per denoise evaluation, to watch reconstruction error amplify or settle across the diffusion trajectory |
| `--restore-tensors a,b` | read the named tensors from the raw shards and everything else from the pack: single-tensor restore for per-tensor ablation (Lane 4 used this) |

### `nq2-infer-triforce`: the split pipeline engine

Mega Tensor's engine: one loop stages raw sections from disc, expansion lanes
reconstruct weights, and the consumer takes tensors in the model's fixed
per-layer order. Adds `--forensics`, `--logits-dump`, `--logits-dump-all` and the
pipeline geometry, which is a runtime choice, never a constant:

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--pipe-slots N` | `NQ2_PIPE_SLOTS` | 2 (device-derived) | tensors that may be in flight (at least 2) |
| `--fetch-ahead N` | `NQ2_FETCH_AHEAD` | 1 (device-derived) | how far the fetcher may run ahead of demand (at least 1) |
| `--prefetch-dist N` | `NQ2_PREFETCH_DIST` | 2 (device-derived) | prefetch distance, 0 or more |
| `--expand-lanes N` | `NQ2_EXPAND_LANES` | 2 (device-derived) | simultaneous expansions, 1 to 64: the residency dial |

CLI flags override environment variables, which override device-derived
defaults. Values are range-checked against the hardware ceiling and clamped
loudly with the reason printed; the startup line prints every effective value
with its source (`cli`, `env` or `device`). The Triforce refuses PQ packs at
open, in plain words. See [TENSOR-HANDSHAKE.md](TENSOR-HANDSHAKE.md).

`nq2-infer-triforce-dbg` is the serial debug build (workers bypassed; the
consumer loads everything itself), kept for bisecting delivery faults: it is how
the 2026-11-24 race was convicted ([PIPELINE](../PIPELINE-2026-11-24.md)).

### Pinned anchors

A change to any engine is judged against these. They must reproduce exactly.

| Anchor | Value |
|---|---|
| Little Nemo hidden-state RMS (3 tokens, 15 layers) | 0.520647 |
| `nq2-streamtest` accumulator over the whole Sigma pack | 1394.1117661067326, bit-identical in every configuration |
| Sigma 1-step anchor, raw | first token 75768 |
| Sigma 1-step anchor, packed | first token 122388 |
| Expansion checksum of the shipped Sigma pack (`nq2-run`) | `7bd9557531df1ab7` |

The packed-engine token anchors were taken with `--ids 1 --tokens 1 --steps 1`;
the receipts in `results/` record the exact recipe of each run. The 32-token
behavioral recipe used by the fidelity battery is `--ids 1 --tokens 32 --steps 1
--guidance 1 --score-temp 1 --seed 42`.

## `nq2-run`: the streaming runner

```
nq2-run model.nq2 model.nq2book            stream the pack and report
nq2-run model.nq2 model.nq2book peek N     expand only block N
```

Streams a pack from disc through a rolling window (megatexture style) and
reports packed against expanded size, peak resident memory, expansion throughput
and a checksum of the full expansion. Expansion is exact: the same index always
calls back the same prototype, so the streamed checksum equals a full
expansion's, at a fraction of the RAM. Measured on the throttled bench: 922.7
MB/s expanded. It refuses v3 packs that use PQ, per-tensor block sizes or a raw
float16 section ("nq2-infer and nq2-infer-lab read it"), rather than misreading
them.

## `nq2-streamtest`: the four-loop harness

```
nq2-streamtest --pack S.nq2 --book S.nq2book --shards DIR [--unpackers N] [--lookahead N] [--prefetch-dist N] [--layers N] [--variant N]
```

JD's four loops (2026-11-24): the main loop does the tensor math, a disc loop
does nothing but lookups, and two unpack loops (`--unpackers`, 0 is the
sequential reference) hand finished tensors to the main loop, which consumes in
one fixed serial order no matter which loop finishes first. Measured on the
shared 1.5-core bench (floor numbers) streaming the full Sigma pack:

| Configuration | Time | Rate |
|---|---|---|
| Sequential, one loop | 33.49 s | 90.4 M weights/s |
| Two unpackers, lookahead 8 | 20.56 s | 147.2 M weights/s |
| Row-pointer table, prefetch distance 2 | 17.93 s | 168.8 M weights/s |

The accumulator is 1394.1117661067326 in every row: concurrency changes the
speed, never the sum. A pack shorter than its header promises fails up front, a
short read stops the pipeline, and the exit is nonzero with no accumulator
published (regression: `tools/test-streamtest-truncation.sh`).

## `nq2-logitcmp`: the behavioral comparator

```
nq2-logitcmp [--allow-partial] raw.logits packed.logits
```

Compares two logits dumps written by `--logits-dump` or `--logits-dump-all`, one
from the raw path and one from the packed path over identical token IDs, seed
and schedule. Per position and per evaluation it reports top-1 agreement, top-5
overlap, cosine similarity, mean absolute logit difference, relative logit L2
and the KL divergence KL(raw || packed) of the softmax distributions, in double
precision and with no tokenizer anywhere: drift is measured in the model's own
output space. It is strict by default (mismatched dumps are refused);
`--allow-partial` accepts a partial comparison. Sample recipe and results:
[ACCEL.md](ACCEL.md) (Lane 4) and `results/behavioral-gate-2026-11-25.txt`.

## Dictionary compute probes

`nq2-dc` and `nq2-dc-adv` live in [DTC.md](DTC.md). The Precision Gate's engine is
in [PRECISION-GATE.md](PRECISION-GATE.md).

## `nq2-gui.py`: the console

A small Python front end for the C tools (a GUI only, by JD's ruling: it does no
model work; every action is a C program called through its command line). It
opens a pack, shows its vital signs, stream-verifies the full expansion with
`nq2-run` (sizes, peak resident memory, throughput, the FNV-1a checksum) and
peeks at any block. It does not chat; chat is [GUI.md](GUI.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pack identity mismatch: the stream carries generation ... and the book carries generation ...` | the `.nq2` and `.nq2book` were not packed together; use the pair from one run |
| `stream and book disagree` | block or entry count differ between the files |
| `unsupported pack geometry` | block above 4096, or entries outside 1 to 256, in a header |
| `this pack uses format v3 options ... that the streaming runner does not expand` | PQ, budgets or fp16-raw pack: use `nq2-infer` or `nq2-infer-lab` |
| Triforce refuses a pack at open | PQ packs are a recorded limit of the Triforce |
| `no safetensors in DIR` | `--dir` must hold the `.safetensors` shards |
| a clamp message on pipeline values | the request passed the hardware ceiling; read the printed reason and source |
| a failed build of a threaded tool on Windows | install a POSIX-thread MinGW (winpthreads) |