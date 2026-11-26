# Little Nemo on CUDA: calibration, DTC by default, predict and expand

Little Nemo can run its projections on the NVIDIA GPU from an NTSE `.nq2` v3 product-quantized pack, with the host side in pure C. This guide covers the lane, the startup calibration, the DTC and EXPAND rules, every flag, the windowless engine, and the measured speed and quiz results. Source: `accel/` (`nemo_gpu.c`/`.h`, `nemo_gpu_kernels.h`, `nemo_calib.c`/`.h`, `nemo_predict.c`/`.h`, `mk_engine_gui_cuda.py`, `build-nemo-gpu.ps1`, `test_nemo_gpu_layers.c`). Implementation: Cortana.

**Status:** this lane landed on 2026-11-26. Source is in `accel/` (`nemo_gpu`, `nemo_calib`, `nemo_predict`, `mk_engine_gui_cuda.py`, `build-nemo-gpu.ps1`, `test_nemo_gpu_layers.c`). The numbers below are from one bench session on one machine (RTX 3050 Ti Laptop, 4 GB). They are measurements, not guarantees, and the quiz is a ten-question smoke test, not a benchmark.

## The honest result first

| Path | Speed | Quiz (10 short questions, greedy, thinking off) |
|---|---|---|
| Raw bf16 weights on CPU | about 0.07 tokens/s | **10/10**, clean stops |
| Compressed pack on CUDA, DTC only (`--expand never`) | 15.50 to 15.81 tokens/s | 7 right, 2 hedged, 1 wrong |
| Compressed pack on CUDA, auto (the default) | 5.17 to 11.24 tokens/s per question (about 12.8 tokens/s in a GUI chat) | **7/10**: 7 pass, 2 hedges, 1 fail |
| Compressed pack on CUDA, full expansion (`--expand always`) | 7.86 to 7.94 tokens/s | the same answers as DTC only, apart from small differences after the answer |

So the compressed pack is about 200 times faster than the raw CPU path, and it is **less accurate**: 7 of 10 against 10 of 10. The loss comes from the compression, not from the GPU. The pack is a 4-dimensional product quantization at 2.2169 bits per weight on the projections, and its projection relative L2 error against the bf16 weights is about 0.32. The CPU reference run on the decoded pack gives the same wrong answer ("Earth" for the largest planet), the same hedges, and the same rambling. Full expansion recovers the pack's exact decode, not the original weights, so it cannot fix what compression removed. This is the fidelity problem [ROADMAP.md](ROADMAP.md) is about, and the Lane 4 results in [ACCEL.md](ACCEL.md) say the same thing.

### The quiz

Chat template on, thinking off, max 16 new tokens, a fresh conversation per question, the same prompts as the raw-CPU quiz. Graded by the first answer line.

| # | Prompt (abridged) | Auto output, first line | Grade |
|---|---|---|---|
| 1 | Sun is a star, one word | `True` then "or False" | hedge (right word first) |
| 2 | spiders have six legs | `False.` | pass |
| 3 | whales are mammals | `True` then "or False" | hedge (right word first) |
| 4 | gold symbol is Ag | `False.` | pass |
| 5 | capital of Texas | Austin | pass |
| 6 | capital of New York State | Albany | pass |
| 7 | capital of California | Sacramento | pass |
| 8 | capital of Ohio | Columbus | pass |
| 9 | 7 times 8 | 56 | pass |
| 10 | largest planet | "Earth" | **fail** (Jupiter) |

Raw CPU bf16 answered all ten correctly and stopped cleanly; the compressed pack keeps going after the answer ("(This is a short sentence.)") on most questions.

Agreement between modes, over the same ten questions:

- auto matched DTC-only token for token on all 10 (132 of 132 tokens).
- auto matched full expansion exactly on 8 of 10; the other two differ only after the answer. First-answer-line agreement with full expansion was 10 of 10 (token-prefix agreement 118 of 132).
- Of 144 step decisions, 37 (26%) were expanded: 35 on margin and 2 on entropy. Full expansion picked the same top-1 in 36 of them; once, on a near tie (margin 0.003), it picked a different token with no effect on any emitted answer.
- Full expansion on the GPU matches the CPU on the decoded pack exactly for questions 3 to 10; questions 1 and 2 differ only after the answer, from a different fp32 accumulation order.

Not run: the five essays, because the gate was above 50 tokens/s and the best measured rate was 15.8.

## Lane selection: the calibrate function

At startup the engine **calibrates**: it probes every lane the machine might offer, checks each for correctness and speed, and picks the fastest correct one, preferring CUDA. The CPU is chosen only when no GPU lane passes. The function never fails; at worst the lane is `cpu`.

| Lane | What is checked | Result on the bench machine |
|---|---|---|
| cuda | the driver and NVRTC load, VRAM free, the DTC and FULL kernels bit-exact against a C emulation on synthetic PQ tensors, then an in_proj-sized GEMV and a memory read are timed | **pass, chosen**: RTX 3050 Ti Laptop (sm_86), driver 13050, NVRTC 13.4, 3296 MiB free; DTC 137.4 Gw/s, FULL 64.5 Gw/s, device read 176.5 GB/s; estimate about 15.3 tokens/s |
| opencl-nv | OpenCL platform and device probe (NVIDIA) | rejected: device found (4096 MiB) but this build has no OpenCL Nemotron forward |
| opencl-igpu | OpenCL probe (Intel/AMD or unified-memory GPU) | rejected: Intel UHD found, same reason |
| cpu | always correct (it is the reference); speed from a single-thread memory-read micro-bench | pass (fallback): 19.4 GB/s read, estimate about 1.08 tokens/s |

CUDA is rejected outright if no `.nq2` pack is given (`--cuda-pack`), because there is nothing to run on the GPU. The verdict is cached in a small text file keyed by GPU name, driver, NVRTC version and pack, so a later launch skips the tests (31 ms for a cached launch on the bench machine; a fresh calibration runs once). The cache is `%LOCALAPPDATA%\NemoChat\nemo-calib.txt` on Windows; `NEMO_CALIB_FILE` overrides the path. Delete the file or pass `--recalibrate` to redo it. The GUI shows `Accel lane: CUDA` and the per-lane verdicts and reasons as engine stats.

### How the CUDA lane runs

- Pure C host. The only CUDA code is in `accel/`; the CUDA driver API and NVRTC are loaded at run time (no CUDA headers or SDK link), kernels compiled for the card's architecture with `--fmad=false`, `--ftz=false`, exact division and square root.
- Zero mutexes, like the rest of the engine ([ZERO-MUTEX.md](ZERO-MUTEX.md)).
- The pack stays resident in VRAM: 2914 MiB total on the bench machine (pack about 1.97 GB, int8 `lm_head` 560 MiB, fp32 books about 88 MB and snapshots about 141 MB for EXPAND). Engine RAM peak about 305 MB. Embeddings are fp16; the `lm_head` is int8 per row on the GPU (logits relative L2 about 2e-2).
- The raw bf16 weights (about 18 GB) cannot be resident in 4 GB, which is why this lane runs from the pack.

## DTC by default, predict decides EXPAND

- **DTC** is the fast fp16-book PQ kernel, about 15.7 tokens/s. It is bit-exact against its own C emulation.
- **EXPAND** (the FULL kernel) is the exact fp32-book, bf16-faithful decode of the same pack, about 7.9 tokens/s. A step is redone from a snapshot of the SSM, convolution and KV state.

The predict rules (`nemo_predict.c`) make two decisions, and every decision goes out as a `DTC ...` or `EXPAND ... reason=...` protocol line and on stderr:

| Level | Rule (default) | Flag |
|---|---|---|
| turn | expand the whole turn if the prompt is longer than 96 tokens | `--dtc-prompt-tokens N` |
| turn | expand if the user asked for reasoning (`/think`) and long-form is on | `--dtc-longform 0\|1` (default 1) |
| turn | expand if the text contains a long-form keyword (essay, write a, story, poem, explain, describe, summarize, compare, step by step, prove, derive, in detail, outline, how does, why does, pros and cons, plan for, and a few more) | `--dtc-longform 0\|1` |
| turn | expand if the complexity score is 6 or more (sentences beyond the first, half the commas, numbers beyond two, operators) | `--dtc-complexity N` |
| step | redo one step at full precision if the DTC logits' top-1 margin is below 0.5 | `--dtc-margin X` |
| step | redo one step at full precision if the softmax entropy is above 4.0 nats | `--dtc-entropy X` |

All ten quiz prompts classified as short-factual (13 to 17 tokens, complexity 1), so they ran in DTC at the turn level, and only the step rule fired. On the capitals, 7 times 8 and the planet question the whole answer line ran in DTC and expansions fired only in the rambling after it. On questions 2 and 4 the True/False token itself had a low margin and was expanded, and full expansion gave the same answer. The GUI counts the lines as "Expansions this session" and "DTC ops this session".

## Flags

The CUDA engines accept every `engine_gui` flag ([LITTLE-NEMO-ENGINE.md](LITTLE-NEMO-ENGINE.md#engine-flags)) plus:

| Flag | Meaning | Default |
|---|---|---|
| `--cuda-pack PACK.nq2` | the NTSE pack the CUDA lane runs from; without it CUDA is rejected | none |
| `--accel auto\|cuda\|cpu` | override the calibrated choice; the cache is still read and written for the record; anything else is a startup error | `auto` |
| `--recalibrate` | ignore the cache and re-run the tests | off |
| `--expand auto\|never\|always` | `auto`: the rules above; `never`: DTC only; `always`: full expansion | `auto` |
| `--dtc-margin X` | step: top-1 margin below this expands | 0.5 |
| `--dtc-entropy X` | step: entropy (nats) above this expands | 4.0 |
| `--dtc-prompt-tokens N` | turn: a prompt longer than this expands | 96 |
| `--dtc-complexity N` | turn: a complexity score at or above this expands | 6 |
| `--dtc-longform 0\|1` | turn: long-form keywords and `/think` expand | 1 |
| `--dtc-trace` | print every step's confidence on stderr | off |
| `--log FILE` | append stderr to FILE (the windowless engine has no console) | off |

Developer aids, read from the environment: `NEMO_CALIB_FILE` (cache path), `NEMO_GPU_NO_EXPAND=1` (do not build the EXPAND resources, saving VRAM), `NEMO_GPU_EXACT=1` and `NEMO_GPU_BOOK` (book precision for the kernels), `NEMO_GPU_HEAD`, `NEMO_GPU_PROFILE=1` (synchronize around every launch and time it), `NEMO_GPU_DBG`, `NEMO_GPU_DUMP` (write the compiled kernel image).

## The binaries and building

| Binary | Notes |
|---|---|
| `accel\bin\nemo-engine-cuda.exe` | the console engine (generated into `accel/engine_gui_cuda.c` from `engine_gui.c` by `mk_engine_gui_cuda.py`, linked with the GPU, calibration and predict objects; the generated file is a build product and is not committed) |
| `accel\bin\nemo-engine-cuda-w.exe` | the same code linked with `-mwindows`: no console window pops up. Use it from a desktop shortcut or the GUI, with `--log FILE` to keep a log |
| `accel\bin\test_nemo_gpu_layers.exe` | per-layer verification of the GPU path against the CPU reference |

```
powershell -File accel\build-nemo-gpu.ps1        # from the repository root, MSYS2 UCRT64 gcc and Python on PATH
```

The script writes new binaries only and rebuilds nothing prior. Example launch, as the desktop shortcut runs it:

```
accel\bin\nemo-engine-cuda-w.exe <checkpoint_dir> <tokenizer.json> --chat --cuda-pack <pack.nq2> --accel auto --log nemo-engine-cuda.log
```

Point the GUI at it with `--engine` ([GUI.md](GUI.md)). Run logs are private and are not part of the repository.

## Verification

| Check | Result |
|---|---|
| DTC and FULL kernels vs the C emulation (calibration self-test) | bit-exact |
| Per-layer, 56 layers, 2 tokens, vs the bf16-rounded CPU reference, FULL | residual relative L2 up to 3.0e-4, pass |
| Per-layer, same, DTC | up to 1.3e-2, so it fails the 1e-3 tolerance **by design**: DTC skips the reference's bf16 rounding of the weights; argmax agrees on 2 of 2 |
| int8 `lm_head` | logits relative L2 about 2e-2 |

## Limits

- The accuracy gap above is real: on this pack the CUDA lane is fast and wrong more often than the raw model. Do not read 15.7 tokens/s as "Little Nemo at that speed with the original quality".
- One machine, one pack, ten short questions, greedy decoding, 16 new tokens. No essay or reasoning evaluation was run.
- The OpenCL lanes have no Nemotron forward in this build and are rejected by calibration.
- The GPU needs a pack of about 2 GB plus the `lm_head` and EXPAND resources in VRAM; on a smaller card calibration rejects CUDA and the engine falls back to the CPU lane.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| lane is `cpu` and the CUDA reason says "no .nq2 pack given" | pass `--cuda-pack` |
| lane is `cpu` after changing GPU, driver or pack | the cache is keyed on those; run with `--recalibrate` if a stale verdict is suspected |
| a rejected CUDA lane with a driver or NVRTC reason | the NVIDIA driver provides the CUDA driver library; NVRTC is loaded from the CUDA toolkit (see `NTSE_NVRTC_DLL` and `CUDA_PATH` in [ACCEL.md](ACCEL.md)) |
| not enough VRAM | set `NEMO_GPU_NO_EXPAND=1` to drop the EXPAND resources, or use a smaller pack |
| the engine seems to do nothing with the windowless exe | it has no console: add `--log FILE` and read the file |
| answers wander after the correct line | expected on this pack (see the honest result); use stop and the first line |