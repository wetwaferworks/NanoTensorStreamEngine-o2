# Accel lanes: CUDA, OpenCL on NVIDIA, OpenCL on the iGPU

`accel/` holds NTSE's GPU lanes: Dense Tensor Compute and the bf16 projection on
the graphics hardware, written so the **CPU path stays the reference and the
official path**. The lanes are an accelerator for it, not a replacement. The pure
C build (`build.bat`, `tools/`, `littlenemo/`) contains no accel code; nothing
outside `accel/` depends on it. Direction and idea: JD. Lane proposals and the
measurement list: Gizmo. NTSE, nq2 and Little Nemo: Elfie. Co-implemented by
Cortana. The full lab notebook, with every receipt, is
[accel/README.md](../accel/README.md); this guide is the map.

## The three lanes

| Lane | API | Device on the bench | Accumulate | Why this API |
|---|---|---|---|---|
| `cuda` | CUDA **driver API** (`nvcuda.dll`) + **NVRTC** (`nvrtc64_*.dll`), both loaded at run time | RTX 3050 Ti Laptop, sm_86, 4 GB | fp32 IEEE (DTC), fp64 (bf16 projection) | no MSVC on the workstation, so `nvcc` cannot host-compile; NVRTC compiles the kernels at open (cubin for the exact SM, PTX-JIT fallback). No CUDA SDK is needed to build |
| `opencl-gpu` | OpenCL 1.2+ (`OpenCL.dll`, run-time loaded), discrete-device filter | the same card through NVIDIA's OpenCL driver | fp32 IEEE / fp64 | the default GPU lane for NVIDIA without CUDA: no NVRTC DLL, a driver mismatch, or a policy reason to avoid CUDA |
| `opencl-igpu` | the same OpenCL backend, integrated filter | Intel UHD (i5-12450H), OpenCL 3.0 NEO, 48 CUs, no fp64 | fp32 IEEE (DTC), float-float (bf16) | unified memory, so the budget is capped at 512 MiB (it is system RAM) |

`auto` order: CUDA, then OpenCL-GPU, then OpenCL-iGPU, then CPU. OpenCL rather
than Vulkan for lanes 2 and 3 because Vulkan compute needs SPIR-V, producing it
needs a shader compiler that is not on the workstation, and hand-written SPIR-V
would be unreviewable; with `FP_CONTRACT OFF` plus `vload_half` OpenCL gives IEEE
results that match the CPU bit for bit. A Vulkan backend can slot into the same
ops vtable later.

## What runs on the GPU

| Op (`ntse_op`) | What it does |
|---|---|
| `dtc-eager` | full tables `T[j][k]` for every segment and code, then apply |
| `dtc-active` | tables only for the (segment, code) pairs the index uses; computed once at upload |
| `dtc-lazy` | tables on demand per call: a mark kernel flags used pairs, the build kernel skips the rest |
| `dtc-direct` | no table: each output gathers its prototypes |
| `expanded` | the reference: materialize `w = mn + sc * book[idx][i]` on the fly and run a fixed 256-wide tree GEMV |
| bf16 projection (`ntse_accel_linear_bf16`) | Little Nemo's projections; double accumulation (CUDA, OpenCL with fp64) or float-float (iGPU); measured correctly rounded |

**Exactness design.** DTC sums are pinned in the CPU kernel's order (ascending `i`
for tables and `A[j]`, ascending `j` for `y`); the GPU kernels keep that order per
output and forbid contraction (`__fadd_rn`/`__fmul_rn`, `--fmad=false`,
`FP_CONTRACT OFF`); fp16 frames convert with `cvt.f32.f16` or `vload_half`. Result:
**every DTC op is bit-identical to the CPU kernel on every lane.** Expanded uses a
tree order, so it differs from the CPU in low bits (relative L2 at most 1.63e-6
against the CPU) and is about 10 times closer to the fp64 oracle (9.3e-8 to 1.3e-7)
than the CPU's sequential sum.

## Safety: validation, budget, fallback

- **Every input is validated** before device work: geometry (non-zero, `cols %
  block == 0` for DTC, products capped at 2^40), every `idx < entries`, fp16 frames
  with exponent 31 (Inf or NaN) rejected, book finite. Bad input is `NTSE_E_INVAL`
  on every lane, the CPU one included.
- **Every API return is checked.** A device error is a sticky fault: the context
  logs once and runs everything on the CPU reference from then on
  (`ntse_accel_failed()`).
- **Bounded memory.** Budget = min(75 percent of free or 50 percent of total, hard
  cap 3 GiB discrete or 512 MiB integrated); on the 4 GB card the derived budget is
  2471 MiB. Larger requests are clamped loudly. Every allocation is charged.
- `BUDGET` and `UNSUPPORTED` are per-call refusals: that call runs on the CPU and
  the context stays on the GPU. Unsupported: v3 PQ tensors, `cols % block != 0`.
- A missing driver or DLL opens on the CPU lane and says why.

Return codes: `NTSE_E_INVAL` (-1), `UNSUPPORTED` (-2), `NOMEM` (-3), `BUDGET`
(-4), `DEVICE` (-5), `NODEV` (-6), `STATE` (-7, an ownership violation refused),
`NONFINITE` (-8).

## Environment

| Variable | Effect |
|---|---|
| `NTSE_ACCEL_LANE` | `cpu`, `cuda`, `igpu` or `opencl-igpu`, `gpu-cl` or `opencl-gpu`, `auto`; an unknown value is reported and ignored |
| `NTSE_ACCEL_DISABLE=1` | force the CPU reference (any value except empty or `0`) |
| `NTSE_ACCEL_VRAM_MB` | lower the device budget (larger values are clamped) |
| `NTSE_ACCEL_PIN_MB` | pinned image arena budget for the stream plan (default 1 GiB, clamp 4 GiB) |
| `NTSE_CUDA_DEVICE`, `NTSE_CL_DEVICE` | pick a device |
| `NTSE_NVRTC_DLL`, `CUDA_PATH` | where to find `nvrtc64_*.dll`; without it the CUDA lane reports NODEV and `auto` moves on to OpenCL |

## Build and run

```
accel\build-accel.bat          (Windows, MinGW gcc, from the repo root)
sh accel/build-accel.sh        (POSIX gcc or clang, from the repo root)
```

Flags `-O2 -std=c99 -Wall -Wextra [-static] -D_FILE_OFFSET_BITS=64`; zero warnings
under MinGW gcc 16.2.0 and Linux gcc 14. It builds only under `accel/bin` and
`accel/obj`. The suites are in [TESTING.md](TESTING.md). The benchmark:

```
ntse-accel-bench [--lane cuda|igpu|gpu-cl|cpu] [--reps N] [--cpu] [--sections abcdsp] [--pack v1.nq2 v1.nq2book shard...]
```

| Section | Measures |
|---|---|
| a | operators on model-shaped tensors |
| b | the break-even map over output rows M |
| c | the activation batch sweep |
| d | the dictionary size and block sweep |
| s | streaming (Tensor Handshake) against resident, slots 1 to 3 |
| p | a full pass over every packed 2-D tensor of a real v1 pack: the linear-layer ceiling |

Output lines starting `R<TAB>` are TSV receipts. `accel/tools/run-bench-gated.ps1`
is the gated, detached runner.

## Results (2026-11-26, ET)

Hardware: RTX 3050 Ti Laptop 4 GB, driver 621.24 (CUDA 13.5, NVRTC 13.4); Intel
UHD (i5-12450H 8C/12T); 16 GB RAM; MSYS2 gcc 16.2.0 `-O2 -std=c99 -static`. Pack:
`sigma-full-b256-v2run`, stream 74,883,852 bytes, book 262,156 bytes. This is
Sigma-3B, not Super VL (which does not fit a 4 GB card). Every timing is end to end
per call (activation upload, kernels, result download). All runs exited 0, with 0
sticky faults and 0 ownership violations.

### Test suites (per lane)

| Suite | cuda | opencl-gpu | opencl-igpu |
|---|---|---|---|
| kernels (shapes x ops, determinism) | pass | pass | pass |
| stream (slots 1 to 3, inline and thread, early-recycle refusal) | 42/42 | 42/42 | 42/42 |
| real Sigma tensors (layer 0, 7 projections x ops) | pass | pass | pass |
| bf16 projection (10 shapes incl. cancellation, refusals, budget) | 47/47 | 47/47 | 47/47 |
| Little Nemo forward (15 layers, 3 tokens) | 10/10 | 10/10 | 10/10 |
| plan | 17/17 | 17/17 | 17/17 |
| guard (lane-independent) | 37 | | |

After Lane 1 added the lazy op the per-lane counts were kernels 43, real 56, plan
17 and guard 37 (`accel/PROGRESS.md`); the earlier receipts table in the lab
notebook records 36, 49, 17 and 36 before it.

### Full Sigma pass (section p): linear-layer ceiling

183 packed 2-D tensors, 3.429 G weights, embeddings excluded. The linear-layer
ceiling is 1 / (time for one matvec through every projection). **This is not
end-to-end decode**: TTFT, prompt tok/s and generation tok/s are N/A because there
is no GPU inference runtime yet.

| Lane | Mode | ms/pass | Linear tok/s ceiling | H2D bytes/token | Divergence vs CPU |
|---|---|---|---|---|---|
| CPU reference (1 thread) | dtc-eager | 105.335 | 9.49 | n/a | reference |
| cuda | resident dtc-eager | 12.167 | **82.19** | 2,994,176 (x only) | 0 values differ |
| cuda | resident dtc-active | 12.560 | 79.62 | 2,994,176 | 0 |
| cuda | resident dtc-direct | 114.033 | 8.77 | 2,994,176 | 0 |
| cuda | old stream, 2 slots | 117 | 8.55 | 168,949,504 | 0 |
| opencl-gpu | resident dtc-eager | 14.337 | 69.75 | 2,994,176 | 0 |
| opencl-gpu | resident dtc-active / direct | 14.924 / 96.742 | 67.00 / 10.34 | 2,994,176 | 0 |
| opencl-igpu | resident dtc-eager | 54.221 | 18.44 | 2,994,176 | 0 |
| opencl-igpu | resident dtc-active / direct | 54.290 / 927.072 | 18.42 / 1.08 | 2,994,176 | 0 |

Peak device memory 166.5 MB charged for the resident pack. Speedups, resident
dtc-eager against the single-thread CPU: CUDA 8.7x, OpenCL-NVIDIA 7.3x, Intel iGPU
1.9x.

### Lane 1: eager against active against lazy (warm runs, full pack)

| Lane | eager | active | lazy | direct | lazy vs best |
|---|---|---|---|---|---|
| cuda resident | 12.635 ms / 79.14 tok/s | 12.842 / 77.87 | 13.620 / 73.42 | 112.982 / 8.85 | -7 percent |
| cuda plan, pinned, 2 slots | 12.818 to 13.027 / 76.8 to 78.0 | 12.712 to 14.516 / 68.9 to 78.7 | 15.116 to 15.244 / 65.6 to 66.2 | n/a | -15 percent |
| opencl-gpu resident | 14.786 / 67.63 | 14.930 / 66.98 | 17.938 / 55.75 | 95.368 / 10.49 | -18 percent |
| opencl-igpu resident | 54.493 / 18.35 | 54.275 / 18.42 | 68.563 / 14.59 | 927.129 / 1.08 | -21 percent |

Verdict, negative for lazy: with a static index lazy builds exactly the entries
active builds and also pays a flag clear plus a mark pass over every block on every
call. It loses 7 to 21 percent. It would only pay when the work set is not known
ahead of time (an index that changes per call). The GPU "lazy" is deliberately not a
spin-wait memo (that needs intra-kernel waiting, which can deadlock without
independent thread scheduling and is not portable to OpenCL); the mark pass uses
idempotent stores, so it is race-free without atomics.

### Lane 2: the prepared stream plan (Tensor Handshake GPU concurrency)

Streaming was the bottleneck: 7.97 tok/s streamed against 80.67 resident. A plan
(`ntse_accel_plan_create` / `_run`) fixed the two causes (the 512 KB book re-sent
with each of the 183 tensors, about 96 MB per token; and per-token CPU relayout):
layout once (121 ms for the whole pack), books resident once (deduplicated by
geometry and content: 1 book for Sigma), a pinned image arena (68.6 MB), and
1 to 4 buffering slots.

| Lane | Path | Slots | ms/pass | Linear tok/s ceiling | H2D bytes/token |
|---|---|---|---|---|---|
| cuda | resident (reference) | n/a | 12.396 | 80.67 | 2,994,176 |
| cuda | old stream | 2 | 125.547 | 7.97 | 168,949,504 |
| cuda | **plan, pinned image, eager** | 1 / **2** / 3 | 18.239 / **13.090** / 12.982 | 54.83 / **76.40** / 77.03 | 71,026,688 |
| cuda | plan, pinned, active | 1 / 2 / 3 | 17.669 / 12.712 / 13.110 | 56.60 / 78.66 / 76.28 | 71,026,688 |
| cuda | plan, pageable, eager | 1 / 2 / 3 | 23.864 / 17.359 / 16.979 | 41.90 / 57.61 / 58.90 | 71,026,688 |
| opencl-gpu | resident | n/a | 14.485 | 69.04 | 2,994,176 |
| opencl-gpu | old stream | 2 | 132.101 | 7.57 | 168,949,504 |
| opencl-gpu | plan, pinned, eager | 1 / 2 / 3 | 31.700 / 24.627 / 24.690 | 31.55 / 40.61 / 40.50 | 71,026,688 |
| opencl-igpu | resident | n/a | 58.593 | 17.07 | 2,994,176 |
| opencl-igpu | plan, pinned / pageable, eager, 2 slots | 2 | 67.433 / 69.190 | 14.83 / 14.45 | 71,026,688 |

All 36 plan rows: bit-identical to the CPU kernel across all 183 tensors, 0
ownership violations. CUDA streamed reaches 95 percent of resident (76.4 against
80.7 tok/s, 9.6 times the old stream) while holding only the book plus the slot
buffers on the device; H2D per token fell 2.38 times (169 to 71 MB). Negatives,
published with the positives: single buffering costs 28 percent; a pageable image
costs 24 percent; **OpenCL on NVIDIA does not overlap copy and compute** (the
driver appears to serialize its two in-order queues, so use CUDA on NVIDIA); on the
iGPU streaming is slower than resident because a "transfer" is a RAM-to-RAM copy;
the plan hides PCIe only while compute is at least as long as the copy (PCIe
12.99 GB/s pinned, 11.05 pageable on the bench), after which the ceiling is about
183 tok/s per 71 MB and scales inversely with bytes.

### Other measured negatives

- The expanded (materializing) reference on the GPU is slow, about 15 to 18
  Gweights/s at batch 1 against 200 to 1270 effective for dtc-eager: it is the
  correctness reference.
- Short matrices: expanded beats DTC at M of 32 and 64 on CUDA; DTC wins from M of
  128 on and by 19 to 69 times at M of 4096 or more. Every GPU DTC call costs at
  least about 0.045 ms (CUDA), 0.055 ms (OpenCL-NVIDIA) and 0.23 ms (iGPU)
  regardless of size.
- iGPU dtc-direct is slower than the single-thread CPU at every M measured.
- The threaded stager gives no benefit even on an idle machine; inline staging is
  the default.
- Batching (section c, CUDA, 4096 squared): dtc-active reaches 2.0 Tweights/s
  effective at batch 64 (0.54 ms) against 26.2 ms for the CPU.

### Lane 4: behavioral fidelity at higher bit rates

Question: what does each bit rate buy in behavior (logits) and not only in tensor
relative L2? Lane 4 compares packs against the raw bf16 model: engine
`nq2-infer-lab`, comparator `nq2-logitcmp`, recipe `--ids 1 --tokens 32 --steps 1
--guidance 1 --score-temp 1 --seed 42 --threads 4`, one denoiser evaluation, 32
positions of 131,072 logits. Raw runs and pack A runs are bit-identical on re-run,
and A's 32 generated IDs equal the Linux battery's.

| Pack | Complete rate (bpw) | Tensor rel L2 (weighted) | top-1 | top-5 | Cosine | Rel logit L2 | Mean KL |
|---|---|---|---|---|---|---|---|
| raw (control) | 16 (bf16) | 0 | 32/32 | 1.000 | 1.000000 | 0.000000 | 0.000000 |
| A: global b256 v2 (shipped) | 0.1563 (stream) | 0.6922 | 0/32 | 0.000 | -0.824 | 1.475 | 5.710 |
| B: per-tensor b256 v3, embed_tokens fp16 | 1.9525 | 0.6626 | 0/32 | 0.013 | 0.763 | 0.693 | 2.589 |
| C: per-tensor PQ d8 RMS, embed_tokens fp16 | 2.8196 | 0.3675 | 0/32 | 0.000 | 0.783 | 0.691 | 2.835 |
| D: per-tensor PQ d4 RMS, embed_tokens fp16 | 3.7145 | 0.2055 | 9/32 | 0.131 | 0.952 | 0.318 | 1.040 |

Rate caveat: B, C and D keep the 402,653,184-weight `encoder.embed_tokens` at fp16
(about 1.68 bits per weight of the total on their own), so their rates are not
comparable to A's; packed-only stream rates are B 1.82, C 2.69, D 3.58 on the 3.43 G
packed weights.

`--restore-tensors` ablations (named tensors raw, the rest from the pack):

| Restored to raw | top-1 | Mean KL |
|---|---|---|
| nothing (B) | 0/32 | 2.589 |
| the replaid embedding matrix (E) | 0/32 | 2.578 |
| diffusion head | 0/32 | 3.711 |
| all 182 layer projections | 0/32 | 1.608 |
| layers + head | **25/32** | 0.0201 |
| layers + E + head | 25/32 | 0.0146 |
| C (PQ8): layers + head raw | **30/32** | 0.0008 |
| C (PQ8): 182 layer projections raw | 23/32 | 0.443 |

**Verdict, negative for every rate tried: no packed configuration reproduces the
raw model's top-1.** Tensor relative L2 improved from 0.66 (B) to 0.37 (C) to 0.21
(D) while behavior moved only a little (D alone shows any top-1 agreement, 28
percent). What changes behavior is keeping a small set of tensors raw: layers raw
and head raw gives 30 of 32 with PQ8 on the rest, but the layer projections are 99
percent of the weights, so that is a diagnosis, not a compression result. The
diffusion head is a single high-leverage tensor. Squared weight error is not
behavioral error (the replaid embedding matrix held 39.8 percent of the squared
weight error and moves KL by only 0.01).

Caveats, all against the strength of the claim: one prompt (id 1), one seed, one
denoiser evaluation, 32 positions (one position is 3 points of top-1); KL is over
a diffusion model's 131,072-way output at one noise level, not an autoregressive
next-token distribution; perplexity and reasoning evals are N/A (Sigma is a
continuous-diffusion LM, and the engine takes token IDs in and out; `tokenizer.json`
is now in the model directory, so a tokenizer-based eval is the next gate); no
multi-step trajectory yet (`--logits-dump-all` exists for it).

## Little Nemo on the lanes

> A separate, end-to-end CUDA lane for Little Nemo (resident compressed pack, startup calibration, DTC by default with predict-driven EXPAND) is documented in [ACCEL-CUDA.md](ACCEL-CUDA.md). The section below is the earlier projection-routing adapter.

`accel/littlenemo/nn_accel.c` routes Little Nemo's projections to a lane while
keeping `nn.c`'s scalar kernel as the golden fallback. The forward test
(15 layers, 3 tokens) reproduces the CPU hidden RMS 0.520647 (tolerance 5e-7, the
anchor's own print precision) and the GPU hidden state is bit-identical to the CPU's
(0 of 4480 values differ) on all three lanes with all 96 projections on the device;
CPU bits are unchanged after a GPU run. Peak working set 739 to 856 MB. But Little
Nemo uploads `W` per call (its weight buffers are reused across layers, so
pointer-keyed caching would be unsound), which makes it PCIe-bound: **there is no
end-to-end GPU inference yet.**

## Limits and unfinished

- No end-to-end GPU inference runtime; section p is the linear-layer ceiling only.
- Lazy loses to active on the GPU (above).
- v3 PQ tensors and `cols % block != 0` fall back to the CPU; there is no real-pack
  test yet for v3 per-tensor books (only synthetic layouts).
- OpenCL reports no free-memory figure, so its budget derives from total (50
  percent). Integrated memory is shared system RAM.
- The vendor drivers use internal locks; "zero mutexes" is a property of our code.
- Next probes: PQ with a residual stage (`nq2-rpq` exists as an experiment, not
  wired into `nq2-quantize`), higher-rate heads only, multi-step trajectories, and a
  tokenizer-based eval.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `NODEV` for the CUDA lane | no `nvrtc64_*.dll`; set `NTSE_NVRTC_DLL` or `CUDA_PATH`, or let `auto` fall to OpenCL |
| everything runs on "cpu reference" | a missing driver or DLL (the open message says why), or `NTSE_ACCEL_DISABLE` / `NTSE_ACCEL_LANE=cpu` is set |
| OpenCL-NVIDIA streaming is slow | the NVIDIA OpenCL driver does not overlap copy and compute; use the CUDA lane |
| the stream plan misses its ceiling | the image fell back to pageable memory (raise `NTSE_ACCEL_PIN_MB`), or a single slot is in use |
| a budget refusal | that call ran on the CPU; raise `NTSE_ACCEL_VRAM_MB` within the hard cap |
| first resident timing looks 20 percent slow | the GPU was ramping its clock from idle; section p now runs an untimed warm-up pass |