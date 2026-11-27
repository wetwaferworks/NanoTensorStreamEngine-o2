# accel/: NTSE-o2 GPU acceleration lanes (research, scratch)

Direction and idea: **JD**. Lane proposals and measurement list: **Gizmo**.
NTSE / nq2 / the converter and Little Nemo: **Elfie**. Co-implemented by
**Cortana**.

Status: merged into `main` (2026-11-26). Nothing outside `accel/` is
modified. The pure
C backend builds and runs with zero GPU code, and **the CPU path stays
the reference and official path**. These lanes are an accelerator for
it, not a replacement.

## Three lanes, and why each is built the way it is

| lane | API | device here | accumulate | why this API |
|---|---|---|---|---|
| `cuda` | CUDA **driver API** (`nvcuda.dll`) + **NVRTC** (`nvrtc64_*.dll`), both loaded at run time | RTX 3050 Ti Laptop, sm_86, 4 GB | fp32 IEEE (DTC), fp64 (bf16 projection) | No MSVC/`cl.exe` on this workstation, so `nvcc` cannot host-compile; NVRTC compiles the kernels at open (cubin for the exact SM, PTX-JIT fallback). No CUDA SDK is needed to *build*; the binary runs, on the CPU lane, on a machine without one. |
| `opencl-igpu` | OpenCL 1.2+ (`OpenCL.dll`, run-time loaded), kernels built from source by the driver | Intel UHD (i5-12450H), OpenCL 3.0 NEO, 48 CUs, no fp64 | fp32 IEEE (DTC), float-float (bf16 projection) | OpenCL is the one compute API the Intel driver ships that compiles kernels from source with no SDK. Unified memory: the budget is capped at 512 MiB because it is system RAM. |
| `opencl-gpu` | same OpenCL backend, device filter = discrete | RTX 3050 Ti via NVIDIA's OpenCL ICD | fp32 IEEE / fp64 | The "default GPU lane for NVIDIA without CUDA". See below. |

**Why OpenCL and not Vulkan for lanes 2 and 3.** Vulkan compute needs
SPIR-V. Producing SPIR-V needs `glslc`/`glslangValidator`/a Vulkan SDK,
and none is on the workstation (downloads were ruled out). Hand-written
SPIR-V blobs would be unreviewable, which fails the "solid, sound,
robust" bar. OpenCL takes source text, both ICDs are installed (Intel
`igdrcl64`, NVIDIA `nvopencl64`), and with `FP_CONTRACT OFF` plus
`vload_half` it gives IEEE results matching the CPU bit for bit (below).
A Vulkan backend can slot into the same ops vtable later if a shader
compiler becomes available.

**About "NVIDIA cards without CUDA cores".** Every NVIDIA GPU since
2006 (G80) has CUDA cores, so that card does not exist in practice. The
real cases are (a) no NVRTC/CUDA toolkit DLL on the machine, (b) a
driver whose CUDA version is too old or new for the card, (c) a
policy or packaging reason to avoid CUDA. Lane 3 covers all three
through the vendor's OpenCL driver, which still runs on the same CUDA
cores. `AUTO` order is CUDA > OpenCL-GPU > OpenCL-iGPU > CPU.

## What runs on the GPU

* **DTC** (dictionary compute, `tools/nq2-dc-kernel.h` algebra) without
  ever materializing W. Three forms:
  * `dtc-eager`: full tables `T[j][k] = Σ_i C[k][i]·x[jB+i]` for every
    segment and code, then apply.
  * `dtc-active`: tables only for the (segment, code) pairs the index
    actually uses. The active set is fixed by `idx`, so it is computed
    once at upload, not per call.
  * `dtc-direct`: no table, each output gathers its prototypes
    directly.
  * `dtc-lazy`: tables built on demand per call (mark, then build
    flagged entries only). See Lane 1.
* **Expanded-weights reference** (`expanded`): materialize
  `w = mn + sc·book[idx][i]` on the fly and run a fixed 256-wide tree
  GEMV. This is the GPU twin of the materializing CPU path.
* **bf16 projection** (`ntse_accel_linear_bf16`) for Little Nemo.
  Double accumulation (CUDA, OpenCL with fp64) or float-float (Dekker
  split + TwoSum, iGPU). Measured correctly rounded (below).

Exactness design: DTC sums are pinned in the CPU kernel's order
(ascending `i` for tables and `A[j]`, ascending `j` for `y`). The GPU
kernels keep that order per output and forbid contraction
(`__fadd_rn/__fmul_rn`, `--fmad=false`; `FP_CONTRACT OFF`). fp16 frames
are converted with `cvt.f32.f16` or `vload_half`. Result: **DTC is
bit-identical to the CPU kernel on every lane.** Expanded uses a
different (tree) order, so it differs from the CPU in low bits, and it
is closer to the fp64 oracle than the CPU's sequential sum is.

## Device layout

One allocation per uploaded tensor. Sections are 256-byte aligned:
`bookT [B][K]`, `book [K][B]`, `idxT / mnT / scT` transposed to
`[nseg][rows]` (coalesced per segment), and the active list (codes
`j·K+k`). Per-call scratch (x, tables, A, y) is a context buffer that
grows by `ensure()` and is charged like everything else.

## GPU-side Tensor Handshake (streaming), zero mutexes

`ntse_accel_stream()` runs a list of tensors through S pinned staging
slots (S = 1..4). Each slot moves `FREE → STAGING → STAGED → INFLIGHT
→ FREE`:

* claims are CAS on a per-slot state word, with release/acquire
  publication, and waits go through the handshake door from
  `tools/nq2-handshake.h` (spin → yield → WaitOnAddress/futex). No
  mutex, condition variable or critical section anywhere in `accel/`.
* each slot has two events, `h2d[s]` (copy done) and `done[s]` (kernels
  done). A slot returns to FREE **only after its `done` event has been
  observed** (`observed[s] == n+1`). S slots means up to S−1 in flight
  plus one staging.
* optional stager thread (`CreateThread`/`pthread`) that only memcpys
  into pinned memory and never calls the GPU API. All API calls stay
  on the driver thread.
* a test hook (`ntse_accel_test_force_early_recycle`) tries to recycle
  a slot still in flight. The guard refuses it, counts a violation, and
  the output stays correct.
* Honest note: the vendor drivers themselves use internal locks. "Zero
  mutexes" is a property of our code, not of the CUDA/OpenCL runtimes.

## Safety: validation, budget, fallback

* **Every input is validated** before any device work: geometry
  (rows/cols/block/entries non-zero, `cols % block == 0` for DTC,
  products capped at 2^40), every `idx < entries`, fp16 frames with
  exponent 31 (Inf/NaN) rejected, book finite. Bad input is
  `NTSE_E_INVAL` on **every** lane, including the CPU one.
* **Every API return is checked.** A device error is a *sticky fault*:
  the context logs once and runs everything on the CPU reference from
  then on (`ntse_accel_failed()`).
* **Bounded memory.** Budget = min(75 % of free or 50 % of total, hard
  cap 3 GiB discrete / 512 MiB integrated). `NTSE_ACCEL_VRAM_MB` and
  `cfg.vram_budget` can lower it. Larger values are clamped loudly.
  Every allocation is charged and also bounded by the device's
  `max_alloc`. On this 4 GB card the derived budget is 2471 MiB.
* `BUDGET` / `UNSUPPORTED` are per-call refusals. That call runs on the
  CPU, and the context stays on the GPU for the next one. An
  over-budget upload yields a host-only handle that always runs on the
  CPU.
* `NTSE_ACCEL_DISABLE=1` or `NTSE_ACCEL_LANE=cpu` forces the CPU
  reference. A missing driver or DLL opens on the CPU lane and says
  why.
* Unsupported → CPU: nq2 v3 PQ (flag 1) tensors and `cols % block != 0`.

## Files

```
accel/include/ntse_accel.h          public API (lanes, ops, codes, stream)
accel/src/accel_internal.h          backend ops vtable
accel/src/accel_core.c              validation, CPU reference, fp64 oracle,
                                    layout, budget, fallback, streaming
accel/src/accel_cuda.c              CUDA driver API + NVRTC backend
accel/src/accel_kernels_cuda.h      CUDA kernel source
accel/src/accel_opencl.c            OpenCL backend (iGPU and GPU)
accel/src/accel_kernels_cl.h        OpenCL kernel source
accel/tests/test_accel_kernels.c    per-kernel correctness vs CPU and oracle
accel/tests/test_accel_guard.c      hostile input, fallback, budget, NODEV
accel/tests/test_accel_stream.c     handshake / streaming
accel/tests/test_accel_real.c       real Sigma pack tensors (+ walk fingerprint)
accel/tests/test_accel_linear.c     bf16 projection kernel
accel/littlenemo/nn_accel.c         Little Nemo shim (projections -> lane)
accel/littlenemo/test_nemo_forward_accel.c   forward vs anchor 0.520647
accel/tools/ntse-accel-bench.c      benchmark receipts (TSV "R" lines)
accel/tools/run-bench-gated.ps1     gated, detached benchmark runner
accel/build-accel.bat / .sh         build into accel/bin, accel/obj only
```

## Build

```
accel\build-accel.bat          (Windows, MinGW gcc; from the repo root)
sh accel/build-accel.sh        (POSIX gcc/clang; from the repo root)
```

Flags `-O2 -std=c99 -Wall -Wextra [-static] -D_FILE_OFFSET_BITS=64`.
Zero warnings under MinGW gcc 16.2.0 and Linux gcc 14. No CUDA, OpenCL
or Vulkan SDK is needed to build. The pure C build (`build.bat`,
`tools/`, `littlenemo/`) is untouched and has no GPU code.

## Tests and results (2026-11-26, ET; workstation busy with the Super VL conversion, all runs at BelowNormal)

| suite | cuda | opencl-gpu | opencl-igpu |
|---|---|---|---|
| kernels (7 shapes × 4 ops, determinism) | 36/36 | 36/36 | 36/36 |
| stream (slots 1–3, inline/thread, 3 ops, early-recycle refusal) | 42/42 | 42/42 | 42/42 |
| real Sigma tensors (layer 0, 7 projections × 4 ops) | 49/49 | 49/49 | 49/49 |
| bf16 projection (10 shapes incl. cancellation, refusals, budget) | 47/47 | 47/47 | 47/47 |
| Little Nemo forward (15 layers, 3 tokens) | 10/10 | 10/10 | 10/10 |
| guard (lane-independent) | 36/36 | | |

Key numbers:

* **DTC eager/active/direct: bit-identical to the CPU kernel** (rel L2 0,
  0 differing values) on all synthetic shapes and on all 7 real
  layer-0 Sigma projections, on all three lanes. vs fp64 oracle: rel L2
  5.05e-6 to 5.36e-6 (the CPU kernel's own fp32 error, reproduced
  exactly).
* **Expanded:** vs CPU rel L2 ≤ 1.63e-6 (gate 1e-5). vs fp64 oracle
  9.3e-8 to 1.3e-7, about 10× closer to the oracle than the CPU's
  sequential sum.
* **bf16 projection:** every output within 0.500 ulp of the exact dot
  product, so it is correctly rounded on all lanes (including
  float-float on the iGPU). Exact cancellation cases recover the tiny
  term. NaN-poisoned outputs are fully overwritten.
* **Little Nemo:** CPU hidden RMS 0.520647 (anchor 0.520647; tolerance
  5e-7, the anchor's own print precision). GPU RMS 0.520647, and **GPU
  vs CPU hidden state bit-identical** (0/4480 values differ) on all
  three lanes, with all 96 projections on the device. CPU bits are
  unchanged after a GPU run. Deterministic across reset. Peak working
  set 739–856 MB. (Bit identity follows from both sides producing the
  correctly rounded fp32 of the same exact sum. It is measured, not
  assumed. The gate is rel L2 ≤ 1e-5.)
* **Walk alignment on the real pack:** weight-space fingerprints equal
  the packer's recorded per-tensor errors (`results/errprobe-baseline-
  2026-11-23.txt`) to 4 decimals (q 0.9921, k 0.9920, v 0.9958, o
  1.0009, gate 0.9952, up 0.9951, down 0.9961). Each beats its own
  one-block-shifted control (1.017–1.024), and cos(recon, raw) is
  0.10–0.14. Note: this v2run pack is near-1.0 lossy in weight space.
  That is a packing-fidelity fact, not an accel one.
* **Guard:** hostile inputs refused on every lane; disable-env fallback
  bit-exact for all ops and the stream; NODEV with NVRTC missing;
  budget refusals per call with no sticky fault; budget clamp to 2471 MiB.
* **Full Sigma pack** (183 2-D tensors, 3.43 G weights, embeddings
  excluded): resident 166 MB device, `differ_vs_cpu = 0` for every op,
  resident and streamed.
* Frozen CPU anchors (tokens 75768/122388, streamtest
  1394.1117661067326) are untouched by construction: no tracked file
  changed and the pure C build has no accel code. The Little Nemo CPU
  anchor is re-verified inside every forward test run.

Tolerances (`tests/test_common.h`): DTC rel L2 ≤ 1e-6 (measured 0),
expanded rel L2 ≤ 1e-5, cosine ≥ 0.999999, Nemo GPU-vs-CPU rel L2 ≤
1e-5, anchor ±5e-7.

## Benchmarks (gated run 10:44–10:48 AM ET, 2026-11-26, after the Super VL HQ conversion finished at 10:43:48 ET)

Receipts: [`bench/receipts/lane-bench-2026-11-26/RECEIPTS.txt`](bench/receipts/lane-bench-2026-11-26/RECEIPTS.txt). The per-run TSV logs stay private (`*.log` is gitignored).
All 9 runs exited 0, sticky_fault = 0, ownership violations = 0.

* rev `ef39281b1496a7b4733aa547468b99c58d6a73db` plus untracked `accel/`
  (no tracked file modified)
* pack `sigma-full-b256-v2run.nq2` sha256 `b6f94b66…fb703e` (74,883,852 B),
  book `.nq2book` sha256 `a80604b5…15f595` (262,156 B). This is
  **Sigma-3B, not Super VL**: Super VL does not fit a 4 GB card.
* RTX 3050 Ti Laptop 4 GB, driver 621.24 (CUDA 13.5, NVRTC 13.4), P0;
  Intel UHD (i5-12450H 8C/12T), OpenCL 3.0 NEO 31.0.101.5081;
  16 GB RAM, about 7 GB free; MSYS2 gcc 16.2.0 `-O2 -std=c99 -static`
* Every timing is end to end per call (activation upload, kernels,
  result download). Thermals were not sampled in this run (added to
  the runner for the next one).

### Full Sigma pack pass (section p): 183 packed 2-D tensors, 3.429 G weights, embeddings excluded

Linear-layer ceiling = 1 / (time for one matvec through every
projection). This is **not end-to-end decode**. TTFT, prompt tok/s
and generation tok/s are N/A because there is no GPU inference runtime
yet.

| lane | mode | ms/pass | linear tok/s ceiling | H2D bytes/token | divergence vs CPU |
|---|---|---|---|---|---|
| CPU reference (1 thread) | dtc-eager | 105.335 | 9.49 | n/a | reference |
| cuda | resident dtc-eager | 12.167 | **82.19** | 2,994,176 (x only) | 0 values differ, relL2 0 |
| cuda | resident dtc-active | 12.560 | 79.62 | 2,994,176 | 0 |
| cuda | resident dtc-direct | 114.033 | 8.77 | 2,994,176 | 0 |
| cuda | stream dtc-eager, 2 slots | 117 | 8.55 | 168,949,504 | 0 |
| cuda | stream dtc-eager, 3 slots | 120 | 8.33 | 168,949,504 | 0 |
| opencl-gpu | resident dtc-eager | 14.337 | 69.75 | 2,994,176 | 0 |
| opencl-gpu | resident dtc-active / direct | 14.924 / 96.742 | 67.00 / 10.34 | 2,994,176 | 0 |
| opencl-gpu | stream 2 / 3 slots | 123 / 132 | 8.13 / 7.56 | 168,949,504 | 0 |
| opencl-igpu | resident dtc-eager | 54.221 | 18.44 | 2,994,176 | 0 |
| opencl-igpu | resident dtc-active / direct | 54.290 / 927.072 | 18.42 / 1.08 | 2,994,176 | 0 |
| opencl-igpu | stream 2 / 3 slots | 138 / 137 | 7.25 / 7.31 | 168,949,504 | 0 |

Peak device memory: 166.5 MB charged for the resident pack (nvidia-smi
delta +257 MiB, including context). Peak host working set: 276–362 MiB.
Every op and every lane reproduces the CPU kernel bit for bit over all
183 tensors.

Speedups (resident dtc-eager vs the single-thread CPU reference): CUDA
8.7×, OpenCL-NVIDIA 7.3×, Intel iGPU 1.9×.

### Negatives (published with the positives)

* **Streaming was the bottleneck: 8.3–8.6 tok/s vs 82 resident on CUDA**
  (fixed by Lane 2 below: 76.4 tok/s streamed).
  It moves 169 MB/token, but the compressed tensors are only about
  67 MB. Two causes, both in our pipeline and not in PCIe: (a) the
  global 256×256 book is re-sent twice (book + bookT, 512 KB) with
  every one of the 183 tensors, about 96 MB/token; (b) every item is
  re-laid-out on the CPU (transposes plus the active-list scan) each
  token. That is the Lane 2 work below.
* **dtc-direct (table-free) loses** everywhere: 9–10× slower than
  eager. It re-reads prototypes per output and has no table reuse.
* **Expanded (materializing) reference is slow on the GPU**: about
  15–18 Gweights/s at batch 1, versus 200–1270 Gweights/s effective
  for dtc-eager. It is the correctness reference, not a fast path.
* **Short matrices: expanded beats DTC** at M = 32 (0.039 vs 0.041
  ms) and M = 64 (0.044 vs 0.050 ms) on CUDA. DTC wins from M = 128
  on, and by 19–69× at M ≥ 4096 (section b, N = 4096, K = B = 256).
  This matches the corrected break-even direction (DTC wins for tall
  matrices). The measured crossover sits below the arithmetic 257
  because GPU calls this small are launch-latency bound (about 40 µs
  floor).
* **Launch-latency floor:** every GPU DTC call costs ≥ about 0.045 ms
  (CUDA), 0.055 ms (OpenCL-NVIDIA) and 0.23 ms (iGPU) regardless of
  size. Small tensors do not speed up.
* **iGPU dtc-direct is slower than the single-thread CPU** at every M
  in section b (e.g. 20.6 ms vs CPU 1.55 ms at M = 16384). iGPU
  expanded is only about 2× faster than the CPU's expanded (18.4 vs
  38.4 ms). On the iGPU only DTC eager/active pay off (0.30 vs CPU
  1.58 ms at M = 16384).
* **Threaded stager:** no benefit even on the idle machine. CUDA 4096²
  × 24 stream is 19.0–20.3 ms inline vs 20.3–22.2 ms threaded
  (earlier 3–5× losses were CPU starvation under the conversion and
  are gone). Inline staging stays the default.
* Batch (section c, CUDA, 4096²): dtc-active reaches 2.0 Tweights/s
  effective at batch 64 (0.54 ms) vs CPU 26.2 ms. expanded 91
  Gweights/s.
* Dictionary sweep (section d, CUDA, 4096²): dtc-eager is
  0.063–0.144 ms for every B ∈ {64..512}, K ∈ {16, 64, 256}. B = 64
  is the slowest (more segments). expanded is 1.1–3.2 ms.

## Lane 1 (Gizmo): CUDA-native DTC, eager vs active vs lazy

All four DTC policies run natively on the GPU from the compressed
form, with no weight expansion and prototype tables on the device. The
expanded-weights kernel is kept as the CUDA (and OpenCL) reference:

| op | tables | when the work set is known |
|---|---|---|
| `dtc-eager` | every (segment, code) | n/a: builds all K per segment |
| `dtc-active` | only codes the index uses | once, at upload (the index is static) |
| `dtc-lazy` | only codes the index uses | at every call: a mark kernel flags (segment, idx) pairs, the build kernel skips unflagged entries |
| `dtc-direct` | none: each output gathers prototypes | n/a |
| `expanded` | reference: materialize W, fixed-tree GEMV | n/a |

The GPU "lazy" is deliberately not a spin-wait memo (compute on first
touch, others wait). That would need intra-kernel waiting between
threads, which can deadlock on GPUs without independent thread
scheduling and is not portable to OpenCL. The mark pass uses
idempotent stores (every writer stores 1), so it is race-free without
atomics. All four DTC ops are **bit-identical to the CPU kernel** on
all three lanes: synthetic suites and all 7 real Sigma layer-0
projections (`real-*-3.log`; those layer logs are not in the committed receipt set), and every full-pack pass
(`differ_vs_cpu = 0`).

Receipts (`bench/receipts/bench-lane1/`, 11:24–11:26 AM ET, and
`bench/receipts/bench-lane1-warm/` 11:28 ET with the warm-up pass), full Sigma
pack:

| lane | eager | active | lazy | direct | lazy vs best |
|---|---|---|---|---|---|
| cuda resident (warm run) | 12.635 ms / 79.14 tok/s | 12.842 / 77.87 | 13.620 / 73.42 | 112.982 / 8.85 | −7 % |
| cuda plan, pinned, 2 slots (two runs) | 12.818–13.027 / 76.8–78.0 | 12.712–14.516 / 68.9–78.7 | 15.116–15.244 / 65.6–66.2 | n/a | −15 % |
| opencl-gpu resident | 14.786 / 67.63 | 14.930 / 66.98 | 17.938 / 55.75 | 95.368 / 10.49 | −18 % |
| opencl-igpu resident | 54.493 / 18.35 | 54.275 / 18.42 | 68.563 / 14.59 | 927.129 / 1.08 | −21 % |

Verdict, negative for lazy: with a static index, lazy builds exactly
the entries active builds, and also pays a flag clear plus a mark pass
over every block on every call. It loses 7–21 %. Lazy only pays when
the work set is not known ahead of time. That is not the case for
packed weights, but it would be for, e.g., an index that changes per
call. Active ≈ eager on the GPU even when only 90 of 256 codes per
segment are used (section a fixtures), because table build is a small
share of a call next to apply and the launch floor. Skipping entries
saves little.

Methodology note, published as a negative: the first gated run timed
the first resident op while the GPU was still ramping its clock from
idle (210 MHz samples): 15.06 ms vs 12.17–12.64 ms warm. Section p now
runs an untimed warm-up pass per op. Section a/b single-tensor timings
sit on the ≈ 40–50 µs launch floor and vary ±50 % between runs (e.g.
eager 3072² 0.045 vs 0.119 ms), so compare ops within one run only.
Full-pack timings vary about ±10 % between runs.

## Lane 2 (Gizmo): Tensor Handshake GPU concurrency, the prepared stream plan

Streaming was the bottleneck: 7.97 tok/s streamed vs 80.67 resident
(CUDA, full Sigma pack). A plan (`ntse_accel_plan_create / _run`,
`src/accel_plan.inc.h`) fixes the two causes found above:

* **Layout once:** every tensor is transposed into a host image of its
  device layout at create time (121 ms for the whole pack). Per token
  the CPU does no layout work.
* **Books resident once:** books are deduplicated by geometry and
  content (1 book, 512 KiB, for Sigma) instead of being re-sent with
  each of the 183 tensors.
* **Pinned image arena** (68.6 MB here) gives direct async DMA from page-locked
  memory. If the image does not fit the pinned budget (1 GiB default,
  `NTSE_ACCEL_PIN_MB`, clamp 4 GiB), it stays pageable and is staged
  through pinned slot buffers.
* **Single, double, triple buffering** (1–4 slots). Uploads run on the
  copy queue. The compute queue waits on a per-slot `h2d` event
  (device side). A slot's device and pinned buffers are reused only
  after its `done` event has been observed on the host. Ownership is
  explicit: FREE → STAGING → STAGED → INFLIGHT → FREE via CAS, with
  zero mutexes. A forced early recycle is refused and counted.
* The device path uses the create-time snapshot (tested: mutating the
  caller's index after create does not change GPU output).

Receipts (`bench/receipts/bench-lane2/`, 11:18–11:19 AM ET, same rev, pack,
hardware and driver as above; GPU 48.9–53.8 °C, max 27.8 W CUDA /
30.1 W OpenCL-NVIDIA / 14.2 W iGPU run; peak host working set
400–617 MiB). PCIe probe, 64 MiB: H2D pinned 12.99 GB/s, pageable
11.05 GB/s, D2H pinned 13.15 GB/s (CUDA).

| lane | path | slots | ms/pass | linear tok/s ceiling | H2D bytes/token |
|---|---|---|---|---|---|
| cuda | resident (reference) | n/a | 12.396 | 80.67 | 2,994,176 |
| cuda | old stream | 2 | 125.547 | 7.97 | 168,949,504 |
| cuda | **plan, pinned image, eager** | 1 / **2** / 3 | 18.239 / **13.090** / 12.982 | 54.83 / **76.40** / 77.03 | 71,026,688 |
| cuda | plan, pinned image, active | 1 / 2 / 3 | 17.669 / 12.712 / 13.110 | 56.60 / 78.66 / 76.28 | 71,026,688 |
| cuda | plan, pageable image, eager | 1 / 2 / 3 | 23.864 / 17.359 / 16.979 | 41.90 / 57.61 / 58.90 | 71,026,688 |
| opencl-gpu | resident | n/a | 14.485 | 69.04 | 2,994,176 |
| opencl-gpu | old stream | 2 | 132.101 | 7.57 | 168,949,504 |
| opencl-gpu | plan, pinned, eager | 1 / 2 / 3 | 31.700 / 24.627 / 24.690 | 31.55 / 40.61 / 40.50 | 71,026,688 |
| opencl-igpu | resident | n/a | 58.593 | 17.07 | 2,994,176 |
| opencl-igpu | plan, pinned / pageable, eager, 2 slots | 2 | 67.433 / 69.190 | 14.83 / 14.45 | 71,026,688 |

All 36 plan rows: on device, `differ_vs_cpu = 0` (bit-identical to the
CPU kernel across all 183 tensors), 0 ownership violations, rc 0.

What it means:

* **CUDA streamed reaches 95 % of resident** (76.4 vs 80.7 tok/s, 9.6×
  the old stream) while holding only the book plus the slot buffers on
  the device, instead of the whole 166 MB pack. Copy (71 MB at 13 GB/s ≈
  5.5 ms) hides behind compute (≈ 12.4 ms) from 2 slots on. Triple
  buffering adds nothing measurable here: compute is the ceiling.
* **H2D per token fell 2.38×** (169 → 71 MB). What remains is the
  compressed index and frames (about 67 MB) plus activations.

Negatives:

* **Single buffering costs 28 %** (54.8 vs 76.4). Overlap is the whole
  game.
* **Pageable image costs 24 %** (57.6 vs 76.4 at 2 slots: the extra
  memcpy into pinned slots runs on the driver thread).
* **OpenCL on NVIDIA does not overlap copy and compute**: 24.6 ms per
  pass is more than compute (14.5 ms resident) plus copy (71 MB at
  12.7 GB/s ≈ 5.6 ms) added together, so there is no overlap plus about
  4.5 ms of extra overhead. 40.6 tok/s even with 2–3 slots. The
  NVIDIA OpenCL driver appears to serialize its two in-order queues.
  The CUDA lane is the one to use on NVIDIA.
* **iGPU: streaming is slower than resident** (14.8 vs 17.1). On
  unified memory a "transfer" is a RAM-to-RAM copy, so on the iGPU
  resident is always right when it fits the budget.
* The plan hides PCIe only while compute is at least as long as the
  copy. For a model whose compressed bytes per token exceed about
  13 GB/s × compute time, the ceiling becomes PCIe: about 183 tok/s per
  71 MB, scaling inversely with bytes.

## Lane 4 (Gizmo): behavioral fidelity at higher bit rates

Question: what does each bit rate buy in *behavior* (logits) and not
only in tensor rel L2? Previous logit comparisons in this project
(dense vs DTC, threads 1 vs default) checked numerical equality of
two engines on the same pack. Lane 4 compares packs **against the raw
bf16 model**.

Setup (receipts `bench/receipts/lane4/RECEIPTS.txt`, `progress.txt`,
`cmp-*-vs-raw.txt`):

* Engine `tools/nq2-infer-lab.c` (Elfie), comparator
  `tools/nq2-logitcmp.c`, quantizer `tools/nq2-quantize.c`. All built
  from this worktree (sha256 in RECEIPTS) into `accel/lane4/bin`, gcc
  16.2.0 `-O2 -std=c99 -static`, 0 warnings.
* Model: Sigma-3B raw safetensors and packs A/B, read-only from the
  2026-11-25 results tree. New packs C/D under `accel/lane4/packs`.
* Recipe (the Linux battery's): `--ids 1 --tokens 32 --steps 1
  --guidance 1 --score-temp 1 --seed 42 --threads 4`, one denoiser
  evaluation dumped, 32 positions × 131,072 logits. Each run takes
  about 40 s, BelowNormal. Peak working set is 0.3 GB (raw, A) to
  1.1 GB (B).
* Metrics per position vs raw: top-1 agreement, top-5 overlap,
  cosine, mean |Δlogit|, relative logit L2, KL(raw‖pack).
* Perplexity and reasoning: **N/A, stated plainly.** Sigma is a
  continuous-diffusion LM, so there is no autoregressive likelihood
  to take a perplexity of. A reasoning eval needs text through a
  tokenizer and a multi-step generation recipe, which this engine
  (token IDs in, token IDs out) does not have yet. `tokenizer.json`
  is now in the model dir, so this is the next gate. Logit-level
  metrics are the behavioral signal here.

Determinism: raw r1 vs r2 and A r1 vs r2 are bit-identical (KL 0,
cosine 1.000000, all 32 token IDs equal). A's 32 generated IDs equal
the WSL/Linux battery's (`463 78431 129796 …`), so the result is
identical across OSes too.

| pack | rate (complete bpw) | tensor rel L2 (weighted) | top-1 | top-5 | cosine | rel logit L2 | mean KL | max KL |
|---|---|---|---|---|---|---|---|---|
| raw r2 (control) | 16 (bf16) | 0 | 32/32 | 1.000 | 1.000000 | 0.000000 | 0.000000 | 0.000000 |
| A global b256 v2 (shipped) | 0.1563 stream | 0.6922 | 0/32 | 0.000 | −0.824 | 1.475 | 5.710 | 6.971 |
| B per-tensor b256 v3, embed_tokens fp16 | 1.9525 | 0.6626 | 0/32 | 0.013 | 0.763 | 0.693 | 2.589 | 3.911 |
| C per-tensor PQ d8 K256 RMS, embed_tokens fp16 | 2.8196 | 0.3675 | 0/32 | 0.000 | 0.783 | 0.691 | 2.835 | 2.981 |
| D per-tensor PQ d4 K256 RMS, embed_tokens fp16 | 3.7145 | 0.2055 | 9/32 | 0.131 | 0.952 | 0.318 | 1.040 | 1.208 |

Where B loses behavior (`--restore-tensors`: named tensors read raw,
everything else from B). This is new evidence:

| restored to raw | top-1 | top-5 | cosine | mean KL |
|---|---|---|---|---|
| nothing (B) | 0/32 | 0.013 | 0.763 | 2.589 |
| E (`replaid_embedding_matrix`) | 0/32 | 0.013 | 0.763 | 2.578 |
| head (`diffusion_head`) | 0/32 | 0.000 | 0.584 | 3.711 |
| E + head | 0/32 | 0.000 | 0.584 | 3.701 |
| all 5 non-layer (E, head, 3 replaid linears) | 0/32 | 0.000 | 0.704 | 3.244 |
| all 182 layer projections | 0/32 | 0.000 | 0.969 | 1.608 |
| layers + E | 0/32 | 0.000 | 0.969 | 1.596 |
| layers + 3 replaid linears | 0/32 | 0.000 | 0.969 | 1.602 |
| **layers + head** | **25/32** | 0.894 | 0.99896 | **0.0201** |
| layers + E + head | 25/32 | 0.938 | 0.99933 | 0.0146 |

Reading:

1. **Two independent failures, both fatal to top-1.** The 182 layer
   projections (b256 flat VQ, each at rel L2 0.93–0.997, i.e. almost
   no signal) destroy the hidden state. Restoring every non-layer
   tensor does not help, and restoring the head alone even raises KL
   (2.59 to 3.71), because a correct head applied to a wrong hidden
   state is just as wrong. The **diffusion head** (rel L2 0.79) by
   itself takes top-1 from 25/32 to 0/32, even with every layer raw.
2. **E is not the lever at the behavioral level.** E
   (`replaid_embedding_matrix`, rel L2 0.9686 in B) held 39.8 % of
   the squared weight error in the November 24 analysis. Restoring it
   moves KL by only 0.01 in either context. Squared weight error is
   not behavioral error. Note: `--embed-fp16` stores
   `encoder.embed_tokens` (402 M weights, most of B's rate), not the
   replaid E that EXPERIMENTS-2026-11-24.md recommended for fp16.
   That flag's naming is worth a look by Elfie.
3. Layers and head raw with E packed: 25/32 top-1, KL 0.02. That is
   the fidelity ceiling of b256 VQ when it is used only on the small
   non-layer tensors.

Rate caveat: A's 0.1563 is stream-only and A packs embed_tokens too;
B, C and D keep the 402,653,184-weight `encoder.embed_tokens` at fp16
(about 1.68 bits per weight of the total on their own), so B/C/D rates
are dominated by that and are not comparable to A. Packed-only stream
rates: B 1.82, C 2.69, D 3.58 bits per weight on the 3.43 G packed
weights.

Higher rates (new packs C and D, built here with the existing
`nq2-quantize --pq D --norm rms`; each pack takes 17–20 min at 12
threads/BelowNormal, 1.3–1.7 GB peak):

| config | top-1 | top-5 | cosine | mean KL |
|---|---|---|---|---|
| B (flat b256) | 0/32 | 0.013 | 0.763 | 2.589 |
| C (PQ8) | 0/32 | 0.000 | 0.783 | 2.835 |
| D (PQ4) | 9/32 | 0.131 | 0.952 | 1.040 |
| C, 182 layer projections raw | 23/32 | 0.569 | 0.9895 | 0.443 |
| C, layers + head raw | 30/32 | 0.931 | 0.99994 | 0.0008 |
| C, head only raw | 0/32 | 0.000 | 0.761 | 2.955 |

Verdict (negative for every rate tried): **no packed configuration
reproduces the raw model's top-1.** Tensor rel L2 improved from 0.66
(B) to 0.37 (C) to 0.21 (D), and behavior moved only a little: KL
2.59 to 2.84 (C is slightly worse than B in KL, better in cosine,
so the two metrics disagree there) to 1.04 (D). Only D shows any
top-1 agreement (28 %). What changes behavior is not rate on the
projections, it is keeping a small set of tensors raw: layers raw +
head raw gives 30/32 with PQ8 on the rest, but the layer projections
are 99 % of the weights, so this is not a compression result, it is
a diagnosis. The diffusion head is a single high-leverage tensor. At
PQ8 its damage is smaller than at b256 (with layers raw: 23/32
top-1 vs 0/32), and cleaning it up moves top-1 from 23 to 30.

Caveats, all negative on the strength of the claim: one prompt (id 1),
one seed, one denoiser evaluation (`--steps 1`), 32 positions. top-1
over 32 positions has a coarse resolution (one position is 3 points).
KL here is over a diffusion model's 131,072-way output at one noise
level, not an autoregressive next-token distribution. There is no
ppl, no reasoning eval (N/A, see above), and no multi-step
trajectory yet (`--logits-dump-all` exists for that). Thermals, TTFT
and H2D are N/A: this lane is CPU only (`nq2-infer-lab`, no GPU).
Rev: tools sources are from the worktree at base `ef39281` plus
uncommitted changes; hashes in RECEIPTS. Next probes: PQ with a
residual stage (nq2-rpq exists as an experiment, not wired into
nq2-quantize), higher-rate heads only, multi-step trajectories, and
a tokenizer-based eval once available.

## Limits, risks, unfinished

* **No end-to-end GPU inference yet.** Little Nemo's projections run on
  the GPU, but it uploads W per call (its weight buffers are reused
  across layers, so pointer-keyed caching would be unsound), which
  makes it PCIe-bound. TTFT and generation tok/s for a GPU runtime are
  therefore **N/A**. Section p gives the *linear-layer ceiling* only.
* Lazy on the GPU is run-time discovery (mark + build), not a
  spin-wait memo (see Lane 1 for why). It is measured and loses to
  active.
* nq2 v3 PQ tensors and `cols % block != 0` fall back to the CPU.
  There is no real-pack test yet for v3 per-tensor books (only
  synthetic layouts).
* The threaded stager gives no benefit (measured above). Inline is
  the default.
* OpenCL reports no free-memory figure, so its budget derives from
  total (50 %). Integrated memory is shared system RAM, and nvidia-smi
  VRAM peaks cover the NVIDIA device only.
* The CUDA lane needs `nvrtc64_*.dll` (found via `CUDA_PATH` or
  `NTSE_NVRTC_DLL`). Without it the lane reports NODEV and AUTO moves
  on to OpenCL.

## Erratum for DICTIONARY-COMPUTE.md (reported, now fixed)

It says the DTC "win holds while M stays under about K·B/(B−1)". The
inequality is reversed. Per segment, DTC costs K·B MACs to build the
tables plus about M lookups, and dense costs M·B MACs. So DTC wins when
**M > K·B/(B−1)** (≈257 at K = B = 256), i.e. for tall matrices, and
loses for short ones (Gizmo's correction). Benchmark section b measures
it. DICTIONARY-COMPUTE.md has since been corrected, with a note.
