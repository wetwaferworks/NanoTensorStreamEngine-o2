# GPU quantizer: `nq2-quantize --gpu`

`nq2-quantize` can run its nearest-code search on an NVIDIA GPU. The output is byte-identical to the CPU path: the same `.nq2`, `.nq2book`, `.nq2manifest` and report, for any thread count. The host side is pure C, and the CUDA kernels are compiled at run time by NVRTC, the same pattern the engine's CUDA lane uses ([ACCEL-CUDA.md](ACCEL-CUDA.md)). Implementation: Cortana, on Elfie's quantizer, at JD's direction.

**Status:** landed on 2026-11-27 in the scratch tree. The GPU lane is off by default and only exists in the GPU build.

## Why the GPU

A read-only measurement of the 2026-11-27 SuperVL quantize run (124.45 B weights, 12 threads, per-tensor books, RMS frame, PQ d4) showed that the run is limited by computation, not by reading from disk:

| | Measured |
|---|---|
| CPU | about 10.4 of 12 logical cores busy on average (user time 174,000 s over 16,700 s of wall time); kernel time about 1 % |
| Disk reads | about 20 MB/s over a 60 s sample, about 28 MB/s averaged over the run; drive D: about 35 % busy |
| Bytes read | about 416 GB by 92 % of the run. Each tensor is read twice (once to sample, once to pack), so a full run reads about twice the 232 GiB of shards |

Nearly all of that compute is the nearest-code search: k-means over each tensor's sample (64 sub-books x 8 iterations x up to 8,192 blocks x 256 codes) and then packing every block against its sub-books. That search is what moved to the GPU. Once it is fast, the floor becomes the two passes over the shards on disk.

## Build and run

```
accel\build-gpuquant.bat
build\gpuquant\nq2-quantize.exe --gpu  [the usual options]
```

The build needs only MinGW gcc. It writes into `build\gpuquant` and never touches the plain build (`build.bat`). The binaries load `nvcuda.dll` and NVRTC (`nvrtc64_*.dll`, found through `CUDA_PATH` or `NTSE_NVRTC_DLL`) at run time.

| Flag | Meaning |
|---|---|
| `--gpu` | nearest-code search on CUDA; everything else stays on the CPU |
| `--gpu-verify` | the same, and every GPU result is checked again on the CPU; the run stops with exit code 3 at the first difference (for testing; slow) |
| `--gpu-device N` | CUDA device index (default 0) |

Without a usable GPU, `--gpu` prints why and exits with code 1. The plain build refuses `--gpu` and names the GPU build. At exit, the GPU build prints how many searches it ran, how many were rescored in double because of near ties, how many took the full double search, and how many fell back to the CPU.

## What runs where

| Step | Where | Order |
|---|---|---|
| Reading shards, sampling (reservoir RNG), normalizing blocks | CPU | unchanged |
| Initial codes for k-means (RNG draws) | CPU | book 0 first, as before |
| Nearest code for each sample row, each k-means iteration | **GPU** | exact (below) |
| Centroid sums and the centroid update | CPU | serial in sample order, as before |
| Nearest code for each block when packing | **GPU** | exact (below) |
| Frames, first-bad-frame check, fidelity sums, writing | CPU | block order, as before |

The PQ path sends all 64 sub-books of a tensor to the GPU in one call for each k-means iteration. Packing goes in chunks of 8,192 blocks. In the per-tensor parallel path, each worker thread has its own CUDA stream and its own device buffers, and nothing is shared or locked.

## Why the bytes match

The CPU picks, for each subvector, the first code k (ascending, strict `<`, starting from best 0 and error 1e30) with the smallest double sum of `((double)x[i] - (double)c[i])^2`, added in the order i = 0..d-1. The GPU returns that same index in two passes per row:

1. **An fp32 screen.** It computes each distance in fp32 and turns it into an interval `[lo, hi]` that is proven to contain the CPU's double value. The margin is `eps = (d + 8) * 2^-22` relative, plus `A = (d + 8) * 2^-140` absolute for underflow. That is about four times the worst-case rounding of a d-term sum of squares in fp32 and in double, and the bounds themselves are computed with round-up and round-down operations. `U` is the smallest `hi`.
2. **An exact rescore.** Any code with `lo <= U` could be the winner. Every other code has a double distance greater than `U`, which is at least the smallest one, so it can never win. Each candidate is rescored in double with the CPU's exact operations and order (`__dsub_rn`, `__dmul_rn`, `__dadd_rn`; no FMA), in ascending k with a strict `<`. So ties are broken the CPU's way.

If any fp32 distance is NaN or overflows, that row skips the screen and runs the full double search. NVRTC compiles with `--fmad=false --ftz=false`. Each thread writes only its own output byte, and there are no atomics.

## Tests

| Test | What it checks |
|---|---|
| `build\gpuquant\test_gq_assign.exe` (`accel\tests\test_gq_assign.c`) | Compiles the quantizer source in unchanged and compares its own `nearest_centroid()` and `learn_book()` with the GPU lane: 975 assignment cases (widths 1 to 4096, 1 to 256 codes, 1 to 3,000 rows, plus the SuperVL shape at full chunk size) over 13 value patterns: Gaussian, duplicate codes, rows equal to codes, one-ulp near ties, grid values, subnormals, underflowing squares, fp32 overflow, NaN and infinity, all zero, mixed magnitude, RMS-normalized, all codes equal. Also 100 k-means cases, flat and 64-book PQ, where the books and the RNG state afterwards must match. `--heavy` makes the cases larger |
| `accel\tests\gpuquant-e2e.ps1` | Synthetic safetensors from `gq_mkfixture` (BF16, F16 and F32; ties, zeros, constants, subnormals, a partial-block tail, and a pack the quantizer must refuse), 11 recipes (global, per-tensor flat v2 and v3, the SuperVL recipe, PQ 8, a small PQ, budgets, residual, block 64, global block 1024), at 1, 4 and 12 threads. Pack files must match the 1-thread CPU reference, and every GPU run must match the CPU run at the same thread count, report included. With `-Shards` it runs on real checkpoints, and `-Prune` deletes the packs after hashing |

The unit test was itself checked against three broken kernels: a flipped tie-break, a screen with no error margin, and an fp32-only winner. It failed all three.

### Results (2026-11-27, RTX 3050 Ti Laptop 4 GB)

- `test_gq_assign`: **1,075 of 1,075 cases byte-identical**, 27.8 M row comparisons. 11.4 M rows had near ties and went to the double rescore, and 2.2 M took the full double search (the NaN, infinity and overflow patterns).
- Fixture e2e, first pass on one seed (run while the SuperVL quantize held the CPU, so the times can't be compared): the plain build of this source matched the shipped `nq2-quantize.exe` on all 9 recipes it reached, report included. On the 8 recipes the GPU path reached, its pack files matched the 1-thread CPU reference. A later quick pass of the small PQ recipe at 1 and 4 threads, including the pack the quantizer must refuse, also matched.
- The full fixture pass (4 seeds, 1, 4 and 12 threads, `--gpu-verify` too), the Sigma-3B pack comparison with wall times, and a re-check of the pinned anchors run after the SuperVL post-pack measurements, so the two don't compete for the GPU and CPU. Their numbers get added here.

## Open issues

- The global-book packing pass (`pack_worker`) and the residual second stage still search on the CPU. Their k-means already uses the GPU.
- On the CPU side, a known difference unrelated to the GPU: the parallel per-tensor writer (`--threads` above 1) leaves the line `product quantization: subvector dim 4, 64 sub-books per tensor` out of the report, while the serial path writes it. The pack files themselves are identical.
