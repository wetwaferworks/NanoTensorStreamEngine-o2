# NanoQuatTwo: Conversion and Typecast Review, 2026-11-23

Requested by JD: multiple code review and bug fix runs over the
tool suite, looking for the usual suspects, mainly CONVERSION...
what types are we typecasting and converting, and are we losing
data? Review and fixes by Elfie. Direction by JD.

Scope: tools/nq2-quantize.c, tools/nq2-run.c, tools/nq2-infer.c,
tools/nq2-audit.c. (nq2-gui.py does no numeric work by design; it
only calls the C tools.)

## The conversion inventory

Every numeric conversion in the suite, and its verdict:

### Lossless by construction (no data lost, ever)

- bf16 to float32 (quantizer tensor_read, engine shard reads):
  a pure 16-bit left shift into the float's high half. Every
  bf16 value is exactly representable in float32. Sigma's
  weights take this path.
- float16 to float32 (all three C tools that decode frames):
  the decoders in nq2-quantize.c, nq2-run.c, and nq2-infer.c
  are the same hand-rolled implementation, verified identical
  in logic, including the subnormal normalization loop and
  the infinity/NaN cases. Every float16 value is exactly
  representable in float32, so decode loses nothing.
- float32 to double (error math, k-means distances, softmax,
  sampler state): widening, exact.
- Integer widening in offsets and counts (uint32 to uint64,
  size_t products): all file offsets and block indexes are
  carried in 64-bit types; seeks use 64-bit helpers on both
  POSIX and Windows.

### Lossy, and by design (the format itself)

- Block min and scale stored as float16 (the frame). The
  encoder (quantizer float_to_half) is proper
  round-to-nearest-even in BOTH the normal path (round bit,
  guard bit, sticky bits all tested) and the subnormal path
  (the sticky-mask bug found in the first review is fixed and
  re-verified at the 2^-25 boundary). Worst case: the scale
  carries a relative error of about 2^-11 (0.05 percent), the
  min an absolute error of half an fp16 ulp at its magnitude.
  This is the format's known price, and the quantizer's
  reported relative L2 is measured AFTER this rounding, the
  way the runner will see it. Nothing is hidden.
- The VQ approximation itself: a block is replaced by
  min + scale * prototype. That is the product, priced openly
  by every report.
- Block tails and tensors smaller than one block are not
  packed (dropped by design; the engine serves them from the
  raw shards). 34 weights of 3.83 billion on the Sigma pack.

### Lossy, matching the reference implementation

- Engine sampler: the latent z is carried in float64 (as the
  reference does) and cast to float32 where it enters the
  encoder input, exactly where the reference casts too.
- Engine matmuls accumulate in float64 and store float32.
- Rope tables and the gamma embedding are computed in
  float64 and stored float32.
- Diffusion head: the prior term z @ E^T is computed with z
  cast to float32 first, matching the reference's dtype flow.

### Lossy, and WRONG (found by this review, fixed)

1. **Diffusion head mean accumulation (nq2-infer.c).**
   mean_from_logits summed 131,072 probability-weighted
   embedding rows into a FLOAT accumulator. Over that many
   terms the rounding error is real and it feeds the guided
   mean, the sampler's center of mass. Fixed: accumulate in
   float64, cast once at the end. A load-time guard now
   rejects embed_dim above 64 (the accumulator's width);
   Sigma's is 16.
2. **Runner rate clock (nq2-run.c).** now_sec used clock(),
   which is CPU time, while the printed "expansion rate"
   reads as a wall-clock disc-to-memory rate. On the full
   pack the CPU figure said 103.5 M weights/sec where the
   wall figure was 72.7. Fixed: wall clock on both platforms
   (clock_gettime MONOTONIC / QueryPerformanceCounter).

### Guards added (corrupt input, not conversion, same pass)

- Runner and engine: a stream index at or above the book's
  entry count is now a hard error instead of an out-of-bounds
  book read. Our packs use 256 entries so the byte index can
  never trip it; a corrupt or mismatched pair now fails loud.
- Runner: a truncated stream now warns that the checksum
  covers a partial expansion.
- Quantizer: --block 0 and --entries 0 are rejected (0
  previously slipped past the entries-above-256 check and
  would have divided by zero). Slab buffers in both block
  walkers and the k-means accumulators are NULL-checked.

### Notes, checked and left alone

- The runner reconstructs in float32 (mn + sc * proto); the
  quantizer's fidelity loop reconstructs the same values in
  float64 to score them. The difference is last-ulp noise in
  the REPORTED L2 only; the runner and the engine agree
  bit-for-bit in float32, and the runner's checksum is the
  ground truth of the artifact.
- The quantizer's dsqrt (Newton, 60 iterations) is more than
  sufficient for the error magnitudes it serves (relative L2
  values of order 1); at extreme arguments near 1e30 it would
  converge slowly, a range this tool never sees.
- Centroid updates divide in float64 and store float32: the
  book IS float32, so the cast is the representation, not a
  loss beyond it.
- Tensor dimension fields narrow from uint64 to int in the
  engine's geometry (rows, cols). Sigma's largest dimension
  is 131,072; a hypothetical dimension above 2^31 would
  overflow. Noted for the format v3 work, not a live risk.
- The half decoder exists in three copies (quantize, run,
  infer). They are identical today; a shared header is the
  tidy fix and belongs with the format v3 cleanup so the
  copies cannot drift.

## Bottom line

No silent data loss anywhere in the pipeline. Every lossy
step is either the format's stated price (frame rounding,
the VQ approximation itself), a cast the reference
implementation also makes, or was the float accumulation in
the diffusion head mean... which is fixed. The conversions
JD asked about are, in order of volume: bf16 to float
(lossless), float to half for the frames (round-to-nearest
-even, priced), float64 sampler state to float at the
encoder boundary (reference parity), and float64 accumulators
behind every float32 store.
