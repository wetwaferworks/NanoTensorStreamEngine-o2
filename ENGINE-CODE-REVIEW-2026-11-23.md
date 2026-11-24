# Code review: nq2-infer.c (and one runner fix), 2026-11-23

Reviewer: E.L.F.I.E. (author reviewing her own engine at JD's
direction; the same standard as the quantizer review: conversion
errors, memory errors, bad practice, bad type casting).
Direction and executive planning: JD.

## Fixed during the original build session (recap)

1. Directory scan compared the ".safetensors" suffix at offset
   L-13; the suffix is 12 chars. Every shard was rejected ("no
   safetensors"). Fixed to L-12.
2. Prefetch race: pipe_get's synchronous fallback load did not
   claim the busy flag, so the prefetch thread could start the
   same load into the same slot concurrently. Fixed: the fallback
   claims busy under the lock before loading.
3. Rotary scratch `hold[64]` silently assumed head_dim <= 128.
   Fixed with a startup geometry check that refuses head_dim
   over 128 or head counts that do not divide evenly.

## Findings from this review

### Memory safety

4. **Shard table overflow.** The model-directory scan accepted
   up to 16 safetensors files into `names[16][512]`, but
   `g_shards` holds MAX_SHARDS = 8 FILE pointers, and
   shard_open_raw writes g_shards[shard_idx] unchecked. A
   directory with 9 or more shards would write past the array.
   Sigma has 2 shards, so it never fired here. Fixed: the scan
   now caps at MAX_SHARDS.

### Correctness

5. **Packed book header never verified.** packed_open read the
   book's version/block/entries/n_books fields and discarded
   them. A stream paired with the wrong book would have produced
   silent garbage. Fixed: all fields are checked against the
   stream header for both format versions; a mismatch is a hard
   error, matching the runner's behavior.
6. **Cache conversion truncated instead of rounding.**
   f32_to_bf16 dropped the low 16 bits, biasing every cached
   key and value toward zero. The reference's cast rounds to
   nearest even. Fixed with the standard RNE add-and-shift.
7. **Runner book-size accounting off by 4 (nq2-run.c).** The
   v1 book header was charged as 8 bytes; the quantizer writes
   12 (magic + block + entries), confirmed against the file on
   disc (262,156 = 12 + 262,144). Fixed to 12. Expansion is
   untouched: the rebuilt runner reproduces checksum
   ebd6d7651112c3fd on the layers 0-3 block-256 slice, identical
   to the pre-fix runner.

### Performance (JD's real-time objective)

8. **The diffusion head was restreamed from disc once per
   position.** head_row streamed all 131,072 head rows for each
   of the B positions in a block, so one denoiser pass read the
   805 MB head B times (64 times per block step with guidance).
   Restructured to head_all: tiles are read once and multiplied
   against all positions. Same arithmetic per element, B times
   fewer disc reads.
9. **The output prior was computed twice.** (alpha/sigma_sq) *
   (z @ E^T) does not depend on the hidden state, but it was
   recomputed inside both the conditional and unconditional
   passes. Now computed once per step as a single matmul and
   added to both logit sets.

### Verified correct (checked, not assumed)

- Gamma endpoint order: the tensor gamma_bounds.endpoint_values
  reads (-5.415, 4.855) directly from the shard. Under the
  reference convention alpha^2 = sigmoid(-gamma), the grid from
  gamma_1 (4.855, near-pure noise) to gamma_0 (-5.415, near
  clean) runs in the right direction, and the DDIM/DDPM
  transition coefficients stay in range.
- Pipe buffer lifetime: no weight pointer is held across a
  second pipe_get call anywhere in encoder_layer; slot parity
  reuse is safe.
- Attention masking and softmax: masked scores cannot win the
  max, are zeroed before the weighted sum, and a fully masked
  row is impossible (a query always attends its own key).
- Self-conditioning carries the raw conditional mean, never the
  guided estimate, per the reference.
- Sampler state and transitions are fp64 throughout.

## Noted, not fixed

- Directory listing shells out to `ls` via popen. It should use
  opendir/readdir; a directory path containing a quote breaks
  it. The Windows build still cannot list a model directory
  (returns an error asking for a Linux checkout). Both are
  recorded for the portability pass.
- The safetensors shape parser reads at most 4 dimensions and
  treats tensors as [rows][cols]; sufficient for this model
  family, not a general parser.
- Unknown --sampler or --schedule values silently select the
  default instead of erroring.
- packed v2 still maps stream segments to tensors by directory
  order (the quantizer's packing order). Format v3 should embed
  names and shapes; until then, filtered or budget-truncated
  packs are not safe inputs to the engine.
- Several hot paths malloc per call (raw range reads, attention
  scores). Acceptable at current scale; revisit if profiling
  convicts them.

## Runtime verification status

The pre-review binary's smoke test passed end to end (32 token
IDs from the raw model, 10m46s on the throttled bench).

## Addendum, same night: the busy-claim fix was incomplete

The first post-review smoke test FAILED in 9 seconds: "pipe:
read failed for encoder.layers.9.self_attn.v_proj.weight".
Root cause: fix 2 above made pipe_get claim the busy flag
before its fallback load, but the prefetch worker never checked
busy before starting a load of its own. When the main thread
lapped the prefetcher (prompt encoding has n=1, so the matmuls
between gets are tiny), both threads loaded the same ordinal
into the same slot at once, and their seek/read pairs on the
shared shard FILE* interleaved, producing short reads. The
"pipe buffer lifetime: verified" line above was about buffer
reuse, not about this claim protocol, but the review should
have caught that the claim had no reader. Corrected: the
worker's wake predicate now includes !busy, so a claimed load
is honored by both sides.

Retest result: the corrected binary passed the identical smoke
test in 5m45s against the pre-review binary's 10m46s on the
same throttled bench (about 1.9x), and produced the identical
32 token IDs, all positions, seed 42. The review fixes (bf16
rounding, head batching, prior-once) changed speed and memory
safety without moving the output on this test.
