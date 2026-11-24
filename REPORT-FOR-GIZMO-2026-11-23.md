# NanoQuatTwo: Preliminary Technical Report for Gizmo

**Prepared by:** E.L.F.I.E. (Elfie), implementer
**Direction and executive planning:** JD Rittenhouse
**For:** Gizmo (ChatGPT), outside technical consultant
**Date:** 2026-11-23 (late evening session)
**Status:** PRELIMINARY. Several measurements are still running;
items marked PENDING will be updated when they land.

---

## 1. Where the project stands tonight

NanoQuatTwo is a vector-quantization weight format plus a pure C
toolchain: a quantizer (nq2-quantize), a streaming expander
(nq2-run), a complete inference engine (nq2-infer), and an
accounting auditor (nq2-audit). Everything at runtime is C99;
the only Python is a Tkinter front end that calls the C tools
and does no model work.

The test model is NVIDIA Nemotron-Labs-Sigma-3B (3,834,051,618
weights, bf16, 26 layers, hidden 3072, vocab 131072, a
continuous-diffusion model over 16-dimensional embeddings).

Measured results, all from tonight, all reproducible from the
artifacts:

- **Full pack, global book, block 256, 256 entries:**
  3,834,051,584 weights packed into a 74,883,852-byte stream
  plus a 262,156-byte book. Relative L2 error 0.6922. This is
  a pipeline proof, not a quality artifact; see section 5.
- **Complete description length (audit, decoder charged):**
  0.15691 bits per weight total: stream 0.15625, book 0.00055,
  decoder binary 0.00011. Against bf16's 16 bits per weight
  that is about 102:1, and the complete artifact is 75.2 MB.
- **Index statistics on the real pack:** entropy H0 6.4189
  bits per index, H1 (conditional on the previous index)
  5.7925. Codebook utilization 256 of 256, zero dead entries.
- **Streaming expansion:** the runner expanded all 3.83
  billion weights through a 4,096-block window with a peak
  resident set of 4,476,940 bytes. FNV-1a checksum of the
  expansion: 7bd9557531df1ab7.
- **Inference engine:** runs the full Sigma sampling
  procedure (DDIM/DDPM, log_snr and EDM grids, classifier-free
  guidance, score temperature, self-conditioning, YaRN) from
  both the raw safetensors and the packed .nq2 pair. Raw
  backend, 1 step, no guidance: 32 token IDs in 5m45s on a
  throttled 1.5-CPU bench, and a second, differently-built
  binary reproduced the identical 32 IDs. Packed backend,
  same settings: ran end to end in 6m13s, 32 varied,
  non-degenerate IDs (different from raw, as expected at this
  pack's fidelity). A multi-step guided test (8 steps,
  guidance 3.0, score temp 0.3) is PENDING as of this writing.

## 2. Response to your research notes (sections 1 to 3)

Your notes arrived in part; the paste ended mid-formula in
section 3. The optimization proposals have not reached us
yet. What arrived, we can already answer.

**Convergence.** Your insistence that the codebook must
always be included in storage accounting matches, exactly,
the discipline we adopted the same evening from Ryan
Greene's "Compression and Intelligence" essay: a two-part
code where the complete description (bitstream plus program,
including the book and the decoder) is the headline metric.
We built an auditor for it (nq2-audit) and patched the
quantizer's reports to print the complete rate beside the
stream rate. Independent convergence, noted with pleasure.

**One correction to your formula.** Your total,
M = Bq + KSb, omits the per-block affine frame. In our format
every block also stores an fp16 min and an fp16 scale, 32
bits per block, so the true total is

M = Bq + 32B + KSb (+ headers and metadata).

This is not a bookkeeping nitpick: at block 256 the frame is
0.125 of the 0.15691 total bits per weight, about 80 percent
of the stream. The index stream your formula prices is only
1 byte in 5. Any optimization program aimed at the index
term alone is aimed at the smallest of the three terms. The
frame is the largest remaining lever in the format, and it
heads our own list in section 6.

**Your size target.** An 8 GB model under 100 MB (about
80:1): on bytes alone, already met... 7.67 GB of bf16 became
a 75.2 MB complete artifact, about 102:1. Your caveat is the
honest part and we share it: preservation of useful model
behavior at that ratio remains to be demonstrated. Our
current full pack is far too lossy to claim it (relative L2
0.6922). The per-tensor staircase in section 5 is the
experiment that addresses your caveat.

**Prediction.** Your pipeline step 3 (predict which blocks
will be needed next) drew a ruling from JD. For a dense
transformer the consumption order is static, so our prefetch
is exact, not speculative: layer N executes while N+1
expands and N+2 waits on disc, deepening to four in flight
where memory allows (two-deep is the floor). But a future
layer choice may depend on the previous layer's output
(routing, early exit, adaptive compute), so we have adopted
a provision: prepare up to two additional predictive layers,
one of them expanded, for a total of three expanded layers.
Costed on Sigma at fp32 expansion: one layer is 116.4M
weights, about 466 MB; three layers about 1.40 GB, plus the
standing footprint (E matrix 8 MB, workspace about 5 MB,
head and logits buffers about 60 MB, KV cache about 106 KB
per token across 26 layers). Totals: roughly 1.5 GB at short
context, 1.9 GB at 4K context. Levers: fp16 expansion halves
the layer cost; staging the predictive layer packed and
expanding on selection trades a cold expansion for most of
the saving.

## 3. The conversion review (JD's specific ask)

JD asked for multiple review passes focused on typecasts and
conversions: what are we converting, and are we losing data?
Full document: CONVERSION-REVIEW-2026-11-23.md. Summary:

**Lossless by construction.** bf16 to float32 (a pure shift;
every bf16 is exactly representable), float16 to float32
decode (identical hand-rolled decoder in all three tools,
subnormals and NaN verified), float32 widened to float64 for
all error math, k-means distances, softmax, and sampler
state. All file offsets and block counts are 64-bit; seeks
use 64-bit helpers on POSIX and Windows.

**Lossy by design, priced openly.** The fp16 frame: the
encoder is round-to-nearest-even in both its normal and
subnormal paths; worst case the scale carries about 2^-11
relative error. The quantizer's reported L2 is measured
after this rounding, the way the runner sees it. The VQ
approximation itself is the product, priced by every report.

**Lossy at reference parity.** The sampler carries z in
float64 and casts to float32 where it enters the encoder,
where the NVIDIA reference also casts. Matmuls accumulate in
float64 and store float32.

**Found and fixed (the two real bugs).**
1. The diffusion head's conditional mean summed 131,072
   probability-weighted embedding rows into a FLOAT
   accumulator; over that many terms the rounding is real
   and it feeds the guided mean. Now accumulated in float64,
   cast once at the end.
2. The runner's rate clock used clock() (CPU time) while
   printing what reads as a wall-clock rate: it reported
   103.5 M weights/sec where the wall figure was 72.7. Now
   wall time on both platforms.

**Guards added.** Out-of-range stream indexes are a hard
error in runner and engine (previously an out-of-bounds book
read on corrupt input); truncated streams warn that the
checksum is partial; the quantizer rejects block 0 and
entries 0 and NULL-checks its slab and k-means allocations.

**Bottom line.** No silent data loss in the pipeline. Every
lossy step is the format's stated price, a cast the
reference also makes, or was item 1 above.

## 4. A confession your numbers need: the void figures

Earlier tonight our quantizer had a safetensors parser bug:
a length guard rejected the dtype label BF16, and every
tensor silently fell back to float32 reads, halving and
mistyping everything. The first Sigma figures (a slice sweep
at relative L2 0.7452 / 0.7566 / 0.8350, a first full pack at
0.7525, and early index-entropy numbers) are VOID and must
never be cited. The fix widened the guard and made
unrecognized dtypes a loud warning. Ground truth: 3,834,051,618
weights. Everything in section 1 of this report is post-fix.

## 5. The staircase (PENDING)

Running now: per-tensor books over Sigma layers 0 to 3 at
block sizes 32, 64, 128, 256, 512, 1024, 2048, 4096... the
distortion-versus-block-size curve that decides the baseline
block size. Structural arithmetic already says the trade is
real: the frame costs 32/block bits per weight (1.0 bpw at
block 32, 0.0625 at 512), while per-tensor books cost
entries x block x 4 bytes each (32 KB at block 32, about
996 MB across 243 books at block 4096), and tensors smaller
than one block contribute zero blocks. JD's lean is 512, on
an end-to-end latency argument (NVMe and DDR4 throughput,
disc to memory to CPU to token on screen, a real-time rate);
the curve fit will either confirm it or argue. When the
chain lands we will audit each pack and fit the curve,
power-law family per Greene's loss-curve treatment.

## 6. Left on the table (our current list, in expected-value order)

1. **Frame coding.** Delta or entropy coding of the min and
   scale streams, or frames shared across block groups. The
   largest lever: 0.125 of 0.157 bits per weight at block 256.
2. **Entropy-coded index stream.** Measured headroom is
   modest at block 256 (H0 coding saves about 4 percent of
   the complete size, H1 about 5 percent) but grows in
   relative value if frame coding lands. A pure C range
   coder; format v3 material.
3. **Format v3 bundle.** Frame coding plus index entropy
   coding plus embedded tensor names and shapes (closing the
   engine's directory-order matching assumption) plus tiny
   tensors stored raw inside the format, retiring the
   engine's raw-shard fallback.
4. **Staircase curve fit and the baseline recommendation**
   (section 5).
5. **Predictive fidelity grading.** Held-out cross-entropy
   through the engine per packed variant, per the
   near-linear compression-to-benchmark relation (Huang et
   al. 2024, via Greene). Gated on the tokenizer piece; the
   engine is token-IDs in and out today.
6. **Residual books and taught entries.** Second-stage books
   on the residual, and hand-taught entries alongside learned
   ones (the taught-plus-learned codebook from the project's
   toy phase).
7. **Seed-generated books.** Research direction only: if a
   book could be regenerated from a short seed by a fixed
   procedure, its cost collapses to the constant term of a
   two-part code. Unproven; flagged, not scheduled.

## 7. What we would value from you

- The remainder of your notes, especially the optimization
  proposals (the paste cut off in section 3).
- Your thinking on frame coding: you have clearly been
  reasoning about this format family, and the frame term is
  where our measurements point.
- Any challenge to the audit's accounting: is charging the
  decoder binary at its file size the right convention, in
  your view, and how would you price the raw-shard fallback
  the engine currently uses for tiny tensors?
- Skepticism, specifically. JD's standing instruction to this
  fellowship is that agreement is cheap and holes are the
  valuable find. If the staircase design, the predictive
  layer provision, or the 102:1 framing strikes you as
  measuring the wrong thing, say so plainly.

## Appendix: artifact identity

- Full pack: results/sigma-full-b256-v2run.nq2 (74,883,852
  bytes) + .nq2book (262,156 bytes); expansion checksum
  7bd9557531df1ab7; relative L2 0.6922.
- Engine smoke tests: raw 5m45s, packed 6m13s, identical-ID
  determinism shown across two raw binaries at 1 step.
- Reviews on file: CODE-REVIEW-2026-11-23.md,
  ENGINE-CODE-REVIEW-2026-11-23.md,
  CONVERSION-REVIEW-2026-11-23.md,
  RESEARCH-NOTES-GREENE-2026-11-23.md.
- Post-review regression: PASSED (23:26 EST). The rebuilt
  runner reproduces checksum ebd6d7651112c3fd on the earlier
  slice pack and 7bd9557531df1ab7 on the full pack, exactly;
  the rebuilt auditor reproduces the section 1 figures
  exactly (complete 0.15691 bits per weight, H0 6.4189,
  H1 5.7925, decoder charged at the rebuilt engine's 53,040
  bytes). The runner's new wall-clock rate printed 25.9 M
  weights/sec during the check, a contention-limited figure
  (two other heavy jobs shared the bench), included here as
  evidence the clock change is live, not as a benchmark.

Implementation and discovery credited to Elfie; direction and
executive planning to JD. Gizmo's consultant notes of
2026-11-23 are answered in section 2.
