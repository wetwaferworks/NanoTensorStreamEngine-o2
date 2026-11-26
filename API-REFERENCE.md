# API Reference: NTSE-o2 command-line and format surface

The documented surface of the Nano Tensor Stream Engine
(NTSE-o2) as it stands in the sources on 2026-11-25. Every
flag below was verified against the tool sources (tools/*.c)
on that date; the Precision Gate section is verified against
the gate implementation, which at the time of writing exists
in a scratch build of the Triforce engine and has not yet
been transplanted into the live tree. That section is marked
where it applies.

Conventions: CLI flags override environment variables, which
override device-derived defaults. The pipeline tools print
every effective value with its source (cli, env, or device)
at startup.

## The engines

Three inference engines run the Sigma-3B packed models. They
share the run-shaping flags:

    --dir DIR            model directory (safetensors shards)
    --packed STREAM BOOK packed stream and book files
    --ids LIST           prompt token IDs
    --tokens N           maximum new tokens
    --steps N            diffusion steps
    --block N            block size
    --sampler NAME       ddim (default) or ddpm
    --schedule NAME      logsnr (default) or edm_sigma
    --guidance F         guidance scale
    --score-temp F       score temperature
    --seed N             seed
    --threads N          worker threads

- nq2-infer: the plain engine.
- nq2-infer-lab: the instrumented engine. Adds --forensics,
  --logits-dump FILE, --logits-dump-all FILE, and
  --restore-tensors LIST (single-tensor restore for
  per-tensor experiments).
- nq2-infer-triforce: the split fetch/expand/consume
  pipeline engine (Mega Tensor, Tensor Handshake stage 2).
  Adds --forensics, --logits-dump, --logits-dump-all, and
  the pipeline geometry below.

## Pipeline geometry (nq2-infer-triforce)

    --pipe-slots N       pipe slots            (env NQ2_PIPE_SLOTS)
    --fetch-ahead N      fetch-ahead depth     (env NQ2_FETCH_AHEAD)
    --prefetch-dist N    prefetch distance     (env NQ2_PREFETCH_DIST)
    --expand-lanes N     expansion lanes, the residency dial,
                         range 1 to 64         (env NQ2_EXPAND_LANES)

Values are range-checked against the hardware ceiling
(--fetch-ahead >= 1, --prefetch-dist >= 0, --expand-lanes
within [1, PIPE_MAX_LANES]) and clamped loudly with the
reason printed when a request passes the ceiling.

## Streaming runner and stream test

- nq2-run STREAM BOOK [peek N]: streams a packed model from
  disc through a rolling window, reporting packed versus
  expanded size, peak resident memory, expansion throughput,
  and a checksum of the full expansion. peek N expands a
  single block on demand. --allow-partial is accepted.
- nq2-streamtest: the four-loop streaming harness.
  Flags: --pack FILE, --book FILE, --shards, --unpackers N
  (0 is the sequential reference), --variant N,
  --lookahead N, --prefetch-dist N, --layers N. Since the
  M-P8 fix, a pack whose bytes fall short of what its header
  promises fails up front, and a short read mid-stream stops
  the pipeline; either way the exit is nonzero and no
  accumulator is published. Regression:
  tools/test-streamtest-truncation.sh.

## The behavioral comparator

- nq2-logitcmp: compares two logits dumps (from
  --logits-dump) and reports top-1 and top-5 agreement, mean
  cosine, relative logit L2, and KL divergence. Accepts
  --allow-partial.

## The quantizer and pack format v3

nq2-quantize writes the pack. Flags verified in source:

    --out PREFIX         output prefix (writes PREFIX.nq2,
                         PREFIX.nq2book, PREFIX.nq2manifest)
    --block N            block size
    --entries N          book entries
    --budget N           weight budget for book learning
                         (full packs pass a large value; the
                         default silently truncates scope)
    --filter STR         tensor name filter
    --sample-blocks N    sample cap for book learning
    --tensor-sample-blocks N
    --books MODE         global or per-tensor
    --iters N            k-means iterations
    --norm MODE          minmax or rms
    --residual           residual second book
    --format N           1, 2, or 3 (default 3)

Format v3 additions (all three require --books per-tensor):

    --pq D               product quantization, subvector
                         dimension D = 4 or 8, RMS sliced,
                         defined at block 256
    --budgets            per-tensor bit budgets: the tensors
                         holding the top 50% of squared error
                         re-pack at block/4, the next 30% at
                         block/2, books re-learned
    --embed-fp16         the embedding tensor stored as raw
                         float16 (default off)

Generation pairing (H3b): a v3 stream and its book carry a
shared generation identity, an FNV-1a 64 computed over the
pack's own contents (geometry, tensor names, book bytes,
packed planes, raw section). Because the identity is
content-derived, re-packing the same inputs with the same
options reproduces the same identity. Every engine checks
the pair at open and refuses a mismatch in plain words,
naming both generations.

Per-tensor digest manifest: <prefix>.nq2manifest, a text
sidecar listing, per tensor, its block range and two FNV-1a
digests: one over the tensor's stored stream bytes (indexes,
then frame minima, then frame scales; the raw bytes for an
fp16-raw tensor) and one over its book's stored floats.
Readers verify both at open when the manifest is present.

Legacy readability: v1 and v2 packs carry no identity.
Readers bind them positionally, exactly as before, and
print a notice that the pack carries no generation
identity.

Reader support by pack kind, as verified in the 2026-11-25
reader sweep: the lab and infer engines run PQ packs; the
Triforce refuses PQ packs at open, in plain words (a
recorded limit). All engines run the v3 global, v3
per-tensor, budgets, fp16, and legacy kinds they accept,
with identity verified where the format carries one.

## Dictionary compute (Dense Tensor Compute)

- nq2-dc PACK BOOK SHARD... : the probe. Positional pack,
  book, and safetensors shard arguments; --match STR selects
  the target tensor (default q_proj); --checksum runs the
  whole-model fingerprint path. Since the 2026-11-25 fix
  pass, malformed shard spans and non-finite book or frame
  values are named refusals with exit 1.
- nq2-dc-adv: the 23-check adversarial campaign. No
  arguments; exit 0 with 23 HOLD / 0 BREAK on success.
- tools/nq2-dc-kernel.h: the one shared kernel. Entry
  points (all static, no allocation; T and A are
  caller-owned workspace):

    float nqdc_seg_sum(const float *xseg, uint32_t block)
        A[j]: the sum of one activation segment.

    float nqdc_entry(const float *book, uint32_t block,
                     const float *xseg, uint32_t k)
        T[j][k]: one table entry, the dot of prototype k
        with the segment. Exposed alone because the lazy and
        active-mask policies build single entries on demand
        and must produce the eager build's bits.

    void nqdc_build(const float *book, uint32_t entries,
                    uint32_t block, const float *x,
                    uint64_t nseg, float *T, float *A)
        Eager build: fill T (nseg * entries floats) and A
        (nseg floats) for activation x against the book.

    void nqdc_apply(const uint8_t *idx, const float *sc,
                    const float *mn, const float *T,
                    const float *A, uint32_t entries,
                    uint64_t nseg, uint64_t rows, float *y)
        y[r] = sum_j ( sc[r][j] * T[j][idx[r][j]]
                     + mn[r][j] * A[j] ).

  The kernel takes the book pointer, entry count, and block
  size as parameters and knows nothing of any pack layout:
  a v1 caller passes its global book; a v2 caller passes the
  tensor's own book pointer per call. Determinism contract:
  segment sums ascend, table entries ascend, apply
  accumulates per row in ascending segment order. Reordering
  any of these changes the output bits and moves the
  campaign's FNV anchors.

## The Precision Gate (scratch implementation, not yet in the live tree)

Implemented in a scratch build of nq2-infer-triforce
(scratch/pgate/impl/) on 2026-11-25; the surface below is
verified against that source. It transplants into the live
tree as its own gated step.

Postures (DTC is the engine's default compute path by JD's
2026-11-25 ruling; the posture decides when expansion is
spent):

    evidence       expanded always, plus pinning and logging
                   of every anchor the run touches. The
                   shipped default posture.
    auto           the gate: DTC default, predictive
                   escalation on near ties and drift.
    fast           DTC always. Never for evidence.
    think-harder   expanded always, no gate.

Selection: --posture NAME, or the environment variable
NQ2_POSTURE. An unknown posture is a named refusal.

Gate knobs:

    --gate-lookahead N   predictive lookahead, 1 to 4 layers
                         (default 2; env NQ2_GATE_LOOKAHEAD)
    --gate-resync N      shadow resync interval in tokens;
                         every N tokens one step runs expanded
                         regardless (env NQ2_GATE_RESYNC)
    --gate-eps F         override the calibrated tie epsilon
                         (positive number; printed as OVERRIDE)
    --gate-budget F      override the calibrated drift budget
                         (positive number; printed as OVERRIDE)
    --gate-log FILE      per-step decision log (ordinal marks,
                         margins, drift estimate)

Calibration carriage: for a v3 pack, gate lines ride in the
manifest after the tensor lines:

    gate 1
    gate-gen <fnv1a64 hex>     must equal the pack generation
    gate-eps <float>           tie epsilon
    gate-budget <float>        drift budget
    gate-layer L <increment> <cumulative>
                               per-layer noise profile

For a legacy pack, the same lines live in a sidecar file
<stream>.nq2gate, keyed by a content FNV over the pack
geometry and stored book bytes. An uncalibrated pack falls
back to expanded; the gate never gates on unmeasured goods.

Gate determinism: every decision is a pure function of
computed values and the calibrated thresholds, never of
wall clock, thread timing, or lane availability. A gated
run reproduces bit-exactly on re-run: same pack, same
prompt, same posture, same tokens.

## Little Nemo engine (littlenemo/)

The Nemotron-Nano-9B-v2 engine speaks a line protocol
(littlenemo/PROTOCOL.md). Verified flags include --mode
minimum|medium|max|ultra (pool lane count), --threads N
(override), --simd / --no-simd (the AVX2 kernel is opt-in
and reached only through a runtime CPUID check), --layers,
--max-new, --max-pos, and --serve / --serve-stdio. The chat
GUI reads littlenemo.ini for the engine, checkpoint, and
tokenizer paths plus the compute mode; arguments override
the file. The engine prints READY only for a checkpoint
that passes whole-geometry validation.

## Release identity (kept separate, always labeled)

Release version: 0.1.0. Rolling build counter: 0001
(Build 0001). Git tag: v0.1.0. The release remains
unpublished at JD's word. Version, counter, and tag are
three different numbers; no document in this set conflates
them.
