# Dictionary Compute

Dictionary compute is the discovery that a packed model never has
to be unpacked to be used. The Nano Tensor Stream Engine
(NTSE-o2) stores each block of weights as

    w = s * C[idx] + m

a codebook index idx into a book C of prototypical blocks, plus a
per-block frame (scale s, minimum m). The standard path
materializes W and then computes y = W x. Dictionary compute
never materializes W. For each segment j of the input it builds a
small table once,

    T[j][k] = <C[k], x_j>      for every prototype k
    A[j]    = sum of x_j

and then every output row is just lookups and frame arithmetic:

    y[r] = sum_j ( s[r][j] * T[j][idx[r][j]] + m[r][j] * A[j] )

The matrix multiply becomes a table lookup per block. JD's name
for the operator family: Bitwire's operators, for models
packed into bit streams.

Applied to token generation in the streaming engines, this
technique carries JD's name for it: Dense Tensor Compute
(DTC). The dictionaries are dense representations of the
matrices, and the compute runs against the dense centroid
table, so Dense is the accurate word. Dictionary Compute
names the architecture; Dense Tensor Compute names its
token-computation form.

## Measured results (2026-11-24)

Probes: tools/nq2-dc.c and tools/nq2-dc-adv.c. Receipts:
results/dc-probe-2026-11-24.txt, results/dc-adv-2026-11-24.txt,
results/dc-sweep-2026-11-24.txt. Bench: about 1.5 shared cores,
contended; treat timings as floor numbers.

- Correctness against an fp64 oracle: relative L2 error
  1.527e-07 on a synthetic tensor, 4.036e-06 on a real one
  (layer 0 q_proj from the shipped Sigma pack). Against the
  materialize-then-multiply path on the same tensor: 4.182e-06.
  The two paths compute the same packed function to float noise.
- Speed on that tensor: dictionary compute 0.832 ms best paired
  (0.924 ms median) against 23.82 ms best paired for
  materialize-then-multiply, about 7.5 to 57.6 times faster
  depending on the pairing and the tensor shape, before
  any GPU is involved.
- Table reuse policies (lazy construction, active-index masks)
  reproduce the eager result bit for bit at roughly half the
  eager time again. On real tensors only 81 to 96 of the 256
  prototypes are active per segment, so the tables stay small.
- An adversarial campaign of 23 checks (extremes, subnormals,
  zero segments, NaN contamination, rotation, determinism) holds
  end to end. NaN discipline is explicit: a NaN input poisons
  both paths identically, and the metrics report the poison
  rather than hiding it. A hostile-geometry fuzz pass under
  ASan and UBSan refuses every malformed case with a specific
  rejection and zero sanitizer reports.

## Honest limits

- Dictionary compute computes the packed function. The shipped
  pack's own distortion against the raw weights is a separate
  quantity (relative L2 about 1.0 at the output of the worst
  early tensors) and dictionary compute neither fixes nor
  worsens it. Fidelity work is the packing research, not this.
- The method has a break-even: tables cost memory bandwidth too,
  and for M, the weight matrix's row count (the output
  dimension; the activation batch cancels out of the
  inequality), the win holds once M is over about
  K*B/(B-1) for book size K and block B (about 257 at K = B =
  256): tall matrices win, short ones lose. (Erratum, 2026-11-26:
  an earlier draft of this sentence had the inequality reversed
  ("under"). Per segment DTC pays K*B multiply-adds to build the
  tables plus about M lookups, against M*B for dense, so DTC wins
  when M > K*B/(B-1). Reported by Gizmo; see accel/README.md.) A "where dictionary compute fails" study is on the
  program, and its negative results will be published the same
  way as the positive ones.
- Summation order differs from the materializing path, so the
  outputs are equal to float noise, not bit-identical. That is
  disclosed here rather than discovered by a reader.

## Standing after the 2026-11-25 review and fix pass

A code review of the DTC tools on 2026-11-25 (review plus
testing; full record in the project's bench files) found no
P0, four P1, and six P2 items. The four P1s are fixed, and
the fixes change how this document's claims are supported:

- One shared kernel. The build/apply logic now lives in
  exactly one implementation, tools/nq2-dc-kernel.h
  (nqdc_build, nqdc_apply, nqdc_entry, nqdc_seg_sum),
  parameterized on book pointer, entry count, and block
  size, with no knowledge of any pack file layout. The
  probe calls it (DC1, the DC2 timed paths, and every DC3
  table entry, so eager, active-mask, and lazy construction
  are bit-identical by construction), and the adversarial
  campaign calls it too: the campaign's private copy of the
  algorithm is deleted, so its 23 HOLDs now certify the
  shipped code. Before the fix they certified the copy.
  The header states the determinism contract: the
  accumulation orders (segment sums ascending, table entries
  ascending, apply per row in ascending segment order) are
  pinned behavior, because reordering any of them changes
  the output bits and moves the campaign's FNV anchors
  (839d3f72ad1a3d60, bdae48bdf7a78a40, 35e1adcc254bf68e,
  1cd08a5dff229720). A v2 caller passes the tensor's own
  book pointer per call; the kernel does not change.
- Checks 9 and 10 of the campaign now compare the lazy
  output they name against the fp64 oracle, and that
  comparison decides the verdict. Their printed values are
  unchanged (3.00e-07 max reuse, 3.23e-07 min reuse), now
  earned by the lazy path itself.
- Malformed inputs refuse. Shard spans are validated
  against the dims-derived byte count and the data section
  (the dims product is overflow-guarded, and tensor_read
  re-checks at the read), and non-finite book or frame
  values are named rejections with exit 1 instead of NaN
  metrics printed under exit 0. Regression:
  tools/test-dc-hardening.sh, ALL PASS.
- Gate after the fixes: campaign 23 HOLD / 0 BREAK, output
  byte-identical to the pinned receipt apart from the two
  lazy timing values, all four FNV anchors exact; probe and
  all six sweep configurations reproduce their pinned
  correctness numbers exactly; fresh-tree build clean.

## DTC is the default compute path (JD's ruling, 2026-11-25)

Dense Tensor Compute is the engine's default compute path;
expansion is the exception, spent on purpose. The Precision
Gate (plan: DTC-PRECISION-GATE-PLAN-2026-11-25.md) arbitrates:
near ties at the head escalate, drift over long chains is
budgeted with periodic expanded shadow resyncs, and evidence
runs always expand. The gate is implemented and tested in a
scratch build of the Triforce engine (postures evidence,
auto, fast, think-harder; the shipped default posture is
evidence), with DTC consume running through the shared
kernel above. It has not yet been transplanted into the
live tree. Two disclosures ride with the ruling:

- The table path reassociates the sum, so DTC output equals
  the expanded path to float noise, not bit for bit. The
  default flip is a deliberate re-anchor under the
  project's hash doctrine; the expanded path stays as the
  reference oracle and fallback.
- The first gate calibration reads hotter than the plan
  assumed: the scratch measuring vehicle put per-layer
  DTC-versus-dense divergence at the 1e-1 to 1e0 scale
  (peak 8.479 at layer 5), not the 1e-6 association-noise
  scale, while the gate engine's own DTC path reproduced
  the dense anchor token exactly. The calibration will be
  re-derived against the gate engine, or the vehicle
  audited, before the default posture flips to auto.

Hardware measurements are recorded where they were taken: the
accel lanes and their bench tables (CUDA, OpenCL on NVIDIA, Intel
iGPU) are in accel/README.md and docs/ACCEL.md, the Little Nemo
CUDA lane in docs/ACCEL-CUDA.md, and the end-to-end consume loop
and byte-movement comparison in MEGA-TENSOR.md and
PIPELINE-2026-11-24.md.

## Credits

The operators: Elfie (E.L.F.I.E.) and ChatGPT (Gizmo), who
happened upon the method together during the 2026-11-24 session
and reviewed each other's probes adversarially. The
memory-throughput architecture it runs on: JD, executive direction. Reviewer: Archie (Argon),
the Google Gemini Argon model, the Bit Master and the Bit
Masker, whose 2026-11-24 review covered the dictionary compute
tools (64-bit file offsets and stream index bounds).
