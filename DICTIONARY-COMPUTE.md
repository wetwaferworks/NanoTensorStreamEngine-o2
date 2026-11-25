# Dictionary Compute

Dictionary compute is the discovery that a packed model never has
to be unpacked to be used. The Nano Quant Engine stores each
block of weights as

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
nanoquanted into bit streams.

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
  materialize-then-multiply, about 16 to 28 times faster
  depending on the pairing, before any GPU is involved.
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
  and for batch width M the win holds while M stays under about
  KB/(B-1) for book size K and block B (about 257 at K = B =
  256). A "where dictionary compute fails" study is on the
  program, and its negative results will be published the same
  way as the positive ones.
- Summation order differs from the materializing path, so the
  outputs are equal to float noise, not bit-identical. That is
  disclosed here rather than discovered by a reader.

## Credits

The operators: Elfie (E.L.F.I.E.) and ChatGPT (Gizmo), who
happened upon the method together during the 2026-11-24 session
and reviewed each other's probes adversarially. The
memory-throughput architecture it runs on: JD (David
Rittenhouse), executive direction. Reviewer: Archie (Argon),
the Google Gemini Argon model, the Bit Master and the Bit
Masker, whose 2026-11-24 review covered the dictionary compute
tools (64-bit file offsets and stream index bounds).
