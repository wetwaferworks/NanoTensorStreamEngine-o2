# Dense Tensor Compute (DTC)

Dictionary compute is the discovery that a packed model never has to be
unpacked to be used. Dense Tensor Compute is JD's name for the technique applied
to token generation in the streaming engines: the dictionaries are dense
representations of the matrices and the compute runs against the dense centroid
table. Dictionary Compute names the architecture; Dense Tensor Compute names its
token-computation form. (JD's name for the operator family: Bitwire's operators,
for models packed into bit streams.) The operators who found it: Elfie and
Gizmo, together, on 2026-11-24.

## The algebra

A packed block is `w = s * C[idx] + m`: an index `idx` into a book `C` of
prototype blocks, and a per-block frame (scale `s`, minimum `m`). The standard
path materializes `W` and then computes `y = W x`. DTC never materializes `W`.
For each segment `j` of the input (one block wide) it builds a small table once,

    T[j][k] = <C[k], x_j>      for every prototype k
    A[j]    = sum of x_j

and then every output row is lookups and frame arithmetic:

    y[r] = sum_j ( s[r][j] * T[j][idx[r][j]] + m[r][j] * A[j] )

The matrix multiply becomes a table lookup per block. The tables are built once
per activation and reused by every output row, so the cost of the tables is paid
per segment, not per row.

## The shared kernel: `tools/nq2-dc-kernel.h`

One implementation, used by the probe, the adversarial campaign, the
Precision Gate engine and mirrored bit for bit by the GPU lanes. All functions
are static and allocation-free; the caller owns the workspace. The kernel takes
the book pointer, entry count and block size as parameters and knows nothing of
any pack layout (a v1 caller passes the global book; a v2 or v3 caller passes
the tensor's own book per call).

| Function | Computes |
|---|---|
| `nqdc_seg_sum(xseg, block)` | `A[j]`, the sum of one activation segment |
| `nqdc_entry(book, block, xseg, k)` | `T[j][k]`, one table entry (exposed alone because the lazy and active policies build single entries on demand and must produce the eager build's bits) |
| `nqdc_build(book, entries, block, x, nseg, T, A)` | the eager build: `T` is `nseg * entries` floats, `A` is `nseg` floats |
| `nqdc_apply(idx, sc, mn, T, A, entries, nseg, rows, y)` | `y[r] = sum_j (sc[r][j] * T[j][idx[r][j]] + mn[r][j] * A[j])` |

**Determinism contract.** Segment sums ascend, table entries ascend, and apply
accumulates per row in ascending segment order. These orders are pinned
behavior: reordering any of them changes the output bits and moves the
campaign's FNV anchors (`839d3f72ad1a3d60`, `bdae48bdf7a78a40`,
`35e1adcc254bf68e`, `1cd08a5dff229720`).

## The compute policies

| Policy | Tables built | Notes |
|---|---|---|
| **eager** | every (segment, prototype) pair | the reference DTC form |
| **active** | only the prototypes the index actually uses | the active set is fixed by the index, so it can be computed once. On real tensors only 81 to 96 of the 256 prototypes are active per segment |
| **lazy** | on demand, per call | a mark pass flags the (segment, index) pairs in use and the build skips the rest; wins only when the work set is not known ahead of time |
| **direct** | none: each output gathers its prototypes | table-free; re-reads prototypes per output, no table reuse |
| **expanded** | the materializing path | the reference oracle and fallback; the GPU twin is a fixed 256-wide tree GEMV |

Eager, active and lazy are bit-identical to each other by construction on the
CPU (the probe calls the shared kernel for all three). On the GPU all four DTC
policies are bit-identical to the CPU kernel on every lane; see
[ACCEL.md](ACCEL.md) for the bench and for why lazy loses there (7 to 21 percent).

## Measured results (CPU, 2026-11-24)

Receipts: `results/dc-probe-2026-11-24.txt`, `results/dc-adv-2026-11-24.txt`,
`results/dc-sweep-2026-11-24.txt`. Bench: about 1.5 shared cores, contended;
treat timings as floor numbers.

- **Correctness** against an fp64 oracle: relative L2 error 1.527e-07 on a
  synthetic tensor, 4.036e-06 on a real one (layer 0 `q_proj` of the shipped
  Sigma pack). Against the materialize-then-multiply path on the same tensor:
  4.182e-06. The two paths compute the same packed function to float noise.
- **Speed** on that tensor: 0.832 ms best paired (0.924 ms median) against
  23.82 ms best paired for materialize-then-multiply, about 7.5 to 57.6 times
  faster depending on the pairing and tensor shape, before any GPU is involved.
- **Adversarial campaign** (`nq2-dc-adv`, B = K = 256): 23 checks hold (extremes,
  subnormals, zero segments, NaN contamination, a learned rotation, two-stage
  residual dictionaries, PQ with a shared frame, determinism, 512-segment
  accumulation), 23 HOLD / 0 BREAK. NaN discipline is explicit: a NaN input
  poisons both paths identically, and the metrics report the poison. Not every
  check is a pass-on-agreement: two are deliberate traps that must look large
  (the PQ offset trap, naive form relative L2 3.083e+01; stale tables diverging,
  0.593).
- A hostile-geometry fuzz pass under ASan and UBSan refuses every malformed
  case with a specific rejection and zero sanitizer reports.

## Break-even: when DTC wins

Per segment, DTC pays `K * B` multiply-adds to build the tables (book size `K`,
block `B`) plus about `M` lookups, where `M` is the weight matrix's row count
(the output dimension; the activation batch cancels out of the inequality).
Dense pays `M * B`. So DTC wins when

    M > K * B / (B - 1)        about 257 at K = B = 256

that is, for tall matrices, and loses for short ones. Measured on CUDA (section b,
N = 4096): expanded beats DTC at M = 32 and M = 64, DTC wins from M = 128 on and
by 19 to 69 times at M of 4096 or more; the measured crossover sits below 257
because calls this small are launch-latency bound. (An earlier draft of this
document stated the inequality reversed; Gizmo's correction and the benchmark
are recorded in [accel/README.md](../accel/README.md), and the erratum note is
in [DICTIONARY-COMPUTE.md](../DICTIONARY-COMPUTE.md).)

## DTC is the default compute path (JD's ruling, 2026-11-25)

DTC is the engine's default compute path; expansion is the exception, spent on
purpose. The [Precision Gate](PRECISION-GATE.md) arbitrates: near ties at the
head escalate, drift over long chains is budgeted with periodic expanded shadow
resyncs, and evidence runs always expand. Two disclosures ride with the ruling:

- The table path reassociates the sum, so DTC output equals the expanded path to
  float noise, not bit for bit. The default flip is a deliberate re-anchor under
  the hash doctrine; the expanded path stays as the reference oracle and fallback.
- The first gate calibration reads hotter than the plan assumed: the scratch
  measuring vehicle put per-layer DTC-versus-dense divergence at the 1e-1 to 1e0
  scale (peak 8.479 at layer 5), not the 1e-6 association-noise scale, while the
  gate engine's own DTC path reproduced the dense anchor token exactly.

## The tools

| Tool | Usage |
|---|---|
| `nq2-dc` | `nq2-dc model.nq2 model.nq2book shard [more shards] [--match substr] [--checksum]`: the probe. `--match` selects the target tensor (default `q_proj`); `--checksum` runs the whole-model fingerprint path. Malformed shard spans and non-finite book or frame values are named refusals with exit 1 (regression: `tools/test-dc-hardening.sh`) |
| `nq2-dc-adv` | no arguments; prints the 23-check campaign and ends `suite done.`; exit 0 with 23 HOLD / 0 BREAK on success |
| `nq2-fused` | the ancestor experiment: a matvec that reads the packed form directly |

```
nq2-dc sigma-full-b256-v2run.nq2 sigma-full-b256-v2run.nq2book <shard-1> <shard-2> --match q_proj
nq2-dc-adv
```

## Honest limits

- DTC computes the packed function. The shipped pack's own distortion against
  the raw weights (relative L2 about 1.0 at the output of the worst early
  tensors) is a separate quantity that DTC neither fixes nor worsens. Fidelity
  is the packing research ([ROADMAP.md](ROADMAP.md)).
- Table building costs memory bandwidth too; see the break-even above.
- Summation order differs from the materializing path, so outputs are equal to
  float noise, not bit-identical. Disclosed here rather than discovered by a
  reader.
- A "where DTC fails" study is on the program; its negatives will be published
  the same way as the positives.

## Credits

The operators: Elfie (E.L.F.I.E.) and ChatGPT (Gizmo). The memory-throughput
architecture it runs on: JD, executive direction. Reviewer: Miss Archie (Archie,
Argon), whose 2026-11-24 review covered the dictionary compute tools (64-bit file
offsets and stream index bounds). The 2026-11-25 DTC code review found no P0, four
P1 and six P2 items; the four P1s are fixed (one shared kernel; the campaign's
private copy deleted so its HOLDs certify the shipped code; the lazy checks now
compare the lazy path against the oracle; malformed inputs refuse).