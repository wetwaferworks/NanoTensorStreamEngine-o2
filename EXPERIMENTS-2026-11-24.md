# Experiments, night of November 23 to 24, 2026

Everything below was measured on the real model (NVIDIA Nemotron-Labs
Sigma-3B, 3,834,051,618 weights, bf16 safetensors) on a small shared
bench VM: about 1.5 shared cores of an AMD EPYC 9D25, AVX2 and
AVX-512F present. Absolute speeds swing by roughly a quarter between
windows on this bench, so orderings inside one window are the
reliable signal. A workstation rerun is the real benchmark. The
shipped baseline (block 256, K 256, global book, full pack at
relative L2 error 0.6922 and 0.15691 bits per weight complete) never
moved during any of this. Every experiment was judged against the
error probe's per-tensor table, not against the aggregate number,
because the aggregate hides which tensors are actually failing.

How to read the error figures: relative L2 error, lower is better.
1.0 means the reconstruction carries no signal. 0.1 would mean
fidelity 0.9. Nothing below reaches 0.1 yet. This file says what
each lever actually bought, including the ones that bought nothing.

Implementation and core mathematics: Elfie (E.L.F.I.E.).
Consultee: Gizmo.
Executive direction: JD (David Rittenhouse).
Bench numbers and verdicts: Elfie.

## The one-paragraph summary

The bottleneck is prototype coverage, not rate, not frame precision,
not initialization luck, and not the size of the book. Product
quantization is the first lever that moves error by a third at a
comparable rate, RMS framing stacks on everything for free, and
the E matrix should simply ship as fp16. A four-loop streaming
engine streams the whole model from disc at up to 168.8 million
weights per second on this little bench, bit-identical in every
configuration. Details and receipts below.

## The staircase: block size is a rate decision, not a fidelity decision

Per-tensor books, layers 0 to 3 (465,592,320 weights), minmax frame:

| block | rel L2 | complete bpw |
|---|---|---|
| 32 | 0.6764 | 1.2703 |
| 64 | 0.7236 | 0.6655 |
| 128 | 0.7423 | 0.3936 |
| 256 | 0.7478 | 0.3184 |
| 512 | 0.7473 | 0.4024 |
| 1024 | 0.7437 | 0.6877 |
| 2048 | 0.7508 | 1.3168 |
| 4096 | 0.9626 | 2.6042 |

Fidelity is flat from block 128 onward while book cost grows
linearly, so the complete-rate curve is U-shaped with its sweet
spot at 256. For contrast, the shipped global pack reaches 0.6922
at 0.15691 bpw complete: per-tensor block 32 spends eight times the
bits to buy 0.016 of error. Rate was never the constraint.

## RMS framing: free improvement, stacks on everything

Replacing the min-max frame with an RMS frame (min stored as 0,
scale stores the block RMS) at identical rate:

| case | minmax | RMS |
|---|---|---|
| slice, block 32 | 0.6764 | 0.6138 |
| slice, block 128 | 0.7423 | 0.6748 |
| E alone, block 256 | 0.9922 | 0.9597 |

About 0.06 of error everywhere, zero rate cost. Adopted as the
default direction for the next format.

## Residual book: a quality-mode option, not the flagship path

A second book trained on the first stage's residuals, measured at
block 32 on the slice: minmax 0.6253 to 0.5599 two-stage; RMS
0.6139 to 0.5490 two-stage, the best pre-PQ slice fidelity we
measured. The price is double rate (2.5405 bpw complete). The
residual stage itself contributes about 0.065 of error reduction
wherever it is stacked. Worth keeping in the toolbox.

## Entries sweep: the book is already big enough

Widening the flat book past 256 entries, minmax frame:

| case | K 256 | K 512 | K 1024 |
|---|---|---|---|
| q_proj, block 256 | 0.9427 | 0.9325 | 0.9210 |
| E, block 256 | 0.9689 | 0.9499 | 0.9174 |

Four times the entries buy about 0.02 of error at two to three
times the rate. The weakest lever tested. 256 entries stands as the
right book size, and the coverage problem is structural, not a
shortage of prototypes. (Receipts: results/entries-sweep-2026-11-24.txt)

## Hadamard transformation (round 3, priority 1): not the lever

Randomized Hadamard inside each block, block 256, K 256: q_proj
0.9427 to 0.9322, E 0.9689 to 0.9725. Both deltas sit inside the
plus/minus 0.05 k-means wobble band. The transform round trip is
numerically clean (max abs error about 1e-7), and we verified the
inverse transform can be folded into the book at encode time so
decode would cost nothing extra if a transform ever won. This one
does not win. There is no outlier structure for it to spread, which
matches the tensor statistics: blocks are not outlier-dominated
anywhere in this model. (Receipts: results/hadamard-p1-2026-11-24.txt)

## Product quantization (round 3, priority 2): the first real lever

Each 256-block is split into subvectors of dimension d, one book per
subspace, frame unchanged. Book bytes are identical to the flat
book at the same K; the cost is index bytes.

q_proj, minmax frame (flat baseline 0.9427 at 0.3229 bpw complete):

| config | rel L2 | complete bpw |
|---|---|---|
| PQ d16 K256 | 0.7667 | 0.7917 |
| PQ d8 K256 | 0.5895 | 1.2917 |
| PQ d8 K64 | 0.6897 | 1.1667 packed 0.9167 |
| PQ d16 K64 | 0.8303 | 0.6667 packed 0.5417 |

With RMS framing stacked (results/pq-rms-2026-11-24.txt):

| config | rel L2 | complete bpw |
|---|---|---|
| PQ d16 K256 RMS | 0.7371 | 0.7917 |
| PQ d8 K256 RMS | 0.5613 | 1.2917 |
| PQ d4 K256 RMS | 0.3148 | 2.2917 |
| PQ d4 K64 RMS | 0.4322 | 2.1667 packed 1.6667 |

PQ d8 beats the staircase's block-32 fidelity at about the same
complete rate. PQ d4 is the first configuration to take a
projection tensor under 0.4. The honest price: the finer
subvectors march up the rate curve as they march down the error
curve (0.94 at 0.32 bpw, 0.59 at 1.29, 0.31 at 2.29). Error 0.1 is
still not in hand, but for the first time the road toward it is
visible instead of theoretical. Next probes: a residual stage on
top of PQ, and learned rotation (OPQ) in place of the fixed
Hadamard. (Receipts: results/pq-p2-2026-11-24.txt)

## Joint additive codebooks (round 3, priority 3): negligible

Two books K 256 at block 256, one frame, q_proj, complete 0.5208
bpw. Sequential residual training: 0.9146, 0.9180, 0.9198 across
seeds 42, 7, 1234. Joint optimization (alternating assignment
sweeps and book updates, 3 rounds) improves every seed by the same
0.0024 or so. Real, consistent, and not worth its complexity in
this formulation; PQ dominates the pair at every rate point.

The experiment's second answer matters more: initialization
sensitivity at tensor level is small. Seed spread is 0.005, an
order of magnitude below the wobble seen across slice experiments,
which lives in sampling coupling across tensors. Initialization is
not what limits fidelity. The ceiling is the representation.
(Receipts: results/additive-p3-2026-11-24.txt)

## Embedding specialization (round 3, priority 4): carve E out

The replaid embedding matrix (2,097,152 weights) holds 39.8 percent
of the shipped pack's total squared error at relative L2 0.9922.
Tensor statistics explain why: it is a sphere field, RMS exactly
0.25, every row of 16 has norm exactly 1.0, and its per-weight
energy is about a thousand times the projection weights'. No VQ
variant fixes it: per-tensor VQ moves the global error by less
than 0.003, and the best VQ result (PQ d8 K256, 0.5070) still
leaves it half unreconstructed.

| representation | E rel L2 | E bytes | global error if swapped |
|---|---|---|---|
| global VQ b256 (shipped) | 0.9922 | 40,960 | 0.6922 |
| PQ d8 K256 | 0.5070 | 557,056 | 0.5816 |
| PQ d4 K256 RMS | 0.2771 | (3.1250 bpw) | n/a |
| fp16 raw | 0.000208 | 4,194,304 | 0.5371 |
| fp32 raw | 0.000000 | 8,388,608 | 0.5371 |

Recommendation for the next format: E ships as fp16 raw. Four
megabytes against a 75 megabyte artifact removes 39.8 percent of
the global squared error outright, and fp16 is indistinguishable
from fp32 here at half the bytes. (Receipts:
results/embedding-p4-2026-11-24.txt)

## Normalization research: rsqrt, fusion, and the frame

Lookup-table reciprocal square root is NOT promoted. One Newton
step brings every method (hardware rsqrt, 256-entry and 1024-entry
tables) to fp32 grade, and in the real workload shape (one rsqrt
serves a whole 256-wide block) every method lands at 360 to 380
cycles per block, inside noise. Hardware rsqrt wins the isolated
throughput race by one to two orders of magnitude. The directive's
own acceptance rule settles it: a table that does not beat hardware
end to end does not ship. (Receipts: results/rsqrtbench-2026-11-24.txt)

Fused reconstruction is bit-identical. Reconstructing each element
inside the dot product instead of materializing the expanded tensor
produces max abs diff 0.000e+00 against the materialize-then-multiply
path, because accumulation order is unchanged. Fusion is numerically
free, holds 0.51 MB packed instead of 50.3 MB expanded, and wins
end to end once the separate expansion pass is priced in.
(Receipts: results/fused-testc-2026-11-24.txt)

The fp16 frame costs nothing measurable. Storing the RMS scale in
fp32 instead of fp16 changes relative L2 by 0.000000 at six
decimals; the two reconstructions differ from each other by 0.000118,
four orders of magnitude below the coverage error. Keep the fp16
frame. The frame question is closed. (Receipts:
results/framed-testd-2026-11-24.txt)

Earlier frame work, for the record: plain entropy coding of the
frame codes prices the frame at 0.0629 bits per weight, halving the
frame term with random access preserved. Delta coding of minima
gains essentially nothing; log-scale deltas work. See
WHAT-WE-CAN-DO.md.

## The streaming engine: four loops, one serialized order

JD's architecture: the main loop does all computation; a second
loop does nothing but disc lookups, grabbing prefetch and predictive
tensors; two loops do the memory unpacks and hand finished tensors
forward. Stages run concurrently, consumption happens in one
serialized order, so the result is deterministic no matter how the
stages interleave. Full model, 26 layers, 182 layer tensors in
forward order, 3,026,190,336 weights, 12.10 GB expanded per pass:

| config | wall | rate | main stall |
|---|---|---|---|
| sequential reference | 33.49 s | 90.4 M weights/s | n/a |
| 1 unpack loop, lookahead 4 | 22.19 s | 136.4 M weights/s | 8.9% |
| 2 unpack loops, lookahead 8 | 20.56 s | 147.2 M weights/s | 6.9% |

The accumulator is 1394.1117661067326 in every configuration, bit
for bit. The disc only moves about 15 MB of packed stream to
produce those 12.1 GB of expanded weights; expansion happens in
flight. Book lookup A/B at the winning configuration: a row-pointer
table with two-block prefetch reaches 168.8 M weights/s (675 MB/s),
beating the plain loop and the SIMD build of it. The winning
prefetch is ported into the engine's packed reader. (Tool:
tools/nq2-streamtest.c)

## Engine test state

The inference engine (tools/nq2-infer.c, pure C) runs the full
Sigma recipe end to end. One-step anchors reproduce exactly across
builds, raw and packed backends alike. An 8-step run at the
reference recipe (guidance 3.0, score temperature 0.3, seed 42)
produced varied, non-degenerate token IDs with prefetch telemetry
of 94.2 percent hits. A clean 8-step determinism repeat with full
output capture is owed as the formal record, and a 32-step recipe
run is in flight as this file is written. Behavioral quality
against NVIDIA's reference implementation is not yet validated:
the model download ships no tokenizer, so the engine is token IDs
in, token IDs out until that piece lands. That gate is stated
plainly rather than papered over. (Round 3, priority 5, waits on
the same gate.)

## What we recommend next

1. Format v3: interleaved per-block records (index, min, scale in
   one sequential run per tensor), embedded tensor names and
   shapes, tiny tensors stored raw, RMS frame as the default, E
   matrix as fp16 raw, frame entropy coding. No experimental
   representation gets promoted into v3 until it demonstrates a
   measurable advantage; PQ has, the rest have not yet.
2. PQ plus a residual stage, and OPQ, as the next fidelity probes,
   judged per-tensor against the error probe.
3. Workstation reruns of every rate in this file. This bench is a
   baseline, not a verdict.
