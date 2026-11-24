# NanoQuatTwo: Response to Gizmo, Round 2

Date: 2026-11-24 (overnight session, 2026-11-23 evening to 2026-11-24 early hours, EST)
Project: NanoQuatTwo (JD's name for the discipline: model lithography)
Direction and executive planning: JD. Implementation and testing: Elfie. Outside review: Gizmo.

This answers your five round-2 asks in order, then reports the streaming work and the inference testing, then closes with the question JD is bringing to you himself. All rates use the two-total accounting from round 1 (model artifact total, and self-contained package total where a decoder is involved). Bench caveat, stated once and applying to every throughput figure: the bench VM is throttled to about 1.5 shared cores and most runs shared it with other jobs, so rates are structural evidence, not workstation benchmarks. Two runtime restarts interrupted long runs during the night; everything interrupted was relaunched from verified checkpoints, and bench binaries now live in a persistent bench/ directory instead of /tmp.

## Ask 1: preserve the error probe as a regression test

Done. nq2-errprobe gained an --all mode that emits the full ranked table (per-tensor parameter count, relative L2, squared-error contribution, representation method) and sits in the baseline manifest (BASELINE-2026-11-23.md) with SHA-256 hashes of the shipped binaries, the pack trio, the tool sources, and the baseline table itself. Baseline reproduced exactly before any round-2 change: total relative L2 0.6922 over 243 tensors, 34 raw-fallback weights.

One correction to your handoff: the tensor count is 243, not 245 (241 packed tensors plus 2 raw-fallback tensors).

What the probe says, plainly: the 0.6922 headline is carried by a few high-energy tensors. The replaid embedding matrix alone holds 39.80 percent of all squared error (relative L2 0.9922), then encoder.embed_tokens.weight (5.32 percent) and diffusion_head.weight (5.31 percent). The top 10 tensors hold 54.6 percent of squared error. The median per-tensor relative L2 is 0.9943: the typical tensor is essentially unreconstructed, and your quality-gate concern is confirmed by measurement. Best tensors are the layernorms (0.023 to 0.026). Worst is encoder.layers.25.mlp.down_proj at 1.0077, marginally worse than storing zeros.

## Ask 2: the ranked table

Produced by errprobe --all and frozen in the baseline (241 rows against the global block-256 book, plus the 2 fallback tensors). Method per tensor is recorded in the table; see Ask 3 for the two tensors you named.

## Ask 3: the embedding matrix and the worst projection, independently

A new tool, nq2-tensorstat, measures block statistics (range/RMS percentiles, top-1 energy share after min-shift, outlier-dominated block fraction, excess kurtosis, optional row norms).

The embedding matrix is a sphere field. Its RMS is exactly 0.25, every row of 16 has norm exactly 1.0 (minimum equals maximum), excess kurtosis is minus 0.35, block range/RMS mean is 5.17, top-1 energy share at the median block is 0.014, and zero percent of its blocks are outlier-dominated. Its per-weight energy is roughly 1000 times that of the projection weights. Verdict: the failure is energy concentration plus a geometry a 256-prototype book cannot tile (each block is 16 unit vectors concatenated), not min-max outlier crushing. Recommendation stands: carve it out of VQ and store it near-raw (8 MB in fp32, 4 MB in fp16). On the global pack's accounting, removing its error entirely would move the headline from 0.6922 to about 0.537.

The worst projection (layers.25 down_proj, 1.0077) is the opposite profile: RMS 0.00627, excess kurtosis 6.94, range/RMS mean 6.02 with a p99 of 12.85 and a max of 79.22. Heavy tails, evenly spread energy. Its failure is the general coverage failure, shared with the other projections (q_proj.0 kurtosis 7.35; diffusion_head 4.36). Across every tensor measured, the median top-1 share is about 0.014: blocks are not outlier-dominated anywhere. Prototype coverage in normalized block space is the bottleneck, everywhere, which is what Ask 4 tested.

## Ask 4: coverage improvements before index-bit shaving

Three coverage levers measured, all per-tensor books on the layers 0-3 slice (465,592,320 weights) unless noted.

RMS normalization (frame stores block RMS as the scale and zero offset; v1/v2 readers handle these packs unmodified). Relative L2, minmax against RMS at identical rate:

| case | minmax | RMS |
|---|---|---|
| slice, block 32 | 0.6764 | 0.6138 |
| slice, block 128 | 0.7423 | 0.6748 |
| E matrix alone, block 32 | 0.8656 | 0.8485 |
| E matrix alone, block 256 | 0.9922 (global book) | 0.9597 |

RMS buys about 0.06 to 0.07 of L2 everywhere at zero rate cost. In a format v3 the RMS frame can also drop the stored min (always zero), halving frame bytes.

Residual book (a second book trained on the first stage's residuals, measured only; the written stream stays single-stage). Two-stage relative L2 at block 32:

| stage 1 | stage 1 L2 | two-stage L2 | two-stage complete rate |
|---|---|---|---|
| slice minmax | 0.6253 | 0.5599 | 2.5405 bpw |
| slice RMS | 0.6139 | 0.5490 | 2.5405 bpw |
| E alone, minmax | 0.8656 | 0.7632 | 2.7502 bpw (E only) |

Honest decomposition: the residual stage contributes about 0.065 of L2 on the slice in both runs. The minmax stage-1 figure here (0.6253) differs from the staircase's 0.6764 because residual sampling draws from the same fixed-seed RNG stream and shifts later tensors' k-means draws; that wobble, about plus or minus 0.05 at this scale, is itself a finding, and it is why your probe doctrine (judge tensors, not aggregates) is the right one. Verdict: the residual stage is the best slice fidelity measured (0.5490) at double rate. A quality mode, not the sub-bit flagship path.

Not yet run: the entries sweep beyond 256 (needs a wide-index experimental stream) and any rotation or product-quantization variant. See the closing question.

## Ask 5: staircase and Test A against the shipped baseline

Staircase, per-tensor books, layers 0-3 slice, relative L2 (stream rate is 5 bytes per block; book bytes grow linearly with block size):

| block | rel L2 | stream bpw | book bytes (slice) |
|---|---|---|---|
| 32 | 0.6764 | 1.2500 | 1,179,668 |
| 64 | 0.7236 | 0.6250 | 2,359,316 |
| 128 | 0.7423 | 0.3125 | 4,718,612 |
| 256 | 0.7478 | 0.1563 | 9,437,204 |
| 512 | 0.7473 | 0.0781 | 18,874,388 |
| 1024 | 0.7437 | 0.0391 | 37,748,756 |
| 2048 | 0.7508 | 0.0195 | 75,497,492 |
| 4096 | in flight (restart-interrupted twice, relaunched) | 0.0098 | about 151 MB |

Fidelity is flat from block 128 onward while book cost doubles per step: in this regime block size is a rate decision, not a fidelity decision. Against the shipped baseline: the global block-256 pack reaches 0.6922 at 0.15691 bpw complete (package total), while per-tensor block 32 spends 1.2700 bpw complete for 0.6764 on the slice. Eight times the bits bought 0.016 of L2. The bottleneck is representation coverage, not rate, which the Ask 4 levers confirm from the other side.

Test A (raw backend, 8 steps, guidance 3.0, score temperature 0.3, seed 42): completed. Generated IDs: 10407 2258 5476 130980 119506 39659 75549 2948 112269 53365 3110 4670 1452 81773 10564 2477 19205 3980 72198 48075 31628 53677 123269 1817 99936 9041 2268 55722 3110 11735 2071 3067. NFE 18. Varied, non-degenerate output at the full recommended recipe; quality against NVIDIA's reference remains unvalidated, as before.

The determinism repeat completed with matching NFE, but a flaw in how the run was captured (piped through a truncating tail) cut the generated-IDs line from the recorded output. Owned plainly: the ID-for-ID repeat comparison therefore rests on the anchor smokes instead. The current engine build reproduced the recorded 1-step raw anchor in full (19254 60401 63533 105033 23760 105343 20733 4517 17696 2664 6704 56671 16168 21525 50272 20939 51470 6302 36933 13528 24055 35590 65165 23536 22866 55578 86117 81793 6201 61076 8458 68977) and the packed anchor in full (463 78431 129796 40418 94723 16470 65356 33407 94768 121566 27725 71821 6099 4619 7043 71501 98186 110236 51519 36477 56899 89689 52412 123872 15984 42082 45739 94642 68400 109740 13798 26618). A clean 8-step repeat with file output is queued. The 32-step recipe run is in flight at writing.

## Streaming: the four-loop architecture and the book lookup

JD designed a four-loop streaming architecture overnight: a main loop doing all CPU computation, a disc loop doing lookups only (grabbing prefetch and predictive tensors), and two memory-unpack loops handing finished tensors forward. His framing: the Timing Handshake applied to streaming... concurrent stages, one serialized consumption order, so concurrency never leaks into the result.

A new harness, nq2-streamtest, implements it over the real pack in the engine's forward demand order (182 layer tensors, 3,026,190,336 weights, 12.10 GB expanded per pass), with a sequential reference mode whose accumulator every pipelined configuration must reproduce.

| configuration | wall | expanded rate | main-loop stall |
|---|---|---|---|
| sequential reference | 33.49 s | 90.4 M weights/s (361 MB/s) | n/a |
| 1 unpack loop, lookahead 4 | 22.19 s | 136.4 M weights/s (546 MB/s) | 8.9% |
| 2 unpack loops, lookahead 4 | 24.32 s | 124.4 M weights/s (498 MB/s) | 12.7% |
| 2 unpack loops, lookahead 8 | 20.56 s | 147.2 M weights/s (589 MB/s) | 6.9% |

The accumulator is 1394.1117661067326 in all four configurations, bit for bit. Lookahead depth was the stronger lever than unpack width on this bench. Engine telemetry from the Test A repeat motivates the design: the engine's single combined prefetch thread hit 94.2 percent of requests with a mean ready slack of 759.6 ms, but 202 misses cost 36,769.6 ms of consumer stall, and the load path sustained 273 MB/sec expanded. Splitting disc from unpack, and doubling unpack, is the direct answer.

Book lookup variants, A/B tested in the harness at 2 unpack loops, lookahead 8 (accumulator identical in every run):

| variant | wall | rate |
|---|---|---|
| baseline loop | 21.70 s | 139.4 M weights/s |
| baseline, SIMD build (-march=native) | 19.31 s | 156.7 M weights/s |
| row-pointer table + prefetch (2 blocks ahead) | 17.93 s | 168.8 M weights/s |
| entry-major (bucket blocks by prototype) | 24.67 s | 122.7 M weights/s |

The prefetch table wins (plus 21 percent); SIMD adds about 12 percent in matched conditions (a same-window rerun pair: plain 125.1, native 138.8). Entry-major is rejected at this book size: the 256 KB book is cache-resident, so streaming each prototype once buys nothing against the counting sort and scattered writes. It may deserve a second look if books outgrow cache. The winning lookup is ported into the engine's packed reader; the resulting build passed both anchor smokes exactly (above), so the optimization is numerically invisible.

## JD's question, and ours

JD's words on seeing the block-256 territory: how do we improve the error from the 0.7 range... could it be pushed toward 0.1-class error (fidelity 0.9)? Our measured levers stack to roughly 0.55 on the slice at 2.54 bpw (RMS plus residual), and an E carve-out takes the global headline to about 0.537. None of that reaches 0.1. The gap looks structural to plain VQ with per-block frames at these rates, but you may see a route we do not: product quantization across sub-blocks, learned rotations before quantization, additive quantization, far more k-means iterations or restarts (our wobble suggests optimization quality is part of the gap), or something else entirely. What does the literature say is the strongest known lever at 0.15 to 2.5 bits per weight, and which of these would you test first against the regression probe?

## Open items at writing

Block 4096 staircase step and the 32-step recipe run in flight; clean 8-step determinism repeat queued behind them; entries sweep awaiting a wide-index experimental format; format v3 (interleaved per-block records, frame entropy coding, embedded tensor directory, RMS frame without a stored min) specified but not built. Package v3 is held until JD finishes testing v2 on his workstation.

## Addendum (2026-11-24, after first writing): staircase complete

The block-4096 step landed: relative L2 0.9626, stream 0.0098 bpw, books 150,994,964 bytes. Folding book bytes into every staircase step makes the complete-rate curve U-shaped: block 32 completes at 1.2703 bpw, 64 at 0.6655, 128 at 0.3936, 256 at 0.3184 (the per-tensor sweet spot), 512 at 0.4024, 1024 at 0.6877, 2048 at 1.3168, 4096 at 2.6042. Blocks past 256 are worse on complete rate and no better on fidelity, and 4096 collapses on fidelity as well (the 3072-weight layernorms contribute zero blocks at that size). The 32-step run remains in flight.

Regards from JD, and from me: your handoff produced the instruments that produced these numbers. Thank you.

... Elfie (implementation and testing), for JD (direction and executive planning)
