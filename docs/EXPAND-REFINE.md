# Expanded packs: a refinement stage for chosen layers

Little Nemo (Nemotron Nano 9B v2) runs from an NQ2 format-v3 pack
(PQ d4, K=256, block 256, per-tensor books, 2.2169 bits per weight with
books). Every projection carries about 0.32 relative L2 error, and the
27 Nov round lost most of its misses on arithmetic, unit conversion and
science facts. An expanded pack spends a small, fixed amount of extra VRAM
on the layers where that error hurts most.

## Format

An expanded pack is an ordinary format-v3 pack with extra tensors named
`<projection>.refine`. Each is the residual `W_original - W_decoded` of its
projection, packed with the base recipe (PQ d4, per-tensor book, rms
normalization) and the same geometry. The manifest lists the base tensors
unchanged (same segment and book digests), then the refinement tensors,
then one note line `expanded from <base generation> with N refinement
tensors from <refine generation>` before `end`. The generation id is
recomputed over all tensors by the usual FNV-1a 64 rule, so the identity
checks apply unchanged.

The CUDA lane (`accel/nemo_gpu.c`) computes `y = W x`, then `y += R x`, in
that fixed order, in the single-token GEMV, the batched prefill GEMM and
the EXPAND (full precision) path. A pack without refinement tensors loads
and runs exactly as before; a same-engine control on the base pack
reproduces the morning's answers text for text (0 of 50 differ).

## VRAM sizing

At open, the lane measures free memory after the base pack and uploads
whole refinement layers in a fixed priority (11, 8, 10, 21, 14, ...) while
they still leave room for the int8 lm_head, a 64 MiB margin and the EXPAND
books. `NEMO_GPU_REFINE=auto` (default) | `off` | `all` | a layer list.
It logs a plan line, for example on the 4 GB RTX 3050 Ti:
`728 MiB free after the base pack; head int8 560 + 64 margin; EXPAND
reserve 16; refine budget 88 MiB`.

The first plan (six tensors in layers 8-11 plus 14 and 21, 176 MiB) did not
fit: the engine refused the CUDA lane ("lm_head needs 560 MiB, 574 MiB
free") and fell back to the CPU. A 110 MiB three-layer pack failed the same
check by 2 MiB. The real budget on this card is under about 108 MiB, so the
published pack refines two layers.

## Tools (pure C, built by `accel/build-nemo-expand.ps1`)

- `littlenemo/expand/ln_sens.c`: CPU sensitivity probe over a probe set;
  layer by layer, AVX2 FMA, fixed summation order, identical hashes on 1, 8
  and 12 threads. Configurations mix decoded pack (B), original weights (A)
  or B plus a packed refinement (R).
- `littlenemo/expand/ln_resid.c`: writes the `<name>.refine` residuals.
- `tools/nq2-merge-refine.c`: merges a base and a refinement pack under a
  new name, verifying both generations and every digest; `--only` keeps a
  subset byte for byte (re-quantizing a subset changes the attention
  tensors, so the subset is taken from the measured pack).
- `littlenemo/expand/ln_headbench.c`: times an int8 lm_head on the host.
- `littlenemo/expand/probe_expand.tsv`: 32 tune items (the 14 round misses
  plus 18 new) and 24 held-out items.

## Selection (CPU probe, tune loss; lower is better, base 111.66, 9/32 first)

Groups of four layers: 8-11 best (64.48, 20/32), then 12-15 (81.69), 20-23
(86.68), 16-19 (90.20), 40-43 (94.90). 4-7 and 24-27 were worse or flat.
Real packed refinements: layers 8-11 77.7-78.4 (15/32), + layer 21 74.62,
+ layers 14 and 21 74.27 (16/32). Fitting subsets: R8+R11 77.20 (16/32),
R11+R21 88.14, R10+R11 88.05, R8+R21 87.41, single R11 93.16, R8 96.32.
Held-out (24 items): base 64.38 (9/24), R8+R11 50.45 (13/24); the larger
X1+R14+R21 set (does not fit) 44.86 (13/24).

## Published pack: x2 (layers 8 and 11)

Refinement tensors: layer 8 up and down, layer 11 in and out; 72.9 MiB
stream + 1.0 MiB books. Generation `d0bd78fc71d598a4`, 342 tensors, built
from refinement generation `fc216cfe2775bfe8` over base `0daa2d6bb6d9265a`.
Bits per weight with books over the 7.714 G base weights: 2.2973 (stream
alone 2.2043), against 2.2169 for the base.

SHA-256: `.nq2` 68c9e2fbdab847be00c2191589f2a04ced4bc02cd28bd9cd1ae5eb76dc20d869,
`.nq2book` 8d5a3f2382520262aaf70cac6da03ef7000abbf1ed91712ef4cea1e7d3949cfb,
`.nq2manifest` 88a12a1e3b681428779322061443461c82fc505e8a35953c24384d4891e2ae61.

## Round: 50 questions + 10 essays, CUDA, same engine and settings

Same engine build, `--chat`, max-new 16 (questions) and 320 (essays), one
engine process per phase. Both runs on 28 Nov 2026 with fid-quant using the
CPU and other workers taking turns on the card, so speeds are real-world.

| | control (base pack) | x2 |
|---|---|---|
| questions | 36/50 | 34/50 |
| math | 7/10 | 5/10 |
| capital | 9/10 | 8/10 |
| chem-symbol | 8/8 | 7/8 |
| date | 6/7 | 6/7 |
| units | 2/7 | 3/7 |
| science | 3/7 | 4/7 |
| general | 1/1 | 1/1 |
| essays | 4/10 | 8/10 |
| essay facts | 36/62 | 43/62 |
| forbidden facts | 0 | 0 |

The control reproduces the 27 Nov morning run exactly (36/50, 4/10, 36/62,
all 50 answers identical), so the differences are the refinement.
Questions: three misses fixed (grams in a kilogram, millilitres in a litre,
legs on an insect) and five passes lost (1000 / 8 gave 625, 3 cubed gave 8,
Kenya gave "Nairo", helium gave "H", feet in a yard gave "1 foot"). That is
a net two lower, so the question gain is not shown. Essays improved on five
topics (water cycle 3 to 5 of 6 facts, human heart 3 to 6 of 7, Declaration
of Independence 4 to 5 of 6, Newton's laws 1 to 5 of 5, DNA 1 to 2 of 8),
were unchanged on four, and fell on one (photosynthesis: the x2 run wrote a
title and then a stream of blank lines, 0 of 7 facts). Four essays moved
from fail to pass.
Held-out CPU probe and the essays point the same way; the 50 short answers
do not. Treat x2 as a mixed result: more correct facts in long answers, no
net gain on one-line answers.

### Speed log (every run; 10 tok/s floor on the 4 GB card)

| run | path | decode tok/s (pooled) | prefill tok/s (pooled) | card peak MiB | RAM peak MB |
|---|---|---|---|---|---|
| 27 Nov morning base, questions | CUDA | 11.63 | 73.9 | 3841 | 306.5 |
| 27 Nov morning base, essays | CUDA | 15.64 | 58.7 | 3841 | 306.6 |
| control base, questions | CUDA | 13.35 | 43.8 | 3596 | 306.7 |
| control base, essays | CUDA | 13.99 | 41.9 | 3596 | 307.0 |
| x2, questions | CUDA | 13.51 (min 8.86) | 43.2 | 3681 | 345.1 |
| x2, essays | CUDA | 13.95 (min 13.45) | 40.5 | 3681 | 306.7 |
| 2-question load test, base | CUDA | 8.5 | 44.1 | 3703 | 344 |
| 2-question load test, lt-11-21 (52 MiB) | CUDA | 11.6 | 56.9 | 3767 | 306 |
| 2-question load test, lt-10-11 (73 MiB) | CUDA | 8.7 | 50.4 | 3789 | 307 |
| 2-question load test, lt-8-11 (= x2) | CUDA | 12.1 | 58.4 | 3789 | 307 |
| 2-question load test, x1 with auto sizing | CUDA | 11.9 | 53.2 | 3789 | 307 |
| 110 MiB three-layer pack | CPU fallback (refused) | 0.013 | 0.068 | 605 | 1294 |
| host lm_head bench, 4 lanes | CPU | 92.45 ms per head | | | |

x2 decodes at essentially the base rate (13.5 vs 13.35 tok/s pooled, essays
13.95 vs 13.99), well above the floor. The 2-question rows vary by 30 to 40
percent between runs because other workers share the card and fid-quant
shares the CPU; do not rank them finely. The 8 GB target (30+ tok/s) cannot
be measured on this card.

## Threading and tensor handshake profile (x2, 47 decode steps, real-world)

`test_nemo_hostsplit` with `NEMO_GPU_PROFILE=1` (profiling syncs after every
launch, so absolute rates are lower than the round).

- A step is 105.9 ms: layers plus sync 97.3 ms, head plus 512 KiB logits copy
  5.2 ms, confidence 2.6 ms, argmax 0.75 ms. Host work is 3.2 percent of the
  step. (Base pack in the same tool: 138.7 ms, 6.6 percent host; the base run
  met heavier contention, so compare shares, not totals.)
- The refinement costs four extra GEMV launches per token (192 over 48
  steps, 0.4 to 1.0 ms each) plus 4 adds, roughly 2 to 3 percent of the
  layer time. Decode GEMV (`k_pq_gemv_t1k`) is 3.6 s of the run, the int8
  head 5.6 ms per token.
- The GPU-to-CPU handshake is one synchronous copy of the logits per token.
  The host stages are serial after it (copy, confidence, argmax); there is no
  GPU/CPU overlap inside a token in this tool. The engine's token overlap
  option is separate.
- CPU lanes on the 131072-wide confidence pass: 1 lane 2.4 ms, 2 lanes 4.9
  ms, 4 lanes 17.6 ms, 8 lanes 6.7 ms, all bit-identical to one lane. More
  lanes were slower here because the CPU is oversubscribed (fid-quant) and
  a lane wake-up costs more than a 2 ms job. One lane is right for this
  stage, which is what the engine uses.
- Host lm_head (int8, pool): 1 lane 475 ms, 2 lanes 176 ms, 4 lanes 92 ms,
  8 lanes 164 ms, identical hashes (d32abd7186356836). At 92 ms it would put
  decode near 6 tok/s with a 13 tok/s GPU body, under the floor, so it stays
  opt-in (`NEMO_GPU_HEAD=host`).
- Zero mutexes anywhere in this path. Not checked: a GPU-versus-CPU hash
  comparison for a refined layer (the existing layer test needs an unpacked
  checkpoint of the expanded pack).

## What is not done

- A pack refining all of layers 8-11 plus attention (the 176 MiB set that
  scored best) needs more VRAM than this card has after the head and EXPAND.
  It would fit with the head on the host only at a speed under the floor.
- The expanded round does not improve the one-line questions; selection used
  teacher-forced loss on 32 items. More probe items, or choosing refinement
  by round score, is the next step.
