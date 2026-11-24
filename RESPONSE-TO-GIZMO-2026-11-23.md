# NanoQuatTwo: Response to Gizmo's Engineering Handoff

Date: 2026-11-23 (late evening EST)
From: Elfie (implementation and testing), for JD (architectural direction and executive planning)
To: Gizmo, outside engineering review
Re: his engineering handoff of 2026-11-23, and our preliminary report of the same evening

Gizmo, your handoff asked for measurements where we had estimates and plain accounting where we had headlines. We ran the program you specified as far as tonight allows. Two new instruments were built for it: a frame probe (your section 2) and a per-tensor error probe (your section 4). The baseline suite was re-run from the shipped binaries under your accounting discipline and reproduces exactly. The headline of this response is not comfortable, and we are leading with it anyway: **the per-tensor breakdown confirms your quality concern in full. The global book at block 256 barely reconstructs the typical tensor, and the fidelity bottleneck is prototype coverage, not the frame.**

## 1. The error probe: where the 0.6922 actually lives

The probe walks the full corrected pack (global book, block 256, 243 tensors, 3,834,051,584 weights) against the original Sigma shards, tensor by tensor. It reproduces the quantizer's total exactly: relative L2 0.6922, fallback 34 weights. The distribution underneath:

| tensor | rel L2 | share of total squared error |
|---|---|---|
| replaid_embedding_matrix.matrix | 0.9922 | 39.80% |
| encoder.embed_tokens.weight | 0.9885 | 5.32% |
| diffusion_head.weight | 0.8491 | 5.31% |
| (next: late-layer gate/up projections) | ~0.994 | ~0.5% each |

- Top 10 tensors hold 54.6% of all squared error.
- **Median per-tensor relative L2 over the 241 packed tensors: 0.9943.** The typical tensor is, to within measurement, not reconstructed at all.
- Worst tensor: encoder.layers.25.mlp.down_proj at 1.0077, which is worse than storing zeros.
- Best tensors: the layernorms, at 0.02 to 0.03 (nearly constant vectors; one prototype shape suffices).
- One tensor, the 2.1M-weight replaid embedding matrix E, carries 39.8% of the entire error budget at rel L2 0.9922. It is small (8 MB raw) and high-energy, and min-max normalization is brutal to it: a single outlier in a 256-wide block stretches the frame and crushes the remaining structure.

Plain statement, per your section 4: the 0.6922 headline is carried by a few high-energy tensors with moderate error. It is not evidence of broad reconstruction, and varied engine output IDs are not fidelity evidence. We agree with your gate and now have the numbers that enforce it.

## 2. The frame study: your experiments A to E, measured

On the same pack: 14,976,764 blocks; the frame (fp16 min + fp16 scale per block) is 59,907,056 bytes, 80.0% of the stream, 0.12500 bits/weight at block 256.

- **Minima:** 1,796 distinct fp16 codes, entropy 7.5178 bits against 16 stored. Adjacent-block deltas: entropy 7.4991 bits, mean jump 267.8 codes, only 1.11% zero. **Your experiment A (delta minima) fails on real data.** The minimum is an extreme statistic; neighboring blocks' minima do not track.
- **Scales:** 3,256 distinct codes, entropy 8.5698 bits. Plain code deltas are worse (8.5845). Log2-domain deltas quantized at 1/64: entropy 6.1412 bits, RMS induced error 0.00451 in log2 (about 0.31% relative on the scale), no clipping at ±2048 symbols. **Your experiment B works.**
- **The finding that reorganizes the plan:** entropy-coding the frame codes *themselves*, no deltas and no dependency chains, prices the frame at (7.5178 + 8.5698) / 256 = **0.0629 bits/weight**, half its current cost, with random access fully preserved. The full delta-plus-checkpoint machinery (your experiment E, priced: every 64 blocks costs 0.05440 frame bpw, every 32 costs 0.05552, against 0.05328 unchained) buys only about 0.0096 bpw beyond the plain route. We recommend the plain route first: most of the gain, none of the chain.
- Estimated complete rates at block 256 if a real entropy coder lands near entropy: about 0.0948 bpw (plain frame coding), about 0.0853 bpw (delta plus checkpoints), against 0.15691 measured today. Estimates, stated as estimates; real coders land above entropy.
- Experiment C (shared frames across block groups) is not yet run; the entropy result deprioritizes it. Experiment D is the plain route above, measured at the entropy level.

## 3. The staircase, first three steps (per-tensor books, layers 0 to 3)

| block | complete rate (stream + books) | rel L2 | per-matrix rel L2 |
|---|---|---|---|
| 32 | 1.2700 bpw | 0.6764 | 0.88 to 0.90 |
| 64 | 0.6654 bpw | 0.7236 | 0.94 to 0.96 |
| 128 | 0.3936 bpw | 0.7423 | 0.95 to 0.99 |

Two honest comparisons. First, per-tensor books at block 32 do reconstruct each matrix far better than the global book does (0.88 to 0.90 against a 0.994 median), but 0.89 is still not faithful; it is a direction, not a destination. Second, the global full-model pack spends 0.15691 bpw complete for an overall 0.6922, while per-tensor block 32 spends 1.27 bpw for 0.6764 on the slice: eight times the bits bought 0.016 of L2. **The bottleneck is the representation's coverage of normalized block space, not the bit budget.** Remaining staircase points (256 through 4096, plus the global-book slice runs) are still computing and will follow as an addendum.

## 4. Your ten questions

1. **Host RAM or VRAM?** Host RAM. The 466 MB per expanded layer and 1.40 GB for three predictive layers are CPU-side fp32 expansions by the C engine. No GPU is in this path; the GPU lane is a separate seat's work.
2. **The 4,476,940 resident bytes:** book 262,156 + expansion window 4,194,304 (4096 blocks × 256 × 4 bytes) + stream buffers 20,480. Exact, and re-verified tonight from the shipped binary.
3. **Does the 74,883,852-byte stream include frames?** Yes. Anatomy: 32 header + 14,976,764 index bytes + 29,953,528 min bytes + 29,953,528 scale bytes.
4. **The 53,040 decoder bytes:** the Linux nq2-infer ELF, engine and decoder in one binary, a shared runtime rather than per-model data. Per your accounting we now report two totals (section 5). The Windows decoder (nq2-infer.exe) is 255,043 bytes; totals are platform-specific and will be reported per platform.
5. **The fallback:** exactly 34 weights (gamma_bounds 2 + noise_schedule 32; 136 bytes as f32), 0.0000009% of the model. The real cost is not bytes; it is the engine's dependency on the 7.67 GB raw shards for those tensors. Format v3 retires this by storing tiny tensors raw inside the bundle.
6. **Random access:** yes. Fixed-size records give O(1) block addressing; the runner's peek demonstrates single-block reads. Formats v1 and v2 have no sequential dependency. (The delta frame coding in section 2 would introduce one; the plain entropy route does not.)
7. **Decode throughput by block size:** pending the staircase's completion; the block-256 full expansion runs at 29.3 M weights/sec from the shipped binary on a contended shared VM, so treat all tonight's timings as soft.
8. **Behavioral quality:** not measured. We have reconstruction L2 (now per-tensor) and pipeline sanity only. No held-out cross-entropy, no benchmark deltas. Stated plainly, per your gate.
9. **Per-tensor breakdown:** section 1. One tensor holds 39.8% of the error; the median tensor is unreconstructed.
10. **Scheduler telemetry:** not yet instrumented. The demand/ready/consumption timestamps you specified are future work; the prefetch pipeline exists but does not yet log its slack.

## 5. Accounting, as adopted

Two totals, always labeled, per your section 1:

- **Model artifact total** (stream + books + fallback bytes): 75,146,008 bytes, 0.15680 bits/weight.
- **Self-contained package total** (artifact + decoder): 75,199,048 bytes, 0.15691 bits/weight, Linux decoder. Windows decoder total: 75,401,051 bytes.

## 6. Where the levers actually are, ranked by tonight's measurements

1. **Prototype coverage is the fidelity lever.** More entries, residual books, per-tensor books, and a kinder normalization are where reconstruction lives. Our next fidelity experiment: scale-only normalization (RMS, no min offset) against min-max, same budget.
2. **Frame entropy coding is the rate lever at fixed fidelity.** It halves the frame term with random access intact.
3. **E deserves special handling.** 8 MB raw, 39.8% of the error. Per-row frames or outright higher-precision storage is cheap insurance for the diffusion geometry.
4. **Format v3** bundles 2 and the tiny-tensor fix, embeds tensor names and shapes, and ends the raw-shard dependency.

Questions back to you: do you see a coverage argument we are missing... a reason 256 prototypes in normalized block space should do better than the measured 0.994 median? And on normalization: min-max versus RMS scale-only, do you have a prior from the VQ literature we should test first?

## 7. Reproducibility record

Repository commits: initial 84273e9a97e3a8f120f4e73b92cfef08301c8455; credited edition 4f7bd3145f60424b974c0d6444626c32e05e84eb. Tools: nq2-quantize, nq2-run, nq2-audit, nq2-infer, plus nq2-frameprobe and nq2-errprobe built for this response (an AddressSanitizer pass caught and fixed a buffer sizing bug in the error probe before its first clean run; the run above is post-fix). Anchors re-verified tonight from shipped binaries: expansion checksum 7bd9557531df1ab7; complete description 0.15691 bpw; index entropy H0 6.4189, H1 5.7925; frame shares as section 2. Model: nvidia/Nemotron-Labs-Sigma-3B, bf16, 3,834,051,618 weights. Environment: shared Linux VM under contention; Windows binaries cross-built with a POSIX-thread MinGW and verified under Wine (frame probe output identical across platforms).

Attribution, as always in this project: architectural direction and executive planning, JD. Implementation, testing, and the measurements above, Elfie. Outside engineering review, Gizmo... whose handoff produced the two instruments that produced the uncomfortable numbers, which is what review is for.
