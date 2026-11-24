# Research notes: Greene, "Compression and Intelligence" (greene.sh, July 2026)

Read 2026-11-23 at JD's direction. JD's framing: model lithography research.
Credit: direction and the research call are JD's; notes and applicability
analysis by E.L.F.I.E.

## What the essay establishes

- Pretraining cross-entropy is literally a compression objective: surprisal
  in bits, total codelength = sum of surprisals, arithmetic coding gets
  within a couple of bits of the bound.
- Honest compression accounting is a TWO-PART code: data bitstream plus
  the program that decodes it. For LLM compressors the program includes
  the weights, and the weights dwarf small datasets. Offline compression
  only wins once the dataset is large enough to amortize the program
  (about 2.88 GB before Qwen3-0.6B beats zstd).
- Online alternative: prequential coding. Encoder and decoder start from
  the same seed and train deterministically in lockstep, so the weights
  are never transmitted. Cost = area under the loss curve (cumulative
  regret R(X) to the final weights) plus a constant program cost c.
  Prequential wins when R(X) < b|theta|: when structure is cheaper to
  learn than to ship.
- Decomposition: L_offline = L_final(X) + b|theta| + c;
  L_online = L_final(X) + R(X) + c.
- Generalization, operationally: the only bits that turn a profit are
  bits reused across many predictions. One program bit must pay for
  itself multiple times in data savings.
- Huang et al. 2024: compressive capacity on held-out data correlates
  nearly linearly with benchmark intelligence (weights cost excluded,
  amortization assumed). Outliers expose benchmark gaming.
- Loss curves fit power laws, loss(t) = H + B t^-beta. Prequential
  codelength inherits the geometry: linear term from H, sublinear
  regret term from B and beta, constant c. Online intelligence =
  data efficiency, measured by prequential codelength on the same
  data in the same order.

## Applicability to NanoQuatTwo

Our setting differs: the essay is lossless data compression via
prediction; NanoQuatTwo is lossy weight quantization (rate-distortion).
The guarantees do not transfer. The accounting and several principles do.

1. **Complete description length is our headline metric.** The .nq2
   stream is the data bitstream; the .nq2book codebooks are program
   cost; the engine is the constant c. Never quote stream-only
   bits/weight as the compression. Report (stream + book + engine)
   per weight. This sharpens the baseline question: at block 512 a
   20B model's stream is roughly 200 MB, but per-tensor books at
   that block size run about 512 KB per book and can dominate the
   total. The book is where the density fight is.
2. **Global vs per-tensor books is a two-part tradeoff.** Bigger
   program (per-tensor books) buys lower distortion; smaller program
   (one global book) costs distortion. MDL gives the decision rule:
   pick per total length at target fidelity, not by stream rate.
3. **Entropy-code the index stream.** Indices are stored flat at
   8 bits per block. If the measured index histogram has entropy
   below 8 bits, an arithmetic/range coder (pure C, fits the house
   rules) cuts the stream by the gap. Context (previous block's
   index) may cut further. Action: measure index entropy on the
   corrected packs; candidate format v3 feature alongside embedded
   tensor names/shapes.
4. **Codebook utilization audit.** Per the reuse principle, a
   prototype only earns its book bytes if many blocks select it.
   Add index-usage counts to the quantizer report; dead or rare
   entries argue for fewer entries or retraining.
5. **Fit the staircase as a curve.** Distortion vs block size
   should be fit (power-law family) and interpolated, so the
   baseline recommendation rests on the curve, not on eyeballing
   eight points. Combine with JD's latency objective: the trip
   disc to memory to CPU to token on screen.
6. **Fidelity ground truth is predictive, not weight-space.**
   Relative L2 is a surrogate. Per Huang et al., the honest measure
   of a packed model is held-out cross-entropy (bits per byte on
   unseen text). Once the engine has a tokenizer, grade packs by
   held-out loss through our own engine.
7. **Seed-generated books (research direction).** Prequential
   coding's trick, adapted: if a codebook can be regenerated from
   a seed plus a short deterministic program at load time, book
   cost collapses toward constant c. This is JD's procedural
   expansion intuition with theory under it. Not current capability;
   k-means books are data, not seeds. Candidate: fit books with a
   parametric generator, or structured books derived from a rule.
8. **Scale honesty.** The essay's amortization result says the
   technique's edge grows with model size and can vanish on small
   models where book overhead dominates. Claims should be sized
   to the model: strongest at the 20B density target, weakest on
   micro models like the 126 MB Nemo class.
9. **For the Mjolnir/Nemo line (separate from the quantizer):**
   prequential codelength is a principled way to compare learners
   (teacher seats, Nemo variants) on identical data in identical
   order, and regret R(X) is the natural score for a continual
   learner. Flagged for that workstream, not applied here.

## Implementation status, 2026-11-23 late

Applied and verified:
- Item 1 (complete accounting): `tools/nq2-audit.c` built and in
  use. The quantizer's own reports now print stream rate AND
  complete rate (source patched; binary rebuilds at packaging,
  the running staircase chain keeps its binary). The quantizer's
  v1 book-size constant was also wrong by 4 (8 vs the true
  12-byte header); fixed in source, same slip the runner had.
- Item 4 (utilization): in the audit tool. Corrected global pack:
  256 of 256 entries used, zero dead.
- First real measurements (corrected full pack, global book,
  block 256): complete description 0.15691 bits/weight (stream
  0.15625 + book 0.00055 + decoder 0.00011). At full-model scale
  the program cost amortizes to nearly nothing, exactly the
  essay's prediction. Index entropy on real data: H0 6.4189,
  H1 5.7925 bits per index; entropy coding would bring the
  complete rate to about 0.1507 (H0) or 0.1483 (H1). Modest,
  because the fp16 frame is 0.125 of the 0.15691 total: the
  frame, not the index, is the biggest remaining stream cost.
- Runner verification of the corrected pack: 3,834,051,584
  weights expanded through a 4,096-block window, peak resident
  4.48 MB, checksum 7bd9557531df1ab7, expansion 103.5 M
  weights/sec on the throttled bench.
- Pack fidelity, honestly: relative L2 0.6922 for the single
  global book at block 256. A pipeline proof, not a quality
  artifact. The per-tensor staircase is the quality question.

Left on the table (in expected-value order):
1. **Frame coding.** The min/scale frame is 80 percent of the
   stream at block 256. Delta or entropy coding of frame values,
   or frames shared across block groups, is the largest single
   lever the accounting exposed. Format v3 material.
2. **Entropy-coded index stream** (item 3). Measured headroom
   above: about 4 to 5 percent of complete size at block 256,
   more at small blocks where indexes dominate less... frame
   share grows as blocks shrink, so recheck per block size when
   the staircase lands. Pure C range coder; format v3.
3. **Format v3 bundle.** Entropy coding + frame coding +
   embedded tensor names/shapes (closes the engine's
   order-matching assumption) + tiny tensors stored raw inside
   the format instead of the engine's raw-shard fallback
   (interim fallback implemented in nq2-infer.c 2026-11-23:
   elements past the last packed block are served raw).
4. **Staircase curve fit and the baseline call** (item 5),
   gated on the staircase chain finishing.
5. **Predictive fidelity grading** (item 6): held-out loss
   through the engine, gated on the tokenizer piece.
6. **Residual books and taught entries** (pre-paper road items):
   the fidelity levers, now that accounting can price them.
7. **Seed-generated books** (item 7): research direction only.
8. **Prequential learner comparison** (item 9): Mjolnir
   workstream, not the format.

## Not applicable / do not overclaim

- No lossless guarantee transfers to .nq2; we are rate-distortion.
- Prequential lockstep training does not shrink an existing
  pretrained model's storage after the fact; item 7 is an
  adaptation, not the essay's method.
- The linear compression-intelligence relation is evaluation
  philosophy for us, not a claim about NanoQuatTwo itself.
