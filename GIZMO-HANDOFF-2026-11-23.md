# NanoQuat-O2 --- Engineering Review & Experimental Handoff

**Prepared for:** Elfie\
**Prepared by:** Gizmo, outside technical consultant\
**Project:** Wet Wafer Works\
**Status:** Working engineering brief --- update measurements as
experiments advance

------------------------------------------------------------------------

## 1. Purpose and scope

NanoQuat-O2 is being developed as a model-weight representation,
reconstruction, and streaming-inference system. The central idea is not
merely to compress a model file: it is to make a very large logical
weight space usable on constrained hardware by keeping a compact
representation on SSD/NVMe, retrieving the required blocks predictively,
reconstructing them in bounded working memory, and delivering them to
computation just before they are needed.

The intended analogy is game-engine virtual-texture or world streaming:

1.  Keep the compact addressable representation on storage.
2.  Predict which weight blocks the next execution stages will require.
3.  Read those blocks ahead of demand.
4.  Decode/reconstruct into a bounded staging window.
5.  Transfer or expose the reconstructed data to the compute backend.
6.  Reuse buffers and continue the stream without stalling execution.

The key engineering question is whether this pipeline can keep compute
supplied while using much less resident memory than the original model.

This brief distinguishes: - **Reported measurements:** values supplied
in the preliminary NanoQuat-O2 report. - **Interpretations:** what those
measurements do and do not establish. - **Proposals:** experiments and
design options that still need implementation and measurement.

## 2. Reported preliminary measurements

The preliminary report describes a Sigma model with approximately
**3.834 billion weights**, originally represented as BF16.

  Metric                                                Reported value
  ---------------------------------- ---------------------------------
  Compressed stream                                   74,883,852 bytes
  Codebook                                               262,156 bytes
  Effective description rate                       0.15691 bits/weight
  Relative L2 reconstruction error                              0.6922
  Index entropy H0                                         6.4189 bits
  Conditional entropy H1                                   5.7925 bits
  Codebook utilization                                       256 / 256
  Block size S                                             256 weights
  Codebook cardinality K                                   256 entries
  Codebook prototype precision                                    FP32
  Streaming expansion window                              4,096 blocks
  Peak resident set                    4,476,940 bytes (about 4.3 MiB)
  Decoder binary                               53,040 bytes (reported)

The frame metadata was described as two FP16 values per block: a minimum
and a scale. The preliminary report also described a four-layer
streaming experiment and a block-size staircase covering S=32 through
S=4096. Some small tensors may use a raw-shard fallback; those bytes
must be included in complete artifact accounting.

**Important qualification:** these are preliminary representation,
reconstruction, and memory figures. They do not by themselves
demonstrate that the compressed model preserves the original model's
task performance, nor do they establish end-to-end streaming inference
speed.

The reported relative L2 error of **0.6922 is substantial**. It needs to
be treated as a major quality signal, not waved away because the decoder
produces varied or non-degenerate outputs.

## 3. Correct storage accounting

The frame overhead is material and must be included. With one FP16
minimum and one FP16 scale per block, the frame consumes 32 bits per
block.

For a simple fixed-width index stream:

\[ M\_{`\mathrm{stream+frames+book}`{=tex}} = Bq + 32B + KSb \]

where: - (B) = number of blocks - (q=`\lceil`{=tex}`\log`{=tex}\_2
K`\rceil`{=tex}) = bits per codebook index - (S) = weights per block -
(K) = codebook entries - (b) = bits per codebook prototype weight -
(32B) = two FP16 frame values per block - (KSb) = codebook storage in
bits

For the reported configuration, K=256, S=256, and FP32 prototypes:

-   Index cost: 8/256 = **0.03125 bits/weight**
-   Frame cost: 32/256 = **0.125 bits/weight**
-   Amortized codebook cost: approximately **0.00055 bits/weight** for
    the reported large model
-   Combined subtotal: approximately **0.1568 bits/weight**, before any
    additional metadata or other format overhead

This explains why frame metadata dominates the index-plus-frame stream
at S=256. The frame component is roughly 80% of those two components.

The complete accounting should be explicit:

\[ M\_{`\mathrm{model}`{=tex}} = M\_{`\mathrm{indices}`{=tex}} +
M\_{`\mathrm{frames}`{=tex}} + M\_{`\mathrm{codebook}`{=tex}} +
M\_{`\mathrm{metadata}`{=tex}} + M\_{`\mathrm{fallback}`{=tex}} +
M\_{`\mathrm{padding/alignment}`{=tex}} \]

Report two totals separately:

-   **Model artifact:** all bytes required to represent and reconstruct
    this model, excluding a separately installed shared runtime if that
    is the deployment model.
-   **Self-contained package:** model artifact plus decoder/runtime
    binaries and any other required executable components.

Do not mix the decoder with the inference engine. State exactly what is
included in each reported size. A raw fallback is not free; count its
bytes and the tensors/regions it covers.

## 4. Block size and codebook size are different variables

Keep these symbols and measurements separate:

-   **S:** weights per block.
-   **K:** number of codebook entries.
-   **q:** index width, normally (`\lceil`{=tex}`\log`{=tex}\_2
    K`\rceil`{=tex}) for fixed-width indices.

Changing S changes frame overhead per weight and codebook size. Changing
K changes index width and codebook size. They are not interchangeable
tuning knobs.

For two FP16 frame values per block, the frame cost alone is:

    Block size S      Frame cost
  -------------- ---------------
              32     1.00000 bpw
              64     0.50000 bpw
             128     0.25000 bpw
             256     0.12500 bpw
             512     0.06250 bpw
            1024     0.03125 bpw
            2048    0.015625 bpw
            4096   0.0078125 bpw

Larger blocks reduce frame overhead per weight, but the codebook grows
as K × S × prototype-bytes. Larger blocks may also alter reconstruction
error, locality, decode cost, and random-access behavior. The table is
therefore an accounting aid, not a prediction that the largest block is
best.

## 5. Highest-value frame-coding experiments

The current frame representation appears to be the largest obvious
storage overhead at S=256. The following are proposals, not claimed
implemented features.

### A. Delta-code adjacent minima

Instead of storing every minimum independently, store:

\[ `\Delta `{=tex}m_i = m_i - m\_{i-1} \]

Measure the distribution, bit cost, decode cost, and reconstruction
exactness. Consider periodic absolute checkpoints so a decoder can begin
at a bounded distance from any requested block.

### B. Encode scale in a transformed domain

Scale values are positive. One candidate is:

\[ `\Delta `{=tex}s_i = `\log`{=tex}\_2(s_i) - `\log`{=tex}*2(s*{i-1})
\]

Measure whether transformed residuals are more compressible than raw
FP16 scales. Preserve enough precision to make the resulting
reconstruction behavior measurable and reproducible.

### C. Shared frames across groups of blocks

Test shared or grouped frame parameters for groups of 2, 4, 8, and 16
blocks. This can reduce metadata, but may increase reconstruction error.
Measure the size/error trade-off rather than assuming shared frames are
harmless.

### D. Entropy-code frame residuals

After measuring residual distributions, test an appropriate coder (for
example, range coding, arithmetic coding, ANS, or a simple
variable-length scheme). Compare: - encoded bytes, - encode/decode
throughput, - random-access cost, - memory use, - deterministic
output, - and error relative to the uncompressed frame representation.

### E. Preserve bounded random access

Do not let delta coding turn the entire model into one long sequential
dependency. Use periodic absolute checkpoints or independently decodable
chunks. Test several checkpoint intervals, such as every 32 or 64
blocks, and measure the extra bytes against seek/restart cost.

**Suggested first combination:** delta-coded frame values + entropy
coding of residuals + periodic absolute checkpoints. Benchmark it
against the current format before changing the default.

## 6. Staircase experiment

Run the block-size staircase with controlled variables. At minimum,
test:

-   S = 32, 64, 128, 256, 512, 1024, 2048, 4096
-   More than one K if practical, while keeping S and K explicitly
    labeled
-   The same source model, calibration/training inputs, seed, and
    evaluation configuration where applicable

For every run, record:

1.  Original tensor/weight count and original byte size.
2.  Index bytes.
3.  Frame bytes.
4.  Codebook bytes.
5.  Metadata and alignment bytes.
6.  Raw-fallback bytes and the fraction of weights using fallback.
7.  Decoder/package bytes separately.
8.  Total model-artifact bytes and self-contained bytes.
9.  Effective bits per weight under both accounting definitions.
10. Relative L2 error and any additional tensor-level error statistics.
11. Encode time, decode time, peak RAM, and peak GPU memory.
12. Random block access latency and sequential throughput.
13. End-to-end inference/task-quality results, once the evaluation path
    is ready.

Do not compare only the headline compression ratio. The useful result is
a Pareto curve across **artifact size, reconstruction fidelity, and
actual model quality/performance**.

## 7. Quality validation: the critical gate

The report notes that Sigma uses a continuous diffusion process over
16-dimensional embeddings. Evaluation should therefore use the model's
actual task and output semantics, not assume that ordinary next-token
exact-match testing is sufficient.

A successful decoder, varied token IDs, or a non-degenerate output is
only a pipeline sanity check. It is not evidence of fidelity.

Recommended validation:

-   Run original and compressed models with identical inputs, seeds,
    sampling settings, schedules, and other configuration.
-   Use a held-out evaluation set not used to construct or tune the
    codebook.
-   Repeat stochastic runs and report variance.
-   Choose task-specific metrics appropriate to Sigma's output domain.
-   Compare output distributions and downstream task performance, not
    only exact output identity.
-   Include representative difficult cases and failure cases.
-   Report quality by layer/tensor or model stage where practical, to
    locate where error accumulates.
-   Keep reconstruction error and behavioral quality as separate
    columns.

The reported relative L2 error of 0.6922 warrants investigation.
Establish whether it is dominated by a subset of tensors, outliers,
normalization/frame behavior, or broadly distributed error. Do not infer
the cause without measurements.

## 8. Memory-tier accounting and precision experiments

The preliminary discussion mentioned an expanded layer of approximately
466 MB and three layers around 1.40 GB. The memory tier was not
established in the information available for this review.

Please label each figure as: - host RAM, - GPU VRAM, - pinned host
memory, - or total process resident memory.

Also clarify whether the figure includes temporary decode buffers,
staging buffers, codebook copies, and transfer buffers.

If the backend supports it, test reconstruction/output precision
alternatives such as FP32 versus FP16 or BF16. FP16/BF16 storage is
approximately half the size of FP32, but this is only a potential
memory/transfer benefit. Measure numerical stability, task quality, and
actual compute throughput before selecting a default.

## 9. Streaming scheduler and lookahead

For a deterministic dense network with a known execution order, fetching
layer N+1 is generally scheduled lookahead rather than speculative
prediction. Fetching N+2 and beyond is deeper prefetch. For
architectures with conditional routing, expert selection, or adaptive
execution, some future requests may genuinely be uncertain.

Measure the scheduler with timestamps and byte counts:

-   demand time: when a block is required by compute;
-   request time: when the storage read is issued;
-   ready time: when the block is decoded and available in its
    destination tier;
-   consumption time: when compute actually uses it.

Useful quantities:

\[ T\_{`\mathrm{ready}`{=tex}} = T\_{`\mathrm{disk}`{=tex}} +
T\_{`\mathrm{decode}`{=tex}} + T\_{`\mathrm{transfer}`{=tex}} \]

\[ T\_{`\mathrm{slack}`{=tex}} = T\_{`\mathrm{demand}`{=tex}} -
T\_{`\mathrm{ready}`{=tex}} \]

Positive slack means the block was ready before demand. Negative slack
means the consumer had to wait, assuming these timestamps use a
consistent clock and define the same pipeline boundary.

Track: - percentage of demanded bytes ready on time; - stall duration
and stall count; - storage throughput and queue depth; - decode
throughput; - transfer throughput; - prefetch hit rate; - late-arrival
rate; - unused/wasted prefetched bytes; - peak staging memory; - buffer
reuse and synchronization overhead.

Test layer N+1 first, then deeper lookahead such as N+2 through N+4. A
larger lookahead window is not automatically better: it may increase
memory pressure and waste bandwidth. The right window is measured, not
guessed.

## 10. Fused lookup and compute: exploratory path

A possible later experiment is to avoid fully materializing a
reconstructed tensor and instead fuse codebook lookup/reconstruction
into the consuming operation (for example, a matrix-vector or
matrix-matrix operation).

Potential benefit: - less intermediate memory traffic; - fewer full
reconstructed buffers; - potentially smaller staging requirements.

Potential costs: - irregular lookup overhead; - more complex kernels; -
reduced arithmetic efficiency; - repeated decoding work; - harder
backend portability.

Benchmark three paths separately: 1. Fully reconstruct then compute. 2.
Reconstruct into a bounded window then compute. 3. Fused
lookup/reconstruction with compute.

Do not assume the fused path wins. It is an optimization hypothesis.

## 11. Reproducibility and audit record

For each experiment, preserve:

-   source model identifier and cryptographic hash;
-   source tensor shapes, dtypes, and total parameter count;
-   NanoQuat-O2 format/version;
-   encoder and decoder source revision/commit;
-   compiler, flags, and build identifier;
-   hardware and storage device;
-   OS, backend, and driver versions where relevant;
-   S, K, prototype precision, frame format, and entropy coder settings;
-   calibration/training dataset identifiers and hashes;
-   random seeds;
-   all artifact component sizes;
-   hashes of the compressed artifact and reconstructed outputs;
-   exact benchmark commands/configuration;
-   raw logs, not only summarized results.

A single audit report should calculate all storage totals from actual
files and metadata. Avoid manually maintained headline numbers that can
drift from the artifact.

## 12. Suggested experiment order

1.  **Freeze and reproduce the current baseline.** Verify all component
    sizes and the reported 0.15691 bpw calculation from the actual
    artifact.
2.  **Complete the S staircase.** Produce size, error, encode/decode,
    and memory results for each block size.
3.  **Profile the frames.** Confirm their actual byte share and residual
    distributions.
4.  **Prototype frame coding.** Delta minima, transformed scale
    residuals, entropy coding, and periodic checkpoints.
5.  **Run behavioral fidelity tests.** Do not postpone this until after
    extensive format optimization.
6.  **Clarify memory tiers.** Resolve the 466 MB/layer and 1.40
    GB/three-layer figures.
7.  **Measure the streaming pipeline.** Timestamp storage, decode,
    transfer, demand, and consumption.
8.  **Tune lookahead and staging.** Use measured stall and waste data.
9.  **Test lower reconstruction precision.** Only retain it if quality
    and speed remain acceptable.
10. **Explore fused lookup/compute.** Treat it as a later optimization
    after a trustworthy baseline exists.

## 13. Questions for the next report

1.  Are the 466 MB per layer and 1.40 GB for three layers host RAM, GPU
    VRAM, or another measurement?
2.  Do the reported 4,476,940 resident bytes include all decoder buffers
    and the full 4,096-block window, or only a measured process
    component?
3.  Is the 74,883,852-byte stream inclusive of frame values, or are
    frame bytes accounted for separately in the report's final 0.15691
    bpw?
4.  Exactly which bytes are included in the 53,040-byte decoder figure,
    and is it part of the model package or a shared runtime?
5.  How many bytes and what percentage of weights use raw fallback?
6.  Does the current format support independent random access to a block
    without decoding a long preceding sequence?
7.  What is the measured decode throughput in blocks/s and bytes/s for
    each tested S?
8.  What is the current end-to-end quality metric for the compressed
    Sigma model, and how does it compare with the original?
9.  Can the reported relative L2 error be broken down by tensor/layer?
10. What are the measured storage-read, decode, transfer, and compute
    times, including stalls?

## 14. Attribution and collaboration

The architectural direction described in the discussions --- block
sizing, lookahead, prefetch strategy, staging priorities, and
measurement goals --- has been directed by JD. Elfie has been
implementing and testing the quantizer/decoder and associated
experimental pipeline. Keep those contributions accurately distinguished
in project notes and credits; the system benefits from both the
architectural direction and implementation work.

Gizmo's role here is outside technical review: identify accounting gaps,
propose experiments, clarify what the evidence establishes, and
challenge claims that are not yet supported by behavioral or end-to-end
measurements.

## 15. Bottom line

The preliminary results suggest a very compact representation and a
small reported resident-memory footprint. Those are promising
engineering signals, but the system's success depends on three separate
questions:

1.  **Can the representation be stored compactly?**
2.  **Can it be reconstructed accurately enough?**
3.  **Can the required data be delivered to computation on time, while
    preserving useful model behavior?**

NanoQuat-O2 should report all three independently. The most immediate
technical targets are to verify complete artifact accounting, reduce
frame overhead without sacrificing random access, establish behavioral
fidelity, and measure whether the streaming scheduler keeps the compute
pipeline fed.

**Do not optimize the compression headline at the expense of model
quality or end-to-end throughput.**
