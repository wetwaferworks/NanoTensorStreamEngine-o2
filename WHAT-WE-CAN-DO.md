# Nano Quant Engine: What Can We Do

*The plan, written 2026-11-23 while the first real model downloads.*

*Status note, 2026-11-24: the plan below was kept as written. What
actually got built and measured lives in EXPERIMENTS-2026-11-24.md
(the fidelity program) and PIPELINE-2026-11-24.md (the streaming
pipeline, its race, and its fix). The short version: the streaming
design below is built and verified against recorded anchors, the
fidelity bottleneck turned out to be prototype coverage rather
than rate, and product quantization is the first lever that moves
it.*

## Where we stand

Three freestanding toys prove the arithmetic closes at 0.4, 0.2, and 0.02 bits per weight, and that taught information (letters frozen into the dictionary) survives the packing intact while learned information degrades gracefully. The toys use made-up weights. The next step is real weights, a real model, and the streaming design.

## The streaming design (megatexture for weights)

A nanoquantized model ships as two files:

- The **index stream** (.nq2): one small index per block of weights, plus a per-block minimum and scale (the affine frame the block was normalized into before packing).
- The **memory file** (.nq2book): the codebook of prototypical blocks. This is the memory that lives next to the weights.

At run time the weights are **never fully expanded**. The runner streams the index stream and materializes only the blocks being executed, into a small rolling window, the way Doom 2016's megatexture keeps only the visible tiles resident. Resident memory is the window, not the model. Expansion is exact: the same index always calls back the same prototype, so streaming changes nothing about the result, only about what is resident.

Why this shape fits: the stored form is nearly all address and almost no substance, so reading it is cheap. The cost moves to decode (expanding prototypes), which is CPU work. Dense models execute layers in order, so the next layer can be expanded while the current one runs. JD's prefetch ruling (2026-11-23, refined same evening): the inference-time streamer runs a **two-step prefetch** as a rolling pipeline, not a bulk lookahead. While layer N executes, layer N+1 is expanding; layer N+2 waits next in line, still just indexes on disc. Only when a layer is dropped does the next one start expanding. The point, in JD's words, is the memory bound: **at most two layers are ever open in memory at once** (the one executing, the one expanding), no matter how large the model grows. For a dense model the layer order is known statically, so this pipeline is exact rather than speculative.

**Adaptive depth (JD, 2026-11-23, later):** where memory allows, the pipeline may deepen to **four layers in flight**: one executing, two expanding, and one falling away as the next enters. Depth is a function of the machine's memory capability, chosen at startup; the two-deep form remains the floor for small machines.

**Predictive layers (JD, 2026-11-23, late, answering Gizmo's step 3):** the dense order N+1, N+2, N+3, N+4 covers the static case, but a future layer choice may depend on the previous layer's output (routing, early exit, adaptive compute). Provision: prepare up to **two additional predictive layers, one of them expanded, for a total of three expanded layers**. Costed on Sigma-3B (fp32 expansion, 116.4M weights per layer): one expanded layer is about 466 MB, three about 1.40 GB, against the current two-slot tensor pipe at 216 MiB. Plus the standing footprint: E matrix 8 MB, workspace about 5 MB, head and logits buffers about 60 MB, books 0.26 MB global (about 62 MB per-tensor at block 256), and the KV cache at about 106 KB per token across 26 layers (7.4 MB at smoke-test scale, about 425 MB at 4K context). Totals: roughly 1.5 GB at short context, 1.9 GB at 4K. Levers if that is too much: fp16 expansion halves the layer cost; tensor-granular staging of the predictive layer (packed segments resident, expand on selection) trades a cold-start expansion for most of the saving.

**The baseline block-size question (JD, 2026-11-23):** JD's lean is **512**, reasoned from NVMe and DDR4 throughput, and his stated objective is end-to-end latency: the book size should reflect how long a tensor takes to travel disc to memory to CPU and back, until the token is on screen, targeting a real-time rate. The staircase experiment (per-tensor books, blocks 32 to 4096) is the evidence; Elfie is to test and recommend the baseline. JD's name for the whole discipline, coined the same night: **model lithography**... build the big model, print it small, expand it at will. (He is renaming the repo to NanoQuantO2 himself.) Mixture-of-experts routing is data-dependent, and that is where a genuine page-fault path (expand on demand, keep hot experts resident) earns its keep.

## Streaming proof: what is built, what is designed

- **Stage 1, disc to memory: built and verified.** `tools/nq2-run.c` streams a packed model from disc through a rolling window of about 4 MB, expanding block by block, and never holds the full expansion. It reports packed versus expanded size, peak resident memory, expansion throughput, and an FNV-1a checksum of the entire expansion, computed on the fly. The checksum was verified against an independent reconstruction, and a `peek` mode proves random access: any single block expands on demand by seeking, without reading the rest. The remaining stage 1 demonstration is to run this against a real packed slice of Sigma and put the peak-resident number beside the original's.
- **Stage 2, memory to GPU: designed; Cortana's lane (JD's ruling, 2026-11-23).** She will build it when she can debug directly on JD's machine. Elfie's scope is stage 1: disc to system memory to CPU. The naive path (expand on the CPU, then copy the expanded weights across the bus) throws the whole advantage away: the expanded form is the big one. The NanoQuatTwo path ships the *packed pair* across the bus instead. The index stream and the memory file together are tens of megabytes even for a multi-billion-parameter model, so they can live resident in VRAM, and a small decode kernel materializes each layer's weights in VRAM, in the rolling window, exactly where the matrix math runs. The bus carries addresses; the substance is born on the device. Nothing about the format changes: the same .nq2 and .nq2book files feed the CPU runner today and the GPU decoder tomorrow.

## The first real run

Subject: **nvidia/Nemotron-Labs-Sigma-3B**, the smallest Sigma (3B-class continuous diffusion model, 26 layers, hidden 3072), weights under the NVIDIA Nemotron Open Model License, about 7.14 GB in bfloat16.

Pipeline:

1. `tools/nq2-quantize.c` (the quantizer, pure C): reads the safetensors shards tensor by tensor, normalizes each block into its affine frame, learns the codebook by k-means over a sample of blocks drawn from across the model, then packs every block to its nearest prototype. Writes the .nq2 stream and the .nq2book memory file, and reports true fidelity (relative L2 error) on the real weights. Per JD's purity rule for this project: everything that runs is C. Python is not used anywhere in NanoQuatTwo; the only scripts allowed are build scripts (build.sh, build.bat, the Makefile). C is the language JD and other humans can read, and it compiles to clean assembly.
2. `tools/nq2-run.c` (the runner, plain C): loads a packed model and streams it back open, reporting packed size versus expanded size, the peak resident window, expansion throughput, and a checksum of the fully expanded stream, computed without ever holding it. Builds with `build.sh` (Linux) or `build.bat` (Windows, MinGW gcc). A prebuilt Windows exe is shipped alongside.

Block size and codebook size are knobs, not constants. The bench run sweeps them (block 128 / 256 / 1024 at 256 entries) and reports the fidelity each rate actually buys on real weights. Rates, book included in the per-block frame cost:

| Block | Index | Frame (min+scale) | Rate |
|---|---|---|---|
| 128 | 8 bits | 32 bits | 0.3125 bits/weight |
| 256 | 8 bits | 32 bits | 0.15625 bits/weight |
| 1024 | 8 bits | 32 bits | 0.0390625 bits/weight |

## What we can do from here

- **Per-layer books.** One global book is the simple start. A book per layer (or per tensor type) matches each layer's own statistics and should buy fidelity at the same rate. The book cost stays small.
- **Entropy-code the stream.** The toy's indexes carried 7.596 bits of information per 8-bit index. A rANS pass over the index stream recovers most of that slack.
- **Residual books.** A second codebook packs the *error* of the first. Two-stage VQ trades rate for fidelity in controlled steps.
- **Taught entries for real structure.** The letters lesson generalizes: freeze prototypes for patterns known to matter (embeddings, norms, the diffusion head) and let k-means learn the rest.
- **A real inference path.** The runner proves streaming and integrity. Wiring the expanded stream into an actual forward pass (the Sigma architecture on CPU, block by block) is the step after.
- **The LithoKraft companion.** A nanoquantized micro model plus its memory file is exactly the artifact a player carries. The companion can be a continual learner because its book can keep learning while its stream stays small.

## Accounting discipline (Greene, "Compression and Intelligence", applied 2026-11-23)

From 2026-11-23, every size claim for a pack is a **complete description length**: stream bytes + book bytes + decoder bytes, divided by weights. The stream alone is never quoted as "the" compression; per Greene's two-part code, the book is program cost and the program is charged. The tool is `tools/nq2-audit.c`: it parses a pack, charges all three parts, measures the index distribution's entropy (H0 marginal, H1 conditional on the previous index within a tensor), and audits codebook utilization (an entry earns its book bytes only if blocks select it).

The audit's first structural finding, independent of any dataset: **the stream is mostly frame.** Each block costs 5 bytes, of which 4 are the fp16 min/scale frame and only 1 is the index. At block 256 the frame is 0.125 of the stream's 0.15625 bits/weight (80 percent); at block 512 it halves to 0.0625. So the block-size baseline question is really two questions: how fast distortion grows with block size (the staircase measures this), against how fast the frame amortizes (pure arithmetic, 32/block bits per weight). Entropy-coding the indexes attacks only the 1 byte in 5; the frame is the bigger target after that (shared or coded frames are an open direction).

## Honest limits, stated up front

- Nobody here claims a 3B model packed at these rates keeps its full quality. The bench run reports what the fidelity actually is. The research question is where the knee of the curve sits for real weights, and which of the refinements above moves it.
- Sigma's own tooling states CPU inference is not supported for the original model. The nanoquantized streaming path is a different artifact with different trade-offs, not a port of NVIDIA's runtime.


## Accounting and findings update (2026-11-23, late; Gizmo handoff round)

Two totals are now reported for every pack, always labeled (adopted from Gizmo's engineering handoff):

- **Model artifact total:** stream + books + fallback bytes. For the corrected full pack: 75,146,008 bytes, 0.15680 bits/weight.
- **Self-contained package total:** artifact + decoder. Linux decoder: 75,199,048 bytes, 0.15691 bits/weight. Windows decoder total: 75,401,051 bytes.

Frame study (nq2-frameprobe, on the full pack): the frame is 80.0% of the stream. Entropy-coding the frame codes themselves prices the frame at 0.0629 bits/weight (half), random access preserved; delta coding minima gains nothing (7.4991 vs 7.5178 bits); scale log2-deltas at 1/64 reach 6.1412 bits. The full delta-plus-checkpoint machinery adds only about 0.0096 bits/weight beyond the plain entropy route.

Error study (nq2-errprobe): the 0.6922 total is dominated by a few tensors. The replaid embedding matrix alone holds 39.80% of the squared error (rel L2 0.9922); the top 10 tensors hold 54.6%; the median per-tensor relative L2 is 0.9943. The fidelity bottleneck is prototype coverage of normalized block space, not the frame and not the bit budget: per-tensor books at block 32 spend 1.27 bits/weight complete for 0.6764 on the layers 0-3 slice. Full treatment: RESPONSE-TO-GIZMO-2026-11-23.md.


## Streaming direction (JD, 2026-11-23 night) and format v3 note

JD's framing: treat weight delivery as a geometry streaming problem, disc to memory to CPU at gaming speed, feeding the sensors and the lookahead with no avoidable throughput stall. Consequences worked tonight: the runner now times the streaming path (read + reconstruct) separately from the checksum verification pass; the engine's packed reader expands block-aligned ranges directly into the destination (no bounce copy) and runs the packed stream through a 256 KB stdio buffer; the prefetch pipe now reports telemetry (hit rate, consumer stall on misses, mean ready slack on hits, load-path MB/sec), so lookahead health is measured rather than assumed.

Format v3 layout note, from the same framing: the v1/v2 stream stores three separate sections (all indexes, all minima, all scales), so every range read is three seeks. Interleaving the per-block record (index byte + fp16 min + fp16 scale, 5 bytes) makes each tensor's records one sequential run, like a vertex buffer: one seek and one linear read per tensor, true single-record random access, and prefetch that behaves like geometry streaming. This joins the v3 bundle already planned (frame entropy coding, embedded tensor names/shapes, tiny tensors stored raw).


## Four-loop streaming architecture (JD, designed 2026-11-24 ~00:05 EST)

JD's design, stated twice from the podium for the record: the MAIN loop does all computation on the CPU (the tensor calculations). A second loop does nothing but disc lookups, grabbing the prefetch and predictive tensors needed next. Two further loops do the memory unpacks and hand the finished tensors forward to the main loop. JD's framing: the Timing Handshake applies to this architecture... the stages run concurrently, but consumption happens in one serialized order, so the forward pass speeds up without the concurrency ever leaking into the result. The end goal in his words: unpack a model and stream it in real time from disc to the CPU, tokens in, tokens out.

Test harness: tools/nq2-streamtest.c implements the four loops over the real pack in the engine's forward demand order, with a sequential reference mode (unpackers 0) whose accumulator every pipelined configuration must reproduce exactly, per-stage busy times, and main-loop stall as the headline metric. First smoke (2 layers, contended bench): sequential 3.41 s at 68.3 M weights/sec; two unpack loops 2.41 s at 96.8 M weights/sec; accumulators identical to 17 significant digits (118.28021318475126). Full 26-layer benchmarks and refinements: NIGHT-LOG-2026-11-23.md.
