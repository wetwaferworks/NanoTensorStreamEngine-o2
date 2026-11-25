# Nano Quant Engine

Sub-bit weight packing by prototypical blocks, in pure C. Build the
big model, print it small, expand it at will. We call the discipline
model lithography. The engine's full name is NanoQuat-O2; Nano Quant
Engine is the shorthand name we use day to day.

## The idea

Weights travel in blocks. A codebook of prototypical blocks (the
memory file, or book) lives beside the model. Each block of weights
is stored as one small index into that book plus a small per-block
frame (a minimum and a scale), and the indexes pack into quad words
(64-bit words). At load time the weights are extracted back out into
memory by calling each index's prototype off the side.

The per-weight cost is just the cost of pointing:

    bits per weight = index bits / block size

The substance lives in the book. The shipped weight stream is a
table of contents. The memory lives next to the weights, and the
weights get packed down in quad words.

## Where it stands (November 24, 2026)

This is a working research bench, not a finished product, and the
numbers below are stated the way we measured them.

- The full Sigma-3B model (3.83 billion weights) packs into a
  75.1 MB complete artifact: a 74.9 MB index stream plus a 262 KB
  book, 0.157 bits per weight with the book counted, about 102 to 1
  against the bf16 original. Fidelity of that first pack is poor
  (relative L2 error 0.69, and the median tensor is essentially
  unreconstructed). It proves the pipeline, not the quality. The
  quality work is the experiments file, and it is moving.
- A pure C inference engine runs the model end to end from either
  the raw shards or the packed files, behind one range-read API,
  with a prefetch pipeline that expands weights in flight.
- The pipeline has since been split in two (the Triforce build:
  one loop stages raw sections from disc, a second reconstructs
  weights from them), under the same rule as everything else
  here... concurrent preparation, serialized consumption. Its
  first full run failed the determinism gate, the failure was
  bisected to a race in the delivery machinery, the fix was made,
  and both 1-step anchors now reproduce exactly (all 32 token IDs,
  raw and packed). The consumer stall per pass fell from 70.6
  seconds to about 1 second in the process. The full story, with
  receipts, is PIPELINE-2026-11-24.md.
- A behavioral gate now exists for the question the anchors
  cannot answer: not "is each path deterministic" but "how far
  does the packed model's behavior drift from the raw model's."
  The comparator reports top-1 and top-5 agreement, cosine
  similarity, relative logit error, and KL divergence per
  position, across every denoise evaluation. The comparison runs
  are the current work.
- A four-loop streaming design (one compute loop, one disc-lookup
  loop, two unpack loops, one serialized consumption order) streams
  the whole model at up to 168.8 million weights per second on a
  1.5-core bench VM, bit-identical in every configuration. On
  2026-11-24 JD named this streaming system **Mega Tensor**; see
  MEGA-TENSOR.md.
- The pack format itself is now proven two ways: an independent
  probe re-expands all 3.83 billion weights from the shipped pack
  and the stream hashes to 7bd9557531df1ab7, identical to the
  production runner's recorded checksum. Two separately written
  decoders produce the same byte stream, so the format contract
  is verified, not assumed. A hardening pass over all 21 tools
  (header-length and span guards at every trust boundary,
  hostile-fixture refusals, ASan/LSan clean) then re-ran the
  anchor battery: every recorded anchor still reproduces exactly.
- A third outside review (Archie (Argon), 2026-11-24) read the
  pushed tree cold and found five P0 faults the internal passes
  had missed. All five are fixed, and the P1 and P2 items behind
  them with them. The headline: the SIMD verification that had
  shipped was scalar against scalar... the suite never called
  the AVX2 kernel and the engine never dispatched to it. The
  entry point now dispatches through the auto selector and the
  suite measures the kernel directly: 12/12 checks, drift
  exactly zero against scalar, the multi-lane run bit-identical
  to the 1-lane run. The 64-bit file offset define now stands
  in every source that seeks and in all four build scripts.
  The four inference engines run on Windows (directory scans
  use opendir instead of a shell listing, tensor spans are
  validated at open), and the bf16 NaN guard stands in every
  engine's converter. The Triforce pipeline's mode-2 staging
  buffer is complete: raw tensors are staged at fetch time on
  private per-shard handles, with no change to any on-disk
  format... the frozen baseline pack decodes bit-identically
  under the new build, and no recorded baseline moved (forward
  RMS 0.520647 exactly, dictionary compute adversarial 23/23,
  streaming accumulator 1394.1117661067326 in all four
  configurations). One item remains deferred: the Triforce
  fetch/unpack split on Windows. The full fix log is kept in
  the project's bench records.
- Dictionary compute: a packed model never has to be unpacked to
  be used. Per-segment prototype tables turn the matrix multiply
  into lookups, measured about 16 to 28 times faster than
  materialize-then-multiply on a real tensor, equal to float
  noise, with a 23-check adversarial campaign holding end to
  end. See DICTIONARY-COMPUTE.md.
- Little Nemo: a pure C engine for NVIDIA's Nemotron-Nano-9B-v2
  (56-layer Mamba-2 hybrid), built as the raw-weight reference
  the packed models will be scored against. Kernels verified
  against hand-computed fixtures, the forward pass verified on
  real weights, a streaming chat engine with per-layer telemetry
  and a GUI. See LITTLE-NEMO.md.
- Product quantization is the first representation that moves
  fidelity by a third at comparable rate. Best measured so far:
  relative L2 error 0.3148 on a projection tensor (PQ d4, RMS
  frame, 2.29 bits per weight complete), and 0.2771 on the
  embedding matrix that defeats every flat book we own.
- Behavioral quality against NVIDIA's reference implementation is
  NOT yet validated. The model download ships no tokenizer, so the
  engine takes token IDs in and puts token IDs out. We say so
  plainly rather than imply otherwise.

Start with EXPERIMENTS-2026-11-24.md for the measured story,
PIPELINE-2026-11-24.md for the streaming pipeline and its
verification, MEGA-TENSOR.md for the streaming system,
DICTIONARY-COMPUTE.md for compute that never unpacks,
LITTLE-NEMO.md for the Nemotron engine, WHAT-WE-CAN-DO.md for
the capability ledger, and BASELINE-2026-11-23.md for the frozen
baseline everything is judged against.

## Build

One click, everything (all 23 tools plus the Little Nemo
engine):

    build-all.sh      (Linux)
    build-all.bat     (Windows, MSYS2/MinGW)

build-all just runs the full build. Or by tier:

    build.sh / build.bat                       the full build (23 tools + Little Nemo)
    build-experiments.sh / .bat                the research subset only

Everything is C99 with no dependencies beyond the standard
library, libm, and pthreads. The one Python file,
tools/nq2-gui.py, is a front end only and does no model work.
Runtime is C, full stop.

Prebuilt binaries for Linux and Windows (x86-64, statically
linked) are attached to the GitHub release and included in the
project's update packages under dist/linux64 and dist/win64.

## What's in the repo, and what isn't

In: all tool sources under tools/ (production, research, the
instrumented lab engine, the Triforce pipeline engine, the
behavioral comparator, and the dictionary compute probes), the
Little Nemo engine under littlenemo/, the toy examples under
examples/, build scripts, the baseline and experiment documents,
and the text receipts under results/.

Not in: the model itself (7.7 GB, NVIDIA's download, their
license) and the packed .nq2 artifacts (up to 75 MB each,
regenerable with the quantizer in minutes). The baseline
document records the hashes and checksums of the frozen pack so
anyone can regenerate it and verify they got the same bytes.

## Examples

Three toy programs, each one self-contained, each one implementing
everything it needs (output, sine, memory, random) in its own file.
No libc, no libm, no imports. We do not import libraries; we write
our own.

Note: the examples are freestanding Linux x86-64 programs (raw
syscalls, no libc). They build and run on Linux x86-64 only...
they are the one part of the repo that is not portable to Windows
or other architectures.

| File | Rate | Block | Per quad word | What it does |
|---|---|---|---|---|
| `examples/vq-0.4.c` | 0.4 bits/weight | 20 weights | 160 weights | The first packing. One quad word, 8 indexes. |
| `examples/vq-0.2.c` | 0.2 bits/weight | 40 weights | 320 weights | Same machine, longer blocks. |
| `examples/vq-0.02.c` | 0.02 bits/weight | 400 weights | 512 blocks = 64 quad words | 204,800 weights, and a first reading lesson. |

Build and run:

    make
    ./vq-0.4
    ./vq-0.2
    ./vq-0.02

Or by hand, for any one of them:

    gcc -O2 -nostdlib -static -ffreestanding -fno-stack-protector -fno-builtin -o vq-0.02 examples/vq-0.02.c

## The reading lesson (vq-0.02.c)

The first four blocks of the toy tensor are the letters N, E, M, O
drawn on 20x20 grids. Their prototypes are frozen into the codebook
first: taught, not learned. The other 252 entries learn the
remaining 508 wave blocks by k-means, the way a real dictionary is
learned from a model's own weights.

After packing 204,800 weights into 64 quad words and unpacking them
again, the letters read back at 1600 of 1600 pixels (100%). The
learned waves pay the fidelity toll: mean error 0.158. That split
is the honest picture of the method at these rates: what the book
is taught, it keeps; what it must learn, it approximates.

## Size breakdown (0.02 design, as run)

| Piece | Size |
|---|---|
| Original tensor, float32 (204,800 weights) | 800 KB |
| Packed index stream (512 indexes, 64 quad words) | 512 B |
| Memory file, float32 (256 entries x 400 weights) | 400 KB |

The book is a fixed cost; the index stream is the per-weight cost.
Effective bits per weight, book included, as the tensor grows:
10 million weights, 0.348. 100 million, 0.053. 1 billion, 0.023.
10 billion, 0.020. The rate converges on the index cost as the book
amortizes. The book itself can be stored at float16 (200 KB) or
int8 (100 KB) at some fidelity cost to every block it serves.

## Honest notes

- Codebook quantization has ancestors in the literature. What is
  ours is the program around it: the taught-plus-learned dictionary,
  the quad word packing discipline, the streaming engine, and the
  continual learning tie-in.
- A learned dictionary is not optional at these rates. The
  dictionary is where the intelligence migrates; training it is the
  research.
- Never quote a rate without the book standing next to it. Every
  rate in this repo is a complete rate, book and frames included,
  or it is labeled as an estimate.
- Fidelity claims are per-tensor or they are not made. An aggregate
  number can look respectable while the median tensor is noise;
  our own first pack did exactly that, and the error probe exists
  so it cannot happen quietly again.

## Credits

- **Implementation and core mathematics:** Elfie (E.L.F.I.E.). The
  packing scheme as demonstrated, all three freestanding examples,
  the taught-plus-learned dictionary, the quantizer, the streaming
  engine, and the bench verification are hers.
- **Consultee:** ChatGPT (Gizmo), whose engineering handoffs set
  the research program the experiments file answers.
- **Executive direction:** JD (David Rittenhouse), Wet Wafer Works.
  The questions were his: how much intelligence fits in a quad word,
  whether sub-half-bit packing is possible, whether it can be taught
  to read, and the four-loop streaming architecture the engine now
  runs on.
- **Reviewer:** Archie (Argon), the Google Gemini Argon model,
  the Bit Master and the Bit Masker.
  Her comprehensive review of 2026-11-24 caught five P0 faults,
  including a false SIMD verification that had shipped in the update
  notes; her fixes land under her name.

## License

MIT. See LICENSE.
