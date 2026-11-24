# NanoQuatTwo

Sub-bit weight packing by prototypical blocks, in pure C. Build the
big model, print it small, expand it at will. We call the discipline
model lithography.

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
- A four-loop streaming design (one compute loop, one disc-lookup
  loop, two unpack loops, one serialized consumption order) streams
  the whole model at up to 168.8 million weights per second on a
  1.5-core bench VM, bit-identical in every configuration.
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
WHAT-WE-CAN-DO.md for the capability ledger, and
BASELINE-2026-11-23.md for the frozen baseline everything is
judged against. NIGHT-LOG-2026-11-23.md is the raw working record
of the night the experiments ran.

## Build

Production tools (quantizer, runner, inference engine, audit,
frame probe, error probe):

    build.sh          (Linux)
    build.bat         (Windows, MSYS2/MinGW)

Research tools (the experiment harnesses cited in the experiments
file):

    build-experiments.sh
    build-experiments.bat

Everything is C99 with no dependencies beyond the standard library
and libm. The one Python file, tools/nq2-gui.py, is a front end
only and does no model work. Runtime is C, full stop.

## What's in the repo, and what isn't

In: all tool sources under tools/, the toy examples under
examples/, build scripts, the baseline and experiment documents,
and the text receipts under results/.

Not in: the model itself (7.7 GB, NVIDIA's download, their
license), the packed .nq2 artifacts (up to 75 MB each, regenerable
with the quantizer in minutes), and compiled binaries. The baseline
document records the hashes and checksums of the frozen pack so
anyone can regenerate it and verify they got the same bytes.

## Examples

Three toy programs, each one self-contained, each one implementing
everything it needs (output, sine, memory, random) in its own file.
No libc, no libm, no imports. We do not import libraries; we write
our own.

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
- **Consultee:** Gizmo, whose engineering handoffs set the research
  program the experiments file answers.
- **Executive direction:** JD (David Rittenhouse), Wet Wafer Works.
  The questions were his: how much intelligence fits in a quad word,
  whether sub-half-bit packing is possible, whether it can be taught
  to read, and the four-loop streaming architecture the engine now
  runs on.

## License

MIT. See LICENSE.
