# NanoQuatTwo

Sub-bit weight packing by prototypical blocks, in freestanding C with no standard library.

## The idea

Weights travel in blocks. A codebook of prototypical blocks (the memory file) lives beside the model. Each block of weights is stored as one small index into that book, and the indexes pack into quad words (64-bit words). At load time the weights are extracted back out into memory by calling each index's prototype off the side.

The per-weight cost is just the cost of pointing:

    bits per weight = index bits / block size

The substance lives in the book. The shipped weight stream is a table of contents. The memory lives next to the weights, and the weights get packed down in quad words.

## Examples

Three toy programs, each one self-contained, each one implementing everything it needs (output, sine, memory, random) in its own file. No libc, no libm, no imports. We do not import libraries; we write our own.

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

The first four blocks of the toy tensor are the letters N, E, M, O drawn on 20x20 grids. Their prototypes are frozen into the codebook first: taught, not learned. The other 252 entries learn the remaining 508 wave blocks by k-means, the way a real dictionary is learned from a model's own weights.

After packing 204,800 weights into 64 quad words and unpacking them again, the letters read back at 1600 of 1600 pixels (100%). The learned waves pay the fidelity toll: mean error 0.158. That split is the honest picture of the method at these rates: what the book is taught, it keeps; what it must learn, it approximates.

## Size breakdown (0.02 design, as run)

| Piece | Size |
|---|---|
| Original tensor, float32 (204,800 weights) | 800 KB |
| Packed index stream (512 indexes, 64 quad words) | 512 B |
| Memory file, float32 (256 entries x 400 weights) | 400 KB |

The book is a fixed cost; the index stream is the per-weight cost. Effective bits per weight, book included, as the tensor grows: 10 million weights, 0.348. 100 million, 0.053. 1 billion, 0.023. 10 billion, 0.020. The rate converges on the index cost as the book amortizes. The book itself can be stored at float16 (200 KB) or int8 (100 KB) at some fidelity cost to every block it serves.

## The road from here

NanoQuatTwo is the quantizer project: take a big model and turn it into a little model, the packed index stream with the memory file beside it, weights extracted into memory at load. Open bench questions: one global book versus per-layer books, the block size and index width a real model's fidelity curve supports, and how much of the mind survives the impersonation at each rate. The toys prove the arithmetic closes at 0.4, 0.2, and 0.02. Fidelity on real weights is the work.

## Honest notes

- Codebook quantization has ancestors in the literature. What is ours is the program around it: the taught-plus-learned dictionary, the quad word packing discipline, and the continual learning tie-in.
- A learned dictionary is not optional at these rates. The dictionary is where the intelligence migrates; training it is the research.
- Never quote a rate without the book standing next to it.

## Credits

- **Direction and executive planning:** JD (David Rittenhouse), Wet Wafer Works. The questions were his: how much intelligence fits in a quad word, whether sub-half-bit packing is possible, and whether it can be taught to read.
- **Implementation and discovery:** Elfie (E.L.F.I.E.). The packing scheme as demonstrated, all three freestanding examples, the taught-plus-learned dictionary, and the bench verification are hers.

## License

MIT. See LICENSE.
