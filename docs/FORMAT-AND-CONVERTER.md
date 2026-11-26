# The NQ2 pack format (v3) and the converter

A pack is two files that travel together, plus an optional manifest:

| File | What it is |
|---|---|
| `PREFIX.nq2` | the index stream: a small header, then for every block of weights an index into a book and a float16 frame |
| `PREFIX.nq2book` | the memory file: the codebook(s) the indexes point into |
| `PREFIX.nq2manifest` | (v3) a text sidecar: the pack's generation id and per-tensor digests (and, when calibrated, the Precision Gate lines) |
| `PREFIX.report.txt` | the converter's own report: counts, rates, relative L2 per tensor |

The converter is `nq2-quantize` (Elfie's, pure C99, built by `build.bat` /
`build.sh`). It reads Hugging Face safetensors shards directly (dtypes BF16,
F16, F32) and writes a pack. Everything below was verified against
`tools/nq2-quantize.c` and `tools/nq2-run.c`.

## The idea in one paragraph

Weights travel in blocks of `--block` consecutive values (256 by default). Each
block is normalized into an affine frame, a minimum and a scale stored as
float16, and the normalized shape is replaced by the nearest of `--entries`
prototype blocks (256 by default, so an index fits one byte) from a book learned
from the model's own blocks by k-means over a random sample. Reconstruction is

    w = s * C[idx] + m

with `C` the book, `idx` the stored index, `(m, s)` the frame. Per block that
costs one byte of index and four bytes of frame; at block 256 the stream alone is
about 0.156 bits per weight. The book is learned, not generic: the dictionary is
where the intelligence migrates, and training it is the research.

## Size and rate: one pinned example

Sigma-3B, global book, block 256, entries 256: 243 tensors, 3,834,051,584
weights, index stream 74,883,852 bytes, book 262,152 bytes. That is 0.1563 bits
per weight for the stream alone and about 0.157 with the book counted
(roughly 102 to 1 against bf16). Relative L2 error on the real weights is
0.6922, so the shipped pack is a pipeline proof, not a quality artifact. Never
quote a rate without the book standing next to it: every rate in this repository
is a complete rate or is labeled an estimate. `nq2-audit` applies the
two-part-code accounting to a pack.

## Converter reference: `nq2-quantize`

```
nq2-quantize --out PREFIX shard.safetensors [more shards] [options]
```

Shard arguments are sorted into canonical (filename) order, so argument order
never changes the pack.

| Option | Default | Meaning |
|---|---|---|
| `--out PREFIX` | required | output prefix for `.nq2`, `.nq2book`, `.nq2manifest`, `.report.txt` |
| `--block N` | 256 | weights per block; 1 to 4096 (every reader enforces the 4096 ceiling) |
| `--entries N` | 256 | prototypes per book; 1 to 256 (an 8-bit index) |
| `--budget N` | 250000000 | weight budget: tensors are taken whole, in order, while the running total is under it. The default silently truncates a full model, so full packs pass a large value (the pinned Sigma pack used 4.2e9) |
| `--filter a,b` | all | keep tensors whose names contain any listed substring |
| `--sample-blocks N` | 32768 | blocks sampled to learn the global book (at least 1) |
| `--tensor-sample-blocks N` | 8192 | blocks sampled per tensor for per-tensor books (at least 1) |
| `--iters N` | 8 | k-means iterations |
| `--books global\|per-tensor` | global | one book for the whole model, or one book per tensor (see below) |
| `--norm minmax\|rms` | minmax | frame: min and range, or RMS with no offset (the stream layout is identical, so existing readers handle RMS packs unchanged) |
| `--format 1\|2\|3` | 3 | layout version; 1 and 2 are legacy writers kept for reproducing older packs |
| `--pq D` | off | product quantization with subvector dimension 4 or 8 (v3, per-tensor books, block 256 only) |
| `--budgets` | off | per-tensor block sizes by sensitivity (v3, per-tensor books, block 128 or more; not with `--pq` or `--residual`) |
| `--embed-fp16` | off | store `embed_tokens` tensors as raw float16 instead of packing them (v3, per-tensor books) |
| `--residual` | off | experimental: train a second book on the first stage's residuals and report the two-stage fidelity and rate; the written stream stays single stage |
| `--threads N` | logical CPUs | worker threads, 1 to 256; output is byte-identical for any N |

Option discipline (refused with a plain message): `--pq`, `--budgets` and
`--embed-fp16` need `--books per-tensor` and `--format 3`; `--pq` takes only 4 or
8 and only block 256; `--pq` and `--budgets` cannot be combined; per-tensor
books are format 2 or 3; the global-book layout is format 1 or 3.

### Per-tensor books (`--books per-tensor`)

Each tensor is sampled, learned and packed with its own codebook, so every book
speaks only its own tensor's language. The global book costs 262,152 bytes once;
per-tensor books cost 262,164 bytes each at block 256 and 256 entries, so a
model with hundreds of tensors pays for the fidelity in book bytes. The report
lists per-tensor relative L2 and the complete rate, stream plus books.
Measured on Sigma (Lane 4, see [ACCEL.md](ACCEL.md) and the behavioral record):
per-tensor books lower the weighted tensor relative L2 from 0.6922 to 0.6626
with `embed_tokens` kept at float16, at a complete rate of 1.9525 bits per
weight, dominated by that float16 embedding (402,653,184 weights).

### Product quantization (`--pq 8` or `--pq 4`)

Each normalized 256-block is split into subvectors of D dimensions (32
subspaces at d8, 64 at d4); each subspace gets its own k-means book of 256
entries, so the representable set grows combinatorially while the book bytes
stay at the flat-book level. The index is block/D bytes. The frame must be
shared across the block (the offset applies once; the "offset trap" in the
adversarial DTC campaign shows what the naive form costs). Measured weight-space
curve on one Sigma tensor (layer 0 `q_proj`, RMS frame, from the behavioral gate
record): flat VQ block 256, relative L2 0.9112 at 0.3229 bits per weight
complete; PQ d8, 0.5613 at 1.2917; PQ d4, 0.3148 at 2.2917.

### Per-tensor budgets (`--budgets`)

The tensors holding the top 50 percent of the squared error re-pack at block/4
and the next 30 percent at block/2, with their books re-learned. The header
carries a per-tensor block size.

### Embedding at float16 (`--embed-fp16`)

Tensors whose names contain `embed_tokens` are stored as a raw float16 section in
place of blocks. Note the naming caveat recorded in Lane 4: on Sigma this keeps
`encoder.embed_tokens` (402 M weights), not the replaid embedding matrix that
the experiments file recommended for float16.

### Multithreading

`--threads` splits the work into disjoint chunks claimed through a lock-free
counter; nothing is locked, a thread that fails to start costs speed and never
correctness, and the flat v3 per-tensor path runs tensor-parallel with bounded
RAM. The output is byte-identical at any thread count, and the generation id is
the same for the same inputs. (Co-implemented by Cortana at JD's direction; the
quantizer itself is Elfie's.)

### Examples

```
:: the pinned Sigma pack: global book, block 256 (the baseline recipe)
nq2-quantize --out sigma-full-b256 --budget 4200000000 shard-1.safetensors shard-2.safetensors

:: per-tensor books, RMS frame, embed_tokens at float16
nq2-quantize --out sigma-pt-b256 --budget 4200000000 --books per-tensor --norm rms --embed-fp16 shard-*.safetensors

:: product quantization d8 and d4 (the recipe of the Lane 4 packs C and D)
nq2-quantize --out sigma-pq8 --budget 4200000000 --books per-tensor --norm rms --embed-fp16 --pq 8 shard-*.safetensors
nq2-quantize --out sigma-pq4 --budget 4200000000 --books per-tensor --norm rms --embed-fp16 --pq 4 shard-*.safetensors
```

Each Lane 4 pack took 17 to 20 minutes at 12 threads (BelowNormal priority) and
1.3 to 1.7 GB peak memory.

## Format v3, byte by byte

All integers little-endian. Verified against the reader in `nq2-run.c`.

**Index stream `.nq2`**

| Field | Type | Notes |
|---|---|---|
| magic | 4 bytes | `NQ2` and a zero byte |
| version | u32 | 1, 2 or 3 |
| block | u32 | 1 to 4096 |
| entries | u32 | 1 to 256 |
| total weights | u64 | |
| blocks | u64 | |
| tensor count `n` | u32 | v2 and v3: the per-tensor segment table follows |
| blocks per tensor | `n` x u64 | v2 and v3 |
| generation id | u64 | v3: content-derived (below) |
| flags | u32 | v3: value 1 = PQ, 2 = per-tensor block sizes, 4 = raw float16 section |
| PQ dimension | u32 | present when the PQ flag (1) is set |
| per-tensor block sizes | `n` x u32 | present when the block-size flag (2) is set |
| raw section | u32 count, then `count` x u32 | present when the raw flag (4) is set |

Then the body: for every block, the indexes, then the frame minima (float16),
then the frame scales (float16); a raw tensor stores its float16 bytes as is.

**Memory file `.nq2book`**

| Field | Type | Notes |
|---|---|---|
| magic | 4 bytes | `NQ2B` |
| v1 | block, entries, then floats | one global book |
| v2 | version, block, entries, book count, then floats | one book per tensor |
| v3 | version, block, entries, book count, generation id (u64), PQ dimension if flagged, then floats | |

The books are float32.

## Pack identity

- **Generation pairing.** Both v3 headers carry a 64-bit generation id: an
  FNV-1a 64 digest of the pair's own content (geometry, tensor names, book bytes,
  index and frame bytes, raw-section bytes), computed before the headers are
  written. Two files packed together share it by construction; a stream and a
  book from different runs disagree, and every v3 reader refuses the pair in
  plain words naming both generations. Because it is content-derived,
  re-packing the same model with the same options reproduces the same id.
- **Digest manifest.** `PREFIX.nq2manifest` lists, per tensor, its block range
  and two FNV-1a digests: one over the tensor's stored stream bytes (indexes,
  then frame minima, then frame scales; the raw bytes for a raw tensor) and one
  over its book's stored floats. Readers verify both at open when the manifest
  is present.
- **Legacy packs (v1, v2)** carry no identity. Readers bind them positionally, as
  before, and print a notice.
- **Gate lines.** A calibrated pack carries Precision Gate lines after the tensor
  lines; see [PRECISION-GATE.md](PRECISION-GATE.md).

## Who reads what

| Reader | v1 / v2 | v3 flat | v3 per-tensor | PQ | budgets | fp16 raw |
|---|---|---|---|---|---|---|
| `nq2-run` | yes | yes | yes | refuses | refuses | refuses |
| `nq2-infer`, `nq2-infer-lab` | yes | yes | yes | yes | yes | yes |
| `nq2-infer-triforce` | yes | yes | yes | refuses at open | yes | yes |
| accel lanes | yes | yes | synthetic layouts only | falls back to CPU | | |

A reader that cannot expand a pack kind refuses at open with a named reason
rather than misreading it. (Reader support is as verified in the 2026-11-25
reader sweep and the 2026-11-26 accel notes.)

## Verification and reproducibility

- `nq2-run PREFIX.nq2 PREFIX.nq2book` streams the pack through a rolling window
  and prints sizes, peak resident memory, expansion throughput and a checksum of
  the full expansion; `peek N` expands one block on demand.
- `nq2-errprobe [--all] pack book shards...` expands tensor by tensor against the
  raw shards and reports each tensor's share of the total squared error. The
  Sigma baseline: the replaid embedding matrix alone holds 39.80 percent of the
  squared error; see `results/errprobe-baseline-2026-11-23.txt`.
- `BASELINE-2026-11-23.md` freezes the Sigma pack's hashes: expansion checksum
  `7bd9557531df1ab7`, total relative L2 0.6922, index entropy H0 6.4189 bits and
  H1 5.7925, frames 80.0 percent of the stream. Any change to the quantizer, the
  formats or the probes is judged against it per tensor, not by the aggregate.
- Hash doctrine: a hash that moves in development is fine; an unstable hash is
  the failure.

## The research tools around the converter

These share the converter's vector-quantization core and measure one idea each;
none writes a pack. Their findings are in
[EXPERIMENTS-2026-11-24.md](../EXPERIMENTS-2026-11-24.md) and `results/`.

| Tool | Question it answers |
|---|---|
| `nq2-tensorstat` | block outlier dominance, range and kurtosis: why some tensors collapse under min-max |
| `nq2-entries` | flat VQ past 256 entries (512, 1024), with ideal and 2-byte index rates |
| `nq2-hadamard` | does a randomized Hadamard transform inside the block improve fidelity? |
| `nq2-pq` | product quantization fidelity and complete rate per tensor |
| `nq2-additive` | joint additive codebooks against sequential residual quantization |
| `nq2-framed` | what fp16 frame rounding costs against an fp32 scale |
| `nq2-fused` | can a matvec eat the packed form directly (the ancestor of DTC)? |
| `nq2-correct` | does a per-entry correction survive converged k-means? |
| `nq2-rpq` | residual product quantization at equal complete rate |
| `nq2-frameprobe` | frame byte share, value distributions and residual entropy |
| `nq2-audit` | two-part-code size accounting and index entropy |
| `nq2-rsqrtbench` | reciprocal square root methods (exact, AVX2, lookup tables) against a scalar reference |

The three freestanding Linux examples in `examples/` (`vq-0.4.c`, `vq-0.2.c`,
`vq-0.02.c`) are the packing scheme as first demonstrated: 0.4, 0.2 and 0.02 bits
per weight, with the last one (204,800 weights in 64 quad words) a first reading
lesson. The archived [README](archive/README-2026-11-25.md) walks through them.