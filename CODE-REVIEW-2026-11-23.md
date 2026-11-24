# NanoQuatTwo Code Review, 2026-11-23

Reviewer and implementer: Elfie (E.L.F.I.E.)
Direction: JD (requested categories: conversion errors, memory errors,
bad practice, bad type casting)
Scope: tools/nq2-quantize.c, tools/nq2-run.c, tools/nq2-gui.py

## Headline

One critical bug was found and fixed earlier the same evening (the
dtype misread, item 1). This review found one more serious bug (the
Windows seek width, item 4), plus a handful of minor and robustness
issues. Every fix below is applied to the sources and verified:
fixture outputs are bit-identical before and after the fixes
(checksums d5b7e3fc21ff6cb3 for format v1, 2ef1c2feb56f3347 for v2),
both tools build warning-free with -Wall -Wextra on Linux and under
MinGW for Windows, and the fixed runner reproduces earlier stream
checksums exactly.

## Conversion errors

1. CRITICAL, fixed. The safetensors header parser measured the
   dtype label with a length guard of 4; the quote-to-quote
   distance for "BF16" is 5, so the label was discarded and every
   tensor silently defaulted to float32. Half the weights were
   read, as the wrong type. All fidelity numbers produced before
   the fix were void and were re-run. The parser now matches
   BF16/F16/F32 explicitly and warns loudly on anything else.
   Lesson recorded: verify a parsed weight count against ground
   truth before trusting any downstream number.
2. Minor, fixed. float_to_half, subnormal branch: the sticky-bit
   mask included the round bit itself, so exact ties always
   rounded up instead of to even. The mask now covers only the
   bits below the round bit. Effect is confined to subnormal
   frames (magnitudes under about 6.1e-5); no current output
   changed.
3. Checked clean. bf16 to fp32 is a pure widening. fp16 to fp32 is
   implemented identically in both tools and matched an
   independent Python (numpy float16) reconstruction exactly for
   both format versions. The fp32 to fp16 normal-range rounding
   is round-to-nearest-even, verified by hand and by the same
   cross-checks.

## Memory errors and bounds

4. SERIOUS, fixed. tensor_read used fseek with a long offset.
   Long is 32 bits on Windows, and the shards are 5.3 GB: on
   Windows every read past 2 GB would have landed at the wrong
   offset or failed. Both tools now use a seek_to helper
   (_fseeki64 on Windows, fseeko elsewhere) and 64-bit positions
   throughout.
5. Fixed. The header buffer malloc in shard_open was unchecked;
   a corrupt header length would have passed NULL to fread.
   Checked now.
6. Fixed. The large allocations are now all checked: the global
   sample (up to about 512 MB at block 4096), the per-tensor
   books, and the pack arrays in both modes. A failed allocation
   reports and exits instead of segfaulting inside memcpy.
7. Fixed. The runner's v2 segment advance could walk off the end
   of the segment table on a corrupt file (segment block counts
   summing to less than the block total). It is bounded at the
   last segment, and the following decrement is guarded against
   underflow.
8. Checked clean. Reservoir sampling never writes past its cap;
   slab walking clamps at tensor ends; per-tensor pack cursors
   are precomputed from the same counts the walks produce; book
   reads verify their element counts; window buffers are checked.

## Type casting and widths

9. Fixed. The runner's seek math mixed long and uint64_t
   (3L * (long)n_blocks). Correct at today's sizes, wrong in
   principle past 2^31. All stream positions are uint64_t now.
10. Checked clean. The quantizer's size products are computed in
    size_t/uint64_t where they can grow; peek's argument is
    range-checked against the block count before use.

## Practice and robustness

11. Fixed. --books treated any unrecognized value as "global"
    silently. It now warns and names the value it was given.
12. Fixed (GUI). The worker thread called Tk widget methods
    directly; Tkinter is not thread-safe. All output is now
    marshalled back to the UI thread through after(0).
13. Noted, accepted. The header walk assumes every top-level
    safetensors value is an object, which the format guarantees
    today. Tensor names are truncated at 250 characters inside
    the tool; names are used only for filtering and reporting,
    never for packing decisions.
14. Noted, accepted. Short-lived CLI allocations (shards, scope,
    books) are intentionally not freed before exit; the runner
    frees its buffers. Reservoir sampling uses modulo reduction;
    the bias is irrelevant for codebook sampling.
15. Noted, format limitation. Tensors smaller than one block
    contribute zero blocks in v2, and block tails are dropped in
    both versions; the loss is visible in the packed weight
    totals. Mixed block sizes are future work, already on the
    road map in WHAT-WE-CAN-DO.md.

## State after this review

The sources carry all fixes. The binaries currently running the
background quantization chain are the pre-review builds; on this
Linux bench they are numerically equivalent for the paths the
chain exercises (the seek bug bites only 32-bit-long platforms,
and no frame value in this model is subnormal). Shipped binaries
will be rebuilt from the reviewed sources at packaging time.
