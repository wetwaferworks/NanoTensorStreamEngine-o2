# Repo update package, 2026-11-24

This is a repository update package, not a release. The release
identity is unchanged: version 0.1.0, rolling build counter 0001
(Build 0001), still unpublished at JD's word.

## What changed since the 2026-11-24 initial sync

- PIPELINE-2026-11-24.md (new): the Triforce streaming pipeline,
  the determinism gate catching a real race, the bisection, the
  root cause, the fix, and the verified re-anchors, written for
  researchers landing on the repo cold.
- README.md: refreshed. Adds the Triforce chapter, the
  behavioral comparator, the one-click build, and the prebuilt
  binaries note.
- EXPERIMENTS-2026-11-24.md: engine test state updated, pointer
  to the pipeline document.
- WHAT-WE-CAN-DO.md: a dated status note at the top; the 2026-11-23
  plan text is kept as written.
- tools/: the experimental line joins the tree. nq2-infer-lab.c
  (instrumented engine with stall forensics, per-evaluation
  logits dumps, and single-tensor restore), nq2-infer-triforce.c
  (the split fetch/expand pipeline), nq2-infer-triforce-dbg.c
  (its serial-mode diagnostic twin), nq2-logitcmp.c (behavioral
  comparator: top-1/top-5, cosine, relative logit L2, KL
  divergence), nq2-correct.c (per-entry correction diagnostic),
  nq2-rpq.c (residual product quantization probe). The pipeline
  sources also carry a read-integrity hardening: the packed
  stream's size and per-tensor block counts are validated at
  open, so a corrupt or truncated pack is refused instead of
  read.
- results/: the receipts for the pipeline story, including the
  failed run, the three bisections, and both fixed re-anchors.
- Build: build-all.sh and build-all.bat build all 21 tools in
  one click on Linux and Windows. build-experiments covers the
  fifteen experimental tools.

## Binaries

dist/linux64 and dist/win64 hold all 21 tools, built fresh from
the sources in this package (Linux: system gcc; Windows: MinGW
POSIX-thread gcc, statically linked). Verified before packing:

- The Linux nq2-run reproduces the frozen pack's recorded
  checksum exactly (FNV-1a 7bd9557531df1ab7, peak resident
  4,476,940 bytes).
- The Linux nq2-audit reads the shipped pack as a complete
  description of 0.15684 bits per weight with this build as the
  decoder (the baseline document's 0.15691 figure used the
  earlier release build as decoder; the artifact itself is
  unchanged).
- All 21 Windows exes import only KERNEL32.dll and msvcrt.dll
  (objdump-verified), so they run without MinGW runtime DLLs.
- MANIFEST-SHA256.txt hashes every binary and source file.

One honest caveat: the Triforce fix itself is anchor-verified
(both 1-step anchors reproduce exactly, receipts in results/).
The read-integrity hardening that rides along in the same
sources has its own anchor exam and negative-test suite
scripted; it runs on the bench next, and nothing in the docs
claims otherwise.

## Publishing

Text files (docs, sources, scripts, receipts) push to
wetwaferworks/nanoquantengine-o2 through the GitHub connector.
The dist/ binaries travel in this zip and attach to the GitHub
release, because the connector's file push is text-only.
Nothing in this package has been pushed yet; it publishes at
JD's word.

## Second update, later on 2026-11-24

Written and pushed the same day, at JD's word. The release
identity is still unchanged: version 0.1.0, rolling build
counter 0001 (Build 0001), unpublished.

- MEGA-TENSOR.md (new): JD named the streaming system under the
  engine **Mega Tensor**. The document covers the four loops,
  the no-walls concurrency rule (fetch, expand, prefetch run
  concurrently; the only boundary is serialized consumption,
  plus the pass-restart epoch), and the measured rates.
- DICTIONARY-COMPUTE.md (new): the discovery that a packed model
  never has to be unpacked to be used, with the math, the
  measured speed and error figures, the adversarial campaign,
  and the honest limits, including the break-even and the
  not-bit-identical disclosure. Sources join the tree:
  tools/nq2-dc.c and tools/nq2-dc-adv.c, with receipts
  results/dc-probe-2026-11-24.txt, results/dc-adv-2026-11-24.txt,
  and results/dc-sweep-2026-11-24.txt.
- LITTLE-NEMO.md (new) and the littlenemo/ tree: the pure C
  engine for Nemotron-Nano-9B-v2. Kernels verified against
  hand-computed fixtures; the forward pass verified on the real
  checkpoint (7/7, bit-exact across reset); the protocol engine
  and GUI included. The document states plainly what is not
  done: the full 56-layer run waits on the checkpoint download
  still in flight.
- README.md: refreshed with the Mega Tensor naming, the
  whole-model checksum identity (an independent probe's
  expansion of all 3.83 billion weights hashes to the runner's
  recorded 7bd9557531df1ab7), the completed hardening pass with
  its anchor re-verification, dictionary compute, and Little
  Nemo.
- The morning note above said the read-integrity hardening's own
  exam was still to run. It has now run: hardened anchors
  reproduce exactly (raw, packed, Triforce packed), hostile
  fixtures are refused, and the sanitizer battery is clean.

## Third update, later on 2026-11-24

A second outside review (ChatGPT (Gizmo), reading the repo cold)
found two things in Little Nemo; both are fixed and pushed the
same day:

- Whole-geometry validation before inference begins:
  nemo_validate_model checks every tensor the steps will fetch
  for presence, shape, and dtype. The engine prints READY only
  for a checkpoint that passes; a partial checkpoint is refused
  up front with the offending tensor named.
- State commitment coherence: every step preflights its tensors
  before the first layer's state moves, so a missing tensor
  commits nothing; any mid-commit failure invalidates the state
  until reset; the engine resets after a failed turn and says
  so on the protocol. New suite: littlenemo/tests/
  test_coherence.c, including bit-identical proof that a
  refused step changed nothing. LITTLE-NEMO.md and
  littlenemo/PROTOCOL.md carry the details.

## Fourth update, 2026-11-24: dual path, Windows build, INI

- Little Nemo goes dual-path: each large projection is split
  across worker lanes and the next layer is prepared while the
  current one computes (preparation concurrent, consumption
  serialized). The engine queries the machine's core count at
  startup; --mode minimum|medium|max|ultra picks the lane count
  and --threads N overrides it. The parallel path is proven
  bit-identical to the serial path on real weights
  (littlenemo/tests/test_par.c), and the forward and coherence
  suites still pass at 7/7 (RMS 0.520647) and 25/25.
- The Windows build is fixed and verified: engine.c no longer
  assumes POSIX (poll, getline, clock_gettime are guarded with
  native Win32 equivalents), the build compiles under MinGW, and
  the resulting .exe runs and prints READY on the full
  checkpoint. The build scripts now build everything (all 23
  tools plus Little Nemo) and stop at the first failure instead
  of reporting success.
- The chat GUI reads littlenemo.ini for the engine, checkpoint,
  and tokenizer paths plus the compute mode, so it launches by
  click with no command-line arguments; arguments still override
  the file.
