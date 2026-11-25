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

## Fifth update, 2026-11-24: SIMD (AVX2), opt-in

- The projection kernel gains an AVX2 twin (src/nn.c, the one
  translation unit the build scripts compile with -mavx2 on
  x86-64). The rest of the engine stays at the baseline
  instruction set, the kernel is reached only through a runtime
  CPUID check, and it runs only when asked: --simd on the engine,
  or simd = true under [engine] in littlenemo.ini. The READY
  line reports the active state (simd=0|1). On hardware without
  AVX2 the engine stays on the scalar path whatever the setting
  says.
- The SIMD kernel sums the same products in a fixed blocked
  order (groups of four doubles), so it is deterministic; the
  scalar path remains the golden reference and the default. A
  new suite (littlenemo/tests/test_simd.c) holds it to both
  standards on real weights: deterministic across reset, and
  faithful to scalar.
- Correction (2026-11-24, Archie's code review): the first
  version of this note reported "drift exactly zero, 6/6
  checks," but that measurement was scalar against scalar...
  the suite never called the AVX2 kernel, and the engine's
  entry point never dispatched to it, so the kernel was
  unreachable in single-thread mode. Both are fixed: the
  entry point now dispatches through the auto selector, and
  the suite calls the kernel directly on synthetic data,
  checks the dispatch itself, and compares a multi-lane SIMD
  forward against the 1-lane run. Corrected measurement,
  kernel against scalar: drift exactly zero (max abs 0.0 on
  the synthetic kernel test and over the hidden state, three
  tokens through 15 layers), 12/12 checks, and the multi-lane
  run is bit-identical to the 1-lane run. The forward suite
  still reproduces the golden hidden RMS 0.520647 exactly, so
  the scalar path did not move under the new build.
- The compiler does not do this on its own: a portable build
  targets the baseline set, and silent floating-point
  reassociation is forbidden, so the vector kernel is written
  by hand and gated, never assumed.

## Sixth update, 2026-11-24 (night): a third review, fixed

A third outside review (Archie (Argon), the Google Gemini
Argon model) read the pushed tree cold and filed five P0
faults, with P1 and P2 items behind them. All five P0s and
the P1/P2 lists are fixed in this push; the full fix log is
kept in the project's bench records.

- SIMD: the Fifth update's correction above is the headline.
  The entry point dispatches through the auto selector, the
  suite measures the kernel itself, 12/12 checks, drift
  exactly zero.
- 64-bit offsets: the _FILE_OFFSET_BITS 64 define now stands
  in the littlenemo loader, forward, and tokenizer sources
  and in the audit and dictionary compute tools, and the flag
  is in all four build scripts.
- The four inference engines run on Windows: the directory
  scan is opendir/readdir instead of a popen shell listing,
  the early exit is gone, tensor spans are validated at open,
  and the bf16 NaN guard from the attention kernel now stands
  in every engine's converter.
- Triforce: a demand-generation protocol fixes a cold-miss
  race, resident ordinals are no longer re-fetched at pass
  restart, and the mode-2 staging buffer is complete... raw
  tensors are staged at fetch time on private per-shard
  handles. No on-disk format changed; the frozen baseline
  pack decodes bit-identically under the new build (streaming
  accumulator 1394.1117661067326 in all four configurations,
  audit output byte-identical before and after).
- Baselines after the whole pass: forward golden RMS
  0.520647 exactly, dictionary compute adversarial 23/23,
  every littlenemo suite green, MinGW cross-compile 18/18 on
  the touched Windows-relevant files.
- One deferral stands: the Triforce fetch/unpack split on
  Windows, a pipe rewrite that cannot be run-verified on this
  bench. The fused Windows worker runs as before.
- Still pending, said plainly: the third battery question
  (Ohio) has not yet run on the post-fix build.
