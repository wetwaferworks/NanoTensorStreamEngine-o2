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
