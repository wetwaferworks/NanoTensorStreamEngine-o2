# accel/ progress note (resume point)

Updated 2026-11-26 12:25 ET. Worktree <scratch>\ntse-o2-accel-2026-11-26,
branch scratch/accel-2026-11-26 off ef39281. This note predates the merge: the work is now on `main`.
`git status`: only `?? accel/`.
(sync: tar on box -> CopyFromBox to <xfer-dir>
-> `tar -xf` in the worktree; the scratch drive is outside the CopyFromBox root).
Run everything at BelowNormal; leave CPU for the GUI (pythonw) and
the baseline runs.

## DONE (do not redo)
- Three lanes (cuda / opencl-gpu / opencl-igpu), all suites pass
  (kernels 36, stream 42, real 49, linear 47, nemo 10, plan 17 per
  lane; guard 36). Logs: accel\logs.
- Gated bench 10:44–10:48 ET: accel\logs\bench\RECEIPTS.txt; README
  Benchmarks section filled (incl. negatives).
- Lane 2 (Gizmo): prepared stream plan, accel\src\accel_plan.inc.h
  (API ntse_accel_plan_* + ntse_accel_probe_bandwidth), test
  accel\tests\test_accel_plan.c (17/17 x 3 lanes), bench section p
  plan rows. Receipts accel\logs\bench-lane2 (11:18 ET):
  CUDA pinned 2 slots 76.40 tok/s (old stream 7.97, resident 80.67),
  H2D 71,026,688 B/token (was 168,949,504), PCIe 12.99 GB/s pinned,
  bit-exact, 0 violations. OpenCL-NVIDIA 40.6, iGPU 14.9.

- README Lane 2 section written.
- Lane 1: NTSE_OP_DTC_LAZY (k_mark + k_build_lazy, CUDA and OpenCL,
  plan support). Bit-exact on all lanes; suites kernels 43, real 56,
  plan 17, guard 37 per lane. Receipts accel\logs\bench-lane1 and
  bench-lane1-warm (warm-up pass added to section p). README Lane 1
  section written: lazy loses 7-21 % (static index => same work as
  active + mark pass).

- Lane 4 STARTED 11:33 ET: accel\lane4\run-lane4.ps1 (detached,
  BelowNormal, 4 threads, resumable: steps with outputs are skipped;
  rerun the same command to resume). Tools built from this worktree's
  tools\*.c into accel\lane4\bin. Raw model and packs A/B are read
  from <repo>\results-2026-11-26 (read-only).
  Configs: raw x2, A global b256 v2 x2, B pt b256 v3 e16, C pt PQ8
  RMS e16 (new pack), D pt PQ4 RMS e16 (new pack). Recipe = Linux
  battery HOUSE: --ids 1 --tokens 32 --steps 1 --guidance 1
  --score-temp 1 --seed 42. Receipts accel\lane4\logs\RECEIPTS.txt.
- Archie: her R5 review (zm\Archie-R5-Verification-2026-11-26.md in
  the 11-25 results) was already verified and fixed in the zero-mutex
  worktree by another agent. It covers nq2-infer, not accel. Not redone.

- Lane 4 DONE for the first scope (12:15 ET): packs A,B,C(PQ8),D(PQ4)
  + 15 --restore-tensors ablations, all vs raw. README Lane 4 section
  written. No config matches raw top-1; D 9/32, layers+head raw 25-30/32.
  Next probes listed there (residual PQ, multi-step, tokenizer eval).

## NEXT (historic list, superseded by the above)
1. Lane 4 fidelity: tooling exists (nq2-infer-lab --logits-dump,
   nq2-logitcmp: top-1/top-5, cosine, rel logit L2, KL). Sigma has no
   tokenizer (ppl/reasoning N/A there); Little Nemo has tokenizer.c.
   Need packs at higher bit rates; scope before running (CPU heavy).
