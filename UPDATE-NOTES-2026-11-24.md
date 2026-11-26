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
wetwaferworks/NanoTensorStreamEngine-o2 through the GitHub connector.
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

## Seventh update, 2026-11-25: two more reviewers, twenty-seven more findings

Archie reviewed the Sixth update's commit and verified all fifteen
of her Round 1 findings fixed, with zero P0 or P1 issues remaining;
her Round 2 report adds five P2 polish items, all fixed in this
pass (the SIMD synthetic test now runs without the checkpoint
present, the Windows builds are warning-clean, and three small
guards were closed in tensorstat, the tokenizer, and the --serve
parser).

Codex (Astra 6.2) then reviewed the same commit independently and
filed ten high-priority and seventeen medium findings, several
demonstrated experimentally. Elfie's triage sustained all
twenty-seven. Twenty-five are fixed here, including the three that
mattered most: the quantizer now packs shards in the canonical
order the readers use and every engine binds a pack to its
directory before trusting it, so a pack can no longer silently
describe the wrong tensors; Little Nemo's tokenizer now implements
the checkpoint's own pre-tokenizer split and ignore_merges lookup
(verified token-for-token against the reference tokenizer on a
seventy-case battery; the two-space prompt that used to encode as
[1256, 1097] now encodes as [1032, 1261]); and zero blocks pack at
scale 0 instead of acquiring weights from a shared book. The
stream verifier now refuses a truncated pack instead of checksumming
its prefix, the quantizer refuses to report success when a write
fails, and the AVX2 kernel lives in its own translation unit so
the scalar fallback contains no AVX instructions. One finding is
held for JD's re-decision rather than fixed (the streamtest's
documented zero-fill on short reads), and one follow-through is
flagged rather than done: the frozen Sigma pack was written before
the zero-block fix, and whether to audit and re-strike it is JD's
call.

The digest printed by the tools is now labeled honestly: it is a
house digest using a shortened FNV-1a offset basis, shared
consistently by four tools and by every published baseline. It has
not been re-based; nq2-run prints the standard-basis FNV-1a
alongside it (on the frozen pack: house 7bd9557531df1ab7,
standard 97bc43e8aef7ea05).

Prefetch is now configuration, not constants. The Triforce pipe
takes its slot count, fetch-ahead, and block prefetch distance
from the command line, the environment (NQ2_PIPE_SLOTS,
NQ2_FETCH_AHEAD, NQ2_PREFETCH_DIST), or defaults derived from the
device, and prints each value with its source. Depth was proven a
performance knob only: the Sigma token and the streamtest
accumulator are identical at every depth tested. The first slice
of the Tensor Handshake (JD's name for the counter-synchronized
prefetch design) has landed: the pipe publishes the consumer's
committed layer counter and aims fetch-ahead from it. Per-thread
tick counters and parallel expansion lanes are designed and
staged, deliberately not landed unproven.

Baselines, all unmoved: forward hidden RMS 0.520647 (7/7);
dc-adv 23/23 with its anchors unchanged; streamtest accumulator
1394.1117661067326 at unpackers 0/1/2/3; Sigma raw token 75768 in
all three engines; loader fingerprint unchanged.

## Eighth update, 2026-11-25: the combined pass (decisions, format v3, Archie Round 4, Boardy)

JD delegated the held engine decisions to Elfie on 2026-11-25;
the calls below are hers, recorded as such, and JD confirmed
the priority the same morning: M-P8 and H3b first. This update
grows as the pass lands; nothing in it ships until the whole
pass passes its gate.

- M-P8 decided and fixed (Codex Astra M-P8, re-raised as
  ChatGPT Sol M8): nq2-streamtest no longer zero-fills short
  reads. A pack whose bytes fall short of what its header
  promises fails up front, by name and shortfall, and any
  short read mid-stream stops the pipeline; either way the
  exit is nonzero and no accumulator is published. The old
  policy let a pack cut by 40 trailing bytes print a
  stable-looking accumulator and exit 0. A regression test,
  tools/test-streamtest-truncation.sh, reproduces that cut
  against the frozen baseline pack and proves the failure
  path on the sequential and pipelined configurations. The
  complete pack's accumulator is unchanged:
  1394.1117661067326 at unpackers 0/1/2/3.
- Archie Round 4 P1-1 and P1-2 fixed in the Little Nemo chat
  GUI: on Windows, a .py engine path is run through the
  current interpreter instead of failing with WinError 193;
  and applying new settings now makes them the session's
  current settings, so Restart engine no longer reverts to
  the startup configuration. Both GUI suites pass.
- Two smaller held items decided the same day. The
  tensorstat statistics lines no longer print negative NaNs
  on zero-variance tensors: an all-zero tensor now reports
  its true zeros with the kurtosis and block-range ratios
  labeled n/a, instead of the 0/0 artifacts. And
  tools/nq2-infer-triforce-dbg.c stays exactly as frozen:
  Sol's H2 remedies for the snapshot are declined, by
  decision, because the file is forensic evidence of the
  bisected Triforce failure, defects included; its header
  now says so in terms meant to stop the finding being
  re-raised, and it is not a live engine.
- A warning we owe the record (Boardy, 2026-11-25):
  measurements taken before the Little Nemo tokenizer fixes
  ran on the pre-fix token stream. The tokenizer was
  corrected twice (Codex H2, push 7; Sol M1, push 9).
  Questionnaire Q01 and Q02 predate the fix and are annotated
  as such in their results file; Q03 ran on the post-fix
  build. All pinned baselines feed raw token IDs and are
  unaffected. The same warning now stands in the README's
  honest notes.
- Format v3 and H3b (Codex Astra H3): a pack and its book
  now carry a shared generation identity, computed from the
  pack's own contents (geometry, tensor names, book bytes,
  and the packed planes), so any two runs of the writer over
  the same inputs produce the same identity and a stream
  packed with one book can never silently pair with another.
  Every engine checks the pair at open and refuses a mismatch
  in plain words, naming both generations. A per-tensor
  digest manifest travels with the pack (one line per tensor:
  its stream segment digest and its book digest), and the
  engines verify it at open; a tampered stream is refused
  before a single weight is served. Legacy format v1 and v2
  packs still load exactly as before, positionally, with a
  printed notice that they carry no generation identity.
  Format v3 also carries the scoped additions JD authorized:
  the PQ pack writer (per-tensor product books at d4 and d8,
  RMS sliced), per-tensor bit budgets that re-pack the
  highest-error tensors at finer blocks, and an embedding
  fp16-raw option (default off). The Triforce refuses PQ
  packs at open for now, in plain words; the lab and infer
  engines run them. Measured results for the new writers are
  appended to results/behavioral-gate-2026-11-25.txt as they
  land; the writer's own geometry probes are already on
  record (PQ d4 K256 rel L2 0.3148, d8 0.5759 on the
  embedding tensor, against 0.6922 for the shipped global
  book at 0.15691 bpw).
- Archie Round 4 P2-1 and P2-2 fixed. P2-1: a Triforce
  pipeline restart now counts a row as resident only when
  its slot actually holds it, so the restart pre-count can
  no longer overstate readiness (cosmetic in effect, as the
  triage found, and now exact). P2-2: the Little Nemo
  tokenizer's is_special now honors an explicit
  "special": false in tokenizer.json instead of treating
  every added token as special. Tokenizer suites stand at
  27/27 and 5/5.
- Boardy, addressed in full (2026-11-25 review, triaged
  blind). Three items sustained, answered by fix or pin:
  the test_simd AVX2 question is closed by proof, the suite
  now runs the AVX2 kernel on this bench with the kernel
  entry counter moving only when SIMD dispatch is on and
  the synthetic drift at exactly zero (7/7); the forward
  depth point is answered by a full-depth logit capture
  (first eight logits plus a hash of the whole 131072-wide
  vector, with a determinism re-run in a second process),
  pinned in results/nemo-full-depth-logits-2026-11-25.txt
  (hash ea501f74b959b77d, reproduced exactly in a second
  process); and the
  tokenizer blast-radius warning stands above. Two items
  were not sustained, answered by the triage verdicts
  themselves: the claimed second instance of the AVX2
  pattern one section up does not exist in the source, and
  the battery-scope claim does not touch any published
  number, since every pinned baseline feeds raw token IDs.
  One item stays open, stated plainly: the Hugging Face
  cross-check of the forward pass is workstation work (the
  bench cannot hold the full checkpoint in that framework)
  and its timing is JD's open decision. Nothing was dropped
  silently and nothing was fake-fixed.

- Format v3 measured results (Sigma 3B full scope; detail
  in results/behavioral-gate-2026-11-25.txt). Behavioral
  points, eval 0 against the raw reference, top-1 agreement
  first: the shipped pack at 0.157 bpw reads 0/32, mean
  cosine -0.794652, mean KL 5.425064. The PQ writer (RMS
  frame, as designed) moves the forward a long way: PQ d8
  at 1.2579 bpw complete (weight rel L2 0.3791) reads 0/32,
  cosine 0.893236, KL 1.817920; PQ d4 at 2.2579 bpw (weight
  rel L2 0.2118) reads 0/32, cosine 0.944810, KL 1.159235.
  The direction is monotone in rate and the forward now
  tracks the raw model's direction closely; top-1
  agreement is still zero at every measured rate, so the
  fidelity criterion is not yet met, and the record says
  so. A v3 global pack in the RMS frame reads cosine
  0.542399 at the shipped rate, better in forward terms
  than the min-max baseline pack despite a worse weight
  error (0.9719 against 0.6922); the two frames do not
  rank the same way in weight space and forward space.
  The v3 global pack at the default min-max frame reproduces the shipped baseline exactly: weight rel L2 0.6922 at 0.1568 bpw complete, the frozen pack's own figures, now inside format v3 with generation identity and the digest manifest. Per-tensor bit budgets under the RMS
  frame move the global RMS figure from 0.9719 to 0.6601
  at 0.4445 bpw; the mechanism works as built and the
  gain is real under that frame, at 2.8 times the shipped
  rate. The embedding fp16 option measured bit-exact on
  this checkpoint (bf16 mantissas fit inside fp16 for
  in-range values): round-trip rel L2 0.0000, at 16 bits
  per stored weight, default off. Reader verification
  closed on a two-layer extraction of the same model:
  every engine runs every pack kind it accepts, lab,
  triforce, and infer agree token-for-token on shared
  packs, and the legacy packs reproduce the v3 packs'
  tokens exactly.

## Eighth update, continued (2026-11-25, later): DTC fixes, the Precision Gate, format phase verification

The Eighth update above was written as the pass landed. This
continuation records what landed after it, in the same pass.
The release identity is still unchanged: version 0.1.0,
rolling build counter 0001 (Build 0001), unpublished.

- Dense Tensor Compute review and fix pass. An Elfie code
  review of the dictionary compute tools (review plus
  testing, at JD's direction) filed no P0, four P1, and six
  P2 findings. The four P1s are fixed:
  (1) The build/apply logic now lives in exactly one place,
  tools/nq2-dc-kernel.h (nqdc_build, nqdc_apply, nqdc_entry,
  nqdc_seg_sum), parameterized on book pointer, entry count,
  and block size, with the accumulation orders stated as a
  determinism contract in the header. The probe calls it in
  DC1, DC2, and DC3; the adversarial campaign's private copy
  of the algorithm is deleted, so the campaign now certifies
  the shipped code (before the fix, its 23 HOLDs certified
  the copy, the same failure shape as the SIMD suite Archie
  caught, one level down).
  (2) Campaign checks 9 and 10 now compare the lazy output
  they name against the fp64 oracle, and that comparison
  decides the verdict. The printed values are unchanged
  (3.00e-07 max reuse, 3.23e-07 min reuse), now earned by the
  lazy path itself.
  (3) Malformed inputs refuse: shard spans are validated
  against the dims-derived byte count and the data section
  (overflow-guarded), and non-finite book or frame values
  are named rejections with exit 1 instead of NaN metrics
  under exit 0. Regression: tools/test-dc-hardening.sh,
  ALL PASS.
  (4) Doc citations corrected: the DICTIONARY-COMPUTE.md
  speedup sentence now reads about 7.5 to 57.6 times faster
  depending on the pairing and the tensor shape (the
  receipts' paired range, k_proj 7.5x to gate_proj 57.6x),
  the break-even M is labeled as the weight matrix's row
  count (the activation batch cancels out), and
  NQDC-FINDINGS-2026-11-24.md now cites materialize+GEMV at
  23.821 ms paired best, replacing a stale 20.8 ms.
  Gate evidence: campaign 23 HOLD / 0 BREAK, output
  byte-identical to the pinned receipt apart from the two
  lazy timing values, all four FNV anchors exact
  (839d3f72ad1a3d60, bdae48bdf7a78a40, 35e1adcc254bf68e,
  1cd08a5dff229720); probe and all six sweep configurations
  reproduce their pinned correctness numbers exactly;
  fresh-tree build-all clean, zero warnings. No baseline
  moved.
- DTC is the default compute path (JD's ruling, 2026-11-25).
  Stated plainly: the table path reassociates the sum, so
  its output equals the expanded path to float noise, not
  bit for bit. Making it the default is a deliberate
  re-anchor under the project's hash doctrine, and the
  expanded path stays in the tree as the reference oracle
  and fallback.
- The Precision Gate (plan:
  DTC-PRECISION-GATE-PLAN-2026-11-25.md) is implemented in a
  scratch build of the Triforce engine and tested there; it
  has not been transplanted into the live tree. Postures:
  evidence (the shipped default), auto, fast, think-harder,
  selected by --posture or NQ2_POSTURE. Knobs:
  --gate-lookahead 1 to 4 (default 2), --gate-resync,
  --gate-eps and --gate-budget (explicit overrides, printed
  as OVERRIDE when used), --gate-log. Calibration is carried
  per pack: gate lines in the format v3 manifest after the
  tensor lines (gate-gen must equal the pack generation),
  or a legacy sidecar file (<stream>.nq2gate) for v1/v2
  packs, keyed by a content FNV over geometry and stored
  book bytes. Scratch test record: Evidence at the anchor
  configuration reproduced token 122388 packed and 75768
  raw, ticks in agreement, gate correctly inert on the raw
  run; Fast ran 130 of 130 layer consumptions through DTC
  to token 122388; Auto produced token 122388 with 1
  escalation in 2 evaluations. One open issue found in
  scratch: an intermittent phase-boundary fault under Auto
  at --threads 1 (1 of 2 attempts; the identical rerun
  completed with the same token and gate summary, and the
  unmodified base engine ran the same setup clean). Root
  cause undetermined; it is carried as an open issue for the
  transplant step, which will instrument the boundary before
  the live-tree gate.
- Calibration warning, stated plainly. The first gate
  calibration (frozen pack, legacy sidecar; gate-gen
  987cac0cda1f2f01, tie epsilon 3.5076e+01, drift budget
  1.2770e+00) was derived from a scratch DTC vehicle whose
  measured divergence from the dense path sits at the 1e-1
  to 1e0 scale per layer (peak 8.479 at layer 5), not the
  1e-6 association-noise scale the plan assumed, and the two
  vehicles' argmax agreed only 8 of 64 times. The gate
  engine's own DTC path nevertheless reproduced the dense
  anchor token exactly, so the divergence may belong to the
  measuring vehicle rather than to DTC. The calibration
  will be re-derived against the gate engine, or the vehicle
  audited, before any decision to flip the default posture
  to Auto. As calibrated today, Auto escalates aggressively
  and sits much closer to Evidence than to Fast in wall
  time.
- Format phase verification landings. H3b proofs on the
  mini model (2-layer Sigma extraction, 27 tensors): the lab
  opens the v3 global pack with "pack identity: generation
  9e0175bc449dbac2; manifest verified for 27 tensors"; a
  mismatched book is refused in plain words naming both
  generations (9e0175bc449dbac2 vs 08e74e772046d9f6); a
  one-byte tamper in the first tensor's index plane is
  refused at open by the manifest segment digest. The frozen
  v1 pack prints the legacy notice and runs under the new
  readers at the pinned anchor exactly (Triforce token
  122388, ticks fetch 181, expand 182 of 182, consume 181,
  with the Archie P2-1 fix in). Mini per-tensor pack: rel
  L2 0.8093 at 0.2107 bpw complete, against 0.9662 at 0.1583
  for the global book on the same embed/head-heavy scope;
  recorded as a fidelity observation, not a reader defect.
- Reader sweep closed. All seven pack kinds exist on the
  mini scope (v3 global, v3 per-tensor, PQ d8, budgets,
  fp16, legacy v1, legacy v2). The lab runs every pack with
  identity verified; cross-format consistency holds (the
  legacy v1 pack through the new lab generates token 4735,
  identical to the v3 global pack; the legacy v2 pack
  generates 103022, identical to the v3 per-tensor pack; the
  fp16 pack runs, token 37172; the PQ pack runs in the lab,
  token 53190, and is refused by the Triforce in plain
  words; the budgets pack runs in the lab and the Triforce,
  token 83478, ticks exact). Every engine is now verified
  against every pack kind it accepts.
- Full-depth Nemo pin landed:
  results/nemo-full-depth-logits-2026-11-25.txt. Complete
  56-layer checkpoint, prompt "What is the capital of
  France?", first eight logits -3.012787 -0.098331 5.376278
  -2.796762 -2.692724 -3.014087 -3.020434 -3.013171, argmax
  1349 (15.690172), whole-vector FNV-1a ea501f74b959b77d,
  reproduced exactly in a second process. The external
  Hugging Face cross-check remains undone; it is
  workstation work and its timing is JD's open decision.
- Final gate pieces: fresh-tree build-all rc=0 (23 tools
  plus Little Nemo); MinGW cross-builds 18/18 for the nine
  changed or neighboring tools across both thread flavors;
  test_par 7/7 on the real checkpoint; anchors as recorded
  above.

- Full-scope PQ behavioral points, landed (the gate
  pipeline's comparisons, appended to
  results/behavioral-gate-2026-11-25.txt as they landed).
  The full PQ d8 pack (243 tensors, RMS frame, per-tensor
  PQ books) measures weight relative L2 0.3791 at 1.2579
  bpw complete; its behavioral point (eval 0, against the
  raw reference) reads top-1 agreement 0/32, mean cosine
  0.893236, mean relative logit L2 0.483437, mean KL
  1.817920. The PQ d4 pack measures weight relative L2
  0.2118 at 2.2579 bpw complete; its point reads top-1
  0/32, mean cosine 0.944810, mean relative logit L2
  0.330032, mean KL 1.159235. The ladder is monotone in
  rate (0.157 bpw: cosine -0.795; 1.2579 bpw: 0.893;
  2.2579 bpw: 0.945), and top-1 agreement is still zero
  at every measured rate, so by the project's fidelity
  criterion the output is still not the same reasoning;
  the record says so plainly. Per-tensor bit budgets at
  full scope (RMS frame): the same-frame global figure
  moves from 0.9719 to 0.6601 at 0.4445 bpw complete
  (4 tensors re-packed at block 64, 63 at block 128, 176
  at block 256); the mechanism works exactly as built
  and the measured gain is small. The v3 global pack at
  the default min-max frame reproduces the frozen
  baseline exactly: weight relative L2 0.6922 at 0.1568
  bpw complete, behavioral point top-1 0/32, mean cosine
  -0.794652, mean relative logit L2 1.423555, mean KL
  5.425064, matching the frozen pack's recorded point to
  the last printed digit. Format v3 reproduces the
  shipped pack class exactly while adding generation
  identity, the digest manifest, and the new writers.
