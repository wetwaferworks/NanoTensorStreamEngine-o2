# Precision Gate implementation: design note (scratch, pre-gate)

Written 2026-11-25 by the gate crew, before the live-tree gate opened.
Source plan: DTC-PRECISION-GATE-PLAN-2026-11-25.md. Consume design:
scratch/dtc/DESIGN.md (Variant A). Kernel: tools/nq2-dc-kernel.h.
Target file: tools/nq2-infer-triforce.c (POSIX triforce path; the
Windows fused worker keeps expanding, as it already ignores the other
POSIX-only pipe knobs, and says so in its startup note).

## 0. Gate state at note time

The format crew is still running (PQ d8 pack, mini pack chain, reader
sweeps; relaunched after the 14:15 EST service restart). No format
completion entry exists in NIGHT-LOG-2026-11-23.md yet, so the live
tree is closed to this crew. Everything below was designed against the
tree as landed through the DTC fix pass (15:06 EST).

## 1. What the engine actually looks like (facts the design rests on)

- encoder_layer consumes 7 pipe ordinals per layer (Q K V O GATE UP
  DOWN) through pipe_get, then dense matmul (double accumulation,
  pool-partitioned by rows). All pipe tensors are fully packed in the
  packs the triforce accepts (it refuses v3 PQ at open; fp16-raw
  tensors carry nblocks 0 and are never pipe tensors in practice).
- The head (diffusion_head) is NOT a pipe ordinal: head_all streams
  row tiles through src_read_range and dense-matmuls them. The head
  is therefore already on the "expanded" side of the line, always.
- Tokens commit by argmax over the conditional logits at the final
  denoise step of each block (main loop, s == steps). Positions are
  evaluated in batches of B = block (32 in the house config): one
  encoder_forward serves all B positions at once. Gate escalation
  granularity is therefore the denoise evaluation, not the token.
- Passes: encode_clean (ATT_WRITE, commits KV) and denoise forwards
  (ATT_READ conditional, ATT_BLOCK_ONLY unconditional). pipe_get(0)
  carries the Tensor Handshake phase boundary: ticks must agree the
  prior pass drained before the list is rebuilt.
- The Stage in each slot already holds exactly what DTC needs after
  a mode-1 fetch: idx bytes and fr halves (minima then scales) for
  the whole tensor, plus the tensor's book via pk_book_of. Block
  index b in the stage is row-major over the tensor: b = r * nseg + j
  with nseg = cols / pk_block_of(t), which is exactly the indexing
  nqdc_apply expects for idx/sc/mn.

## 2. Variant A consume path (live design, per the brief)

Marks live on the priority list. Pipe gains:

    uint8_t *ord_mark;   /* [max_ord+1] MARK_* */
    enum { MARK_UNDECIDED = 0, MARK_EXPAND = 1, MARK_DTC = 2 };

- MARK_EXPAND is today's behavior, bit for bit: lanes claim, expand
  into buf[slot], pipe_get returns the buffer, dense matmul.
- MARK_DTC: lanes never claim the ordinal (claim predicate gains
  mark == MARK_EXPAND). The fetcher still stages it (fetch is mark
  blind). The consumer computes from the stage.
- MARK_UNDECIDED is never claimable either; the scorer finalizes
  marks ahead of the claim window (section 3), so an undecided
  ordinal inside the window is a design fault, and the consumer's
  demand path treats it as EXPAND rather than waiting forever.

encoder_layer routes each projection through matmul_pipe(ord, x, y,
n, rows, cols):

- EXPAND: w = pipe_get(ord); matmul(x, w, y, n, rows, cols).
- DTC: dtc_consume(ord, x, y, n, rows, cols):
  * Mirror pipe_get's accounting: req_max, demand post, wait until
    the stage holds ord (st_owner[slot] == ord) with the same
    demand_seen discipline; claim as the consumer (ord_claim =
    PIPE_MAX_LANES) so claim exclusivity has no exceptions.
  * Fault fallback: if have[slot] == ord (the boundary drain
    expanded it), run the dense matmul on buf and finish with
    pipe_get-style bookkeeping. A mark never decides correctness,
    only which arithmetic runs; the drain is a fault path and is
    documented as such.
  * Compute: convert the stage's fr halves to float mn/sc arrays
    (per-slot workspace, nb_max floats each, allocated in pipe_init
    next to the staging buffers and counted in g_pipe_staging_bytes),
    then per position i, pool-partitioned by position:
    nqdc_build(book, entries, block, x + i*cols, nseg, T_i, A_i)
    into a per-worker workspace, then nqdc_apply(stage idx, sc, mn,
    T_i, A_i, entries, nseg, rows, y + i*rows). The kernel's pinned
    accumulation orders are untouched; partitioning by position
    cannot change any output bit (each output element is produced
    by exactly one nqdc_apply row loop).
  * Finish: ord_state = ORD_READY, count the preparation through
    pipe_mark_ready with load_bytes_last = staged packed bytes
    (nb * (idxw + 4)) and load_ms_last = the stage's fetch_ms, then
    publish layer_counter = ord like pipe_get does.
  * ord == 0 runs the same phase-boundary + restart block as
    pipe_get(0). The block is factored into pipe_pass_boundary()
    and called by both entry points before anything else.
- DTC eligibility per ordinal, decided once at pipe_init from the
  pack (never mid-pass): packed mode in use, private stream handle
  open (pk_priv), tensor fully packed (nblocks * block == count),
  not fp16-raw, block divides cols. Ineligible ordinals are
  permanently MARK_EXPAND. Raw (unpacked) runs: everything EXPAND,
  the gate is inert and says so in the startup line.

Restart interplay: ord_mark is per-pass state, reset to UNDECIDED in
the same restart block that rebuilds ord_state from have[]. DTC
ordinals never enter have[], so a drained pass counts them through
the preparation tick only, exactly like an expanded ordinal whose
buffer was evicted. n_ready semantics become "preparations
completed this pass", which is what the boundary check actually
needs (fetch tick, preparation tick, consume tick agree).

## 3. The scorer (consumer only, deterministic)

The consumer is the only writer of marks. Finalization points:

- In pipe_pass_boundary(): finalize layers 0 .. lookahead.
- In encoder_layer, before the layer's first matmul_pipe: finalize
  layers up to layer + lookahead that are still UNDECIDED.

The claim window reaches at most req_max + slots - 1; with slots
at most 8 and 7 ordinals per layer, only the next layer's first
ordinal can be claimed before its layer starts, and it is finalized
by then. Marks are immutable once finalized within a pass.

Score for layer l (Auto posture, calibrated pack), in order:
1. Forced sets first: a step escalation or shadow resync in force
   (section 4) finalizes every remaining layer EXPAND. Fast posture
   finalizes DTC (subject to eligibility). Think Harder and
   Evidence finalize EXPAND.
2. Hot layer: profile[l] above hot_factor (4x) the profile median
   finalizes EXPAND. This is the static sensitivity map.
3. Drift projection: drift2 accumulates delta[l]^2 for each layer
   consumed DTC, where delta[l] is the calibrated per-layer
   divergence increment (section 5). Projected final divergence
   with layer l DTC is sqrt(drift2 + delta[l]^2 + sum of delta^2
   over the rest of the lookahead window). If the projection
   exceeds the calibrated budget, layer l and the rest of the
   window finalize EXPAND (the plan's crossing rule).
4. Otherwise DTC.
When a layer is consumed all-EXPAND, drift2 resets to 0 at the
layer's end (the state is the exact quantized state again). Mixed
layers cannot occur: marks are per layer, applied to all 7
ordinals, because calibration is per layer.

## 4. Margin gate, drift resync, escalation

Margin gate (at the head, per denoise evaluation): denoise() gains
a file-scope flag g_gate_final, set by main when s == steps. When
the posture is Auto, the pack is calibrated, and g_gate_final is
set, denoise copies the conditional pass's input h before the
forward (B * H floats, only on final steps). After the conditional
logits and prior are composed, it computes each position's margin
(top1 - top2). If the minimum margin is below tie_epsilon:
- set g_gate_force_expand for the rest of this denoise call,
- re-run the conditional forward from the saved h (the scorer sees
  the force flag and finalizes EXPAND), recompose logits, prior,
  and cond_mean,
- count one escalation (printed in the run summary; logged per
  step in gate debug output).
The unconditional pass, when guidance != 1, runs under the same
force flag. The logits dump then records the escalated logits,
which is what the sampler consumes: the dump stays honest.
Intermediate steps are never margin-escalated: their outputs feed
z, and the long-chain controls below are the plan's answer there.

Drift estimator: drift2 as in section 3, plus a profile scale
factor (starts 1.0). Shadow resync: every g_gate_resync denoise
evaluations (counting conditional forwards; default 32, 0 = off),
in Auto only, the conditional forward runs twice: once under the
current marks, once forced EXPAND, from the same input h. The
relative L2 between the two outputs is the measured divergence;
the run proceeds on the EXPANDED output, the scale factor updates
to measured / predicted (clamped to [0.25, 4]), drift2 resets,
and the event is logged. Resync never applies to ATT_WRITE passes
(encode_clean): a double run would double-commit KV. Encode passes
rely on the within-pass budget only. This is a stated v1 limit.

## 5. Calibration storage (as landed H3b formats allow)

All three landed readers (infer, infer-lab, infer-triforce) parse
exactly g_ndir tensor lines from the manifest and close the file;
trailing lines are never read. So:

- v3 packs: the calibrator appends gate lines to the pack's
  .nq2manifest after the tensor lines:
      gate 1
      gate-gen <generation hex16>
      gate-eps <tie_epsilon>
      gate-budget <drift budget, final-layer rel L2 units>
      gate-layer <l> <delta rel L2 increment>   (one per layer)
  The engine's gate loader reads them only after the manifest
  verifies, and requires gate-gen to match the pack generation.
- Legacy packs (v1/v2, no manifest is ever read): the same gate
  lines go in a sidecar <stream minus .nq2>.nq2gate, keyed by
  gate-gen = FNV-1a over (version, block, entries, nblocks,
  total book floats, book bytes) computed at open from values
  the engine already holds. A mismatched or missing key means
  no calibration: the gate falls back to expanded, per the plan.
- The quantizer never writes gate lines; re-packing a model
  drops its calibration, which is the correct semantics (the
  numbers describe a specific pack's bytes).

Calibration production: scratch harness (this crew, pre-gate) on
the frozen pack; the harness's analyzer emits the gate lines.
The same analyzer becomes a small resident tool only if JD asks;
for this landing the lines for the frozen pack are produced by
the harness and stored beside it, and the derivation is recorded
in results/ so the numbers are auditable.

tie_epsilon derivation (from the harness data): per position and
evaluation, dmax = max over vocab of |logit_DTC - logit_dense|.
A tie can flip only when margin < 2 * dmax. tie_epsilon =
4 * (the largest dmax observed across the calibration set), the
factor 4 covering unseen prompts at the same pack. The drift
budget = 0.5 * (final-layer rel L2 divergence of always-DTC vs
dense on the calibration passes), in the same units as the
profile increments.

## 6. Postures and knobs

--posture auto|fast|think-harder|evidence; env NQ2_POSTURE below
the CLI, same three-layer family as the pipe knobs, strict
parsing, startup line confesses the effective posture and its
source. Default at first landing: evidence (which is bit-identical
to today's engine: every ordinal EXPAND, no gate state consulted).
The default flips to auto only through the validation gate in
section 7, as a deliberate re-anchor.
--gate-lookahead 1..4 (default 2), env NQ2_GATE_LOOKAHEAD.
--gate-resync N (default 32 evaluations, 0 off), env likewise.
--gate-eps and --gate-budget overrides exist for validation
experiments only; the startup line prints OVERRIDE when used.
Gate debug logging (--gate-log PATH): per pass, the marks, per
step the margins and drift, each escalation and resync with its
measured numbers. Diagnostic doctrine: present in this build,
documented as the debug channel; release stripping is a build
flag away and out of this landing's scope.

## 7. Validation and the default decision (plan section 8)

Runs, all on the frozen pack unless noted, house anchor config
(--dir sigma-3b --packed FROZEN --ids 1 --tokens 32 --steps 1
--guidance 1 --score-temp 1 --seed 42) plus the behavioral prompt
set from results/behavioral-gate-2026-11-25.txt practice
(--logits-dump comparisons through nq2-logitcmp):
1. Evidence regression (non-negotiable): tokens 122388 packed /
   75768 raw configuration, streamtest and dc-adv untouched by
   this change but re-run in the final gate; littlenemo forward
   RMS is a different binary and is re-run by build-all's suites.
2. Agreement battery: Auto vs Think Harder (always expanded) vs
   Fast (always DTC): token agreement over the anchor run and
   the behavioral set, escalation rate under Auto, logit
   agreement via logitcmp.
3. Speed retention: wall time of Auto and Fast vs Evidence on
   the anchor config, median of 3, bench contention stated.
4. Tie behavior: positions whose dense margin falls below
   tie_epsilon in the calibration data must show Auto escalating
   and matching the expanded tokens there.
5. Long-chain spot check: a longer generation (tokens 256,
   steps 1) under the three postures; report where Fast's tokens
   first diverge from expanded, and whether Auto tracks expanded
   throughout.
Default flip rule (from the brief): Auto becomes the default
posture only if gated token agreement is indistinguishable from
expanded self-agreement (identical tokens on every validation
run, reruns included) AND most of the DTC speed is retained
(Auto wall time closer to Fast than to Evidence). The flip is
recorded as a re-anchor: old pins, new pins, stability across
reruns, in the night log and the update notes. Otherwise the
gate ships with the default unchanged and the report says why.

## 8. Variant B (head DTC), only if A lands clean

head_all would read the head's packed sections in row tiles
(packed_fetch_range shape) and run the kernel per tile instead of
expanding tiles. The head tensor is 1,572,864 blocks at Sigma
geometry; its sections are read once per denoise evaluation
today, so the win is real but the change touches the one path
every logit flows through. It is reported as its own increment
or as not attempted, never blended into A's numbers.

## 9. Risks named in advance

- Bench contention: the format crew's packs and this crew's runs
  share about 1.5 CPUs. Timing numbers carry that caveat; token
  and agreement numbers do not (determinism is load-blind).
- The claim window arithmetic (slots vs ordinals per layer) is
  the one place a mark could be read late. The scorer finalizes
  a full lookahead window ahead, and UNDECIDED is not claimable,
  so a late read degrades to a stall the demand path resolves as
  EXPAND, never to a wrong answer. The final gate's config
  matrix (slots 2 and 8, lanes 1 and 2, fetch-ahead 1 and 18,
  the shapes that historically bit) must pass under Auto.
- Float determinism of DTC across thread counts: nqdc_apply's
  row loop is whole per position; positions partition across
  workers without splitting any accumulation. The gate runs at
  --threads 1 and the device default must produce identical
  tokens; the validation battery checks both.
