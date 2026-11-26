# The Precision Gate: when the engine expands, and when it trusts the table

A plan for predictive precision in the Nano Tensor Stream Engine (NTSE-o2).
Directed by JD, 2026-11-25. Written by Elfie the same day.
Status: PLAN. Nothing in this document is implemented yet. The DTC default
ruling it serves was given by JD on 2026-11-25: Dense Tensor Compute is the
engine's default compute path; expansion is the exception, spent on purpose.

## 1. The principle

DTC and dense expansion of the same pack compute the same quantized model.
They differ only in float association: the table path reassociates the sum,
so its result matches the expanded path to float noise (measured around
1e-6 relative L2 per matmul in the probe), never bit-exactly. The
quantization error itself (packed versus the raw model, rel L2 0.6922 at
0.15691 bpw) is five orders of magnitude larger and is NOT affected by
which path runs. Expansion therefore buys exactly one thing: exactness
relative to the quantized model. The gate exists to spend that exactness
only where association noise could change an outcome, and nowhere else.

Corollary, stated once so nobody has to rediscover it: if a future goal is
closeness to the raw model, the levers are books and bit budgets. The gate
does not pretend to be that lever.

## 2. The three trigger classes

### 2.1 Near ties (per token, at the head)

When the top two logits of a token sit within the noise of each other, the
fast path's argmax can differ from the expanded path's argmax, and one
flipped token reroutes the whole generation after it. The margin between
first and second place is computed anyway during sampling; it is the
cheapest uncertainty signal the engine will ever produce.

Rule: if (top1 - top2) < tie_epsilon for the current step, escalate that
step to the expanded path and recompute before committing the token.
tie_epsilon is calibrated per pack (section 5), not guessed.

### 2.2 Long chains (per generation, compounding)

Association noise compounds across an autoregressive chain. Any single
step is harmless; thousands of steps are a random walk, and a near tie
somewhere in the middle is a fork. Two controls:

- Drift budget: a running estimate of accumulated divergence between the
  fast state and where the expanded state would be, fed by the per-layer
  noise profile (section 5) and the margins actually observed. When the
  projected drift crosses the budget line, the next step escalates.
- Shadow resync: every N tokens (N a posture setting), one step is run
  expanded regardless, and the hidden states are compared. The measured
  divergence recalibrates the running estimate, so the budget is anchored
  in measurement, not only in the model of the noise.

### 2.3 Evidence runs (per invocation, absolute)

Any run whose output is evidence runs expanded, always, no gate:
anchor pins, behavioral gates, cross-checks against external references,
and any artifact another party will verify. A pin must mean one exact
thing. The gate never sits between a verification and its reference path.

## 3. The predictive lookahead (1 to 4 layers)

Deciding at consume time is too late: expansion takes work, and a gate
that stalls the consumer to expand on demand hands back the speed the
default was chosen for. The pipe already runs ahead of the consumer
(fetch depth is decoupled from expansion width; the Tensor Handshake
priority list schedules preparation by ordinal). The gate therefore
decides EARLY, on a lookahead horizon of 1 to 4 layers:

- At layer L, the gate scores layers L+1 .. L+lookahead and marks each
  ordinal in the window DTC or EXPAND.
- The priority list expresses the marks directly: an EXPAND ordinal is
  staged and expanded by the lanes as today; a DTC ordinal is staged as
  packed sections only, and the consume step runs the table path.
- Lookahead 1 is the reactive floor: correct, but a late tie signal can
  still cost a stall. Lookahead 2 to 4 trades signal freshness for a
  stall-free schedule: the decision rides the prefetch horizon the pipe
  already maintains. Default proposed: 2, tunable per posture.

Signals the predictor scores, all available before the layer runs:

- The per-layer noise profile (calibration, section 5): some layers are
  known hot. The sensitivity work already in flight (top 10 of 241
  tensors holding 54.6 percent of squared error) is the seed of this map.
- The running drift estimate (2.2), projected forward across the window:
  if the projection crosses budget inside the window, the crossing layer
  and everything after it in the window escalate.
- The previous tokens' margins: a generation that has been producing
  near ties keeps producing them for a while; a run of wide margins
  predicts a quiet window.
- The posture flag (section 6): user intent outranks prediction.

## 4. Escalation mechanics

Escalation is per ordinal, never whole-model by default:

- Head-only escalation: recompute the final projection and logits
  expanded for a tied step. Cheapest; resolves the tie with exact
  quantized semantics.
- Layer escalation: one layer's projections run expanded; the table
  path resumes after it. The state after an expanded layer is the exact
  quantized state, which also zeroes the running drift estimate honestly
  (the estimate restarts from a known state, it is not merely forgiven).
- Pass escalation: the whole step runs expanded (shadow resync, 2.2).

Table lifetime note (from the DTC review, readiness items): a DTC table
is per (tensor book, activation) and lives inside the pipe_get window;
expansion marks must respect the same window, and table builds complete
before lane dispatch, with per-row ascending order preserved, so the
fast path stays deterministic under the Tensor Handshake.

## 5. Calibration (per pack, at pack time or first load)

The gate's numbers come from measurement, once per pack:

- Run both paths over a fixed calibration set of prompts. Record, per
  layer, the hidden-state divergence between paths, and at the head the
  distribution of top1-top2 margins under both paths.
- Derive tie_epsilon from the observed head divergence (the margin below
  which the two paths were seen to disagree, with a safety factor).
- Store the noise profile and tie_epsilon in the pack's manifest, next
  to the per-tensor digests the H3b work already adds. A pack carries
  its own gate calibration; engines refuse to gate on a pack that lacks
  one and fall back to expanded (the honest default for unmeasured
  goods).

## 6. Postures (user-facing)

Four postures, presented beside the existing pool modes in the GUI:

- Auto: the gate as specified. DTC default, predictive escalation on.
- Fast: DTC always. For drafting and play; never for evidence.
- Think Harder: expanded always. JD's requested mode: the user asks,
  the engine does the math the hard way, no gate second-guessing.
- Evidence: expanded always, plus pinning and logging of every anchor
  the run touches. For gates, cross-checks, and releases.

## 7. Determinism of the gate itself

The gate is part of the engine and obeys the engine's law:

- Every gate decision is a pure function of computed values and the
  calibrated thresholds in the manifest. Never of wall clock, thread
  timing, or lane availability.
- Decisions are logged per step (ordinal marks, margins, drift estimate)
  in debug builds, stripped in release, per the diagnostic doctrine.
- A gated run reproduces bit-exactly on re-run: same pack, same prompt,
  same posture, same tokens. The gate changes WHICH exact computation
  runs at a step; it never introduces a race into the arithmetic.

## 8. Validation plan (before the gate may default to Auto)

- Agreement battery: gated versus always-expanded versus always-DTC on
  the behavioral prompt set. Report token agreement of gated against
  expanded (the target: agreement indistinguishable from expanded
  against itself across reruns), and the escalation rate (the design
  point: a small single-digit percentage of steps).
- Speed retention: gated throughput against always-DTC on the same
  runs. The gate earns its place only if it keeps most of the table
  win; report the retained fraction, do not round it into a slogan.
- Long-chain test: a multi-thousand-token generation under Auto,
  Fast, and Think Harder. Plot divergence of Fast from expanded over
  position; show where Auto's resyncs fire and that its output tracks
  the expanded run.
- Tie injection: prompts chosen to produce near ties (calibration set
  plus adversarial picks). The gate must escalate on the tied steps and
  match the expanded run's tokens there.
- Anchor discipline: flipping the default to Auto is a deliberate
  re-anchor, documented like any other, with new pins verified stable
  across reruns before they are called pins.

## 9. Open questions (for JD, and for the CUDA session)

- Default lookahead: 2 proposed. Is the stall-versus-freshness trade
  JD's to set per posture, or one global value?
- Escalation granularity at a tie: head-only first, layer on repeated
  ties? Or straight to layer?
- Shadow resync interval N: fixed, or scaled by the drift estimate?
- GPU semantics: a CUDA DTC path needs the same gate story. Do gate
  decisions stay on the CPU consume side (recommended: the predictor
  is cheap, the tables are the GPU's work), and what does an EXPAND
  ordinal mean when the expanded path is also on device?
- Does PCHIE get a vote? A harness-verified run may want Evidence
  posture enforced from outside the engine rather than requested.

## 10. What this plan is not

Not an approximation scheme: both paths compute the same quantized
model, and the gate only arbitrates association noise. Not a quality
knob for the pack: books and budgets own quality. Not a claim: every
number above that is not already in the receipts is a target for the
validation plan to prove or refute.

Credits: direction and the default ruling, JD (executive direction).
Plan and design, Elfie. Dense Tensor Compute implementation and core
mathematics, Elfie; DTC review evidence, the review crew of 2026-11-25.
