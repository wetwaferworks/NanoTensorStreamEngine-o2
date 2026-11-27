# The Precision Gate

The Precision Gate decides when the engine **expands** a packed tensor and when
it trusts the **table** (Dense Tensor Compute). It exists because of JD's ruling
of 2026-11-25: DTC is the engine's default compute path and expansion is the
exception, spent on purpose. The plan, design note, scratch implementation,
calibration record and test logs are in [`docs/precision-gate/`](precision-gate/);
this guide is the user-facing summary.

**Status, honestly.** The gate lives in the live Triforce engine,
`tools/nq2-infer-triforce.c`, transplanted from the scratch-phase copy that
remains at `docs/precision-gate/impl/nq2-infer-triforce.c` as the record of
that work. The shipped default posture is `evidence` (expand always), so an
unflagged run is the historical engine, bit for bit. The first calibration
reads hotter than the plan assumed, and Auto is not the default until that
is re-derived against this engine. On the POSIX split pipeline the postures
take effect; the Windows fused worker always expands and says so at startup.

## The principle

DTC and dense expansion of the same pack compute the same quantized model. They
differ only in float association: the table path reassociates the sum, so its
result matches the expanded path to float noise (measured around 1e-6 relative L2
per matmul in the probe), never bit-exactly. The quantization error itself
(packed versus the raw model, relative L2 0.6922 at 0.157 bits per weight) is five
orders of magnitude larger and is not affected by which path runs. Expansion buys
exactly one thing: exactness relative to the quantized model. The gate spends that
exactness only where association noise could change an outcome. If the goal is
closeness to the *raw* model, the levers are books and bit budgets
([ROADMAP.md](ROADMAP.md)); the gate does not pretend to be that lever.

## Postures

| Posture | Behavior | Use it for |
|---|---|---|
| `evidence` | expanded always, plus pinning and logging of every anchor the run touches. **The shipped default posture.** | gates, cross-checks, releases, anything another party will verify |
| `auto` | the gate: DTC by default, predictive escalation on near ties and drift | normal use, once calibrated and validated |
| `fast` | DTC always | drafting and play; never for evidence |
| `think-harder` | expanded always, no gate second-guessing | JD's requested mode: the user asks, the engine does the math the hard way |

An unknown posture is a named refusal. Select with `--posture NAME` or the
environment variable `NQ2_POSTURE`; the flag wins.

## The three trigger classes

1. **Near ties (per token, at the head).** If `top1 - top2 < tie_epsilon` for the
   current step, escalate that step to the expanded path and recompute before
   committing the token. `tie_epsilon` is calibrated per pack, not guessed.
2. **Long chains (per generation).** A drift budget (a running estimate of
   accumulated divergence, fed by the per-layer noise profile and the observed
   margins) escalates the next step when the projection crosses the budget; a
   shadow resync runs one step expanded every N tokens and re-anchors the
   estimate in measurement.
3. **Evidence runs (per invocation, absolute).** Any run whose output is evidence
   runs expanded, always, with no gate. A pin must mean one exact thing.

## Predictive lookahead

Deciding at consume time is too late, so the gate decides early on a horizon of 1
to 4 layers: at layer L it scores layers L+1 through L+lookahead and marks each
ordinal DTC or EXPAND on the Tensor Handshake priority list. An EXPAND ordinal is
staged and expanded by the lanes as usual; a DTC ordinal is staged as packed
sections only and the consumer runs the table path. Lookahead 1 is the reactive
floor; the default is 2. Signals scored: the per-layer noise profile, the running
drift projection, previous tokens' margins, and the posture flag (user intent
outranks prediction).

Escalation is per ordinal, never whole-model by default: head-only (recompute the
final projection and logits expanded), layer (one layer's projections run
expanded), or pass (the whole step, the shadow resync).

## Flags and environment

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--posture NAME` | `NQ2_POSTURE` | `evidence` | one of the four postures |
| `--gate-lookahead N` | `NQ2_GATE_LOOKAHEAD` | 2 | predictive lookahead, 1 to 4 layers |
| `--gate-resync N` | `NQ2_GATE_RESYNC` | | shadow resync interval in tokens |
| `--gate-eps F` | | calibrated | override the tie epsilon (positive; printed as `OVERRIDE`) |
| `--gate-budget F` | | calibrated | override the drift budget (positive; printed as `OVERRIDE`) |
| `--gate-log FILE` | | off | per-step decision log: ordinal marks, margins, drift estimate |

The engine also takes every `nq2-infer-triforce` flag (see
[INFERENCE-TOOLS.md](INFERENCE-TOOLS.md)). At startup it prints a confession line
(`calibration: eps ..., budget ...`) and at the end a summary line such as
`2 evaluations, 1 escalations, 0 resyncs; layer consumptions via DTC 72 of 156`.

## Calibration carriage

A pack carries its own gate calibration, and an uncalibrated pack falls back to
expanded: the gate never gates on unmeasured goods.

- **v3 packs:** gate lines ride in the manifest after the tensor lines.

```
gate 1
gate-gen <fnv1a64 hex>        must equal the pack generation
gate-eps <float>              tie epsilon
gate-budget <float>           drift budget
gate-layer L <increment> <cumulative>     per-layer noise profile
```

- **Legacy packs:** the same lines live in a sidecar `<stream>.nq2gate`, keyed by a
  content FNV-1a 64 over the pack geometry and stored book bytes.

Derivation (`calib/DERIVATION-2026-11-25.md`): per-layer divergence is the relative
L2 between DTC and dense hidden states after each layer of the last denoise pass;
increments are `sqrt(max(div[l]^2 - div[l-1]^2, 0))`; the drift budget is
`0.5 * div[final]`; `tie_epsilon = 4 * max(dmax)` where `dmax` is the largest
logit difference between the two paths per position.

## Determinism of the gate itself

Every decision is a pure function of computed values and the calibrated
thresholds, never of wall clock, thread timing or lane availability. A gated run
reproduces bit-exactly on re-run: same pack, same prompt, same posture, same
tokens. The gate changes *which* exact computation runs at a step; it never
introduces a race into the arithmetic.

## Where it stands (scratch test record, 2026-11-25)

| Test | Result |
|---|---|
| Evidence, packed (anchor config `--ids 1 --tokens 1 --steps 1`) | token 122388, ticks in agreement: PASS |
| Evidence, raw | token 75768, ticks in agreement, gate inert: PASS |
| Fast with the calibration sidecar | token 122388, 130 of 130 layer consumptions via DTC, 1 m 53 s wall: PASS |
| Auto | token 122388; "2 evaluations, 1 escalations, 0 resyncs; layer consumptions via DTC 72 of 156"; 12 m 12 s wall: PASS as a run |
| Auto at `--threads 1` | 2026-11-25 record: first attempt FAULTED at the first pass boundary in 4 s ("the prior pass did not drain"); an identical rerun completed with token 122388. The unmodified base engine at `--threads 1` completed cleanly. **Fixed 2026-11-26:** `pipe_pass_boundary()` was called twice at ordinal 0; the consume-tick count (`pass_consumed == 0`) makes later calls a no-op, in the scratch impl and the tools copy |

The `--threads 1` fault in the table is the 2026-11-25 scratch record, not an undetermined race. Miss Archie's Round 6 note identified the root cause: a second `pipe_pass_boundary()` call at ordinal 0, from both `matmul_pipe` and the consume path, while the fresh pass was already in motion. The 2026-11-26 zero-mutex Handshake conversion counts consume ticks so the boundary runs once per pass (`if (g_pipe.pass_consumed == 0) return;`). That fix is in the live `tools/` engine. The fault existed only on the POSIX split path; the Windows fused worker has no `pipe_pass_boundary()`.

After the transplant into `tools/` (2026-11-27), the live engine was rebuilt and re-run on the POSIX split path (WSL). Packed evidence at the anchor config, `--posture evidence`, produced token 122388 with ticks in agreement (fetch 181, expand 182 of 182 ready, consume 181) at both `--threads 12` and `--threads 1`. The `--threads 1` run crossed every pass boundary without the 2026-11-25 fault and exited 0. `tools/test-precision-gate-flags.sh` covers the named refusals (unknown posture, lookahead, resync, non-positive eps). On Windows the same source compiles (one existing snprintf truncation warning) and accepts the flags; the fused worker always expands and says so at startup.

First calibration (Sigma pack, scratch measuring vehicle): `tie_epsilon` 35.076,
drift budget 1.277, per-layer divergence from 7.89e-02 at layer 0 to a peak of
8.479 at layer 5 (largest increments at layers 4 and 5), argmax agreement between
the two vehicles 8 of 64. As calibrated, Auto sits much closer to Evidence than
to Fast in wall time and would behave like Think Harder with gate overhead. The
gate engine's own DTC path (Fast) reproduced the dense anchor token exactly, so the
large divergence may be a property of the scratch vehicle's consume path rather
than of DTC. **The calibration must be re-derived against the gate engine, or the
vehicle audited, before the default posture flips to Auto** (plan section 8: an
agreement battery, speed retention, a long-chain test, tie injection and anchor
discipline).

## Record notes (2026-11-27)

**Frozen forensic engine.** `tools/nq2-infer-triforce-dbg.c` was frozen at
commit `04a4640` as evidence of the bisected Triforce failure (SHA-256
`f09b813d32c201f609391f65de33e0547ef6a419357c6f73c0c6ef5d39f7c700`, the hash in
`MANIFEST-SHA256.txt`). Commit `ef39281` (2026-11-26, the zero-mutex Handshake
conversion) rewrote that file in place, so the record pointed at evidence that
no longer matched it. The frozen bytes are restored unchanged at
`tools/frozen/nq2-infer-triforce-dbg.04a4640.c.txt` (not compiled; see
[tools/frozen/README.md](../tools/frozen/README.md)). The zero-mutex rewrite
stays at `tools/nq2-infer-triforce-dbg.c` and is the buildable debug tool; it
supersedes the frozen copy as something to run, never as evidence.

**Which boundary fix is live.** Two fixes were written for the 2026-11-25
`--threads 1` boundary fault. The live one is the consume-tick count
(`pass_consumed`), which runs the pass boundary once per pass; it is in both
`tools/nq2-infer-triforce.c` and [`precision-gate/impl/`](precision-gate/impl/)
and is the fix verified above (token 122388, ticks 181/182/181 at `--threads 12`
and `--threads 1`). A local 2026-11-26 transplant fixed the tools copy with a
boundary epoch stamp instead. That transplant never reached this tree and is
superseded by the consume-tick count; nothing here depends on it.

**Re-derived calibration.** A later review cites a re-derived Sigma calibration
(`tie_epsilon` 4.699707e-03, drift budget 3.308205e-05, argmax agreement 64 of
64). Its run record was not found on the bench machine on 2026-11-27, so those
numbers are not adopted here. The scratch calibration above, with its record in
[`precision-gate/calib/DERIVATION-2026-11-25.md`](precision-gate/calib/DERIVATION-2026-11-25.md),
stays the only calibration on file, and the default posture stays Evidence until
a re-derivation lands with its record.

## Credits

Directed by JD (2026-11-25). Plan and design: Elfie. Scratch-phase implementation
and tests: the scratch-phase crew.