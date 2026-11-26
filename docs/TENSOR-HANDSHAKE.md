# Mega Tensor and the Tensor Handshake

**Mega Tensor** is JD's name (2026-11-24) for the streaming system underneath
NTSE: the layer that moves the tensors. A packed model is small enough to stream
like media, but only if something streams it well. Mega Tensor fetches packed
sections from disc, expands them back into weights, and feeds them to the
compute loops in exactly one order while everything else runs concurrently. Its
rule, in Gizmo's formulation: **concurrent preparation, serialized consumption,
verifiable results.**

The **Tensor Handshake** is the coordination protocol Mega Tensor runs on. The
idea and shape are JD's: a cousin of the timing handshake, in which the tick and
control loops ask the core loop "is it time to tick?" and "can I tick?" instead
of taking a lock. The engines that use it are Elfie's; Tensor Handshake stage 2
was Cortana's lane (JD's ruling of 2026-11-23); the zero-mutex conversion was
co-implemented by the assistants. The primitives are in `tools/nq2-handshake.h`
([ZERO-MUTEX.md](ZERO-MUTEX.md)).

## The four loops

The streaming design (JD, 2026-11-24, designed at the podium before bed):

- one disc-lookup loop, staging raw packed sections;
- two unpack loops, reconstructing weights from the staged sections;
- one compute loop, consuming finished tensors in a fixed order.

Preparation is parallel and opportunistic. Consumption is serial and exact. There
are no phase boundaries and no barriers between the arms: fetch, expand and
prefetch run concurrently with no walls between them. The only hard boundary is
the consumption point itself, plus one epoch counter that bumps at every pass
restart so staged work from an older pass is discarded rather than consumed
stale. That is Time Serialization applied to a weight stream: the discipline, not
a lock, makes the output deterministic. `nq2-streamtest` is the harness; its
table (sequential 33.49 s down to 17.93 s, accumulator 1394.1117661067326 in
every row) is in [INFERENCE-TOOLS.md](INFERENCE-TOOLS.md).

## The Triforce split

The inference pipeline built on Mega Tensor (`nq2-infer-triforce`) splits the
work three ways: a fetcher stages raw sections from disc, expansion lanes
reconstruct weights, and the consumer takes tensors in the model's fixed
per-layer order. Pipeline geometry is runtime-configurable (`--pipe-slots`,
`--fetch-ahead`, `--prefetch-dist`, `--expand-lanes`; flags in
[INFERENCE-TOOLS.md](INFERENCE-TOOLS.md)).

### The race the anchor caught (2026-11-24)

The first full Triforce run diverged from the recorded 1-step packed anchor at
position 8. Bisection, one variable at a time: the fused lab engine matched (pack,
book and sampler innocent); the Triforce in raw mode diverged worse (the defect
was in delivery generally); the Triforce with workers bypassed matched exactly
(the arithmetic was innocent, the delivery was convicted). Root cause: the
fetcher's branch for not-fully-packed tensors read directly into a consumption
slot at fetch time, and with prefetch nudging it two ordinals ahead, at two slots
ordinal N+2's slot is ordinal N's slot: the fetch overwrote weights the consumer
was still using. A second defect: the fetcher's high-water mark never reset
across denoise evaluations, so it silently stopped working after the first. The
fix: fetch only marks the stage and the load happens at expand time under the
demand discipline, and pass restarts bump an epoch counter that resets the
fetcher's marks and makes a stale worker discard its publish. Both 1-step anchors
then matched exactly: packed, 485 of 546 requests served from the pipe (88.8
percent, up from 23.4), consumer stall 1,035 ms (down from 70,610), load path
1,876 MB/s; raw, 362 of 546 hits (66.3 percent). Full story:
[PIPELINE-2026-11-24.md](../PIPELINE-2026-11-24.md).

## The layer counter and stage 1

The pipe publishes a layer counter: the consumer's committed ordinal, on the
existing ordinal machinery. Fetch-ahead targets are that counter plus the
configured distance, so prefetch presses N+2, N+3, N+4 by count rather than by
guess.

## Stage 2 (landed 2026-11-25)

- **Ticks.** Each pipeline thread publishes a tick: the fetcher its highest
  staged ordinal, the expansion lanes the count of ordinals ready, the consumer
  its committed ordinal. A pass may not roll over until the three ticks agree the
  prior pass drained; that agreement is the commit condition for the restart, and
  disagreement is a printed fault, never a silent wait.
- **The prefetch priority list.** Work selection runs through a global list, one
  row per layer: its state, which thread holds its claim, and each worker's
  current and next intended layer. Priority derives from the consumer's counter,
  so the nearest unready layer is always next.
- **Claims.** A layer is claimed by exactly one thread, taken with
  compare-and-swap and cleared only by its holder; no two threads ever expand the
  same layer at the same time.
- **The door.** Every change to the list bumps one generation counter and the
  door word, so there is a single door for every waiter.
- **The pass boundary is a handshake of its own.** The consumer asks, every worker
  parks at its decision point with no claim held and says so, the consumer alone
  rebuilds the list, then opens the next pass.
- **Scheduling only.** The list governs scheduling, never arithmetic or commit
  order: consumption commits in ordinal order, which is why the anchors stay
  bit-identical at every width.
- **Expand lanes.** Expansion width is its own runtime knob (1 to 64), separate
  from pipeline depth: staging may run far ahead while only a few layers expand at
  once, because expansion is bounded by buffer lifetime, not fetch depth. The
  ladder {2, 4, 8, 16, 32, 64} is the residency dial: more memory buys more layers
  in flight, exactly as far as the hardware ceiling allows, and the guardrails
  make the dial safe to turn. A full pass expands a 75 MB stream into about 15 GB
  of fp32 weights through small tensor buffers, at most three expanded layers
  provisioned at once.

## Hardware awareness

Hardware awareness is Mega Tensor's defining property. Every Tensor Handshake
default (slots, fetch-ahead, prefetch distance, expand lanes) is derived from the
machine underneath: core count and RAM today, structured so a GPU's memory can
join the inputs. Overrides are range-checked against the same hardware, clamped
loudly with the reason printed rather than honored into an allocation failure,
and the startup line reports every effective value with its source (`cli`, `env`
or `device`), so a run on any machine confesses its own shape.

## The same protocol on the GPU

The accel lanes apply the Handshake to host-to-device streaming: pinned staging
slots with an explicit ownership word per slot (`FREE`, `STAGING`, `STAGED`,
`INFLIGHT`), claims by compare-and-swap, a slot recycled only after its `done`
event has been observed, and a forced early recycle that the guard refuses and
counts. The prepared stream plan reached 76.40 tok/s (linear-layer ceiling) on
the RTX 3050 Ti against 7.97 for the old stream. See [ACCEL.md](ACCEL.md).

## Handshake counters in the GUI

Protocol v3's `HS` line carries the Tensor Handshake counters cumulatively per
session (pass, ordinal, prefetch hits and misses, stall time, DTC requests and
DTC time, staged buffers); the stats panel shows them with a hit rate. DTC
requests count as `dtc`, not as prefetch misses, and DTC compute time counts as
`dtc_ms`, not stall time. See [GUI.md](GUI.md) and `littlenemo/PROTOCOL.md`.

## Measured load-path figures (throttled bench, floor numbers)

- dedicated streaming runner: 922.7 MB/s expanded;
- the fixed packed re-anchor run: 1,876 MB/s load, 88.8 percent prefetch hits,
  consumer stall about 1 second per pass (the broken build stalled 70.6 seconds).

## Why it matters

A frontier model in bf16 moves tens of gigabytes per generated token. The same
model packed by this engine moves a few hundred megabytes, and Mega Tensor is
what turns that smaller river into tokens without dropping the determinism that
makes the results verifiable.

## Credits

Architecture and the name: JD. Implementation: Elfie. Stage 2 lane and the GPU
side: Cortana. Research program consultee and the governing rule: ChatGPT
(Gizmo). Reviewer: Miss Archie (Argon), whose 2026-11-24 review hardened the
pipeline engine (the demand-protocol race, pass-restart re-fetch and the mode-2
staging buffer).