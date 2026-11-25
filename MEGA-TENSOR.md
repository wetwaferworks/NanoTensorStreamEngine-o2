# Mega Tensor

Mega Tensor is the name JD gave, on 2026-11-24, to the streaming
system underneath the Nano Quant Engine: the layer that moves the
tensors. A packed model is small enough to stream like media, but
only if something streams it well. Mega Tensor is that something.
It fetches packed sections from disc, expands them back into
weights, and feeds them to the compute loops in exactly one order,
while everything else runs concurrently.

The rule it lives by, in Gizmo's formulation: concurrent
preparation, serialized consumption, verifiable results.

## The four loops

The streaming design has four loops:

- one disc-lookup loop, staging raw packed sections;
- two unpack loops, reconstructing weights from the staged
  sections;
- one compute loop, consuming finished tensors in a fixed order.

Preparation is parallel and opportunistic. Consumption is serial
and exact. There are no phase boundaries and no barriers between
the arms: fetch, expand, and prefetch run concurrently with no
walls between them. The only hard boundary is the consumption
point itself, plus one epoch counter that bumps at every pass
restart so staged work from an older pass is discarded rather
than consumed stale. That is Time Serialization applied to a
weight stream: the discipline, not a lock, is what makes the
output deterministic.

Measured on the bench VM (about 1.5 shared cores, so these are
floor numbers, not peaks), streaming the full Sigma-3B pack:

| Configuration | Time | Rate |
|---|---|---|
| Sequential, one loop | 33.49 s | 90.4 M weights/s |
| Two unpackers, lookahead 8 | 20.56 s | 147.2 M weights/s |
| Row-pointer table, prefetch distance 2 | 17.93 s | 168.8 M weights/s |

The accumulator over the whole stream is bit-identical in every
configuration: 1394.1117661067326. Concurrency changes the speed,
never the sum.

## The Triforce split

The inference pipeline built on Mega Tensor splits the work three
ways (the Triforce build): one loop stages raw sections from
disc, a second reconstructs weights from them, and the consumer
takes tensors in the model's fixed per-layer order. Pipeline
geometry is runtime-configurable (slots and fetch-ahead depth are
flags, never hard coded), and the split earned its keep the hard
way: its first determinism failure was bisected to a race in this
delivery machinery, fixed, and both 1-step anchors now reproduce
exactly. The full story is PIPELINE-2026-11-24.md.

**The Tensor Handshake.** The pipe publishes a layer counter: the
consumer's committed ordinal, on the existing ordinal machinery.
Fetch-ahead targets are that counter plus the configured
distance, so prefetch presses N+2, N+3, N+4 by count rather than
by guess.

Stage 2 of the Handshake landed on 2026-11-25, with the
bit-identical proof the design demanded at every configuration
it was run. Each pipeline thread now publishes a tick: the
fetcher its highest staged ordinal, the expansion lanes the
count of ordinals ready, the consumer its committed ordinal. A
pass may not roll over until the three ticks agree the prior
pass drained; that agreement is the commit condition for the
restart, and disagreement is a printed fault, never a silent
wait. Work selection runs through a global prefetch priority
list, one row per layer: its state, which thread holds its
claim, and each worker's current and next intended layer.
Priority is derived from the consumer's counter, so the nearest
unready layer is always next. A layer is claimed by exactly one
thread, stamped under the lock; no two threads ever expand the
same layer at the same time. Every change to the list bumps one
generation counter and wakes the one wait point, so there is a
single door for every waiter and the phase boundary is that
door. The list governs scheduling only, never arithmetic or
commit order: consumption still commits in ordinal order, which
is why the anchors stay bit-identical at every width. Expansion
width is its own runtime knob (expand lanes, 1 to 64), separate
from pipeline depth: staging may run far ahead while only a few
layers expand at once, because expansion is bounded by buffer
lifetime, not by fetch depth. The lane ladder {2, 4, 8, 16, 32,
64} is the residency dial: more memory buys more layers in
flight, exactly as far as the hardware ceiling allows, and the
guardrails below are what make the dial safe to turn.

**Hardware awareness is Mega Tensor's defining property.** A
streaming engine is a tensor pipeline, and a tensor pipeline
has to configure itself to the hardware it runs on. Every
Tensor Handshake default (slots, fetch-ahead, prefetch
distance, expand lanes) is derived from the machine underneath:
core count and RAM today, structured so a GPU's memory can join
the inputs when that lane exists. Overrides are range-checked
against the same hardware, clamped loudly with the reason
printed rather than honored into an allocation failure, and the
startup line reports every effective value with its source, so
a run on any machine confesses its own shape. The Tensor
Handshake is the coordination protocol; hardware awareness is
what the protocol serves.

Measured load-path figures, same throttled bench:

- dedicated streaming runner: 922.7 MB/s expanded;
- fixed packed re-anchor run: 1,876 MB/s load, 88.8% prefetch
  hits, consumer stall about 1 second per pass (the broken build
  stalled 70.6 seconds);
- one full pass expands a 75 MB stream into about 15 GB of fp32
  weights through small tensor buffers, at most three expanded
  layers provisioned at once.

## Why it matters

A frontier model in bf16 moves tens of gigabytes per generated
token. The same model packed by this engine moves a few hundred
megabytes, and Mega Tensor is what turns that smaller river into
tokens without dropping the determinism that makes the results
verifiable. The GPU decode stage (packed stream resident in
VRAM, expansion on the device) is the next arm of the same
design; it is being built on local hardware now.

## Credits

Architecture: JD (David Rittenhouse), who also named it.
Implementation: Elfie (E.L.F.I.E.). Research program consultee:
ChatGPT (Gizmo). Reviewer: Archie (Argon), the Google Gemini
Argon model, the Bit Master and the Bit Masker, whose
2026-11-24 review hardened the pipeline engine (the
demand-protocol race, pass-restart re-fetch, and the mode-2
staging buffer).
