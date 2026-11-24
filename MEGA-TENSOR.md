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
ChatGPT (Gizmo).
