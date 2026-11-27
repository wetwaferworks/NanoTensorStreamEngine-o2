# GPU expert stream (SuperNemo CUDA path)

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.

## What it does, in plain words

SuperNemo has 512 small "expert" networks in each of its 40 MoE layers, and every token picks 22 of them per layer.
That is far too much to keep on a 4 GB card, and before this change the experts were always computed on the CPU.

With the stream on, the engine works like this for every MoE layer of every token:

1. **Pick.** The router chooses the experts, exactly as before.
2. **Check the card.** Experts that are still sitting in a card slot from an earlier token are used straight away.
3. **Send the compressed form.** Experts that are missing are copied into pinned RAM staging buffers, then sent to the card
   in their *compressed* form: index bytes, per-block min/scale and the codebook, about 1.9 MB per expert (up and down
   matrices together). The copy comes from the warm expert cache in system RAM when the expert is there, otherwise from the pack.
4. **Expand on the card.** CUDA kernels expand the compressed blocks into weights on the card itself. For decode the expansion
   happens inside the compute kernel, in registers (the same DTC method the CPU uses), so nothing bigger than the compressed
   form ever crosses the bus or is written to VRAM. For prefill (many tokens at once) the kernel writes the expanded matrix
   to a work buffer on the card and multiplies all the tokens that chose that expert in one go.
5. **Compute on the card.** relu squared and the weighted sum happen on the card too; only the layer's output row comes back.
6. **Let go.** The slots are not pinned to anything: the next token reuses them. The least recently used slot is the one that
   gets the next expert, so experts that keep getting picked stay on the card for free.

Sending and computing overlap: there are two staging halves and two CUDA streams. While the card computes one chunk of experts,
the CPU fills the other half and the copy engine sends it. Events (no locks anywhere) keep a staging half or a slot from being
overwritten while the card still reads it.

## How many slots

The slot count is not a hard-coded number. It comes from the hardware calibration file NemoChat writes
(`%LOCALAPPDATA%\NemoChat\nemo-calib.txt`, or the file named by `NEMO_CALIB_FILE`):

    slots = (min(free VRAM now, free VRAM at calibration) - dense tensors already on the card
             - the VRAM expert cache - work buffers - 64 MB headroom) / bytes per expert slot

capped at the number of experts in the model. The engine refuses to stream when the calibration says the card is not the lane
to use. Staging is two chunks of up to 22 experts (or half the slots, whichever is smaller). The startup line on stderr says
what it chose and why, for example:

    sn-engine: GPU expert stream on: N expert slots (... MB VRAM, 1.89 MB each), M pinned staging buffers (... MB RAM), ...; sized from <calibration file> ...

and after each turn a `GPUX` line reports slot hits, loads, MB sent and how many experts ran on the card.

## Switches

| Want | Flag | Ini key (`[supernemo-tuning]`) |
|---|---|---|
| Stream experts through the card (default whenever `--gpu-mb` is above 0) | `--gpu-experts` | `gpu_experts = 1` |
| Keep experts on the CPU lanes (the old behaviour) | `--no-gpu-experts` | `gpu_experts = 0` |
| Stream experts only, no dense tensors on the card | `--gpu-mb 0 --gpu-experts` | |

The stream needs DTC on (the default) and replaces the older VRAM expert tier (`--gpu-cache-mb`), which is ignored while the
stream is on. If anything on the card fails mid-run the engine finishes that layer on the CPU, prints a note on stderr and
carries on with the stream switched off.

## Is it exact?

Yes, in the sense the project uses:

- The on-card expansion writes the same bytes as `sn_expand` on the CPU, checked on every resident tensor and on streamed
  experts of the real pack.
- Decode through slots is bit-identical to the CPU DTC path; prefill through slots is bit-identical to the CPU prefill GEMM
  (the kernel copies the CPU's fused multiply-add order lane for lane and its sum order; NVRTC is run with `--fmad=false`).
- A whole-model run with the stream on gives the same logits hash as the CPU at any thread count, and the same hash as
  today's dense CUDA lane when both are on.

One older limit is unchanged and has nothing to do with the stream: the dense-tensor CUDA lane (`--gpu-mb`) has always been
float-close but not bit-identical to the CPU, so "CPU only" and "`--gpu-mb 3000`" have different hashes, with or without the stream.

Test: `test_sn_xstream.exe PREFIX SHAPES [CONFIG]` (kernel checks, slot reuse and eviction, model hashes across thread counts).
`SN_XS_MAX_SLOTS=N` forces a small slot count to exercise eviction.

## Speed and memory on this machine

Measured on 2026-11-27 on the development laptop (12 threads, RTX 3050 Ti 4 GB, 16 GB RAM), same prompt, 32 new tokens
allowed, each pair run back to back. Another job was using the CPU during all of these, so treat them as real-world numbers;
the before/after ratio is the useful part, not the absolute speed.

| Run | Decode tok/s | Prefill tok/s | Card VRAM peak | Engine RAM peak |
|---|---|---|---|---|
| `--gpu-mb 3000 --no-gpu-experts` (before) | 0.154 | 0.301 | 3446 MiB | 686 MB |
| `--gpu-mb 3000` (after, stream on) | 0.253 | 0.813 | 3749 MiB | 609 MB |
| `--gpu-mb 3000 --no-gpu-experts` (before, again) | 0.092 | 0.265 | 3445 MiB | 686 MB |
| `--gpu-mb 3000` (after, again) | 0.104 | 0.487 | 3708 MiB | 608 MB |
| `--gpu-mb 0 --gpu-experts` (experts only) | 0.030 | 0.454 | 3701 MiB | 650 MB |

Every run wrote the same answer. With the dense tensors on the card there is room for about 115 to 119 expert slots, fewer
than the 880 experts one token touches, so slots are rarely reused from one token to the next and the warm tier that matters
is the RAM expert cache. The experts-only setting fills the card with about 1684 slots and does reuse them, but its decode
is slow on this card when the slot pool fills the whole card (see item 5 below).
Use it only to experiment; the default (`--gpu-mb` with the stream on) is the one to use.

The quiz set was run again with the stream on: the five short questions (square root of 144, capital of Australia, symbol
for helium, year of the Declaration of Independence, largest planet) scored 5 of 5, and the two essays (the seasons, DNA)
scored 2 of 2, the same as before. The CPU was shared with another job during that run, so its speed is not comparable
with the earlier quiet run; the paired runs above are the fair comparison.

| Quiz run | Questions prefill / decode tok/s | Essays prefill / decode tok/s | Card peak | Engine RAM peak |
|---|---|---|---|---|
| Earlier, quiet, stream off | 0.917 / 0.209 | 0.981 / 0.351 | 3822 MiB | 710 MB |
| Stream on, CPU shared | 0.365 / 0.078 | 0.356 / 0.125 | 3884 MiB | 616 MB |

## Where the time goes (profile, 2026-11-27)

`SN_PROF=1` in the environment makes the engine print `PROF` lines on stderr after each turn: time per kind of work,
how predictable the router is, the stages of the expert stream, and how the MTP drafts fared. It only measures; results
are unchanged (checked on the test packs with it on). All numbers below were taken while another job used about four
CPU cores, so they are real-world numbers; same prompt as the speed table, 17 answer tokens.

**Decode, per token, stream on (about 3.4 to 3.7 s):** routed experts 65 to 67%, router 9%, Mamba scan 9%, Mamba in/out
projections 9% (on the card), shared expert 3%, lm_head 1.4%, the rest under 1% each. Inside the expert part, filling the
pinned staging buffers on the CPU takes about 75% of the stream's time, queueing copies and kernels about 8%, waiting
for the card at the end of a layer about 17%, and waiting for a staging buffer to come free almost nothing. So the card
is not the bottleneck: getting the compressed experts into pinned memory is.

**One issue found and fixed:** on Windows the RAM expert cache was silently off. The engine computed the budget in a
32-bit number and 4096 MB came out as 0, while the startup line still said "on". It now works: about 56% of the experts
a decode token asks for come from RAM, and the engine's RAM use rises to the intended 4 to 5 GB. Results are identical
with or without it.

**Can the router be guessed ahead of time?** Of the 22 experts a layer picks for a decode token:

| Guess | Share of the picks it gets right |
|---|---|
| The same experts as the last token in that layer | 33 to 42% |
| The next layer's router run one layer early (on this layer's input) | 65 to 68% |
| The layer's most used experts so far | 18 to 29% |
| The first two together (about 37 experts) | 75 to 78% |

**Speculative decoding (MTP head, 3 drafts):** 80% of drafts accepted (first draft 100%, second 80%, third 75%). The
misses were ends of the answer (it drafted "." where the model wanted to stop) and a list order slip (" Earth" drafted
where the model wanted " Venus"). It is token-identical by design.

**Expand on the card or keep as dense compute?** Single-call times on the card versus one CPU lane (DTC = the exact
lookup kernels):

| Tensor | Card, plain float | Card, DTC | CPU DTC, 1 lane | Best home |
|---|---|---|---|---|
| Mamba in_proj 18560 x 4096 | 5.1 ms | 1.5 ms | 77 ms | resident on the card (used every token) |
| Mamba out_proj 4096 x 8192 | 1.8 ms | 1.5 ms | 72 ms | resident on the card |
| MoE fc1 / fc2 latent | 1.4 / 1.5 ms | 3.6 / 2.1 ms | 14 / 8.5 ms | resident on the card |
| Routed expert up / down | 0.5 / 0.8 ms | 0.7 / 1.5 ms | 4.3 / 8.0 ms | streamed compressed, computed on the card |

The plain-float card kernels are sometimes quicker but are not bit-identical, so the card uses DTC. Every tensor used
on every token belongs on the card for good; experts are each used rarely, so they are sent compressed when picked.

**Measured with the cache fix (real-world):**

| Run | Decode tok/s | Prefill tok/s | Engine RAM peak |
|---|---|---|---|
| Experts on the CPU lanes (`--no-gpu-experts`) | 0.183 | 0.382 | 4.8 GB |
| Expert stream on | 0.274 | 0.591 | 4.4 GB |
| Expert stream on, `--spec --draft-len 3` | 0.314 | 0.863 | 5.2 GB |

All three wrote the same answer.

**What would help next, most expected gain first:**

1. Turn speculative decoding on with the stream (`--spec --draft-len 3`): +15 to 33% decode measured, token-identical.
   It is a default change, so it waits for JD's go.
2. Start the next layer's experts early: run the next layer's router one layer ahead (65 to 68% right), and fill
   staging and the RAM cache for those experts while the card is still working on this layer. Expected 10 to 25%
   on decode; results unchanged, because the compute still uses the real picks.
3. Let the card copy straight from the RAM cache: keep the cache's buffers in pinned memory, so a cached expert needs
   no CPU copy at all. That targets the biggest cost above.
4. Router on the card with an exact fixed-order kernel (9 to 14% of decode). Needs about 370 MB of VRAM for the
   router weights, so the dense budget has to give some room.
5. Experts-only mode (`--gpu-mb 0`): cap the slot pool well below the full card. 300 slots decoded at 0.054 to
   0.060 tok/s against 0.030 with 1684 slots on the 4 GB card, which also drives the display.

A "hot" slot policy is available to try with `SN_XS_POLICY=hot` (keeps the most used experts in their slots). It raised
slot hits from 0 to about 270 to 370 at about 100 slots and from 0 to 2257 at 300 slots, with no clear speed change, so
the default stays least recently used.

## Every run on the real pack today

Same prompt for runs 1 to 14 ("List the eight planets of the solar system in order from the Sun, separated by commas.",
32 new tokens allowed, 12 threads, below-normal priority); every one of them answered "Mercury, Venus, Earth, Mars,
Jupiter, Saturn, Uranus, Neptune" (the 4-token warm-up stopped after "Venus"). Prefill and decode are pooled tok/s from
the engine's own STATS line; wall is the whole process including load; card peak is the whole card (the display and other
programs included); RAM is the engine's peak working set. Another job was using about four CPU cores during all of these,
so every timing here is real-world. "Budget issue" marks runs from before the RAM cache fix, when the 4096 MB cache was
silently off on Windows.

| # | Run | Path | RAM cache | Slots (policy) | Prefill tok/s | Decode tok/s | Wall | Card peak | Engine RAM peak | Conditions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | warm-up (4 tokens) | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.333 | 0.1459 | 148 s | 3446 MiB | 684 MB | real-world (another job on the CPU) |
| 2 | before-gpu3000 | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.301 | 0.1537 | 249 s | 3446 MiB | 686 MB | real-world (another job on the CPU) |
| 3 | after-gpu3000-stream | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.813 | 0.2534 | 132 s | 3749 MiB | 609 MB | real-world (another job on the CPU) |
| 4 | after-stream-only | CUDA expert stream only, dense on CPU | off (budget issue) | 1684 (least recently used) | 0.454 | 0.0300 | 690 s | 3701 MiB | 650 MB | real-world (another job on the CPU) |
| 5 | before-gpu3000-again | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.265 | 0.0918 | 345 s | 3445 MiB | 686 MB | real-world (another job on the CPU) |
| 6 | after-gpu3000-stream-again | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 115 (least recently used) | 0.487 | 0.1042 | 271 s | 3708 MiB | 608 MB | real-world (another job on the CPU) |
| 7 | streamonly-300-lru | CUDA expert stream only, dense on CPU | off (budget issue) | 300 (least recently used) | 0.620 | 0.0543 | 389 s | 1239 MiB | 649 MB | real-world (another job on the CPU) |
| 8 | gpu3000-hot-prof | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (hot, prototype) | 0.820 | 0.2496 | 132 s | 3899 MiB | 609 MB | real-world (another job on the CPU) |
| 9 | gpu3000-hot-prof-spec3 | CUDA dense 3000 MB on card + expert stream, MTP spec 3, accept 80% | off (budget issue) | 119 (hot, prototype) | 0.695 | 0.3312 | 122 s | 3899 MiB | 1127 MB | real-world (another job on the CPU) |
| 10 | streamonly-300-hot | CUDA expert stream only, dense on CPU | off (budget issue) | 300 (hot, prototype) | 0.551 | 0.0601 | 366 s | 1271 MiB | 648 MB | real-world (another job on the CPU) |
| 11 | gpu3000-lru-prof | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 101 (least recently used) | 0.846 | 0.2841 | 124 s | 3829 MiB | 609 MB | real-world (another job on the CPU) |
| 12 | gpu3000-cachefix-prof | CUDA dense 3000 MB on card + expert stream | on (fixed) | 101 (hot, prototype) | 0.591 | 0.2736 | 150 s | 3837 MiB | 4441 MB | real-world (another job on the CPU) |
| 13 | gpu3000-cachefix-prof-spec3 | CUDA dense 3000 MB on card + expert stream, MTP spec 3, accept 80% | on (fixed) | 119 (hot, prototype) | 0.863 | 0.3139 | 116 s | 3863 MiB | 5231 MB | real-world (another job on the CPU) |
| 14 | gpu3000-cachefix-nostream | CUDA dense 3000 MB on card, experts on CPU lanes | on (fixed) | - | 0.382 | 0.1829 | 213 s | 3617 MiB | 4783 MB | real-world (another job on the CPU) |
| 15 | quiz, 5 questions (16 new tokens max) | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.365 | 0.078 | quiz run total 3070 s | 3707 MiB | 615 MB | real-world (another job on the CPU) |
| 16 | quiz, 2 essays (320 new tokens max) | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.356 | 0.125 | (same run) | 3884 MiB | 616 MB | real-world (another job on the CPU) |

Real-pack test runs (not speed runs): the kernel checks took about 25 s each; the whole-model hash run took 729.7 s
(CPU, 12 threads), 620.4 s (stream only, 12 threads), 935.2 s (stream only, 4 threads), 1088.8 s (CPU, 4 threads),
249.8 s (dense 3000 MB) and 87.1 s (dense 3000 MB plus stream), with a CPU scan from another worker running at the time.
