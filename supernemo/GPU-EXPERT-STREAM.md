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
is slow on this card (an open issue being looked at: the likely cause is the card running nearly full next to the display).
Use it only to experiment; the default (`--gpu-mb` with the stream on) is the one to use.

The five short quiz questions (square root of 144, capital of Australia, symbol for helium, year of the Declaration of
Independence, largest planet) were run again with the stream on: 5 of 5 right, same answers as before. The CPU was shared
with another job during that run, so its speed (prefill 0.365 tok/s, decode 0.078 tok/s pooled, card peak 3707 MiB, engine
RAM peak 615 MB) is not comparable with the earlier quiet run (prefill 0.917, decode 0.209, 3685 MiB, 708 MB); the paired
runs above are the fair comparison.
