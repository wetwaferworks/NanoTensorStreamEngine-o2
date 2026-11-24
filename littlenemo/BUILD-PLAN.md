# Little Nemo: build plan

Goal (JD, 2026-11-24): a downloadable tool JD can talk to, built as big as we can
make it here. Target model: NVIDIA-Nemotron-Nano-9B-v2 (Nemotron-H hybrid:
27 Mamba-2 + 25 MLP + 4 attention layers, hidden 4480, vocab 131072).
Architecture study: ../NEMOTRON-NANO-9B-V2-ARCH-2026-11-24.md (build-ready).

Language law (JD, restated 2026-11-24): everything runtime is strictly C.
The only Python allowed is a GUI front end that calls the C engine.

Lanes: PCHIE integration is Elfie + JD here. Cortana: CUDA and JD's local
hardware only.

## Phases, each gated before the next begins

- P0 Checkpoint inspector (nemo inspect): hardened safetensors header reader,
  prints the real tensor inventory of a shard. Gate: inventory of Nano shard 1
  matches the names and shapes the architecture study predicts.
- P1 Weight path: whole-checkpoint walk via model.safetensors.index.json;
  raw bf16 load; NanoQuant pack of the Nano with nq2-quantize; errprobe map of
  the new pack. Gate: pack checksum recorded, per-tensor errors reported.
- P2 Block math, one block at a time, checked against NVIDIA's reference
  semantics from modeling_nemotron_h.py: RMSNorm, Mamba-2 (in_proj split,
  causal conv1d, SSM recurrent step, gated norm, out_proj), MLP (relu2),
  attention (GQA, NoPE). Gate: prefill chunked scan reproduces the recurrent
  step on a synthetic state (the harness self-check from the study).
- P3 Full forward: prefill + greedy decode, fp64 sampler state, deterministic.
  Gate: the PCHIE reasoning battery protocol (raw vs packed vs packed+NQDC
  trajectories, first-divergence positions, answer agreement), every run
  hash-chained: prompt hash, pack checksum, step log, output hash.
- P4 Chat loop: tokenizer in C (tokenizer.json from the checkpoint), prompt
  in, tokens out, streaming token by token so conversation feels live at
  whatever rate the hardware gives; the true rate is always measured and
  shown, never rounded up.
- P5 Interaction layer (JD's direction, 2026-11-24 ~10:33 EST): a chat GUI
  in the spirit of the Llama front ends (Python/Tkinter, the one allowed
  Python piece, calling the C engine). It must show three things: the
  conversation, the model's THINKING (reasoning trace in its own pane),
  and a stats readout (tokens, tokens/sec, ms per token, elapsed).
  Engine and GUI speak PROTOCOL.md (line protocol over pipes or TCP), so
  the same GUI works two ways: against an engine on JD's own machine from
  the download, or connected to the engine running on Elfie's computer.
- P6 Package: a single download for JD, in the established delivery
  naming discipline.

## Discipline

- One pen in the source. Deterministic consumption, serialized order.
- Every loader treats file headers as untrusted (the 2026-11-24 hardening
  patterns: header-length guard, span checks, geometry validation).
- No number is claimed before its receipt exists.
