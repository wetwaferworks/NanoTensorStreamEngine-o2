# Little Nemo

Little Nemo is a pure C inference engine for NVIDIA's
Nemotron-Nano-9B-v2, built alongside the Nano Quant Engine. It
exists for one question, in JD's formulation: we do not get to
keep the original weights at these pack rates, so the packed
model has to prove that its output is still the same reasoning.
Little Nemo is the raw-weight reference that the packed model
will be measured against, and the conversation front end JD
asked for: talk to the model in real time, watch it think, and
see the machine work while it does.

Source: littlenemo/ in this repo. Runtime is C, full stop. The
one Python file is the GUI front end; it renders and relays, and
does no model work.

## The model

Nemotron-Nano-9B-v2 is a 56-layer hybrid: 27 Mamba-2 layers, 25
MLP layers, 4 attention layers (at indices 14, 21, 30, 39),
hidden width 4480, vocabulary 131072 with an untied lm_head.
Every block is the same shape: RMSNorm, one mixer, residual add.
The Mamba-2 mixer is the interesting one (a gated, grouped
RMSNorm after the state-space step, norm applied after the gate
in this checkpoint), and its exact decode math is written down
in littlenemo/BLOCK-MATH-SPEC.md, derived from NVIDIA's
reference implementation and verified against hand-computed
fixtures.

## What is verified (2026-11-24)

- Checkpoint inspector and loader: the real checkpoint's tensor
  inventory matches the architecture study exactly. Loader
  fingerprints on real tensors: layer 0 A_log FNV-1a
  f73e4acba693588f; layer 21 q_proj RMS 0.017538, FNV-1a
  66a6639137e13b79.
- Kernels, each against hand-computed expectations: Mamba-2
  convolution, SSM step, and gated norm (19/19 checks); grouped
  query attention decode, including a closed-form proof of the
  head-to-KV-group mapping (19/19); byte-level BPE tokenizer
  over tokenizer.json (27/27, including malformed-input
  refusals).
- The assembled forward pass: three tokens through the first 15
  layers of the real checkpoint (all three block types,
  attention included). Hidden state finite and nonzero (RMS
  0.520647), bit-exact deterministic across a state reset, and
  token-sensitive. 7/7 checks, re-verified after the hardening
  pass below with identical results.
- A two-round code review with a completed fix pass: span and
  overflow guards at index time, 64-bit file seeks throughout
  (a 32-bit long would silently misread the 4.9 GB shards on
  Windows), geometry validation at the kernel doors, decode
  error channels instead of silent failures, and NaN that
  propagates instead of being zeroed away. Five crafted hostile
  checkpoints are refused cleanly; the full suite is silent
  under ASan and UBSan. The review also caught a real bug of
  ours: the bf16 converter canonicalized NaN to a bit pattern
  that is a finite 2^125 in bf16, not a NaN at all. Fixed, with
  a regression check standing guard.
- The engine binary and its line protocol (littlenemo/
  PROTOCOL.md): greedy streaming generation with per-layer
  telemetry, per-token timing, and a think/answer split, built
  and exercised against the truncated 15-layer stack. The
  protocol run reaches exactly the boundary of the partial
  checkpoint on this bench (the final norm and lm_head live in
  the last shard) and reports it as an error rather than
  inventing output.
- A second outside review (ChatGPT (Gizmo), 2026-11-24) found
  two things, both fixed the same day. First, geometry is now
  validated whole before inference begins: every tensor the
  steps will fetch is checked for presence, shape, and dtype
  against the architecture, and the engine prints READY only
  for a checkpoint that passes. Second, state commitment is
  coherent: every step preflights its tensors before the first
  layer's state moves, so a missing tensor is a preparation
  failure that commits nothing, and any failure mid-commit
  invalidates the state until reset. The coherence suite proves
  the refused step bit-identical to never attempting it. His
  formulation is the standing rule: preparation can be
  concurrent; state commitment must be coherent.
- A third outside review (Archie (Argon), 2026-11-24) caught
  what the earlier rounds and our own suites had missed in this
  engine. The shipped SIMD verification was scalar against
  scalar: the suite never called the AVX2 kernel, and the
  engine's entry point never dispatched to it, so the kernel
  was unreachable in single-thread mode. Both are fixed. The
  suite now calls the kernel directly on synthetic data,
  checks the dispatch itself, and compares a multi-lane SIMD
  forward against the 1-lane run: 12/12 checks, kernel drift
  against scalar exactly zero, multi-lane bit-identical, and
  the forward suite still reproduces the golden RMS 0.520647.
  The same review completed the 64-bit offset work (the define
  now stands in the loader, forward, and tokenizer sources and
  in all four build scripts) and hardened the tokenizer's ID
  validation and pair maps. In the sibling pipeline engine the
  same night, the Triforce mode-2 staging buffer was completed
  with no change to any on-disk format, and the frozen
  baseline pack decodes bit-identically under the new build.

## What is not done yet, said plainly

- The full checkpoint is downloaded and validated, and the
  engine prints READY on it. A three-question battery (the
  capitals of France, Japan, and Ohio, asked in a raw
  question/answer scaffold) stands at two passed, France and
  Japan. The third question has not yet run on the post-review
  build, and nothing broader about generation quality is
  claimed until it and the behavioral battery below have run.
- Chat-template wrapping arrives with tokenizer_config.json;
  prompts are currently encoded as-is.
- The NanoQuant packed weight source plugs into the same fetch
  points the raw source uses. When it lands, the behavioral
  battery runs the same prompts three ways (raw, packed, packed
  with dictionary compute) and scores token-for-token agreement,
  first divergence, and final-answer agreement, hash-chained
  per PCHIE discipline.

## The interface

The GUI (littlenemo/gui/nemo_chat.py) is a dark chat window in
the style JD specified: conversation on the left, the model's
thinking on the right, a 56-cell strip that lights each layer as
the pass moves through it (Mamba blue, MLP gray, attention
orange), and a readout line carrying the live numbers... layer
and type, next layer, active layers, predict-ahead depth,
tokens, tokens per second, milliseconds per token, estimated
memory throughput, elapsed time. Its protocol client is verified
headlessly, 11/11 checks against a mock engine.

## Credits

Implementation: Elfie (E.L.F.I.E.). Direction and the
interaction design: JD (David Rittenhouse). The CUDA and
local-hardware lane belongs to Cortana; the harness integration
(PCHIE) is Elfie and JD's work. Reviewer of the 2026-11-24
pass: Archie (Argon), the Google Gemini Argon model, the Bit
Master and the Bit Masker, whose review caught five P0 faults
across this engine and its benchmates, including the false
SIMD verification above.
