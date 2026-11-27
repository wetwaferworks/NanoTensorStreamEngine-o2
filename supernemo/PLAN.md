# SuperNemo engine: plan and status (2026-11-27)

Goal (JD, 6:46 AM ET Nov 27): get Nemotron 3.5 Super VL 120B-A12B running in NTSE, talk to it
for real from the NTSE chat window. Pure C, zero mutexes, no downloads, local commits only.
Work lives in D:\scratch\ntse-o2-supervl-quant-2026-11-27 (branch scratch/supervl-quant-2026-11-27).

## The model, as the checkpoint says it is

| Part | Facts (config.json and the 43,079 tensor headers) |
|---|---|
| Backbone | 88 layers: 40 Mamba-2, 40 MoE, 8 attention. hidden 4096, vocab 131072, untied lm_head, eps 1e-5 |
| Mamba-2 | in_proj 18560x4096 = z 8192 + conv 10240 + dt 128; 128 heads x head_dim 64, state 128, 8 groups, conv width 4, conv channels 10240 = 8192 + 2x1024, norm 8192, out_proj 4096x8192 |
| Attention | 32 query heads, 2 KV heads, head_dim 128 (q 4096x4096, k and v 256x4096, o 4096x4096), no bias, NoPE like Little Nemo |
| MoE | 512 routed experts, top 22, 1 shared expert (5376), routed scaling 5.0, relu2, latent width 1024 (fc1_latent_proj 1024x4096 in, fc2_latent_proj 4096x1024 out), each expert up 2688x1024 and down 1024x2688, gate 512x4096 plus a float e_score_correction_bias |
| MTP head | language_model.mtp.*: one attention + one MoE layer; optional (speculative decoding), not needed for plain chat |
| Vision | RADIO ViT-H/16 (32 blocks, 1280 wide, 16 heads, MLP 5120, 7 registers, 3 CLS tokens, CPE up to 2048), pixel-shuffle downsample 0.5, mlp1 projector 5120 -> 20480 -> 4096 |
| Total | 124,453,395,976 weights; 232 GiB bf16 |

## What exists today and what does not

| Piece | State |
|---|---|
| Pack writer (nq2-quantize) | Now handles this model: PQ in the parallel path, raw fp16 for small and sensitive tensors, skip of the I64 tensor (commits 20b9614, 2885e39) |
| Pack reader for the engine (`src/sn_pack.*`) | Done and verified: 15 tensors of the real pack reproduce the quantizer's own relative L2 (0.30 to 0.32 packed, 0.0000 raw); unaligned partial expansion equals the slice of the full expansion |
| Pack preflight (`tests/sn_preflight.c`) | Done: checks every tensor of the language path and the vision path by name and element count against the geometry above |
| MoE block math (`src/sn_moe.*`) | Done, 15 fixture checks pass (routing rules, unbiased weights, ties, double-precision reference). The router tie and epsilon rules are from the published structure; the upstream reference is not on this machine, so confirm against a reference run |
| Mamba-2 and attention kernels | Exist in `littlenemo/src` (parameterised math: heads, head_dim, groups, state); the forward driver there is hard-coded to Nano dims |
| Tokenizer, protocol, GUI | Exist (Little Nemo tokenizer reads tokenizer.json; protocol v3 already has `MOE` lines and `moe=` READY key; GUI shows "SuperNemo engine not available yet") |
| SuperNemo forward driver (`src/sn_model.*`) | Done: geometry from config.json, fused PQ matvec straight from the pack, experts per lane; 35 stack checks on a tiny model (lane-count bit-identical, dense reference, exact reset replay). Real-pack speed unmeasured until the quantize publishes |
| Protocol v3 adapter (`src/sn_engine.c`, `supernemo.ini`) | Done: READY (`moe=512`, `media=image` when the pack has vision tensors), HW, PREFILL, LAYER/MOE/TICK telemetry, THINK/TOK split, STATS, MEM, STOP polling, `--below-normal`; chat template is the checkpoint's own ChatML (system block, `<think>` handling); images via MEDIA (stb_image decode, the processor's dynamic-resolution rule, torch-style antialiased bicubic) |
| RADIO vision tower + projector (`src/sn_vision.*`, `src/sn_image.*`) | Done: 32 blocks, CPE position resample, pixel shuffle v2, mlp1; matches an independent double-precision reference to 1e-6 relative on a tiny pack (non-square grids included) and is bit-identical on 1 and 5 lanes. Video and audio are not built |
| CUDA lane (`src/sn_cuda.*`) | Done for the dense tensors and lm_head: NVRTC-compiled PQ4 and fp16 matvec kernels reading the packed form straight from VRAM, loaded through the driver API at run time (no toolkit headers or host compiler needed to build; NVRTC from CUDA_PATH to run). `--gpu-mb N` in sn-engine fills the card with lm_head, then each layer's Mamba/attention mixers and the MoE latent + shared projections. GPU vs CPU kernel: 1.8e-7 relative on real tensors, same text on the tiny model. Routed experts stay on the CPU lanes: 20480 experts cannot live in 4 GB, and a miss is bound by reading the pack, not by arithmetic |

## Phases

1. **Text path on CPU from the pack** (done, see above; real-pack numbers pending): `sn_forward.c` with a geometry struct (from config.json, no hard-coded 4480s),
   per-layer fetch through `sn_expand` on the 12-lane pool, expert expansion on demand with an LRU of expanded experts,
   layer-major prefill, chat template from tokenizer_config. Gate: stack runs, logits finite, state reset bit-exact, thread count
   does not change a bit; then a prompt battery against the raw bf16 checkpoint on a few layers (per-layer relative error), and
   argmax agreement on short prompts as the honest quality number.
2. **Engine adapter** (done): `engine_gui`-style binary speaking protocol v3 (READY with moe=512, MOE telemetry lines), wired into the
   `[supernemo]` section of littlenemo.ini so the NTSE chat window can select it.
3. **Vision** (done for images): RADIO blocks (LayerNorm, qkv, GELU MLP), CPE position embedding resample, pixel shuffle, mlp1; image tokens into
   the stream at `<image>` positions.
4. **Speed**: CUDA expert kernels (PQ4 decode in registers as DTC does for dense), hot-expert residency, MTP speculative decode.

## Expected speed and memory, honestly

16 GB RAM and a 4 GB RTX 3050 Ti. The pack is about 46 GB, so it streams from NVMe through the page cache. Per token the router
touches 22 experts in each of 40 MoE layers: 880 expert pairs, about 5.5 M weights each, plus the dense mixers. A rough bound is
1 to 3 s per token on the CPU lane once warm; first-token latency is dominated by reading. This is a "talk to it for real" speed,
not a fast one.

## Quality, honestly

The recipe is the repository's documented CUDA pack (per-tensor books, RMS frame, PQ d4). Weight relative L2 is about 0.30
per packed tensor. On Sigma-3B the same recipe matched the raw model's top token on 9 of 32 positions, and the project's own
notes call that class of pack "not yet a quality artifact". Expect a SuperNemo pack that runs and produces text but is clearly
degraded against bf16. Raising quality needs a higher-rate recipe (an experts-only book per layer, smaller PQ subvectors, or
residual PQ), which is a converter research item and needs JD's go because it multiplies pack size.
