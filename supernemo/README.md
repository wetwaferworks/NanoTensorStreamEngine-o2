# SuperNemo engine (Nemotron 3.5 Super VL 120B-A12B on NTSE)

Pure C99, zero mutexes (the Little Nemo pool), reads the NQ2 pack written by `nq2-quantize` (per-tensor books, PQ d4).
Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. See `PLAN.md` for the geometry, status and
honest notes on speed and quality.

## Build and test

    build.bat                       (MinGW-w64 gcc; no CUDA toolkit needed to build)

    test_sn_moe.exe                                          MoE math fixtures
    test_sn_model.exe  PREFIX SHAPES CONFIG                  stack vs dense reference, lanes, reset, batched prefill (tiny pack)
    test_sn_vision.exe PREFIX SHAPES CONFIG                  RADIO + mlp1 vs independent double reference (tiny vision pack)
    test_sn_image.exe  [PNG]                                 target-grid rule, resize invariants, decode
    test_sn_cuda.exe   PREFIX SHAPES [budget_mb]             GPU kernels vs the CPU kernel on every tensor they can take
    sn_preflight.exe   PREFIX SHAPES CONFIG                  does the pack carry every tensor the engine needs, with the right sizes
    sn_packcheck.exe   ...                                   pack vs source shards (reconstruction error, fused matvec)

The tiny packs come from `sn_mktiny.exe DIR` (language) and `sn_mkvis.exe DIR` (vision), then `nq2-quantize` with
`--books per-tensor --norm rms --pq 4 --raw-names embeddings.weight,lm_head,gate.weight,pos_embed,mlp1.,embedder,cls_token --raw-small 4096`.

## Run

    sn-engine.exe --pack D:\models\nemotron-3.5-super-vl-nanotensor\nemotron-3.5-super-vl ^
        --config D:\models\nemotron-3.5-super-vl-120b-a12b-bf16\original\config.json ^
        --tokenizer D:\models\nemotron-3.5-super-vl-120b-a12b-bf16\original\tokenizer.json ^
        [--shapes TSV] [--threads N] [--max-pos 4096] [--max-new 256] [--think] [--below-normal]
        [--gpu-mb 3000] [--max-patches 1024] [--no-vision] [--ask "one-shot question"]

Without `--ask` it speaks NTSE chat protocol v3 on stdin/stdout (`supernemo.ini` has the GUI line). `--ask` prints the reply
and the measured speed and exits.

* Prompts are prefilled in layer-major batches of 128 tokens: each weight matrix is expanded once per batch and each routed
  expert once for every token that chose it.
* Decode uses the fused PQ kernels straight from the pack (no matrix is rebuilt), routed experts split across the lanes.
* `--gpu-mb` keeps lm_head and the dense mixers / latent / shared projections resident in VRAM in packed form (NVRTC from
  `CUDA_PATH` is needed at run time). Routed experts stay on the CPU lanes.
* Images: PNG, JPEG, BMP, GIF via MEDIA. The checkpoint's dynamic-resolution rule picks the tile; `--max-patches` caps the
  size this machine will chew on (default 1024 patches = 256 image tokens). Video and audio are not built.

## Speed features (DTC, hot-expert cache, speculative decoding)

All three are exact or deterministic, pure C and lock-free, and each has an off switch.

| Feature | Flag | Ini key (`[supernemo-tuning]`) | Default |
|---|---|---|---|
| Dense Tensor Compute straight on the packed PQ blocks (CPU, and GPU for tall tensors and cached experts) | `--no-dtc`, `--dtc-force` | `dtc = off/auto/force` | auto |
| Hot-expert cache: RAM tier (resident packed experts) and VRAM tier (mirror of the hottest) | `--cache-mb N`, `--gpu-cache-mb N` | `cache_mb`, `gpu_cache_mb` | 4096 / 0 |
| Speculative decoding with the checkpoint's MTP head (greedy, verify in one fused pass) | `--spec`, `--no-spec`, `--draft-len K` (1..7) | `spec`, `draft_len` | off, 3 |

`--ini FILE` reads the `[supernemo-tuning]` section; command-line flags win. The cache budgets are separate from `--gpu-mb`
(dense tensors); on the 4 GB card `--gpu-mb 3000` leaves roughly 0.8 GB, so keep `--gpu-cache-mb` at or below about 600.

What is guaranteed (tests: `test_sn_pqk`, `test_sn_spec`, `test_sn_cache`, `test_sn_cuda`):
- DTC equals the direct kernels to float noise (about 1e-6 relative; bounds 3e-2 per layer, 5e-2 on logits, argmax must match).
  The GPU DTC kernels are bit-identical to the CPU DTC, and every DTC result is bit-identical across lane counts, row splits and
  file versus resident sources.
- The cache never changes a result: the logits hash is identical with it off, roomy, evicting, single-lane and with the VRAM tier.
  Admission and eviction are decided on the main thread from token counts between pool jobs, so the state is deterministic.
- Speculative decoding is token-identical to plain greedy by construction (exact verification, rollback of Mamba/KV state).
  It only pays when the draft acceptance rate is high enough: verifying k+1 tokens reads the union of their experts. Measure on
  the real pack (`SPEC` and `CACHE` lines appear next to `STATS`) before leaving it on.
