# SuperNemo: a 120B-parameter model on a 4 GB card

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.

SuperNemo is NVIDIA's Nemotron 3.5 Super VL (120B-A12B): a hybrid Mamba, attention and mixture-of-experts model with a vision tower. Each of its MoE layers holds 512 small expert networks and each token uses 22 of them, so one token touches about 880 experts. The engine in `supernemo/` is pure C99 with zero mutexes, reads the packed model from disk, and speaks the same chat protocol as Little Nemo. This page is the plain-language overview. The detailed guides are [supernemo/README.md](../supernemo/README.md) and [supernemo/GPU-EXPERT-STREAM.md](../supernemo/GPU-EXPERT-STREAM.md); how much intelligence survives is in [INTELLIGENCE-RETENTION.md](INTELLIGENCE-RETENTION.md).

## The pieces

| piece | what it does | default |
|---|---|---|
| Dense Tensor Compute (DTC) | multiplies straight from the compressed blocks, so a weight matrix is never rebuilt in memory | on |
| Expert cache | keeps recently used experts in RAM (4096 MB budget), in pinned memory laid out like a card slot so one copy sends an expert to the card; an optional VRAM tier mirrors the hottest. Which experts stay is decided on the main thread from token counts, so the cache never changes an answer | on |
| GPU expert stream | missing experts go to the card in their compressed form (about 1.9 MB each), are expanded and computed there, and their slots are reused by the next token | on when `--gpu-mb` is above 0 |
| MTP speculative decoding | the checkpoint's own multi-token head drafts 3 tokens, and the main pass checks them in one go; output is identical to plain greedy decoding. About 80 percent of drafts were accepted | on (`--no-spec` turns it off) |
| Early expert fetch (EEF) | while the card works on one layer, the next layer's router is run early and its likely experts are sent ahead | on, mode 4 |

JD's rule: a speed feature that is verified to give identical results is on by default, and each one has an off switch.

## Early expert fetch modes

`SN_XS_PFMODE` picks how early fetch works. The router is right about two times in three, and a wrong guess costs a copy, never a wrong answer.

| mode | what it does |
|---|---|
| 1 | guesses are read from the pack on the main thread and not kept (the first version) |
| 2 | the same reads, but the guesses are kept in the RAM cache |
| 3 | only guesses already in the RAM cache are sent early; the disk is never read for a guess |
| 4 (default) | mode 3, plus the other guesses are read from the pack on a helper thread while the main thread computes, and become cache entries at the next planning step |

The first version made decoding slower than no early fetch at all, because it read from the disk on the main thread and starved the cache. Mode 4 fixes the cache (5,600 hits against 4,812 with early fetch off) and overlaps the reads.

## Speed, real-world

Median decode speed over 17 generated tokens, MTP on, `--gpu-mb 3000`, 8 threads at below-normal priority, same prompt for every run. Other jobs were using the CPU and the card, and one configuration moved by up to 60 percent from load alone (0.150 to 0.251 tok/s for early fetch off), so read the ratios.

| early fetch | runs | median decode tok/s | RAM cache hits | pack reads | early sends (wrong) |
|---|---|---|---|---|---|
| off | 6 | 0.189 | 4,812 | 17,975 | 0 |
| **mode 4 (default)** | 6 | 0.183 | 5,600 | 18,249 | 2,043 (258) |
| mode 3 | 4 | 0.171 | 4,812 | 17,975 | 2,021 (239) |
| mode 2 | 2 | 0.161 | 5,544 | 19,209 | 5,148 (1,372) |
| mode 1 | 2 | 0.165 | 3,817 | 20,145 | 5,148 (1,372) |

Honest reading: modes 1 to 3 are clearly slower, and mode 4 ties with early fetch off within the noise (0.183 against 0.189 median); it does not beat it by a margin on this disk-bound machine. It stays on by default, as JD decided, while profiling looks for the next gain. Draft lengths of 1, 2 and 3 tokens all landed inside the noise, so the default stays 3.

JD agreed that about 0.1 to 0.3 tokens per second is realistic for this model on this card. The 10 and 30 tokens per second goals are for Little Nemo.

## Same answers, checked

| check | result |
|---|---|
| whole-model logits hash on the real pack, CPU 12 threads, GPU stream 12 threads, stream 4 threads, CPU 4 threads | all `b0c32d2989f86f69` |
| dense 3000 MB lane, with and without the stream | `5dacc350426a9506` (its own float kernels differ from the CPU lanes, known and unchanged) |
| tiny test packs, all four early fetch modes, 3-slot and 5-slot runs that force eviction | identical hashes across 1, 4 and 8 threads (116 and 166 checks, 0 failed) |
| quiz subset: square root, capital of Australia, helium symbol, 1776, largest planet | 5 of 5 on the final build |

The hash comparison shows the speed features do not change the model's output. It compares the pack with itself on another path; it does not compare against the raw checkpoint.

## Run it

```
supernemo\build.bat
sn-engine.exe --pack <pack-dir> --config <checkpoint>\config.json --tokenizer <checkpoint>\tokenizer.json ^
    --gpu-mb 3000 --below-normal --ask "your question"
```

Without `--ask` it speaks chat protocol v3 on stdin and stdout. Useful switches: `--no-spec`, `--no-prefetch`, `--no-pinned-cache`, `--no-gpu-experts`, `--cache-mb N`, `--draft-len K`.

## Limits

- Under 0.4 tokens per second on a 4 GB card. This is a correctness and residency demonstration, not a fast chat.
- The disk is the bottleneck: reading experts from the pack is the slowest step.
- Images work; video and audio are not built.
- Intelligence against the raw checkpoint is pending (the raw model is far larger than this machine).
