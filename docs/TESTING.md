# Testing

Every suite in the tree, what it needs, and how to run it. Suites exit nonzero on failure. Weights are never in the repository, so suites that need a real checkpoint or pack take its path on the command line.

## Build first

| Script | Builds |
|---|---|
| `build.bat` / `build.sh` (repo root) | all 23 tools in `tools/` and Little Nemo with its suites; the threaded tools need a POSIX-thread MinGW on Windows |
| `build-all.bat` / `build-all.sh` | the one-click name for the same build |
| `build-experiments.bat` / `build-experiments.sh` | the research-tool subset |
| `littlenemo/build.bat` / `build.sh` | `nemo-engine`, `nemo-inspect`, 13 suites and the fixture builder `mk_fixtures` into `littlenemo/bin/` |
| `accel/build-accel.bat` / `build-accel.sh` | the accel suites, `ntse-accel-bench` and the Little Nemo accel smoke into `accel/bin/` (no GPU SDK needed; vendor runtimes load at run time, and with no GPU the suites run on the CPU reference) |
| `littlenemo/gui/native/build.bat` / `build.sh` | the C scaler library and `test_nemo_scale` |
| `parakeet/build.bat` | `parakeet-c.exe` and `parakeet-tests.exe` |

## Little Nemo C suites (`littlenemo/bin/`)

| Suite | Needs | Checks |
|---|---|---|
| `test_mamba` | nothing | Mamba-2 conv, SSM step and gated norm against hand-computed values |
| `test_attn` | nothing | GQA attention decode with closed-form softmax weights |
| `test_tokenizer <fixture.json>` | `tests/fixture_tokenizer.json` | JSON reader and BPE against a hand-built fixture, malformed-input refusals |
| `test_tokenizer_split <tokenizer.json>` | the real `tokenizer.json` | the pre-tokenizer split and `ignore_merges` path |
| `test_guards` | nothing | geometry guards at the mamba entry points, the gated norm's anti-hang floor, attention NaN propagation |
| `test_loader <checkpoint_dir>` | a checkpoint dir | synthetic loader checks, then fingerprints on real weights (the regression anchors) |
| `test_forward <checkpoint_dir>` | a checkpoint dir | truncated 15-layer forward: finite, reset-reproducible, token-sensitive |
| `test_hostile <refuse\|i32> <dir>` | crafted fixture dirs (`mk_fixtures` writes them) | hostile checkpoints are refused cleanly |
| `test_coherence <checkpoint_dir> <dtype_dir>` | a partial checkpoint and a wrong-dtype dir | whole-geometry validation and state-commit coherence |
| `test_par <checkpoint_dir> [lanes] [layers]` | a checkpoint dir | multi-lane forward is memcmp-identical to the single-lane reference |
| `test_simd <checkpoint_dir> [layers]` | a checkpoint dir | the opt-in AVX2 path: deterministic, faithful to scalar, dispatch proven by entry counters |
| `test_prefill <checkpoint_dir> [layers]` | a checkpoint dir | layer-major prefill is bit-identical to token-by-token stepping, across a chunk boundary |
| `test_tokeneval <fixture.json>` | `tests/fixture_tokenizer.json` | tokenizer-based eval scorer (top-1, top-5, first divergence, perplexity) and fixture encode/decode; no checkpoint |

```
cd littlenemo
build.bat
bin\test_mamba.exe && bin\test_attn.exe && bin\test_guards.exe
bin\test_tokenizer.exe tests\fixture_tokenizer.json
bin\test_loader.exe <checkpoint_dir>
```

### Fixture-driven suites

`test_hostile`, `test_coherence` and `test_prefill` used to be skipped on a machine without hand-made fixtures. `mk_fixtures <checkpoint_dir> <out_dir>` (pure C, built by `build.bat`) now writes everything they need from the checkpoint you already have; no weights are copied or downloaded, and the output belongs outside the repository.

| Fixture | What it is | Used by |
|---|---|---|
| `hostile/refuse-*` (ten files) | tiny crafted safetensors: short header, zero header, huge header, truncated name, negative shape, size wrap, reversed offsets, span mismatch, short payload, unknown dtype | `test_hostile refuse` (each must be refused with a named reason) |
| `hostile/i32` | an int32 tensor where a float tensor is required | `test_hostile i32` |
| `dtype/model.safetensors` | a **sparse** file (1.2 GB long, almost nothing written) that declares the embeddings as bf16 and layer 0's norm as int32, with no model data | `test_coherence` (whole-geometry validation) |
| `partial/` | hard links (symlinks off Windows) to the first two shards of the checkpoint: layers 0 to 32 only | `test_coherence` (state-commit coherence on an incomplete checkpoint) |

`littlenemo/tests/run-fixture-suites.bat <checkpoint_dir> [fixture_dir]` (`.sh` beside it) builds the fixtures, runs `test_hostile` eleven times, `test_coherence` and `test_prefill`, and ends `fixture suites: ALL PASS` or `FAILURES`. Fixtures made from a private checkpoint stay private: keep the fixture directory out of git.

## NTSE tool tests (repo root)

| Test | Run | Checks |
|---|---|---|
| `nq2-dc-adv` | `nq2-dc-adv` (no arguments; built by `build.bat`) | the dictionary-compute adversarial suite: 23 HOLD, 0 BREAK, with FNV anchors; ends `suite done.` |
| `tools/test-dc-hardening.sh` | `sh tools/test-dc-hardening.sh` | `nq2-dc` refuses a NaN book, a shard whose span contradicts its shape, and other malformed input with a nonzero exit and a named reason |
| `tools/test-streamtest-truncation.sh` | `sh tools/test-streamtest-truncation.sh` | `nq2-streamtest` treats an incomplete pack as a hard failure and never publishes an accumulator for it (sequential and one pipelined configuration) |
| pinned anchors | see [INFERENCE-TOOLS.md](INFERENCE-TOOLS.md#pinned-anchors) | the engines' reference outputs on the frozen baseline pack |

## Accel suites (`accel/bin/`)

| Suite | Run | Checks |
|---|---|---|
| `test_accel_guard` | `test_accel_guard` | hostile inputs, the budget, and `NTSE_ACCEL_DISABLE=1` forcing the CPU lane; any machine |
| `test_accel_kernels` | `test_accel_kernels [--lane cuda\|igpu\|gpu-cl\|all] [--big]` | per-kernel correctness of every present lane against the CPU reference, plus repeated-run determinism |
| `test_accel_stream` | `test_accel_stream [--lane L]` | the GPU side of the Tensor Handshake: 1, 2 and 3 pinned slots, inline and with the stager thread |
| `test_accel_real <v1.nq2> <v1.nq2book> <shard> [...] [--match S]` | needs the real pack and shards | the lanes on real packed tensors |
| `test_accel_linear` | `test_accel_linear [--lane L]` | the bf16 projection kernel Little Nemo routes to |
| `test_accel_plan` | `test_accel_plan [--lane L]` | the prepared stream plan: every (op, slots 1 to 4, batch 1 and 3, pinned and pageable) run bit-identical to the resident path and the CPU |
| `test_nemo_forward_accel <checkpoint-dir> [--lane L]` | a checkpoint dir | Little Nemo's truncated forward with projections on a lane, against the CPU run and the frozen anchor |
| `test_nemo_gpu_layers <ckpt> <pack.nq2> [tokens] [layers] [dtc\|full\|all]` | CUDA, an unpacked checkpoint of the pack and the pack | per-layer residuals and logits of each CUDA lane against the CPU reference, with the argmax: DTC under 3e-2, FULL under 1e-3, logits under 5e-2 (reasoning in the source); `NEMO_GPU_GEMV_FULL=exact` runs the exact FULL path |
| `test_nemo_prefill <ckpt> <pack.nq2> [tokens]` | CUDA, checkpoint and pack | batched prefill equals stepped decode bit for bit, DTC and FULL |
| `test_nemo_hostsplit <ckpt> <pack.nq2> [steps]` | CUDA, checkpoint and pack | the host lanes (`--host-threads`) give bit-identical confidence at 1 to 8 lanes |
| `test_nemo_tokpipe <tokenizer.json>` | the real tokenizer | every safe cut gives the one-shot ids; the pipelined encoder matches over many lengths |
| `test_nemo_pack_ident` | nothing | pack identity checks of the manifest |

Bench tables and what the numbers mean: [ACCEL.md](ACCEL.md). The CUDA lane's per-layer verification (`test_nemo_gpu_layers`) is covered in [ACCEL-CUDA.md](ACCEL-CUDA.md#verification).

## GUI suites (Python, `littlenemo/tests/`)

Run each from `littlenemo/tests`, for example `python test_gui_protocol.py`. They use the fixture engines (`fake_engine.py`, `mock_engine.py`), need no weights, and exit nonzero on failure.

| Suite | Checks |
|---|---|
| `test_gui_protocol.py` | protocol v2 and v3 parsing, escapes, percent-encoding, size limits |
| `test_gui_client.py` | the engine client over pipes and TCP, send gating |
| `test_gui_media.py` | attachment inspection: limits, extension versus leading bytes, video frames |
| `test_gui_stats.py` | the stats model and the thinking-line headline |
| `test_gui_settings.py` | INI read and write, comment preservation, validation, the CUDA options (blank passes no flag, flags reach only a `cuda` engine) |
| `test_gui_about.py` | the About and credits text, and that it agrees with the README and `docs/CREDITS.md` (order and wording) |
| `test_gui_icons.py` | icon packs, manifests, mascots and theme packs; **needs the C scaler built** (it fails with `KeyError: 32` in "pack assets resolve" otherwise) |
| `test_gui_scale.py` | the ctypes scaler front end |
| `test_gui_ui_smoke.py` | the Tk window, driven through `--demo smoke`, `unavailable`, `resize` and `sizes` (needs a display) |

`python gui/nemo_chat.py --demo sizes --shots DIR` opens the window at several sizes (minimum, 1003x701, 1280x720, 1366x768, 1600x900, maximized, and 1000x700 and 900x600 forced below the minimum) with the stats panel open and a three-line message, and checks that the composer stays clear of the stats panel and the window edge and that the send button is whole; it writes `30-size-*.png`.

`python gui/nemo_chat.py --demo composer --shots DIR` (needs a display) checks the multi-line input bar (placeholder on the first line, growth to six lines, scrolling, word wrap), that the About and Settings dialogs fit on the screen, and writes the screenshots.

```
cd littlenemo/gui/native && build.bat && test_nemo_scale.exe && cd ../../tests
python test_gui_icons.py
```

`littlenemo/gui/native/test_nemo_scale` is the C test for the scaler (`ALL PASS`).

## Quantize dialog (Python, `tools/`)

| Suite | Run | Checks |
|---|---|---|
| `tools/test_nq2_gui.py` | `python tools/test_nq2_gui.py` | the option rules and command line, progress parsing, the job runner (a fake converter: success, failure, Stop, missing exe), settings kept outside the repo; **with a built `nq2-quantize`** (beside the script, in the repo root, on PATH or in `NQ2_QUANTIZE`) it also quantizes a small synthetic safetensors at 1, 6 and 12 threads in three recipes and requires the `.nq2`, `.nq2book` and `.nq2manifest` to be hash-identical, then drives the dialog through one run (needs a display) |

## Parakeet suites

| Suite | Run | Checks |
|---|---|---|
| `parakeet-tests.exe <model-dir>` | `bin\parakeet-tests.exe %PARAKEET_MODEL%` | 16 unit tests (FFT, GEMM at 1 and 8 threads, pool, resampler, WAV, mel, SPM, checkpoint dims) |
| `parakeet/tests/stdio_smoke.py` | `python parakeet\tests\stdio_smoke.py` | READY, PING, FILE, START and STOP, QUIT over `--stdio` (needs the model and its `samples\`) |
| `parakeet/tests/gui_dictation_test.py [wav] [ref.txt]` | `python parakeet\tests\gui_dictation_test.py <wav> <ref.txt>` | the mic button end to end, with a reference WAV replayed as the microphone (needs a display, the model, and a UTF-8 reference text without a byte-order mark) |

## SuperNemo suites (`supernemo/`, built by `supernemo\build.bat`)

The suites run on tiny random-weight packs written by `sn_mktiny` and `nq2-quantize` (no checkpoint is needed or shipped);
`test_sn_pqk` and `test_sn_cuda` also take a few real-shaped tensors written by `nq2-quantize` from any checkpoint.

| Suite | Checks |
|---|---|
| `test_sn_moe`, `test_sn_image` | the MoE block math and the image preprocessing |
| `test_sn_pqk PREFIX SHAPES` | the packed matrix kernels: multi-vector direct equals single-vector bit for bit, DTC independent of vector count, row split and file or RAM source, DTC within float noise of direct |
| `test_sn_cuda PREFIX SHAPES` | the GPU DTC kernels are bit-identical to the CPU DTC; the VRAM cache tier uploads and drops cleanly |
| `test_sn_model PREFIX SHAPES CONFIG` | the forward pass: deterministic, reset-reproducible, identical at 1 to 12 lanes |
| `test_sn_spec PREFIX SHAPES CONFIG` | DTC bounds (3e-2 per layer, 5e-2 on logits, argmax equal), fused steps equal single steps, rollback, speculative output token-identical to greedy |
| `test_sn_cache PREFIX SHAPES CONFIG` | the logits hash is identical with the expert cache off, roomy, evicting, single-lane and with the VRAM tier |
| `test_sn_vision PREFIX SHAPES CONFIG` | the RADIO tower against a double-precision reference |
| `sn_packcheck`, `sn_preflight` | a finished pack: every block readable and in range; geometry and tensor coverage against the checkpoint |

## Little Nemo tokenizer-based eval

Quality on a fixed passage of text, not the top-1 of one diffusion denoiser step: [LITTLE-NEMO-TOKENEVAL.md](LITTLE-NEMO-TOKENEVAL.md). The unit suite needs no checkpoint:

    bin\test_tokeneval.exe tests\fixture_tokenizer.json

A tokenizer-only gold-id check and a teacher-forced live run need a local official tokenizer.json (and, for the live path only, a checkpoint). No model path is stored in the tree. Live runs default to one thread.

## Little Nemo quiz

The 10-question smoke quiz (raw 10 of 10 at about 0.07 tokens/s; compressed pack 7 of 10) is recorded in [LITTLE-NEMO-QUIZ.md](LITTLE-NEMO-QUIZ.md). The grader unit test needs no checkpoint:

    python littlenemo/tests/quiz/test_quiz_grade.py

A live rerun needs a local Nemotron Nano 9B v2 tree and, for the pack path, a local .nq2. See the guide for the command lines. No model path is stored in the tree.

## Sanitizers and thread checking

The C suites run silent under ASan and UBSan (a review pass in 2026-11-24). ThreadSanitizer receipts for the self-contained zero-mutex engines and Little Nemo unit suites are in [docs/tsan/](tsan/RECEIPT-2026-11-27.md); rerun with `./docs/tsan/run-tsan.sh` on Linux or WSL. See [ZERO-MUTEX.md](ZERO-MUTEX.md) for what that record covers and what it leaves out.