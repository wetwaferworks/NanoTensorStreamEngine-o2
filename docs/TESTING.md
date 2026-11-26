# Testing

Every suite in the tree, what it needs, and how to run it. Suites exit nonzero on failure. Weights are never in the repository, so suites that need a real checkpoint or pack take its path on the command line.

## Build first

| Script | Builds |
|---|---|
| `build.bat` / `build.sh` (repo root) | all 23 tools in `tools/` and Little Nemo with its suites; the threaded tools need a POSIX-thread MinGW on Windows |
| `build-all.bat` / `build-all.sh` | the one-click name for the same build |
| `build-experiments.bat` / `build-experiments.sh` | the research-tool subset |
| `littlenemo/build.bat` / `build.sh` | `nemo-engine`, `nemo-inspect` and 11 suites into `littlenemo/bin/` |
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
| `test_hostile <refuse\|i32> <dir>` | crafted fixture dirs | hostile checkpoints are refused cleanly |
| `test_coherence <checkpoint_dir> <dtype_dir>` | a partial checkpoint and a wrong-dtype dir | whole-geometry validation and state-commit coherence |
| `test_par <checkpoint_dir> [lanes] [layers]` | a checkpoint dir | multi-lane forward is memcmp-identical to the single-lane reference |
| `test_simd <checkpoint_dir> [layers]` | a checkpoint dir | the opt-in AVX2 path: deterministic, faithful to scalar, dispatch proven by entry counters |
| `test_prefill <checkpoint_dir> [layers]` | a checkpoint dir; built by hand, not by `build.bat` | layer-major prefill is bit-identical to token-by-token stepping, across a chunk boundary |

```
cd littlenemo
build.bat
bin\test_mamba.exe && bin\test_attn.exe && bin\test_guards.exe
bin\test_tokenizer.exe tests\fixture_tokenizer.json
bin\test_loader.exe <checkpoint_dir>
```

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

Bench tables and what the numbers mean: [ACCEL.md](ACCEL.md). The CUDA lane's per-layer verification (`test_nemo_gpu_layers`) is covered in [ACCEL-CUDA.md](ACCEL-CUDA.md#verification).

## GUI suites (Python, `littlenemo/tests/`)

Run each from `littlenemo/tests`, for example `python test_gui_protocol.py`. They use the fixture engines (`fake_engine.py`, `mock_engine.py`), need no weights, and exit nonzero on failure.

| Suite | Checks |
|---|---|
| `test_gui_protocol.py` | protocol v2 and v3 parsing, escapes, percent-encoding, size limits |
| `test_gui_client.py` | the engine client over pipes and TCP, send gating |
| `test_gui_media.py` | attachment inspection: limits, extension versus leading bytes, video frames |
| `test_gui_stats.py` | the stats model and the thinking-line headline |
| `test_gui_settings.py` | INI read and write, comment preservation, validation |
| `test_gui_icons.py` | icon packs, manifests, mascots and theme packs; **needs the C scaler built** (it fails with `KeyError: 32` in "pack assets resolve" otherwise) |
| `test_gui_scale.py` | the ctypes scaler front end |
| `test_gui_ui_smoke.py` | the Tk window, driven through `--demo smoke` (needs a display) |

```
cd littlenemo/gui/native && build.bat && test_nemo_scale.exe && cd ../../tests
python test_gui_icons.py
```

`littlenemo/gui/native/test_nemo_scale` is the C test for the scaler (`ALL PASS`).

## Parakeet suites

| Suite | Run | Checks |
|---|---|---|
| `parakeet-tests.exe <model-dir>` | `bin\parakeet-tests.exe %PARAKEET_MODEL%` | 17 unit tests (FFT, GEMM at 1 and 8 threads, pool, resampler, WAV, mel, SPM, checkpoint dims) |
| `parakeet/tests/stdio_smoke.py` | `python parakeet\tests\stdio_smoke.py` | READY, PING, FILE, START and STOP, QUIT over `--stdio` (needs the model and its `samples\`) |
| `parakeet/tests/gui_dictation_test.py [wav] [ref.txt]` | `python parakeet\tests\gui_dictation_test.py` | the mic button end to end, with a reference WAV replayed as the microphone (needs a display) |

## Sanitizers and thread checking

The C suites run silent under ASan and UBSan (a review pass in 2026-11-24). **There are no ThreadSanitizer receipts in this repository**; see [ZERO-MUTEX.md](ZERO-MUTEX.md) for what is and is not established about the zero-mutex design. Running them is on the [roadmap](ROADMAP.md).