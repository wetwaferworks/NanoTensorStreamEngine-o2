# Roadmap

What is done, what is in progress, and what is next. Items here are plans, not results. Where something has been measured, the guide that holds the numbers is linked.

## Fidelity: keeping the model's intelligence at very low bits per weight

The open problem. Packing works and is fast; **fidelity at low bits per weight is not yet proven** (the Lane 4 results in [ACCEL.md](ACCEL.md), and the 7 of 10 versus 10 of 10 quiz in [ACCEL-CUDA.md](ACCEL-CUDA.md)). JD's target is full intelligence at roughly 0.2 bits per weight on average; that is a goal, not a measurement. The plan is to implement each method below in pure C, test it against the raw model, and keep the best.

| # | Method | Idea |
|---|---|---|
| 1 | Fit for outputs, not weights | calibrated, activation-aware codebooks (in the style of GPTQ and AQLM) that minimize layer-output error rather than weight error; the favorite |
| 2 | Critical tensors raw | keep sensitive tensors at high precision and squeeze the rest harder to hold the average; the evidence so far is that raw projections gave 25 to 30 of 32 top-1 agreement |
| 3 | Residual codebooks | a second tiny codebook that encodes the first's error |
| 4 | Shared codebooks | one cross-layer codebook plus a small per-layer delta |
| 5 | Short distillation | tune codebooks and scales only, against the raw model's logits |
| 6 | DTC plus predict | expand only low-confidence tokens to full precision; implemented for Little Nemo on CUDA ([ACCEL-CUDA.md](ACCEL-CUDA.md)), where it did not close the gap because the pack itself is lossy |

### How each method will be judged

- top-1 agreement and KL divergence against the raw model over thousands of tokens;
- perplexity on a fixed text (WikiText) for Little Nemo;
- the quiz and essays side by side with raw, and small MMLU, GSM8K and ARC slices;
- a per-layer sensitivity sweep.

Every result will be published with its negatives, as Lane 4 was.

## Engines and lanes

- **Little Nemo CUDA lane:** done and measured; source is in `accel/` ([ACCEL-CUDA.md](ACCEL-CUDA.md)).
- **CUDA runtime for Sigma:** the lanes are verified, but there is no end-to-end GPU decode for the Sigma model yet ([ACCEL.md](ACCEL.md)).
- **SuperNemo engine:** Nemotron 3.5 Super VL 120B-A12B (512-expert MoE and a vision tower). The engine and its tests are in `supernemo/`; real-pack speed and quality are not measured yet ([supernemo/PLAN.md](../supernemo/PLAN.md)).
- **GPU-accelerated quantizer:** landed in scratch on 2026-11-27 as `nq2-quantize --gpu` (GPU build only, off by default). The nearest-code search for codebook fitting and packing runs on CUDA, and the output is byte-identical to the CPU path ([GPU-QUANTIZER.md](GPU-QUANTIZER.md)). The SuperVL run measured compute-bound: about 10.4 of 12 cores busy against about 28 MB/s of disk reads. Still open: the Sigma-3B timings, and moving the global-book packing pass and the residual stage onto the GPU.
- **Precision Gate:** the scratch implementation now lives in `tools/nq2-infer-triforce.c`. The `--threads 1` boundary fault was a double `pipe_pass_boundary()` call at ordinal 0 (Miss Archie's Round 6 note), fixed by counting consume ticks so the boundary runs once per pass; verified after the transplant on the POSIX path at `--threads 12` and `--threads 1`, both producing the packed Sigma anchor token 122388 ([PRECISION-GATE.md](PRECISION-GATE.md)). The Windows fused worker still always expands.
- **Per-pack calibration of the DTC predict thresholds:** `dtc_margin` (0.5) and `dtc_entropy` (4.0) are hand-picked heuristics from one pack and a ten-question quiz ([ACCEL-CUDA.md](ACCEL-CUDA.md#dtc-by-default-predict-decides-expand)). Derive them per pack instead: measure, over a fixed prompt set, how often a DTC step's top-1 differs from full expansion at each margin and entropy, choose the thresholds that meet a stated disagreement budget, and ship them with the pack the way the Precision Gate ships its calibration.
- **OpenCL Nemotron forward:** the OpenCL lanes have no Nemotron forward in the Little Nemo build; calibration rejects them.

## Verification

- **ThreadSanitizer:** done for the self-contained zero-mutex pool, the DTC adversarial suite, and the Little Nemo unit suites that need no private checkpoint. Receipt and runner are in [docs/tsan/](tsan/RECEIPT-2026-11-27.md) ([ZERO-MUTEX.md](ZERO-MUTEX.md)).
- **Tokenizer-based Little Nemo eval:** done as a harness. A committed original passage (140 official-tokenizer ids), a first-divergence / top-1 / top-5 / perplexity scorer, and `test_tokeneval` score quality on decoded text under teacher forcing, not on the top-1 of one step. Live 9B runs are opt-in ([LITTLE-NEMO-TOKENEVAL.md](LITTLE-NEMO-TOKENEVAL.md)).
- **Scaler cache:** the Python wrapper's scaled-image cache is a lock-free snapshot dictionary; the one `threading.Lock` in the tree is gone ([ZERO-MUTEX.md](ZERO-MUTEX.md)).

## Documentation

- Keep [CREDITS.md](CREDITS.md) current as people and assistants join.
- **Little Nemo quiz record:** done. Exact prompts, first-line grading and a path-free runner are in [LITTLE-NEMO-QUIZ.md](LITTLE-NEMO-QUIZ.md) and `littlenemo/tests/quiz/`.

See also [CONTRIBUTING in the README](../README.md#contributing).