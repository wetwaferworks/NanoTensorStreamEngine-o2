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
- **SuperNemo engine:** Nemotron 3.5 Super VL 120B-A12B (512-expert MoE and a vision tower). The GUI side exists; the engine path does not.
- **Precision Gate:** move the scratch implementation into `tools/` and chase the open intermittent `--threads 1` boundary fault ([PRECISION-GATE.md](PRECISION-GATE.md)).
- **OpenCL Nemotron forward:** the OpenCL lanes have no Nemotron forward in the Little Nemo build; calibration rejects them.

## Verification

- ThreadSanitizer runs for the zero-mutex engines, with receipts committed ([ZERO-MUTEX.md](ZERO-MUTEX.md)).
- A tokenizer-based evaluation of Little Nemo, so quality is scored on text and not only on top-1 of one diffusion denoiser step.
- Replace the Python scaler cache lock (the one lock in the tree) with a lock-free structure, or drop the cache.

## Documentation

- Keep [CREDITS.md](CREDITS.md) current as people and assistants join.
- Commit the Little Nemo quiz record so the 10 of 10 raw result and the 0.07 tokens/s figure can be reproduced from the tree.

See also [CONTRIBUTING in the README](../README.md#contributing).