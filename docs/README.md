# Documentation index

Start with the [project README](../README.md): the credits, an honest status table, a quick start and a one-page architecture. This folder holds one guide per feature. Every flag and number in these guides comes from the code or from a measurement receipt in the tree; where something is not proven, the guide says so.

## Guides

| Guide | Covers |
|---|---|
| [INTELLIGENCE-RETENTION.md](INTELLIGENCE-RETENTION.md) | how much intelligence the compressed models keep, with speed beside accuracy for every configuration, and every table behind it |
| [COMPARISON.md](COMPARISON.md) | how NTSE compares with llama.cpp, ExLlamaV2, MLC-LLM, vLLM, KTransformers and PowerInfer, using only their published figures |
| [SUPERNEMO.md](SUPERNEMO.md) | the 120B SuperNemo engine: expert cache, GPU expert stream, MTP, early expert fetch, results |
| [EXPAND-REFINE.md](EXPAND-REFINE.md) | expanded packs: a refinement stage for chosen layers, sized to the free VRAM |
| [RETENTION-BATTERY-2026-11-27.md](RETENTION-BATTERY-2026-11-27.md) | the full retention battery report |
| [CREDITS.md](CREDITS.md) | everyone who built this, including the assistants as co-implementers, and the third-party papers and models with full author lists |
| [FORMAT-AND-CONVERTER.md](FORMAT-AND-CONVERTER.md) | the NQ2 format v3, the `nq2-quantize` converter and every option (per-tensor books, `--embed-fp16`, `--budgets`, `--pq`, multithreading), the manifest and pairing |
| [INFERENCE-TOOLS.md](INFERENCE-TOOLS.md) | `nq2-infer`, `nq2-run`, the Triforce engine, `nq2-infer-lab`, `nq2-logitcmp`, `nq2-streamtest`, pinned anchors |
| [DTC.md](DTC.md) | Dense Tensor Compute: the algebra, the shared kernel, the policies (eager, active, lazy, direct, expanded), break-even, results |
| [TENSOR-HANDSHAKE.md](TENSOR-HANDSHAKE.md) | the stream plan: ticks, priority list, claims, the door, expand lanes |
| [PRECISION-GATE.md](PRECISION-GATE.md) | the confidence gate that decides when to compute at higher precision; scratch status and calibration |
| [ACCEL.md](ACCEL.md) | the accel lanes (CUDA, OpenCL on NVIDIA, Intel iGPU), ops, environment variables, bench tables, Lane 4 fidelity results |
| [ACCEL-CUDA.md](ACCEL-CUDA.md) | Little Nemo on CUDA: startup calibration, DTC by default, predict and expand, flags, the windowless engine, speed and quiz results |
| [ZERO-MUTEX.md](ZERO-MUTEX.md) | the zero-mutex design, what is established and what is not |
| [LITTLE-NEMO-ENGINE.md](LITTLE-NEMO-ENGINE.md) | the pure C Nemotron-Nano-9B-v2 engine: block math, layer-major prefill, `engine_gui`, flags |
| [LITTLE-NEMO-TOKENEVAL.md](LITTLE-NEMO-TOKENEVAL.md) | tokenizer-based Little Nemo eval: a fixed passage, teacher-forced top-1/top-5, first divergence, perplexity, and decoded-text agreement |
| [GUI.md](GUI.md) | the NTSE Chat GUI: models, attachments, thinking line, stats panel, themes, mascots, shape customizer, scaler, settings, shortcuts, and every protocol v3 message |
| [DICTATION.md](DICTATION.md) | `parakeet-c` speech to text: flags, stdio protocol, microphone selection, `PARAKEET_MODEL`, streaming partials |
| [TESTING.md](TESTING.md) | every suite and how to run it |
| [ROADMAP.md](ROADMAP.md) | the fidelity roadmap and what comes next |

## Specifications and references

| Document | Covers |
|---|---|
| [../API-REFERENCE.md](../API-REFERENCE.md) | the tool command lines |
| [../littlenemo/PROTOCOL.md](../littlenemo/PROTOCOL.md) | the engine line protocol v2 and v3 |
| [../littlenemo/BLOCK-MATH-SPEC.md](../littlenemo/BLOCK-MATH-SPEC.md) | the exact Nemotron-H block math |
| [../littlenemo/gui/ICONS.md](../littlenemo/gui/ICONS.md), [THEMES.md](../littlenemo/gui/THEMES.md) | icon pack and theme formats |
| [../parakeet/PARAKEET-C.md](../parakeet/PARAKEET-C.md) | the Parakeet runtime's engineering notes |
| [../accel/README.md](../accel/README.md) | the accel research record |

## Research record

[DICTIONARY-COMPUTE.md](../DICTIONARY-COMPUTE.md), [MEGA-TENSOR.md](../MEGA-TENSOR.md), [PIPELINE-2026-11-24.md](../PIPELINE-2026-11-24.md), [BASELINE-2026-11-23.md](../BASELINE-2026-11-23.md), [EXPERIMENTS-2026-11-24.md](../EXPERIMENTS-2026-11-24.md), [LITTLE-NEMO.md](../LITTLE-NEMO.md), [WHAT-WE-CAN-DO.md](../WHAT-WE-CAN-DO.md), [TESTING-NOTES.md](../TESTING-NOTES.md), [UPDATE-NOTES-2026-11-24.md](../UPDATE-NOTES-2026-11-24.md); [the precision-gate plan and record](precision-gate/PLAN.md); [the previous README](archive/README-2026-11-25.md), preserved so no earlier credit or number is lost.

## Troubleshooting

Each guide ends with a troubleshooting table for its own feature: [inference tools](INFERENCE-TOOLS.md), [accel](ACCEL.md), [CUDA lane](ACCEL-CUDA.md), [Little Nemo engine](LITTLE-NEMO-ENGINE.md), [GUI](GUI.md), [dictation](DICTATION.md).