# NanoTensor Stream Engine (NTSE-o2)

NTSE runs language models on your own computer. The model is compressed into a small packed file and streamed through the compute loop instead of being held whole, so a model too big for your memory can still run, and nothing has to leave the machine. The runtime is pure C99 (a C compiler and nothing else). The chat window and a few test tools are Python front ends that draw and relay and do no model work.

Formerly the Nano Quant Engine (renamed 2026-11-25). The `nq2` tool names and the `.nq2` pack format did not change with the name.

This is a research bench, not a finished product. Every number below has a receipt in the repository, and anything we have not measured is marked pending.

## Where things stand (28 November 2026)

| Piece | What works | Honest limit |
|---|---|---|
| Packs and the converter | `nq2-quantize` packs a model into the NQ2 v3 format, multithreaded and byte-identical at any thread count, with an optional GPU search that gives the CPU's exact bytes. | A 2.2 bits-per-weight pack loses accuracy, measured below. |
| Dense Tensor Compute (DTC) | Multiplies straight from the compressed blocks, never unpacking the model. Bit-identical on every GPU lane. | Equals the unpacked path to float noise, not bit for bit. |
| Little Nemo (Nemotron Nano 9B v2) | Chat on the CUDA lane at about 13 tokens per second on a 4 GB RTX 3050 Ti (real-world, shared machine). | Less accurate than the original: see the table below. The 30+ tokens per second goal for 8 GB cards is not measured. |
| SuperNemo (Nemotron 3.5 Super VL, 120B-A12B) | Runs on the same 4 GB card with an expert cache, a GPU expert stream, speculative decoding (MTP, on by default) and early expert fetch (mode 4, on by default). Same answers as the CPU path (identical hash), 5 of 5 on its quiz subset. | About 0.1 to 0.3 tokens per second on this card. |
| Chat window and dictation | Text, image and video attachments, themes, a live speech-to-text button (`parakeet-c`). | Dictation needs your own copy of the Parakeet weights. |

## How much intelligence is kept

Little Nemo, exact scoring against the original model (119 items, first answer word, the 8-bit head the card runs):

| configuration | fits a 4 GB card | decode tok/s | first-answer hits | share of the original |
|---|---|---|---|---|
| original (bf16) | no | n/a | 106 of 119 | 100% |
| shipping pack | yes | 16.2 quieter morning, about 13 shared | 31 of 119 | 29% |
| pack plus x2 refinement | yes | about 13, same conditions as the shared shipping-pack run | 52 of 119 | 49% |
| pack plus x1 refinement | no (CPU only) | n/a | 59 of 119 | 56% |
| pack with the original Mamba projections restored | not built yet | pending | quiz 47 of 50, same as the original | pending |

What we learned:

- The loss is almost all in the Mamba `in_proj` and `out_proj` weights. Restoring them lifts the quiz from 17 to 47 of 50 and brings back the model's habit of stopping when it is done (stop probability 0.017 to 0.9997).
- The output head is not the problem (8-bit against 16-bit moves the loss by 0.008 per token or less).
- The packed model is less sure of itself and drifts in wording, but it still says "I don't know" for made-up things (8 of 8).
- Next step: give those projections more bits. All of them at 8 bits makes a 4,784.5 MiB pack that does not fit 4 GB, so we are testing smaller slices.

Full tables, with speed beside accuracy for every configuration, are in [docs/INTELLIGENCE-RETENTION.md](docs/INTELLIGENCE-RETENTION.md). How this is similar to llama.cpp is in [docs/COMPARISON.md](docs/COMPARISON.md).

## Quick start

Windows with MSYS2 UCRT64 gcc on PATH (any C99 compiler works; the threaded tools need a POSIX-thread MinGW):

```
build.bat                      :: the nq2 tools, the Little Nemo engine and its tests
accel\build-accel.bat          :: GPU lanes
supernemo\build.bat            :: the SuperNemo engine and its tests
parakeet\build.bat             :: parakeet-c and its tests
```

Linux: `sh build-all.sh` (see [docs/TESTING.md](docs/TESTING.md)).

See the chat window first, with no model at all:

```
python littlenemo\gui\nemo_chat.py --model littlenemo --engine "python littlenemo/tests/fake_engine.py --stdio --model littlenemo"
```

You need NVIDIA's model files under their own licenses: the Nemotron Nano 9B v2 checkpoint and its tokenizer for Little Nemo, and a packed Super VL checkpoint with its `config.json` and `tokenizer.json` for SuperNemo. They are not in this repository.

Build the CUDA Little Nemo engine (after `accel\build-accel.bat`, with the CUDA toolkit available):

```
powershell -File accel\build-nemo-gpu.ps1
```

Chat with Little Nemo from the window:

```
python littlenemo\gui\nemo_chat.py --model littlenemo --engine "accel\bin\nemo-engine-cuda-w.exe <checkpoint-dir> <tokenizer.json> --chat --cuda-pack <pack.nq2> --accel auto"
```

The same arguments work on the console binary, `accel\bin\nemo-engine-cuda.exe`. Details, flags and the ten-question smoke test are in [docs/ACCEL-CUDA.md](docs/ACCEL-CUDA.md).

SuperNemo:

```
supernemo\build.bat
sn-engine.exe --pack <pack-dir> --config <checkpoint-dir>\config.json --tokenizer <checkpoint-dir>\tokenizer.json --gpu-mb 3000 --below-normal --ask "your question"
```

Without `--ask` it speaks the chat protocol on stdin and stdout, so the same window can drive it. Details are in [docs/SUPERNEMO.md](docs/SUPERNEMO.md).

## Where the reports live

- [docs/INTELLIGENCE-RETENTION.md](docs/INTELLIGENCE-RETENTION.md): how much intelligence the compressed models keep, with speed beside accuracy
- [docs/EXPAND-REFINE.md](docs/EXPAND-REFINE.md): the small refinement pack for Little Nemo
- [docs/SUPERNEMO.md](docs/SUPERNEMO.md): the 120B engine
- [docs/COMPARISON.md](docs/COMPARISON.md): how this is similar to llama.cpp
- Scrubbed quiz and retention receipts: `littlenemo/tests/quiz/receipts/` and `littlenemo/tests/retention/receipts/`

The documentation index is [docs/README.md](docs/README.md).

## Ideas in one page

- **NQ2 pack.** Weights travel in blocks. A small book of prototype blocks sits beside the model, and each block is an index into the book plus a tiny scale and offset. [docs/FORMAT-AND-CONVERTER.md](docs/FORMAT-AND-CONVERTER.md)
- **Dense Tensor Compute.** Build one small table per input segment, then every output row is lookups. [docs/DTC.md](docs/DTC.md)
- **Expanded packs.** A small refinement for chosen layers adds back some of what compression lost, sized to the free memory on the card. [docs/EXPAND-REFINE.md](docs/EXPAND-REFINE.md)
- **Expert stream and cache (SuperNemo).** Compressed experts go to the card, are expanded there, and their slots are reused. [docs/SUPERNEMO.md](docs/SUPERNEMO.md)
- **Tensor Handshake and zero mutexes.** Concurrent preparation, serialized use, no locks anywhere. [docs/TENSOR-HANDSHAKE.md](docs/TENSOR-HANDSHAKE.md), [docs/ZERO-MUTEX.md](docs/ZERO-MUTEX.md)
- **Precision Gate and accel lanes.** When to trust the table, and the CUDA and OpenCL lanes. [docs/PRECISION-GATE.md](docs/PRECISION-GATE.md), [docs/ACCEL.md](docs/ACCEL.md)

The research record (experiments, baselines, the fidelity roadmap) is linked from the index, and the earlier long README is kept in [docs/archive/](docs/archive/) so no credit or number is lost.

## Layout

```
tools/       the nq2 tools: quantizer, inference engines, runner, probes
littlenemo/  the Nemotron Nano 9B v2 engine, tests (quiz, retention), chat GUI
supernemo/   the Nemotron 3.5 Super VL engine and its expert stream
accel/       GPU lanes (CUDA, OpenCL): sources, tests, benches
parakeet/    parakeet-c, the native speech runtime
docs/        the documentation set
```

Not in the repository: the models, packed `.nq2` files and raw run logs.

## Contributing

Issues and pull requests are welcome. The rules:

- The runtime is C99. Python is for front ends and test tooling only. GPU code lives in `accel/`.
- Zero mutexes in shipped code.
- Every size is a complete size (stream, book and frames counted) or labeled an estimate. Report timings as real-world when other jobs share the machine.
- Do not edit pinned anchors to make a test pass. A moved hash needs a reason.
- No weights, run logs, secrets or local paths in commits.
- Credit people and tools by name. Keep [docs/CREDITS.md](docs/CREDITS.md) current.

## License and release

MIT, Copyright (c) 2026 Wet Wafer Works (see [LICENSE](LICENSE)). Models you download keep their own licenses. Release version 0.1.0, build counter 0001, tag v0.1.0; see [VERSION.txt](VERSION.txt).

## Credits

Everyone below built this. Full detail is in [docs/CREDITS.md](docs/CREDITS.md).

- **JD (Jaydee Wetwork), Wet Wafer Works: idea, direction and shape.** The questions were his: how much intelligence fits in a quad word, whether sub-half-bit packing is possible. The four-loop streaming architecture, Mega Tensor, Dense Tensor Compute and the Tensor Handshake are his, and so is every ruling that set the direction: DTC as the default compute path, the zero-mutex rule, the pure-C runtime, the chat window and the dictation button. Earlier releases credited JD under another name; those credits are kept in the archived documents.
- **Cortana: implementer.** The converter's multithreading, the accel GPU lanes, the Little Nemo CUDA runtime and its throughput work, the SuperNemo engine with its GPU expert stream, expert cache, speculative decoding and early expert fetch, expanded packs, the retention battery, the chat window and dictation runtime (with the other assistants), the Precision Gate transplant, and these docs. Co-implementation by Cortana, built with Grok Bot by SpaceX.
- **Elfie (Muse Wonder): the converter and NTSE.** The packing scheme, the three freestanding examples, the taught-plus-learned dictionary, `nq2-quantize`, the streaming engine, Dense Tensor Compute, the Little Nemo engine, the Precision Gate design, and the bench verification.
- **Miss Archie (Google Argon): code review and review fixes.** Her review of 2026-11-24 caught five serious faults, including a false SIMD verification, and her fixes land under her name. Rounds 2 through 7 kept moving real code.
- **ChatGPT (Gizmo): consultee and the DTC lane proposals.** His handoffs set the research program; with Elfie he happened upon dictionary compute, and he specified the accel lanes and the behavioral fidelity lane.
- **CloudAtlas (Technical Art Director): GUI art, themes and mascots.**
- **Grok Bot, by SpaceX: the assistant platform Cortana is built with.**
- **Reviewers and contributors:** Codex (Astra 6.2), ChatGPT 6.2 Sol, Boardy, Sonnet, Neon, and the review crew of 2026-11-25. The memory-for-speed ladder of kimi-k3-in-c inspired the expand-lane residency dial.
- **Third-party work this project stands on:** NVIDIA Nemotron-Nano-9B-v2 (Little Nemo) and its paper (arXiv:2508.14444); NVIDIA Nemotron 3.5 Super VL (SuperNemo); NVIDIA Nemotron-Labs-Sigma-3B (arXiv:2610.02665); NVIDIA Parakeet TDT 0.6B v2 (CC-BY-4.0) with arXiv:2304.06795 and arXiv:2305.05084; Mamba-2 (arXiv:2405.21060, Tri Dao and Albert Gu); and Ryan Greene's "Compression and Intelligence" (greene.sh, 2026), whose two-part-code accounting is our rule for every size claim. Full author lists are in the credits guide.
