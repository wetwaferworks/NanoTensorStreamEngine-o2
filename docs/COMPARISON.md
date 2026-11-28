# How NTSE compares with other engines

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.

**Read this first.** Every number in the tables below is a figure that someone else published, copied here with its source, hardware and quantization. **None of them is our measurement.** We did not download, build or run any other engine or any other model to write this page. The hardware, models and quantization differ from ours in almost every row, so the speed columns are context, not a ranking. A same-card head-to-head on our RTX 3050 Ti would settle it; that is **pending JD's go**.

Our own numbers, for reference (real-world, one 4 GB RTX 3050 Ti laptop, 16 GB RAM, see [INTELLIGENCE-RETENTION.md](INTELLIGENCE-RETENTION.md)):

| model | size on disk | decode | what it is |
|---|---|---|---|
| Little Nemo (Nemotron Nano 9B v2) | pack 2,039 MiB (2.2 bits per weight) plus a 560 MiB 8-bit head on the card | 12.6 to 13.9 tok/s | 31 of 119 first-answer hits against 106 for the original |
| SuperNemo (Nemotron 3.5 Super VL, 120B-A12B) | packed, streamed from disk | 0.16 to 0.19 tok/s median | same hash as the CPU path, quiz subset 5 of 5 |

## What others report

### llama.cpp (and Ollama, which bundles it)

llama.cpp is the closest relative: C/C++, very widely used, runs on CPU and GPU, and supports the same hybrid Mamba architecture (`nemotron_h`) since release b6315 ([Hugging Face thread](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/discussions/4)). Ollama's source tree contains Nemotron-H code for the same architecture ([ollama source](https://github.com/ollama/ollama/blob/d319227d/llama/llama.cpp/src/models/nemotron-h.cpp)).

| reported figure | hardware | model and quantization | source |
|---|---|---|---|
| 15.23 tok/s generation (128 tokens), 207 tok/s prompt | GTX 1050, 4 GB (the only 4 GB card we found in the scoreboard) | Llama 2 7B, Q4_0 (3.56 GiB), all layers on the card | [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013), a user entry |
| 186 tok/s generation, 11,993 tok/s prompt | RTX 4090, 24 GB | same model and quantization | same scoreboard |
| 48 to 53 tok/s (Q4_K_M, 6.1 GB) and 49 to 60 tok/s (IQ3_M, 4.9 GB) | RTX 3090, 24 GB, all 56 layers on the card | Nemotron Nano 9B v2 GGUF, short runs, temperature 0.7 | [GGUF bench report](https://huggingface.co/weathermanj/NVIDIA-Nemotron-Nano-9B-v2-gguf/blob/main/REPORT.md) |
| 7 to 11 tok/s | CPU only, 16 threads | Nemotron Nano 9B v2 GGUF, IQ3_M down to Q8_0 | same report |
| 19.9 tok/s generation, 769 tok/s prompt | NVIDIA DGX Spark, 128 GB unified memory | Nemotron 3 Super 120B-A12B, Q4_K (65.1 GiB) | [llama.cpp bench notes](https://github.com/turbo-tan/llama.cpp-tq3/blob/main/benches/nemotron/nemotron-dgx-spark.md) and [aggregator page](https://llamaperf.com/model/nemotron-3) |
| about 17 tok/s | RTX 4090, 24 GB (the file is larger than the card, so it is partly offloaded) | Nemotron 3 Super 120B-A12B, Unsloth Q2_K_XL (54.7 GB) | [forum post quoted on llamaperf](https://llamaperf.com/model/nemotron-3) (a user report) |

What this tells us. On a 4 GB card, llama.cpp fits a 7B model at 4 bits and a GTX 1050 reports about 15 tok/s. Nemotron Nano 9B v2 files at 4.9 GB (IQ3_M) and 6.1 GB (Q4_K_M) do not fit a 4 GB card in full, so a 4 GB machine would run it with partial offload, and we found no published number for that. Our pack is about 2 GiB for the same model, which is why it fits with room for the head. The cost is accuracy: those files are 3 to 4.5 bits per weight against our 2.2, and the quality notes in that report are quick output checks, not a retention battery, so there is nothing published to compare our 29 percent first-answer retention with. For the 120B model, the published llama.cpp numbers use machines with 64 GB or more of memory or a 24 GB card; we found no published 4 GB-card figure.

### ExLlamaV2

Reported in its README for Llama 2 7B on an RTX 4090: 257 tok/s at 3.0 bits per weight, 211 at 4.0 and 179 at 5.0 (EXL2 format, decode only) ([README](https://github.com/turboderp-org/exllamav2/blob/master/README.md)). It is an NVIDIA-only, transformer-focused engine; we found no Mamba or hybrid support in its documentation.

### MLC-LLM

Its published benchmark reports 4-bit Llama 2 decode on an RTX 4090 of 204.8 tok/s (7B) and 113.5 tok/s (13B), against 177.46 and 105.94 for ExLlamaV2 and 151.1 and 88.0 for llama.cpp in the same table, with fp16 compute and 256 tokens decoded ([mlc-ai/llm-perf-bench](https://github.com/mlc-ai/llm-perf-bench)). Its strength is compiling one model to many backends. We did not confirm Nemotron-H or Mamba support.

### vLLM

A serving engine for GPUs with plenty of memory, built for many concurrent requests rather than one user on a small card. Its supported-model list includes Nemotron-H (`NemotronHForCausalLM`) and Mamba2 models ([supported models](https://docs.vllm.ai/en/stable/models/supported_models/)). We found no published figure for a 4 GB card, and it is not aimed at one.

### KTransformers (MoE offload)

Reported for DeepSeek-R1/V3 671B at 4 bits (Q4_K_M) on one 24 GB RTX 4090 with dual Xeon Gold 6454S CPUs and 382 GB to 1 TB of DRAM: 13.9 to 15.9 tok/s decode and 101 to 112.5 tok/s prefill on 1K to 8K prompts with six experts per token (version 0.2.1), and 12.4 to 13.5 decode with the model's default eight experts ([KTransformers guide](https://github.com/kvcache-ai/ktransformers/blob/main/doc%2Fen%2FDeepseekR1_V3_tutorial.md)). The same guide lists llama.cpp at 4.51 tok/s decode on its 2x32-core machine. The design computes the experts on a very large server CPU and keeps the dense part on the card. That needs hundreds of gigabytes of RAM, a different regime from a 16 GB laptop where the experts live on disk.

### PowerInfer

Reported on an RTX 4090 with an i9-13900K and 192 GB of RAM: up to 11.69 times the speed of llama.cpp on Falcon(ReLU)-40B in FP16, on the same machine ([paper, SOSP 2024](https://arxiv.org/html/2312.12456)). It relies on activation sparsity in ReLU-family models, so it does not apply directly to Nemotron's hybrid layers.

### A vendor post worth noting

A forum post from a new engine named glyd claims 43 tok/s for Nemotron 3 Super at about 3 bits per weight on an RTX 4090, 35 tok/s on a 16 GB card and 13 to 18 tok/s on a 32 GB RAM PC, with hot experts on the card, the rest on the CPU, and the remainder streamed from SSD ([post](https://www.reddit.com/r/LocalLLM/comments/1x2uid4/nemotron_3_super_120b_at_43_toks_on_one_rtx_4090/)). It is a self-report that we cannot verify, and it names no 4 GB result. It is the same family of idea as ours (hot experts resident, the rest streamed), on much bigger hardware.

## Features side by side

"Not found" means we looked in the project's public documentation and did not find it, not that it is absent; "not compared" or "not checked" means we did not look. Implementation languages are general knowledge, not checked on this visit. NTSE entries are measured or tested in this repository.

| | NTSE | llama.cpp / Ollama | ExLlamaV2 | MLC-LLM | vLLM | KTransformers | PowerInfer |
|---|---|---|---|---|---|---|---|
| runtime written in pure C99, no other dependencies | yes (GPU code builds at run time through the driver) | C/C++ | Python with CUDA extensions | compiler stack (TVM) | Python with CUDA | Python and C++ | C/C++ (a llama.cpp fork) |
| no mutexes in shipped code | yes, by rule, with a documented thread-sanitizer receipt | not a stated goal | not found | not found | not found | not found | not found |
| identical output across thread counts, with a published hash | yes (exact scorer and generated answers; SuperNemo whole-model hash identical on CPU and GPU, 4 and 12 threads) | not found | not found | not found | not found | not found | not found |
| Mamba / hybrid support | yes (Little Nemo, SuperNemo) | yes (`nemotron_h`) | not found | not confirmed | yes (Nemotron-H, Mamba2) | not confirmed | no (ReLU-sparsity models) |
| runs a 9B hybrid in about 2 GiB of weights on a 4 GB card | yes (2.2 bits per weight; accuracy cost measured above) | smallest 9B files in the report above are 4.7 GB (Q2_K) and 4.9 GB (IQ3_M) | not found | not found | no | not applicable | not applicable |
| MoE experts streamed from disk through a small card | yes, 120B on a 4 GB card at 0.1 to 0.3 tok/s | expert offload to system RAM (`--cpu-moe`); no 4 GB figure found | no | not found | no | yes, with hundreds of GB of RAM | not applicable |
| compute straight from compressed blocks (no unpacking) | yes (DTC) | not compared | not compared | not compared | not compared | not compared | not compared |
| publishes accuracy retention against the original, with misses | yes, in this repository | not checked | not checked | not found | not applicable | not found | states accuracy is retained; no battery found |
| speed on a 4 GB card | measured | one 4 GB figure for a 7B model (GTX 1050) | not found | not found | not aimed at it | not aimed at it | not aimed at it |

## What we can and cannot say

- We can say our design choices are different (pure C, mutex-free, identical results at any thread count, a hybrid model and a 120B MoE on a 4 GB card), and that the specific measurements in this repository are repeatable by anyone who has the checkpoints.
- We cannot say we are faster or more accurate than any of them. The published figures use faster cards, more memory and gentler quantization, and none measures the same retention battery. On a card with more memory than ours, llama.cpp and the others are very likely faster than NTSE today.
- **Pending JD's go:** a same-card head-to-head on this machine (llama.cpp running Nemotron Nano 9B v2 at the largest quantization that fits 4 GB, scored on our retention battery and timed the same way). It needs a download of the other engine and a model file, so we have not started it.

## Sources used

- llama.cpp CUDA scoreboard (Llama 2 7B Q4_0): https://github.com/ggml-org/llama.cpp/discussions/15013
- Nemotron-H support in llama.cpp (release b6315): https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2/discussions/4
- Nemotron Nano 9B v2 GGUF benchmark report: https://huggingface.co/weathermanj/NVIDIA-Nemotron-Nano-9B-v2-gguf/blob/main/REPORT.md
- Nemotron 3 Super on llama.cpp: https://github.com/turbo-tan/llama.cpp-tq3/blob/main/benches/nemotron/nemotron-dgx-spark.md and https://llamaperf.com/model/nemotron-3
- Ollama Nemotron-H implementation: https://github.com/ollama/ollama/blob/d319227d/llama/llama.cpp/src/models/nemotron-h.cpp
- ExLlamaV2 README: https://github.com/turboderp-org/exllamav2/blob/master/README.md
- MLC-LLM performance benchmark: https://github.com/mlc-ai/llm-perf-bench
- vLLM supported models: https://docs.vllm.ai/en/stable/models/supported_models/
- KTransformers DeepSeek-R1/V3 guide: https://github.com/kvcache-ai/ktransformers/blob/main/doc%2Fen%2FDeepseekR1_V3_tutorial.md
- PowerInfer paper: https://arxiv.org/html/2312.12456
- glyd forum post: https://www.reddit.com/r/LocalLLM/comments/1x2uid4/nemotron_3_super_120b_at_43_toks_on_one_rtx_4090/

Figures were read on 28 November 2026. Pages change; check the source before quoting.
