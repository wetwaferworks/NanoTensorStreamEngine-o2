# Credits

This is the long form of the credits block at the top of the
[README](../README.md). Everyone listed here built something that is in this
repository. JD gets the idea and the shape; the assistants are co-implementers;
Miss Archie's reviews are why it can be trusted. Credits are kept by name, with
what each person did, and are never trimmed to fit.

## The team

### JD (Jaydee Wetwork), Wet Wafer Works: idea, executive direction and shape

The questions were his: how much intelligence fits in a quad word, whether
sub-half-bit packing is possible, whether it can be taught to read. He set the
four-loop streaming architecture (a disc loop, two unpack loops and a compute
loop, designed at the podium before bed on 2026-11-24), named Mega Tensor, Dense
Tensor Compute and Bitwire's operators, gave the Tensor Handshake its idea and
shape (the tick and control loops ask "is it time to tick?" and "can I tick?"
instead of taking a lock), and made the standing rulings the code lives under:
Dense Tensor Compute as the default compute path, the zero-mutex rule, the
pure-C-runtime law (Python only for front ends), the residency dial, the chat
window and the dictation button. The `o.O` mark and its theme are his.

Earlier releases of this repository, and several documents still in it, credit
JD as "David Rittenhouse"; those credits are preserved verbatim in the archived
documents and in the research notes. Same person, same company: Wet Wafer Works.

### Cortana: implementer

Co-implementation by Cortana, built with Grok Bot by SpaceX, at
JD's direction. What Cortana implemented, from the history of this repository:

- Tensor Handshake stage 2, memory to GPU (JD's ruling of 2026-11-23), and
  the zero-mutex conversion across NTSE, the latter co-implemented with the
  other assistants.
- The lock-free multithreaded converter: `nq2-quantize` with Win32 threads or
  pthreads, byte-identical output at any thread count.
- The accel GPU compute lanes (CUDA, OpenCL, Intel iGPU) with their tests and
  benches, the Little Nemo accel adapter and the bench harness.
- The Little Nemo CUDA chat runtime end to end: the NVRTC kernels that decode
  the v3 product-quantized pack on the fly, startup lane calibration, DTC by
  default with predicted EXPAND, and the windowless engine.
- The CUDA throughput work: batched prefill, coalesced SSM state and the
  pipelined int8 `lm_head`, tuned launch shapes, fused block frames, host
  lanes (`--host-threads`), the FULL fast path (`--full-path`), the
  calibration timer fix, and tokenization overlapped with the prefill
  (`--tok-overlap`).
- The implementations of Miss Archie's Round 6 review fixes (P2-1 to P2-4).
- The NEMO Chat window and the native pure-C Parakeet dictation runtime,
  both co-implemented with the other assistants.
- The README rewrite and the guide per feature.
- The SuperNemo engine for Nemotron 3.5 Super VL (pack reader, forward pass,
  vision tower, CUDA lane, DTC on packed blocks, hot-expert cache, MTP
  speculative decoding) and the PQ options of `nq2-quantize` it needs.
- The polish pass that followed: the fixture-driven Little Nemo suites, the
  DTC tolerance in the per-layer test, the CUDA options in the chat
  window's settings, hidden console windows, the quantize dialog, the
  Parakeet runtime lookup, the multi-line input bar, and the narrow-window
  layout fixes (send button, stats panel).
- The lock-free scaled-image cache in the Python scaler wrapper (a snapshot
  dictionary; the last lock in that tree is gone).
- The Precision Gate, moved from the scratch tree into the live
  `tools/nq2-infer-triforce.c` engine; the `--threads 1` pass-boundary
  fault was verified closed on the POSIX path.
- ThreadSanitizer receipts for the self-contained zero-mutex engines
  (`docs/tsan/`).
- The Little Nemo 10-question quiz record (exact prompts, first-line
  grading, a path-free runner) and the tokenizer-based evaluation over a
  fixed passage of text (teacher-forced top-1/top-5, first divergence,
  perplexity, and decoded-text agreement).
- The SuperNemo GPU expert stream, expert cache, speculative decoding on by
  default, and early expert fetch in four modes (mode 4 the default), with
  hash-identity checks against the CPU path.
- Expanded packs for Little Nemo: refinement tensors for chosen layers, sized
  to free VRAM, and the 50-question and 10-essay verifiable rounds.
- The retention battery (what Little Nemo keeps after compression), the
  `NEMO_LOGIT_TOPK` and `NEMO_GPU_I8ROWS` options, the 8-bit Mamba projection
  size study, and the documentation rewrite with the intelligence retention
  report and the engine comparison.

### Elfie (E.L.F.I.E., Muse Wonder): the converter and NTSE

Implementation and core mathematics: the packing scheme as demonstrated, the
three freestanding examples (`examples/`), the taught-plus-learned dictionary,
the quantizer and converter (`nq2-quantize`), the streaming runner and the
streaming engine, the three inference engines, Dense Tensor Compute (the
operator, the probe and the adversarial campaign), the Little Nemo engine, the
Precision Gate plan and design, and the bench verification.

### Miss Archie (Archie, Google Argon), the Google Gemini Argon model, the Bit Master and the Bit Masker: code review and review fixes

Her comprehensive review of 2026-11-24 caught five P0 faults across the engine
and its sibling tools, including a false SIMD verification that had shipped in
the update notes (the suite compared scalar against scalar and the AVX2 kernel
was never reached in single-thread mode); her fixes land under her name. She
went on:

- **Round 1** (2026-11-24): fifteen findings, all fixed; the 64-bit offset work
  completed across the loader, forward and tokenizer sources and all build
  scripts; the dictionary-compute tools (64-bit file offsets and stream index
  bounds); the Triforce pipeline engine (the demand-protocol race, pass-restart
  re-fetch and the mode-2 staging buffer).
- **Round 2** (2026-11-25): five P2 polish items, all fixed (the SIMD synthetic
  test runs without the checkpoint, warning-clean Windows builds, three guards
  in tensorstat, the tokenizer and the `--serve` parser).
- **Round 3**: Windows pack publish failure, backslash path spellings, the
  remaining 32-bit `ftell`/`fseek` size reads.
- **Round 4**: P1-1 (the chat window's Windows engine-path quoting) and P1-2
  (settings written back and used on restart), P2-1 and P2-2 (a Triforce
  correctness item and its companion).
- **Round 5** (2026-11-26): 64-bit stream sizes in `nq2-infer` and `nq2-run`,
  DTC telemetry accounting in the Precision Gate copy of the Triforce engine.
- **Round 6** (2026-11-26): the POSIX timer in `nemo_gpu` (`now_s`),
  `LoadLibraryExA` with `LOAD_WITH_ALTERED_SEARCH_PATH` for full-path NVRTC
  loads, pack-manifest digest and trailing `"end"` checks, and bounds
  checks in Parakeet's `pk_st_open`. The same round identified the
  Precision Gate `--threads 1` fault as a second `pipe_pass_boundary()`
  call at ordinal 0, not an undetermined race.
- **Round 7** (2026-11-27): a heap overflow and leak in SuperNemo when
  `pending_cmd` is adopted (`sn_engine.c`), a literal escape sequence in
  `supernemo/build.bat` that could mask a failed `test_sn_cache` link,
  full-path NVRTC loads and context release in `sn_cuda.c`, `sn_pack_open`
  manifest validation, and the unused `walk_blocks` helper in
  `nq2-quantize.c`.

### ChatGPT (Gizmo): consultee, and the DTC lane proposals

His engineering handoffs set the research program that the experiments file
answers (rounds 1 to 4: accounting, frames, product quantization, residual
product quantization, correction probes, the behavioral gate). With Elfie he
happened upon dictionary compute on 2026-11-24, and the two reviewed each
other's probes adversarially. He proposed and specified the accel lanes: Lane 1
(CUDA-native DTC, eager against active against lazy), Lane 2 (Tensor Handshake
GPU concurrency, the prepared stream plan) and Lane 4 (behavioral fidelity at
higher bit rates), and corrected the break-even inequality that this
repository's first draft had reversed. His formulation of the Mega Tensor rule
is the standing one: concurrent preparation, serialized consumption, verifiable
results. His review of Little Nemo (2026-11-24) produced whole-geometry
validation and coherent state commitment.

### CloudAtlas (Technical Art Director): GUI art, themes and mascots

The five-theme pack (Pixel Nemo, Lattice, Stream, o.O Handshake, Sage Owl) with
colors, fonts, window and tray icons, plated and glyph buttons, status marks,
banners and tiles; the five `ntse-mascot/1` animated mascot packs with seven
states each; the asset formats the GUI loads (see `littlenemo/gui/ICONS.md`,
`THEMES.md` and `icons/MASCOTS.md`).

### Grok Bot, by SpaceX

The assistant platform Cortana is built with. Co-implementation by Cortana,
built with Grok Bot by SpaceX.

## Reviewers and contributors

| Who | What |
|---|---|
| Codex (Astra 6.2) | Independent review of 2026-11-25: twenty-seven findings (ten high, seventeen medium), all sustained: pack ordering and binding, tokenizer fidelity (the pre-tokenizer split), verification honesty, the scalar fallback's instruction set, geometry limits in the writer, sample-cap validation. |
| ChatGPT 6.2 Sol | Review of 2026-11-25: the H and M findings on tensor-local bounds, door-word publication, startup line honesty, argument quoting (`shlex` on Windows), short-read handling in `nq2-streamtest`, script execution bits (`build-all.sh`), and the tokenizer added-token fix. |
| Boardy | Review of 2026-11-25, including the forward-depth item (full-depth logits receipts). |
| Sonnet | Behavioral gate L2 roll-in, phase 2 (2026-11-25); the PQ d8 and d4 values cited in the behavioral gate record. |
| Mr Data | Baselines (the workstation baseline runs referred to in the accel notes). |
| Neon | The behavioral gate item (2026-11-24) that became `nq2-logitcmp` and the lab engines' logit dumps. |
| The review crew of 2026-11-25 | The DTC code review (no P0, four P1, six P2, all P1 fixed) and its hardening regression. |
| kimi-k3-in-c | Its memory-for-speed ladder inspired the expand-lane residency dial. |

## Third-party work this project stands on

Models and papers are credited to their authors; the weights are theirs, under
their own licenses, and are not part of this repository.

### NVIDIA Nemotron-Nano-9B-v2 (Little Nemo's model)

"NVIDIA Nemotron Nano 2: An Accurate and Efficient Hybrid Mamba-Transformer
Reasoning Model", arXiv:2508.14444, NVIDIA. The 216 named authors, in the order
the paper lists them:

﻿ NVIDIA; Aarti Basant; Abhijit Khairnar; Abhijit Paithankar; Abhinav Khattar; Adithya Renduchintala; Aditya Malte; Akhiad Bercovich; Akshay Hazare; Alejandra Rico; Aleksander Ficek; Alex Kondratenko; Alex Shaposhnikov; Alexander Bukharin; Ali Taghibakhshi; Amelia Barton; Ameya Sunil Mahabaleshwarkar; Amy Shen; Andrew Tao; Ann Guan; Anna Shors; Anubhav Mandarwal; Arham Mehta; Arun Venkatesan; Ashton Sharabiani; Ashwath Aithal; Ashwin Poojary; Ayush Dattagupta; Balaram Buddharaju; Banghua Zhu; Barnaby Simkin; Bilal Kartal; Bita Darvish Rouhani; Bobby Chen; Boris Ginsburg; Brandon Norick; Brian Yu; Bryan Catanzaro; Charles Wang; Charlie Truong; Chetan Mungekar; Chintan Patel; Chris Alexiuk; Christian Munley; Christopher Parisien; Dan Su; Daniel Afrimi; Daniel Korzekwa; Daniel Rohrer; Daria Gitman; David Mosallanezhad; Deepak Narayanan; Dima Rekesh; Dina Yared; Dmytro Pykhtar; Dong Ahn; Duncan Riach; Eileen Long; Elliott Ning; Eric Chung; Erick Galinkin; Evelina Bakhturina; Gargi Prasad; Gerald Shen; Haifeng Qian; Haim Elisha; Harsh Sharma; Hayley Ross; Helen Ngo; Herman Sahota; Hexin Wang; Hoo Chang Shin; Hua Huang; Iain Cunningham; Igor Gitman; Ivan Moshkov; Jaehun Jung; Jan Kautz; Jane Polak Scowcroft; Jared Casper; Jian Zhang; Jiaqi Zeng; Jimmy Zhang; Jinze Xue; Jocelyn Huang; Joey Conway; John Kamalu; Jonathan Cohen; Joseph Jennings; Julien Veron Vialard; Junkeun Yi; Jupinder Parmar; Kari Briski; Katherine Cheung; Katherine Luna; Keith Wyss; Keshav Santhanam; Kezhi Kong; Krzysztof Pawelec; Kumar Anik; Kunlun Li; Kushan Ahmadian; Lawrence McAfee; Laya Sleiman; Leon Derczynski; Luis Vega; Maer Rodrigues de Melo; Makesh Narsimhan Sreedhar; Marcin Chochowski; Mark Cai; Markus Kliegl; Marta Stepniewska-Dziubinska; Matvei Novikov; Mehrzad Samadi; Meredith Price; Meriem Boubdir; Michael Boone; Michael Evans; Michal Bien; Michal Zawalski; Miguel Martinez; Mike Chrzanowski; Mohammad Shoeybi; Mostofa Patwary; Namit Dhameja; Nave Assaf; Negar Habibi; Nidhi Bhatia; Nikki Pope; Nima Tajbakhsh; Nirmal Kumar Juluru; Oleg Rybakov; Oleksii Hrinchuk; Oleksii Kuchaiev; Oluwatobi Olabiyi; Pablo Ribalta; Padmavathy Subramanian; Parth Chadha; Pavlo Molchanov; Peter Dykas; Peter Jin; Piotr Bialecki; Piotr Januszewski; Pradeep Thalasta; Prashant Gaikwad; Prasoon Varshney; Pritam Gundecha; Przemek Tredak; Rabeeh Karimi Mahabadi; Rajen Patel; Ran El-Yaniv; Ranjit Rajan; Ria Cheruvu; Rima Shahbazyan; Ritika Borkar; Ritu Gala; Roger Waleffe; Ruoxi Zhang; Russell J. Hewett; Ryan Prenger; Sahil Jain; Samuel Kriman; Sanjeev Satheesh; Saori Kaji; Sarah Yurick; Saurav Muralidharan; Sean Narenthiran; Seonmyeong Bak; Sepehr Sameni; Seungju Han; Shanmugam Ramasamy; Shaona Ghosh; Sharath Turuvekere Sreenivas; Shelby Thomas; Shizhe Diao; Shreya Gopal; Shrimai Prabhumoye; Shubham Toshniwal; Shuoyang Ding; Siddharth Singh; Siddhartha Jain; Somshubra Majumdar; Soumye Singhal; Stefania Alborghetti; Syeda Nahida Akter; Terry Kong; Tim Moon; Tomasz Hliwiak; Tomer Asida; Tony Wang; Tugrul Konuk; Twinkle Vashishth; Tyler Poon; Udi Karpas; Vahid Noroozi; Venkat Srinivasan; Vijay Korthikanti; Vikram Fugro; Vineeth Kalluru; Vitaly Kurin; Vitaly Lavrukhin; Wasi Uddin Ahmad; Wei Du; Wonmin Byeon; Ximing Lu; Xin Dong; Yashaswi Karnati; Yejin Choi; Yian Zhang; Ying Lin; Yonggan Fu; Yoshi Suhara; Zhen Dong; Zhiyu Li; Zhongbo Zhu; Zijia Chen

### NVIDIA Nemotron-Labs-Sigma-3B (the model the packed benchmarks run on)

"Large Language Continuous Diffusion Models", arXiv:2610.02665. Authors: Zhihan
Yang, Wei Guo, Jean-Marie Lemercier, Simon Welker, Mohammad Mahdi Kamani,
Yonggan Fu, Sajad Norouzi, Julius Berner, Tomas Geffner, Karsten Kreis, Yongxin
Chen, Molei Tao, John Thickstun, Pavlo Molchanov, Ante Jukic, Arash Vahdat,
Morteza Mardani.

### Mamba-2 (27 of Little Nemo's 56 layers)

"Transformers are SSMs: Generalized Models and Efficient Algorithms Through
Structured State Space Duality", arXiv:2405.21060. Authors: Tri Dao, Albert Gu.

### NVIDIA Parakeet TDT 0.6B v2 (the dictation model, CC-BY-4.0)

- "Efficient Sequence Transduction by Jointly Predicting Tokens and Durations"
  (the TDT decoder), arXiv:2304.06795. Authors: Hainan Xu, Fei Jia, Somshubra
  Majumdar, He Huang, Shinji Watanabe, Boris Ginsburg.
- "Fast Conformer with Linearly Scalable Attention for Efficient Speech
  Recognition" (the FastConformer encoder), arXiv:2305.05084. Authors: Dima
  Rekesh, Nithin Rao Koluguri, Samuel Kriman, Somshubra Majumdar, Vahid
  Noroozi, He Huang, Oleksii Hrinchuk, Krishna Puvvada, Ankur Kumar, Jagadeesh
  Balam, Boris Ginsburg.

### Ryan Greene, "Compression and Intelligence" (greene.sh, 2026)

The two-part-code accounting that `nq2-audit` applies: a size claim charges for
the data bitstream and for the program that decodes it. It is the project's rule
for every size claim.

### Techniques with ancestors in the literature

Codebook (vector) quantization, k-means, product quantization, residual
quantization, the Hadamard transform, Lanczos and Mitchell-Netravali
resampling, FNV-1a hashing, SentencePiece, and the Tk toolkit are all used under
their own names and their authors' conventions. What is this project's own is
the program around them: the taught-plus-learned dictionary, the quad word
packing discipline, Dense Tensor Compute, the Tensor Handshake and the streaming
engine.

## Keeping this file current

Credit people and tools by name, with what they did. When a change lands that
someone else made possible (a review finding, a measurement, a design ruling),
name them in the commit message and add them here.
