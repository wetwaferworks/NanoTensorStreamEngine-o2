# Retention battery, 2026-11-27: what Little Nemo keeps after compression

JD approved this battery on 2026-11-27 at 5:51 PM ET (all of Gizmo's lanes, plus extras). It asks one question: after NanoTensor compression, what does Nemotron Nano 9B v2 ("Little Nemo") still do the way the original does? SuperNemo (Nemotron 3.5 Super VL) is covered only by a small subset run and by a pointer to the expert-residency work. Everything here was run on the bench machine (RTX 3050 Ti Laptop 4 GB, i5-12450H, 16 GB), with the battery in `littlenemo/tests/retention/` and the scrubbed answers in `littlenemo/tests/retention/receipts/2026-11-28/`. No weights, packs, raw logs, logit dumps or machine paths are in the tree.

## Headline

1. **The model stops talking because of the compression, not because of the engine.** After the gold answer and its newline, the original bf16 model puts probability 1.0 on its end-of-turn token (`<SPECIAL_12>`, id 12, already in the engine's stop set) on **50 of 50** items. The packed model puts a mean of **0.017** there (median 0.001), and the stop token is the top choice on **0 of 50**. It writes another newline (27 times) or opens a parenthesis (14 times) instead. That is why packed answers run to the token cap with blank lines and "(150 words)" loops. The cause is in the Mamba projections: put the original bf16 `in_proj` and `out_proj` back in all 27 Mamba layers (nothing else) and the stop token returns to a mean probability of **0.9997**, top choice on **50 of 50**. Restoring the MLP projections instead gets only 0.08 (4 of 50), and the x1 and x2 expansions, which refine a handful of layers, do not move it (0.024 and 0.022). See [Termination](#p0-termination-the-end-of-turn-token).
2. **The damage is in the layers, not in the output head.** Swapping the lm_head between bf16 and the int8 per-row head the CUDA lane uses moves the loss by 0.008 nats per token or less, and the first-token hits by at most 1 of 119, in every arm. Gizmo's higher-precision-head test says: no gain there.
3. **Three tensor classes carry the loss, and Mamba carries almost all of it.** Putting the original bf16 weights back into the Mamba `in_proj` and `out_proj` of all 27 Mamba layers (and nothing else) takes the quiz from 17/50 to **47/50, the same as the original**, and closes 100 percent of the loss gap. MLP `down_proj` alone closes 41 percent, Mamba `in_proj` alone 46 percent, the four attention layers 10 percent.
4. **Expanded packs help, and two layers get most of the way there.** Six refined layers (x1, teacher-forced on the CPU only, it does not fit the 4 GB card) halve the loss (NLL per token 1.62 to 1.10, raw 0.27; first-token hits 32 to 60 of 119, raw 106) and fix long-context recall (0/4 to 4/4). The x2 pack that does fit the card (refine tensors on layers 8 and 11 only, 73.4 MiB) gets NLL 1.22 and 53 of 119 first-token hits with the bf16 head (52 of 119, NLL 1.22 with the int8 head the card uses; quiz 29/50 against 17/50 packed, 34/50 for x1), so about three quarters of x1's gain for a third of the refined layers. On CUDA, the full generation battery for x2 against packed: quiz 34/50 against 36/50 (the lenient grader hides the improvement, see headline 5), multi-step 3/12 against 1/12, long context 4/4 against 3/4, essays 8/10 by the keyword grader against 4/10 (4/10 against 3/10 when read by hand). Generated differences this small are inside what one run can show.
5. **The packed model writes a sentence where the original writes the answer.** Teacher-forced on "answer with just the answer", the original starts with the bare answer on 47 of 50 quiz items. The packed pack starts with the bare answer on 17 of 50, and on 18 of the 33 misses its first token is "The". Generated with a lenient grader (answer found in the first line) the same pack passes 36 of 50. Both numbers are real; they measure different things, see [How to read the numbers](#how-to-read-the-numbers).
6. **Abstention survives, confidence does not.** On made-up entities the packed model still answers "I don't know" on 8 of 8, in generation and teacher-forced. But its confidence is no longer tied to being right: mean top-1 confidence 0.53 with 27 percent of first tokens right (expected calibration error 0.256), against 0.93 and 89 percent for the original (ECE 0.065).
7. **Speed against JD's targets.** Little Nemo on CUDA decodes at **12.6 to 13.7 tok/s (median)** and prefills at 56 to 70 tok/s on the 4 GB card: above the **10 tok/s floor**, well under the **30+ tok/s target**, which is for 8 GB cards and was not measured here (this machine has only the 4 GB card). These runs shared the CPU with other jobs, so they read lower than the quiet 12:52 PM round (16.2 tok/s decode). The SuperNemo subset decoded at 0.17 tok/s, far under the floor. See [Speeds](#speeds).
8. **Results are repeatable.** The teacher-forced evaluator gives the same hash with 1 thread and with 12 (`8f3435aec98819af`), and 72 of 72 generated answers were identical between the default host threads and a 1-thread run.

## Accuracy and speed per configuration

One line per configuration, accuracy next to speed, for picking the best setup. Teacher-forced columns are exact and repeatable (CPU); generated columns are CUDA greedy runs; speeds are real-world (shared CPU, see [Speeds](#speeds)). Higher is better for hits and passes, lower for NLL. Configurations that cannot run on the card have no speed.

| configuration | where it runs | first-token hits, all 119 items (teacher-forced) | NLL per token | quiz first token (TF) | quiz pass (generated) | multi-step (generated) | essays (generated) | answers that stop on their own | decode tok/s (median) | prefill tok/s (median) | card peak MiB | against the 10 tok/s floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| raw bf16 (original) | CPU, teacher-forced only | 106/119 (89%) | 0.2675 | 47/50 | - | - | - | - | - | - | - | no speed on the card: 18 GB, does not fit |
| raw bf16, int8 head | CPU, teacher-forced only | 106/119 (89%) | 0.2669 | 47/50 | - | - | - | - | - | - | - |  |
| packed, int8 head (the shipping CUDA lane) | CUDA 4 GB | 31/119 (26%) | 1.6309 | 16/50 | 36/50 | 1/12 | 4/10 keyword, 3/10 read by hand | 8/141 | 12.6 | 66.2 | 3672 | meets the 10 tok/s floor |
| packed, bf16 head | CPU only (head 1,120 MiB does not fit) | 32/119 (27%) | 1.6232 | 17/50 | - | - | - | - | - | - | - | no gain over int8 head |
| x1 expanded, 6 layers, int8 head | CPU, teacher-forced only | 59/119 (50%) | 1.1071 | 34/50 | - | - | - | - | - | - | - | does not fit the 4 GB card |
| x1 expanded, bf16 head | CPU, teacher-forced only | 60/119 (50%) | 1.1031 | 34/50 | - | - | - | - | - | - | - |  |
| x2 expanded, 4 tensors, int8 head | CUDA 4 GB | 52/119 (44%) | 1.2238 | 28/50 | 34/50 | 3/12 | 8/10 keyword, 4/10 read by hand | 13/141 | 13.0 | 70.3 | 3681 | meets the 10 tok/s floor; fits the card |
| x2 expanded, bf16 head | CPU, teacher-forced only | 53/119 (45%) | 1.2187 | 29/50 | - | - | - | - | - | - | - |  |
| SuperNemo (120B MoE) subset | CUDA (sn-engine) | - | - | - | 5/5 on 5 questions (earlier round) | - | - | - | 0.17 | 0.58 | 3777 | BELOW the 10 tok/s floor |

Reading it: only two Little Nemo configurations run on the 4 GB card at all, the packed pack and x2, and both decode at about 13 tok/s, above the floor. x2 costs no measurable speed. Its accuracy gain on the generated quiz is within noise; the teacher-forced columns show what the expansion buys when it is measured exactly. The original and x1 are the accuracy ceiling and the middle point, not options for this card.

For choosing: on this card x2 is the better of the two runnable configurations on every exact measure at no speed cost (first-token hits 52 against 31 of 119, NLL 1.22 against 1.63, same 13 tok/s). Neither fixes the stopping problem. The biggest lever the sweep found is the Mamba `in_proj` and `out_proj` weights: with those restored, both accuracy (quiz 47/50) and turn-end behaviour (stop on 50 of 50) return to the original. A pack that gives more bits to those tensors is the configuration this battery points to, but it was not built or timed here, so its size and speed are unknown. The x2 expansion refines only 2 of the 27 Mamba layers.

## How to read the numbers

- **Teacher-forced (TF)** runs feed the gold answer to the model one token at a time and score what it would have said at each step. They are exact, repeatable, and do not depend on sampling. "First token" means the model's top choice at the first answer position equals the gold's first token, so a model that writes "The capital of Australia is Canberra." misses even though it knows the answer. "Whole answer" needs every gold token to be top-1. NLL per token is the average surprise (lower is better). "Gold in top 5" is a looser check. All TF runs ran on the CPU, layer by layer, because the bf16 model does not fit the card.
- **Generation** runs let the model write greedy (chat template on, thinking off) on the CUDA lane, and a grader reads the first answer line. A pass means the gold answer (or a listed alias) is found there, so a full sentence containing the answer passes.
- **Raw bf16 was measured teacher-forced only.** A 9B bf16 model is about 18 GB, more than this machine's 16 GB RAM, so a raw generation arm was not run. Where a table has a raw row it comes from TF; generation tables compare packed against x2.
- Arms: **raw bf16** (original weights); **packed** (`nemo9b-pq4-L.nq2`, PQ d4, 2.2 bits per weight, about 0.32 relative L2 error on the large layers); **x1** (refine tensors on layers 8, 9, 10, 11, 14 and 21; too large for the card's VRAM budget, so CPU only); **x2** (refine tensors on layer 8 up/down and layer 11 in/out, 73.4 MiB, fits the card and runs with the engine's expand logic); **int8 head** (the lm_head the CUDA lane uses, built like `accel/nemo_gpu.c`) against **bf16 head**.
- Battery: 203 items (50 quiz, 12 candidate-wrong, 39 paraphrase, 12 multi-step arithmetic, 10 novel-passage reasoning, 16 calibration, 10 instruction following, 4 long-context recall, 50 turn-end probes) plus 10 essays. 193 are teacher-forced, 141 generated.

## P0: Termination, the end-of-turn token

Question: why does Little Nemo not stop? The instrumented engine (`NEMO_LOGIT_TOPK`, read-only, off by default) prints the stop-token probability and rank at every generated step. The stop set is {12, 2, 11} (`<SPECIAL_12>` plus the generation-config ids); the tokenizer has no `<|end_of_text|>`.

Teacher-forced at the right place (the 50 quiz gold answers, each followed by a newline, which is what both models write first):

| arm | items | mean P(stop) | median P(stop) | stop is top-1 | stop in top 5 | median stop rank | what is top-1 instead |
|---|---|---|---|---|---|---|---|
| raw bf16 | 50 | 1.0 | 1.0 | 50/50 | 50/50 | 0 | <SPECIAL_12> x50 |
| packed | 50 | 0.0171 | 0.001 | 0/50 | 6/50 | 19 | \n x27;  ( x14;   x1 |
| expanded x1 (6 layers) | 50 | 0.0238 | 0.0046 | 1/50 | 5/50 | 16 |  ( x24; \n x8;   x6 |
| expanded x2 (2 layers, 4 tensors) | 50 | 0.0219 | 0.0019 | 1/50 | 1/50 | 19 |  ( x20; \n x19;   x1 |
| packed + original Mamba in/out_proj | 50 | 0.9997 | 0.9999 | 50/50 | 50/50 | 0 | <SPECIAL_12> x50 |
| packed + original MLP up/down_proj | 50 | 0.0805 | 0.0031 | 4/50 | 12/50 | 16 | \n x23;  ( x14; <SPECIAL_12> x4 |

In generation, the same measurement at the step after the first newline of each answer:

| generation arm | answers that ended on their own (of cap-limited runs) | answers with a newline | mean P(stop) after the newline | median | stop is top-1 | median stop rank |
|---|---|---|---|---|---|---|
| packed | 8/141 | 139 | 0.0266 | 0.00044 | 5/139 | 55 |
| packed-det-ht1 | 3/72 | 70 | 0.0178 | 0.00048 | 2/70 | 47 |
| expanded-x2 | 13/141 | 138 | 0.0766 | 0.00154 | 12/138 | 25 |

What it means: the original model ends its turn with certainty right after the answer line. The compressed model has lost that habit. Its stop token has a median rank around 19 to 55 and a probability of about 1 in a thousand, so under greedy decoding it almost never leaves. Raising the lm_head precision does not help (see the head table), so the end-of-turn behaviour is stored in the compressed layers, and run G (below) narrows it to the Mamba input and output projections. The x2 expansion raises the generated stop probability somewhat (top choice in 12 of 138 generated answers against 5 of 139) and raises the number of answers that end on their own from 8 to 13 of 141, but the teacher-forced probe on the clean position shows no real change (mean 0.022 against 0.017 packed), so that gain is small and unreliable. A cheap practical fix is a serving-side one: stop at the first blank line after a non-empty answer line for short-answer prompts. That changes the reply the user sees, not the model, so it is a mitigation and not a retention result.

**Which layers carry it (run G).** Same 50 probes, teacher-forced, base plus three variants. Finished: x2 (layers 8 and 11 refined): mean P(stop) 0.0219, stop is top-1 on 1 of 50, median stop rank 19; packed with the original Mamba in/out projections: mean P(stop) 0.9997, stop is top-1 on 50 of 50, median stop rank 0; packed with the original MLP up/down projections: mean P(stop) 0.0805, stop is top-1 on 4 of 50, median stop rank 16. Restoring the Mamba projections brings the stop probability to a mean of 0.9997 against 0.0171 packed and 1.0 raw, so the end-of-turn habit lives mostly in those layers, consistent with the class sweep.

## P0: matched raw, packed and expanded battery

Teacher-forced first-token results (set sizes: quiz 50, paraphrase 39, multi-step 12, passage 10, calibration 8 answerable; "all" is 119 items):

| arm | quiz first token | quiz whole answer | paraphrase | multi-step | passage | calibration (answerable) | all first token | NLL / token | gold in top 5 | ECE | mean top-1 confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw bf16 | 47/50 | 45/50 | 34/39 | 9/12 | 8/10 | 8/8 | 106/119 | 0.2675 | 118/119 | 0.065 | 0.9306 |
| raw bf16, int8 head | 47/50 | 45/50 | 34/39 | 9/12 | 8/10 | 8/8 | 106/119 | 0.2669 | 118/119 | 0.0547 | 0.9303 |
| packed | 17/50 | 16/50 | 11/39 | 1/12 | 0/10 | 3/8 | 32/119 | 1.6232 | 84/119 | 0.2564 | 0.5253 |
| packed, int8 head | 16/50 | 14/50 | 11/39 | 1/12 | 0/10 | 3/8 | 31/119 | 1.6309 | 84/119 | 0.2648 | 0.5254 |
| expanded x1 (6 layers) | 34/50 | 32/50 | 16/39 | 2/12 | 3/10 | 5/8 | 60/119 | 1.1031 | 95/119 | 0.1449 | 0.5939 |
| x1, int8 head | 34/50 | 32/50 | 16/39 | 2/12 | 2/10 | 5/8 | 59/119 | 1.1071 | 95/119 | 0.1311 | 0.5901 |
| expanded x2 (2 layers, 4 tensors) | 29/50 | 26/50 | 15/39 | 2/12 | 2/10 | 5/8 | 53/119 | 1.2187 | 90/119 | 0.1785 | 0.595 |
| x2, int8 head | 28/50 | 25/50 | 15/39 | 2/12 | 2/10 | 5/8 | 52/119 | 1.2238 | 90/119 | 0.1815 | 0.592 |

Generated on CUDA, graded on the first answer line:

| arm (CUDA, greedy) | quiz | paraphrase | multi-step | passage | calibration | instructions (full) | instructions (first line only) | long context |
|---|---|---|---|---|---|---|---|---|
| packed | 36/50 | 25/39 | 1/12 | 4/10 | 15/16 | 2/10 | 4/10 | 3/4 |
| packed-det-ht1 | 36/50 | - | 1/12 | 4/10 | - | - | - | - |
| expanded-x2 | 34/50 | 23/39 | 3/12 | 4/10 | 13/16 | 2/10 | 6/10 | 4/4 |

What it means: the original gets 106 of 119 first tokens right, the packed pack 32, x1 60. Multi-step arithmetic and novel-passage reasoning are hit hardest (packed 1/12 and 0/10 teacher-forced, 1/12 and 4/10 generated with the lenient grader, against 9/12 and 8/10 for the original). Expansion recovers part of it (x1 60 and x2 53 of 119 first tokens, against 32 packed). The generated quiz score for x2 differs from packed by two questions, which is noise at n = 50, and the lenient grader accepts a sentence that contains the answer, so the teacher-forced columns are the better measure of what the expansion restores.

## P1: head precision (Gizmo's specific ask)

Little Nemo with a higher-precision output head, everything else held constant. A bf16 lm_head needs 1,120 MiB; only 922 MiB are free on the 4 GB card after the pack, so the CUDA lane cannot hold it (the first attempt silently fell back to the CPU and was stopped before scoring any item; its log is kept as evidence). The test is therefore teacher-forced on the CPU, with the same layers and only the head changed.

| weights | bf16-class head: first token | NLL/tok | int8 per-row head: first token | NLL/tok | NLL change |
|---|---|---|---|---|---|
| raw bf16 | 106/119 | 0.2675 | 106/119 | 0.2669 | -0.0006 |
| packed | 32/119 | 1.6232 | 31/119 | 1.6309 | 0.0077 |
| expanded x1 (6 layers) | 60/119 | 1.1031 | 59/119 | 1.1071 | 0.004 |
| expanded x2 (2 layers, 4 tensors) | 53/119 | 1.2187 | 52/119 | 1.2238 | 0.0051 |

What it means: the head is not where the quality goes. Raising it from the int8 per-row head to bf16 buys at most 0.008 nats per token and changes at most one item. Spending 1.1 GB of VRAM on a bf16 head is not worth it.

## P1: sensitivity by tensor class

Restoring one class of tensors to the original bf16 values in an otherwise packed model (teacher-forced, quiz, 50 items). "Class" is across every layer where the tensor exists (27 Mamba layers carry `in_proj` and `out_proj`, 25 MLP layers carry `up_proj` and `down_proj`, 4 attention layers).

| restored to original bf16 | quiz first token | quiz NLL (50 items) | share of the packed-to-raw gap closed |
|---|---|---|---|
| base | 17/50 | 140.209 | 0% |
| cls_in | 29/50 | 87.162 | 46% |
| cls_out | 18/50 | 137.027 | 3% |
| cls_up | 19/50 | 106.644 | 29% |
| cls_down | 30/50 | 92.636 | 41% |
| cls_attn | 24/50 | 128.846 | 10% |
| cls_mamba | 47/50 | 23.786 | 100% |
| cls_mlp | 32/50 | 73.908 | 57% |
| ref (from run A) | 47/50 | 23.753 | 100% |

What it means: Mamba `in_proj` and `out_proj` together are the problem. Restoring both gets the original's score back (47/50, NLL 23.8 against 23.75), though each alone gives only a part (46 percent and 3 percent), so the errors combine rather than add. MLP weights are second (57 percent together). The four attention layers matter least. This agrees with where the expansion work put its refinement (layers 8, 9, 10, 11, 14, 21) and says the cheapest big win is to give the Mamba projections more bits, not the head and not attention.

## P1: top-k on the questions the packed pack got wrong

The 14 quiz questions the 12:52 PM round failed (4, 5, 8, 14, 31, 37, 38, 39, 41, 42, 44, 45, 48, 49), teacher-forced: where does the gold first token rank?

| arm | gold is top-1 | gold in top 5 | median gold rank | gold more likely than the packed lane's wrong answer |
|---|---|---|---|---|
| raw bf16 | 12/14 | 13/14 | 0 | 11/14 |
| packed | 2/14 | 7/14 | 6 | 1/14 |
| expanded x1 (6 layers) | 4/14 | 9/14 | 3 | 5/14 |
| expanded x2 (2 layers, 4 tensors) | 3/14 | 9/14 | 3 | 2/14 |

| quiz item | raw bf16 | packed | expanded x1 (6 layers) | expanded x2 (2 layers, 4 tensors) |
|---|---|---|---|---|
| 4 | rank 0 (p=1.000), top-1 '9' | rank 8 (p=0.003), top-1 '7' | rank 9 (p=0.001), top-1 '4' | rank 9 (p=0.002), top-1 '4' |
| 5 | rank 0 (p=0.999), top-1 '1' | rank 2 (p=0.041), top-1 '4' | rank 0 (p=0.507), top-1 '1' | rank 1 (p=0.127), top-1 '4' |
| 8 | rank 0 (p=0.997), top-1 '3' | rank 6 (p=0.004), top-1 '1' | rank 4 (p=0.004), top-1 '4' | rank 2 (p=0.006), top-1 '4' |
| 14 | rank 0 (p=0.990), top-1 'O' | rank 7 (p=0.001), top-1 'Toronto' | rank 7 (p=0.002), top-1 'Toronto' | rank 6 (p=0.004), top-1 'Toronto' |
| 31 | rank 0 (p=0.989), top-1 '1' | rank 0 (p=0.629), top-1 '1' | rank 0 (p=0.880), top-1 '1' | rank 0 (p=0.756), top-1 '1' |
| 37 | rank 0 (p=0.999), top-1 '1' | rank 2 (p=0.203), top-1 '4' | rank 0 (p=0.739), top-1 '1' | rank 0 (p=0.809), top-1 '1' |
| 38 | rank 0 (p=1.000), top-1 '3' | rank 7 (p=0.005), top-1 '6' | rank 1 (p=0.194), top-1 '6' | rank 1 (p=0.128), top-1 '6' |
| 39 | rank 0 (p=0.999), top-1 '1' | rank 3 (p=0.082), top-1 '6' | rank 1 (p=0.302), top-1 '3' | rank 1 (p=0.168), top-1 '3' |
| 41 | rank 0 (p=0.991), top-1 '3' | rank 9 (p=0.021), top-1 'There' | rank 2 (p=0.107), top-1 '6' | rank 3 (p=0.086), top-1 '6' |
| 42 | rank 0 (p=1.000), top-1 '1' | rank 0 (p=0.362), top-1 '1' | rank 0 (p=0.671), top-1 '1' | rank 0 (p=0.606), top-1 '1' |
| 44 | rank 0 (p=0.999), top-1 'J' | rank 3 (p=0.010), top-1 'The' | rank 5 (p=0.009), top-1 'The' | rank 6 (p=0.005), top-1 'Mars' |
| 45 | rank 0 (p=0.998), top-1 'Merc' | rank 3 (p=0.017), top-1 'Mars' | rank 3 (p=0.014), top-1 'Earth' | rank 3 (p=0.008), top-1 'Earth' |
| 48 | rank 1 (p=0.417), top-1 'Six' | rank 57 (p=0.001), top-1 'The' | rank 15 (p=0.004), top-1 'The' | rank 35 (p=0.001), top-1 'The' |
| 49 | rank 7 (p=0.001), top-1 'Diamond' | rank 241 (p=0.000), top-1 'The' | rank 117 (p=0.000), top-1 'The' | rank 140 (p=0.000), top-1 'The' |

What it means: for the original the gold answer is the first choice on 12 of the 14 (the other two are hard even for it: "legs of an insect", "hardest natural substance"). The packed pack ranks the gold answer first on 2 of 14 and in the top 5 on 7. The wrong answers it gave are not random slips: the gold sits at rank 2 to 9 on most, so the knowledge is damaged rather than gone. x1 lifts some (items 5 and 37 reach rank 0, 38, 39 and 41 reach rank 1 to 2) but 4, 14, 44, 45, 48 and 49 stay low.

## P1: expert residency and I/O telemetry

Not repeated here. The SuperNemo expert-residency, early-fetch and pinned-cache work is covered by the gpustream round-6 record (scratch commits 0e50902 and 1f52c02, receipts in `littlenemo/tests/quiz/receipts/2026-11-27/supernemo/`). JD's note that MTP and early expert fetch (EEF) should be tested again is for that worker. This battery only ran a small SuperNemo subset (below).

## P2: semantic essay grading

The keyword grader counts required facts. A human-style read checks whether the essay says something false. Both are shown, for the packed pack and for x2 (10 essays each, 320 tokens, CUDA). Hand grades, with every false claim quoted, are in `receipts/2026-11-28/semantic-*.json`; a pass needs the main facts right and no false claim.

| essay | topic | packed: keyword grader | packed: semantic | packed: false claims | x2: keyword grader | x2: semantic | x2: false claims |
|---|---|---|---|---|---|---|---|
| 1 | photosynthesis | 4/7 fail | fail | 0 | 0/7 fail | fail | 0 |
| 2 | water cycle | 3/6 fail | fail | 0 | 5/6 pass | pass | 0 |
| 3 | seasons | 4/5 pass | fail | 2 | 4/5 pass | fail | 1 |
| 4 | human heart | 3/7 fail | fail | 1 | 6/7 pass | pass | 0 |
| 5 | world war ii | 6/7 pass | pass | 1 | 6/7 pass | fail | 3 |
| 6 | declaration of independence | 4/6 fail | fail | 2 | 5/6 pass | fail | 2 |
| 7 | dna | 1/8 fail | fail | 2 | 2/8 fail | fail | 0 |
| 8 | solar system | 5/6 pass | pass | 0 | 5/6 pass | fail | 1 |
| 9 | vaccines | 5/5 pass | pass | 0 | 5/5 pass | pass | 0 |
| 10 | newton's laws | 1/5 fail | fail | 0 | 5/5 pass | pass | 1 |
|  | **passes** | 4/10 | 3/10 | 8 | 8/10 | 4/10 | 8 |

What it means: the keyword grader flatters the compressed model. x2 jumps from 4/10 to 8/10 on keywords and from 3/10 to 4/10 on a careful read, because half of x2's keyword passes still contain a false claim (the Sun as "90 percent of Earth's mass", Japan surrendering in 1947, an invented line quoted from the Declaration, the Moon or the equator driving the seasons). Eight false claims in each arm. Both arms degenerate after the first paragraph (repeated "(150 words)" lines, blank lines, a stray `</think>`), which is the termination problem above.

## P2: novel passage reasoning, paraphrase stability

Passage reasoning (10 invented passages that need a small calculation or comparison, for example opening times that chain from one to the next) is in the tables above: original 8/10, packed 0/10 first-token, x1 3/10; generated 4/10 for packed and x2 alike.

Paraphrase stability: the same fact asked three ways (13 groups of three).

| arm (teacher-forced first token) | groups of 3 | all three right | mixed | all three wrong |
|---|---|---|---|---|
| raw bf16 | 13 | 11 | 1 | 1 |
| raw bf16, int8 head | 13 | 11 | 1 | 1 |
| packed | 13 | 2 | 3 | 8 |
| packed, int8 head | 13 | 2 | 3 | 8 |
| expanded x1 (6 layers) | 13 | 4 | 2 | 7 |
| x1, int8 head | 13 | 4 | 2 | 7 |
| expanded x2 (2 layers, 4 tensors) | 13 | 4 | 2 | 7 |
| x2, int8 head | 13 | 4 | 2 | 7 |

| generation arm | groups of 3 | all three right | mixed | all three wrong |
|---|---|---|---|---|
| packed | 13 | 5 | 6 | 2 |
| expanded-x2 | 13 | 6 | 4 | 3 |

What it means: the original is stable (11 of 13 groups right all three ways). The packed pack is right all three ways on 2 of 13 groups teacher-forced and on 5 of 13 generated, and gets 6 of 13 groups right only some of the ways, so what it knows depends on how the question is phrased. x2 is about the same (6 right three ways, 4 mixed, 3 wrong).

## Extras

**Calibration and abstention.** Eight questions about made-up entities (a country "Zantoria", an element "vorinium") with the instruction to reply exactly "I don't know" when there is no real answer, and eight real questions with the same instruction.

| generation arm | made-up questions answered "I don't know" | real questions answered right | real questions wrongly refused |
|---|---|---|---|
| packed | 8/8 | 7/8 | 1/8 |
| expanded-x2 | 8/8 | 5/8 | 1/8 |

| arm | "I don't know" is the top first token | mean P(first token I) | mean NLL of the full phrase |
|---|---|---|---|
| raw bf16 | 8/8 | 1.0 | 0.0 |
| raw bf16, int8 head | 8/8 | 1.0 | 0.0 |
| packed | 8/8 | 0.8932 | 0.129 |
| packed, int8 head | 8/8 | 0.8959 | 0.126 |
| expanded x1 (6 layers) | 8/8 | 0.9881 | 0.024 |
| x1, int8 head | 8/8 | 0.9886 | 0.024 |
| expanded x2 (2 layers, 4 tensors) | 8/8 | 0.9752 | 0.032 |
| x2, int8 head | 8/8 | 0.9761 | 0.031 |

Abstention is intact in every arm. The cost shows up on the answerable ones: x2 refused one and got 5 of 8 right against 7 of 8 for packed (the 8 are too few to separate them). Confidence quality is in the ECE column of the matched table above.

**Multi-step arithmetic, instruction following, long context.** From the tables above: multi-step 1/12 (packed) and 3/12 (x2) generated, 9/12 for the original teacher-forced; instruction following ("reply with the word yes in capitals and nothing else", "list three fruits, one per line", "answer in exactly one word") 2/10 fully right for both generated arms, 4/10 (packed) and 6/10 (x2) on the first line alone; long-context recall of one fact (a number, time or year) among about 20 sentences of invented notes (about 860 tokens), generated 3/4 (packed) and 4/4 (x2).

Teacher-forced long context (4 passages, about 860 tokens each; the strict first token):

| arm | first token | whole answer | mean NLL | gold ranks | top-1 instead |
|---|---|---|---|---|---|
| raw bf16 | 4/4 | 4/4 | 0.03 | 0, 0, 0, 0 | '4', '2', '7', '6' |
| raw bf16, int8 head | 4/4 | 4/4 | 0.03 | 0, 0, 0, 0 | '4', '2', '7', '6' |
| packed | 0/4 | 0/4 | 4.219 | 2, 1, 1, 1 | 'The', 'The', 'The', 'The' |
| packed, int8 head | 0/4 | 0/4 | 4.218 | 2, 1, 1, 1 | 'The', 'The', 'The', 'The' |
| expanded x1 (6 layers) | 4/4 | 4/4 | 0.427 | 0, 0, 0, 0 | '4', '2', '7', '6' |

The packed pack's long-context miss is a format miss (it starts "The code for the red door is ..." and the gold is the bare number), not a recall failure; its gold token ranks 1 or 2. x1 gets all four with the bare number.

**Determinism across thread counts.** Teacher-forced: the same 6 quiz items scored with 1 thread (930 s) and 12 threads (149 s) produce the same hash, `8f3435aec98819af`. Generated: packed with the default host threads against packed with `--host-threads 1`, 72 of 72 answers identical. The evaluator uses fixed-order dot products, so thread count changes only the time.

## Speeds

Recorded against JD's floor (**10 tok/s on the 4 GB card**) and target (**30+ tok/s on 8 GB cards**). All timings are real-world: another worker's fid-quant was running on the CPU for the whole battery, the teacher-forced runs shared the CPU (8 threads, BelowNormal), and JD may have been on the machine. The quiet 12:52 PM round decoded at 16.2 tok/s on the same pack; these runs read about 20 percent lower.

Generation (CUDA lane, `accel=cuda`, int8 lm_head, DTC on). Tok/s are per-item medians and means from the engine's own counters; wall time includes engine load and calibration; card peak is the whole card (Windows and other programs hold about 0.6 GiB).

| run | group | path | prefill tok/s (median / mean) | decode tok/s (median / mean) | decode pooled | wall s | card peak MiB | RAM peak MB | against the targets |
|---|---|---|---|---|---|---|---|---|---|
| Little Nemo packed | short (cap 48) | CUDA | 66.2 / 66.1 | 12.6 / 12.6 | 12.4 | 996 | 3672 | 306.9 | meets 10 tok/s floor; below 30 tok/s target |
| Little Nemo packed | essays | CUDA | 57.2 / 60.2 | 13.7 / 13.6 | 13.6 | 361 | 3761 | 307.0 | meets 10 tok/s floor; below 30 tok/s target |
| Little Nemo packed-det-ht1 | short (cap 48) | CUDA | 70.0 / 68.2 | 12.8 / 12.6 | 12.3 | 442 | 3673 | 346.3 | meets 10 tok/s floor; below 30 tok/s target |
| Little Nemo expanded-x2 | short (cap 48) | CUDA | 70.3 / 68.5 | 13.0 / 13.3 | 13.0 | 742 | 3681 | 347.3 | meets 10 tok/s floor; below 30 tok/s target |
| Little Nemo expanded-x2 | essays | CUDA | 56.5 / 58.2 | 13.1 / 13.0 | 13.0 | 306 | 3681 | 306.9 | meets 10 tok/s floor; below 30 tok/s target |
| SuperNemo (120B MoE) subset, 24 tokens | short (10 items) | CUDA (sn-engine) | 0.58 / 0.57 | 0.17 / 0.17 | - | 1222 | 3777 | 4432.9 | BELOW 10 tok/s floor; below 30 tok/s target |

Little Nemo is above the 10 tok/s floor on every CUDA arm (12.6 to 13.7 median decode) and about 2.3 times under the 30 tok/s target. The target is for 8 GB cards, and this machine has only the 4 GB one, so the target is unmeasured. The expansion costs no measurable decode speed here (13.0 for x2 against 12.6 for packed) and the card peak is the same within 10 MiB. The SuperNemo subset (10 questions, 24-token answers, the engine as built at 7:26 PM on 11/27, not rerun on later builds) decodes at about 0.17 tok/s on the 4 GB card, so it is far below the floor; it is a correctness check on a few questions, not a speed claim, and the residency work above is what moves that number.

Teacher-forced CPU runs (layer-major, the whole probe set per configuration, 8 CPU threads at BelowNormal; tok/s is tokens scored per second, not decode speed; the hash is the evaluator's fixed-order result):

| run | arm | tokens scored | seconds | teacher-forced tok/s (CPU) | hash |
|---|---|---|---|---|---|
| A | base | 5387 | 1930.8 | 2.79 | 8db9f55f7ff5f1ed |
| A | base_h8 | 5387 | 20.3 | n/a (head only, layer outputs reused) | 9faa9a54e773ceba |
| A | X1 | 5387 | 2923.2 | 1.84 | 5bdc50289998cbce |
| A | X1_h8 | 5387 | 2892.5 | 1.86 | eddabe141efe81f1 |
| A | ref | 5387 | 3291.0 | 1.64 | c5745d4ad8cb7f91 |
| A | ref_h8 | 5387 | 3221.8 | 1.67 | dabe31241cc36bd2 |
| B | base | 1927 | 1011.9 | 1.90 | 6d559932df49ea49 |
| B | cls_in | 1927 | 926.1 | 2.08 | 23f04d27a588ee98 |
| B | cls_out | 1927 | 906.1 | 2.13 | d9ef4db5cc044274 |
| B | cls_up | 1927 | 983.6 | 1.96 | 269e8b91e3361b78 |
| B | cls_down | 1927 | 697.0 | 2.76 | e891e81dee35d4ec |
| B | cls_attn | 1927 | 544.3 | 3.54 | 49862408bbe2f297 |
| B | cls_mamba | 1927 | 926.9 | 2.08 | 2dc7dd7c5a1aa50d |
| B | cls_mlp | 1927 | 787.9 | 2.45 | a1cf3498b8023e22 |
| C | base | 3449 | 1886.3 | 1.83 | 83d9e002eb8d6847 |
| C | base_h8 | 3449 | 12.4 | n/a (head only, layer outputs reused) | 982b9cfd0d376bf6 |
| C | X1 | 3449 | 1542.2 | 2.24 | e589deed576d0222 |
| C | ref | 3449 | 2060.2 | 1.67 | 8baa1469e7aa1785 |
| C | ref_h8 | 3449 | 2690.5 | 1.28 | 48c17fba743f4704 |
| D1 | base | 198 | 930.0 | 0.21 | 8f3435aec98819af |
| D12 | base | 198 | 149.2 | 1.33 | 8f3435aec98819af |
| E | base | 1602 | 613.0 | 2.61 | 2dbb65ab1fdc090b |
| E | X1 | 1602 | 792.1 | 2.02 | e722ac0a7d2b3ed3 |
| E | ref | 1602 | 902.9 | 1.77 | 6d76c8f8052da1a5 |
| F | base | 5387 | 3555.6 | 1.52 | 8db9f55f7ff5f1ed |
| F | X2 | 5387 | 3035.5 | 1.77 | 144b547ef9b64541 |
| F | X2_h8 | 5387 | 3454.9 | 1.56 | a8ec97d1103350cb |
| G | base | 1602 | 1234.6 | 1.30 | 2dbb65ab1fdc090b |
| G | X2 | 1602 | 899.2 | 1.78 | a11f392eed364f21 |
| G | cls_mamba | 1602 | 935.3 | 1.71 | 20d91f1a27967094 |
| G | cls_mlp | 1602 | 878.7 | 1.82 | 1fdcffbabf577074 |

TF peak RAM: 3.4 GB (run A), 2.4 GB (B), 2.8 GB (C), 1.8 GB (D). Path: CPU for every teacher-forced run.

## Conditions and limits

- Engine: `nemo-engine-cuda` built from scratch-branch head `1f52c02` plus the logit-report patch committed with this doc (`littlenemo/src/engine_gui.c`, off unless `NEMO_LOGIT_TOPK` is set; quiz and essay answers were checked identical to the 12:52 PM round with it on). The packed arms and the first round used an earlier build of the same engine with the patch (`r1`); x2 used the rebuild. Greedy, chat template on, thinking off.
- Generation caps: 48 new tokens for short items, 320 for essays. "Natural stop" counts answers that ended before the cap.
- Raw bf16 has no generation arm (RAM). x1 has no generation arm (does not fit 4 GB). Sample sizes are small (n = 50 quiz; 8 to 12 in the extras): read differences of one or two items as noise.
- The int8 head test, the class sweep and the turn-end probes are teacher-forced on the CPU. They say what the model would choose, not what the CUDA lane's accumulated rounding would; the CUDA generation arms were run separately and agree on direction.
- Hand grades are mine (Grok Bot), from the essay text against the fact list; false claims are quoted in the receipts so they can be rechecked.

## Receipts

`littlenemo/tests/retention/receipts/2026-11-28/`: `gen-<arm>.json` (answers, grades, termination flags, speeds, engine command with placeholder paths), `gen-supernemo-subset.json`, `tf-<run>.tsv` (run tables with hashes), `tf-<run>-items.tsv` (per-item scores, no top-k vectors), `semantic-<arm>.json` (hand grades), `summary.json` (all tables as data). Rebuild the tables with `analyze.py` and the receipts with `make_receipts.py`.

Run hashes (teacher-forced evaluator):

| run | arm | set | hash |
|---|---|---|---|
| A | base | quiz, cand, para, multi, pass, calib (139 items) | `8db9f55f7ff5f1ed` |
| A | base_h8 | quiz, cand, para, multi, pass, calib (139 items) | `9faa9a54e773ceba` |
| A | X1 | quiz, cand, para, multi, pass, calib (139 items) | `5bdc50289998cbce` |
| A | X1_h8 | quiz, cand, para, multi, pass, calib (139 items) | `eddabe141efe81f1` |
| A | ref | quiz, cand, para, multi, pass, calib (139 items) | `c5745d4ad8cb7f91` |
| A | ref_h8 | quiz, cand, para, multi, pass, calib (139 items) | `dabe31241cc36bd2` |
| B | base | quiz, cand (62) | `6d559932df49ea49` |
| B | cls_in | quiz, cand (62) | `23f04d27a588ee98` |
| B | cls_out | quiz, cand (62) | `d9ef4db5cc044274` |
| B | cls_up | quiz, cand (62) | `269e8b91e3361b78` |
| B | cls_down | quiz, cand (62) | `e891e81dee35d4ec` |
| B | cls_attn | quiz, cand (62) | `49862408bbe2f297` |
| B | cls_mamba | quiz, cand (62) | `2dc7dd7c5a1aa50d` |
| B | cls_mlp | quiz, cand (62) | `a1cf3498b8023e22` |
| C | base | long (4) | `83d9e002eb8d6847` |
| C | base_h8 | long (4) | `982b9cfd0d376bf6` |
| C | X1 | long (4) | `e589deed576d0222` |
| C | ref | long (4) | `8baa1469e7aa1785` |
| C | ref_h8 | long (4) | `48c17fba743f4704` |
| D1 | base | 6 quiz items, 1 thread | `8f3435aec98819af` |
| D12 | base | 6 quiz items, 12 threads | `8f3435aec98819af` |
| E | base | term (50) | `2dbb65ab1fdc090b` |
| E | X1 | term (50) | `e722ac0a7d2b3ed3` |
| E | ref | term (50) | `6d76c8f8052da1a5` |
| F | base | A set (139) | `8db9f55f7ff5f1ed` |
| F | X2 | A set (139) | `144b547ef9b64541` |
| F | X2_h8 | A set (139) | `a8ec97d1103350cb` |
| G | base | term (50) | `2dbb65ab1fdc090b` |
| G | X2 | term (50) | `a11f392eed364f21` |
| G | cls_mamba | term (50) | `20d91f1a27967094` |
| G | cls_mlp | term (50) | `1fdcffbabf577074` |

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.
