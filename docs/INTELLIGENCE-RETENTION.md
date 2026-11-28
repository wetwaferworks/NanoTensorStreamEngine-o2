# Intelligence retention: how much the compressed models keep

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.

NTSE squeezes big models into small packs. This report answers the question that matters most: after squeezing, how much of the original model's intelligence is left, and how fast does it run? Speed and accuracy are shown side by side for every configuration, against the original ("raw") model.

In short, the shipping Little Nemo pack keeps about 29 percent of the original model's first-answer hits (31 of 119, against 106). The small x2 refinement that still fits a 4 GB card keeps 49 percent (52 of 119), and the larger x1 refinement, which only runs on the CPU, keeps 56 percent (59 of 119). Both packs that fit the card decode at about 13 tokens per second when the machine is shared, and 16.2 on a quieter morning (pooled decode on the 50-question set). Almost all of the loss comes from compressing the Mamba `in_proj` and `out_proj` weights: putting the original values back in those alone returns the quiz to the original's score (47 of 50) and restores the habit of stopping at the end of a turn. Giving those projections more bits, in a size that still fits 4 GB, is the next step. SuperNemo's answers on the card match the CPU path (identical whole-model hash); the speed features do not change accuracy.

Everything here was measured on one bench machine (RTX 3050 Ti Laptop 4 GB, i5-12450H, 16 GB RAM) on 27 and 28 November 2026. All timings are real-world: other jobs shared the CPU and the card while these ran, so read the ratios more than the last digit. Anything we did not measure is marked pending. The full detail, with every table and receipt, is in [RETENTION-BATTERY-2026-11-27.md](RETENTION-BATTERY-2026-11-27.md).

## The short version

1. **Little Nemo (Nemotron Nano 9B v2) keeps about 29 percent of the original's first-answer hits in the shipping pack, and about 49 percent with the small "x2" refinement that still fits the 4 GB card.** Raw gets 106 of 119 right on the first answer word, the shipping pack 31, x2 52, and the larger x1 (CPU only) 59.
2. **Speed is fine, accuracy is the gap.** Little Nemo decodes at about 12.6 to 13.9 tokens per second on the card while sharing the machine (16.2 on a quieter morning, pooled decode on the 50-question set), above the 10 tokens per second floor. x2 costs no measurable speed.
3. **We found where the loss lives.** The Mamba `in_proj` and `out_proj` weights are almost all of it. Put the original weights back in just those, in all 27 Mamba layers, and the quiz returns to the original's score (17 to 47 of 50) and the model stops talking at the right moment again (stop probability 0.017 to 0.9997). That pack was not built at speed yet; it is the next step.
4. **The output head is not the problem.** An 8-bit head against a 16-bit head changes the loss by at most 0.008 per token.
5. **SuperNemo (Nemotron 3.5 Super VL, 120B-A12B) gives the same answers on the card as on the CPU** (identical whole-model hash) and passes its 5-question quiz subset, at about 0.1 to 0.3 tokens per second on this card. A raw SuperNemo comparison was not possible here.

## How we measure

- **Teacher-forced (exact).** We hand the model the correct answer one token at a time and score what it would have said at each step. This is repeatable to the last bit and does not depend on sampling. "First-token hit" means the model's top choice for the first answer word is the right one, so a model that writes "The capital of Australia is Canberra." misses even though it knows the answer. "Loss per token" is the average surprise: lower is better.
- **Generated (what a user sees).** The model writes its own answer on the card, and a grader looks for the right answer in the first line. This is more forgiving: a sentence that contains the answer passes.
- **Raw is teacher-forced only.** The original 9B model is about 18 GB in bf16 and does not fit this machine's 16 GB, so it has exact scores but no generated run and no card speed.
- The test set has 203 items (50 quiz questions, 12 earlier misses, 39 reworded questions, 12 multi-step sums, 10 passage questions, 16 calibration, 10 instruction, 4 long-context and 50 stop-habit probes) plus 10 essays. Small samples: treat a one or two item difference as noise.

## Little Nemo (9B): speed and accuracy side by side

Configurations: **raw** is the original bf16 model. **Packed** is the shipping 2.2-bit pack. **x2** adds a small refinement to layers 8 and 11 (73.4 MiB) and fits the card. **x1** refines six layers, which does not fit the 4 GB card. All rows use the 8-bit output head the card uses (the 16-bit head changes almost nothing, see below).

| configuration | runs on the 4 GB card | decode tok/s (real-world) | first-token hits, of 119 | share of raw | loss per token | quiz first token, of 50 | share of raw gap in loss closed |
|---|---|---|---|---|---|---|---|
| raw bf16 (original) | no (18 GB) | n/a | 106 (89%) | 100% | 0.267 | 47 | 100% |
| packed (shipping) | yes | 12.6 median (16.2 quieter morning) | 31 (26%) | 29% | 1.631 | 16 | 0% |
| x2, layers 8 and 11 | yes | 13.0 median (13.5 pooled in the expansion round) | 52 (44%) | 49% | 1.224 | 28 | 30% |
| x1, six layers | no (CPU only) | n/a | 59 (50%) | 56% | 1.107 | 34 | 38% |
| packed + original Mamba in/out | speed pending (not built as a pack) | pending | pending | pending | pending | 47 (16-bit head, quiz only) | 100% |

"Share of raw" is the configuration's hits divided by the original's 106. The last column is how much of the distance between packed and raw loss the configuration closes. The last row comes from a test that swaps the original weights into the packed model on the CPU, so it shows what is possible, not a pack you can download yet.

### Full scorecard, Little Nemo (exact scoring, 8-bit head, all the measures in one place)

| measure | raw | packed | x2 | x1 | restored Mamba in/out |
|---|---|---|---|---|---|
| first-token hits, all 119 | 106 | 31 | 52 | 59 | not run |
| quiz first token (of 50) | 47 | 16 | 28 | 34 | 47 (16-bit head) |
| loss per token (NLL) | 0.267 | 1.631 | 1.224 | 1.107 | not run |
| expected calibration error (ECE) | 0.055 | 0.265 | 0.182 | 0.131 | not run |
| mean top-1 confidence | 0.93 | 0.53 | 0.59 | 0.59 | not run |
| multi-step sums (of 12) | 9 | 1 | 2 | 2 | not run |
| passage reasoning (of 10) | 8 | 0 | 2 | 2 | not run |
| calibration questions answered right (of 8) | 8 | 3 | 5 | 5 | not run |
| reworded groups right all three ways (of 13) | 11 | 2 | 4 | 4 | not run |
| long-context recall, first token (of 4) | 4 | 0 | not run | 4 | not run |
| mean chance of stopping after the answer | 1.0 | 0.017 | 0.022 | 0.024 | 0.9997 |
| stop is the top choice (of 50) | 50 | 0 | 1 | 1 | 50 |
| quiz, generated on the card (of 50) | not run | 36 | 34 | not run | not run |
| essays, keyword grader (of 10) | not run | 4 | 8 | not run | not run |
| essays, read by hand (of 10) | not run | 3 | 4 | not run | not run |
| essay facts found (of 62) | not run | 36 | 43 | not run | not run |
| held-out loss, 24 items (base pack 64.38) | not run | 64.38 | 50.45 | not run | not run |
| held-out first token (of 24) | not run | 9 | 13 | not run | not run |
| decode tok/s on the card (real-world) | n/a | 12.6 | 13.0 | n/a | pending |
| card peak (MiB) | n/a | 3672 | 3681 | n/a | pending |
| same result at 1 and 12 threads | not run | identical (exact scorer, 6 items; 72 of 72 generated answers) | not run | not run | not run |

The ECE and confidence rows use the 8-bit-head runs (raw with the 16-bit head: ECE 0.065). The held-out rows compare the base pack against x2 on 24 items the refinement was not tuned on. The thread check was run on the packed pack only; every other configuration used the same fixed-order scorer, whose hashes are listed in appendix K. "Not run" means we did not measure it, not that the result is zero. The `restored Mamba in/out` column is a CPU experiment on the 50 stop-habit and quiz items only.

### What the user sees (generated on the card)

| test | packed | x2 | raw |
|---|---|---|---|
| 50 quiz questions, first line | 36/50 | 34/50 | not run |
| same-engine control for the quiz | 36/50 | n/a | n/a |
| 10 essays, keyword grader | 4/10 | 8/10 | not run |
| 10 essays, read by hand for false claims | 3/10 | 4/10 | not run |
| essay facts found | 36/62 | 43/62 | not run |
| 12 multi-step sums | 1/12 | 3/12 | 9/12 (exact) |
| long-context recall | 3/4 | 4/4 | 4/4 (exact) |
| 39 reworded questions | 25/39 | 23/39 | 34/39 (exact) |
| answers that stop on their own | 8 of 141 | 13 of 141 | not run |
| says "I don't know" for made-up things | 8/8 | 8/8 | 8/8 (exact) |
| decode tok/s, questions (pooled) | 12.4 | 13.0 | n/a |
| decode tok/s, essays (pooled) | 13.6 | 13.0 | n/a |
| card peak | 3672 MiB | 3681 MiB | n/a |

x2 is a mixed result and we say so: more correct facts in long answers (essays 4 to 8 of 10 by keyword, though only 3 to 4 of 10 when read carefully, because half of the passes still contain a false claim), no net gain on one-line questions (36 to 34 of 50: three fixed, five lost). On a held-out set of 24 items that the refinement was not tuned on, loss fell from 64.38 to 50.45 and first-token hits rose from 9 to 13 of 24.

## Where the loss lives

We put the original weights back, one group of tensors at a time, into an otherwise packed model and scored the 50 quiz questions (exact, on the CPU).

| restored to original | quiz first token, of 50 | share of the packed-to-raw gap closed |
|---|---|---|
| nothing (packed) | 17 | 0% |
| Mamba `in_proj` | 29 | 46% |
| Mamba `out_proj` | 18 | 3% |
| MLP `up_proj` | 19 | 29% |
| MLP `down_proj` | 30 | 41% |
| attention layers | 24 | 10% |
| all MLP projections | 32 | 57% |
| **Mamba `in_proj` and `out_proj` together** | **47** | **100%** |

The Mamba projections only work as a pair: each alone closes a part, together they restore the original's score. The 27 Mamba layers are the cheapest big win.

### The model forgets to stop

After the correct answer and its newline, the original model puts probability 1.0 on its end-of-turn token on 50 of 50 items. The packed model puts a mean of 0.017 there and chooses it on 0 of 50, so it keeps writing blank lines or "(150 words)". This comes from the compression, not the engine.

| configuration | mean chance of stopping | stop is the top choice |
|---|---|---|
| raw | 1.0 | 50/50 |
| packed | 0.017 | 0/50 |
| x1 (six layers refined) | 0.024 | 1/50 |
| x2 (two layers refined) | 0.022 | 1/50 |
| MLP up/down restored | 0.081 | 4/50 |
| **Mamba in/out restored** | **0.9997** | **50/50** |

### Other findings

- **Head precision does not matter.** Swapping the 8-bit head for a 16-bit one moves the loss by at most 0.008 per token and changes at most one item in 119, for raw, packed, x1 and x2 alike.
- **Confidence drifts.** The packed model's confidence no longer tracks being right: expected calibration error 0.256 against 0.065 for the original (mean confidence 0.53 against 0.93).
- **Wording drift.** Asked to "answer with just the answer", the original writes the bare answer on 47 of 50. The packed model does on 17, and on 18 of its 33 misses the first word is "The". That is why the exact first-token score (16 of 50) sits well under the forgiving generated score (36 of 50). Both are real; they measure different things.
- **Reworded questions.** The original answers the same fact right all three ways in 11 of 13 groups. Packed manages 2 of 13 (exact), x2 4 of 13.
- **Results repeat.** The exact scorer gives the same hash on 1 thread and on 12 (`8f3435aec98819af`), and 72 of 72 generated answers matched between default threads and one thread.

## SuperNemo (120B-A12B): speed and accuracy

SuperNemo is a mixture of 512 experts per layer. Its engine, expert cache and early expert fetch are described in [SUPERNEMO.md](SUPERNEMO.md).

| measure | result | note |
|---|---|---|
| same answers as the CPU path | **yes**, whole-model hash `b0c32d2989f86f69` on CPU and on the card, 4 and 12 threads | proves the speed features do not change the output; it compares against the same pack on the CPU, not against the raw model |
| quiz subset, 5 questions (final build, MTP on, early fetch mode 4) | **5/5** | prefill 0.55 and decode 0.081 tok/s pooled, real-world; card peak 3694 MiB |
| essays, 2 | 2/2 | earlier round, not rerun after the latest build |
| 10-question mixed subset (earlier build) | 7/10 | misses: one arithmetic answer (220 for 260) and two made-up-entity items answered "I don not know." instead of the exact phrase |
| decode speed, 17-token runs | 0.16 to 0.19 tok/s median | see the table in SUPERNEMO.md |
| against the raw model | pending | the raw checkpoint is far beyond this machine |
| relative weight error of the pack | about 0.30 | at the recipe used |

JD's call: about 0.1 to 0.3 tokens per second is realistic for SuperNemo on a 4 GB card. The 10 tokens per second floor was never a SuperNemo target.

## Speed against the targets

| goal | result |
|---|---|
| Little Nemo at least 10 tok/s on the 4 GB card | met: 12.6 to 13.9 tok/s median, shared machine; 16.2 on a quieter morning (pooled decode, questions) |
| Little Nemo 30+ tok/s on 8 GB cards | not measured (only a 4 GB card here) |
| SuperNemo | 0.1 to 0.3 tok/s on the 4 GB card, as agreed |

## Next: give the Mamba projections more bits

The finding above says to store the Mamba `in_proj` and `out_proj` at higher precision. Doing it everywhere at 8 bits does not fit. Sizes below are exact counts for the 27 Mamba layers; the new pack is the current 2,038.6 MiB pack plus the extra.

| variant | extra over the current pack | resulting pack | fits the 4 GB card |
|---|---|---|---|
| current (PQ, 2.2 bits) | 0 | 2,038.6 MiB | yes |
| 8-bit `dt` rows only | +10.7 MiB | 2,049.3 MiB | yes (about 14.8 MiB on the card) |
| 8-bit `B`, `C` and `dt` rows | +181.7 MiB | 2,220.3 MiB | no (the card has about 88 to 162 MiB to spare after the head) |
| all Mamba projections at 3 bits | +431 MiB | 2,469.6 MiB | no |
| all Mamba projections at 4 bits | +905 MiB | 2,943.6 MiB | no |
| `z` or `x` rows at 8 bits (each) | +855 MiB | 2,893.6 MiB | no |
| all Mamba projections at 6 bits | +1,854 MiB | 3,892.6 MiB | no |
| **all Mamba projections at 8 bits** | **+2,746 MiB** | **4,784.5 MiB** | **no** |

Moving the output head to the CPU would free about 560 MiB, but it ran at about 5.9 tokens per second in our test, under the floor. So the open question is which small slice of the projections carries the loss. The accuracy and speed of each variant are **pending**: the runs that would answer it were stopped before they finished, and nothing from them is reported here.

### New options for this work

| option | what it does | default |
|---|---|---|
| `NEMO_LOGIT_TOPK=K` (1 to 20) | print the top K guesses and the stop probability for every generated token; read-only | off |
| `NEMO_GPU_I8ROWS=<slice>[@layers]` | keep a slice of every Mamba `in_proj` (`dt`, `B`, `C`, `BC`, `BCdt`, `x`, `z`, `all`) at 8 bits per row, built from the original weights; logs its size and hash | off |

The CPU scorer takes matching specs, and its result for the whole `in_proj` matches the restored-weights run bit for bit, which checks the new path.

## Not yet measured

- Accuracy and speed of the 8-bit and partial Mamba-projection packs (sizes are known; the scoring runs were stopped before they finished).
- Generated, user-visible answers from the original (raw) models. The 9B original is about 18 GB and does not fit this machine's 16 GB of RAM.
- Little Nemo speed on 8 GB or larger cards (the 30+ tok/s goal). Only a 4 GB card was available here.
- A same-card comparison with llama.cpp. That is waiting on a go to download and run it.
- SuperNemo against its raw checkpoint (the raw model is far larger than this machine).
- Samples are small (50 quiz questions, 8 to 12 in the extras). One SuperNemo run can move by up to 60 percent from machine load alone.

## Receipts

Scrubbed answers, grades, speeds and run hashes are in `littlenemo/tests/retention/receipts/2026-11-28/` and `littlenemo/tests/quiz/receipts/2026-11-27/`. No weights, packs, raw logs or machine paths are in the repository.

---

# Appendix: every table

The sections above are the summary. Everything below is the full record, one table per test, for every configuration that ran. Numbers are copied from the detailed battery report ([RETENTION-BATTERY-2026-11-27.md](RETENTION-BATTERY-2026-11-27.md)) and the expansion report ([EXPAND-REFINE.md](EXPAND-REFINE.md)).

Terms: **x1** is the six-layer refinement (CPU only), **x2** the two-layer refinement that fits the card, **int8 head** the 8-bit output head the card uses, **bf16 head** the 16-bit head, **ECE** expected calibration error (lower is better), **NLL** loss per token (lower is better).

## A. Exact scores for every configuration (teacher-forced, CPU)

Set sizes: quiz 50, reworded questions 39, multi-step 12, passage 10, calibration 8 answerable; "all" is 119 items.

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

## B. Generated on the card (greedy, graded on the first answer line)

| arm (CUDA, greedy) | quiz | paraphrase | multi-step | passage | calibration | instructions (full) | instructions (first line only) | long context |
|---|---|---|---|---|---|---|---|---|
| packed | 36/50 | 25/39 | 1/12 | 4/10 | 15/16 | 2/10 | 4/10 | 3/4 |
| packed-det-ht1 | 36/50 | - | 1/12 | 4/10 | - | - | - | - |
| expanded-x2 | 34/50 | 23/39 | 3/12 | 4/10 | 13/16 | 2/10 | 6/10 | 4/4 |

What it means: the original gets 106 of 119 first tokens right, the packed pack 32, x1 60. Multi-step arithmetic and novel-passage reasoning are hit hardest (packed 1/12 and 0/10 teacher-forced, 1/12 and 4/10 generated with the lenient grader, against 9/12 and 8/10 for the original). Expansion recovers part of it (x1 60 and x2 53 of 119 first tokens, against 32 packed). The generated quiz score for x2 differs from packed by two questions, which is noise at n = 50, and the lenient grader accepts a sentence that contains the answer, so the teacher-forced columns are the better measure of what the expansion restores.

## C. The stop habit, in generation

| generation arm | answers that ended on their own (of cap-limited runs) | answers with a newline | mean P(stop) after the newline | median | stop is top-1 | median stop rank |
|---|---|---|---|---|---|---|
| packed | 8/141 | 139 | 0.0266 | 0.00044 | 5/139 | 55 |
| packed-det-ht1 | 3/72 | 70 | 0.0178 | 0.00048 | 2/70 | 47 |
| expanded-x2 | 13/141 | 138 | 0.0766 | 0.00154 | 12/138 | 25 |

## D. Head precision, all four weight sets

| weights | bf16-class head: first token | NLL/tok | int8 per-row head: first token | NLL/tok | NLL change |
|---|---|---|---|---|---|
| raw bf16 | 106/119 | 0.2675 | 106/119 | 0.2669 | -0.0006 |
| packed | 32/119 | 1.6232 | 31/119 | 1.6309 | 0.0077 |
| expanded x1 (6 layers) | 60/119 | 1.1031 | 59/119 | 1.1071 | 0.004 |
| expanded x2 (2 layers, 4 tensors) | 53/119 | 1.2187 | 52/119 | 1.2238 | 0.0051 |

What it means: the head is not where the quality goes. Raising it from the int8 per-row head to bf16 buys at most 0.008 nats per token and changes at most one item. Spending 1.1 GB of VRAM on a bf16 head is not worth it.

## E. Sensitivity by tensor class (with the loss)

Row names: `cls_in` and `cls_out` are the Mamba `in_proj` and `out_proj`, `cls_up` and `cls_down` the MLP `up_proj` and `down_proj`, `cls_attn` the attention layers, `cls_mamba` and `cls_mlp` each pair together, `ref` the original model.

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
| ref (the matched main-set run) | 47/50 | 23.753 | 100% |

What it means: Mamba `in_proj` and `out_proj` together are the problem. Restoring both gets the original's score back (47/50, NLL 23.8 against 23.75), though each alone gives only a part (46 percent and 3 percent), so the errors combine rather than add. MLP weights are second (57 percent together). The four attention layers matter least. This agrees with where the expansion work put its refinement (layers 8, 9, 10, 11, 14, 21) and says the cheapest big win is to give the Mamba projections more bits, not the head and not attention.

## F. The 14 quiz questions the shipping pack got wrong: where the right answer ranks

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

## G. Essays, keyword grader against a careful read

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

## H. Passages and reworded questions

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

## I. Calibration, abstention, multi-step, instructions, long context, repeatability

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

## J. Speed of every run (real-world)

Recorded against JD's floor (**10 tok/s on the 4 GB card**) and target (**30+ tok/s on 8 GB cards**). All timings are real-world: another worker's fid-quant was running on the CPU for the whole battery, the teacher-forced runs shared the CPU (8 threads, BelowNormal), and JD may have been on the machine. The quieter 12:52 PM ET 27 November round decoded at 16.2 tok/s pooled on the 50-question set (same pack, decode time only; see the note under appendix M); these later runs read about 20 percent lower under a more heavily shared machine.

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

Teacher-forced CPU runs (the whole probe set per configuration, 8 CPU threads at below-normal priority; tok/s here is tokens scored per second, not decode speed; the hash is the scorer's fixed-order result):

| test | arm | tokens scored | seconds | teacher-forced tok/s (CPU) | hash |
|---|---|---|---|---|---|
| main set | base | 5387 | 1930.8 | 2.79 | 8db9f55f7ff5f1ed |
| main set | base_h8 | 5387 | 20.3 | n/a (head only, layer outputs reused) | 9faa9a54e773ceba |
| main set | X1 | 5387 | 2923.2 | 1.84 | 5bdc50289998cbce |
| main set | X1_h8 | 5387 | 2892.5 | 1.86 | eddabe141efe81f1 |
| main set | ref | 5387 | 3291.0 | 1.64 | c5745d4ad8cb7f91 |
| main set | ref_h8 | 5387 | 3221.8 | 1.67 | dabe31241cc36bd2 |
| class sweep | base | 1927 | 1011.9 | 1.90 | 6d559932df49ea49 |
| class sweep | cls_in | 1927 | 926.1 | 2.08 | 23f04d27a588ee98 |
| class sweep | cls_out | 1927 | 906.1 | 2.13 | d9ef4db5cc044274 |
| class sweep | cls_up | 1927 | 983.6 | 1.96 | 269e8b91e3361b78 |
| class sweep | cls_down | 1927 | 697.0 | 2.76 | e891e81dee35d4ec |
| class sweep | cls_attn | 1927 | 544.3 | 3.54 | 49862408bbe2f297 |
| class sweep | cls_mamba | 1927 | 926.9 | 2.08 | 2dc7dd7c5a1aa50d |
| class sweep | cls_mlp | 1927 | 787.9 | 2.45 | a1cf3498b8023e22 |
| long context | base | 3449 | 1886.3 | 1.83 | 83d9e002eb8d6847 |
| long context | base_h8 | 3449 | 12.4 | n/a (head only, layer outputs reused) | 982b9cfd0d376bf6 |
| long context | X1 | 3449 | 1542.2 | 2.24 | e589deed576d0222 |
| long context | ref | 3449 | 2060.2 | 1.67 | 8baa1469e7aa1785 |
| long context | ref_h8 | 3449 | 2690.5 | 1.28 | 48c17fba743f4704 |
| thread check, 1 thread | base | 198 | 930.0 | 0.21 | 8f3435aec98819af |
| thread check, 12 threads | base | 198 | 149.2 | 1.33 | 8f3435aec98819af |
| stop probe | base | 1602 | 613.0 | 2.61 | 2dbb65ab1fdc090b |
| stop probe | X1 | 1602 | 792.1 | 2.02 | e722ac0a7d2b3ed3 |
| stop probe | ref | 1602 | 902.9 | 1.77 | 6d76c8f8052da1a5 |
| x2 main set | base | 5387 | 3555.6 | 1.52 | 8db9f55f7ff5f1ed |
| x2 main set | X2 | 5387 | 3035.5 | 1.77 | 144b547ef9b64541 |
| x2 main set | X2_h8 | 5387 | 3454.9 | 1.56 | a8ec97d1103350cb |
| x2 stop probe | base | 1602 | 1234.6 | 1.30 | 2dbb65ab1fdc090b |
| x2 stop probe | X2 | 1602 | 899.2 | 1.78 | a11f392eed364f21 |
| x2 stop probe | cls_mamba | 1602 | 935.3 | 1.71 | 20d91f1a27967094 |
| x2 stop probe | cls_mlp | 1602 | 878.7 | 1.82 | 1fdcffbabf577074 |

Peak RAM of the exact scorer: 3.4 GB (main set), 2.4 GB (class sweep), 2.8 GB (long context), 1.8 GB (thread check). Path: CPU for every teacher-forced run.

## K. Run hashes (exact scorer)

Run hashes (teacher-forced evaluator):

| test | arm | set | hash |
|---|---|---|---|
| main set | base | quiz, cand, para, multi, pass, calib (139 items) | `8db9f55f7ff5f1ed` |
| main set | base_h8 | quiz, cand, para, multi, pass, calib (139 items) | `9faa9a54e773ceba` |
| main set | X1 | quiz, cand, para, multi, pass, calib (139 items) | `5bdc50289998cbce` |
| main set | X1_h8 | quiz, cand, para, multi, pass, calib (139 items) | `eddabe141efe81f1` |
| main set | ref | quiz, cand, para, multi, pass, calib (139 items) | `c5745d4ad8cb7f91` |
| main set | ref_h8 | quiz, cand, para, multi, pass, calib (139 items) | `dabe31241cc36bd2` |
| class sweep | base | quiz, cand (62) | `6d559932df49ea49` |
| class sweep | cls_in | quiz, cand (62) | `23f04d27a588ee98` |
| class sweep | cls_out | quiz, cand (62) | `d9ef4db5cc044274` |
| class sweep | cls_up | quiz, cand (62) | `269e8b91e3361b78` |
| class sweep | cls_down | quiz, cand (62) | `e891e81dee35d4ec` |
| class sweep | cls_attn | quiz, cand (62) | `49862408bbe2f297` |
| class sweep | cls_mamba | quiz, cand (62) | `2dc7dd7c5a1aa50d` |
| class sweep | cls_mlp | quiz, cand (62) | `a1cf3498b8023e22` |
| long context | base | long (4) | `83d9e002eb8d6847` |
| long context | base_h8 | long (4) | `982b9cfd0d376bf6` |
| long context | X1 | long (4) | `e589deed576d0222` |
| long context | ref | long (4) | `8baa1469e7aa1785` |
| long context | ref_h8 | long (4) | `48c17fba743f4704` |
| thread check, 1 thread | base | 6 quiz items, 1 thread | `8f3435aec98819af` |
| thread check, 12 threads | base | 6 quiz items, 12 threads | `8f3435aec98819af` |
| stop probe | base | term (50) | `2dbb65ab1fdc090b` |
| stop probe | X1 | term (50) | `e722ac0a7d2b3ed3` |
| stop probe | ref | term (50) | `6d76c8f8052da1a5` |
| x2 main set | base | A set (139) | `8db9f55f7ff5f1ed` |
| x2 main set | X2 | A set (139) | `144b547ef9b64541` |
| x2 main set | X2_h8 | A set (139) | `a8ec97d1103350cb` |
| x2 stop probe | base | term (50) | `2dbb65ab1fdc090b` |
| x2 stop probe | X2 | term (50) | `a11f392eed364f21` |
| x2 stop probe | cls_mamba | term (50) | `20d91f1a27967094` |
| x2 stop probe | cls_mlp | term (50) | `1fdcffbabf577074` |

## L. SuperNemo, the 10-item mixed subset (earlier build, real-world)

Seven of ten pass. Item 3 is a real miss (220 for 260). Items 7 and 8 reply "I don not know." where the instruction asked for the exact phrase "I don't know", so they fail the exact-match grader even though the model abstained. Wall 1,222 s, card peak 3777 MiB, engine RAM peak 4,433 MB, decode pooled 0.129 tok/s, prefill pooled 0.587 tok/s.

| item | kind | answer given | grade | decode tok/s | prefill tok/s |
|---|---|---|---|---|---|
| 1 | science | Jupiter | pass | 0.1491 | 0.496 |
| 2 | capital | Ottawa | pass | 0.1442 | 0.411 |
| 3 | arithmetic | 220 | fail | 0.1733 | 0.627 |
| 4 | arithmetic | 30 | pass | 0.2089 | 0.583 |
| 5 | passage | Dov | pass | 0.206 | 0.686 |
| 6 | passage | 19 | pass | 0.158 | 0.737 |
| 7 | made-up entity | I don not know. | fail | 0.1817 | 0.651 |
| 8 | made-up entity | I don not know. | fail | 0.1394 | 0.578 |
| 9 | real fact, abstain allowed | Leonardo da Vinci | pass | 0.1722 | 0.504 |
| 10 | format | { | pass | 0.1625 | 0.473 |

The 5-question quiz subset on the final build (MTP on, early fetch mode 4) passes 5 of 5: 12 (square root of 144), Canberra, He, 1776, Jupiter. Pooled prefill 0.547 and decode 0.081 tok/s, card peak 3694 MiB, engine RAM peak 5087 MB, wall 575 s. The earlier quiz round, taken before the speed features, also passed 5 of 5 and the two essays (seasons, DNA) 2 of 2, at prefill 0.917 and decode 0.209 tok/s on a quiet machine. Speed per early-fetch mode is in [SUPERNEMO.md](SUPERNEMO.md).


## M. Little Nemo, base pack against x2, 50 questions and 10 essays on the card

Same engine, same flags, one engine per phase; the base pack control reproduces the morning round exactly (all 50 answers identical).

| | control (base pack) | x2 |
|---|---|---|
| questions | 36/50 | 34/50 |
| math | 7/10 | 5/10 |
| capital | 9/10 | 8/10 |
| chemical symbol | 8/8 | 7/8 |
| date | 6/7 | 6/7 |
| units | 2/7 | 3/7 |
| science | 3/7 | 4/7 |
| general | 1/1 | 1/1 |
| essays | 4/10 | 8/10 |
| essay facts | 36/62 | 43/62 |
| forbidden facts | 0 | 0 |
| held-out loss (24 items) | 64.376 | 50.451 |
| held-out first token (24 items) | 9/24 | 13/24 |

Questions: three misses fixed (grams in a kilogram, millilitres in a litre, legs on an insect) and five passes lost (1000 / 8 gave 625, 3 cubed gave 8, Kenya gave "Nairo", helium gave "H", feet in a yard gave "1 foot"), a net two lower. Essays improved on five topics, stayed level on four and fell on one (photosynthesis: a title and then blank lines). Which layers to refine was chosen by loss on 32 tuning items; layers 8 to 11 together were best (loss 64.48 against 111.66 for the base pack on that set), and the pair of layers 8 and 11 is the best two-layer choice that fits the card (77.20 on the tuning set).

| speed run | path | decode tok/s (pooled) | prefill tok/s (pooled) | card peak MiB | RAM peak MB |
|---|---|---|---|---|---|
| morning base, questions | CUDA | 16.22 | 73.9 | 3841 | 306.5 |
| morning base, essays | CUDA | 16.10 | 58.7 | 3841 | 306.6 |
| control base, questions | CUDA | 13.35 | 43.8 | 3596 | 306.7 |
| control base, essays | CUDA | 13.99 | 41.9 | 3596 | 307.0 |
| x2, questions | CUDA | 13.51 (min 8.86) | 43.2 | 3681 | 345.1 |
| x2, essays | CUDA | 13.95 (min 13.45) | 40.5 | 3681 | 306.7 |

All timings are real-world. The morning row is the 12:52 PM ET 27 November quiz on the shipping pack: 16.22 tok/s questions and 16.10 tok/s essays, pooled generated tokens over decode time only, from `littlenemo/tests/quiz/receipts/2026-11-27/little-nemo/` (`pooled.decode_tps_pooled`; also 16.2 and 16.1 in [LITTLE-NEMO-ROUND-2026-11-27.md](LITTLE-NEMO-ROUND-2026-11-27.md)). That same receipt has a `summary.decode_tps_pooled` field of 11.63 and 15.64, which divides generated tokens by prefill time plus decode time and is not a decode rate; it was copied into an earlier table by mistake. No new run was needed. The later control and x2 rows were measured back to back on a more heavily shared machine (about 13.4 to 14.0 tok/s) and are the fair comparison with each other.
