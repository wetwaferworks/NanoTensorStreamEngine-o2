# Little Nemo and SuperNemo verifiable round, 2026-11-27

JD asked for a profiling and testing round with answers that can be checked. This is the Little Nemo half: 50 short questions with one gold answer each and 10 short essays graded against a fixed list of facts. The question set, rubric, graders and runner are in `littlenemo/tests/quiz/` (`questions50.json`, `essays.json`, `grade.py`, `grade_essay.py`, `run_round.py`). No model, pack or machine path is stored in the tree. The SuperNemo half (5 questions and 2 essays from the same files) is [further down](#supernemo-the-same-graders-on-a-smaller-set).

## Headline

| Part | Score | Notes |
|---|---|---|
| 50 short questions | **36/50** | 36 pass, 14 fail, graded on the first answer line |
| 10 essays | **4/10 pass**, 36 of 62 required facts | 0 forbidden claims; a pass needs 70 percent of the facts and no forbidden claim |
| Decode speed | 16.2 tokens/s pooled on questions, 16.1 on essays | CUDA lane, RTX 3050 Ti Laptop 4 GB |
| Prefill speed | 73.9 tokens/s pooled on questions, 58.7 on essays | batched prefill, 64-token chunks |
| Wall time | 321 s for the whole round | 92 s questions and 228 s essays, each including about 22 s engine load and calibration |
| Memory | engine RAM peak 306.6 MB; whole-card VRAM peak 3841 MiB of 4096 | VRAM is the whole card (Windows and other programs hold about 0.6 GiB) |

This is the compressed pack, not the raw model. The 2026-11-26 record already showed the pack losing facts the raw bf16 weights keep (7/10 against 10/10 on the smoke quiz, see [ACCEL-CUDA.md](ACCEL-CUDA.md)), and this round says the same thing at a larger size. The losses are in the pack's 2.2-bit projections, not in the GPU: the 2026-11-26 CPU reference on the decoded pack gave the same wrong answers.

## Conditions

- Engine: `nemo-engine-cuda` built from scratch-branch head `5214304` with the same steps as `accel/build-nemo-gpu.ps1`, written to a private build folder.
- Lane: CUDA, chosen by calibration (`accel=cuda`, `dtc=on expand=auto`, `full_path=fast`, `prefill_chunk=64`, int8 `lm_head`). Pack: the Nemotron Nano 9B v2 PQ d4 pack from the 2026-11-26 CUDA probe (`nemo9b-pq4-L.nq2`, 2,048,980,932 bytes), the same pack as the published 7/10 record.
- Greedy, chat template on, thinking off, `RESET` between items, one engine process per part.
- Questions: `--max-new 16`, graded on the first non-empty answer line. Essays: `--max-new 320`.
- Engine and runner at BelowNormal priority. Started 12:52 PM ET. Other load during the run: the SuperVL post-pack check (`sn_packcheck`, CPU) was finishing, and a LithoKraft window was open (GPU load about 20 percent before the run started). Other GPU runners were parked. Mean GPU load during the round was 84 percent.
- Reset check: questions 4, 5 and 8 were rerun in a fresh engine in a different order and gave the same answers (7, 42, 49), so the wrong answers are the model's, not state left over from the question before.

## Short questions

| # | Category | Gold | First answer line | Grade | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|---|---|
| 1 | math | 144 | 144 | pass | 44.5 | 15.1 |
| 2 | math | 42 | 42 | pass | 81.5 | 16.3 |
| 3 | math | 63 | 63 | pass | 83.0 | 17.1 |
| 4 | math | 9 | 7 | **fail** | 68.8 | 16.1 |
| 5 | math | 12 | 42 | **fail** | 83.8 | 16.3 |
| 6 | math | 1024 | 2 to the power of 10 is 1024. | pass | 86.3 | 20.6 |
| 7 | math | 50 | 50 | pass | 74.1 | 20.1 |
| 8 | math | 39 | 49 | **fail** | 67.8 | 17.5 |
| 9 | math | 125 | 125 | pass | 86.1 | 18.9 |
| 10 | math | 27 | 3 cubed is 27. | pass | 65.0 | 17.8 |
| 11 | capital | Paris | Paris | pass | 75.8 | 15.5 |
| 12 | capital | Tokyo | Tokyo | pass | 75.6 | 14.7 |
| 13 | capital | Rome | Rome | pass | 76.9 | 15.7 |
| 14 | capital | Ottawa | Toronto | **fail** | 76.4 | 13.9 |
| 15 | capital | Canberra | The capital of Australia is Canberra. | pass | 76.5 | 13.9 |
| 16 | capital | Berlin | Berlin | pass | 75.7 | 16.5 |
| 17 | capital | Cairo | Cairo | pass | 65.0 | 13.2 |
| 18 | capital | Madrid | The capital of Spain is Madrid. | pass | 77.2 | 16.7 |
| 19 | capital | Nairobi | The capital of Kenya is Nairobi. | pass | 76.6 | 17.8 |
| 20 | capital | Brasilia | The capital of Brazil is Brasília. | pass | 77.4 | 14.8 |
| 21 | chem-symbol | Au | The chemical symbol for gold is Au. | pass | 78.4 | 15.6 |
| 22 | chem-symbol | Ag | The chemical symbol for silver is Ag. | pass | 78.3 | 15.7 |
| 23 | chem-symbol | Fe | The chemical symbol for iron is Fe. | pass | 79.3 | 14.7 |
| 24 | chem-symbol | Na | Na | pass | 66.7 | 17.8 |
| 25 | chem-symbol | K | The chemical symbol for potassium is K. | pass | 78.4 | 14.7 |
| 26 | chem-symbol | O | The chemical symbol for oxygen is O. | pass | 78.7 | 19.0 |
| 27 | chem-symbol | Pb | The chemical symbol for lead is Pb. | pass | 78.3 | 15.7 |
| 28 | chem-symbol | He | The chemical symbol for helium is He. | pass | 78.8 | 13.9 |
| 29 | date | 1969 | The first time humans landed on the Moon was in 1969. | pass | 71.8 | 19.0 |
| 30 | date | 1945 | World War II ended in 1945. | pass | 79.8 | 17.7 |
| 31 | date | 1776 | 1789 | **fail** | 84.9 | 18.9 |
| 32 | date | 1989 | The Berlin Wall fell in 1989. | pass | 79.1 | 17.4 |
| 33 | date | 1492 | 1492 | pass | 82.0 | 18.7 |
| 34 | date | 1912 | 1912 | pass | 66.9 | 16.3 |
| 35 | date | 1914 | World War I began in 1914. | pass | 79.3 | 18.9 |
| 36 | units | 100 | 100 | pass | 79.0 | 16.5 |
| 37 | units | 1000 | 44.39 grams | **fail** | 66.2 | 12.6 |
| 38 | units | 3600 | 60 | **fail** | 77.4 | 17.5 |
| 39 | units | 12 | 60 | **fail** | 66.6 | 17.6 |
| 40 | units | 3 | There are 3 feet in one yard. | pass | 77.5 | 14.5 |
| 41 | units | 366 | There are 365 days in a leap year. | **fail** | 68.6 | 16.5 |
| 42 | units | 1000 | 1 | **fail** | 83.0 | 13.7 |
| 43 | science | 100 | The boiling point of water at sea level is approximately 100 degrees Celsius | pass | 78.2 | 19.8 |
| 44 | science | Jupiter | The largest planet in our solar system is Earth. | **fail** | 68.1 | 16.0 |
| 45 | science | Mercury | Mars. | **fail** | 65.4 | 12.3 |
| 46 | science | H2O | The chemical formula for water is H₂O. | pass | 66.1 | 17.7 |
| 47 | science | 7 | There are 7 continents on Earth. | pass | 66.4 | 15.6 |
| 48 | science | 6 | The number of legs an insect has varies by species, but most insects have two | **fail** | 66.1 | 18.9 |
| 49 | science | diamond | The hardest natural substance is a mineral or mineral. | **fail** | 76.3 | 14.7 |
| 50 | general | Shakespeare | The play "Rome and Juliet" was written by William Shakespeare. | pass | 77.8 | 17.6 |

By category: math 7/10, capital 9/10, chem-symbol 8/8, date 6/7, units 2/7, science 3/7, general 1/1.

No question stopped on its own inside 16 tokens; as in the earlier record, the pack keeps going after the answer line, which is why grading uses the first line only.

## Essays

| # | Topic | Facts | Missing | Forbidden | Grade | Prefill tok/s | Decode tok/s |
|---|---|---|---|---|---|---|---|
| 1 | photosynthesis | 4/7 | uses water; releases oxygen; happens in chloroplasts | none | **fail** | 34.9 | 16.2 |
| 2 | water cycle | 3/6 | condensation; collection, runoff or infiltration; clouds form | none | **fail** | 77.3 | 16.3 |
| 3 | seasons | 4/5 | hemispheres have opposite seasons | none | pass | 56.2 | 16.2 |
| 4 | human heart | 3/7 | atria; ventricles; pumps blood to the lungs for oxygen; arteries or veins or aorta | none | **fail** | 54.7 | 16.2 |
| 5 | world war ii | 6/7 | Pearl Harbor attack | none | pass | 59.8 | 15.4 |
| 6 | declaration of independence | 4/6 | Thomas Jefferson drafted it; Philadelphia | none | **fail** | 59.6 | 16.2 |
| 7 | dna | 1/8 | double helix; nucleotides; adenine; thymine; guanine; cytosine; A pairs with T and G with C | none | **fail** | 68.7 | 16.1 |
| 8 | solar system | 5/6 | gravity holds it together | none | pass | 75.8 | 16.1 |
| 9 | vaccines | 5/5 | none | none | pass | 76.2 | 16.1 |
| 10 | newton's laws | 1/5 | first law: inertia; second law: F = ma; third law: equal and opposite reaction; acceleration | none | **fail** | 56.6 | 16.1 |

Every essay ran to the 320-token cap, and the predictor sent every essay step through EXPAND (the fp32-book FULL kernels), so the essay decode rate is the FULL fast path. Read by hand, the essays start on topic and in clear prose, then degrade: several fall into repeated blank lines or a repeated phrase, one prints a stray `</think>` and restarts, and the DNA essay names the sugar-phosphate backbone but none of the four bases. The rubric only counts facts, so a fluent essay that leaves the key facts out scores low; no forbidden claim (seasons from distance, A pairs with C, nine planets and so on) appeared.

## Profile

| | Questions | Essays |
|---|---|---|
| Prefill tokens/s, per item mean (min to max) | 74.8 (44.5 to 86.3) | 62.0 (34.9 to 77.3) |
| Decode tokens/s, per item mean (min to max) | 16.5 (12.3 to 20.6) | 16.1 (15.4 to 16.3) |
| Decode tokens/s, pooled | 16.22 | 16.10 |
| Prompt tokens, generated tokens | 1438, 800 | 344, 3200 |
| DTC steps, EXPAND steps | 673, 177 | 0, 3216 |
| Wall time (load and calibration) | 92 s (22 s) | 228 s (23 s) |
| Engine RAM peak | 306.5 MB | 306.6 MB |
| Whole-card VRAM peak, mean GPU load | 3841 MiB, 75% | 3841 MiB, 88% |

The first question's prefill (about 44 tokens/s) and the first essay's (about 35) include warm-up; the rest sit near 65 to 86 tokens/s for these 20 to 40 token prompts. Decode matches the published 15 to 22 tokens/s range.

## How to rerun

    python littlenemo/tests/quiz/run_round.py \
        --engine path/to/nemo-engine-cuda.exe \
        --checkpoint /path/to/nemotron-nano-9b-v2 \
        --cuda-pack /path/to/the.nq2 --out results.json

`python littlenemo/tests/quiz/test_round_grade.py` checks the question set and rubric without a model: every gold answer and alias passes, every listed wrong answer fails, no gold answer appears in its own prompt (so an echo cannot pass), element symbols are matched case-sensitively, and a sample essay with a forbidden claim fails. The full essay texts and per-item numbers from this run stay in the private run folder with the other run data.

## SuperNemo, the same graders on a smaller set

JD asked for 5 questions and 2 essays on SuperNemo, since it runs at well under one token per second. The items come from the same files and are graded the same way: questions 5, 15, 28, 31 and 44 (one each from math, capitals, element symbols, dates and science, four of them ones the Little Nemo pack got wrong or that are easy to confuse), and essays 3 (seasons) and 7 (DNA).

| Part | Score | Notes |
|---|---|---|
| 5 short questions | **5/5** | every answer was the bare gold answer and the turn ended on its own |
| 2 essays | **2/2 pass**, 11 of 13 required facts | no forbidden claim; both essays ended on their own (119 and 154 tokens) |

| # | Gold | SuperNemo said | Little Nemo pack said |
|---|---|---|---|
| 5 | 12 | 12 | 42 |
| 15 | Canberra | Canberra | Canberra (in a sentence) |
| 28 | He | He | He (in a sentence) |
| 31 | 1776 | 1776 | 1789 |
| 44 | Jupiter | Jupiter | Earth |

| # | Topic | Facts | Missing | Read by hand |
|---|---|---|---|---|
| 3 | seasons | 4/5 | hemispheres | correct and on point (tilt of about 23.5 degrees, not distance, orbit, sunlight); the word "hemisphere" came out misspelled as "Hemisium" twice, so the hemisphere fact did not match; "the tilt changes" is loose wording for the tilt's direction relative to the Sun |
| 7 | DNA | 7/8 | nucleotides | double helix, sugar-phosphate backbone, all four bases, A with T and C with G, Watson and Crick 1953; "deoxyribonucleic" came out as "deoxyriucleic" |

The two misspellings are the only visible damage; they look like single dropped or swapped tokens, which fits the 2.2-bit pack noise rather than a knowledge gap.

### Conditions

- Engine: `sn-engine` built from the scratch branch with the loader fix (`supernemo: accept raw k and v projections`), into a private build folder; the binaries the post-pack script uses were not touched.
- Pack: the new SuperVL pack (124.45B weights, per-tensor books, RMS frame, PQ d4; tensors of 1,048,576 weights or fewer kept raw).
- CUDA lane, `--gpu-mb 3000` (273 tensors, 2830 MB resident: lm_head and the dense mixer, latent and shared projections), routed experts on the CPU, `--threads 12`, `--below-normal`, DTC on auto, RAM expert cache 4096 MB, no VRAM expert cache, speculative decoding off. Greedy, chat on, thinking off, `RESET` between items.
- One long-lived engine per part (questions at `--max-new 16`, essays at `--max-new 320`), so the expert cache stayed warm across the items in each part.
- Started 1:21 PM ET. A LithoKraft window was open for most of the run (it closed around 1:26 PM and a new one was open again by 1:35 PM). The GPU-quantizer runner stayed gated on the game the whole time; a gate check before and after every item found no GPU or heavy step running, so there were no pauses and no overlap.
- `sn-engine` answers `RESET` with `DONE` and `nemo-engine` does not; the runner's `--style supernemo` reads that acknowledgement. A first attempt without it read each acknowledgement as the next answer and was discarded.

### Profile

| | Questions | Essays |
|---|---|---|
| Prefill tokens/s, per item | 0.88 to 0.96 (about 35 s for 29 to 34 tokens) | 0.97 to 0.99 (about 37 s for 35 to 37 tokens) |
| Decode tokens/s, engine `gen_tps` per item | 0.26 to 0.33 | 0.354 and 0.355 |
| Decode tokens/s, pooled | 0.21 (only 11 tokens in all, so the first decode step dominates) | **0.351** over 273 tokens |
| Wall time (engine load) | 244 s (5 s) | 861 s (5 s) |
| Engine RAM peak | 708 MB | 710 MB |
| Whole-card VRAM peak, mean GPU load | 3685 MiB, 9% | 3822 MiB, 24% |

The whole set took 1106 s (about 18.5 minutes), well under the 45-minute estimate, because the answers stopped on their own instead of running to the token caps. The essays' 0.35 tokens/s is the steady decode rate to plan with; a 320-token answer would take about 15 minutes. The GPU is mostly idle: the routed experts run on the CPU and dominate the time.

### Speed probe before the run

One-shot runs on a 35-token prompt (list the eight planets), 17 tokens out, each a fresh process. All four answered correctly.

| Lane | Prefill tok/s | Decode tok/s | Notes |
|---|---|---|---|
| CUDA, `--gpu-mb 3000` | 0.78 | 0.335 | card peak 3677 MiB |
| CUDA, speculative, draft 1 | 0.83 | 0.31 | MTP acceptance 89 percent |
| CUDA, speculative, draft 3 | 0.82 | 0.28 | MTP acceptance 80 percent |
| CPU only | 0.83 | 0.14 | |

Speculative decoding is token-identical but slower here: verifying several tokens reads the union of their experts, and on this machine expert reads are the cost.

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.

