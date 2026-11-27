# Little Nemo 10-question smoke quiz

The 2026-11-26 bench session scored the raw bf16 CPU engine **10 of 10** at
about **0.07 tokens/s**, and the compressed CUDA pack **7 of 10** on the
same questions. This file is the record that lets those figures be
reproduced from the tree. The numbers themselves remain measurements from
that session, documented in [ACCEL-CUDA.md](ACCEL-CUDA.md); they are not
a guarantee on another machine.

Exact prompts, expected answers and the published pack first lines live in
[`littlenemo/tests/quiz/questions.json`](../littlenemo/tests/quiz/questions.json).
The first-line grader is [`grade.py`](../littlenemo/tests/quiz/grade.py).
The runner is [`run_quiz.py`](../littlenemo/tests/quiz/run_quiz.py). No
checkpoint, pack, or machine path is stored in any of those files.

## Conditions (the published record)

- Greedy decoding (the engine default).
- Chat template on (`engine_gui --chat`, or the CUDA engine with `--chat`).
- Thinking off (do not pass `--think`).
- `--max-new 16`.
- A fresh conversation per question (`RESET` between prompts).
- Grade the first non-empty answer line after joining `TOK` pieces.

Headline score counts only clean **passes**. A **hedge** is the right word
first with extra hedging on that line (for example `True or False`). A
**fail** is a wrong or missing answer. The pack's 7 of 10 is 7 pass, 2
hedge (questions 1 and 3), 1 fail (question 10, "Earth" instead of
Jupiter). The CPU reference on the decoded pack gave the same wrong
planet; the loss is from compression, not from the GPU.

## How to rerun the raw CPU path

The checkpoint is not in this repository. Point the runner at a local
Nemotron Nano 9B v2 tree (or set `NEMO_CKPT` and, if needed,
`NEMO_TOKENIZER`):

    python littlenemo/tests/quiz/run_quiz.py \
        --engine littlenemo/bin/engine_gui.exe \
        --checkpoint /path/to/nemotron-nano-9b-v2 \
        --tokenizer /path/to/nemotron-nano-9b-v2/tokenizer.json

That command line is the published raw-CPU record. A full ten-question
pass at about 0.07 tokens/s takes on the order of half an hour to an
hour; the 2026-11-26 figure was about 0.07 tokens/s with clean stops and
10 of 10 first lines.

## How to rerun a compressed CUDA pack

Same runner, CUDA engine, plus the pack (and any `--expand` flags you
are measuring). The GPU must be free. Set `--cuda-pack` to a local `.nq2`
and do not commit that path.

    python littlenemo/tests/quiz/run_quiz.py \
        --engine accel/bin/nemo-engine-cuda.exe \
        --checkpoint /path/to/nemotron-nano-9b-v2 \
        --tokenizer /path/to/nemotron-nano-9b-v2/tokenizer.json \
        --cuda-pack /path/to/the.nq2

The published pack score is 7 of 10 under the default auto expand policy.

## Tests that do not need a checkpoint

    python littlenemo/tests/quiz/test_quiz_grade.py

This checks the ten exact prompts, the first-line rules, and that the
published pack first lines still score 7 of 10 (7 pass, 2 hedge, 1 fail)
while clean expected answers score 10 of 10.

## The larger verifiable round

A 50-question set with gold answers and aliases (`questions50.json`), a 10-essay fact rubric (`essays.json`, graded by `grade_essay.py`) and a profiling runner (`run_round.py`) live beside this quiz. The 2026-11-27 results on the CUDA pack are in [LITTLE-NEMO-ROUND-2026-11-27.md](LITTLE-NEMO-ROUND-2026-11-27.md). `test_round_grade.py` checks the set and the rubric without a model.
