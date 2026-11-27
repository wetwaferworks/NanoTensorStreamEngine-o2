# Little Nemo tokenizer-based evaluation

Sigma's Lane 4 score is top-1 agreement on one denoiser evaluation
(32 positions). Little Nemo is autoregressive, so its quality score
belongs on a stretch of text: tokenize a fixed passage, take every
next-token decision under teacher forcing, decode the predicted ids
back through the same tokenizer, and report whether that text holds
together.

This record is the harness and the fixed passage. It is not a
WikiText number (that measurement is still on the
[roadmap](ROADMAP.md) as a fidelity judgement) and it is not a live
9B result: a teacher-forced pass over the 140-token passage reads
the raw checkpoint once per token and should not be run as an
all-core marathon on a busy machine.

## What is committed

- The passage: [`littlenemo/tests/tokeneval/passage.txt`](../littlenemo/tests/tokeneval/passage.txt),
  original to this repository.
- Gold token ids for that passage, derived with the official
  Nemotron Nano 9B v2 `tokenizer.json`:
  [`passage.ids`](../littlenemo/tests/tokeneval/passage.ids) (140 ids).
  Encode-then-decode of the passage is identical to the source text.
- The scorer: [`score.c`](../littlenemo/tests/tokeneval/score.c)
  (top-1, top-5, first divergence, double log-softmax perplexity;
  ties go to the lowest index).
- The suite and driver: `littlenemo/bin/test_tokeneval`.

No checkpoint path and no `tokenizer.json` path is stored in the tree.

## Unit tests (no checkpoint)

From the `littlenemo/` directory, after `build.bat` or `build.sh`:

    bin/test_tokeneval tests/fixture_tokenizer.json

This checks the scorer against hand-chosen logits (including a
balanced two-class NLL of log 2), encode/decode on the fixture
tokenizer, that a synthetic teacher-forced run decodes to the same
text as the true suffix, and that the committed passage is present.
It does not need the real model or the real tokenizer.

## Tokenizer-only check against the gold ids

Needs a local official `tokenizer.json`, still no weights:

    bin/test_tokeneval --tokenizer /path/to/tokenizer.json \
        --text tests/tokeneval/passage.txt \
        --ids tests/tokeneval/passage.ids

Expect `tokens 140`, `encode/decode identity: yes`, and
`gold match: yes`. To regenerate the gold file (do not commit a
machine path with it):

    bin/test_tokeneval --tokenizer /path/to/tokenizer.json \
        --text tests/tokeneval/passage.txt \
        --write-ids tests/tokeneval/passage.ids

## Live teacher-forced eval (optional, heavy)

    bin/test_tokeneval --checkpoint /path/to/nemotron-nano-9b-v2 \
        --tokenizer /path/to/tokenizer.json \
        --text tests/tokeneval/passage.txt \
        --threads 1 --out tokeneval-report.txt

`--max-tokens N` shortens the passage for a smoke run. The default
lane count is 1. The report prints top-1, top-5, first divergence,
perplexity, and whether the decoded predicted text equals the
decoded true suffix.

`--checkpoint` is the only path that loads weights. Leave it off
unless the machine is free for a long CPU pass.
