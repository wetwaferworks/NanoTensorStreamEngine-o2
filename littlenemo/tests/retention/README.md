# NanoTensor retention battery

This battery measures what Little Nemo (Nemotron Nano 9B v2) and SuperNemo (Nemotron 3.5 Super VL) still know and can do after NanoTensor compression. It runs the same items on each arm: raw bf16, packed, expanded, and the higher-precision output head. The results write-up is [docs/RETENTION-BATTERY-2026-11-27.md](../../../docs/RETENTION-BATTERY-2026-11-27.md). The scrubbed receipts are in [receipts/](receipts/).

## Files

| File | What it is |
|---|---|
| `build_battery.py` | Builds `battery.json` and `probe.tsv`. The build is deterministic, and the long-context haystacks use a fixed seed. |
| `battery.json` | All 153 items with gold answers, accepted spellings and the checks each item gets graded by. |
| `probe.tsv` | The 143 teacher-forced items for `ln_retain`, as id, set, category, prompt and answer. |
| `ln_retain.c` | The teacher-forced scorer, in pure C. It's a fork of `littlenemo/expand/ln_sens.c`; see its header. |
| `run_battery.py` | Runs generation on the Little Nemo CUDA engine, with `NEMO_LOGIT_TOPK` telemetry and speed for every item. |
| `grade_retention.py` | Graders for the battery: short answers, abstention, and instruction rules checked both strictly and on the first line only. |
| `analyze.py` | Turns the teacher-forced and generation outputs into the summary tables. |
| `sn_subset.json` | The ten items SuperNemo runs, through `littlenemo/tests/quiz/run_round.py --style supernemo`. |
| `test_retention_grade.py` | Checks the battery and graders without a model. |

## Sets

| Set | Items | Lane |
|---|---|---|
| `quiz` | 50 | The 2026-11-27 round, with the same prompts and gold answers. This is the matched raw-vs-packed set. |
| `cand` | 12 | The packed lane's own wrong answers on that round, scored next to the gold answers. This is the top-k lane. |
| `para` | 39 | 13 facts asked three ways each. This is the paraphrase stability lane. |
| `multi` | 12 | Multi-step arithmetic and word problems. |
| `pass` | 10 | Questions about made-up passages. The answer has to be reasoned from the text because it can't have been memorized. |
| `calib` | 16 | 8 made-up entities that should get "I don't know" and 8 real questions under the same instruction. |
| `instr` | 10 | Instruction following, checked by rule. Generation only. |
| `long` | 4 | Needle-in-a-haystack recall over about 600 to 1,200 tokens. |

## ln_retain output

The `--detail` file has one line per configuration and item, with these tab-separated columns:

- configuration and item id
- set and category
- gold NLL in nats, summed over the answer tokens
- whether greedy picks the first gold token
- whether greedy picks every gold token
- greedy's first token
- answer length
- the first gold token's rank (0 means top) and its log-probability
- the end-of-turn log-probability right after the full gold answer, summed over the engine's stop set (12, 2 and 11)
- the best rank of a stop token at that point, and the argmax token there
- the top five first tokens, each as `id:logprob:text`

Plan specs are the same as ln_sens (`base`, `ref`, `L<n>[.<leaf>]`, `R<n>[.<leaf>]`), plus two more:

- `head8` scores through the CUDA lane's int8 per-row `lm_head`.
- `class:<leaf>` puts one tensor class back to the original weights in every layer. The leaf can be `in_proj`, `out_proj`, `up_proj`, `down_proj` or `attn`.

## Engine telemetry

When `NEMO_LOGIT_TOPK=K` is set, `littlenemo/src/engine_gui.c` (and so the generated CUDA engine) prints one line per generated token:

    LOGIT step=N tok=ID lp=... margin=... stop_lp=... stop_rank=R top=id:lp,...

It only reads the logits. With the variable unset, the engine's output is byte-for-byte unchanged.

No model, pack or machine path is stored in this folder.

Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX.
