# Quiz receipts

Every answer behind the scores in
[docs/LITTLE-NEMO-ROUND-2026-11-27.md](../../../../docs/LITTLE-NEMO-ROUND-2026-11-27.md),
copied from the bench machine (RTX 3050 Ti Laptop 4 GB, i5-12450H, 16 GB) on
2026-11-27. They are text only: no weights, packs, logit dumps, engine stderr
or run-ledger data.

| Folder | Session (ET) | Score | Files |
|---|---|---|---|
| `2026-11-27/little-nemo/` | 12:52 to 12:58 PM | 36/50 questions, 4/10 essays (36 of 62 facts) | `RECEIPTS.md`, `receipts.json`, `reset-check.json` |
| `2026-11-27/supernemo/` | 1:21 to 1:40 PM | 5/5 questions, 2/2 essays (11 of 13 facts) | `RECEIPTS.md`, `receipts.json`, `speed-probe.json` (1:02 to 1:14 PM) |

## Format

- `RECEIPTS.md` is the readable copy: run conditions, the exact engine
  command with placeholder paths, the engine's READY line, then every item
  with its prompt, gold answer, graded first line or fact list, timing and
  the model's full answer (essays in full) in a text block.
- `receipts.json` holds the same data for tools. Per item: `prompt`, `gold`
  with `accepted_aliases` and `listed_wrong_answers` (questions), or the
  `found` facts with the text that matched, the `missed` facts and the
  forbidden claims checked and found (essays); `answer_full`; `grade`; and
  `profile`.
- `profile` fields come from the engine's own `PREFILL`, `STATS` and `MEM`
  lines: `prompt_tokens`, `prefill_ms`, `prefill_tps`, `gen_tokens`,
  `gen_ms` (the whole turn, prefill included), `decode_tps` (the engine's
  `gen_tps`), `ram_peak_mb` (engine peak working set), plus the runner's
  `wall_s` and, from nvidia-smi sampled once a second, the whole-card
  `vram_used_peak_mib` and mean `gpu_util_mean`. Whole-card VRAM includes
  about 0.6 GiB that Windows and other programs hold.
- Each part also has pooled numbers: decode tokens over decode time
  (`gen_ms` minus `prefill_ms`), and prompt tokens over prefill time. With
  only a few tokens per answer, the pooled decode figure is dominated by the
  first decode step; the essays give the steady rate.
- `other_job_gate` (SuperNemo only) records the check of the GPU-quantizer
  runner before and after each item: `clear` means it was not running a GPU
  or heavy step, and `overlap` is true if one started during the item.
- `reset-check.json` reruns three wrong Little Nemo answers in a fresh
  engine in another order; the answers match, so they are the model's, not
  state left from the question before.
- `speed-probe.json` keeps the engine's `STATS`, `SPEC` and `MEM` lines and
  replies from the one-shot SuperNemo speed probes run before the quiz.

## Grading

The graders and item files are next to this folder:
[`grade.py`](../grade.py), [`grade_essay.py`](../grade_essay.py),
[`questions50.json`](../questions50.json), [`essays.json`](../essays.json).

- **Questions** are graded on the first non-empty answer line. A pass needs
  the gold answer or an accepted alias as a whole word, with no listed wrong
  answer before it. Element symbols are matched case-sensitively so `He`
  cannot be the pronoun. No gold answer appears in its own prompt, so an echo
  of the question cannot pass.
- **Essays** are graded on facts, not style. A required fact is found when any
  of its case-insensitive patterns matches the essay; a forbidden claim (a
  common misconception such as seasons from distance, or adenine pairing with
  cytosine) is flagged the same way. An essay passes with 70 percent of its
  required facts and no forbidden claim. A misspelled key word does not
  match, which is how SuperNemo's seasons essay lost its hemisphere fact.
- `python littlenemo/tests/quiz/test_round_grade.py` checks the item files
  and graders without a model. Rerunning a receipt's prompts through
  `run_round.py` with the same engine and pack should give the same answers;
  decoding is greedy.

## Scrubbing

Machine paths are replaced by placeholders (`<models>` for the model and
pack folder, `<scratch>` for the working tree, `<user-home>`), and the user
and host names by `<user>` and `<host>`. The engine's stderr, the runner's
logs and the private run folders are not copied. Answers, prompts and
numbers are not edited.

## Not here

- The first SuperNemo attempt, which misread `sn-engine`'s `DONE`
  acknowledgement of `RESET` and was discarded (see the round doc).
- The 2026-11-27 post-pack speed runs, which logged no output.
