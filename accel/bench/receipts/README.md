# Bench receipts

The raw logs behind the speed and agreement numbers in
[accel/README.md](../../README.md) and
[docs/ACCEL-CUDA.md](../../../docs/ACCEL-CUDA.md), copied from the bench
machine (RTX 3050 Ti Laptop 4 GB, i5-12450H, 16 GB) on 2026-11-27. They are
text only: no weights, packs, logit dumps or run-ledger data.

Scrubbed before commit: machine paths are replaced by placeholders
(`<results>` for the results folder that holds the packs and the Sigma-3B
model, `<scratch>`, `<user-home>`, `<models>`), and the user and host names by
`<user>` and `<host>`. Nothing else was edited. Paths inside a receipt that
read `accel\logs\...` or `accel\lane4\...` are where the file sat when it was
written; the files now live here.

| Folder | Session (ET) | Backs |
|---|---|---|
| `lane-bench-2026-11-26/` | 2026-11-26 10:44 to 10:48 | accel/README.md "Benchmarks": the gated 9-run bench, `RECEIPTS.txt` |
| `bench-lane1/`, `bench-lane1-warm/` | 2026-11-26 11:24 to 11:28 | accel/README.md Lane 1 table (eager, active, lazy, direct) |
| `bench-lane2/` | 2026-11-26 11:18 to 11:19 | accel/README.md Lane 2 (create-time snapshot, PCIe probe) |
| `lane4/` | 2026-11-26 11:33 to 12:15 | accel/README.md Lane 4 (packs against the raw bf16 model): `RECEIPTS.txt`, `progress.txt`, `cmp-*-vs-raw.txt`, run outputs |
| `cuda-throughput-2026-11-26/` | 2026-11-26 evening and 2026-11-27 early morning | docs/ACCEL-CUDA.md: "Speed after the throughput pass" and the CUDA quiz speeds (see below) |

## Which file holds which ACCEL-CUDA number

Each `*.txt` is the engine's protocol output (one `STATS` line per prompt),
`*.in.txt` is the exact input, and `*.err.txt` is the engine's stderr.

| Number in ACCEL-CUDA.md | Receipt |
|---|---|
| Baseline DTC prefill 11.2 / 18.3 / 18.1 tokens/s, decode 15.9 to 15.0, time to first token 712 ms / 11.6 s | `base-never.txt` (`prompt_tps`, `gen_tps`, `ttft_ms` for 8, 52 and 211 prompt tokens) |
| Baseline FULL prefill 8.4 tokens/s, decode 7.7 | `base-always.txt` |
| After the pass: DTC prefill 21.3 / 76.1 / 80.7 tokens/s, decode 21.7 to 20.2, time to first token 376 ms / 2.62 s | `c5-never.txt` (`f1-never.txt` repeats it on the fast FULL build: 80.5 at 211 tokens) |
| FULL prefill 72.1 tokens/s on the fast path, decode 16.0 to 16.6 | `f1-always.txt` |
| FULL prefill 43.4 tokens/s on `exact`, decode 8.1 to 8.4 | `c5-always.txt` |
| Quiz, DTC only: about 20 to 22 tokens/s | `qfinal-never.txt` (21.35 to 21.76 over the ten questions) |
| Quiz, auto: 9 to 17 tokens/s per question | `qfinal-auto.txt` (9.12 to 16.84) |
| Quiz, full expansion fast path: about 16 to 17 tokens/s | `qfinal-always.txt` (16.87 to 17.18) |

## Numbers with no receipt here

Said plainly so nobody reads them as receipted:

- the raw bf16 CPU quiz (about 0.07 tokens/s, 10/10); the roadmap already
  lists committing that record;
- the speeds from the day the lane first landed (15.5 to 15.8 DTC, 5 to 11
  auto, 7.9 FULL exact on the quiz);
- the calibration probe line (DTC 137.4 Gw/s, FULL 64.5 Gw/s, device read
  176.5 GB/s, estimate about 15.3 tokens/s; CPU 19.4 GB/s);
- the quiz grading itself (7 right, 2 hedged, 1 wrong) is a reading of the
  answers; the answers for the post-pass runs are in the `qfinal-*.txt`
  files, the grading is not recorded separately;
- the Verification table (per-layer residuals, batched equals stepped,
  fast equals exact token counts);
- the 7 real Sigma layer-0 projection logs (`real-*-3.log`) named in
  accel/README.md Lane 1. They were not in the staged receipt set, so they
  are not in this folder;
- the per-run TSV `*.log` files named in the original bench output.
  `*.log` is gitignored, so those logs stay private. Each folder's
  `RECEIPTS.txt` is the committed summary.
