# SuperNemo GPU expert stream, round 7 receipts (2026-11-28)

## Round 7: early fetch that keeps its guesses (2026-11-28)

JD's rule (1:25 AM): MTP and early fetch (EEF) stay on by default. Round 6 showed EEF made the default slower than EEF off, so this round
profiled why and changed how early fetch works. Everything below is **real-world** timing (fid-quant and other workers on the CPU,
Little Nemo and the retention battery taking turns on the card), 8 threads, below-normal, MTP on, `--gpu-mb 3000`, the same prompt
as every run today (17 tokens came back before the model's end-of-text, 5 MTP cycles). Single runs of this length move by up to
60% from load alone (the same EEF-off run gave 0.150 to 0.251 tok/s; prefill, which EEF does not touch, ranged from 46 s to 95 s),
so the table below also carries counters that do not depend on load.

### Why early fetch was slower (profile)

- Early fetch read its guesses from the pack **on the main thread**, so no time overlapped: the disk is the bottleneck (the CPU
  fill of the pinned staging is about 75% of stream time) and every wrong guess (about a third) was an extra read.
- The guessed experts were **not admitted to the RAM cache**, so with EEF on the cache hit rate fell (8,887 to 5,000 of 15,840 in
  round 6) and later tokens read the same experts from disk again.
- The next layer's router, run early on the main thread, costs 6 to 12 s per run on a CPU-oversubscribed machine (the "early fetch"
  stage time), about as much as the reads it saves.

### What changed (`SN_XS_PFMODE`, results unchanged)

| Mode | What early fetch does |
|---|---|
| 1 | the round 6 behaviour: guesses are read from the pack on the main thread and not kept |
| 2 | same reads, but the guesses are admitted to the RAM cache (one shared `cache_admit` rule for demand loads and guesses) |
| 3 | only guesses already in the pinned RAM cache are sent early (one DMA, no CPU copy); the pack is never read for a guess |
| 4 (default) | mode 3, plus the other guesses are read from the pack on the pool's prep thread into pinned blocks while the main thread computes, and become RAM cache entries at the next `cache_plan` (so cache state never depends on disk speed) |

No mutexes: the prep thread is the pool's existing handshake job, and the main thread waits for it only at the next `cache_plan`.

### Table (MTP on, EEF mode as shown; decode tok/s over 17 tokens)

| Mode | runs | decode tok/s (median / mean / range) | RAM-cache hits | pack reads, demand + early | early sends (wrong) |
|---|---|---|---|---|---|
| EEF off | 6 | 0.189 / 0.191 / 0.150-0.251 | 4812 | 17975 | 0 (0) |
| mode 4 (default) | 6 | 0.183 / 0.206 / 0.154-0.352 | 5600 | 18249 | 2043 (258) |
| mode 3 | 4 | 0.171 / 0.173 / 0.167-0.183 | 4812 | 17975 | 2021 (239) |
| mode 2 | 2 | 0.161 / 0.161 / 0.156-0.166 | 5544 | 19209 | 5148 (1372) |
| mode 1 (old) | 2 | 0.165 / 0.165 / 0.125-0.205 | 3817 | 20145 | 5148 (1372) |

The counters repeat exactly from run to run (same cache hits, same reads), which is also what a deterministic cache gives; only the
clock moves.

All runs (every one, nothing dropped):

| run | mode | draft | decode tok/s | wall s | cache hits | pack reads | early DMAs (wrong) | background reads kept | load |
|---|---|---|---|---|---|---|---|---|---|
| x7-a1-admit | 2 | 3 | 0.1656 | 203 | 5544 | 19209 | 1967 (254) | 0 | fid-quant |
| x7-b1-legacy | 1 | 3 | 0.2049 | 166 | 3817 | 20145 | 1316 (206) | 0 | fid-quant |
| x7-c1-cachedonly | 3 | 3 | 0.1697 | 235 | 4812 | 17975 | 2021 (239) | 0 | fid-quant |
| x7-d1-eefoff | off | 3 | 0.2016 | 192 | 4812 | 17975 | 0 (0) | 0 | fid-quant |
| x7-d2-eefoff | off | 3 | 0.1496 | 234 | 4812 | 17975 | 0 (0) | 0 | fid-quant |
| x7-c2-cachedonly | 3 | 3 | 0.1835 | 173 | 4812 | 17975 | 2021 (239) | 0 | fid-quant |
| x7-b2-legacy | 1 | 3 | 0.1249 | 228 | 3817 | 20145 | 1316 (206) | 0 | fid-quant |
| x7-a2-admit | 2 | 3 | 0.1563 | 209 | 5544 | 19209 | 1967 (254) | 0 | fid-quant |
| x8-e1-bg | 4 | 3 | 0.1846 | 213 | 5600 | 18249 | 2043 (258) | 1069 | fid-quant |
| x8-d1-eefoff | off | 3 | 0.1993 | 192 | 4812 | 17975 | 0 (0) | 0 | fid-quant |
| x8-c1-cached | 3 | 3 | 0.1731 | 227 | 4812 | 17975 | 2021 (239) | 0 | fid-quant |
| x8-e2-bg | 4 | 3 | 0.1811 | 189 | 5600 | 18249 | 2043 (258) | 1069 | fid-quant |
| x8-d2-eefoff | off | 3 | 0.2506 | 158 | 4812 | 17975 | 0 (0) | 0 | fid-quant |
| x8-c2-cached | 3 | 3 | 0.1674 | 190 | 4812 | 17975 | 2021 (239) | 0 | fid-quant |
| x9-default-draft3 | 4 | 3 | 0.1725 | 188 | 5600 | 18249 | 2043 (258) | 1069 | fid-quant |
| x9-draft1 | 4 | 1 | 0.1721 | 188 | 8032 | 17933 | 4194 (523) | 1446 | fid-quant |
| x9-draft2 | 4 | 2 | 0.1586 | 217 | 6022 | 17767 | 2416 (309) | 1233 | no fid-quant |
| x9-default-draft3-b | 4 | 3 | 0.1915 | 209 | 5600 | 18249 | 2043 (258) | 1069 | no fid-quant |
| x9e-off-1 | off | 3 | 0.1787 | 216 | 4812 | 17975 | 0 (0) | 0 | no fid-quant |
| x9e-default-1 | 4 | 3 | 0.1536 | 236 | 5600 | 18249 | 2043 (258) | 1069 | no fid-quant |
| x9e-default-2 | 4 | 3 | 0.3517 | 114 | 5600 | 18249 | 2043 (258) | 1069 | fid-quant |
| x9e-off-2 | off | 3 | 0.1656 | 228 | 4812 | 17975 | 0 (0) | 0 | no fid-quant |

### What it shows

- **Mode 4 keeps the cache learning**: 5,600 RAM-cache hits against 4,812 with EEF off (+16%) and 3,817 in the round 6 mode 1, for
  about 1.5% more pack reads than EEF off (18,249 against 17,975), because the 1,069 guesses read in the background replace later
  demand reads. The main thread waited 0.75 to 2.9 s in total for those background reads.
- **Modes 1 and 2 read more** (20,145 and 19,209) and do not overlap those reads, so they are the slow ones. Mode 3 reads exactly what EEF off reads and
  adds 2,021 early sends that were right 88% of the time.
- **Speed: mode 4 and EEF off are tied within the noise.** Median 0.183 against 0.189 tok/s, mean 0.206 against 0.191
  (the mean for mode 4 includes one unusually quiet-machine run at 0.352). Mode 4 is clearly better than modes 1 to 3, but on this
  disk-bound box it does **not** beat EEF off, so the "at least as fast as off" goal is met only within the noise, not shown by a margin.
  JD decides whether EEF stays on at this cost; the numbers are here for that call.
- **MTP draft length** (mode 4): draft 1 0.1721 tok/s (89% accepted, 9 cycles), draft 2 0.1586 (92%, 6 cycles),
  draft 3 0.1725 and 0.1915 (80%, 5 cycles). All inside the noise, so the default stays at 3.

### Accuracy

- Hash-identical, so accuracy is unchanged: with mode 4 the whole-model logits hash on the real pack is `b0c32d2989f86f69` for the stream
  alone (300 slots) and for the stream with every speed-up off (1,684 slots), both equal to the CPU path; the dense 3000 MB lane
  with and without the stream is `5dacc350426a9506` (the dense lane's own float kernels differ from the CPU lanes, known and
  unchanged). 6 checks, 0 failed, 8 threads.
- Tiny packs, modes 1, 2, 3 and 4 (and the final default build): `b194b652178e8593` and `740d6b3b8fd53103` (116 checks), tiny MTP
  `274289f8c1635e2f` and `3ea686ea2e00e053` (166 checks), 3-slot and 5-slot runs that force eviction, early fetch and background
  reads; threads 1, 4 and 8 all agree. CPU tests: model 43/0, spec 234/0, cache 14/0.
- **Quiz subset (SuperNemo, final build, defaults: MTP on, early fetch mode 4): 5 of 5 questions pass** (Q5 "12", Q15 "Canberra", Q28 "He", Q31 "1776", Q44 "Jupiter"; the same five questions as the 13:21 and 16:23 runs, which also scored 5 of 5). Pooled prefill 0.547 tok/s and decode 0.081 tok/s, card peak 3694 MiB, engine RAM peak 5087 MB, wall 575 s, real-world. The essays (E3, E7) were not rerun tonight: they take about 50 minutes and the hash identity above already shows the output cannot differ from the earlier passes.
