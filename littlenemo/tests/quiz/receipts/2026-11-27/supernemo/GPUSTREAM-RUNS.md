# SuperNemo GPU expert stream: every run, 2026-11-27

Scrubbed receipts (no machine paths). Co-implementation by Cortana, built with Grok Bot by SpaceX.


Same prompt for runs 1 to 14 ("List the eight planets of the solar system in order from the Sun, separated by commas.",
32 new tokens allowed, 12 threads, below-normal priority); every one of them answered "Mercury, Venus, Earth, Mars,
Jupiter, Saturn, Uranus, Neptune" (the 4-token warm-up stopped after "Venus"). Prefill and decode are pooled tok/s from
the engine's own STATS line; wall is the whole process including load; card peak is the whole card (the display and other
programs included); RAM is the engine's peak working set. Another job was using about four CPU cores during all of these,
so every timing here is real-world. "Budget issue" marks runs from before the RAM cache fix, when the 4096 MB cache was
silently off on Windows.

| # | Run | Path | RAM cache | Slots (policy) | Prefill tok/s | Decode tok/s | Wall | Card peak | Engine RAM peak | Conditions |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | warm-up (4 tokens) | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.333 | 0.1459 | 148 s | 3446 MiB | 684 MB | real-world (another job on the CPU) |
| 2 | before-gpu3000 | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.301 | 0.1537 | 249 s | 3446 MiB | 686 MB | real-world (another job on the CPU) |
| 3 | after-gpu3000-stream | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.813 | 0.2534 | 132 s | 3749 MiB | 609 MB | real-world (another job on the CPU) |
| 4 | after-stream-only | CUDA expert stream only, dense on CPU | off (budget issue) | 1684 (least recently used) | 0.454 | 0.0300 | 690 s | 3701 MiB | 650 MB | real-world (another job on the CPU) |
| 5 | before-gpu3000-again | CUDA dense 3000 MB on card, experts on CPU lanes | off (budget issue) | - | 0.265 | 0.0918 | 345 s | 3445 MiB | 686 MB | real-world (another job on the CPU) |
| 6 | after-gpu3000-stream-again | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 115 (least recently used) | 0.487 | 0.1042 | 271 s | 3708 MiB | 608 MB | real-world (another job on the CPU) |
| 7 | streamonly-300-lru | CUDA expert stream only, dense on CPU | off (budget issue) | 300 (least recently used) | 0.620 | 0.0543 | 389 s | 1239 MiB | 649 MB | real-world (another job on the CPU) |
| 8 | gpu3000-hot-prof | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (hot, prototype) | 0.820 | 0.2496 | 132 s | 3899 MiB | 609 MB | real-world (another job on the CPU) |
| 9 | gpu3000-hot-prof-spec3 | CUDA dense 3000 MB on card + expert stream, MTP spec 3, accept 80% | off (budget issue) | 119 (hot, prototype) | 0.695 | 0.3312 | 122 s | 3899 MiB | 1127 MB | real-world (another job on the CPU) |
| 10 | streamonly-300-hot | CUDA expert stream only, dense on CPU | off (budget issue) | 300 (hot, prototype) | 0.551 | 0.0601 | 366 s | 1271 MiB | 648 MB | real-world (another job on the CPU) |
| 11 | gpu3000-lru-prof | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 101 (least recently used) | 0.846 | 0.2841 | 124 s | 3829 MiB | 609 MB | real-world (another job on the CPU) |
| 12 | gpu3000-cachefix-prof | CUDA dense 3000 MB on card + expert stream | on (fixed) | 101 (hot, prototype) | 0.591 | 0.2736 | 150 s | 3837 MiB | 4441 MB | real-world (another job on the CPU) |
| 13 | gpu3000-cachefix-prof-spec3 | CUDA dense 3000 MB on card + expert stream, MTP spec 3, accept 80% | on (fixed) | 119 (hot, prototype) | 0.863 | 0.3139 | 116 s | 3863 MiB | 5231 MB | real-world (another job on the CPU) |
| 14 | gpu3000-cachefix-nostream | CUDA dense 3000 MB on card, experts on CPU lanes | on (fixed) | - | 0.382 | 0.1829 | 213 s | 3617 MiB | 4783 MB | real-world (another job on the CPU) |
| 15 | quiz, 5 questions (16 new tokens max) | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.365 | 0.078 | quiz run total 3070 s | 3707 MiB | 615 MB | real-world (another job on the CPU) |
| 16 | quiz, 2 essays (320 new tokens max) | CUDA dense 3000 MB on card + expert stream | off (budget issue) | 119 (least recently used) | 0.356 | 0.125 | (same run) | 3884 MiB | 616 MB | real-world (another job on the CPU) |

Real-pack test runs (not speed runs): the kernel checks took about 25 s each; the whole-model hash run took 729.7 s
(CPU, 12 threads), 620.4 s (stream only, 12 threads), 935.2 s (stream only, 4 threads), 1088.8 s (CPU, 4 threads),
249.8 s (dense 3000 MB) and 87.1 s (dense 3000 MB plus stream), with a CPU scan from another worker running at the time.
