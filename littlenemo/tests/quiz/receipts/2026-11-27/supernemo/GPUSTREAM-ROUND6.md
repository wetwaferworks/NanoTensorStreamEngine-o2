# SuperNemo GPU expert stream: round 6 runs, 2026-11-27

Scrubbed receipts (no machine paths). Co-implementation by Cortana, built with Grok Bot by SpaceX.

Same prompt as every run above ("List the eight planets ...", 32 new tokens allowed, 8 CPU threads, below-normal priority).
All of these are **real-world** timings: fid-quant and other workers (Little Nemo expansion and retention runs) were using the
CPU and the disk the whole time, so the same configuration can differ by a factor of two from one run to the next (compare rows 2
and 8, which are the same setup). Read the ratios, not the absolute numbers. Prefill and decode are tok/s from the engine's STATS
line; "card peak" is the whole card as nvidia-smi sees it; "MB sent" is everything copied to the card.
| # | Run | MTP | Early fetch | Pinned cache | Slots | Prefill tok/s | Decode tok/s | Wall | Card peak | Engine RAM | MB sent to card | Early fetch sent / right | RAM cache hits / misses |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | defaults: MTP + early fetch + pinned cache | on (80% accepted) | on | on | 119 | 0.307 | 0.0933 | 338 s | 3801 MiB | 3610 MB | 45541 | 5148 / 3776 (73%) | 3817 / 8937 |
| 2 | MTP off | off | on | on | 119 | 0.242 | 0.0611 | 491 s | 3730 MiB | 3653 MB | 57912 | 15444 / 10496 (68%) | 5000 / 10840 |
| 3 | MTP off, early fetch off | off | off | on | 119 | 0.303 | 0.1171 | 308 s | 3730 MiB | 3659 MB | 48538 | - | 8887 / 6953 |
| 4 | MTP off, pinned cache off | off | on | off | 119 | 0.310 | 0.0573 | 471 s | 3730 MiB | 3682 MB | 57912 | 15444 / 10496 (68%) | 5000 / 10840 |
| 5 | MTP off, early fetch off, pinned cache off | off | off | off | 119 | 0.283 | 0.0916 | 365 s | 3730 MiB | 4451 MB | 48538 | - | 8887 / 6953 |
| 6 | stream only, defaults (MTP on) | on (80% accepted) | on | on | 300 (cap) | 0.216 | 0.0350 | 765 s | 1102 MiB | 4130 MB | 45488 | 5148 / 3776 (73%) | 3800 / 8954 |
| 7 | stream only, MTP off | off | on | on | 300 (cap) | 0.232 | 0.0245 | 894 s | 1102 MiB | 2985 MB | 57912 | 15444 / 10496 (68%) | 5000 / 10840 |
| 8 | MTP off, SN_PROF=1 | off | on | on | 119 | 0.578 | 0.1003 | 265 s | 3730 MiB | 4799 MB | 57912 | 15444 / 10496 (68%) | 5000 / 10840 |

All timings real-world (fid-quant and other workers running).
