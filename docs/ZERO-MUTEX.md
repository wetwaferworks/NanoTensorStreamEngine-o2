# The zero-mutex design

**The rule (JD, 2026-11-26): shipped userspace code contains no mutex.** No
mutex, condition variable, critical section, read-write lock or spinlock in the C
sources, and nothing behind them. The pure-C runtime and the Tensor Handshake
earned the rule: coordination is built from four rules, and the primitives live in
one header, `tools/nq2-handshake.h` (also vendored as
`littlenemo/src/nq2-handshake.h`).

## The four rules

1. **Phase-boundary programming.** Work proceeds in phases. Inside a phase every
   thread owns disjoint data; hand-off happens only at a boundary.
2. **Single-writer state, published with release stores and read with acquire
   loads.** Epoch counters answer "is it time?"; claim words taken with
   compare-and-swap answer "can I?", and exactly one owner clears each claim.
3. **Waiting is a spin with a pause, then a yield, then a sleep on the word
   itself** (a futex on Linux, `WaitOnAddress` on Windows). The sleep is only a
   sleep: it guards nothing and is never needed for correctness, which comes from
   re-checking the published state. The Linux sleep is bounded (50 ms).
4. **Results are reduced in a fixed order after the join.**

## The primitives (`nq2-handshake.h`)

| Primitive | What it is |
|---|---|
| `HS_LD(p)` / `HS_ST(p, v)` | acquire load / release store |
| `HS_LDX(p)` / `HS_STX(p, v)` | relaxed load / store |
| `HS_ADD(p, v)` / `HS_SUB(p, v)` | acq-rel add or subtract, returning the new value |
| `hs_cas_int(p, from, to)` | take a claim word from `from` to `to`; 1 if this caller won |
| `hs_raise_long(p, v)` | raise `*p` to at least `v`: a monotonic "seen" acknowledgement several threads may advance |
| `hs_pause()`, `hs_yield()` | the first two waiting steps |
| `hs_sleep_on(w, old)`, `hs_wake_all(w)` | the third step: futex or `WaitOnAddress` on the word |
| `hs_wait(w, old)` | spin (128 pauses), yield (8 times), then sleep until `*w` is no longer `old` |
| `hs_door_open(door)`, `hs_door_wait(door, seen)`, `hs_door_bump(door)` | the door (below) |
| `hs_exit_tick(exited)`, `hs_wait_count(ctr, target)` | a thread's last act is to count itself out; the joiner waits on a counter, never on a kernel object |
| `HS_THREAD_FN`, `HS_THREAD_RETURN` | one thread entry signature for Win32 threads and pthreads |
| the engine pool (`NQ2HS_ENGINE_POOL`) | `pool_init`, `pool_run`, `pool_shutdown`: one dispatch is one phase |

**The door.** Every waiter loads a 32-bit epoch word *before* it evaluates its
predicate and sleeps only while the word still holds that value; every writer
publishes its state first and bumps the word after. A bump between the load and
the sleep makes the sleep return at once, so no wakeup can be lost.

**The compute pool.** `pool_run(fn, ctx)` runs `fn(ctx, worker, nworkers)` on
every worker and returns when all are done. The core loop writes `fn` and `ctx`,
arms `pending`, and opens the phase by bumping the dispatch epoch; each worker
asks "is it time?" (the epoch moved), runs its disjoint share and counts itself
done. The core loop's next write to `fn` and `ctx` happens only after it has seen
`pending` reach zero. This replaced the mutex and two condition variables of the
original pool; behavior and partitions are unchanged, and results are
bit-identical to the single-lane path. If some threads fail to start the pool
shrinks to the contiguous prefix that did start (Codex review M-I5), and zero
started workers is served serially on the caller.

## Where the rule applies

| Component | How it coordinates |
|---|---|
| Triforce engine | ticks, the priority list, claim words, one door ([TENSOR-HANDSHAKE.md](TENSOR-HANDSHAKE.md)) |
| Little Nemo engine | the engine pool above; preparation concurrent, consumption serialized |
| `nq2-quantize` | disjoint chunks claimed through a lock-free counter (`atomic_next`); per-thread RNG state replays each tensor's serial draws, so output is byte-identical at any thread count |
| accel lanes | per-slot CAS ownership words; waits through the handshake door; an optional stager thread that only `memcpy`s into pinned memory and never calls the GPU API |
| `parakeet-c` | lock-free fork/join pool: atomics only, created and joined around each decode (phase boundaries) |

## Honest notes

- "Zero mutexes" is a property of our code, not of the CUDA and OpenCL runtimes:
  the vendor drivers use internal locks.
- The Python front end is outside the rule's scope (it renders and relays, and
  does no model work), and it is not lock-free: `nemo_scale.py` keeps one
  `threading.Lock` around its small scaled-image cache dictionary, never around the
  C call and never while scaling. The C scaler itself has no globals, no locks and
  no threads.
- Thread creation itself uses `CreateThread` / `pthread_create`; there is no
  blocking primitive after that.
- **Sanitizer records.** This repository carries no ThreadSanitizer run record.
  The Precision Gate status notes record one intermittent failure that is still
  open: under the gate engine at `--threads 1`, one of two attempts faulted at the
  first pass boundary ("the prior pass did not drain"); an identical rerun
  completed with the anchor token, and the unmodified base engine at `--threads 1`
  completed cleanly. Root cause undetermined; the transplant crew is told to
  instrument the boundary before the live-tree gate. The DTC hostile-geometry fuzz
  pass ran under ASan and UBSan with zero reports, and the Little Nemo suite is
  silent under both. Running the engines under ThreadSanitizer is on the
  [roadmap](ROADMAP.md).
- A grep for the usual lock names (`pthread_mutex`, `pthread_cond`,
  `CRITICAL_SECTION`, `SRWLOCK`, `CONDITION_VARIABLE`, `CreateMutex`,
  `CreateEvent`) across the C sources finds none; the only lock anywhere in the
  tree is the Python cache lock above.

## Credits

The Tensor Handshake: idea and shape, JD. The engines that use it: Elfie. The
zero-mutex conversion (2026-11-26): co-implemented by the assistants. Reviews that
shaped the pool and the door: Codex (Astra 6.2) M-I5, ChatGPT 6.2 Sol H1 and H5,
Miss Archie's Round 5.