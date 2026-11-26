/* pool.h - Little Nemo worker pool, pure C.
 *
 * Two services over one small pool:
 *
 *   nemo_pool_for    splits a row range across the compute lanes
 *                    (the caller is lane 0). Each lane's rows are
 *                    computed by the same serial inner loop, so the
 *                    result is bit-identical to the serial path:
 *                    parallelism changes who computes a row, never
 *                    the arithmetic of a row.
 *
 *   nemo_pool_prep   runs one background job on a dedicated prep
 *                    thread (fetch layer n+1 while layer n computes:
 *                    concurrent preparation, serialized consumption)
 *                    and nemo_pool_prep_wait collects it. With one
 *                    lane the prep job runs inline at submit time,
 *                    which keeps the serial model the reference.
 *
 * Thread counts come from the machine: nemo_cores() queries the OS,
 * and the mode names map onto it (JD's modes, 2026-11-24):
 *   minimum = 1 lane, medium = half the cores (rounded up),
 *   max = all but one core, ultra = every core.
 * An explicit lane count overrides the mode.
 */
#ifndef NEMO_POOL_H
#define NEMO_POOL_H

/* A parallel-for body: compute items [begin, end) of ctx. */
typedef void (*nemo_pool_fn)(void *ctx, int begin, int end);
/* A background prep body. */
typedef void (*nemo_prep_fn)(void *ctx);

#include <stdint.h>
#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN /* keep GDI et al. out of our namespace */
#endif
#include <windows.h>
typedef HANDLE nemo_thread;
#else
#include <pthread.h>
typedef pthread_t nemo_thread;
#endif

/* Zero mutex (Tensor Handshake, 2026-11-26): no lock or condvar
 * anywhere in the pool. One dispatch is one phase: the caller
 * (lane 0, the core loop) writes fn/ctx/n_items/lanes, zeroes
 * lanes_done and opens the phase by bumping gen (release). Every
 * worker asks "is it time?" (gen moved), runs its disjoint row
 * range if it has one, and counts itself in lanes_done; the
 * caller's next write to the dispatch fields happens only after
 * every lane, the caller included, has counted in. The prep job
 * is the same handshake on prep_gen / prep_busy. Waits are a
 * spin, a yield, then a futex / WaitOnAddress sleep on the word. */
typedef struct {
    int nthreads; /* compute lanes, >= 1 */
    int spawned;  /* worker threads exist */
    nemo_thread *workers;
    nemo_thread prep_thread;
    int made_workers; /* worker threads actually started */
    int made_prep;    /* prep thread started */
    nemo_pool_fn fn;  /* dispatch fields: caller-written between phases */
    void *ctx;
    int n_items, lanes;
    uint32_t gen;        /* dispatch epoch; the caller is the one writer */
    uint32_t lanes_done; /* lanes counted in this phase */
    nemo_prep_fn prep_fn;
    void *prep_ctx;
    uint32_t prep_busy;  /* 1 from submit until the job has run */
    int prep_submitted;
    uint32_t prep_gen;   /* prep epoch; the caller is the one writer */
    uint32_t shutdown;
    uint32_t exited;     /* exit ticks (the Windows join) */
    void *worker_args; /* per-worker lane structs, freed at shutdown */
} nemo_pool;

/* Number of logical cores the OS reports (>= 1). */
int nemo_cores(void);

/* Lanes for a mode name on a machine with `cores` cores:
 * "minimum" 1, "medium" ceil(cores/2), "max" cores-1 (at least 1),
 * "ultra" cores. Returns -1 for an unknown mode name. */
int nemo_threads_for_mode(const char *mode, int cores);

/* Bring the pool up with nthreads compute lanes (clamped to >= 1).
 * One lane means no threads at all: everything runs inline and the
 * serial path stays the reference implementation. Returns 0 on
 * success. Safe on a zeroed struct; shutdown is safe to call once. */
int nemo_pool_init(nemo_pool *p, int nthreads);
void nemo_pool_shutdown(nemo_pool *p);
int nemo_pool_threads(const nemo_pool *p);

/* Split [0, n_items) across the lanes and wait for all of them. */
void nemo_pool_for(nemo_pool *p, int n_items, nemo_pool_fn fn, void *ctx);

/* Queue one background job (at most one outstanding) and collect it.
 * prep_wait blocks until the queued job has finished; calling it
 * with nothing queued returns at once. */
void nemo_pool_prep(nemo_pool *p, nemo_prep_fn fn, void *ctx);
void nemo_pool_prep_wait(nemo_pool *p);

#endif
