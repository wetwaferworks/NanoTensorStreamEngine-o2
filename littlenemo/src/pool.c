/* pool.c - see pool.h. One implementation for both platforms
 * behind the same contract; the arithmetic a lane runs is the
 * caller's serial inner loop, so lanes never change a result,
 * only who computes it.
 *
 * Zero mutex (2026-11-26): the pool is a Tensor Handshake, the
 * cousin of JD's timing handshake. Workers ask the core loop
 * "is it time?" (the dispatch epoch moved) instead of waiting on
 * a condition variable under a lock. Original pool: Elfie;
 * handshake idea and shape: JD; conversion co-implemented by
 * the assistants. */
#include "pool.h"
#include "nq2-handshake.h"

#include <stdlib.h>
#include <string.h>

/* Per-worker lane ticket. */
typedef struct {
    nemo_pool *p;
    int lane;
} worker_arg;

int nemo_threads_for_mode(const char *mode, int cores) {
    if (cores < 1) cores = 1;
    if (!mode) return -1;
    if (!strcmp(mode, "minimum")) return 1;
    if (!strcmp(mode, "medium")) return (cores + 1) / 2;
    if (!strcmp(mode, "max")) return cores > 1 ? cores - 1 : 1;
    if (!strcmp(mode, "ultra")) return cores;
    return -1;
}

int nemo_pool_threads(const nemo_pool *p) { return p ? p->nthreads : 1; }

#if defined(_WIN32)
int nemo_cores(void) {
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
}
#else
#include <unistd.h>
int nemo_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}
#endif

static HS_THREAD_FN(worker_main, arg) {
    worker_arg *wa = (worker_arg *)arg;
    nemo_pool *p = wa->p;
    int lane = wa->lane;
    uint32_t seen = 0;
    for (;;) {
        uint32_t g;
        for (;;) {
            if (HS_LD(&p->shutdown)) goto out;
            g = HS_LD(&p->gen);
            if (g != seen) break;
            hs_wait(&p->gen, seen);
        }
        if (HS_LD(&p->shutdown)) goto out;
        seen = g;
        /* The dispatch fields were written before gen's release
         * and do not change until this lane has counted in. */
        nemo_pool_fn fn = p->fn;
        void *ctx = p->ctx;
        int n = p->n_items, lanes = p->lanes;
        if (lane < lanes) {
            int b = (int)((long)lane * n / lanes);
            int e = (int)((long)(lane + 1) * n / lanes);
            if (e > b) fn(ctx, b, e);
        }
        /* Every worker counts in, with or without rows this phase,
         * so the caller never rewrites the dispatch fields under a
         * worker that has yet to read them. */
        if (HS_ADD(&p->lanes_done, 1u) == (uint32_t)p->nthreads)
            hs_wake_all(&p->lanes_done);
    }
out:
    hs_exit_tick(&p->exited);
    HS_THREAD_RETURN;
}

static HS_THREAD_FN(prep_main, arg) {
    nemo_pool *p = (nemo_pool *)arg;
    uint32_t seen = 0;
    for (;;) {
        uint32_t g;
        for (;;) {
            if (HS_LD(&p->shutdown)) goto out;
            g = HS_LD(&p->prep_gen);
            if (g != seen) break;
            hs_wait(&p->prep_gen, seen);
        }
        if (HS_LD(&p->shutdown)) goto out;
        seen = g;
        nemo_prep_fn fn = p->prep_fn;
        void *ctx = p->prep_ctx;
        fn(ctx);
        HS_ST(&p->prep_busy, 0u);       /* the job's result is published */
        hs_wake_all(&p->prep_busy);
    }
out:
    hs_exit_tick(&p->exited);
    HS_THREAD_RETURN;
}

static int start_thread(nemo_thread *t, int is_prep, void *arg) {
#if defined(_WIN32)
    *t = CreateThread(NULL, 0, is_prep ? prep_main : worker_main, arg, 0,
                      NULL);
    return *t ? 0 : -1;
#else
    return pthread_create(t, NULL, is_prep ? prep_main : worker_main, arg)
        == 0 ? 0 : -1;
#endif
}

/* Stop and join whichever threads were started. */
static void stop_and_join(nemo_pool *p) {
    HS_ST(&p->shutdown, 1u);
    hs_door_bump(&p->gen);
    hs_door_bump(&p->prep_gen);
#if defined(_WIN32)
    /* The join: every started thread's last act is its exit tick.
     * Closing a handle of a thread that is still returning is
     * fine; it touches no pool memory after the tick. */
    hs_wait_count(&p->exited,
                  (uint32_t)(p->made_workers + p->made_prep));
    for (int i = 0; i < p->made_workers; i++) CloseHandle(p->workers[i]);
    if (p->made_prep) CloseHandle(p->prep_thread);
#else
    for (int i = 0; i < p->made_workers; i++)
        pthread_join(p->workers[i], NULL);
    if (p->made_prep) pthread_join(p->prep_thread, NULL);
#endif
    free(p->workers);
    free(p->worker_args);
    p->workers = NULL;
    p->worker_args = NULL;
}

int nemo_pool_init(nemo_pool *p, int nthreads) {
    memset(p, 0, sizeof *p);
    if (nthreads < 1) nthreads = 1;
    p->nthreads = nthreads;
    if (nthreads == 1) return 0;
    p->workers = (nemo_thread *)calloc((size_t)(nthreads - 1),
                                       sizeof(nemo_thread));
    p->worker_args = calloc((size_t)(nthreads - 1), sizeof(worker_arg));
    if (!p->workers || !p->worker_args) goto degrade;
    for (int i = 0; i < nthreads - 1; i++) {
        worker_arg *wa = &((worker_arg *)p->worker_args)[i];
        wa->p = p;
        wa->lane = i + 1;
        if (start_thread(&p->workers[i], 0, wa) != 0) goto degrade;
        p->made_workers++;
    }
    if (start_thread(&p->prep_thread, 1, p) != 0) goto degrade;
    p->made_prep = 1;
    p->spawned = 1;
    return 0;
degrade:
    /* A pool that cannot spawn runs serial rather than not at all;
     * nemo_pool_threads then reports the honest lane count. Any
     * threads that did start are stopped and joined first, and
     * only threads that started are ever joined or closed (Archie
     * P2-5). */
    stop_and_join(p);
    memset(p, 0, sizeof *p);
    p->nthreads = 1;
    return 0;
}

void nemo_pool_shutdown(nemo_pool *p) {
    if (!p || !p->spawned) {
        if (p) memset(p, 0, sizeof *p);
        return;
    }
    stop_and_join(p);
    memset(p, 0, sizeof *p);
}

void nemo_pool_for(nemo_pool *p, int n_items, nemo_pool_fn fn, void *ctx) {
    if (n_items <= 0 || !fn) return;
    if (!p->spawned) {
        fn(ctx, 0, n_items);
        return;
    }
    int lanes = p->nthreads;
    if (lanes > n_items) lanes = n_items;
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n_items;
    p->lanes = lanes;
    HS_ST(&p->lanes_done, 0u);
    hs_door_bump(&p->gen);              /* "it is time" */
    int e = (int)((long)n_items / lanes);
    if (e > 0) fn(ctx, 0, e);
    HS_ADD(&p->lanes_done, 1u);         /* lane 0 counts in */
    hs_wait_count(&p->lanes_done, (uint32_t)p->nthreads);  /* the join */
}

void nemo_pool_prep(nemo_pool *p, nemo_prep_fn fn, void *ctx) {
    if (!p->spawned) {
        fn(ctx);
        return;
    }
    /* At most one outstanding job: wait for the last to finish. */
    for (;;) {
        uint32_t b = HS_LD(&p->prep_busy);
        if (!b) break;
        hs_wait(&p->prep_busy, b);
    }
    p->prep_fn = fn;
    p->prep_ctx = ctx;
    p->prep_submitted = 1;
    HS_ST(&p->prep_busy, 1u);
    hs_door_bump(&p->prep_gen);
}

void nemo_pool_prep_wait(nemo_pool *p) {
    if (!p->spawned) return;
    for (;;) {
        uint32_t b = HS_LD(&p->prep_busy);
        if (!b) return;
        hs_wait(&p->prep_busy, b);
    }
}
