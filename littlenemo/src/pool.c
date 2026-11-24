/* pool.c - see pool.h. One implementation per platform behind the
 * same contract; the arithmetic a lane runs is the caller's serial
 * inner loop, so lanes never change a result, only who computes it. */
#include "pool.h"

#include <stdlib.h>
#include <string.h>

/* Per-worker lane ticket, shared by both platform halves. */
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

static DWORD WINAPI worker_main(LPVOID arg) {
    worker_arg *wa = (worker_arg *)arg;
    nemo_pool *p = wa->p;
    int lane = wa->lane;
    unsigned seen = 0;
    EnterCriticalSection(&p->mu);
    for (;;) {
        while (!p->shutdown && p->gen == seen)
            SleepConditionVariableCS(&p->cv_work, &p->mu, INFINITE);
        if (p->shutdown) break;
        seen = p->gen;
        nemo_pool_fn fn = p->fn;
        void *ctx = p->ctx;
        int n = p->n_items, lanes = p->lanes;
        int mine = lane < lanes;
        int b = (int)((long)lane * n / lanes);
        int e = (int)((long)(lane + 1) * n / lanes);
        LeaveCriticalSection(&p->mu);
        if (mine && e > b) fn(ctx, b, e);
        EnterCriticalSection(&p->mu);
        if (mine && ++p->lanes_done == p->lanes)
            WakeConditionVariable(&p->cv_done);
    }
    LeaveCriticalSection(&p->mu);
    return 0;
}

static DWORD WINAPI prep_main(LPVOID arg) {
    nemo_pool *p = (nemo_pool *)arg;
    unsigned seen = 0;
    EnterCriticalSection(&p->mu);
    for (;;) {
        while (!p->shutdown && p->prep_gen == seen)
            SleepConditionVariableCS(&p->cv_prep, &p->mu, INFINITE);
        if (p->shutdown) break;
        seen = p->prep_gen;
        nemo_prep_fn fn = p->prep_fn;
        void *ctx = p->prep_ctx;
        LeaveCriticalSection(&p->mu);
        fn(ctx);
        EnterCriticalSection(&p->mu);
        p->prep_busy = 0;
        WakeAllConditionVariable(&p->cv_prep_done);
    }
    LeaveCriticalSection(&p->mu);
    return 0;
}

int nemo_pool_init(nemo_pool *p, int nthreads) {
    memset(p, 0, sizeof *p);
    if (nthreads < 1) nthreads = 1;
    p->nthreads = nthreads;
    if (nthreads == 1) return 0;
    InitializeCriticalSection(&p->mu);
    InitializeConditionVariable(&p->cv_work);
    InitializeConditionVariable(&p->cv_done);
    InitializeConditionVariable(&p->cv_prep);
    InitializeConditionVariable(&p->cv_prep_done);
    p->workers = (HANDLE *)calloc((size_t)(nthreads - 1), sizeof(HANDLE));
    p->worker_args = calloc((size_t)(nthreads - 1), sizeof(worker_arg));
    if (!p->workers || !p->worker_args) goto degrade;
    for (int i = 0; i < nthreads - 1; i++) {
        worker_arg *wa = &((worker_arg *)p->worker_args)[i];
        wa->p = p;
        wa->lane = i + 1;
        p->workers[i] = CreateThread(NULL, 0, worker_main, wa, 0, NULL);
        if (!p->workers[i]) goto degrade;
    }
    p->prep_thread = CreateThread(NULL, 0, prep_main, p, 0, NULL);
    if (!p->prep_thread) goto degrade;
    p->spawned = 1;
    return 0;
degrade:
    /* A pool that cannot spawn runs serial rather than not at all;
     * nemo_pool_threads then reports the honest lane count. Any
     * threads that did start are shut down first (NULL handle slots
     * are no-ops for the join). */
    if (p->workers) p->spawned = 1;
    nemo_pool_shutdown(p);
    p->nthreads = 1;
    return 0;
}

void nemo_pool_shutdown(nemo_pool *p) {
    if (!p || !p->spawned) {
        if (p) memset(p, 0, sizeof *p);
        return;
    }
    EnterCriticalSection(&p->mu);
    p->shutdown = 1;
    WakeAllConditionVariable(&p->cv_work);
    WakeAllConditionVariable(&p->cv_prep);
    LeaveCriticalSection(&p->mu);
    for (int i = 0; i < p->nthreads - 1; i++) {
        WaitForSingleObject(p->workers[i], INFINITE);
        CloseHandle(p->workers[i]);
    }
    WaitForSingleObject(p->prep_thread, INFINITE);
    CloseHandle(p->prep_thread);
    free(p->workers);
    free(p->worker_args);
    DeleteCriticalSection(&p->mu);
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
    EnterCriticalSection(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n_items;
    p->lanes = lanes;
    p->lanes_done = 0;
    p->gen++;
    WakeAllConditionVariable(&p->cv_work);
    LeaveCriticalSection(&p->mu);
    int e = (int)((long)n_items / lanes);
    if (e > 0) fn(ctx, 0, e);
    EnterCriticalSection(&p->mu);
    if (++p->lanes_done < p->lanes)
        while (p->lanes_done < p->lanes)
            SleepConditionVariableCS(&p->cv_done, &p->mu, INFINITE);
    LeaveCriticalSection(&p->mu);
}

void nemo_pool_prep(nemo_pool *p, nemo_prep_fn fn, void *ctx) {
    if (!p->spawned) {
        fn(ctx);
        return;
    }
    EnterCriticalSection(&p->mu);
    while (p->prep_busy)
        SleepConditionVariableCS(&p->cv_prep_done, &p->mu, INFINITE);
    p->prep_fn = fn;
    p->prep_ctx = ctx;
    p->prep_busy = 1;
    p->prep_submitted = 1;
    p->prep_gen++;
    WakeConditionVariable(&p->cv_prep);
    LeaveCriticalSection(&p->mu);
}

void nemo_pool_prep_wait(nemo_pool *p) {
    if (!p->spawned) return;
    EnterCriticalSection(&p->mu);
    while (p->prep_busy)
        SleepConditionVariableCS(&p->cv_prep_done, &p->mu, INFINITE);
    LeaveCriticalSection(&p->mu);
}

#else /* POSIX */

#include <unistd.h>

int nemo_cores(void) {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
}

static void *worker_main(void *arg) {
    worker_arg *wa = (worker_arg *)arg;
    nemo_pool *p = wa->p;
    int lane = wa->lane;
    unsigned seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->shutdown && p->gen == seen)
            pthread_cond_wait(&p->cv_work, &p->mu);
        if (p->shutdown) break;
        seen = p->gen;
        nemo_pool_fn fn = p->fn;
        void *ctx = p->ctx;
        int n = p->n_items, lanes = p->lanes;
        int mine = lane < lanes;
        int b = (int)((long)lane * n / lanes);
        int e = (int)((long)(lane + 1) * n / lanes);
        pthread_mutex_unlock(&p->mu);
        if (mine && e > b) fn(ctx, b, e);
        pthread_mutex_lock(&p->mu);
        if (mine && ++p->lanes_done == p->lanes)
            pthread_cond_signal(&p->cv_done);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

static void *prep_main(void *arg) {
    nemo_pool *p = (nemo_pool *)arg;
    unsigned seen = 0;
    pthread_mutex_lock(&p->mu);
    for (;;) {
        while (!p->shutdown && p->prep_gen == seen)
            pthread_cond_wait(&p->cv_prep, &p->mu);
        if (p->shutdown) break;
        seen = p->prep_gen;
        nemo_prep_fn fn = p->prep_fn;
        void *ctx = p->prep_ctx;
        pthread_mutex_unlock(&p->mu);
        fn(ctx);
        pthread_mutex_lock(&p->mu);
        p->prep_busy = 0;
        pthread_cond_broadcast(&p->cv_prep_done);
    }
    pthread_mutex_unlock(&p->mu);
    return NULL;
}

int nemo_pool_init(nemo_pool *p, int nthreads) {
    memset(p, 0, sizeof *p);
    if (nthreads < 1) nthreads = 1;
    p->nthreads = nthreads;
    if (nthreads == 1) return 0;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv_work, NULL);
    pthread_cond_init(&p->cv_done, NULL);
    pthread_cond_init(&p->cv_prep, NULL);
    pthread_cond_init(&p->cv_prep_done, NULL);
    p->workers = (pthread_t *)calloc((size_t)(nthreads - 1),
                                     sizeof(pthread_t));
    p->worker_args = calloc((size_t)(nthreads - 1), sizeof(worker_arg));
    if (!p->workers || !p->worker_args) goto degrade;
    int made = 0;
    for (int i = 0; i < nthreads - 1; i++) {
        worker_arg *wa = &((worker_arg *)p->worker_args)[i];
        wa->p = p;
        wa->lane = i + 1;
        if (pthread_create(&p->workers[i], NULL, worker_main, wa) != 0)
            break;
        made++;
    }
    if (made != nthreads - 1) goto degrade_join;
    if (pthread_create(&p->prep_thread, NULL, prep_main, p) != 0)
        goto degrade_join;
    p->spawned = 1;
    return 0;
degrade_join:
    /* Join whoever started, then fall through to the serial degrade.
     * p->nthreads still names the intended lanes so shutdown joins
     * the right count only if every worker started; track via made. */
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_cond_broadcast(&p->cv_prep);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < made; i++) pthread_join(p->workers[i], NULL);
    free(p->workers);
    free(p->worker_args);
    p->workers = NULL;
    p->worker_args = NULL;
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    pthread_cond_destroy(&p->cv_prep);
    pthread_cond_destroy(&p->cv_prep_done);
    memset(p, 0, sizeof *p);
    p->nthreads = 1;
    return 0;
degrade:
    free(p->workers);
    free(p->worker_args);
    p->workers = NULL;
    p->worker_args = NULL;
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    pthread_cond_destroy(&p->cv_prep);
    pthread_cond_destroy(&p->cv_prep_done);
    memset(p, 0, sizeof *p);
    p->nthreads = 1;
    return 0;
}

void nemo_pool_shutdown(nemo_pool *p) {
    if (!p || !p->spawned) {
        if (p) memset(p, 0, sizeof *p);
        return;
    }
    pthread_mutex_lock(&p->mu);
    p->shutdown = 1;
    pthread_cond_broadcast(&p->cv_work);
    pthread_cond_broadcast(&p->cv_prep);
    pthread_mutex_unlock(&p->mu);
    for (int i = 0; i < p->nthreads - 1; i++)
        pthread_join(p->workers[i], NULL);
    pthread_join(p->prep_thread, NULL);
    free(p->workers);
    free(p->worker_args);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv_work);
    pthread_cond_destroy(&p->cv_done);
    pthread_cond_destroy(&p->cv_prep);
    pthread_cond_destroy(&p->cv_prep_done);
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
    pthread_mutex_lock(&p->mu);
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n_items;
    p->lanes = lanes;
    p->lanes_done = 0;
    p->gen++;
    pthread_cond_broadcast(&p->cv_work);
    pthread_mutex_unlock(&p->mu);
    int e = (int)((long)n_items / lanes);
    if (e > 0) fn(ctx, 0, e);
    pthread_mutex_lock(&p->mu);
    if (++p->lanes_done < p->lanes)
        while (p->lanes_done < p->lanes)
            pthread_cond_wait(&p->cv_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

void nemo_pool_prep(nemo_pool *p, nemo_prep_fn fn, void *ctx) {
    if (!p->spawned) {
        fn(ctx);
        return;
    }
    pthread_mutex_lock(&p->mu);
    while (p->prep_busy) pthread_cond_wait(&p->cv_prep_done, &p->mu);
    p->prep_fn = fn;
    p->prep_ctx = ctx;
    p->prep_busy = 1;
    p->prep_submitted = 1;
    p->prep_gen++;
    pthread_cond_signal(&p->cv_prep);
    pthread_mutex_unlock(&p->mu);
}

void nemo_pool_prep_wait(nemo_pool *p) {
    if (!p->spawned) return;
    pthread_mutex_lock(&p->mu);
    while (p->prep_busy) pthread_cond_wait(&p->cv_prep_done, &p->mu);
    pthread_mutex_unlock(&p->mu);
}

#endif
