/* nq2-handshake.h - the Tensor Handshake primitives, pure C.
 *
 * The Tensor Handshake is a cousin of the timing handshake: the
 * idea and its shape are JD's (the tick and control loops ask the
 * core loop "is it time to tick?" and "can I tick?" instead of
 * taking a lock). The engines that use it are Elfie's; the
 * zero-mutex conversion (2026-11-26) was co-implemented by the
 * assistants.
 *
 * There are no mutexes, condition variables, critical sections,
 * read-write locks or spinlocks here or behind anything here.
 * Coordination is built from four rules only:
 *
 *   1. Phase-boundary programming. Work proceeds in phases. Inside
 *      a phase every thread owns disjoint data; hand-off happens
 *      only at a boundary.
 *   2. Single-writer state published with release stores and read
 *      with acquire loads. Epoch counters answer "is it time?";
 *      claim words taken with compare-and-swap answer "can I?",
 *      and exactly one owner clears each claim.
 *   3. Waiting is a spin with a pause, then a yield, then a sleep
 *      on the word itself (futex on Linux, WaitOnAddress on
 *      Windows). The sleep is only a sleep: it guards nothing and
 *      is never needed for correctness, which comes from
 *      re-checking the published state.
 *   4. Results are reduced in a fixed order after the join.
 *
 * The door: every waiter loads a 32-bit epoch word BEFORE it
 * evaluates its predicate and sleeps only while the word still
 * holds that value; every writer publishes its state first and
 * bumps the word after. A bump between the load and the sleep
 * makes the sleep return at once, so no wakeup can be lost.
 *
 * GCC and Clang (MinGW included) __atomic builtins; the build
 * flags stay -std=c99.
 */
#ifndef NQ2_HANDSHAKE_H
#define NQ2_HANDSHAKE_H

#include <stdint.h>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#include <time.h>
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#error "the Tensor Handshake needs the GCC/Clang __atomic builtins"
#endif

/* Acquire / release / relaxed access and read-modify-write. */
#define HS_LD(p)        __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define HS_LDX(p)       __atomic_load_n((p), __ATOMIC_RELAXED)
#define HS_ST(p, v)     __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define HS_STX(p, v)    __atomic_store_n((p), (v), __ATOMIC_RELAXED)
#define HS_ADD(p, v)    __atomic_add_fetch((p), (v), __ATOMIC_ACQ_REL)
#define HS_SUB(p, v)    __atomic_sub_fetch((p), (v), __ATOMIC_ACQ_REL)

/* Claim word: take it from `from` to `to`; 1 if this caller won. */
static inline int hs_cas_int(int *p, int from, int to)
{
    return __atomic_compare_exchange_n(p, &from, to, 0,
                                       __ATOMIC_ACQ_REL,
                                       __ATOMIC_ACQUIRE);
}

/* Raise *p to at least v (a monotonic "seen" acknowledgement that
 * several threads may advance). Returns 1 if it changed. */
static inline int hs_raise_long(long *p, long v)
{
    long cur = HS_LD(p);
    while (cur < v)
        if (__atomic_compare_exchange_n(p, &cur, v, 1,
                                        __ATOMIC_ACQ_REL,
                                        __ATOMIC_ACQUIRE))
            return 1;
    return 0;
}

static inline void hs_pause(void)
{
#if defined(__x86_64__) || defined(__i386__)
    __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
    __asm__ volatile("yield" ::: "memory");
#else
    __asm__ volatile("" ::: "memory");
#endif
}

static inline void hs_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    sched_yield();
#endif
}

/* ---- the sleep: futex / WaitOnAddress, a sleep primitive only ---- */
#ifdef _WIN32
typedef BOOL (WINAPI *hs_wait_on_address_fn)(volatile VOID *, PVOID,
                                             SIZE_T, DWORD);
typedef VOID (WINAPI *hs_wake_by_address_fn)(PVOID);
/* Resolved once at first use; a racing first use resolves the same
 * values, and the pointers are published atomically. */
static hs_wait_on_address_fn hs_win_wait_p;
static hs_wake_by_address_fn hs_win_wake_p;
static int hs_win_state; /* 0 unresolved, 1 resolved, 2 missing */

static inline void hs_win_resolve(void)
{
    if (HS_LD(&hs_win_state)) return;
    HMODULE m = LoadLibraryA("api-ms-win-core-synch-l1-2-0.dll");
    if (!m) m = GetModuleHandleA("kernelbase.dll");
    hs_wait_on_address_fn w = NULL;
    hs_wake_by_address_fn k = NULL;
    if (m) {
        FARPROC fw = GetProcAddress(m, "WaitOnAddress");
        FARPROC fk = GetProcAddress(m, "WakeByAddressAll");
        /* memcpy-free function pointer conversion via a union. */
        union { FARPROC f; hs_wait_on_address_fn w; } uw;
        union { FARPROC f; hs_wake_by_address_fn k; } uk;
        uw.f = fw; uk.f = fk;
        if (fw && fk) { w = uw.w; k = uk.k; }
    }
    __atomic_store_n(&hs_win_wait_p, w, __ATOMIC_RELEASE);
    __atomic_store_n(&hs_win_wake_p, k, __ATOMIC_RELEASE);
    HS_ST(&hs_win_state, (w && k) ? 1 : 2);
}

static inline void hs_sleep_on(uint32_t *w, uint32_t old)
{
    hs_win_resolve();
    hs_wait_on_address_fn f = __atomic_load_n(&hs_win_wait_p,
                                              __ATOMIC_ACQUIRE);
    if (f) f((volatile VOID *)w, &old, sizeof old, 50);
    else Sleep(1);
}

static inline void hs_wake_all(uint32_t *w)
{
    hs_win_resolve();
    hs_wake_by_address_fn f = __atomic_load_n(&hs_win_wake_p,
                                              __ATOMIC_ACQUIRE);
    if (f) f((PVOID)w);
}
#else
#if defined(__linux__) && defined(__x86_64__)
/* futex(2) by hand: -std=c99 hides syscall(), and assembly is
 * allowed. FUTEX_WAIT_PRIVATE 128, FUTEX_WAKE_PRIVATE 129. */
static inline long hs_futex(uint32_t *addr, long op, long val, void *ts)
{
    long r;
    register long r10 __asm__("r10") = (long)ts;
    __asm__ volatile("syscall"
                     : "=a"(r)
                     : "a"(202L), "D"(addr), "S"(op), "d"(val),
                       "r"(r10)
                     : "rcx", "r11", "memory");
    return r;
}
#define HS_HAVE_FUTEX 1
#elif defined(__linux__) && defined(__aarch64__)
static inline long hs_futex(uint32_t *addr, long op, long val, void *ts)
{
    register long x8 __asm__("x8") = 98;
    register long x0 __asm__("x0") = (long)addr;
    register long x1 __asm__("x1") = op;
    register long x2 __asm__("x2") = val;
    register long x3 __asm__("x3") = (long)ts;
    __asm__ volatile("svc 0"
                     : "+r"(x0)
                     : "r"(x8), "r"(x1), "r"(x2), "r"(x3)
                     : "memory");
    return x0;
}
#define HS_HAVE_FUTEX 1
#endif

static inline void hs_sleep_on(uint32_t *w, uint32_t old)
{
#ifdef HS_HAVE_FUTEX
    /* A bounded sleep (50 ms): correctness never depends on the
     * wake, only on re-checking the published state. */
    struct timespec ts = { 0, 50000000L };
    hs_futex(w, 128, (long)old, &ts);
#else
    struct timespec ts = { 0, 50000L };
    (void)w; (void)old;
    nanosleep(&ts, NULL);
#endif
}

static inline void hs_wake_all(uint32_t *w)
{
#ifdef HS_HAVE_FUTEX
    hs_futex(w, 129, 0x7fffffffL, NULL);
#else
    (void)w;
#endif
}
#endif

/* Wait until *w no longer holds `old`: spin with a pause, then
 * yield, then sleep on the word. */
static inline void hs_wait(uint32_t *w, uint32_t old)
{
    for (int i = 0; i < 128; i++) {
        if (HS_LD(w) != old) return;
        hs_pause();
    }
    for (int i = 0; i < 8; i++) {
        if (HS_LD(w) != old) return;
        hs_yield();
    }
    while (HS_LD(w) == old) hs_sleep_on(w, old);
}

/* The door: open() before evaluating a predicate, wait(seen) when
 * it is false, bump() after every published change. */
static inline uint32_t hs_door_open(uint32_t *door)
{
    return __atomic_load_n(door, __ATOMIC_SEQ_CST);
}
static inline void hs_door_wait(uint32_t *door, uint32_t seen)
{
    hs_wait(door, seen);
}
static inline void hs_door_bump(uint32_t *door)
{
    __atomic_add_fetch(door, 1u, __ATOMIC_SEQ_CST);
    hs_wake_all(door);
}

/* Thread entry signature and the exit tick: a thread's last act
 * is to count itself out, so the joiner waits on a counter and
 * never on a kernel object. */
#ifdef _WIN32
#define HS_THREAD_FN(name, arg) DWORD WINAPI name(LPVOID arg)
#define HS_THREAD_RETURN return 0
#else
#define HS_THREAD_FN(name, arg) void *name(void *arg)
#define HS_THREAD_RETURN return NULL
#endif

static inline void hs_exit_tick(uint32_t *exited)
{
    HS_ADD(exited, 1u);
    hs_wake_all(exited);
}

static inline void hs_wait_count(uint32_t *ctr, uint32_t target)
{
    for (;;) {
        uint32_t v = HS_LD(ctr);
        if (v >= target) return;
        hs_wait(ctr, v);
    }
}

#ifdef NQ2HS_ENGINE_POOL
/* ---------------- the engines' compute pool ----------------
 *
 * pool_run(fn, ctx) runs fn(ctx, worker, nworkers) on every worker
 * and returns when all are done. One dispatch is one phase: the
 * core loop writes fn/ctx, arms `pending`, and opens the phase by
 * bumping the dispatch epoch (release). Each worker asks "is it
 * time?" (the epoch moved), runs its disjoint share, and counts
 * itself done (acq_rel). The core loop's next write to fn/ctx
 * happens only after it has seen pending reach zero, so fn/ctx
 * never change under a worker. Replaces the mutex + two condvars
 * of Elfie's original pool; behaviour and partitions unchanged.
 */
typedef void (*work_fn)(void *ctx, int worker, int nworkers);

typedef struct {
#ifdef _WIN32
    HANDLE *threads;
#else
    pthread_t *threads;
#endif
    int nworkers;
    int started;        /* threads actually launched (M-I5):
                         * shutdown joins exactly these */
    work_fn fn;         /* written by the core loop between phases */
    void *ctx;
    uint32_t dispatch;  /* phase epoch: core loop is the one writer */
    uint32_t pending;   /* workers that still owe a completion */
    uint32_t stop;
    uint32_t exited;    /* exit ticks */
} Pool;

static Pool g_pool;

static HS_THREAD_FN(pool_worker, arg)
{
    int me = (int)(intptr_t)arg;
    uint32_t seen = 0;
    for (;;) {
        uint32_t d;
        for (;;) {
            if (HS_LD(&g_pool.stop)) {
                hs_exit_tick(&g_pool.exited);
                HS_THREAD_RETURN;
            }
            d = HS_LD(&g_pool.dispatch);
            if (d != seen) break;
            hs_wait(&g_pool.dispatch, seen);
        }
        if (HS_LD(&g_pool.stop)) {       /* the shutdown bump */
            hs_exit_tick(&g_pool.exited);
            HS_THREAD_RETURN;
        }
        seen = d;
        work_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        int nw = g_pool.nworkers;
        fn(ctx, me, nw);
        if (HS_SUB(&g_pool.pending, 1u) == 0)
            hs_wake_all(&g_pool.pending);
    }
}

static void pool_init(int nworkers)
{
    if (nworkers < 1) nworkers = 1;
    g_pool.nworkers = nworkers;
    /* Launch failures shrink the pool to the workers that
     * actually started (Codex Astra review M-I5, 2026-11-25):
     * launches are sequential, so the started set is always the
     * contiguous prefix [0, started) and dispatch partitions
     * stay exact. Zero started workers leaves nworkers at 1,
     * which pool_run serves serially on the caller. nworkers is
     * written before any worker starts and never again while
     * they run, except by the shrink below, which happens
     * before the first dispatch. */
    int started = 0;
#ifdef _WIN32
    g_pool.threads = (HANDLE *)malloc(sizeof(HANDLE) * nworkers);
    for (int i = 0; i < nworkers; i++) {
        g_pool.threads[i] = CreateThread(NULL, 0, pool_worker,
                                         (LPVOID)(intptr_t)i, 0, NULL);
        if (!g_pool.threads[i]) break;
        started++;
    }
#else
    g_pool.threads = (pthread_t *)malloc(sizeof(pthread_t) * nworkers);
    for (int i = 0; i < nworkers; i++) {
        if (pthread_create(&g_pool.threads[i], NULL, pool_worker,
                           (void *)(intptr_t)i) != 0)
            break;
        started++;
    }
#endif
    if (started < nworkers) {
        fprintf(stderr, "pool: %d of %d workers started; "
                "continuing with the smaller pool\n",
                started, nworkers);
        g_pool.started = started;
        g_pool.nworkers = started > 0 ? started : 1;
    } else {
        g_pool.started = started;
    }
}

/* Run fn(ctx, worker, nworkers) on every worker and wait. */
static void pool_run(work_fn fn, void *ctx)
{
    if (g_pool.nworkers <= 1) { fn(ctx, 0, 1); return; }
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    HS_ST(&g_pool.pending, (uint32_t)g_pool.nworkers);
    hs_door_bump(&g_pool.dispatch);         /* "it is time" */
    for (;;) {                              /* the join */
        uint32_t p = HS_LD(&g_pool.pending);
        if (p == 0) break;
        hs_wait(&g_pool.pending, p);
    }
}

static void pool_shutdown(void)
{
    HS_ST(&g_pool.stop, 1u);
    hs_door_bump(&g_pool.dispatch);
#ifdef _WIN32
    hs_wait_count(&g_pool.exited, (uint32_t)g_pool.started);
    for (int i = 0; i < g_pool.started; i++)
        CloseHandle(g_pool.threads[i]);
#else
    for (int i = 0; i < g_pool.started; i++)
        pthread_join(g_pool.threads[i], NULL);
#endif
}
#endif /* NQ2HS_ENGINE_POOL */

#endif /* NQ2_HANDSHAKE_H */
