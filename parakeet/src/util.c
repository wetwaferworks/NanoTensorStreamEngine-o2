/* parakeet-c utils: clock, aligned alloc, lock-free fork/join pool, safetensors mmap */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define PK_PAUSE() _mm_pause()
#else
#define PK_PAUSE() ((void)0)
#endif

double pk_now(void) {
#ifdef _WIN32
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

void *pk_alloc(size_t n) {
    if (n == 0) n = 64;
    n = (n + 63) & ~(size_t)63;
#ifdef _WIN32
    void *p = _aligned_malloc(n, 64);
#else
    void *p = NULL; if (posix_memalign(&p, 64, n)) p = NULL;
#endif
    if (!p) { fprintf(stderr, "parakeet-c: out of memory (%zu bytes)\n", n); exit(3); }
    memset(p, 0, n);
    return p;
}
void pk_free(void *p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
float pk_bf16_to_f32(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static float f16_to_f32(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, u;
    if (e == 0) { if (!m) u = s << 31; else { e = 127 - 15 + 1; while (!(m & 1024)) { m <<= 1; e--; } m &= 1023; u = (s << 31) | (e << 23) | (m << 13); } }
    else if (e == 31) u = (s << 31) | 0x7f800000 | (m << 13);
    else u = (s << 31) | ((e + 112) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}

/* ------------------------------------------------------------------ pool
 * Fork/join with atomics only. Workers exist only between pk_pool_start and
 * pk_pool_stop (created/joined = phase boundaries). Inside, a job is a
 * generation bump; workers grab task indices with fetch_add and check out.
 * The caller waits until every task is done AND every worker checked out,
 * so no worker can ever see a mix of two jobs. No mutex, no condvar. */
static struct {
    int nworkers;
    _Atomic unsigned gen;
    _Atomic int next, done, out, quit;
    int ntasks; pk_task_fn fn; void *ctx;
#ifdef _WIN32
    HANDLE th[64];
#else
    pthread_t th[64];
#endif
} P;

static void spin_relax(unsigned *spins) {
    PK_PAUSE();
    if (++*spins > 4000) {
        *spins = 0;
#ifdef _WIN32
        SwitchToThread();
#else
        sched_yield();
#endif
    }
}

#ifdef _WIN32
static DWORD WINAPI worker_main(LPVOID arg)
#else
static void *worker_main(void *arg)
#endif
{
    (void)arg;
    unsigned seen = 0, spins = 0;
    for (;;) {
        unsigned g;
        while ((g = atomic_load_explicit(&P.gen, memory_order_acquire)) == seen) {
            if (atomic_load_explicit(&P.quit, memory_order_acquire)) return 0;
            spin_relax(&spins);
        }
        seen = g;
        pk_task_fn fn = P.fn; void *ctx = P.ctx; int nt = P.ntasks;
        int i;
        while ((i = atomic_fetch_add_explicit(&P.next, 1, memory_order_acq_rel)) < nt) {
            fn(ctx, i);
            atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
        }
        atomic_fetch_add_explicit(&P.out, 1, memory_order_release);
    }
}

void pk_pool_start(int nthreads) {
    if (nthreads < 1) nthreads = 1;
    if (nthreads > 64) nthreads = 64;
    P.nworkers = nthreads - 1;
    atomic_store(&P.quit, 0);
    atomic_store(&P.gen, 0);
    for (int i = 0; i < P.nworkers; i++) {
#ifdef _WIN32
        P.th[i] = CreateThread(NULL, 0, worker_main, NULL, 0, NULL);
#else
        pthread_create(&P.th[i], NULL, worker_main, NULL);
#endif
    }
}
void pk_pool_stop(void) {
    atomic_store_explicit(&P.quit, 1, memory_order_release);
    for (int i = 0; i < P.nworkers; i++) {
#ifdef _WIN32
        WaitForSingleObject(P.th[i], INFINITE); CloseHandle(P.th[i]);
#else
        pthread_join(P.th[i], NULL);
#endif
    }
    P.nworkers = 0;
}
int pk_pool_threads(void) { return P.nworkers + 1; }

void pk_parallel_for(int ntasks, pk_task_fn fn, void *ctx) {
    if (ntasks <= 0) return;
    if (P.nworkers == 0 || ntasks == 1) { for (int i = 0; i < ntasks; i++) fn(ctx, i); return; }
    P.fn = fn; P.ctx = ctx; P.ntasks = ntasks;
    atomic_store_explicit(&P.next, 0, memory_order_relaxed);
    atomic_store_explicit(&P.done, 0, memory_order_relaxed);
    atomic_store_explicit(&P.out, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&P.gen, 1, memory_order_acq_rel);
    int i;
    while ((i = atomic_fetch_add_explicit(&P.next, 1, memory_order_acq_rel)) < ntasks) {
        fn(ctx, i);
        atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
    }
    unsigned spins = 0;
    while (atomic_load_explicit(&P.done, memory_order_acquire) < ntasks ||
           atomic_load_explicit(&P.out, memory_order_acquire) < P.nworkers)
        spin_relax(&spins);
}

/* ------------------------------------------------------------ safetensors */
static const char *skip_ws(const char *p, const char *e) { while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t' || *p == ',' || *p == ':')) p++; return p; }
static const char *read_str(const char *p, const char *e, char *out, int cap) {
    int n = 0;
    if (p >= e || *p != '"') return NULL;
    p++;
    while (p < e && *p != '"') { if (*p == '\\' && p + 1 < e) p++; if (n < cap - 1) out[n++] = *p; p++; }
    out[n] = 0;
    return p < e ? p + 1 : NULL;
}
static const char *skip_value(const char *p, const char *e) {
    p = skip_ws(p, e);
    if (p >= e) return e;
    if (*p == '"') { char tmp[4]; (void)tmp; p++; while (p < e && *p != '"') { if (*p == '\\') p++; p++; } return p + 1; }
    if (*p == '{' || *p == '[') {
        int depth = 0;
        do {
            if (*p == '"') { p++; while (p < e && *p != '"') { if (*p == '\\') p++; p++; } }
            else if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') depth--;
            p++;
        } while (p < e && depth > 0);
        return p;
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']') p++;
    return p;
}

int pk_st_open(pk_st *st, const char *path) {
    memset(st, 0, sizeof *st);
#ifdef _WIN32
    HANDLE f = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS, NULL);
    if (f == INVALID_HANDLE_VALUE) return -1;
    LARGE_INTEGER sz; GetFileSizeEx(f, &sz);
    HANDLE m = CreateFileMappingA(f, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!m) { CloseHandle(f); return -1; }
    const uint8_t *base = (const uint8_t *)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
    if (!base) { CloseHandle(m); CloseHandle(f); return -1; }
    st->file_handle = f; st->map_handle = m; st->size = (size_t)sz.QuadPart;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    struct stat sb; fstat(fd, &sb);
    const uint8_t *base = mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { close(fd); return -1; }
    st->file_handle = (void *)(intptr_t)fd; st->size = sb.st_size;
#endif
    st->base = base;
    uint64_t hl; memcpy(&hl, base, 8);
    if (st->size < 8 || hl > st->size - 8) return -2;
    const char *p = (const char *)base + 8, *e = p + hl;
    const uint8_t *data0 = base + 8 + hl;
    int cap = 1024; st->t = calloc(cap, sizeof(pk_tensor));
    p = skip_ws(p, e);
    if (*p != '{') return -3;
    p++;
    for (;;) {
        p = skip_ws(p, e);
        if (p >= e || *p == '}') break;
        char key[160];
        p = read_str(p, e, key, sizeof key);
        if (!p) return -4;
        p = skip_ws(p, e);
        if (!strcmp(key, "__metadata__")) { p = skip_value(p, e); continue; }
        if (st->n == cap) { cap *= 2; st->t = realloc(st->t, cap * sizeof(pk_tensor)); }
        pk_tensor *t = &st->t[st->n]; memset(t, 0, sizeof *t);
        strncpy(t->name, key, sizeof t->name - 1);
        if (*p != '{') return -5;
        p++;
        uint64_t off0 = 0, off1 = 0;
        for (;;) {
            p = skip_ws(p, e);
            if (*p == '}') { p++; break; }
            char f[32]; p = read_str(p, e, f, sizeof f); if (!p) return -6;
            p = skip_ws(p, e);
            if (!strcmp(f, "dtype")) {
                char d[16]; p = read_str(p, e, d, sizeof d);
                t->dtype = !strcmp(d, "F32") ? PK_F32 : !strcmp(d, "BF16") ? PK_BF16 : !strcmp(d, "F16") ? PK_F16 : !strcmp(d, "I64") ? PK_I64 : PK_OTHER;
            } else if (!strcmp(f, "shape") || !strcmp(f, "data_offsets")) {
                int isshape = f[0] == 's'; int k = 0;
                p++; /* [ */
                for (;;) {
                    p = skip_ws(p, e);
                    if (*p == ']') { p++; break; }
                    uint64_t v = strtoull(p, (char **)&p, 10);
                    if (isshape) { if (k < 4) t->shape[k] = (int64_t)v; k++; }
                    else { if (k == 0) off0 = v; else off1 = v; k++; }
                }
                if (isshape) t->ndim = k;
            } else p = skip_value(p, e);
        }
        if (off0 > off1 || off1 > (uint64_t)st->size - 8 - hl) return -7;
        t->data = data0 + off0; t->nbytes = (size_t)(off1 - off0);
        /* The shape must fit the four stored dimensions and describe
         * exactly the bytes the offsets name, or a later copy-out reads
         * past the tensor (or the mapping). */
        if (t->ndim > 4) return -8;
        {
            uint64_t n = 1, es = t->dtype == PK_F32 ? 4 : (t->dtype == PK_BF16 || t->dtype == PK_F16) ? 2 : t->dtype == PK_I64 ? 8 : 0;
            for (int i = 0; i < t->ndim; i++) {
                if (t->shape[i] < 0 || (t->shape[i] && n > UINT64_MAX / (uint64_t)t->shape[i])) return -9;
                n *= (uint64_t)t->shape[i];
            }
            if (es && (n > UINT64_MAX / es || n * es != (uint64_t)t->nbytes)) return -9;
        }
        st->n++;
    }
    return 0;
}
void pk_st_close(pk_st *st) {
#ifdef _WIN32
    if (st->base) UnmapViewOfFile(st->base);
    if (st->map_handle) CloseHandle(st->map_handle);
    if (st->file_handle) CloseHandle(st->file_handle);
#else
    if (st->base) munmap((void *)st->base, st->size);
    if (st->file_handle) close((int)(intptr_t)st->file_handle);
#endif
    free(st->t);
    memset(st, 0, sizeof *st);
}
const pk_tensor *pk_st_get(const pk_st *st, const char *name) {
    for (int i = 0; i < st->n; i++) if (!strcmp(st->t[i].name, name)) return &st->t[i];
    return NULL;
}
int64_t pk_tensor_numel(const pk_tensor *t) { int64_t n = 1; for (int i = 0; i < t->ndim; i++) n *= t->shape[i]; return n; }
float *pk_tensor_f32(const pk_tensor *t) {
    int64_t n = pk_tensor_numel(t);
    float *o = pk_alloc(n * sizeof(float));
    if (t->dtype == PK_F32) memcpy(o, t->data, n * 4);
    else if (t->dtype == PK_BF16) for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, t->data + 2 * i, 2); o[i] = pk_bf16_to_f32(h); }
    else if (t->dtype == PK_F16) for (int64_t i = 0; i < n; i++) { uint16_t h; memcpy(&h, t->data + 2 * i, 2); o[i] = f16_to_f32(h); }
    return o;
}
