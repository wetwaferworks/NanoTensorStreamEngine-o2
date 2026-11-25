/* nq2-infer.c ... NanoQuatTwo inference engine.
 *
 * Implemented by Elfie (E.L.F.I.E.): the engine and this code.
 * Direction and executive planning by JD.
 *
 * A pure C99 inference engine for continuous-diffusion language
 * models of the Nemotron-Labs-Sigma family, built to run on
 * NanoQuatTwo packed models (.nq2 + .nq2book) as well as on the
 * original safetensors weights. The architecture and sampling
 * procedure follow NVIDIA's published reference implementation
 * (modeling_nemotron_labs_sigma.py, NVIDIA Open Model License):
 * blockwise denoising of 16-dimensional continuous embeddings
 * with a KV-cached clean prefix, self-conditioning, optional
 * classifier-free guidance, and DDIM/DDPM transitions.
 *
 * Streaming design (JD's prefetch ruling, 2026-11-23): weights
 * flow disc -> memory -> CPU through a two-deep pipeline. The
 * forward pass consumes tensors in a fixed order; while one
 * tensor computes, the next is already materializing, and the
 * one after waits on disc. At most two weight tensors are ever
 * open at once, however large the model grows. The two big
 * matrices (token embedding, diffusion head) are never wholly
 * materialized: embeddings gather rows, the head multiplies in
 * row tiles.
 *
 * Thread-smart (JD's ruling): a worker pool sizes itself to the
 * machine's cores and partitions every matmul by rows; a
 * dedicated prefetch thread keeps the pipeline fed. Buffer sizes
 * are derived from the model's own tensor shapes at load, never
 * hardcoded.
 *
 * Scope of this first engine: token IDs in, token IDs out.
 * The text tokenizer is the next piece, on its own.
 *
 * Build: cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer \
 *          tools/nq2-infer.c -lm
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <dirent.h> /* opendir/readdir directory scan (Archie P0-3) */

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif

/* ================= threads ================= */

static int cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

typedef void (*work_fn)(void *ctx, int worker, int nworkers);

typedef struct {
#ifdef _WIN32
    HANDLE *threads;
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE work_cv, done_cv;
#else
    pthread_t *threads;
    pthread_mutex_t lock;
    pthread_cond_t work_cv, done_cv;
#endif
    int nworkers;
    work_fn fn;
    void *ctx;
    int pending;        /* workers that still owe a completion */
    uint64_t generation;
    int stop;
} Pool;

static Pool g_pool;

#ifdef _WIN32
static DWORD WINAPI pool_worker(LPVOID arg)
#else
static void *pool_worker(void *arg)
#endif
{
    int me = (int)(intptr_t)arg;
    uint64_t seen = 0;
    for (;;) {
#ifdef _WIN32
        EnterCriticalSection(&g_pool.lock);
        while (!g_pool.stop && g_pool.generation == seen)
            SleepConditionVariableCS(&g_pool.work_cv, &g_pool.lock, INFINITE);
#else
        pthread_mutex_lock(&g_pool.lock);
        while (!g_pool.stop && g_pool.generation == seen)
            pthread_cond_wait(&g_pool.work_cv, &g_pool.lock);
#endif
        if (g_pool.stop) {
#ifdef _WIN32
            LeaveCriticalSection(&g_pool.lock);
            return 0;
#else
            pthread_mutex_unlock(&g_pool.lock);
            return NULL;
#endif
        }
        seen = g_pool.generation;
        work_fn fn = g_pool.fn;
        void *ctx = g_pool.ctx;
        int nw = g_pool.nworkers;
#ifdef _WIN32
        LeaveCriticalSection(&g_pool.lock);
#else
        pthread_mutex_unlock(&g_pool.lock);
#endif
        fn(ctx, me, nw);
#ifdef _WIN32
        EnterCriticalSection(&g_pool.lock);
        if (--g_pool.pending == 0) WakeConditionVariable(&g_pool.done_cv);
        LeaveCriticalSection(&g_pool.lock);
#else
        pthread_mutex_lock(&g_pool.lock);
        if (--g_pool.pending == 0) pthread_cond_signal(&g_pool.done_cv);
        pthread_mutex_unlock(&g_pool.lock);
#endif
    }
}

static void pool_init(int nworkers)
{
    if (nworkers < 1) nworkers = 1;
    g_pool.nworkers = nworkers;
#ifdef _WIN32
    InitializeCriticalSection(&g_pool.lock);
    InitializeConditionVariable(&g_pool.work_cv);
    InitializeConditionVariable(&g_pool.done_cv);
    g_pool.threads = (HANDLE *)malloc(sizeof(HANDLE) * nworkers);
    for (int i = 0; i < nworkers; i++)
        g_pool.threads[i] = CreateThread(NULL, 0, pool_worker,
                                         (LPVOID)(intptr_t)i, 0, NULL);
#else
    pthread_mutex_init(&g_pool.lock, NULL);
    pthread_cond_init(&g_pool.work_cv, NULL);
    pthread_cond_init(&g_pool.done_cv, NULL);
    g_pool.threads = (pthread_t *)malloc(sizeof(pthread_t) * nworkers);
    for (int i = 0; i < nworkers; i++)
        pthread_create(&g_pool.threads[i], NULL, pool_worker,
                       (void *)(intptr_t)i);
#endif
}

/* Run fn(ctx, worker, nworkers) on every worker and wait. */
static void pool_run(work_fn fn, void *ctx)
{
    if (g_pool.nworkers <= 1) { fn(ctx, 0, 1); return; }
#ifdef _WIN32
    EnterCriticalSection(&g_pool.lock);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.pending = g_pool.nworkers;
    g_pool.generation++;
    WakeAllConditionVariable(&g_pool.work_cv);
    while (g_pool.pending > 0)
        SleepConditionVariableCS(&g_pool.done_cv, &g_pool.lock, INFINITE);
    LeaveCriticalSection(&g_pool.lock);
#else
    pthread_mutex_lock(&g_pool.lock);
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.pending = g_pool.nworkers;
    g_pool.generation++;
    pthread_cond_broadcast(&g_pool.work_cv);
    while (g_pool.pending > 0)
        pthread_cond_wait(&g_pool.done_cv, &g_pool.lock);
    pthread_mutex_unlock(&g_pool.lock);
#endif
}

static void pool_shutdown(void)
{
#ifdef _WIN32
    EnterCriticalSection(&g_pool.lock);
    g_pool.stop = 1;
    WakeAllConditionVariable(&g_pool.work_cv);
    LeaveCriticalSection(&g_pool.lock);
    for (int i = 0; i < g_pool.nworkers; i++) {
        WaitForSingleObject(g_pool.threads[i], INFINITE);
        CloseHandle(g_pool.threads[i]);
    }
#else
    pthread_mutex_lock(&g_pool.lock);
    g_pool.stop = 1;
    pthread_cond_broadcast(&g_pool.work_cv);
    pthread_mutex_unlock(&g_pool.lock);
    for (int i = 0; i < g_pool.nworkers; i++)
        pthread_join(g_pool.threads[i], NULL);
#endif
}

/* ================= small math ================= */

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;

static void rng_seed(uint64_t s) { g_rng = s ? s : 0x9E3779B97F4A7C15ull; }

static uint64_t rng_next(void)
{
    uint64_t x = g_rng;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_rng = x;
    return x * 0x2545F4914F6CDD1Dull;
}

/* Standard normal by Box-Muller; spare cached per call chain
 * (single-threaded use: the sampler only). */
static double randn(void)
{
    static int have_spare = 0;
    static double spare = 0.0;
    if (have_spare) { have_spare = 0; return spare; }
    double u1, u2;
    do { u1 = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0); }
    while (u1 <= 0.0);
    u2 = (double)(rng_next() >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double t = 2.0 * 3.14159265358979323846 * u2;
    spare = r * sin(t);
    have_spare = 1;
    return r * cos(t);
}

static float bf16_to_float(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float f16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float silu_f(float x) { return x / (1.0f + expf(-x)); }

/* y = x @ W^T, W row-major [rows][cols], x [n][cols], y [n][rows].
 * Rows of y are partitioned across the pool. */
typedef struct {
    const float *x;
    const float *w;
    float *y;
    int n, rows, cols;
} MatmulCtx;

static void matmul_work(void *vctx, int worker, int nw)
{
    MatmulCtx *c = (MatmulCtx *)vctx;
    int r0 = (int)((int64_t)c->rows * worker / nw);
    int r1 = (int)((int64_t)c->rows * (worker + 1) / nw);
    for (int r = r0; r < r1; r++) {
        const float *wrow = c->w + (size_t)r * c->cols;
        for (int i = 0; i < c->n; i++) {
            const float *xrow = c->x + (size_t)i * c->cols;
            double acc = 0.0;
            for (int k = 0; k < c->cols; k++) acc += (double)xrow[k] * wrow[k];
            c->y[(size_t)i * c->rows + r] = (float)acc;
        }
    }
}

static void matmul(const float *x, const float *w, float *y,
                   int n, int rows, int cols)
{
    MatmulCtx c;
    c.x = x; c.w = w; c.y = y; c.n = n; c.rows = rows; c.cols = cols;
    pool_run(matmul_work, &c);
}

/* RMSNorm over rows of x in place-ish (out may alias x). */
static void rmsnorm(float *out, const float *x, const float *w,
                    int n, int dim, float eps)
{
    for (int i = 0; i < n; i++) {
        const float *row = x + (size_t)i * dim;
        float *orow = out + (size_t)i * dim;
        double ss = 0.0;
        for (int k = 0; k < dim; k++) ss += (double)row[k] * row[k];
        double inv = 1.0 / sqrt(ss / dim + eps);
        for (int k = 0; k < dim; k++) orow[k] = (float)(row[k] * inv) * w[k];
    }
}

/* ================= weight sources =================
 *
 * Two backends behind one range-read API:
 *   raw:    the original safetensors shards (bf16/f16/f32)
 *   packed: a NanoQuatTwo .nq2 stream + .nq2book pair
 *
 * The packed format (v1/v2) carries no tensor names or shapes,
 * so the packed backend rebuilds its directory from the model
 * directory's safetensors headers, matching tensors to stream
 * segments in the same order the quantizer packed them. That
 * gap is recorded for a future format v3.
 */

enum { DT_BF16 = 0, DT_F16 = 1, DT_F32 = 2 };

typedef struct {
    char name[256];
    uint64_t count;
    int rows, cols;
    int shard;              /* raw: shard index */
    uint64_t byte_off;      /* raw: absolute offset of the data */
    int dtype;              /* raw */
    uint64_t block0;        /* packed: first stream block */
    uint64_t nblocks;       /* packed: blocks this tensor owns */
} TEnt;

#define MAX_TENSORS 1024
static TEnt g_dir[MAX_TENSORS];
static int g_ndir = 0;

static TEnt *dir_find(const char *name)
{
    for (int i = 0; i < g_ndir; i++)
        if (strcmp(g_dir[i].name, name) == 0) return &g_dir[i];
    return NULL;
}

/* ---- raw safetensors ---- */

#define MAX_SHARDS 8
static FILE *g_shards[MAX_SHARDS];
static int g_nshards = 0;

static int js_string_end(const char *s, long n, long pos)
{
    if (pos >= n || s[pos] != '"') return -1;
    pos++;
    while (pos < n) {
        if (s[pos] == '\\') { pos += 2; continue; }
        if (s[pos] == '"') return (int)(pos + 1);
        pos++;
    }
    return -1;
}

static int shard_open_raw(const char *path, int shard_idx)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    /* File size, for the data_offsets span check below. */
    uint64_t fsize;
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    { long long s = _ftelli64(f); if (s < 0) { fclose(f); return -1; } fsize = (uint64_t)s; }
    if (_fseeki64(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
#else
    if (fseeko(f, 0, SEEK_END) != 0) { fclose(f); return -1; }
    { off_t s = ftello(f); if (s < 0) { fclose(f); return -1; } fsize = (uint64_t)s; }
    if (fseeko(f, 0, SEEK_SET) != 0) { fclose(f); return -1; }
#endif
    uint8_t lenbuf[8];
    if (fread(lenbuf, 1, 8, f) != 8) { fclose(f); return -1; }
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)lenbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); return -1; }
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) { fclose(f); return -1; }
    if (fread(hdr, 1, hlen, f) != hlen) { free(hdr); fclose(f); return -1; }
    hdr[hlen] = '\0';
    uint64_t data_start = 8 + hlen;
    g_shards[shard_idx] = f;

    long n = (long)hlen, pos = 0;
    while (pos < n) {
        while (pos < n && hdr[pos] != '"') pos++;
        if (pos >= n) break;
        int ke = js_string_end(hdr, n, pos);
        if (ke < 0) break;
        long klen = ke - pos - 2;
        char key[256];
        if (klen > 250) klen = 250;
        memcpy(key, hdr + pos + 1, klen);
        key[klen] = '\0';
        long p = ke;
        while (p < n && (hdr[p] == ' ' || hdr[p] == ':' || hdr[p] == '\t')) p++;
        if (p >= n || hdr[p] != '{') { pos = ke; continue; }
        const char *obj = hdr + p;
        long depth = 0, q = p;
        int instr = 0;
        while (q < n) {
            char c = hdr[q];
            if (instr) {
                if (c == '\\') { q += 2; continue; }
                if (c == '"') instr = 0;
            } else if (c == '"') instr = 1;
            else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') { depth--; if (depth == 0) break; }
            q++;
        }
        long olen = q - p + 1;
        pos = q + 1;
        if (strcmp(key, "__metadata__") == 0) continue;
        if (g_ndir >= MAX_TENSORS) { free(hdr); return -1; }
        TEnt *t = &g_dir[g_ndir];
        memset(t, 0, sizeof(*t));
        memcpy(t->name, key, strlen(key) + 1);
        t->shard = shard_idx;
        const char *dt = strstr(obj, "\"dtype\"");
        if (dt && (long)(dt - obj) < olen) {
            const char *colon = strchr(dt + 7, ':');
            const char *q1 = colon ? strchr(colon, '"') : NULL;
            const char *q2 = q1 ? strchr(q1 + 1, '"') : NULL;
            char dts[8] = {0};
            if (q1 && q2 && q2 - q1 - 1 >= 1 && q2 - q1 - 1 <= 7)
                memcpy(dts, q1 + 1, q2 - q1 - 1);
            if (strcmp(dts, "BF16") == 0) t->dtype = DT_BF16;
            else if (strcmp(dts, "F16") == 0) t->dtype = DT_F16;
            else if (strcmp(dts, "F32") == 0) t->dtype = DT_F32;
            else {
                fprintf(stderr, "warning: tensor %s dtype '%s' "
                        "unrecognized, assuming F32\n", key, dts);
                t->dtype = DT_F32;
            }
        }
        const char *sh = strstr(obj, "\"shape\"");
        if (sh && (long)(sh - obj) < olen) {
            const char *lb = strchr(sh, '[');
            const char *rb = lb ? strchr(lb, ']') : NULL;
            if (lb && rb) {
                char buf[128];
                long bl = rb - lb - 1;
                if (bl > 120) bl = 120;
                memcpy(buf, lb + 1, bl);
                buf[bl] = '\0';
                uint64_t dims[4];
                int nd = 0;
                char *tok = strtok(buf, ", ");
                while (tok && nd < 4) {
                    dims[nd++] = strtoull(tok, NULL, 10);
                    tok = strtok(NULL, ", ");
                }
                if (nd == 1) { t->rows = (int)dims[0]; t->cols = 1; }
                else if (nd >= 2) { t->rows = (int)dims[0]; t->cols = (int)dims[1]; }
                uint64_t cnt = 1;
                for (int i = 0; i < nd; i++) cnt *= dims[i];
                t->count = cnt;
            }
        }
        const char *df = strstr(obj, "\"data_offsets\"");
        if (df && (long)(df - obj) < olen) {
            const char *lb = strchr(df, '[');
            if (lb) {
                uint64_t begin = strtoull(lb + 1, NULL, 10);
                const char *comma = strchr(lb, ',');
                uint64_t end = comma ? strtoull(comma + 1, NULL, 10) : 0;
                uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
                /* Span validation (Archie P0-4): only begin was
                 * read before, unchecked. The span must be
                 * ordered, lie inside the shard file, and (when
                 * both shape and dtype were parsed) match the
                 * shape-derived element count exactly. */
                if (end < begin || end > fsize
                    || data_start + end > fsize
                    || (dt && (long)(dt - obj) < olen && t->count > 0
                        && end - begin != t->count * esz)) {
                    fprintf(stderr, "reject: tensor %s has "
                            "data_offsets [%llu, %llu) inconsistent "
                            "with the shard or its shape\n", key,
                            (unsigned long long)begin,
                            (unsigned long long)end);
                    free(hdr);
                    return -1;
                }
                t->byte_off = data_start + begin;
            }
        }
        g_ndir++;
    }
    free(hdr);
    return 0;
}

static int seek_to(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* Read elements [elem0, elem0+nelems) of a raw tensor as fp32. */
static int raw_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                          float *out)
{
    int esz = t->dtype == DT_F32 ? 4 : 2;
    uint8_t *raw = (uint8_t *)malloc(nelems * (uint64_t)esz);
    if (!raw) return -1;
    FILE *f = g_shards[t->shard];
    if (seek_to(f, t->byte_off + elem0 * (uint64_t)esz) != 0
            || fread(raw, esz, nelems, f) != nelems) {
        free(raw);
        return -1;
    }
    if (t->dtype == DT_BF16) {
        const uint16_t *h = (const uint16_t *)raw;
        for (uint64_t i = 0; i < nelems; i++) out[i] = bf16_to_float(h[i]);
    } else if (t->dtype == DT_F16) {
        const uint16_t *h = (const uint16_t *)raw;
        for (uint64_t i = 0; i < nelems; i++) out[i] = f16_to_float(h[i]);
    } else {
        memcpy(out, raw, nelems * 4);
    }
    free(raw);
    return 0;
}

/* ---- packed .nq2 ---- */

static FILE *g_pk_stream = NULL;
static float *g_pk_book = NULL;         /* all books, fp32 */
static uint32_t g_pk_block = 0, g_pk_entries = 0, g_pk_version = 0;
static uint64_t g_pk_header = 0;        /* stream header bytes */
static uint64_t g_pk_nblocks = 0;

static uint32_t pk_rd_u32(FILE *f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4) { fprintf(stderr, "short read\n"); exit(1); }
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8)
         | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

static uint64_t pk_rd_u64(FILE *f)
{
    uint64_t lo = pk_rd_u32(f), hi = pk_rd_u32(f);
    return lo | (hi << 32);
}

static float pk_half(const uint8_t *p)
{
    uint16_t h = (uint16_t)(p[0] | (p[1] << 8));
    return f16_to_float(h);
}

/* Expand packed elements [elem0, elem0+nelems) of one tensor. */
static int packed_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                             float *out)
{
    uint32_t block = g_pk_block;
    uint64_t b0 = t->block0 + elem0 / block;
    uint64_t b1 = t->block0 + (elem0 + nelems - 1) / block;
    uint64_t nb = b1 - b0 + 1;
    uint8_t *idx = (uint8_t *)malloc(nb);
    uint8_t *fr = (uint8_t *)malloc(nb * 4);
    float *tmp = (float *)malloc(block * sizeof(float));
    if (!idx || !fr || !tmp) { free(idx); free(fr); free(tmp); return -1; }
    if (seek_to(g_pk_stream, g_pk_header + b0) != 0
            || fread(idx, 1, nb, g_pk_stream) != nb) goto fail;
    if (seek_to(g_pk_stream, g_pk_header + g_pk_nblocks + 2 * b0) != 0
            || fread(fr, 2, nb, g_pk_stream) != nb) goto fail;
    if (seek_to(g_pk_stream, g_pk_header + 3 * g_pk_nblocks + 2 * b0) != 0
            || fread(fr + nb * 2, 2, nb, g_pk_stream) != nb) goto fail;
    {
        const float *book = g_pk_book;
        if (g_pk_version == 2) {
            /* find the segment: segments are tensors in dir order */
            book = g_pk_book + (size_t)(t - g_dir) * g_pk_entries * block;
        }
        for (uint64_t b = 0; b < nb; b++) {
            if (idx[b] >= g_pk_entries) goto fail;
            /* Book lookup optimization (streamtest variant 1,
             * measured +21 percent in the harness): pull the
             * prototype row two blocks ahead toward cache while
             * the current row streams out, so the lookup never
             * waits on memory. */
            if (b + 2 < nb) {
#if defined(__GNUC__) || defined(__clang__)
                __builtin_prefetch(book + (size_t)idx[b + 2] * block,
                                   0, 1);
#endif
            }
            float mn = pk_half(fr + b * 2);
            float sc = pk_half(fr + nb * 2 + b * 2);
            const float *proto = book + (size_t)idx[b] * block;
            uint64_t g0 = (b0 + b) * block - t->block0 * block;
            uint64_t lo = elem0 > g0 ? elem0 : g0;
            uint64_t hi = elem0 + nelems < g0 + block ? elem0 + nelems
                                                      : g0 + block;
            if (lo == g0 && hi == g0 + block) {
                /* Block-aligned and fully wanted (the streaming
                 * hot path: whole tensors for the pipe): expand
                 * straight into the destination, no bounce copy. */
                const float *restrict src = proto;
                float *restrict dst = out + (g0 - elem0);
                for (uint32_t i = 0; i < block; i++)
                    dst[i] = mn + sc * src[i];
            } else {
                for (uint32_t i = 0; i < block; i++)
                    tmp[i] = mn + sc * proto[i];
                for (uint64_t e = lo; e < hi; e++)
                    out[e - elem0] = tmp[e - g0];
            }
        }
    }
    free(idx); free(fr); free(tmp);
    return 0;
fail:
    free(idx); free(fr); free(tmp);
    return -1;
}

static int g_use_packed = 0;

static int src_read_range(const TEnt *t, uint64_t elem0, uint64_t nelems,
                          float *out)
{
    if (!g_use_packed) return raw_read_range(t, elem0, nelems, out);
    /* The stream holds only whole blocks: elements past the last
     * packed block (tensors smaller than one block, and block
     * tails) were never packed. Serve the packed portion from
     * the stream and the remainder from the raw shards, which
     * are always open. Without this, a small tensor's read would
     * silently return the next tensor's first block. */
    uint64_t packed_elems = t->nblocks * g_pk_block;
    if (elem0 >= packed_elems)
        return raw_read_range(t, elem0, nelems, out);
    uint64_t in = nelems;
    if (elem0 + in > packed_elems) in = packed_elems - elem0;
    if (packed_read_range(t, elem0, in, out) != 0) return -1;
    if (in < nelems)
        return raw_read_range(t, elem0 + in, nelems - in, out + in);
    return 0;
}

/* Open the packed pair and assign stream segments to directory
 * tensors in directory order (the quantizer's packing order). */
static int packed_open(const char *stream_path, const char *book_path)
{
    g_pk_stream = fopen(stream_path, "rb");
    if (!g_pk_stream) return -1;
    /* The packed reader makes three section reads per range;
     * a large stdio buffer keeps those streaming instead of
     * syscall-bound. */
    setvbuf(g_pk_stream, NULL, _IOFBF, 1 << 18);
    char magic[4];
    if (fread(magic, 1, 4, g_pk_stream) != 4
            || memcmp(magic, "NQ2\0", 4) != 0) return -1;
    g_pk_version = pk_rd_u32(g_pk_stream);
    g_pk_block = pk_rd_u32(g_pk_stream);
    g_pk_entries = pk_rd_u32(g_pk_stream);
    (void)pk_rd_u64(g_pk_stream);           /* total weights */
    g_pk_nblocks = pk_rd_u64(g_pk_stream);
    /* T2 hardening (CODE-REVIEW-2026-11-24): validate the pack
     * geometry before any division or allocation trusts it. */
    if (g_pk_block == 0 || g_pk_block > 4096 || g_pk_entries == 0
        || g_pk_entries > 256) {
        fprintf(stderr, "packed stream header has unsupported "
                "geometry (block %u, entries %u)\n",
                g_pk_block, g_pk_entries);
        return -1;
    }
    uint32_t n_books = 1;
    uint64_t *seg = NULL;
    if (g_pk_version == 2) {
        n_books = pk_rd_u32(g_pk_stream);
        seg = (uint64_t *)malloc(sizeof(uint64_t) * n_books);
        if (!seg) return -1;
        for (uint32_t i = 0; i < n_books; i++) seg[i] = pk_rd_u64(g_pk_stream);
    } else if (g_pk_version != 1) {
        return -1;
    }
    {
        long hb = ftell(g_pk_stream);
        g_pk_header = hb > 0 ? (uint64_t)hb : 0;
    }
    FILE *bf = fopen(book_path, "rb");
    if (!bf) return -1;
    if (fread(magic, 1, 4, bf) != 4 || memcmp(magic, "NQ2B", 4) != 0)
        return -1;
    uint64_t total_book_floats;
    if (g_pk_version == 2) {
        uint32_t bver = pk_rd_u32(bf);
        uint32_t b_block = pk_rd_u32(bf);
        uint32_t b_entries = pk_rd_u32(bf);
        uint32_t b_books = pk_rd_u32(bf);
        if (bver != 2 || b_block != g_pk_block
                || b_entries != g_pk_entries || b_books != n_books) {
            fprintf(stderr, "stream and book disagree\n");
            return -1;
        }
        total_book_floats = (uint64_t)n_books * g_pk_entries * g_pk_block;
    } else {
        uint32_t b_block = pk_rd_u32(bf);
        uint32_t b_entries = pk_rd_u32(bf);
        if (b_block != g_pk_block || b_entries != g_pk_entries) {
            fprintf(stderr, "stream and book disagree\n");
            return -1;
        }
        total_book_floats = (uint64_t)g_pk_entries * g_pk_block;
    }
    g_pk_book = (float *)malloc(total_book_floats * sizeof(float));
    if (!g_pk_book) return -1;
    if (fread(g_pk_book, sizeof(float), total_book_floats, bf)
            != total_book_floats) return -1;
    fclose(bf);

    uint64_t cursor = 0;
    if (g_pk_version == 2) {
        if ((int)n_books != g_ndir) {
            fprintf(stderr, "packed model has %u books but the "
                    "directory has %d tensors\n", n_books, g_ndir);
            return -1;
        }
        for (int i = 0; i < g_ndir; i++) {
            g_dir[i].block0 = cursor;
            g_dir[i].nblocks = seg[i];
            cursor += seg[i];
        }
    } else {
        for (int i = 0; i < g_ndir; i++) {
            g_dir[i].block0 = cursor;
            g_dir[i].nblocks = g_dir[i].count / g_pk_block;
            cursor += g_dir[i].nblocks;
        }
    }
    free(seg);
    return 0;
}

/* ================= prefetch pipeline =================
 *
 * JD's two-step prefetch, at tensor granularity: the forward
 * pass consumes weight tensors in a fixed ordinal order. One
 * slot computes while the prefetch thread materializes the
 * next ordinal into the other slot; the ordinal after that is
 * still on disc. Two tensors open at a time, ever.
 */

enum { SLOT_Q = 0, SLOT_K, SLOT_V, SLOT_O, SLOT_GATE, SLOT_UP,
       SLOT_DOWN, SLOTS_PER_LAYER };

/* Wall clock for pipeline telemetry only (never touches the math). */
static double pipe_now(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static int have_freq = 0;
    LARGE_INTEGER c;
    if (!have_freq) { QueryPerformanceFrequency(&freq); have_freq = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#endif
}

typedef struct {
#ifdef _WIN32
    CRITICAL_SECTION lock;
    CONDITION_VARIABLE cv;
    HANDLE thread;
#else
    pthread_mutex_t lock;
    pthread_cond_t cv;
    pthread_t thread;
#endif
    float *buf[2];
    size_t cap_elems;
    int have[2];
    int want;
    int busy;
    int stop;
    int max_ord;
    /* Telemetry (JD's lookahead, measured): when each ordinal
     * became ready, how long the consumer stalled on misses, how
     * long ready tensors waited on hits (slack), and the load
     * path's own cost. Loads are serialized by the busy flag, so
     * the last-load fields are safe. */
    double *ready_at;
    double load_ms_last;
    uint64_t load_bytes_last;
    double stat_stall_s, stat_slack_s, stat_load_ms;
    uint64_t stat_bytes;
    long stat_hits, stat_miss;
} Pipe;

static Pipe g_pipe;

static const TEnt *ord_ent(int ord)
{
    int layer = ord / SLOTS_PER_LAYER;
    int slot = ord % SLOTS_PER_LAYER;
    static const char *names[SLOTS_PER_LAYER] = {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight",
        "mlp.down_proj.weight"
    };
    char full[256];
    snprintf(full, sizeof(full), "encoder.layers.%d.%s", layer, names[slot]);
    return dir_find(full);
}

static void pipe_load(int slot, int ord)
{
    const TEnt *t = ord_ent(ord);
    if (!t || t->count > g_pipe.cap_elems) {
        fprintf(stderr, "pipe: tensor for ordinal %d missing/oversize\n", ord);
        exit(1);
    }
    double t0 = pipe_now();
    if (src_read_range(t, 0, t->count, g_pipe.buf[slot]) != 0) {
        fprintf(stderr, "pipe: read failed for %s\n", t->name);
        exit(1);
    }
    g_pipe.load_ms_last = (pipe_now() - t0) * 1000.0;
    g_pipe.load_bytes_last = t->count * sizeof(float);
}

/* Called under the lock wherever an ordinal is marked ready. */
static void pipe_mark_ready(int ord)
{
    g_pipe.ready_at[ord] = pipe_now();
    g_pipe.stat_load_ms += g_pipe.load_ms_last;
    g_pipe.stat_bytes += g_pipe.load_bytes_last;
}

#ifdef _WIN32
static DWORD WINAPI pipe_worker(LPVOID unused)
#else
static void *pipe_worker(void *unused)
#endif
{
    (void)unused;
    for (;;) {
#ifdef _WIN32
        EnterCriticalSection(&g_pipe.lock);
        while (!g_pipe.stop
               && !(g_pipe.want >= 0 && g_pipe.want <= g_pipe.max_ord
                    && g_pipe.have[g_pipe.want & 1] != g_pipe.want
                    && !g_pipe.busy))
            SleepConditionVariableCS(&g_pipe.cv, &g_pipe.lock, INFINITE);
        if (g_pipe.stop) { LeaveCriticalSection(&g_pipe.lock); return 0; }
        int ord = g_pipe.want;
        g_pipe.busy = 1;
        LeaveCriticalSection(&g_pipe.lock);
        pipe_load(ord & 1, ord);
        EnterCriticalSection(&g_pipe.lock);
        g_pipe.have[ord & 1] = ord;
        pipe_mark_ready(ord);
        g_pipe.busy = 0;
        WakeAllConditionVariable(&g_pipe.cv);
        LeaveCriticalSection(&g_pipe.lock);
#else
        pthread_mutex_lock(&g_pipe.lock);
        while (!g_pipe.stop
               && !(g_pipe.want >= 0 && g_pipe.want <= g_pipe.max_ord
                    && g_pipe.have[g_pipe.want & 1] != g_pipe.want
                    && !g_pipe.busy))
            pthread_cond_wait(&g_pipe.cv, &g_pipe.lock);
        if (g_pipe.stop) { pthread_mutex_unlock(&g_pipe.lock); return NULL; }
        int ord = g_pipe.want;
        g_pipe.busy = 1;
        pthread_mutex_unlock(&g_pipe.lock);
        pipe_load(ord & 1, ord);
        pthread_mutex_lock(&g_pipe.lock);
        g_pipe.have[ord & 1] = ord;
        pipe_mark_ready(ord);
        g_pipe.busy = 0;
        pthread_cond_broadcast(&g_pipe.cv);
        pthread_mutex_unlock(&g_pipe.lock);
#endif
    }
}

static void pipe_init(int layers)
{
    size_t max_elems = 0;
    int max_ord = layers * SLOTS_PER_LAYER - 1;
    for (int o = 0; o <= max_ord; o++) {
        const TEnt *t = ord_ent(o);
        if (t && t->count > max_elems) max_elems = t->count;
    }
    g_pipe.cap_elems = max_elems;
    g_pipe.buf[0] = (float *)malloc(max_elems * sizeof(float));
    g_pipe.buf[1] = (float *)malloc(max_elems * sizeof(float));
    if (!g_pipe.buf[0] || !g_pipe.buf[1]) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    g_pipe.have[0] = g_pipe.have[1] = -1;
    g_pipe.want = -1;
    g_pipe.max_ord = max_ord;
    g_pipe.ready_at = (double *)calloc((size_t)max_ord + 1, sizeof(double));
    if (!g_pipe.ready_at) {
        fprintf(stderr, "pipe: out of memory\n");
        exit(1);
    }
    g_pipe.stat_stall_s = g_pipe.stat_slack_s = g_pipe.stat_load_ms = 0.0;
    g_pipe.stat_bytes = 0;
    g_pipe.stat_hits = g_pipe.stat_miss = 0;
#ifdef _WIN32
    InitializeCriticalSection(&g_pipe.lock);
    InitializeConditionVariable(&g_pipe.cv);
    g_pipe.thread = CreateThread(NULL, 0, pipe_worker, NULL, 0, NULL);
#else
    pthread_mutex_init(&g_pipe.lock, NULL);
    pthread_cond_init(&g_pipe.cv, NULL);
    pthread_create(&g_pipe.thread, NULL, pipe_worker, NULL);
#endif
}

/* Get the materialized weights for ordinal ord, and set the
 * prefetch of ord+1 in motion. The returned buffer is valid
 * until the request for ord+2. */
static const float *pipe_get(int ord)
{
    /* If the prefetch thread has not produced ord, claim the
     * load (busy) before dropping the lock, so the prefetcher
     * cannot start the same load into the same slot. */
    double t_entry = pipe_now();
    int was_ready = 0;
#ifdef _WIN32
    EnterCriticalSection(&g_pipe.lock);
    was_ready = (g_pipe.have[ord & 1] == ord);
    if (was_ready) {
        g_pipe.stat_hits++;
        g_pipe.stat_slack_s += pipe_now() - g_pipe.ready_at[ord];
    }
    while (g_pipe.have[ord & 1] != ord && g_pipe.busy)
        SleepConditionVariableCS(&g_pipe.cv, &g_pipe.lock, INFINITE);
    if (g_pipe.have[ord & 1] != ord) {
        g_pipe.busy = 1;
        LeaveCriticalSection(&g_pipe.lock);
        pipe_load(ord & 1, ord);
        EnterCriticalSection(&g_pipe.lock);
        g_pipe.have[ord & 1] = ord;
        pipe_mark_ready(ord);
        g_pipe.busy = 0;
        WakeAllConditionVariable(&g_pipe.cv);
    }
    if (ord + 1 <= g_pipe.max_ord) {
        g_pipe.want = ord + 1;
        WakeAllConditionVariable(&g_pipe.cv);
    }
    if (!was_ready) {
        g_pipe.stat_miss++;
        g_pipe.stat_stall_s += pipe_now() - t_entry;
    }
    LeaveCriticalSection(&g_pipe.lock);
#else
    pthread_mutex_lock(&g_pipe.lock);
    was_ready = (g_pipe.have[ord & 1] == ord);
    if (was_ready) {
        g_pipe.stat_hits++;
        g_pipe.stat_slack_s += pipe_now() - g_pipe.ready_at[ord];
    }
    while (g_pipe.have[ord & 1] != ord && g_pipe.busy)
        pthread_cond_wait(&g_pipe.cv, &g_pipe.lock);
    if (g_pipe.have[ord & 1] != ord) {
        g_pipe.busy = 1;
        pthread_mutex_unlock(&g_pipe.lock);
        pipe_load(ord & 1, ord);
        pthread_mutex_lock(&g_pipe.lock);
        g_pipe.have[ord & 1] = ord;
        pipe_mark_ready(ord);
        g_pipe.busy = 0;
        pthread_cond_broadcast(&g_pipe.cv);
    }
    if (ord + 1 <= g_pipe.max_ord) {
        g_pipe.want = ord + 1;
        pthread_cond_broadcast(&g_pipe.cv);
    }
    if (!was_ready) {
        g_pipe.stat_miss++;
        g_pipe.stat_stall_s += pipe_now() - t_entry;
    }
    pthread_mutex_unlock(&g_pipe.lock);
#endif
    return g_pipe.buf[ord & 1];
}

static void pipe_report(void)
{
    long total = g_pipe.stat_hits + g_pipe.stat_miss;
    if (total == 0) return;
    printf("prefetch telemetry: %ld requests, %ld hits (%.1f%%), "
           "%ld misses\n",
           total, g_pipe.stat_hits,
           100.0 * (double)g_pipe.stat_hits / (double)total,
           g_pipe.stat_miss);
    printf("  consumer stall on misses: %.1f ms total; "
           "mean ready slack on hits: %.1f ms\n",
           g_pipe.stat_stall_s * 1000.0,
           g_pipe.stat_hits
               ? g_pipe.stat_slack_s * 1000.0 / g_pipe.stat_hits : 0.0);
    if (g_pipe.stat_load_ms > 0)
        printf("  load path (disc + reconstruct): %.1f ms for "
               "%.2f GB expanded, %.0f MB/sec\n",
               g_pipe.stat_load_ms,
               (double)g_pipe.stat_bytes / 1e9,
               (double)g_pipe.stat_bytes
                   / (g_pipe.stat_load_ms / 1000.0) / 1e6);
}

static void pipe_shutdown(void)
{
#ifdef _WIN32
    EnterCriticalSection(&g_pipe.lock);
    g_pipe.stop = 1;
    WakeAllConditionVariable(&g_pipe.cv);
    LeaveCriticalSection(&g_pipe.lock);
    WaitForSingleObject(g_pipe.thread, INFINITE);
    CloseHandle(g_pipe.thread);
#else
    pthread_mutex_lock(&g_pipe.lock);
    g_pipe.stop = 1;
    pthread_cond_broadcast(&g_pipe.cv);
    pthread_mutex_unlock(&g_pipe.lock);
    pthread_join(g_pipe.thread, NULL);
#endif
}

/* ================= model ================= */

static int M_H = 3072;          /* hidden size */
static int M_L = 26;            /* layers */
static int M_HEADS = 32;
static int M_KVHEADS = 8;
static int M_HD = 128;          /* head dim */
static int M_FFN = 9216;
static int M_VOCAB = 131072;
static int M_ED = 16;           /* continuous embedding dim */
static float M_EPS = 1e-5f;

static float *g_norm_in = NULL;     /* [L][H] */
static float *g_norm_post = NULL;   /* [L][H] */
static float *g_norm_final = NULL;  /* [H] */
static float *g_in_linear = NULL;   /* [H][ED] */
static float *g_sc_linear = NULL;   /* [H][ED] */
static float *g_gamma_linear = NULL;/* [H][64] */
static float *g_E = NULL;           /* [VOCAB][ED], rows normalized */
static double g_gamma0 = -3.0, g_gamma1 = 6.0;

static float *load_small(const char *name)
{
    TEnt *t = dir_find(name);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    float *buf = (float *)malloc(t->count * sizeof(float));
    if (!buf || src_read_range(t, 0, t->count, buf) != 0) {
        fprintf(stderr, "cannot load %s\n", name);
        exit(1);
    }
    return buf;
}

/* YaRN rotary tables, following the reference exactly:
 * factor = max_pos / original_max_pos (0.25 for this model),
 * correction ramp between beta_fast and beta_slow dims. */
static float *g_rope_cos = NULL, *g_rope_sin = NULL;  /* [maxpos][HD] */
static int g_rope_max = 0;

static void rope_init(int max_pos)
{
    const double theta = 1000000.0;
    const double orig_max = 16384.0;
    const double factor = 4096.0 / orig_max;
    const double beta_fast = 32.0, beta_slow = 1.0;
    int half = M_HD / 2;
    double inv_freq[64];
    double corr = 2.0 * log(theta);
    double low_d = M_HD * log(orig_max / (beta_fast * 2.0 * 3.141592653589793))
                   / corr;
    double high_d = M_HD * log(orig_max / (beta_slow * 2.0 * 3.141592653589793))
                    / corr;
    int low = (int)floor(low_d); if (low < 0) low = 0;
    int high = (int)ceil(high_d); if (high > M_HD - 1) high = M_HD - 1;
    for (int i = 0; i < half; i++) {
        double pos_freq = pow(theta, (double)(2 * i) / M_HD);
        double inv_extrap = 1.0 / pos_freq;
        double inv_interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low > 0.001 ? high - low : 0.001);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double ef = 1.0 - ramp;
        inv_freq[i] = inv_interp * (1.0 - ef) + inv_extrap * ef;
    }
    g_rope_max = max_pos;
    g_rope_cos = (float *)malloc((size_t)max_pos * M_HD * sizeof(float));
    g_rope_sin = (float *)malloc((size_t)max_pos * M_HD * sizeof(float));
    if (!g_rope_cos || !g_rope_sin) { fprintf(stderr, "oom rope\n"); exit(1); }
    for (int p = 0; p < max_pos; p++) {
        for (int i = 0; i < half; i++) {
            double ang = inv_freq[i] * p;
            float c = (float)cos(ang), s = (float)sin(ang);
            g_rope_cos[(size_t)p * M_HD + i] = c;
            g_rope_cos[(size_t)p * M_HD + half + i] = c;
            g_rope_sin[(size_t)p * M_HD + i] = s;
            g_rope_sin[(size_t)p * M_HD + half + i] = s;
        }
    }
}

/* KV cache: bf16 storage, fp32 math. K is stored post-rotary,
 * exactly as the reference caches it. */
typedef struct {
    uint16_t **k, **v;      /* [layer][cap * kv_dim] */
    int cap;
    int len;                /* committed positions */
} KVCache;

static KVCache g_cache;

static void cache_init(int cap)
{
    int kv_dim = M_KVHEADS * M_HD;
    g_cache.k = (uint16_t **)malloc(sizeof(uint16_t *) * M_L);
    g_cache.v = (uint16_t **)malloc(sizeof(uint16_t *) * M_L);
    g_cache.cap = cap;
    g_cache.len = 0;
    for (int l = 0; l < M_L; l++) {
        g_cache.k[l] = (uint16_t *)malloc((size_t)cap * kv_dim * 2);
        g_cache.v[l] = (uint16_t *)malloc((size_t)cap * kv_dim * 2);
        if (!g_cache.k[l] || !g_cache.v[l]) {
            fprintf(stderr, "oom cache\n");
            exit(1);
        }
    }
}

static uint16_t f32_to_bf16(float f)
{
    /* round to nearest even, matching the reference's cast;
     * plain truncation biased every cached key and value down */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    if ((bits & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; /* NaN -> bf16 quiet NaN; without this the rounding add carries into the sign bit and erases the NaN (Archie P0-5, guard from littlenemo/src/attn.c) */
    bits += 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

/* ================= encoder ================= */

typedef struct {
    float *xn;        /* [maxn][H] normed input */
    float *q;         /* [maxn][HEADS*HD] */
    float *kcur;      /* [maxn][KVHEADS*HD] */
    float *vcur;      /* [maxn][KVHEADS*HD] */
    float *att;       /* [maxn][HEADS*HD] */
    float *gate;      /* [maxn][FFN] */
    float *up;        /* [maxn][FFN] */
    float *tmp;       /* [maxn][H] */
    int maxn;
} WS;

static WS ws_alloc(int maxn)
{
    WS w;
    w.maxn = maxn;
    w.xn = (float *)malloc((size_t)maxn * M_H * sizeof(float));
    w.q = (float *)malloc((size_t)maxn * M_HEADS * M_HD * sizeof(float));
    w.kcur = (float *)malloc((size_t)maxn * M_KVHEADS * M_HD * sizeof(float));
    w.vcur = (float *)malloc((size_t)maxn * M_KVHEADS * M_HD * sizeof(float));
    w.att = (float *)malloc((size_t)maxn * M_HEADS * M_HD * sizeof(float));
    w.gate = (float *)malloc((size_t)maxn * M_FFN * sizeof(float));
    w.up = (float *)malloc((size_t)maxn * M_FFN * sizeof(float));
    w.tmp = (float *)malloc((size_t)maxn * M_H * sizeof(float));
    if (!w.xn || !w.q || !w.kcur || !w.vcur || !w.att || !w.gate
            || !w.up || !w.tmp) {
        fprintf(stderr, "oom workspace\n");
        exit(1);
    }
    return w;
}

enum { ATT_WRITE = 0, ATT_READ = 1, ATT_BLOCK_ONLY = 2 };

typedef struct {
    const float *q;         /* [n][HEADS*HD], post-rotary */
    const float *kcur;      /* [n][KV] current block keys */
    const float *vcur;
    const int *pos;         /* [n] query positions */
    float *att;             /* [n][HEADS*HD] out */
    int n, layer, mode, clen;
} AttCtx;

static void att_work(void *vctx, int worker, int nw)
{
    AttCtx *c = (AttCtx *)vctx;
    int kv_dim = M_KVHEADS * M_HD;
    int groups = M_HEADS / M_KVHEADS;
    int total_qh = c->n * M_HEADS;
    int a0 = (int)((int64_t)total_qh * worker / nw);
    int a1 = (int)((int64_t)total_qh * (worker + 1) / nw);
    int total_keys = (c->mode == ATT_BLOCK_ONLY) ? c->n : c->clen + c->n;
    const uint16_t *ck = c->layer >= 0 ? g_cache.k[c->layer] : NULL;
    const uint16_t *cv = c->layer >= 0 ? g_cache.v[c->layer] : NULL;
    float *scores = (float *)malloc(sizeof(float) * total_keys);
    if (!scores) exit(1);
    for (int a = a0; a < a1; a++) {
        int i = a / M_HEADS;
        int h = a % M_HEADS;
        int kh = h / groups;
        const float *qv = c->q + (size_t)i * M_HEADS * M_HD + h * M_HD;
        double mx = -1e30;
        for (int j = 0; j < total_keys; j++) {
            int allowed;
            if (c->mode == ATT_WRITE) {
                int kpos = j < c->clen ? j : c->pos[j - c->clen];
                allowed = kpos <= c->pos[i];
            } else {
                allowed = 1;
            }
            if (!allowed) { scores[j] = -1e30f; continue; }
            double dot = 0.0;
            if (j < c->clen && c->mode != ATT_BLOCK_ONLY) {
                const uint16_t *kv = ck + (size_t)j * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++)
                    dot += (double)qv[d] * bf16_to_float(kv[d]);
            } else {
                const float *kv = c->kcur
                    + (size_t)(j - (c->mode == ATT_BLOCK_ONLY ? 0 : c->clen))
                          * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++) dot += (double)qv[d] * kv[d];
            }
            double s = dot * (1.0 / sqrt((double)M_HD));
            scores[j] = (float)s;
            if (s > mx) mx = s;
        }
        double sum = 0.0;
        for (int j = 0; j < total_keys; j++) {
            if (scores[j] <= -1e29f) { scores[j] = 0.0f; continue; }
            scores[j] = (float)exp((double)scores[j] - mx);
            sum += scores[j];
        }
        float *ov = c->att + (size_t)i * M_HEADS * M_HD + h * M_HD;
        for (int d = 0; d < M_HD; d++) ov[d] = 0.0f;
        if (sum <= 0.0) continue;
        double inv = 1.0 / sum;
        for (int j = 0; j < total_keys; j++) {
            if (scores[j] == 0.0f) continue;
            double p = scores[j] * inv;
            if (j < c->clen && c->mode != ATT_BLOCK_ONLY) {
                const uint16_t *vv = cv + (size_t)j * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++)
                    ov[d] += (float)(p * bf16_to_float(vv[d]));
            } else {
                const float *vv = c->vcur
                    + (size_t)(j - (c->mode == ATT_BLOCK_ONLY ? 0 : c->clen))
                          * kv_dim + kh * M_HD;
                for (int d = 0; d < M_HD; d++) ov[d] += (float)(p * vv[d]);
            }
        }
    }
    free(scores);
}

/* hidden: [n][H] in/out (residual stream). pos: [n] positions. */
static void encoder_layer(float *hidden, const int *pos, int n, int layer,
                          int mode, WS *w)
{
    int qdim = M_HEADS * M_HD, kv_dim = M_KVHEADS * M_HD;
    int base = layer * SLOTS_PER_LAYER;
    rmsnorm(w->xn, hidden, g_norm_in + (size_t)layer * M_H, n, M_H, M_EPS);
    matmul(w->xn, pipe_get(base + SLOT_Q), w->q, n, qdim, M_H);
    matmul(w->xn, pipe_get(base + SLOT_K), w->kcur, n, kv_dim, M_H);
    matmul(w->xn, pipe_get(base + SLOT_V), w->vcur, n, kv_dim, M_H);
    /* rotary, in place on q and kcur */
    for (int i = 0; i < n; i++) {
        const float *cs = g_rope_cos + (size_t)pos[i] * M_HD;
        const float *sn = g_rope_sin + (size_t)pos[i] * M_HD;
        float *vecs[2] = { w->q + (size_t)i * qdim, NULL };
        vecs[1] = w->kcur + (size_t)i * kv_dim;
        int dims[2] = { qdim, kv_dim };
        for (int v = 0; v < 2; v++) {
            float *x = vecs[v];
            for (int off = 0; off < dims[v]; off += M_HD) {
                float *xh = x + off;
                float hold[64];
                for (int d = 0; d < M_HD / 2; d++) hold[d] = xh[d];
                for (int d = 0; d < M_HD / 2; d++) {
                    float x1 = hold[d], x2 = xh[M_HD / 2 + d];
                    xh[d] = x1 * cs[d] - x2 * sn[d];
                    xh[M_HD / 2 + d] = x2 * cs[M_HD / 2 + d]
                                       + x1 * sn[M_HD / 2 + d];
                }
            }
        }
        /* Llama-4 style query scaling (1.0 below 16384) */
        double sc = 1.0 + 0.1 * log(1.0 + floor(pos[i] / 16384.0));
        if (sc != 1.0) {
            float *x = w->q + (size_t)i * qdim;
            for (int d = 0; d < qdim; d++) x[d] = (float)(x[d] * sc);
        }
    }
    if (mode == ATT_WRITE) {
        int kv_dim2 = kv_dim;
        for (int i = 0; i < n; i++) {
            uint16_t *kd = g_cache.k[layer] + (size_t)(g_cache.len + i) * kv_dim2;
            uint16_t *vd = g_cache.v[layer] + (size_t)(g_cache.len + i) * kv_dim2;
            for (int d = 0; d < kv_dim2; d++) {
                kd[d] = f32_to_bf16(w->kcur[(size_t)i * kv_dim2 + d]);
                vd[d] = f32_to_bf16(w->vcur[(size_t)i * kv_dim2 + d]);
            }
        }
    }
    AttCtx ac;
    ac.q = w->q; ac.kcur = w->kcur; ac.vcur = w->vcur; ac.pos = pos;
    ac.att = w->att; ac.n = n; ac.layer = layer; ac.mode = mode;
    ac.clen = g_cache.len;
    pool_run(att_work, &ac);
    matmul(w->att, pipe_get(base + SLOT_O), w->tmp, n, M_H, qdim);
    for (int i = 0; i < n * M_H; i++) hidden[i] += w->tmp[i];

    rmsnorm(w->xn, hidden, g_norm_post + (size_t)layer * M_H, n, M_H, M_EPS);
    matmul(w->xn, pipe_get(base + SLOT_GATE), w->gate, n, M_FFN, M_H);
    matmul(w->xn, pipe_get(base + SLOT_UP), w->up, n, M_FFN, M_H);
    for (int i = 0; i < n * M_FFN; i++)
        w->gate[i] = silu_f(w->gate[i]) * w->up[i];
    matmul(w->gate, pipe_get(base + SLOT_DOWN), w->tmp, n, M_H, M_FFN);
    for (int i = 0; i < n * M_H; i++) hidden[i] += w->tmp[i];
}

/* Full backbone pass. ATT_WRITE commits KV and advances the
 * cache length after the pass. Returns hidden states after the
 * final norm in `hidden`. */
static void encoder_forward(float *hidden, const int *pos, int n, int mode,
                            WS *w)
{
    for (int l = 0; l < M_L; l++)
        encoder_layer(hidden, pos, n, l, mode, w);
    rmsnorm(w->tmp, hidden, g_norm_final, n, M_H, M_EPS);
    memcpy(hidden, w->tmp, (size_t)n * M_H * sizeof(float));
    if (mode == ATT_WRITE) g_cache.len += n;
}

/* ================= diffusion ================= */

static TEnt *g_head_ent = NULL;
static TEnt *g_embed_ent = NULL;

/* Diffusion-head logits for all B hidden rows at once, streaming
 * the head matrix in row tiles: each tile is read from disc once
 * and multiplied against every position. The full matrix is
 * never materialized, and the head crosses the disc-to-CPU path
 * once per pass instead of once per position. */
static void head_all(const float *h, int B, float *logits)
{
    const int TILE = 2048;
    float *tile = (float *)malloc((size_t)TILE * M_H * sizeof(float));
    float *tmp = (float *)malloc((size_t)B * TILE * sizeof(float));
    if (!tile || !tmp) { fprintf(stderr, "oom head tile\n"); exit(1); }
    for (int r0 = 0; r0 < M_VOCAB; r0 += TILE) {
        int rows = M_VOCAB - r0 < TILE ? M_VOCAB - r0 : TILE;
        if (src_read_range(g_head_ent, (uint64_t)r0 * M_H,
                           (uint64_t)rows * M_H, tile) != 0) {
            fprintf(stderr, "head read failed\n");
            exit(1);
        }
        matmul(h, tile, tmp, B, rows, M_H);
        for (int i = 0; i < B; i++)
            for (int j = 0; j < rows; j++)
                logits[(size_t)i * M_VOCAB + r0 + j]
                    = tmp[(size_t)i * rows + j];
    }
    free(tile);
    free(tmp);
}

/* softmax(logits) @ E for one position: the model's clean
 * embedding estimate. */
static void mean_from_logits(const float *logits, float *mean_out)
{
    double mx = -1e30;
    for (int v = 0; v < M_VOCAB; v++)
        if (logits[v] > mx) mx = logits[v];
    double sum = 0.0;
    /* Accumulate in double: summing 131,072 probability-weighted
     * rows into a float loses real digits off the mean. */
    double acc[64];
    for (int d = 0; d < M_ED; d++) acc[d] = 0.0;
    for (int v = 0; v < M_VOCAB; v++) {
        double p = exp((double)logits[v] - mx);
        sum += p;
        const float *erow = g_E + (size_t)v * M_ED;
        for (int d = 0; d < M_ED; d++) acc[d] += p * erow[d];
    }
    for (int d = 0; d < M_ED; d++) mean_out[d] = (float)(acc[d] / sum);
}

/* One denoiser evaluation, mirroring the reference
 * _denoise_block. z is [B][ED] in double. Fills the conditional
 * logits [B][VOCAB], the guided estimate x_guided [B][ED], and
 * the raw conditional mean cond_mean [B][ED]. */
static void denoise(const double *z, double gamma, const float *selfcond,
                    double guidance, const int *bpos, int B,
                    float *logits, float *x_guided, float *cond_mean,
                    WS *w)
{
    double alpha_sq = 1.0 / (1.0 + exp(gamma));
    double sigma_sq = 1.0 / (1.0 + exp(-gamma));
    double alpha = sqrt(alpha_sq);
    double zvar = alpha_sq / M_ED + sigma_sq;
    double zscale = 1.0 / sqrt(zvar);

    float *xin = (float *)malloc((size_t)B * M_ED * sizeof(float));
    float *h = (float *)malloc((size_t)B * M_H * sizeof(float));
    float *hu = (float *)malloc((size_t)B * M_H * sizeof(float));
    float *tmp = (float *)malloc((size_t)B * M_H * sizeof(float));
    if (!xin || !h || !hu || !tmp) { fprintf(stderr, "oom denoise\n"); exit(1); }
    for (int i = 0; i < B * M_ED; i++) xin[i] = (float)(z[i] * zscale);
    matmul(xin, g_in_linear, h, B, M_H, M_ED);
    memcpy(hu, h, (size_t)B * M_H * sizeof(float));
    if (selfcond) {
        float *scin = (float *)malloc((size_t)B * M_ED * sizeof(float));
        if (!scin) { fprintf(stderr, "oom denoise\n"); exit(1); }
        double s = sqrt((double)M_ED);
        for (int i = 0; i < B * M_ED; i++) scin[i] = (float)(selfcond[i] * s);
        matmul(scin, g_sc_linear, tmp, B, M_H, M_ED);
        for (int i = 0; i < B * M_H; i++) h[i] += tmp[i];
        free(scin);
    }
    /* gamma (log-SNR) embedding: sin/cos of exp-spaced freqs */
    {
        float ge[64];
        float *grow = (float *)malloc((size_t)M_H * sizeof(float));
        if (!grow) { fprintf(stderr, "oom denoise\n"); exit(1); }
        for (int j = 0; j < 32; j++) {
            double fr = exp(-5.0 + (10.0 * j) / 31.0);
            ge[j] = (float)sin(fr * gamma);
            ge[32 + j] = (float)cos(fr * gamma);
        }
        matmul(ge, g_gamma_linear, grow, 1, M_H, 64);
        for (int i = 0; i < B; i++)
            for (int d = 0; d < M_H; d++) {
                h[(size_t)i * M_H + d] += grow[d];
                hu[(size_t)i * M_H + d] += grow[d];
            }
        free(grow);
    }
    /* The output prior (alpha / sigma_sq) * (z @ E^T) does not
     * depend on the hidden states, so it is computed once and
     * added to both the conditional and unconditional logits. */
    double prior = alpha / sigma_sq;
    float *prior_dot = (float *)malloc((size_t)B * M_VOCAB * sizeof(float));
    float *zf = (float *)malloc((size_t)B * M_ED * sizeof(float));
    if (!prior_dot || !zf) { fprintf(stderr, "oom denoise\n"); exit(1); }
    for (int i = 0; i < B * M_ED; i++) zf[i] = (float)z[i];
    matmul(zf, g_E, prior_dot, B, M_VOCAB, M_ED);
    for (uint64_t i = 0; i < (uint64_t)B * M_VOCAB; i++)
        prior_dot[i] = (float)(prior * prior_dot[i]);
    free(zf);

    /* conditional pass: block attends the clean prefix cache */
    encoder_forward(h, bpos, B, ATT_READ, w);
    head_all(h, B, logits);
    for (int i = 0; i < B; i++) {
        float *lg = logits + (size_t)i * M_VOCAB;
        const float *pd = prior_dot + (size_t)i * M_VOCAB;
        for (int v = 0; v < M_VOCAB; v++) lg[v] += pd[v];
        mean_from_logits(lg, cond_mean + (size_t)i * M_ED);
    }
    if (guidance != 1.0) {
        float *logits_u = (float *)malloc((size_t)B * M_VOCAB * sizeof(float));
        float *x_u = (float *)malloc((size_t)B * M_ED * sizeof(float));
        if (!logits_u || !x_u) { fprintf(stderr, "oom denoise\n"); exit(1); }
        encoder_forward(hu, bpos, B, ATT_BLOCK_ONLY, w);
        head_all(hu, B, logits_u);
        for (int i = 0; i < B; i++) {
            float *lg = logits_u + (size_t)i * M_VOCAB;
            const float *pd = prior_dot + (size_t)i * M_VOCAB;
            for (int v = 0; v < M_VOCAB; v++) lg[v] += pd[v];
            mean_from_logits(lg, x_u + (size_t)i * M_ED);
        }
        for (int i = 0; i < B * M_ED; i++)
            x_guided[i] = (float)(x_u[i] + guidance * (cond_mean[i] - x_u[i]));
        free(logits_u);
        free(x_u);
    } else {
        memcpy(x_guided, cond_mean, (size_t)B * M_ED * sizeof(float));
    }
    free(prior_dot);
    free(xin); free(h); free(hu); free(tmp);
}

static void encode_clean(const int *ids, int n, const int *pos, WS *w)
{
    float *hidden = (float *)malloc((size_t)n * M_H * sizeof(float));
    if (!hidden) { fprintf(stderr, "oom encode\n"); exit(1); }
    for (int i = 0; i < n; i++) {
        if (src_read_range(g_embed_ent, (uint64_t)ids[i] * M_H, M_H,
                           hidden + (size_t)i * M_H) != 0) {
            fprintf(stderr, "embed read failed\n");
            exit(1);
        }
    }
    encoder_forward(hidden, pos, n, ATT_WRITE, w);
    free(hidden);
}

/* ================= main ================= */

static int cmp_str(const void *a, const void *b)
{
    return strcmp((const char *)a, (const char *)b);
}

int main(int argc, char **argv)
{
    const char *dir = NULL;
    const char *pk_stream = NULL, *pk_book = NULL;
    const char *ids_arg = NULL;
    int max_new = 64, steps = 32, block = 32;
    int sampler_ddim = 1, schedule_lognsnr = 1;
    double guidance = 3.0, score_temp = 0.3;
    uint64_t seed = 1234;
    int threads = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) dir = argv[++i];
        else if (strcmp(argv[i], "--packed") == 0 && i + 2 < argc) {
            pk_stream = argv[i + 1];
            pk_book = argv[i + 2];
            i += 2;
        }
        else if (strcmp(argv[i], "--ids") == 0 && i + 1 < argc) ids_arg = argv[++i];
        else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) max_new = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = atoi(argv[++i]);
        else if (strcmp(argv[i], "--sampler") == 0 && i + 1 < argc) sampler_ddim = strcmp(argv[++i], "ddpm") != 0;
        else if (strcmp(argv[i], "--schedule") == 0 && i + 1 < argc) schedule_lognsnr = strcmp(argv[++i], "edm_sigma") != 0;
        else if (strcmp(argv[i], "--guidance") == 0 && i + 1 < argc) guidance = atof(argv[++i]);
        else if (strcmp(argv[i], "--score-temp") == 0 && i + 1 < argc) score_temp = atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) threads = atoi(argv[++i]);
        else {
            fprintf(stderr, "unknown or incomplete option: %s\n", argv[i]);
            return 1;
        }
    }
    if (steps < 1) steps = 1;
    if (block < 1) block = 1;
    if (block > 4096) {
        fprintf(stderr, "warning: block clamped to 4096\n");
        block = 4096;
    }
    if (!dir || !ids_arg) {
        fprintf(stderr,
            "usage: nq2-infer --dir MODEL_DIR [--packed STREAM BOOK]\n"
            "       --ids 1,25,882 [--tokens 64] [--steps 32]\n"
            "       [--block 32] [--sampler ddim|ddpm]\n"
            "       [--schedule log_snr|edm_sigma] [--guidance 3.0]\n"
            "       [--score-temp 0.3] [--seed 1234] [--threads N]\n");
        return 1;
    }

    /* directory: every .safetensors in the model dir, sorted so
     * shard order matches the quantizer's packing order */
    {
        char names[16][512];
        int nn = 0;
        /* opendir/readdir, not popen("ls ...") (Archie P0-3): no
         * shell in the path, so --dir cannot inject commands, and
         * the same code runs on Windows (MinGW carries dirent.h),
         * which also retires the old _WIN32 early exit that refused
         * to run there. loader.c scans its directory the same way. */
        DIR *dd = opendir(dir);
        if (!dd) { fprintf(stderr, "cannot list %s\n", dir); return 1; }
        struct dirent *de;
        while ((de = readdir(dd)) != NULL && nn < MAX_SHARDS) {
            size_t L = strlen(de->d_name);
            if (L > 12 && strcmp(de->d_name + L - 12, ".safetensors") == 0) {
                snprintf(names[nn], 512, "%s", de->d_name);
                nn++;
            }
        }
        closedir(dd);
        qsort(names, nn, 512, cmp_str);
        if (nn == 0) { fprintf(stderr, "no safetensors in %s\n", dir); return 1; }
        for (int s = 0; s < nn; s++) {
            char path[1024];
            snprintf(path, sizeof(path), "%s/%s", dir, names[s]);
            if (shard_open_raw(path, s) != 0) {
                fprintf(stderr, "cannot open %s\n", path);
                return 1;
            }
        }
        g_nshards = nn;
    }

    /* derive the model's shape from its own tensors */
    g_embed_ent = dir_find("encoder.embed_tokens.weight");
    g_head_ent = dir_find("diffusion_head.weight");
    if (!g_embed_ent || !g_head_ent) {
        fprintf(stderr, "not a Sigma-family model directory\n");
        return 1;
    }
    M_VOCAB = g_embed_ent->rows;
    M_H = g_embed_ent->cols;
    {
        TEnt *e = dir_find("replaid_embedding_matrix.matrix");
        TEnt *g0 = dir_find("encoder.layers.0.mlp.gate_proj.weight");
        TEnt *q0 = dir_find("encoder.layers.0.self_attn.q_proj.weight");
        TEnt *k0 = dir_find("encoder.layers.0.self_attn.k_proj.weight");
        if (!e || !g0 || !q0 || !k0) {
            fprintf(stderr, "model directory is incomplete\n");
            return 1;
        }
        M_ED = e->cols;
        if (M_ED > 64) {
            fprintf(stderr, "embed_dim %d above the supported 64\n",
                    M_ED);
            return 1;
        }
        M_FFN = g0->rows;
        if (q0->rows % M_HD != 0 || k0->rows % M_HD != 0 || M_HD > 128) {
            fprintf(stderr, "unsupported head geometry\n");
            return 1;
        }
        M_HEADS = q0->rows / M_HD;
        M_KVHEADS = k0->rows / M_HD;
        int maxl = -1;
        for (int i = 0; i < g_ndir; i++) {
            int li;
            if (sscanf(g_dir[i].name, "encoder.layers.%d.", &li) == 1
                    && li > maxl)
                maxl = li;
        }
        M_L = maxl + 1;
    }

    if (pk_stream) {
        g_use_packed = 1;
        if (packed_open(pk_stream, pk_book) != 0) {
            fprintf(stderr, "cannot open packed model\n");
            return 1;
        }
    }

    /* resident small tensors */
    g_norm_in = (float *)malloc((size_t)M_L * M_H * sizeof(float));
    g_norm_post = (float *)malloc((size_t)M_L * M_H * sizeof(float));
    if (!g_norm_in || !g_norm_post) { fprintf(stderr, "oom\n"); return 1; }
    for (int l = 0; l < M_L; l++) {
        char nm[256];
        snprintf(nm, sizeof(nm), "encoder.layers.%d.input_layernorm.weight", l);
        float *b = load_small(nm);
        memcpy(g_norm_in + (size_t)l * M_H, b, M_H * sizeof(float));
        free(b);
        snprintf(nm, sizeof(nm), "encoder.layers.%d.post_attention_layernorm.weight", l);
        b = load_small(nm);
        memcpy(g_norm_post + (size_t)l * M_H, b, M_H * sizeof(float));
        free(b);
    }
    g_norm_final = load_small("encoder.norm.weight");
    g_in_linear = load_small("replaid_input_linear.weight");
    g_sc_linear = load_small("replaid_selfcond_linear.weight");
    g_gamma_linear = load_small("replaid_gamma_linear.weight");
    {
        float *raw_e = load_small("replaid_embedding_matrix.matrix");
        g_E = (float *)malloc((size_t)M_VOCAB * M_ED * sizeof(float));
        if (!g_E) { fprintf(stderr, "oom\n"); return 1; }
        for (int v = 0; v < M_VOCAB; v++) {
            const float *r = raw_e + (size_t)v * M_ED;
            double nrm = 0.0;
            for (int d = 0; d < M_ED; d++) nrm += (double)r[d] * r[d];
            double inv = 1.0 / (sqrt(nrm) + 1e-8);
            for (int d = 0; d < M_ED; d++)
                g_E[(size_t)v * M_ED + d] = (float)(r[d] * inv);
        }
        free(raw_e);
        float *gb = load_small("gamma_bounds.endpoint_values");
        g_gamma0 = gb[0];
        g_gamma1 = gb[1];
        free(gb);
    }

    /* prompt ids */
    int prompt[8192];
    int plen = 0;
    {
        char buf[65536];
        snprintf(buf, sizeof(buf), "%s", ids_arg);
        char *tok = strtok(buf, ", ");
        while (tok && plen < 8192) {
            prompt[plen++] = atoi(tok);
            tok = strtok(NULL, ", ");
        }
    }
    if (plen == 0) { fprintf(stderr, "empty prompt\n"); return 1; }
    for (int i = 0; i < plen; i++)
        if (prompt[i] < 0 || prompt[i] >= M_VOCAB) {
            fprintf(stderr, "token id %d out of range\n", prompt[i]);
            return 1;
        }

    int cap = plen + max_new + block + 8;
    if (threads <= 0) threads = cpu_count();
    pool_init(threads);
    pipe_init(M_L);
    rope_init(cap);
    cache_init(cap);
    WS w = ws_alloc(plen > block ? plen : block);

    size_t pipe_bytes = 2 * g_pipe.cap_elems * sizeof(float);
    size_t cache_bytes = (size_t)M_L * 2 * cap * (M_KVHEADS * M_HD) * 2;
    printf("NanoQuatTwo inference engine\n");
    printf("model: %d layers, hidden %d, vocab %d, embed_dim %d, "
           "source: %s\n", M_L, M_H, M_VOCAB, M_ED,
           g_use_packed ? "packed .nq2" : "raw safetensors");
    printf("threads: %d workers + 1 prefetch; derived budget: "
           "pipe %.1f MB, KV cache %.1f MB, embedding matrix %.1f MB\n",
           threads, pipe_bytes / 1048576.0, cache_bytes / 1048576.0,
           (size_t)M_VOCAB * M_ED * 4 / 1048576.0);

    /* gamma grid, fp64 */
    double *grid = (double *)malloc(sizeof(double) * (steps + 1));
    if (!grid) { fprintf(stderr, "oom\n"); return 1; }
    if (schedule_lognsnr) {
        for (int i = 0; i <= steps; i++)
            grid[i] = g_gamma1 + (g_gamma0 - g_gamma1) * ((double)i / steps);
    } else {
        double rho = 7.0;
        double smax = exp(g_gamma1 / 2.0), smin = exp(g_gamma0 / 2.0);
        double a = pow(smax, 1.0 / rho), b = pow(smin, 1.0 / rho);
        for (int i = 0; i <= steps; i++) {
            double s = pow(a + (b - a) * ((double)i / steps), rho);
            if (s < 1e-30) s = 1e-30;
            grid[i] = 2.0 * log(s);
        }
    }

    rng_seed(seed);
    {
        int *pos = (int *)malloc(sizeof(int) * (plen > block ? plen : block));
        for (int i = 0; i < plen; i++) pos[i] = i;
        encode_clean(prompt, plen, pos, &w);

        int *out_ids = (int *)malloc(sizeof(int) * max_new);
        double *z = (double *)malloc((size_t)block * M_ED * sizeof(double));
        float *selfcond = (float *)malloc((size_t)block * M_ED * sizeof(float));
        float *logits = (float *)malloc((size_t)block * M_VOCAB * sizeof(float));
        float *x_guided = (float *)malloc((size_t)block * M_ED * sizeof(float));
        float *cond_mean = (float *)malloc((size_t)block * M_ED * sizeof(float));
        if (!pos || !out_ids || !z || !selfcond || !logits || !x_guided
                || !cond_mean) {
            fprintf(stderr, "oom\n");
            return 1;
        }
        long nfe = 0;
        int generated = 0;
        int nblocks = (max_new + block - 1) / block;
        for (int bidx = 0; bidx < nblocks; bidx++) {
            for (int i = 0; i < block * M_ED; i++) z[i] = randn();
            int have_sc = 0;
            for (int i = 0; i < block; i++)
                pos[i] = plen + generated + i;
            for (int s = 0; s <= steps; s++) {
                double gt = grid[s];
                denoise(z, gt, have_sc ? selfcond : NULL, guidance,
                        pos, block, logits, x_guided, cond_mean, &w);
                nfe += guidance != 1.0 ? 2 : 1;
                if (s == steps) {
                    for (int i = 0; i < block; i++) {
                        const float *lg = logits + (size_t)i * M_VOCAB;
                        int best = 0;
                        for (int v = 1; v < M_VOCAB; v++)
                            if (lg[v] > lg[best]) best = v;
                        if (generated + i < max_new)
                            out_ids[generated + i] = best;
                    }
                    break;
                }
                memcpy(selfcond, cond_mean,
                       (size_t)block * M_ED * sizeof(float));
                have_sc = 1;
                double a_sq_t = 1.0 / (1.0 + exp(gt));
                double a_t = sqrt(a_sq_t);
                double s_t = sqrt(1.0 / (1.0 + exp(-gt)));
                double gs = grid[s + 1];
                double a_sq_s = 1.0 / (1.0 + exp(gs));
                double a_s = sqrt(a_sq_s);
                double s_s = sqrt(1.0 / (1.0 + exp(-gs)));
                for (int i = 0; i < block * M_ED; i++) {
                    double x = x_guided[i];
                    double eps = (z[i] - a_t * x) / (s_t < 1e-8 ? 1e-8 : s_t);
                    if (score_temp != 1.0) {
                        eps /= score_temp;
                        x = (z[i] - s_t * eps) / (a_t < 1e-8 ? 1e-8 : a_t);
                    }
                    if (sampler_ddim) {
                        z[i] = a_s * x + s_s * eps;
                    } else {
                        double c = -expm1(gs - gt);
                        double noise = randn();
                        double var = c * (1.0 - a_sq_s);
                        if (var < 0.0) var = 0.0;
                        z[i] = (1.0 - c) * sqrt(a_sq_s / a_sq_t) * z[i]
                             + c * a_s * x + sqrt(var) * noise;
                    }
                }
            }
            generated += block;
            if (bidx + 1 < nblocks) {
                int blk_ids[4096];
                for (int i = 0; i < block; i++)
                    blk_ids[i] = out_ids[bidx * block + i];
                encode_clean(blk_ids, block, pos, &w);
            }
        }
        printf("generated token ids:");
        for (int i = 0; i < max_new; i++) printf(" %d", out_ids[i]);
        printf("\n");
        printf("nfe: %ld (denoiser evaluations), seed %llu\n",
               nfe, (unsigned long long)seed);
    }

    pipe_report();
    pipe_shutdown();
    pool_shutdown();
    return 0;
}
