/* fid-common.h ... shared pieces of the fidelity lab tools.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27, for the
 * fidelity roadmap (docs/ROADMAP.md). Pure C99 plus POSIX threads
 * (MinGW winpthreads on Windows). Zero mutexes: workers are created
 * and joined, and hand out work through one atomic counter.
 *
 * Determinism contract: every parallel loop splits its work into
 * items whose boundaries do not depend on the thread count, each
 * item is computed by exactly one worker in a fixed order, and any
 * reduction across items happens afterwards in item order. Results
 * are therefore bit-identical for any --threads value.
 */
#ifndef FID_COMMON_H
#define FID_COMMON_H
#pragma GCC diagnostic ignored "-Wunused-function"

#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* ---------------- time, files ---------------- */

static double fid_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static int fid_seek(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static uint64_t fid_fsize(FILE *f)
{
#ifdef _WIN32
    if (_fseeki64(f, 0, SEEK_END) != 0) return 0;
    long long s = _ftelli64(f);
#else
    if (fseeko(f, 0, SEEK_END) != 0) return 0;
    off_t s = ftello(f);
#endif
    return s < 0 ? 0 : (uint64_t)s;
}

static void *fid_alloc(size_t n)
{
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", n); exit(2); }
    return p;
}

static void *fid_calloc(size_t n, size_t sz)
{
    void *p = calloc(n ? n : 1, sz);
    if (!p) { fprintf(stderr, "out of memory (%zu x %zu bytes)\n", n, sz); exit(2); }
    return p;
}

/* ---------------- threads, zero mutex ---------------- */

static int g_fid_threads = 1;

static int fid_cpu_count(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors > 0 ? (int)si.dwNumberOfProcessors : 1;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

typedef void (*fid_item_fn)(void *ctx, int64_t item);
typedef struct {
    fid_item_fn fn;
    void *ctx;
    int64_t n;
    int64_t next;   /* atomic */
} FidPar;

static void *fid_par_worker(void *p)
{
    FidPar *j = (FidPar *)p;
    for (;;) {
        int64_t it = __atomic_fetch_add(&j->next, 1, __ATOMIC_RELAXED);
        if (it >= j->n) break;
        j->fn(j->ctx, it);
    }
    return NULL;
}

/* Run fn(ctx, item) for item in [0, n). */
static void fid_par_for(int64_t n, fid_item_fn fn, void *ctx)
{
    if (n <= 0) return;
    FidPar j;
    j.fn = fn; j.ctx = ctx; j.n = n; j.next = 0;
    int nt = g_fid_threads;
    if (nt > n) nt = (int)n;
    if (nt <= 1) { fid_par_worker(&j); return; }
    pthread_t *th = (pthread_t *)fid_alloc(sizeof(pthread_t) * nt);
    int *ok = (int *)fid_calloc(nt, sizeof(int));
    for (int t = 1; t < nt; t++)
        ok[t] = pthread_create(&th[t], NULL, fid_par_worker, &j) == 0;
    fid_par_worker(&j);
    for (int t = 1; t < nt; t++)
        if (ok[t]) pthread_join(th[t], NULL);
    free(th); free(ok);
}

/* ---------------- number formats ---------------- */

static float fid_bf16_to_f32(uint16_t h)
{
    uint32_t b = (uint32_t)h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

static uint16_t fid_f32_to_bf16(float f)
{
    uint32_t b;
    memcpy(&b, &f, 4);
    if ((b & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
    b += 0x7FFFu + ((b >> 16) & 1u);
    return (uint16_t)(b >> 16);
}

static float fid_f16_to_f32(uint16_t h)
{
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff, b;
    if (e == 0) {
        if (m == 0) b = s;
        else {
            int sh = 0;
            while (!(m & 0x400)) { m <<= 1; sh++; }
            m &= 0x3ff;
            b = s | ((uint32_t)(113 - sh) << 23) | (m << 13);
        }
    } else if (e == 31) b = s | 0x7f800000u | (m << 13);
    else b = s | ((e + 112) << 23) | (m << 13);
    float f;
    memcpy(&f, &b, 4);
    return f;
}

/* Round to nearest even float16 (same rounding as nq2-quantize). */
static uint16_t fid_f32_to_f16(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 112;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u
                && ((half & 1u) || (mant & ((1u << (shift - 1)) - 1))))
            half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (mant & 0x1000u && ((mant & 0x2000u) || (mant & 0xfffu))) mant += 0x2000u;
    if (mant & 0x800000u) { mant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static float fid_round_f16(float f) { return fid_f16_to_f32(fid_f32_to_f16(f)); }

/* dir/name into dst, or exit: a path that does not fit is an error,
 * never a silent truncation */
static void fid_join(char *dst, size_t n, const char *dir, const char *name)
{
    int k = snprintf(dst, n, "%s/%s", dir, name);
    if (k < 0 || (size_t)k >= n) { fprintf(stderr, "path too long: %s/%s\n", dir, name); exit(1); }
}

static uint64_t fid_fnv(const void *p, size_t n, uint64_t h)
{
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
#define FID_FNV_SEED 14695981039346656037ull

static uint64_t fid_mix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

/* ---------------- safetensors ---------------- */

enum { FID_BF16 = 0, FID_F16 = 1, FID_F32 = 2 };

typedef struct {
    char name[256];
    int dtype;
    int ndim;
    uint64_t dims[4];
    uint64_t count;
    uint64_t off;     /* absolute byte offset in the shard file */
    uint64_t bytes;
} FidTensor;

typedef struct {
    char path[2200];
    FILE *f;
    uint64_t data_start;
    int n;
    FidTensor *t;
} FidShard;

static int fid_js_string_end(const char *s, long n, long pos)
{
    long i = pos + 1;
    while (i < n) {
        if (s[i] == '\\') { i += 2; continue; }
        if (s[i] == '"') return (int)(i + 1);
        i++;
    }
    return -1;
}

static int fid_shard_open(FidShard *sh, const char *path)
{
    memset(sh, 0, sizeof(*sh));
    snprintf(sh->path, sizeof(sh->path), "%s", path);
    sh->f = fopen(path, "rb");
    if (!sh->f) return -1;
    uint64_t fsz = fid_fsize(sh->f);
    fid_seek(sh->f, 0);
    uint8_t lb[8];
    if (fread(lb, 1, 8, sh->f) != 8) return -1;
    uint64_t hl = 0;
    for (int i = 0; i < 8; i++) hl |= (uint64_t)lb[i] << (8 * i);
    if (hl == 0 || hl > (1ull << 28) || 8 + hl > fsz) return -1;
    char *h = (char *)fid_alloc(hl + 1);
    if (fread(h, 1, hl, sh->f) != hl) { free(h); return -1; }
    h[hl] = 0;
    sh->data_start = 8 + hl;
    int cap = 64;
    sh->t = (FidTensor *)fid_calloc(cap, sizeof(FidTensor));
    long n = (long)hl, pos = 0;
    while (pos < n) {
        while (pos < n && h[pos] != '"') pos++;
        if (pos >= n) break;
        int ke = fid_js_string_end(h, n, pos);
        if (ke < 0) break;
        long kl = ke - pos - 2;
        char key[256];
        if (kl > 250) kl = 250;
        memcpy(key, h + pos + 1, kl);
        key[kl] = 0;
        long p = ke;
        while (p < n && (h[p] == ' ' || h[p] == ':' || h[p] == '\t')) p++;
        if (p >= n || h[p] != '{') { pos = ke; continue; }
        long q = p, depth = 0;
        int instr = 0;
        while (q < n) {
            char c = h[q];
            if (instr) { if (c == '\\') { q += 2; continue; } if (c == '"') instr = 0; }
            else if (c == '"') instr = 1;
            else if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') { depth--; if (depth == 0) break; }
            q++;
        }
        char obj[1024];
        long ol = q - p + 1;
        if (ol > 1023) ol = 1023;
        memcpy(obj, h + p, ol);
        obj[ol] = 0;
        pos = q + 1;
        if (!strcmp(key, "__metadata__")) continue;
        if (sh->n == cap) {
            cap *= 2;
            sh->t = (FidTensor *)realloc(sh->t, cap * sizeof(FidTensor));
            if (!sh->t) { free(h); return -1; }
        }
        FidTensor *t = &sh->t[sh->n];
        memset(t, 0, sizeof(*t));
        snprintf(t->name, sizeof(t->name), "%s", key);
        const char *dt = strstr(obj, "\"dtype\"");
        if (!dt) { free(h); return -1; }
        if (strstr(dt, "\"BF16\"")) t->dtype = FID_BF16;
        else if (strstr(dt, "\"F16\"")) t->dtype = FID_F16;
        else if (strstr(dt, "\"F32\"")) t->dtype = FID_F32;
        else { fprintf(stderr, "unsupported dtype in %s\n", key); free(h); return -1; }
        const char *sp = strstr(obj, "\"shape\"");
        const char *lb2 = sp ? strchr(sp, '[') : NULL;
        if (!lb2) { free(h); return -1; }
        const char *c = lb2 + 1;
        t->count = 1;
        while (*c && *c != ']') {
            while (*c == ' ' || *c == ',') c++;
            if (*c == ']') break;
            uint64_t v = strtoull(c, (char **)&c, 10);
            if (t->ndim < 4) t->dims[t->ndim] = v;
            t->ndim++;
            t->count *= v;
        }
        const char *df = strstr(obj, "\"data_offsets\"");
        const char *lb3 = df ? strchr(df, '[') : NULL;
        if (!lb3) { free(h); return -1; }
        uint64_t b0 = strtoull(lb3 + 1, NULL, 10);
        const char *cm = strchr(lb3, ',');
        uint64_t b1 = cm ? strtoull(cm + 1, NULL, 10) : 0;
        uint64_t esz = t->dtype == FID_F32 ? 4 : 2;
        if (b1 < b0 || b1 - b0 != t->count * esz || sh->data_start + b1 > fsz) {
            fprintf(stderr, "inconsistent data_offsets for %s\n", key);
            free(h);
            return -1;
        }
        t->off = sh->data_start + b0;
        t->bytes = b1 - b0;
        sh->n++;
    }
    free(h);
    return 0;
}

static int fid_tensor_read(FidShard *sh, const FidTensor *t, uint64_t e0,
                           uint64_t ne, float *out)
{
    if (e0 > t->count || ne > t->count - e0) return -1;
    int esz = t->dtype == FID_F32 ? 4 : 2;
    const uint64_t CH = 1ull << 22;
    uint8_t *buf = (uint8_t *)fid_alloc((size_t)(ne < CH ? ne : CH) * esz);
    for (uint64_t done = 0; done < ne; done += CH) {
        uint64_t m = ne - done < CH ? ne - done : CH;
        if (fid_seek(sh->f, t->off + (e0 + done) * esz) != 0
                || fread(buf, esz, m, sh->f) != m) { free(buf); return -1; }
        if (t->dtype == FID_BF16)
            for (uint64_t i = 0; i < m; i++) out[done + i] = fid_bf16_to_f32(((uint16_t *)buf)[i]);
        else if (t->dtype == FID_F16)
            for (uint64_t i = 0; i < m; i++) out[done + i] = fid_f16_to_f32(((uint16_t *)buf)[i]);
        else memcpy(out + done, buf, m * 4);
    }
    free(buf);
    return 0;
}

/* ---------------- dense kernels ----------------
 * C[m][n] += A[m][k] * B[k][n], all row-major float with leading
 * dimensions. Work items are fixed 32-row x 256-column tiles of C;
 * each walks k in fixed 256-wide chunks, keeps a 4 x 16 accumulator
 * tile per chunk, and adds it into C in chunk order. Tile and chunk
 * boundaries do not depend on the thread count, so neither does the
 * result. upper=1 computes only tiles that touch the upper triangle
 * (j >= i), for symmetric products; the caller mirrors. */
#define FID_TM 32
#define FID_TN 256
#define FID_TK 256
typedef struct {
    float *C; const float *A; const float *B;
    int64_t m, k, n, ldc, lda, ldb;
    int64_t ntn;
    int upper;
} FidGemm;

static void fid_gemm_item(void *vc, int64_t it)
{
    FidGemm *g = (FidGemm *)vc;
    int64_t rb = it / g->ntn, cb = it % g->ntn;
    int64_t i_beg = rb * FID_TM, j_beg = cb * FID_TN;
    int64_t i_end = i_beg + FID_TM < g->m ? i_beg + FID_TM : g->m;
    int64_t j_end = j_beg + FID_TN < g->n ? j_beg + FID_TN : g->n;
    if (g->upper && j_end <= i_beg) return;
    for (int64_t k0 = 0; k0 < g->k; k0 += FID_TK) {
        int64_t k1 = k0 + FID_TK < g->k ? k0 + FID_TK : g->k;
        for (int64_t i0 = i_beg; i0 < i_end; i0 += 4) {
            int64_t mi = i_end - i0 < 4 ? i_end - i0 : 4;
            for (int64_t j0 = j_beg; j0 < j_end; j0 += 16) {
                int64_t nj = j_end - j0 < 16 ? j_end - j0 : 16;
                float acc[4][16];
                for (int a = 0; a < 4; a++)
                    for (int b = 0; b < 16; b++) acc[a][b] = 0.0f;
                if (mi == 4 && nj == 16) {
                    const float *a0 = g->A + i0 * g->lda;
                    const float *a1 = a0 + g->lda, *a2 = a1 + g->lda, *a3 = a2 + g->lda;
                    for (int64_t kk = k0; kk < k1; kk++) {
                        const float *br = g->B + kk * g->ldb + j0;
                        float v0 = a0[kk], v1 = a1[kk], v2 = a2[kk], v3 = a3[kk];
                        for (int b = 0; b < 16; b++) {
                            float bb = br[b];
                            acc[0][b] += v0 * bb;
                            acc[1][b] += v1 * bb;
                            acc[2][b] += v2 * bb;
                            acc[3][b] += v3 * bb;
                        }
                    }
                } else {
                    for (int64_t kk = k0; kk < k1; kk++) {
                        const float *br = g->B + kk * g->ldb + j0;
                        for (int a = 0; a < mi; a++) {
                            float v = g->A[(i0 + a) * g->lda + kk];
                            for (int b = 0; b < nj; b++) acc[a][b] += v * br[b];
                        }
                    }
                }
                for (int a = 0; a < mi; a++) {
                    float *cr = g->C + (i0 + a) * g->ldc + j0;
                    for (int b = 0; b < nj; b++) cr[b] += acc[a][b];
                }
            }
        }
    }
}

static void fid_gemm_ex(float *C, int64_t ldc, const float *A, int64_t lda,
                        const float *B, int64_t ldb, int64_t m, int64_t k,
                        int64_t n, int upper)
{
    FidGemm g;
    g.C = C; g.A = A; g.B = B; g.m = m; g.k = k; g.n = n;
    g.ldc = ldc; g.lda = lda; g.ldb = ldb; g.upper = upper;
    g.ntn = (n + FID_TN - 1) / FID_TN;
    int64_t ntm = (m + FID_TM - 1) / FID_TM;
    fid_par_for(ntm * g.ntn, fid_gemm_item, &g);
}

/* C[m][n] = A[m][k] . B[n][k]^T (both row-major, dot products along
 * k): the shape of every projection y = x W^T. Items are fixed
 * 32 x 64 tiles of C; k runs in 256-wide chunks with eight float
 * lanes per dot product, the lanes summed in a fixed order and the
 * chunks accumulated into C in order. C is overwritten. */
typedef struct {
    float *C; const float *A; const float *B;
    int64_t m, k, n, ldc, lda, ldb, ntn;
} FidGemmNT;

static void fid_gemm_nt_item(void *vc, int64_t it)
{
    FidGemmNT *g = (FidGemmNT *)vc;
    int64_t rb = it / g->ntn, cb = it % g->ntn;
    int64_t i_beg = rb * 32, j_beg = cb * 64;
    int64_t i_end = i_beg + 32 < g->m ? i_beg + 32 : g->m;
    int64_t j_end = j_beg + 64 < g->n ? j_beg + 64 : g->n;
    for (int64_t i = i_beg; i < i_end; i++)
        for (int64_t j = j_beg; j < j_end; j++) g->C[i * g->ldc + j] = 0.0f;
    for (int64_t k0 = 0; k0 < g->k; k0 += 256) {
        int64_t k1 = k0 + 256 < g->k ? k0 + 256 : g->k;
        int64_t k8 = k0 + ((k1 - k0) & ~(int64_t)7);
        for (int64_t i0 = i_beg; i0 < i_end; i0 += 4) {
            int mi = (int)(i_end - i0 < 4 ? i_end - i0 : 4);
            for (int64_t j0 = j_beg; j0 < j_end; j0 += 4) {
                int nj = (int)(j_end - j0 < 4 ? j_end - j0 : 4);
                float acc[4][4][8];
                memset(acc, 0, sizeof(acc));
                const float *ar[4], *br[4];
                for (int a = 0; a < 4; a++) ar[a] = g->A + (i0 + (a < mi ? a : 0)) * g->lda;
                for (int b = 0; b < 4; b++) br[b] = g->B + (j0 + (b < nj ? b : 0)) * g->ldb;
                for (int64_t kk = k0; kk < k8; kk += 8)
                    for (int a = 0; a < 4; a++)
                        for (int b = 0; b < 4; b++)
                            for (int l = 0; l < 8; l++)
                                acc[a][b][l] += ar[a][kk + l] * br[b][kk + l];
                for (int a = 0; a < mi; a++)
                    for (int b = 0; b < nj; b++) {
                        float s = ((acc[a][b][0] + acc[a][b][1]) + (acc[a][b][2] + acc[a][b][3]))
                                + ((acc[a][b][4] + acc[a][b][5]) + (acc[a][b][6] + acc[a][b][7]));
                        for (int64_t kk = k8; kk < k1; kk++) s += ar[a][kk] * br[b][kk];
                        g->C[(i0 + a) * g->ldc + j0 + b] += s;
                    }
            }
        }
    }
}

static void fid_gemm_nt(float *C, int64_t ldc, const float *A, int64_t lda,
                        const float *B, int64_t ldb, int64_t m, int64_t k, int64_t n)
{
    FidGemmNT g;
    g.C = C; g.A = A; g.B = B; g.m = m; g.k = k; g.n = n;
    g.ldc = ldc; g.lda = lda; g.ldb = ldb; g.ntn = (n + 63) / 64;
    fid_par_for(((m + 31) / 32) * g.ntn, fid_gemm_nt_item, &g);
}

static void fid_gemm(float *C, int64_t ldc, const float *A, int64_t lda,
                     const float *B, int64_t ldb, int64_t m, int64_t k, int64_t n)
{
    fid_gemm_ex(C, ldc, A, lda, B, ldb, m, k, n, 0);
}

#endif
