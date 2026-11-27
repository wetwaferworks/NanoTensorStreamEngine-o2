/* sn_gemm.c - see sn_gemm.h. Pure C99 (GCC target attributes for the AVX2 kernels), zero mutexes. */
#include "sn_gemm.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SN_HAVE_X86 1
#endif

#define CHUNK_ROWS 32

static int g_avx2 = -1;
static int have_avx2(void)
{
    if (g_avx2 < 0) {
#ifdef SN_HAVE_X86
        __builtin_cpu_init();
        g_avx2 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
        g_avx2 = 0;
#endif
    }
    return g_avx2;
}

static float apply_act(float x, int act)
{
    if (act == 1) return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
    if (act == 2) { float r = x > 0 ? x : 0; return r * r; }
    return x;
}

#ifdef SN_HAVE_X86
__attribute__((target("avx2,fma")))
static inline float hsum8(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_hadd_ps(lo, lo); lo = _mm_hadd_ps(lo, lo);
    return _mm_cvtss_f32(lo);
}

__attribute__((target("avx2,fma")))
static void rows_avx2(const float *W, int rows, int K, const float *X, int N, const float *bias,
                      int r0, int R, float *Y, int act)
{
    int n0 = 0;
    for (; n0 + 4 <= N; n0 += 4) {
        const float *x0 = X + (size_t)n0 * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
        for (int r = 0; r < rows; r++) {
            const float *w = W + (size_t)r * K;
            __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
            for (int k = 0; k < K; k += 8) {
                __m256 wv = _mm256_loadu_ps(w + k);
                a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + k), a0);
                a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + k), a1);
                a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + k), a2);
                a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + k), a3);
            }
            float b = bias ? bias[r0 + r] : 0.0f;
            Y[(size_t)(n0 + 0) * R + r0 + r] = apply_act(hsum8(a0) + b, act);
            Y[(size_t)(n0 + 1) * R + r0 + r] = apply_act(hsum8(a1) + b, act);
            Y[(size_t)(n0 + 2) * R + r0 + r] = apply_act(hsum8(a2) + b, act);
            Y[(size_t)(n0 + 3) * R + r0 + r] = apply_act(hsum8(a3) + b, act);
        }
    }
    for (; n0 < N; n0++) {
        const float *x0 = X + (size_t)n0 * K;
        for (int r = 0; r < rows; r++) {
            const float *w = W + (size_t)r * K;
            __m256 a0 = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + k), _mm256_loadu_ps(x0 + k), a0);
            float b = bias ? bias[r0 + r] : 0.0f;
            Y[(size_t)n0 * R + r0 + r] = apply_act(hsum8(a0) + b, act);
        }
    }
}

__attribute__((target("avx2,fma")))
static float dot_avx2(const float *a, const float *b, int n)
{
    __m256 s = _mm256_setzero_ps();
    for (int i = 0; i < n; i += 8) s = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s);
    return hsum8(s);
}
__attribute__((target("avx2,fma")))
static void axpy_avx2(float *y, float a, const float *x, int n)
{
    __m256 av = _mm256_set1_ps(a);
    for (int i = 0; i < n; i += 8) _mm256_storeu_ps(y + i, _mm256_fmadd_ps(av, _mm256_loadu_ps(x + i), _mm256_loadu_ps(y + i)));
}
#endif

static void rows_scalar(const float *W, int rows, int K, const float *X, int N, const float *bias,
                        int r0, int R, float *Y, int act)
{
    for (int n = 0; n < N; n++) {
        const float *x = X + (size_t)n * K;
        for (int r = 0; r < rows; r++) {
            const float *w = W + (size_t)r * K;
            float s = 0.0f;
            for (int k = 0; k < K; k++) s += w[k] * x[k];
            Y[(size_t)n * R + r0 + r] = apply_act(s + (bias ? bias[r0 + r] : 0.0f), act);
        }
    }
}

int sn_gemm_simd(void) { return have_avx2(); }

void sn_gemm_rows(const float *W, int rows, int K, const float *X, int N, const float *bias, int r0, int R, float *Y, int act)
{
#ifdef SN_HAVE_X86
    if (have_avx2()) { rows_avx2(W, rows, K, X, N, bias, r0, R, Y, act); return; }
#endif
    rows_scalar(W, rows, K, X, N, bias, r0, R, Y, act);
}

float sn_dot(const float *a, const float *b, int n)
{
#ifdef SN_HAVE_X86
    if (have_avx2()) return dot_avx2(a, b, n);
#endif
    float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
void sn_axpy(float *y, float a, const float *x, int n)
{
#ifdef SN_HAVE_X86
    if (have_avx2()) { axpy_avx2(y, a, x, n); return; }
#endif
    for (int i = 0; i < n; i++) y[i] += a * x[i];
}

/* ------------------------------------------------------------ the pack-backed product */

typedef struct { sn_rd rd; float *book; float *wbuf; const sn_tensor *booked; } glane;

struct sn_gemm {
    const sn_pack *pack;
    nemo_pool *pool;
    int nthreads;
    glane *lanes;
};

static int lane_of(int n_items, int nthreads, int begin)
{
    int lanes = nthreads < n_items ? nthreads : n_items;
    for (int l = 0; l < lanes; l++)
        if ((int)((long)l * n_items / lanes) == begin) return l;
    return 0;
}

void sn_gemm_close(sn_gemm *g)
{
    if (!g) return;
    if (g->lanes) for (int i = 0; i < g->nthreads; i++) { sn_rd_close(&g->lanes[i].rd); free(g->lanes[i].book); free(g->lanes[i].wbuf); }
    free(g->lanes); free(g);
}

sn_gemm *sn_gemm_open(const sn_pack *p, nemo_pool *pool, int nthreads, size_t max_cols)
{
    sn_gemm *g = (sn_gemm *)calloc(1, sizeof *g);
    if (!g) return NULL;
    g->pack = p; g->pool = pool; g->nthreads = nthreads < 1 ? 1 : nthreads;
    g->lanes = (glane *)calloc((size_t)g->nthreads, sizeof(glane));
    if (!g->lanes) { free(g); return NULL; }
    for (int i = 0; i < g->nthreads; i++) {
        if (sn_rd_open(p, &g->lanes[i].rd) != 0) { sn_gemm_close(g); return NULL; }
        g->lanes[i].book = (float *)malloc(p->book_floats * sizeof(float));
        g->lanes[i].wbuf = (float *)malloc((size_t)CHUNK_ROWS * max_cols * sizeof(float));
        if (!g->lanes[i].book || !g->lanes[i].wbuf) { sn_gemm_close(g); return NULL; }
    }
    return g;
}

typedef struct {
    sn_gemm *g; const sn_tensor *t; const float *bias; const float *X; int N, K, R; float *Y; int act; int fail;
} gemm_ctx;

static void gemm_job(void *vc, int b, int e)
{
    gemm_ctx *c = (gemm_ctx *)vc;
    sn_gemm *g = c->g;
    int nchunks = (c->R + CHUNK_ROWS - 1) / CHUNK_ROWS;
    glane *ln = &g->lanes[lane_of(nchunks, g->nthreads, b)];
    if (!c->t->raw && ln->booked != c->t) {
        if (sn_load_book(g->pack, &ln->rd, c->t, ln->book) != 0) { c->fail = 1; return; }
        ln->booked = c->t;
    }
    for (int ch = b; ch < e; ch++) {
        int r0 = ch * CHUNK_ROWS, r1 = r0 + CHUNK_ROWS < c->R ? r0 + CHUNK_ROWS : c->R;
        if (sn_expand_book(g->pack, &ln->rd, c->t, ln->book, (uint64_t)r0 * c->K, (uint64_t)(r1 - r0) * c->K, ln->wbuf) != 0) { c->fail = 1; return; }
        sn_gemm_rows(ln->wbuf, r1 - r0, c->K, c->X, c->N, c->bias, r0, c->R, c->Y, c->act);
    }
}

int sn_gemm_linear(sn_gemm *g, const sn_tensor *t, const float *bias, const float *X, int N, int K, int R, float *Y, int act)
{
    if ((uint64_t)K * (uint64_t)R != t->count || (K & 7)) return -2;
    gemm_ctx c = { g, t, bias, X, N, K, R, Y, act, 0 };
    nemo_pool_for(g->pool, (R + CHUNK_ROWS - 1) / CHUNK_ROWS, gemm_job, &c);
    return c.fail ? -1 : 0;
}
