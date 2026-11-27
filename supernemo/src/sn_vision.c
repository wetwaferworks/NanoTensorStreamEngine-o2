/* sn_vision.c - see sn_vision.h. Pure C99 (GCC target attributes for the AVX2 kernels), zero mutexes. */
#include "sn_vision.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define SN_HAVE_X86 1
#endif

#define VPFX "vision_model.radio_model.model."
#define CHUNK_ROWS 32

/* ------------------------------------------------------------ config */

static const char *find_after(const char *from, const char *key)
{
    char k[128]; snprintf(k, sizeof k, "\"%s\"", key);
    const char *p = strstr(from, k);
    if (!p) return NULL;
    p = strchr(p + strlen(k), ':');
    return p ? p + 1 : NULL;
}
static int get_int(const char *from, const char *key, int *out)
{
    const char *p = find_after(from, key);
    if (!p) { fprintf(stderr, "sn_vision: config has no \"%s\"\n", key); return -1; }
    *out = (int)strtol(p, NULL, 10); return 0;
}
static int get_dbl(const char *from, const char *key, double *out)
{
    const char *p = find_after(from, key);
    if (!p) { fprintf(stderr, "sn_vision: config has no \"%s\"\n", key); return -1; }
    *out = strtod(p, NULL); return 0;
}

int sn_vgeom_load(sn_vgeom *g, const char *path, int llm_hidden)
{
    memset(g, 0, sizeof *g);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sn_vision: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = (char *)malloc((size_t)sz + 1);
    if (!t || fread(t, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(t); return -1; }
    t[sz] = 0; fclose(f);
    const char *V = strstr(t, "\"vision_config\"");
    if (!V) { fprintf(stderr, "sn_vision: config has no vision_config\n"); free(t); return -1; }
    int rc = 0; double mr = 4.0, ds = 0.5, eps = 1e-6;
    rc |= get_int(V, "hidden_size", &g->hidden);
    rc |= get_int(V, "num_hidden_layers", &g->layers);
    rc |= get_int(V, "num_attention_heads", &g->heads);
    rc |= get_int(V, "num_cls_tokens", &g->ncls);
    rc |= get_int(V, "num_registers", &g->nreg);
    rc |= get_int(V, "patch_size", &g->patch);
    rc |= get_int(V, "max_img_size", &g->max_img);
    rc |= get_dbl(V, "mlp_ratio", &mr);
    rc |= get_dbl(V, "layer_norm_eps", &eps);
    rc |= get_int(t, "projector_hidden_size", &g->proj_hidden);
    rc |= get_dbl(t, "downsample_ratio", &ds);
    free(t);
    if (rc) return -1;
    g->mlp = (int)(g->hidden * mr + 0.5); g->eps = (float)eps; g->llm_hidden = llm_hidden;
    g->shuffle = (int)(1.0 / ds + 0.5);
    if (g->heads <= 0 || g->hidden % g->heads || (g->hidden / g->heads) % 8 || g->hidden % 8 || g->mlp % 8 || g->shuffle < 1) {
        fprintf(stderr, "sn_vision: unsupported vision geometry (hidden %d, heads %d)\n", g->hidden, g->heads); return -1;
    }
    return 0;
}

/* ------------------------------------------------------------ kernels */

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

static float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f)); }
static float apply_act(float x, int act)
{
    if (act == 1) return gelu_erf(x);
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

/* rows of W (row-major, K floats each) times 4 tokens at once; K % 8 == 0 */
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
static float dot_avx2(const float *a, const float *b, int n)   /* n % 8 == 0 */
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
static float dot_scalar(const float *a, const float *b, int n) { float s = 0; for (int i = 0; i < n; i++) s += a[i] * b[i]; return s; }
static void axpy_scalar(float *y, float a, const float *x, int n) { for (int i = 0; i < n; i++) y[i] += a * x[i]; }

static float vdot(const float *a, const float *b, int n)
{
#ifdef SN_HAVE_X86
    if (have_avx2()) return dot_avx2(a, b, n);
#endif
    return dot_scalar(a, b, n);
}
static void vaxpy(float *y, float a, const float *x, int n)
{
#ifdef SN_HAVE_X86
    if (have_avx2()) { axpy_avx2(y, a, x, n); return; }
#endif
    axpy_scalar(y, a, x, n);
}

/* ------------------------------------------------------------ the tower */

typedef struct { sn_rd rd; float *book; float *wbuf; float *sbuf; size_t sbuf_n; const sn_tensor *booked; } vlane;

typedef struct {
    const sn_tensor *qkv, *proj, *fc1, *fc2;
    float *n1w, *n1b, *n2w, *n2b, *qkvb, *projb, *fc1b, *fc2b;
} vlayer;

struct sn_vision {
    sn_vgeom g;
    const sn_pack *pack;
    nemo_pool *pool;
    int nthreads;
    vlane *lanes;
    vlayer *L;
    const sn_tensor *pos_t, *mlp1_a, *mlp1_b;
    float *embedder;                 /* [hidden][3*patch*patch], resident */
    float *cls;                      /* [ncls+nreg][hidden] */
    float *fnw, *fnb;                /* final LayerNorm */
    float *m1n;                      /* mlp1 RMSNorm weight */
    int fail;
};

static int lane_of(int n_items, int nthreads, int begin)
{
    int lanes = nthreads < n_items ? nthreads : n_items;
    for (int l = 0; l < lanes; l++)
        if ((int)((long)l * n_items / lanes) == begin) return l;
    return 0;
}

typedef struct {
    sn_vision *v; const sn_tensor *t; const float *bias; const float *X; int N, K, R; float *Y; int act; int fail;
} gemm_ctx;

static void gemm_job(void *vc, int b, int e)
{
    gemm_ctx *c = (gemm_ctx *)vc;
    sn_vision *v = c->v;
    int nchunks = (c->R + CHUNK_ROWS - 1) / CHUNK_ROWS;
    vlane *ln = &v->lanes[lane_of(nchunks, v->nthreads, b)];
    if (!c->t->raw && ln->booked != c->t) {
        if (sn_load_book(v->pack, &ln->rd, c->t, ln->book) != 0) { c->fail = 1; return; }
        ln->booked = c->t;
    }
    for (int ch = b; ch < e; ch++) {
        int r0 = ch * CHUNK_ROWS, r1 = r0 + CHUNK_ROWS < c->R ? r0 + CHUNK_ROWS : c->R;
        if (sn_expand_book(v->pack, &ln->rd, c->t, ln->book, (uint64_t)r0 * c->K, (uint64_t)(r1 - r0) * c->K, ln->wbuf) != 0) { c->fail = 1; return; }
#ifdef SN_HAVE_X86
        if (have_avx2()) { rows_avx2(ln->wbuf, r1 - r0, c->K, c->X, c->N, c->bias, r0, c->R, c->Y, c->act); continue; }
#endif
        rows_scalar(ln->wbuf, r1 - r0, c->K, c->X, c->N, c->bias, r0, c->R, c->Y, c->act);
    }
}

int sn_vision_linear(sn_vision *v, const sn_tensor *t, const float *bias, const float *X, int N, int K, int R, float *Y, int act)
{
    if ((uint64_t)K * (uint64_t)R != t->count || (K & 7)) return -2;
    gemm_ctx c = { v, t, bias, X, N, K, R, Y, act, 0 };
    nemo_pool_for(v->pool, (R + CHUNK_ROWS - 1) / CHUNK_ROWS, gemm_job, &c);
    return c.fail ? -1 : 0;
}

/* LayerNorm per token (double accumulation), y = (x - mean) / sqrt(var + eps) * w + b */
typedef struct { const float *x; float *y; const float *w, *b; int dim; float eps; } ln_ctx;
static void ln_job(void *vc, int b, int e)
{
    ln_ctx *c = (ln_ctx *)vc;
    for (int n = b; n < e; n++) {
        const float *x = c->x + (size_t)n * c->dim; float *y = c->y + (size_t)n * c->dim;
        double m = 0, s2 = 0;
        for (int i = 0; i < c->dim; i++) m += x[i];
        m /= c->dim;
        for (int i = 0; i < c->dim; i++) { double d = x[i] - m; s2 += d * d; }
        float inv = (float)(1.0 / sqrt(s2 / c->dim + c->eps));
        for (int i = 0; i < c->dim; i++) y[i] = (float)(x[i] - m) * inv * c->w[i] + c->b[i];
    }
}
static void layer_norm(sn_vision *v, const float *x, float *y, const float *w, const float *b, int N, int dim, float eps)
{
    ln_ctx c = { x, y, w, b, dim, eps };
    nemo_pool_for(v->pool, N, ln_job, &c);
}

/* RMSNorm per token (mlp1's norm): y = x / sqrt(mean(x^2) + eps) * w */
typedef struct { const float *x; float *y; const float *w; int dim; float eps; } rms_ctx;
static void rms_job(void *vc, int b, int e)
{
    rms_ctx *c = (rms_ctx *)vc;
    for (int n = b; n < e; n++) {
        const float *x = c->x + (size_t)n * c->dim; float *y = c->y + (size_t)n * c->dim;
        double s2 = 0;
        for (int i = 0; i < c->dim; i++) s2 += (double)x[i] * x[i];
        float inv = (float)(1.0 / sqrt(s2 / c->dim + c->eps));
        for (int i = 0; i < c->dim; i++) y[i] = x[i] * inv * c->w[i];
    }
}

/* Full (non-causal) attention, fused qkv rows [q | k | v], each hidden wide, heads of hd. One query per item. */
typedef struct { sn_vision *v; const float *qkv; float *out; int N, hidden, heads, hd; } attn_ctx;
static void attn_job(void *vc, int b, int e)
{
    attn_ctx *c = (attn_ctx *)vc;
    sn_vision *v = c->v;
    vlane *ln = &v->lanes[lane_of(c->N, v->nthreads, b)];
    int H = c->hidden, hd = c->hd, N = c->N, stride = 3 * H;
    float scale = 1.0f / sqrtf((float)hd);
    float *sc = ln->sbuf;
    for (int q = b; q < e; q++) {
        for (int h = 0; h < c->heads; h++) {
            const float *qv = c->qkv + (size_t)q * stride + h * hd;
            float mx = -3.4e38f;
            for (int j = 0; j < N; j++) {
                float s = vdot(qv, c->qkv + (size_t)j * stride + H + h * hd, hd) * scale;
                sc[j] = s; if (s > mx) mx = s;
            }
            double sum = 0;
            for (int j = 0; j < N; j++) { float ex = expf(sc[j] - mx); sc[j] = ex; sum += ex; }
            float inv = (float)(1.0 / sum);
            float *o = c->out + (size_t)q * H + h * hd;
            memset(o, 0, (size_t)hd * sizeof(float));
            for (int j = 0; j < N; j++) vaxpy(o, sc[j] * inv, c->qkv + (size_t)j * stride + 2 * H + h * hd, hd);
        }
    }
}

typedef struct { float *x; const float *y; size_t n; } add_ctx;
static void add_job(void *vc, int b, int e)
{
    add_ctx *c = (add_ctx *)vc;
    size_t chunk = 4096;
    for (int i = b; i < e; i++) {
        size_t lo = (size_t)i * chunk, hi = lo + chunk < c->n ? lo + chunk : c->n;
        for (size_t k = lo; k < hi; k++) c->x[k] += c->y[k];
    }
}
static void add_inplace(sn_vision *v, float *x, const float *y, size_t n)
{
    add_ctx c = { x, y, n };
    nemo_pool_for(v->pool, (int)((n + 4095) / 4096), add_job, &c);
}

/* ------------------------------------------------------------ open / close */

static float *load_raw(const sn_pack *p, sn_rd *rd, const char *name, uint64_t want)
{
    const sn_tensor *t = sn_pack_find(p, name);
    if (!t) { fprintf(stderr, "sn_vision: pack has no %s\n", name); return NULL; }
    if (!t->raw || t->count != want) { fprintf(stderr, "sn_vision: %s is not a raw tensor of %llu weights\n", name, (unsigned long long)want); return NULL; }
    float *o = (float *)malloc(want * sizeof(float));
    if (!o || sn_expand_book(p, rd, t, NULL, 0, want, o) != 0) { free(o); fprintf(stderr, "sn_vision: cannot read %s\n", name); return NULL; }
    return o;
}
static const sn_tensor *need(const sn_pack *p, const char *name, uint64_t want)
{
    const sn_tensor *t = sn_pack_find(p, name);
    if (!t) { fprintf(stderr, "sn_vision: pack has no %s\n", name); return NULL; }
    if (t->count != want) { fprintf(stderr, "sn_vision: %s has %llu weights, expected %llu\n", name, (unsigned long long)t->count, (unsigned long long)want); return NULL; }
    return t;
}

void sn_vision_close(sn_vision *v)
{
    if (!v) return;
    if (v->lanes) for (int i = 0; i < v->nthreads; i++) { sn_rd_close(&v->lanes[i].rd); free(v->lanes[i].book); free(v->lanes[i].wbuf); free(v->lanes[i].sbuf); }
    if (v->L) for (int i = 0; i < v->g.layers; i++) {
        vlayer *l = &v->L[i];
        free(l->n1w); free(l->n1b); free(l->n2w); free(l->n2b); free(l->qkvb); free(l->projb); free(l->fc1b); free(l->fc2b);
    }
    free(v->lanes); free(v->L); free(v->embedder); free(v->cls); free(v->fnw); free(v->fnb); free(v->m1n);
    free(v);
}

sn_vision *sn_vision_open(const sn_pack *p, nemo_pool *pool, int nthreads, const sn_vgeom *g)
{
    sn_vision *v = (sn_vision *)calloc(1, sizeof *v);
    if (!v) return NULL;
    v->g = *g; v->pack = p; v->pool = pool; v->nthreads = nthreads < 1 ? 1 : nthreads;
    sn_rd mrd;
    if (sn_rd_open(p, &mrd) != 0) { free(v); return NULL; }
    int Hd = g->hidden, ps = g->patch, ok = 1;
    char nm[256];
    v->lanes = (vlane *)calloc((size_t)v->nthreads, sizeof(vlane));
    v->L = (vlayer *)calloc((size_t)g->layers, sizeof(vlayer));
    if (!v->lanes || !v->L) ok = 0;
    size_t wmax = (size_t)CHUNK_ROWS * (size_t)(g->proj_hidden > g->mlp ? g->proj_hidden : g->mlp);
    if (ok) for (int i = 0; i < v->nthreads; i++) {
        if (sn_rd_open(p, &v->lanes[i].rd) != 0) { ok = 0; break; }
        v->lanes[i].book = (float *)malloc(p->book_floats * sizeof(float));
        v->lanes[i].wbuf = (float *)malloc(wmax * sizeof(float));
        if (!v->lanes[i].book || !v->lanes[i].wbuf) { ok = 0; break; }
    }
    if (ok) {
        v->embedder = load_raw(p, &mrd, VPFX "patch_generator.embedder.weight", (uint64_t)Hd * 3 * ps * ps);
        v->cls = load_raw(p, &mrd, VPFX "patch_generator.cls_token.token", (uint64_t)(g->ncls + g->nreg) * Hd);
        v->pos_t = need(p, VPFX "patch_generator.pos_embed", (uint64_t)(g->max_img / ps) * (g->max_img / ps) * Hd);
        v->fnw = load_raw(p, &mrd, "vision_projector.vision_final_layernorm.weight", Hd);
        v->fnb = load_raw(p, &mrd, "vision_projector.vision_final_layernorm.bias", Hd);
        int cat = Hd * g->shuffle * g->shuffle;
        v->m1n = load_raw(p, &mrd, "mlp1.0.weight", (uint64_t)cat);
        v->mlp1_a = need(p, "mlp1.1.weight", (uint64_t)g->proj_hidden * cat);
        v->mlp1_b = need(p, "mlp1.3.weight", (uint64_t)g->llm_hidden * g->proj_hidden);
        if (!v->embedder || !v->cls || !v->pos_t || !v->fnw || !v->fnb || !v->m1n || !v->mlp1_a || !v->mlp1_b) ok = 0;
    }
    for (int i = 0; ok && i < g->layers; i++) {
        vlayer *l = &v->L[i];
#define LN_(field, suffix, cnt) do { snprintf(nm, sizeof nm, VPFX "blocks.%d." suffix, i); l->field = load_raw(p, &mrd, nm, (uint64_t)(cnt)); if (!l->field) ok = 0; } while (0)
#define WT_(field, suffix, cnt) do { snprintf(nm, sizeof nm, VPFX "blocks.%d." suffix, i); l->field = need(p, nm, (uint64_t)(cnt)); if (!l->field) ok = 0; } while (0)
        LN_(n1w, "norm1.weight", Hd); LN_(n1b, "norm1.bias", Hd); LN_(n2w, "norm2.weight", Hd); LN_(n2b, "norm2.bias", Hd);
        LN_(qkvb, "attn.qkv.bias", 3 * Hd); LN_(projb, "attn.proj.bias", Hd); LN_(fc1b, "mlp.fc1.bias", g->mlp); LN_(fc2b, "mlp.fc2.bias", Hd);
        WT_(qkv, "attn.qkv.weight", (uint64_t)3 * Hd * Hd); WT_(proj, "attn.proj.weight", (uint64_t)Hd * Hd);
        WT_(fc1, "mlp.fc1.weight", (uint64_t)g->mlp * Hd); WT_(fc2, "mlp.fc2.weight", (uint64_t)Hd * g->mlp);
#undef LN_
#undef WT_
    }
    sn_rd_close(&mrd);
    if (!ok) { sn_vision_close(v); return NULL; }
    return v;
}

const sn_vgeom *sn_vision_geom(const sn_vision *v) { return &v->g; }

/* ------------------------------------------------------------ embed */

typedef struct { const float *pix; float *X; int H, W, gh, gw, ps, Hd; const float *emb; const float *pos; int nprefix; } patch_ctx;

static void patch_job(void *vc, int b, int e)
{
    patch_ctx *c = (patch_ctx *)vc;
    int K = 3 * c->ps * c->ps;
    float *tmp = (float *)malloc((size_t)K * sizeof(float));
    if (!tmp) return;
    for (int t = b; t < e; t++) {
        int i = t / c->gw, j = t % c->gw;
        for (int ch = 0; ch < 3; ch++)
            for (int py = 0; py < c->ps; py++)
                memcpy(tmp + (ch * c->ps + py) * c->ps, c->pix + ((size_t)ch * c->H + (size_t)i * c->ps + py) * c->W + (size_t)j * c->ps, (size_t)c->ps * sizeof(float));
        float *x = c->X + (size_t)(c->nprefix + t) * c->Hd;
        const float *pe = c->pos + (size_t)t * c->Hd;
        for (int h = 0; h < c->Hd; h++) x[h] = vdot(c->emb + (size_t)h * K, tmp, K) + pe[h];
    }
    free(tmp);
}

/* torch bilinear, align_corners=False: src = (dst + 0.5) * in/out - 0.5, clamped at 0 */
static void bilin_axis(int in, int out, int d, int *i0, int *i1, float *lam)
{
    float scale = (float)in / (float)out;
    float s = ((float)d + 0.5f) * scale - 0.5f;
    if (s < 0) s = 0;
    int a = (int)s;
    int b = a + (a < in - 1 ? 1 : 0);
    *i0 = a; *i1 = b; *lam = s - (float)a;
}

/* position embedding for a gh x gw grid: resample the max x max table to m x m (m = max(gh, gw)), crop top-left */
static float *make_pos(sn_vision *v, int gh, int gw)
{
    const sn_vgeom *g = &v->g;
    int Hd = g->hidden, M = g->max_img / g->patch, m = gh > gw ? gh : gw;
    size_t n = (size_t)M * M * Hd;
    float *tab = (float *)malloc(n * sizeof(float));
    float *pos = (float *)malloc((size_t)gh * gw * Hd * sizeof(float));
    if (!tab || !pos) { free(tab); free(pos); return NULL; }
    if (sn_expand_book(v->pack, &v->lanes[0].rd, v->pos_t, NULL, 0, n, tab) != 0) { free(tab); free(pos); return NULL; }
    for (int i = 0; i < gh; i++) {
        int r0, r1; float lr; bilin_axis(M, m, i, &r0, &r1, &lr);
        for (int j = 0; j < gw; j++) {
            int c0, c1; float lc; bilin_axis(M, m, j, &c0, &c1, &lc);
            const float *p00 = tab + ((size_t)r0 * M + c0) * Hd, *p01 = tab + ((size_t)r0 * M + c1) * Hd;
            const float *p10 = tab + ((size_t)r1 * M + c0) * Hd, *p11 = tab + ((size_t)r1 * M + c1) * Hd;
            float *o = pos + ((size_t)i * gw + j) * Hd;
            for (int h = 0; h < Hd; h++) {
                float top = p00[h] * (1.0f - lc) + p01[h] * lc, bot = p10[h] * (1.0f - lc) + p11[h] * lc;
                o[h] = top * (1.0f - lr) + bot * lr;
            }
        }
    }
    free(tab);
    return pos;
}

int sn_vision_embed(sn_vision *v, const float *pix, int H, int W, float *out)
{
    const sn_vgeom *g = &v->g;
    int ps = g->patch, sh = g->shuffle, Hd = g->hidden;
    if (H <= 0 || W <= 0 || H % (ps * sh) || W % (ps * sh) || H > g->max_img * 8 || W > g->max_img * 8) return -2;
    int gh = H / ps, gw = W / ps, npatch = gh * gw, np = g->ncls + g->nreg, N = np + npatch;
    int rc = 0;
    size_t sN = (size_t)N * Hd;
    float *X = (float *)malloc(sN * 4), *T = (float *)malloc(sN * 4), *A = (float *)malloc(sN * 4);
    float *QKV = (float *)malloc(sN * 3 * 4), *MH = (float *)malloc((size_t)N * g->mlp * 4);
    float *pos = make_pos(v, gh, gw);
    if (!X || !T || !A || !QKV || !MH || !pos) { rc = -1; goto done; }
    for (int i = 0; i < v->nthreads; i++) {
        if (v->lanes[i].sbuf_n < (size_t)N) {
            float *nb = (float *)realloc(v->lanes[i].sbuf, (size_t)N * sizeof(float));
            if (!nb) { rc = -1; goto done; }
            v->lanes[i].sbuf = nb; v->lanes[i].sbuf_n = (size_t)N;
        }
    }
    memcpy(X, v->cls, (size_t)np * Hd * sizeof(float));
    { patch_ctx pc = { pix, X, H, W, gh, gw, ps, Hd, v->embedder, pos, np }; nemo_pool_for(v->pool, npatch, patch_job, &pc); }
    for (int li = 0; li < g->layers && !rc; li++) {
        const vlayer *l = &v->L[li];
        layer_norm(v, X, T, l->n1w, l->n1b, N, Hd, g->eps);
        if ((rc = sn_vision_linear(v, l->qkv, l->qkvb, T, N, Hd, 3 * Hd, QKV, 0)) != 0) break;
        { attn_ctx ac = { v, QKV, A, N, Hd, g->heads, Hd / g->heads }; nemo_pool_for(v->pool, N, attn_job, &ac); }
        if ((rc = sn_vision_linear(v, l->proj, l->projb, A, N, Hd, Hd, T, 0)) != 0) break;
        add_inplace(v, X, T, sN);
        layer_norm(v, X, T, l->n2w, l->n2b, N, Hd, g->eps);
        if ((rc = sn_vision_linear(v, l->fc1, l->fc1b, T, N, Hd, g->mlp, MH, 1)) != 0) break;
        if ((rc = sn_vision_linear(v, l->fc2, l->fc2b, MH, N, g->mlp, Hd, T, 0)) != 0) break;
        add_inplace(v, X, T, sN);
    }
    if (rc) goto done;
    {
        /* drop the prefix tokens, final LayerNorm, then 2x2 pixel shuffle (ps v2) */
        const float *feat = X + (size_t)np * Hd;
        layer_norm(v, feat, T, v->fnw, v->fnb, npatch, Hd, g->eps);
        int oh = gh / sh, ow = gw / sh, cat = Hd * sh * sh, Nt = oh * ow;
        float *PS = (float *)malloc((size_t)Nt * cat * 4), *PN = (float *)malloc((size_t)Nt * cat * 4);
        float *H1 = (float *)malloc((size_t)Nt * g->proj_hidden * 4);
        if (!PS || !PN || !H1) { free(PS); free(PN); free(H1); rc = -1; goto done; }
        for (int I = 0; I < oh; I++)
            for (int J = 0; J < ow; J++) {
                float *dst = PS + ((size_t)I * ow + J) * cat;
                for (int a = 0; a < sh; a++)          /* row within the group */
                    for (int b = 0; b < sh; b++)      /* column within the group */
                        memcpy(dst + (size_t)(a * sh + b) * Hd, T + ((size_t)(I * sh + a) * gw + (J * sh + b)) * Hd, (size_t)Hd * sizeof(float));
            }
        { rms_ctx rcx = { PS, PN, v->m1n, cat, 1e-5f }; nemo_pool_for(v->pool, Nt, rms_job, &rcx); }
        rc = sn_vision_linear(v, v->mlp1_a, NULL, PN, Nt, cat, g->proj_hidden, H1, 2);
        if (!rc) rc = sn_vision_linear(v, v->mlp1_b, NULL, H1, Nt, g->proj_hidden, g->llm_hidden, out, 0);
        free(PS); free(PN); free(H1);
    }
done:
    free(X); free(T); free(A); free(QKV); free(MH); free(pos);
    return rc;
}
