/* ln_sens.c - Little Nemo precision sensitivity probe, pure C.
 *
 * Scores how much each layer (or each projection) of the compressed
 * pack costs on a probe set, by putting higher-precision weights back
 * one layer or one tensor at a time and measuring the teacher-forced
 * negative log-likelihood of the gold answers.
 *
 * Weight sources per projection:
 *   B  the decoded pack (bf16 checkpoint unpacked from the .nq2 pack;
 *      what the CUDA lane's FULL kernel computes),
 *   A  the original bf16 weights,
 *   R  B plus a refinement tensor "<name>.refine" decoded from a PQ
 *      pack (--refine), i.e. what the expanded pack computes.
 * Everything that is not a layer projection (norms, conv, SSM
 * parameters, embeddings, final norm, lm_head) always comes from A, as
 * on the CUDA lane.
 *
 * Evaluation is layer-major over every probe item at once: a layer's
 * weights are read once per configuration and applied to all tokens of
 * all items (projections as one blocked GEMM, the Mamba scan and
 * attention per item), so the cost of a configuration is one pass over
 * the weights plus the arithmetic. The residual stream entering each
 * layer under the base configuration is kept, so a configuration that
 * first differs at layer k only recomputes layers k..55.
 *
 * Determinism: every output element is one fixed-order dot product
 * (8-lane fused multiply-add partials, fixed-order horizontal sum);
 * threads only split rows or items, so results are bit-identical at
 * any thread count (--hash prints a digest to check that).
 *
 * Usage:
 *   ln_sens --ref DIR --pack-ckpt DIR --tokenizer FILE --probe FILE
 *           --plan FILE --out FILE [--detail FILE] [--refine PACK.nq2]
 *           [--set tune|hold|all] [--threads N] [--layers N]
 *           [--max-items N] [--pause-file PATH] [--hash]
 *
 * Plan file: one configuration per line, "name spec[+spec...]", where a
 * spec is base, ref, L<n> (layer n from A), L<n>.<leaf> (one projection
 * from A), R<n> or R<n>.<leaf> (refined). Leaves: in_proj, out_proj,
 * up_proj, down_proj, q_proj, k_proj, v_proj, o_proj. The first line
 * must be the base configuration.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "attn.h"
#include "forward.h"
#include "loader.h"
#include "mamba.h"
#include "nn.h"
#include "pool.h"
#include "tokenizer.h"

#define H NEMO_HIDDEN
#define NL NEMO_LAYERS
#define VOCAB NEMO_VOCAB
#define PROJ 22656
#define INNER 10240
#define MLPD 15680
#define NQ 40
#define NKV 8
#define AD 128
#define EPS 1e-5f
#define MAXITEMS 256
#define MAXTOK 128

/* ---------------- small utilities ---------------- */

static double now_s(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

static void sleep_s(int s) {
#ifdef _WIN32
    Sleep((DWORD)s * 1000);
#else
    sleep((unsigned)s);
#endif
}

static uint64_t fnv(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
#define FNV_SEED 14695981039346656037ull

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "ln_sens: out of memory (%zu bytes)\n", n); exit(3); }
    return p;
}

static const char *g_pause = NULL;
static void pause_gate(void) {
    if (!g_pause) return;
    int said = 0;
    for (;;) {
        FILE *f = fopen(g_pause, "rb");
        if (!f) break;
        fclose(f);
        if (!said) { fprintf(stderr, "ln_sens: paused (%s present)\n", g_pause); said = 1; }
        sleep_s(30);
    }
    if (said) fprintf(stderr, "ln_sens: resumed\n");
}

/* ---------------- blocked GEMM: Y[t][r] = sum_c W[r][c] X[t][c] -------- */

typedef struct {
    const float *W, *X;
    float *Y;
    int R, C, T;
} gemm_ctx;

static inline float hsum8(__m256 v) {
    float a[8];
    _mm256_storeu_ps(a, v);
    return ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]));
}

/* One row against up to four tokens; each (row, token) sum is its own
 * accumulator, so grouping never changes a result. */
static void dot_row(const float *w, const float *X, int C, int t0, int nt,
                    float *out) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const float *x0 = X + (size_t)t0 * C;
    const float *x1 = x0 + C, *x2 = x1 + C, *x3 = x2 + C;
    if (nt == 4) {
        for (int c = 0; c < C; c += 8) {
            __m256 wv = _mm256_loadu_ps(w + c);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x0 + c), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x1 + c), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x2 + c), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(x3 + c), a3);
        }
        out[0] = hsum8(a0); out[1] = hsum8(a1); out[2] = hsum8(a2); out[3] = hsum8(a3);
        return;
    }
    for (int k = 0; k < nt; k++) {
        const float *xk = X + (size_t)(t0 + k) * C;
        __m256 a = _mm256_setzero_ps();
        for (int c = 0; c < C; c += 8)
            a = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(xk + c), a);
        out[k] = hsum8(a);
    }
}

#define TB 32
static void gemm_chunk(void *vctx, int begin, int end) {
    gemm_ctx *g = (gemm_ctx *)vctx;
    float o[4];
    for (int tb = 0; tb < g->T; tb += TB) {
        int te = tb + TB < g->T ? tb + TB : g->T;
        for (int r = begin; r < end; r++) {
            const float *w = g->W + (size_t)r * g->C;
            for (int t = tb; t < te; t += 4) {
                int nt = te - t < 4 ? te - t : 4;
                dot_row(w, g->X, g->C, t, nt, o);
                for (int k = 0; k < nt; k++) g->Y[(size_t)(t + k) * g->R + r] = o[k];
            }
        }
    }
}

static nemo_pool g_pool;
static void gemm(const float *W, const float *X, float *Y, int R, int C, int T) {
    gemm_ctx g = {W, X, Y, R, C, T};
    nemo_pool_for(&g_pool, R, gemm_chunk, &g);
}

/* ---------------- refinement pack reader (v3 PQ d4) ---------------- */

typedef struct {
    FILE *s;
    uint64_t header, nblocks;
    int nt;
    char (*name)[256];
    uint64_t *b0, *nb;
    FILE *bk;
    uint64_t book_hdr;
} rpack;

static int rd32(FILE *f, uint32_t *v) { uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return -1; *v = b[0] | (b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); return 0; }
static int rd64(FILE *f, uint64_t *v) { uint32_t lo, hi; if (rd32(f, &lo) || rd32(f, &hi)) return -1; *v = lo | ((uint64_t)hi << 32); return 0; }

static int rpack_open(rpack *k, const char *path) {
    memset(k, 0, sizeof *k);
    k->s = fopen(path, "rb");
    if (!k->s) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char mg[4]; uint32_t ver, block, entries, nseg, flags, pq; uint64_t total, gen, sg;
    if (fread(mg, 1, 4, k->s) != 4 || memcmp(mg, "NQ2\0", 4) || rd32(k->s, &ver) || ver != 3
        || rd32(k->s, &block) || rd32(k->s, &entries) || rd64(k->s, &total) || rd64(k->s, &k->nblocks)
        || rd32(k->s, &nseg) || block != 256 || entries != 256) return -1;
    for (uint32_t i = 0; i < nseg; i++) if (rd64(k->s, &sg)) return -1;
    if (rd64(k->s, &gen) || rd32(k->s, &flags) || flags != 1u || rd32(k->s, &pq) || pq != 4) {
        fprintf(stderr, "%s: need a PQ d4 per-tensor v3 pack\n", path); return -1;
    }
    k->header = (uint64_t)ftello(k->s);
    char mp[1200]; snprintf(mp, sizeof mp, "%s", path);
    size_t L = strlen(mp);
    if (L < 5 || strcmp(mp + L - 4, ".nq2")) return -1;
    strcpy(mp + L - 4, ".nq2manifest");
    FILE *m = fopen(mp, "r");
    if (!m) { fprintf(stderr, "manifest %s missing\n", mp); return -1; }
    char line[1200]; unsigned long long mgen; int nt;
    if (!fgets(line, sizeof line, m) || !fgets(line, sizeof line, m) || sscanf(line, "generation %llx", &mgen) != 1
        || mgen != gen || !fgets(line, sizeof line, m) || sscanf(line, "tensors %d", &nt) != 1 || nt != (int)nseg) {
        fclose(m); fprintf(stderr, "manifest does not describe %s\n", path); return -1;
    }
    k->nt = nt;
    k->name = xmalloc(sizeof(*k->name) * (size_t)nt);
    k->b0 = xmalloc(8 * (size_t)nt); k->nb = xmalloc(8 * (size_t)nt);
    for (int i = 0; i < nt; i++) {
        unsigned long long b0, nb, sd, bd; unsigned blk; char kind[16];
        if (!fgets(line, sizeof line, m) || sscanf(line, "tensor %255s %llu %llu %u %llx %llx %15s", k->name[i], &b0, &nb, &blk, &sd, &bd, kind) != 7) {
            fclose(m); return -1;
        }
        k->b0[i] = b0; k->nb[i] = nb;
    }
    fclose(m);
    strcpy(mp + L - 4, ".nq2book");
    k->bk = fopen(mp, "rb");
    if (!k->bk) { fprintf(stderr, "book %s missing\n", mp); return -1; }
    k->book_hdr = 4 + 4 + 4 + 4 + 4 + 8 + 4;
    return 0;
}

/* Decode tensor `name` and add it into dst[count] (bf16-rounded decode,
 * exactly the FULL kernel's weight). Returns 0, 1 if absent, -1 on error. */
static float bfr(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) u = 0x7fc00000u;
    else { u += 0x7fffu + ((u >> 16) & 1u); u &= 0xffff0000u; }
    memcpy(&f, &u, 4); return f;
}
static float h2f(uint16_t h) {
    uint32_t sgn = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, m = h & 1023u, u;
    if (e == 0) { float f = (float)m * 5.9604644775390625e-8f; memcpy(&u, &f, 4); u |= sgn; }
    else if (e == 31) u = sgn | 0x7f800000u | (m << 13);
    else u = sgn | ((e + 112u) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static int rpack_add(rpack *k, const char *name, float *dst, uint64_t count) {
    int i = -1;
    for (int j = 0; j < k->nt; j++) if (!strcmp(k->name[j], name)) { i = j; break; }
    if (i < 0) return 1;
    uint64_t nb = k->nb[i];
    if (nb * 256 != count) { fprintf(stderr, "%s: refine geometry mismatch\n", name); return -1; }
    uint8_t *idx = xmalloc((size_t)nb * 64);
    uint16_t *mn = xmalloc((size_t)nb * 2), *sc = xmalloc((size_t)nb * 2);
    float *book = xmalloc(256 * 256 * sizeof(float));
    int ok = fseeko(k->s, (off_t)(k->header + k->b0[i] * 64), SEEK_SET) == 0 && fread(idx, 1, nb * 64, k->s) == nb * 64
          && fseeko(k->s, (off_t)(k->header + k->nblocks * 64 + 2 * k->b0[i]), SEEK_SET) == 0 && fread(mn, 2, nb, k->s) == nb
          && fseeko(k->s, (off_t)(k->header + k->nblocks * 66 + 2 * k->b0[i]), SEEK_SET) == 0 && fread(sc, 2, nb, k->s) == nb
          && fseeko(k->bk, (off_t)(k->book_hdr + (uint64_t)i * 256 * 256 * 4), SEEK_SET) == 0 && fread(book, 4, 65536, k->bk) == 65536;
    if (ok) {
        for (uint64_t b = 0; b < nb; b++) {
            float m = h2f(mn[b]), c = h2f(sc[b]);
            for (int sub = 0; sub < 64; sub++) {
                const float *q = book + ((size_t)sub * 256 + idx[b * 64 + sub]) * 4;
                float *d = dst + b * 256 + sub * 4;
                for (int t = 0; t < 4; t++) { volatile float pr = c * q[t]; volatile float s = m + pr; d[t] += bfr(s); }
            }
        }
    }
    free(idx); free(mn); free(sc); free(book);
    return ok ? 0 : -1;
}

/* ---------------- configuration ---------------- */

static const char *leaf_of(char t, int j) {
    if (t == 'M') return j == 0 ? "in_proj" : j == 1 ? "out_proj" : NULL;
    if (t == '-') return j == 0 ? "up_proj" : j == 1 ? "down_proj" : NULL;
    static const char *a[4] = {"q_proj", "k_proj", "v_proj", "o_proj"};
    return j < 4 ? a[j] : NULL;
}
static int nslots(char t) { return t == '*' ? 4 : 2; }

typedef struct { char name[96]; unsigned char src[NL][4]; } config;   /* 0 B, 1 A, 2 R */

static int parse_spec(config *c, const char *spec) {
    if (!strcmp(spec, "base")) return 0;
    if (!strcmp(spec, "ref")) {
        for (int l = 0; l < NL; l++) for (int j = 0; j < 4; j++) c->src[l][j] = 1;
        return 0;
    }
    int code;
    if (spec[0] == 'L') code = 1; else if (spec[0] == 'R') code = 2; else return -1;
    char *e; long l = strtol(spec + 1, &e, 10);
    if (e == spec + 1 || l < 0 || l >= NL) return -1;
    char t = nemo_layer_type((int)l);
    if (*e == 0) { for (int j = 0; j < nslots(t); j++) c->src[l][j] = (unsigned char)code; return 0; }
    if (*e != '.') return -1;
    for (int j = 0; j < nslots(t); j++) if (!strcmp(e + 1, leaf_of(t, j))) { c->src[l][j] = (unsigned char)code; return 0; }
    return -1;
}

/* ---------------- probe items ---------------- */

typedef struct {
    char id[32], set[8], cat[24], answer[96];
    int n, ans0, nans;         /* tokens, first answer token index, answer length */
    int ids[MAXTOK];
    int t0;                    /* offset into the global token list */
} item;

static item g_items[MAXITEMS];
static int g_nitems, g_T;

/* ---------------- model pieces ---------------- */

static nemo_checkpoint g_A, g_B;
static rpack g_R; static int g_have_R;
static int g_layers = NL;

static float *load_f32_from(const nemo_checkpoint *cp, const char *name, uint64_t *count) {
    const nemo_tensor_info *t = nemo_find(cp, name);
    if (!t) { fprintf(stderr, "tensor %s missing\n", name); exit(4); }
    float *f = nemo_load_f32(cp, t);
    if (!f) { fprintf(stderr, "tensor %s failed to load\n", name); exit(4); }
    if (count) *count = t->count;
    return f;
}

/* A projection in fp32 from its configured source. */
static float *load_proj(int l, int j, int src) {
    char nm[160];
    snprintf(nm, sizeof nm, "backbone.layers.%d.mixer.%s.weight", l, leaf_of(nemo_layer_type(l), j));
    uint64_t cnt;
    if (src == 1) return load_f32_from(&g_A, nm, &cnt);
    float *w = load_f32_from(&g_B, nm, &cnt);
    if (src == 2) {
        char rn[176]; snprintf(rn, sizeof rn, "%s.refine", nm);
        int rc = g_have_R ? rpack_add(&g_R, rn, w, cnt) : 1;
        if (rc != 0) { fprintf(stderr, "refinement %s %s\n", rn, rc > 0 ? "not in the refine pack" : "failed"); exit(4); }
    }
    return w;
}

typedef struct {
    float *P, *Inner, *Q, *K, *V, *Mid, *Hn, *Yb;
    nemo_mamba_state *ms;     /* one per lane slot */
    nemo_attn_cache *ac;
    int nslot;
    /* per-layer small tensors */
    float *conv_w, *conv_b, *a_log, *d, *dt_bias, *gnorm_w;
    int layer, fail;
} work;
static work g_w;

static void mamba_items(void *ctx, int begin, int end) {
    work *w = (work *)ctx;
    for (int s = begin; s < end; s++) {
        nemo_mamba_state *st = &w->ms[s];
        for (int i = s; i < g_nitems; i += w->nslot) {
            memset(st->ssm_state, 0, (size_t)st->heads * st->head_dim * st->state_size * sizeof(float));
            memset(st->conv_win, 0, (size_t)st->conv_dim * 4 * sizeof(float));
            item *it = &g_items[i];
            for (int t = 0; t < it->n; t++) {
                size_t g = (size_t)(it->t0 + t);
                if (nemo_mamba_decode(st, w->P + g * PROJ, 0, w->conv_w, w->conv_b, w->a_log, w->d,
                                      w->dt_bias, w->gnorm_w, 8, EPS, w->Inner + g * INNER) != 0) w->fail = 1;
            }
        }
    }
}
static void attn_items(void *ctx, int begin, int end) {
    work *w = (work *)ctx;
    for (int s = begin; s < end; s++) {
        nemo_attn_cache *c = &w->ac[s];
        for (int i = s; i < g_nitems; i += w->nslot) {
            c->len = 0;
            item *it = &g_items[i];
            for (int t = 0; t < it->n; t++) {
                size_t g = (size_t)(it->t0 + t);
                if (nemo_attn_push(c, w->K + g * NKV * AD, w->V + g * NKV * AD) != 0
                    || nemo_attn_decode(c, w->Q + g * NQ * AD, w->Inner + g * NQ * AD) != 0) w->fail = 1;
            }
        }
    }
}

typedef struct { const float *X; float *Hn; const float *wn; } norm_ctx;
static void norm_chunk(void *ctx, int b, int e) {
    norm_ctx *n = (norm_ctx *)ctx;
    for (int t = b; t < e; t++) nemo_rmsnorm(n->X + (size_t)t * H, n->wn, n->Hn + (size_t)t * H, H, EPS);
}

/* One layer over every token of every item: X += mixer(norm(X)). */
static void run_layer(int l, const unsigned char *src, float *X) {
    work *w = &g_w;
    char t = nemo_layer_type(l), nm[160];
    int T = g_T;
    snprintf(nm, sizeof nm, "backbone.layers.%d.norm.weight", l);
    float *wn = load_f32_from(&g_A, nm, NULL);
    norm_ctx nc = {X, w->Hn, wn};
    nemo_pool_for(&g_pool, T, norm_chunk, &nc);
    free(wn);
    w->fail = 0;
    if (t == 'M') {
#define SMALL(field, leaf) snprintf(nm, sizeof nm, "backbone.layers.%d.mixer.%s", l, leaf), w->field = load_f32_from(&g_A, nm, NULL)
        SMALL(conv_w, "conv1d.weight"); SMALL(conv_b, "conv1d.bias"); SMALL(a_log, "A_log");
        SMALL(d, "D"); SMALL(dt_bias, "dt_bias"); SMALL(gnorm_w, "norm.weight");
#undef SMALL
        float *Win = load_proj(l, 0, src[0]);
        gemm(Win, w->Hn, w->P, PROJ, H, T);
        free(Win);
        nemo_pool_for(&g_pool, w->nslot, mamba_items, w);
        float *Wout = load_proj(l, 1, src[1]);
        gemm(Wout, w->Inner, w->Yb, H, INNER, T);
        free(Wout);
        free(w->conv_w); free(w->conv_b); free(w->a_log); free(w->d); free(w->dt_bias); free(w->gnorm_w);
    } else if (t == '-') {
        float *Wup = load_proj(l, 0, src[0]);
        gemm(Wup, w->Hn, w->Mid, MLPD, H, T);
        free(Wup);
        for (size_t i = 0; i < (size_t)T * MLPD; i++) w->Mid[i] = nemo_relu2(w->Mid[i]);
        float *Wdn = load_proj(l, 1, src[1]);
        gemm(Wdn, w->Mid, w->Yb, H, MLPD, T);
        free(Wdn);
    } else {
        float *Wq = load_proj(l, 0, src[0]); gemm(Wq, w->Hn, w->Q, NQ * AD, H, T); free(Wq);
        float *Wk = load_proj(l, 1, src[1]); gemm(Wk, w->Hn, w->K, NKV * AD, H, T); free(Wk);
        float *Wv = load_proj(l, 2, src[2]); gemm(Wv, w->Hn, w->V, NKV * AD, H, T); free(Wv);
        nemo_pool_for(&g_pool, w->nslot, attn_items, w);
        float *Wo = load_proj(l, 3, src[3]); gemm(Wo, w->Inner, w->Yb, H, NQ * AD, T); free(Wo);
    }
    if (w->fail) { fprintf(stderr, "layer %d: mixer failed\n", l); exit(5); }
    for (size_t i = 0; i < (size_t)T * H; i++) X[i] += w->Yb[i];
}

/* ---------------- scoring ---------------- */

static uint16_t *g_head;      /* lm_head bf16, resident */
static float *g_normf;

typedef struct { double nll; int first_ok, all_ok, pred0; } iscore;

typedef struct { const float *Hn; float *L; int P, r0, rn; const uint16_t *W; float *Wf; } head_ctx;

static void score(const float *X, iscore *sc, uint64_t *hash) {
    int P = 0;
    for (int i = 0; i < g_nitems; i++) P += g_items[i].nans;
    float *Hn = xmalloc((size_t)P * H * 4);
    int p = 0;
    for (int i = 0; i < g_nitems; i++) {
        item *it = &g_items[i];
        for (int j = 0; j < it->nans; j++, p++)
            nemo_rmsnorm(X + (size_t)(it->t0 + it->ans0 - 1 + j) * H, g_normf, Hn + (size_t)p * H, H, EPS);
    }
    float *Lg = xmalloc((size_t)P * VOCAB * 4);
    const int CH = 4096;
    float *Wf = xmalloc((size_t)CH * H * 4), *Lc = xmalloc((size_t)P * CH * 4);
    for (int r0 = 0; r0 < VOCAB; r0 += CH) {
        for (size_t i = 0; i < (size_t)CH * H; i++) Wf[i] = nemo_bf16_to_f32(g_head[(size_t)r0 * H + i]);
        gemm(Wf, Hn, Lc, CH, H, P);
        for (int q = 0; q < P; q++) memcpy(Lg + (size_t)q * VOCAB + r0, Lc + (size_t)q * CH, CH * 4);
    }
    free(Wf); free(Lc);
    p = 0;
    uint64_t h = FNV_SEED;
    h = fnv(X, (size_t)g_T * H * 4, h);
    for (int i = 0; i < g_nitems; i++) {
        item *it = &g_items[i];
        sc[i].nll = 0; sc[i].first_ok = 0; sc[i].all_ok = 1;
        for (int j = 0; j < it->nans; j++, p++) {
            const float *lg = Lg + (size_t)p * VOCAB;
            int am = 0; float mx = lg[0];
            for (int v = 1; v < VOCAB; v++) if (lg[v] > mx) { mx = lg[v]; am = v; }
            double s = 0; for (int v = 0; v < VOCAB; v++) s += exp((double)lg[v] - mx);
            int gold = it->ids[it->ans0 + j];
            sc[i].nll += (log(s) + mx) - lg[gold];
            if (j == 0) { sc[i].first_ok = am == gold; sc[i].pred0 = am; }
            if (am != gold) sc[i].all_ok = 0;
        }
    }
    h = fnv(Lg, (size_t)P * VOCAB * 4, h);
    *hash = h;
    free(Lg); free(Hn);
}

/* ---------------- main ---------------- */

static int load_probe(const char *path, const char *set, nemo_tokenizer *tk, int max_items) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open probe %s\n", path); return -1; }
    char line[2048];
    g_nitems = 0; g_T = 0;
    while (fgets(line, sizeof line, f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char *fld[5]; int k = 0; char *s = line;
        for (; k < 5; k++) { fld[k] = s; char *tab = strchr(s, '\t'); if (!tab) { k++; break; } *tab = 0; s = tab + 1; }
        if (k < 5) { fprintf(stderr, "probe line needs 5 fields: %s\n", line); fclose(f); return -1; }
        if (strcmp(set, "all") && strcmp(set, fld[1])) continue;
        if (g_nitems >= MAXITEMS || (max_items > 0 && g_nitems >= max_items)) break;
        item *it = &g_items[g_nitems];
        snprintf(it->id, sizeof it->id, "%s", fld[0]);
        snprintf(it->set, sizeof it->set, "%s", fld[1]);
        snprintf(it->cat, sizeof it->cat, "%s", fld[2]);
        snprintf(it->answer, sizeof it->answer, "%s", fld[4]);
        char tpl[1024], full[1200];
        /* the engine's first-turn chat template, thinking off */
        snprintf(tpl, sizeof tpl, "<SPECIAL_10>System\n\n<SPECIAL_11>User\n%s\n<SPECIAL_11>Assistant\n<think></think>", fld[3]);
        snprintf(full, sizeof full, "%s%s", tpl, fld[4]);
        int a = nemo_tokenizer_encode(tk, tpl, it->ids, MAXTOK);
        int b = nemo_tokenizer_encode(tk, full, it->ids, MAXTOK);
        if (a < 1 || b <= a) { fprintf(stderr, "item %s does not tokenize cleanly\n", it->id); fclose(f); return -1; }
        int chk[MAXTOK]; nemo_tokenizer_encode(tk, tpl, chk, MAXTOK);
        if (memcmp(chk, it->ids, (size_t)a * sizeof(int))) { fprintf(stderr, "item %s: answer changes the prompt tokens\n", it->id); fclose(f); return -1; }
        it->n = b; it->ans0 = a; it->nans = b - a; it->t0 = g_T;
        g_T += b;
        g_nitems++;
    }
    fclose(f);
    return g_nitems > 0 ? 0 : -1;
}

int main(int argc, char **argv) {
    const char *ref = NULL, *pck = NULL, *tokp = NULL, *probe = NULL, *plan = NULL, *out = NULL, *detail = NULL, *refine = NULL, *set = "tune";
    int threads = 0, max_items = 0, want_hash = 0;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define ARG(flag, var) else if (!strcmp(a, flag) && i + 1 < argc) var = argv[++i]
        if (0) {}
        ARG("--ref", ref); ARG("--pack-ckpt", pck); ARG("--tokenizer", tokp); ARG("--probe", probe);
        ARG("--plan", plan); ARG("--out", out); ARG("--detail", detail); ARG("--refine", refine);
        ARG("--set", set); ARG("--pause-file", g_pause);
        else if (!strcmp(a, "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--layers") && i + 1 < argc) g_layers = atoi(argv[++i]);
        else if (!strcmp(a, "--max-items") && i + 1 < argc) max_items = atoi(argv[++i]);
        else if (!strcmp(a, "--hash")) want_hash = 1;
        else { fprintf(stderr, "ln_sens: unknown argument %s\n", a); return 2; }
#undef ARG
    }
    if (!ref || !pck || !tokp || !probe || !plan || !out) {
        fprintf(stderr, "usage: ln_sens --ref DIR --pack-ckpt DIR --tokenizer FILE --probe FILE --plan FILE --out FILE [--detail FILE] [--refine PACK.nq2] [--set tune|hold|all] [--threads N] [--layers N] [--max-items N] [--pause-file PATH] [--hash]\n");
        return 2;
    }
    if (g_layers < 1 || g_layers > NL) g_layers = NL;
    if (threads <= 0) threads = nemo_cores();
    if (nemo_pool_init(&g_pool, threads) != 0) { fprintf(stderr, "pool init failed\n"); return 1; }
    threads = nemo_pool_threads(&g_pool);
    nemo_tokenizer *tk = nemo_tokenizer_load(tokp);
    if (!tk) { fprintf(stderr, "tokenizer load failed\n"); return 1; }
    if (load_probe(probe, set, tk, max_items)) return 1;
    if (nemo_checkpoint_open(&g_A, ref) || nemo_checkpoint_open(&g_B, pck)) { fprintf(stderr, "checkpoint open failed\n"); return 1; }
    if (refine) { if (rpack_open(&g_R, refine)) return 1; g_have_R = 1; }

    /* plan */
    FILE *pf = fopen(plan, "r");
    if (!pf) { fprintf(stderr, "cannot open plan %s\n", plan); return 1; }
    static config cfg[512]; int nc = 0; char line[4096];
    while (fgets(line, sizeof line, pf) && nc < 512) {
        char nm[96], specs[3900];
        if (line[0] == '#' || sscanf(line, "%95s %3899s", nm, specs) != 2) continue;
        memset(&cfg[nc], 0, sizeof cfg[nc]);
        snprintf(cfg[nc].name, sizeof cfg[nc].name, "%s", nm);
        for (char *s = strtok(specs, "+"); s; s = strtok(NULL, "+"))
            if (parse_spec(&cfg[nc], s)) { fprintf(stderr, "bad spec %s in %s\n", s, nm); return 2; }
        nc++;
    }
    fclose(pf);
    if (nc == 0) { fprintf(stderr, "empty plan\n"); return 2; }
    for (int l = 0; l < NL; l++) for (int j = 0; j < 4; j++)
        if (cfg[0].src[l][j]) { fprintf(stderr, "the first plan line must be the base configuration\n"); return 2; }

    fprintf(stderr, "ln_sens: %d items (%s), %d tokens, %d configurations, %d threads, %d layers\n",
            g_nitems, set, g_T, nc, threads, g_layers);

    /* resident pieces */
    { uint64_t c; g_normf = load_f32_from(&g_A, "backbone.norm_f.weight", &c);
      const nemo_tensor_info *t = nemo_find(&g_A, "lm_head.weight");
      if (!t) { fprintf(stderr, "lm_head missing\n"); return 1; }
      g_head = nemo_load_bf16(&g_A, t);
      if (!g_head) { fprintf(stderr, "lm_head load failed\n"); return 1; } }
    work *w = &g_w;
    int T = g_T;
    w->P = xmalloc((size_t)T * PROJ * 4); w->Inner = xmalloc((size_t)T * INNER * 4);
    w->Q = xmalloc((size_t)T * NQ * AD * 4); w->K = xmalloc((size_t)T * NKV * AD * 4); w->V = xmalloc((size_t)T * NKV * AD * 4);
    w->Mid = xmalloc((size_t)T * MLPD * 4); w->Hn = xmalloc((size_t)T * H * 4); w->Yb = xmalloc((size_t)T * H * 4);
    w->nslot = threads;
    w->ms = xmalloc(sizeof(nemo_mamba_state) * (size_t)w->nslot);
    w->ac = xmalloc(sizeof(nemo_attn_cache) * (size_t)w->nslot);
    for (int s = 0; s < w->nslot; s++)
        if (nemo_mamba_state_init(&w->ms[s], 128, 80, 128, 8) || nemo_attn_cache_init(&w->ac[s], NQ, NKV, AD, MAXTOK)) {
            fprintf(stderr, "state init failed\n"); return 1;
        }

    /* embeddings -> cache[0] */
    float **cache = xmalloc(sizeof(float *) * (NL + 1));
    for (int l = 0; l <= NL; l++) cache[l] = NULL;
    cache[0] = xmalloc((size_t)T * H * 4);
    {
        const nemo_tensor_info *e = nemo_find(&g_A, "backbone.embeddings.weight");
        FILE *f = e ? fopen(e->shard_path, "rb") : NULL;
        if (!f) { fprintf(stderr, "embeddings missing\n"); return 1; }
        uint16_t row[H];
        for (int i = 0; i < g_nitems; i++)
            for (int t = 0; t < g_items[i].n; t++) {
                if (fseeko(f, (off_t)(e->data_off + (uint64_t)g_items[i].ids[t] * H * 2), SEEK_SET) || fread(row, 2, H, f) != H) {
                    fprintf(stderr, "embedding read failed\n"); return 1;
                }
                float *x = cache[0] + (size_t)(g_items[i].t0 + t) * H;
                for (int k = 0; k < H; k++) x[k] = nemo_bf16_to_f32(row[k]);
            }
        fclose(f);
    }

    FILE *of = fopen(out, "a"), *df = detail ? fopen(detail, "a") : NULL;
    if (!of || (detail && !df)) { fprintf(stderr, "cannot open outputs\n"); return 1; }
    float *X = xmalloc((size_t)T * H * 4);
    iscore *sc = xmalloc(sizeof(iscore) * (size_t)g_nitems);
    for (int c = 0; c < nc; c++) {
        double t0 = now_s();
        int start = 0;
        if (c > 0) {
            start = g_layers;
            for (int l = 0; l < g_layers && start == g_layers; l++)
                for (int j = 0; j < 4; j++) if (cfg[c].src[l][j]) { start = l; break; }
        }
        memcpy(X, cache[start], (size_t)T * H * 4);
        for (int l = start; l < g_layers; l++) {
            pause_gate();
            run_layer(l, cfg[c].src[l], X);
            if (c == 0) { cache[l + 1] = xmalloc((size_t)T * H * 4); memcpy(cache[l + 1], X, (size_t)T * H * 4); }
        }
        uint64_t hh;
        score(X, sc, &hh);
        double nll[2] = {0, 0}; int ntok[2] = {0, 0}, f1[2] = {0, 0}, all[2] = {0, 0}, ni[2] = {0, 0};
        for (int i = 0; i < g_nitems; i++) {
            int s = strcmp(g_items[i].set, "hold") == 0;
            nll[s] += sc[i].nll; ntok[s] += g_items[i].nans; f1[s] += sc[i].first_ok; all[s] += sc[i].all_ok; ni[s]++;
            if (df) {
                char *p0 = nemo_tokenizer_decode(tk, &sc[i].pred0, 1, 0);
                fprintf(df, "%s\t%s\t%s\t%s\t%.6f\t%d\t%d\t%s\n", cfg[c].name, g_items[i].id, g_items[i].set, g_items[i].cat,
                        sc[i].nll, sc[i].first_ok, sc[i].all_ok, p0 ? p0 : "?");
                free(p0);
            }
        }
        double dt = now_s() - t0;
        fprintf(of, "%s\tstart=%d\ttune_nll=%.6f\ttune_tok=%d\ttune_first=%d/%d\ttune_all=%d/%d\thold_nll=%.6f\thold_tok=%d\thold_first=%d/%d\thold_all=%d/%d\tsec=%.1f\thash=%016llx\n",
                cfg[c].name, start, nll[0], ntok[0], f1[0], ni[0], all[0], ni[0], nll[1], ntok[1], f1[1], ni[1], all[1], ni[1], dt,
                (unsigned long long)hh);
        fflush(of); if (df) fflush(df);
        fprintf(stderr, "%-20s start %2d  tune nll %9.4f first %d/%d all %d/%d  hold nll %9.4f first %d/%d  %.1f s%s%016llx\n",
                cfg[c].name, start, nll[0], f1[0], ni[0], all[0], ni[0], nll[1], f1[1], ni[1], dt, want_hash ? "  hash " : "  ", (unsigned long long)hh);
    }
    fclose(of); if (df) fclose(df);
    nemo_pool_shutdown(&g_pool);
    return 0;
}
