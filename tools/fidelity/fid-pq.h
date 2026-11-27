/* fid-pq.h ... fidelity lab: framed product quantization of one
 * weight matrix, weight-fit or output-fit.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27, for the
 * fidelity roadmap (docs/ROADMAP.md). Pure C99, zero mutexes,
 * results independent of the thread count.
 *
 * Representation (the same family as nq2-quantize --pq D --norm rms):
 *   - each row is cut into frame blocks of 256 weights; a block
 *     carries one float16 scale s;
 *   - each block is cut into 256/d subvectors; subvector position
 *     p in the block has its own book of K entries of dim d
 *     (float16), so a block costs 256/d indices of log2 K bits;
 *   - optional residual stage (method 3): a second book per
 *     position with K2 entries; the weight is s * (c1 + c2);
 *   - books may come from outside (method 4, shared across layers).
 *
 * What changes between methods is how books, indices and scales are
 * chosen:
 *   weight fit (today's recipe): k-means on the normalized
 *     subvectors, nearest-entry assignment, scale = block RMS;
 *   output fit (method 1): the objective is the layer-output error
 *     tr((W - Q) H (W - Q)^T) with H = E[x x^T] from calibration:
 *       a. books from k-means weighted by s^2 * diag(H);
 *       b. assignment with error feedback (GPTQ / GPTVQ): subvectors
 *          are fixed left to right, each choice minimizes
 *          ||(w - q) A||^2 with A = U_gg^-1, and its error is pushed
 *          onto the unquantized columns through the inverse-Hessian
 *          factor U;
 *       c. books refit by conjugate gradients on the full quadratic
 *          output objective (AQLM-style codebook update);
 *       d. scales refit per row by solving the small normal
 *          equations of the same objective.
 */
#ifndef FID_PQ_H
#define FID_PQ_H
#include "fid-linalg.h"

#define FID_BLOCK 256

typedef struct {
    int d, K, K2;          /* K2 = 0: one stage */
    int fit_act;           /* k-means weights s^2 * diag(H) */
    int gptq;              /* error-feedback assignment */
    int refit;             /* CG iterations for the books, 0 = off */
    int scale_refit;
    int rounds;            /* assignment + refit rounds */
    int km_iters;
    int64_t sample;        /* blocks sampled for k-means */
    double damp;
    uint64_t seed;
    const float *ext_b1;   /* external books (method 4), or NULL */
    const float *ext_b2;
    int learn_only;        /* return after the books are learned */
} PqCfg;

typedef struct {
    int64_t r, c, nb, nsub, ng;   /* rows, cols, blocks/row, subs/block, groups/row */
    int d, K, K2;
    float *b1, *b2;        /* nsub*K*d, nsub*K2*d (float16-exact after finish) */
    float *s;              /* r*nb, float16-exact */
    uint16_t *k1, *k2;     /* r*ng */
} PqState;

typedef struct {
    double wrel;           /* ||W-Q|| / ||W|| */
    double orel;           /* tr(dW H dW^T) / tr(W H W^T), undamped H */
    double orel_stage[8];  /* output error after each stage (damped H), -1 = n/a */
    double bits;           /* total bits for this tensor */
    double secs;
} PqStats;

static int pq_ilog2(int K) { int b = 0; while ((1 << b) < K) b++; return b; }

static double pq_bits(int64_t r, int64_t c, int d, int K, int K2, int books_owned)
{
    int64_t nsub = FID_BLOCK / d;
    double idx = (double)r * c / d * (pq_ilog2(K) + (K2 ? pq_ilog2(K2) : 0));
    double frame = (double)r * (c / FID_BLOCK) * 16.0;
    double books = books_owned ? (double)nsub * (K + K2) * d * 16.0 : 0.0;
    return idx + frame + books;
}


/* ---------- nearest entry: float distances, fixed scan order ----------
 * Squared (optionally diagonally weighted) distance from x to each of
 * the K entries of C (K x d, row-major), lowest index wins ties. The
 * dimension is a compile-time constant in the common cases so the
 * compiler can unroll and vectorize the distance. */
#define PQ_NEAR_BODY(D)                                                      \
    for (int e = 0; e < K; e++) {                                            \
        const float *ce = C + (int64_t)e * (D);                              \
        float dd = 0.0f;                                                     \
        for (int j = 0; j < (D); j++) {                                      \
            float t = x[j] - ce[j];                                          \
            dd += (w ? w[j] : 1.0f) * t * t;                                 \
        }                                                                    \
        if (dd < best) { best = dd; bk = e; }                                \
    }
static int pq_nearest(const float *x, const float *w, const float *C, int K, int d, float *best_out)
{
    float best = INFINITY;
    int bk = 0;
    switch (d) {
    case 4: PQ_NEAR_BODY(4) break;
    case 8: PQ_NEAR_BODY(8) break;
    case 16: PQ_NEAR_BODY(16) break;
    default: PQ_NEAR_BODY(d) break;
    }
    if (best_out) *best_out = best;
    return bk;
}

/* ---------- weighted k-means over subvectors, one book per position ---------- */

typedef struct {
    const float *W; const float *s0; const float *hdiag;
    int64_t r, c, nb, nsamp;
    int d, K, iters, act;
    uint64_t seed;
    float *book;           /* nsub*K*d */
    const float *base;     /* stage-1 book for residual learning, or NULL */
    int Kb;
} KmCtx;

static void km_item(void *vc, int64_t sub)
{
    KmCtx *k = (KmCtx *)vc;
    int d = k->d, K = k->K;
    int64_t ns = k->nsamp, tot = k->r * k->nb;
    float *pts = (float *)fid_alloc((size_t)ns * d * sizeof(float));
    float *wts = (float *)fid_alloc((size_t)ns * d * sizeof(float));
    const float *bb = k->base ? k->base + sub * (int64_t)k->Kb * d : NULL;
    for (int64_t i = 0; i < ns; i++) {
        int64_t blk = ns >= tot ? i : (int64_t)((double)i * tot / ns);
        int64_t row = blk / k->nb, b = blk % k->nb;
        float s = k->s0[row * k->nb + b];
        int64_t col0 = b * FID_BLOCK + sub * d;
        for (int j = 0; j < d; j++) {
            float u = s != 0.0f ? k->W[row * k->c + col0 + j] / s : 0.0f;
            pts[i * d + j] = u;
            wts[i * d + j] = k->act ? s * s * k->hdiag[col0 + j] : 1.0f;
        }
        if (bb) {   /* residual of the stage-1 nearest entry, same metric */
            double best = 1e300; int bk = 0;
            for (int e = 0; e < k->Kb; e++) {
                double dd = 0;
                for (int j = 0; j < d; j++) {
                    double t = pts[i * d + j] - bb[e * d + j];
                    dd += wts[i * d + j] * t * t;
                }
                if (dd < best) { best = dd; bk = e; }
            }
            for (int j = 0; j < d; j++) pts[i * d + j] -= bb[bk * d + j];
        }
    }
    float *cent = k->book + sub * (int64_t)K * d;
    /* init: K distinct sample positions from a keyed permutation walk */
    uint64_t key = fid_mix(k->seed ^ fid_mix((uint64_t)sub + 77));
    for (int e = 0; e < K; e++) {
        int64_t src = ns > 0 ? (int64_t)(fid_mix(key + (uint64_t)e) % (uint64_t)ns) : 0;
        for (int j = 0; j < d; j++) cent[e * d + j] = pts[src * d + j];
    }
    int *as = (int *)fid_alloc(sizeof(int) * ns);
    double *err = (double *)fid_alloc(sizeof(double) * ns);
    double *sum = (double *)fid_alloc(sizeof(double) * K * d);
    double *wsum = (double *)fid_alloc(sizeof(double) * K * d);
    for (int it = 0; it <= k->iters; it++) {
        for (int64_t i = 0; i < ns; i++) {
            float bf;
            int bk = pq_nearest(pts + i * d, wts + i * d, cent, K, d, &bf);
            double best = bf;
            as[i] = bk; err[i] = best;
        }
        if (it == k->iters) break;
        memset(sum, 0, sizeof(double) * K * d);
        memset(wsum, 0, sizeof(double) * K * d);
        for (int64_t i = 0; i < ns; i++)
            for (int j = 0; j < d; j++) {
                sum[as[i] * d + j] += (double)wts[i * d + j] * pts[i * d + j];
                wsum[as[i] * d + j] += wts[i * d + j];
            }
        for (int e = 0; e < K; e++) {
            double wt = 0;
            for (int j = 0; j < d; j++) wt += wsum[e * d + j];
            if (wt > 0) {
                for (int j = 0; j < d; j++)
                    if (wsum[e * d + j] > 0) cent[e * d + j] = (float)(sum[e * d + j] / wsum[e * d + j]);
            } else {
                /* empty entry: move it to the worst-served sample */
                int64_t wi = 0;
                for (int64_t i = 1; i < ns; i++) if (err[i] > err[wi]) wi = i;
                for (int j = 0; j < d; j++) cent[e * d + j] = pts[wi * d + j];
                err[wi] = 0;
            }
        }
    }
    for (int64_t i = 0; i < K * d; i++) cent[i] = fid_round_f16(cent[i]);
    free(pts); free(wts); free(as); free(err); free(sum); free(wsum);
}

static void pq_block_scales(const float *W, int64_t r, int64_t c, float *s)
{
    int64_t nb = c / FID_BLOCK;
    for (int64_t i = 0; i < r; i++)
        for (int64_t b = 0; b < nb; b++) {
            const float *x = W + i * c + b * FID_BLOCK;
            double ss = 0;
            for (int j = 0; j < FID_BLOCK; j++) ss += (double)x[j] * x[j];
            s[i * nb + b] = fid_round_f16((float)sqrt(ss / FID_BLOCK));
        }
}

/* ---------- assignment: plain or diag-weighted nearest entry ---------- */

typedef struct {
    const float *W; const float *hdiag; PqState *st; int act;
} AsCtx;

static void as_row(void *vc, int64_t row)
{
    AsCtx *a = (AsCtx *)vc;
    PqState *st = a->st;
    int d = st->d;
    for (int64_t b = 0; b < st->nb; b++) {
        float s = st->s[row * st->nb + b];
        for (int64_t sub = 0; sub < st->nsub; sub++) {
            int64_t col0 = b * FID_BLOCK + sub * d, g = col0 / d;
            float u[16], w[16];
            for (int j = 0; j < d; j++) {
                u[j] = s != 0.0f ? a->W[row * st->c + col0 + j] / s : 0.0f;
                w[j] = a->act ? a->hdiag[col0 + j] : 1.0f;
            }
            const float *B1 = st->b1 + sub * (int64_t)st->K * d;
            int bk = pq_nearest(u, a->act ? w : NULL, B1, st->K, d, NULL);
            st->k1[row * st->ng + g] = (uint16_t)bk;
            if (st->K2) {
                const float *B2 = st->b2 + sub * (int64_t)st->K2 * d;
                float rres[16];
                for (int j = 0; j < d; j++) rres[j] = u[j] - B1[bk * d + j];
                int k2 = pq_nearest(rres, a->act ? w : NULL, B2, st->K2, d, NULL);
                st->k2[row * st->ng + g] = (uint16_t)k2;
            }
        }
    }
}

/* ---------- error-feedback assignment (GPTQ / GPTVQ) ---------- */

typedef struct {
    float *W;              /* working copy, updated in place */
    const float *U;        /* c x c upper */
    const float *Ag;       /* ng x d x d: inverse of U's diagonal d-blocks */
    PqState *st;
    float *cA1, *cA2;      /* nsub x K x d: entries times A for this block */
    float *Err;            /* r x 256 */
    int64_t blk;
} GqCtx;

static void gq_row(void *vc, int64_t row)
{
    GqCtx *q = (GqCtx *)vc;
    PqState *st = q->st;
    int d = st->d;
    int64_t c = st->c, i1 = q->blk * FID_BLOCK;
    float *w = q->W + row * c + i1;
    double ss = 0;
    for (int j = 0; j < FID_BLOCK; j++) ss += (double)w[j] * w[j];
    float s = fid_round_f16((float)sqrt(ss / FID_BLOCK));
    st->s[row * st->nb + q->blk] = s;
    float *err = q->Err + row * FID_BLOCK;
    for (int64_t sub = 0; sub < st->nsub; sub++) {
        int64_t lc = sub * d, col0 = i1 + lc, g = col0 / d;
        const float *A = q->Ag + g * d * d;
        float uA[16], ea[16];
        for (int j = 0; j < d; j++) {
            double t = 0;
            for (int i = 0; i <= j; i++)
                t += (double)(s != 0.0f ? w[lc + i] / s : 0.0f) * A[i * d + j];
            uA[j] = (float)t;
        }
        const float *C1 = q->cA1 + sub * (int64_t)st->K * d;
        int bk = pq_nearest(uA, NULL, C1, st->K, d, NULL);
        st->k1[row * st->ng + g] = (uint16_t)bk;
        for (int j = 0; j < d; j++) ea[j] = uA[j] - C1[bk * d + j];
        if (st->K2) {
            const float *C2 = q->cA2 + sub * (int64_t)st->K2 * d;
            int k2 = pq_nearest(ea, NULL, C2, st->K2, d, NULL);
            st->k2[row * st->ng + g] = (uint16_t)k2;
            for (int j = 0; j < d; j++) ea[j] -= C2[k2 * d + j];
        }
        for (int j = 0; j < d; j++) { ea[j] *= s; err[lc + j] = ea[j]; }
        if (s == 0.0f) for (int j = 0; j < d; j++) { ea[j] = 0; err[lc + j] = 0; }
        /* push the error onto the rest of this block */
        for (int j = 0; j < d; j++) {
            if (ea[j] == 0.0f) continue;
            const float *Ur = q->U + (col0 + j) * c + i1;
            for (int64_t t = lc + d; t < FID_BLOCK; t++) w[t] -= ea[j] * Ur[t];
        }
    }
}

/* inverse of an upper-triangular d x d block (row-major), double */
static void pq_inv_upper(const float *Ublk, int64_t ld, int d, float *out)
{
    double M[16][16], X[16][16];
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) {
        M[i][j] = j >= i ? Ublk[i * ld + j] : 0.0; X[i][j] = 0.0;
    }
    for (int j = 0; j < d; j++) {
        X[j][j] = 1.0 / M[j][j];
        for (int i = j - 1; i >= 0; i--) {
            double t = 0;
            for (int k = i + 1; k <= j; k++) t += M[i][k] * X[k][j];
            X[i][j] = -t / M[i][i];
        }
    }
    for (int i = 0; i < d; i++) for (int j = 0; j < d; j++) out[i * d + j] = (float)X[i][j];
}

static void pq_gptq(PqState *st, const float *W0, const float *U)
{
    int d = st->d;
    int64_t r = st->r, c = st->c;
    float *W = (float *)fid_alloc((size_t)r * c * sizeof(float));
    memcpy(W, W0, (size_t)r * c * sizeof(float));
    float *Ag = (float *)fid_alloc((size_t)st->ng * d * d * sizeof(float));
    for (int64_t g = 0; g < st->ng; g++)
        pq_inv_upper(U + (g * d) * c + g * d, c, d, Ag + g * d * d);
    float *cA1 = (float *)fid_alloc((size_t)st->nsub * st->K * d * sizeof(float));
    float *cA2 = st->K2 ? (float *)fid_alloc((size_t)st->nsub * st->K2 * d * sizeof(float)) : NULL;
    float *Err = (float *)fid_alloc((size_t)r * FID_BLOCK * sizeof(float));
    GqCtx q;
    q.W = W; q.U = U; q.Ag = Ag; q.st = st; q.cA1 = cA1; q.cA2 = cA2; q.Err = Err;
    for (int64_t blk = 0; blk < st->nb; blk++) {
        for (int64_t sub = 0; sub < st->nsub; sub++) {
            int64_t g = (blk * FID_BLOCK + sub * d) / d;
            const float *A = Ag + g * d * d;
            for (int stage = 0; stage < (st->K2 ? 2 : 1); stage++) {
                int K = stage ? st->K2 : st->K;
                const float *B = (stage ? st->b2 : st->b1) + sub * (int64_t)K * d;
                float *CA = (stage ? cA2 : cA1) + sub * (int64_t)K * d;
                for (int e = 0; e < K; e++)
                    for (int j = 0; j < d; j++) {
                        double t = 0;
                        for (int i = 0; i <= j; i++) t += (double)B[e * d + i] * A[i * d + j];
                        CA[e * d + j] = (float)t;
                    }
            }
        }
        q.blk = blk;
        fid_par_for(r, gq_row, &q);
        int64_t i2 = (blk + 1) * FID_BLOCK;
        if (i2 < c) {
            for (int64_t i = 0; i < r * FID_BLOCK; i++) Err[i] = -Err[i];
            fid_gemm(W + i2, c, Err, FID_BLOCK, U + (blk * FID_BLOCK) * c + i2, c,
                     r, FID_BLOCK, c - i2);
        }
    }
    free(W); free(Ag); free(cA1); free(cA2); free(Err);
}

/* ---------- reconstruction and the quadratic objective ---------- */

static void pq_recon_rows(const PqState *st, int64_t r0, int64_t nr, int scaled,
                          const float *b1, const float *b2, float *Q)
{
    int d = st->d;
    for (int64_t i = 0; i < nr; i++) {
        int64_t row = r0 + i;
        for (int64_t g = 0; g < st->ng; g++) {
            int64_t col0 = g * d, b = col0 / FID_BLOCK, sub = (col0 % FID_BLOCK) / d;
            float s = scaled ? st->s[row * st->nb + b] : 1.0f;
            const float *c1 = b1 + (sub * (int64_t)st->K + st->k1[row * st->ng + g]) * d;
            const float *c2 = st->K2 && b2 ? b2 + (sub * (int64_t)st->K2 + st->k2[row * st->ng + g]) * d : NULL;
            for (int j = 0; j < d; j++)
                Q[i * st->c + col0 + j] = s * (c1[j] + (c2 ? c2[j] : 0.0f));
        }
    }
}

/* sum over rows of x H x^T for X (nr x c), with Hd = H + lam I */
static double pq_quad_rows(const float *X, int64_t nr, int64_t c, const float *H,
                           double lam, float *tmp)
{
    memset(tmp, 0, (size_t)nr * c * sizeof(float));
    fid_gemm(tmp, c, X, c, H, c, nr, c, c);
    double s = 0;
    for (int64_t i = 0; i < nr * c; i++) s += (double)tmp[i] * X[i] + lam * (double)X[i] * X[i];
    return s;
}

#define PQ_CHUNK 2048

/* f = sum_r (Q_r - W_r) Hd (Q_r - W_r)^T for the current state */
static double pq_objective(const PqState *st, const float *W, const float *H, double lam)
{
    int64_t c = st->c;
    float *Q = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    float *T = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    double f = 0;
    for (int64_t r0 = 0; r0 < st->r; r0 += PQ_CHUNK) {
        int64_t nr = st->r - r0 < PQ_CHUNK ? st->r - r0 : PQ_CHUNK;
        pq_recon_rows(st, r0, nr, 1, st->b1, st->b2, Q);
        for (int64_t i = 0; i < nr * c; i++) Q[i] -= W[r0 * c + i];
        f += pq_quad_rows(Q, nr, c, H, lam, T);
    }
    free(Q); free(T);
    return f;
}

static double pq_wquad(const float *W, int64_t r, int64_t c, const float *H, double lam)
{
    float *T = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    double f = 0;
    for (int64_t r0 = 0; r0 < r; r0 += PQ_CHUNK) {
        int64_t nr = r - r0 < PQ_CHUNK ? r - r0 : PQ_CHUNK;
        f += pq_quad_rows(W + r0 * c, nr, c, H, lam, T);
    }
    free(T);
    return f;
}

/* A(p) = M^T (M(p) Hd), with M the code map; also used for b = M^T(W Hd).
 * p == NULL: use Win (r x c) in place of M(p). Output g (double, nparam). */
static void pq_apply(const PqState *st, const double *p, const float *Win,
                     const float *H, double lam, double *g)
{
    int d = st->d;
    int64_t c = st->c, n1 = st->nsub * (int64_t)st->K * d;
    int64_t np = n1 + st->nsub * (int64_t)st->K2 * d;
    float *pb1 = NULL, *pb2 = NULL;
    if (p) {
        pb1 = (float *)fid_alloc(sizeof(float) * n1);
        for (int64_t i = 0; i < n1; i++) pb1[i] = (float)p[i];
        if (st->K2) {
            pb2 = (float *)fid_alloc(sizeof(float) * (np - n1));
            for (int64_t i = 0; i < np - n1; i++) pb2[i] = (float)p[n1 + i];
        }
    }
    float *Y = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    float *Z = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    memset(g, 0, sizeof(double) * np);
    for (int64_t r0 = 0; r0 < st->r; r0 += PQ_CHUNK) {
        int64_t nr = st->r - r0 < PQ_CHUNK ? st->r - r0 : PQ_CHUNK;
        if (p) pq_recon_rows(st, r0, nr, 1, pb1, pb2, Y);
        else memcpy(Y, Win + r0 * c, (size_t)nr * c * sizeof(float));
        memset(Z, 0, (size_t)nr * c * sizeof(float));
        fid_gemm(Z, c, Y, c, H, c, nr, c, c);
        for (int64_t i = 0; i < nr * c; i++) Z[i] += (float)(lam * Y[i]);
        for (int64_t i = 0; i < nr; i++) {
            int64_t row = r0 + i;
            for (int64_t gg = 0; gg < st->ng; gg++) {
                int64_t col0 = gg * d, b = col0 / FID_BLOCK, sub = (col0 % FID_BLOCK) / d;
                double s = st->s[row * st->nb + b];
                const float *z = Z + i * c + col0;
                double *g1 = g + (sub * (int64_t)st->K + st->k1[row * st->ng + gg]) * d;
                for (int j = 0; j < d; j++) g1[j] += s * z[j];
                if (st->K2) {
                    double *g2 = g + n1 + (sub * (int64_t)st->K2 + st->k2[row * st->ng + gg]) * d;
                    for (int j = 0; j < d; j++) g2[j] += s * z[j];
                }
            }
        }
    }
    free(Y); free(Z); free(pb1); free(pb2);
}

/* CG on A theta = b over all book entries. Keeps the best iterate by
 * the true objective; never returns books worse than it was given. */
static void pq_refit_books(PqState *st, const float *W, const float *H, double lam,
                           int iters, double *f_io)
{
    int d = st->d;
    int64_t n1 = st->nsub * (int64_t)st->K * d;
    int64_t np = n1 + st->nsub * (int64_t)st->K2 * d;
    double *th = (double *)fid_alloc(sizeof(double) * np);
    for (int64_t i = 0; i < n1; i++) th[i] = st->b1[i];
    for (int64_t i = n1; i < np; i++) th[i] = st->b2[i - n1];
    double *b = (double *)fid_alloc(sizeof(double) * np);
    double *rr = (double *)fid_alloc(sizeof(double) * np);
    double *pp = (double *)fid_alloc(sizeof(double) * np);
    double *Ap = (double *)fid_alloc(sizeof(double) * np);
    pq_apply(st, NULL, W, H, lam, b);
    pq_apply(st, th, NULL, H, lam, Ap);
    double rs = 0;
    for (int64_t i = 0; i < np; i++) { rr[i] = b[i] - Ap[i]; pp[i] = rr[i]; rs += rr[i] * rr[i]; }
    for (int it = 0; it < iters && rs > 0; it++) {
        pq_apply(st, pp, NULL, H, lam, Ap);
        double pAp = 0;
        for (int64_t i = 0; i < np; i++) pAp += pp[i] * Ap[i];
        if (!(pAp > 0)) break;
        double al = rs / pAp, rs2 = 0;
        for (int64_t i = 0; i < np; i++) { th[i] += al * pp[i]; rr[i] -= al * Ap[i]; rs2 += rr[i] * rr[i]; }
        double be = rs2 / rs;
        rs = rs2;
        for (int64_t i = 0; i < np; i++) pp[i] = rr[i] + be * pp[i];
    }
    float *o1 = (float *)fid_alloc(sizeof(float) * n1);
    float *o2 = st->K2 ? (float *)fid_alloc(sizeof(float) * (np - n1)) : NULL;
    memcpy(o1, st->b1, sizeof(float) * n1);
    if (o2) memcpy(o2, st->b2, sizeof(float) * (np - n1));
    for (int64_t i = 0; i < n1; i++) st->b1[i] = fid_round_f16((float)th[i]);
    for (int64_t i = n1; i < np; i++) st->b2[i - n1] = fid_round_f16((float)th[i]);
    double f = pq_objective(st, W, H, lam);
    if (f <= *f_io) *f_io = f;
    else {   /* keep what we had */
        memcpy(st->b1, o1, sizeof(float) * n1);
        if (o2) memcpy(st->b2, o2, sizeof(float) * (np - n1));
    }
    free(th); free(b); free(rr); free(pp); free(Ap); free(o1); free(o2);
}

/* per-row scale refit: minimize (sum_b s_b v_b - w) Hd (.)^T over s */
typedef struct {
    const PqState *st; const float *P; const float *Qt; const float *W;
    double *G; double *h; int64_t r0, b;
} ScCtx;

static void sc_acc_row(void *vc, int64_t i)
{
    ScCtx *x = (ScCtx *)vc;
    const PqState *st = x->st;
    int64_t c = st->c, nb = st->nb, b = x->b;
    const float *p = x->P + i * c;                    /* v_b Hd for this row */
    const float *q = x->Qt + i * c;
    const float *w = x->W + (x->r0 + i) * c;
    double *G = x->G + i * nb * nb, *h = x->h + i * nb;
    for (int64_t b2 = 0; b2 < nb; b2++) {
        double t = 0;
        for (int j = 0; j < FID_BLOCK; j++) t += (double)p[b2 * FID_BLOCK + j] * q[b2 * FID_BLOCK + j];
        G[b * nb + b2] = t;
    }
    double t = 0;
    for (int64_t j = 0; j < c; j++) t += (double)p[j] * w[j];
    h[b] = t;
}

static void sc_solve_row(void *vc, int64_t i)
{
    ScCtx *x = (ScCtx *)vc;
    PqState *st = (PqState *)x->st;
    int64_t nb = st->nb;
    double *G = x->G + i * nb * nb, *h = x->h + i * nb;
    double A[64][64], y[64];
    double tr = 0;
    for (int64_t a = 0; a < nb; a++) tr += G[a * nb + a];
    for (int64_t a = 0; a < nb; a++) {
        for (int64_t b = 0; b < nb; b++) A[a][b] = 0.5 * (G[a * nb + b] + G[b * nb + a]);
        A[a][a] += 1e-9 * tr / nb + 1e-30;
        y[a] = h[a];
    }
    /* Cholesky solve */
    for (int64_t j = 0; j < nb; j++) {
        double s = A[j][j];
        for (int64_t k = 0; k < j; k++) s -= A[j][k] * A[j][k];
        if (!(s > 0)) return;   /* leave the row's scales as they are */
        A[j][j] = sqrt(s);
        for (int64_t i2 = j + 1; i2 < nb; i2++) {
            double t = A[i2][j];
            for (int64_t k = 0; k < j; k++) t -= A[i2][k] * A[j][k];
            A[i2][j] = t / A[j][j];
        }
    }
    for (int64_t j = 0; j < nb; j++) {
        double t = y[j];
        for (int64_t k = 0; k < j; k++) t -= A[j][k] * y[k];
        y[j] = t / A[j][j];
    }
    for (int64_t j = nb - 1; j >= 0; j--) {
        double t = y[j];
        for (int64_t k = j + 1; k < nb; k++) t -= A[k][j] * y[k];
        y[j] = t / A[j][j];
    }
    int64_t row = x->r0 + i;
    for (int64_t b = 0; b < nb; b++)
        if (st->s[row * nb + b] != 0.0f) st->s[row * nb + b] = fid_round_f16((float)y[b]);
}

static void pq_refit_scales(PqState *st, const float *W, const float *H, double lam,
                            double *f_io)
{
    int64_t c = st->c, nb = st->nb;
    if (nb > 64) return;
    float *old = (float *)fid_alloc(sizeof(float) * st->r * nb);
    memcpy(old, st->s, sizeof(float) * st->r * nb);
    float *Qt = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    float *P = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
    double *G = (double *)fid_alloc(sizeof(double) * PQ_CHUNK * nb * nb);
    double *h = (double *)fid_alloc(sizeof(double) * PQ_CHUNK * nb);
    ScCtx x;
    x.st = st; x.P = P; x.Qt = Qt; x.W = W; x.G = G; x.h = h;
    for (int64_t r0 = 0; r0 < st->r; r0 += PQ_CHUNK) {
        int64_t nr = st->r - r0 < PQ_CHUNK ? st->r - r0 : PQ_CHUNK;
        pq_recon_rows(st, r0, nr, 0, st->b1, st->b2, Qt);
        x.r0 = r0;
        for (int64_t b = 0; b < nb; b++) {
            memset(P, 0, (size_t)nr * c * sizeof(float));
            fid_gemm(P, c, Qt + b * FID_BLOCK, c, H + (b * FID_BLOCK) * c, c, nr, FID_BLOCK, c);
            for (int64_t i = 0; i < nr; i++)
                for (int j = 0; j < FID_BLOCK; j++)
                    P[i * c + b * FID_BLOCK + j] += (float)(lam * Qt[i * c + b * FID_BLOCK + j]);
            x.b = b;
            fid_par_for(nr, sc_acc_row, &x);
        }
        fid_par_for(nr, sc_solve_row, &x);
    }
    double f = pq_objective(st, W, H, lam);
    if (f <= *f_io) *f_io = f;
    else memcpy(st->s, old, sizeof(float) * st->r * nb);
    free(old); free(Qt); free(P); free(G); free(h);
}

/* Final reconstruction (bf16, as the engine will read it) and the
 * weight and output errors against W (H undamped, may be NULL). */
static void pq_finish(const float *W, const float *H, const PqState *st, float *Q,
                      PqStats *ps)
{
    int64_t r = st->r, c = st->c;
    for (int64_t r0 = 0; r0 < r; r0 += PQ_CHUNK) {
        int64_t nr = r - r0 < PQ_CHUNK ? r - r0 : PQ_CHUNK;
        pq_recon_rows(st, r0, nr, 1, st->b1, st->b2, Q + r0 * c);
    }
    double e2 = 0, w2 = 0;
    for (int64_t i = 0; i < r * c; i++) {
        Q[i] = fid_bf16_to_f32(fid_f32_to_bf16(Q[i]));
        double dd = (double)W[i] - Q[i];
        e2 += dd * dd; w2 += (double)W[i] * W[i];
    }
    ps->wrel = w2 > 0 ? sqrt(e2 / w2) : 0;
    ps->orel = 0;
    if (H) {
        float *D = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
        float *T = (float *)fid_alloc((size_t)PQ_CHUNK * c * sizeof(float));
        double num = 0, den = 0;
        for (int64_t r0 = 0; r0 < r; r0 += PQ_CHUNK) {
            int64_t nr = r - r0 < PQ_CHUNK ? r - r0 : PQ_CHUNK;
            for (int64_t i = 0; i < nr * c; i++) D[i] = W[r0 * c + i] - Q[r0 * c + i];
            num += pq_quad_rows(D, nr, c, H, 0.0, T);
            den += pq_quad_rows(W + r0 * c, nr, c, H, 0.0, T);
        }
        ps->orel = den > 0 ? num / den : 0;
        free(D); free(T);
    }
}

/* ---------- state files (.fidq): what a decoder needs, for the
 * distillation step (method 5) and for rebuilding a variant ---------- */

static int pq_state_write(const char *path, const PqState *st)
{
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    uint32_t hdr[2] = { 0x51444946u, 1u };   /* "FIDQ", version 1 */
    int64_t dims[3] = { st->r, st->c, 0 };
    int32_t cfg[3] = { st->d, st->K, st->K2 };
    int ok = fwrite(hdr, 4, 2, f) == 2 && fwrite(dims, 8, 3, f) == 3 && fwrite(cfg, 4, 3, f) == 3;
    size_t n1 = (size_t)st->nsub * st->K * st->d, n2 = (size_t)st->nsub * st->K2 * st->d;
    ok = ok && fwrite(st->b1, sizeof(float), n1, f) == n1;
    if (st->K2) ok = ok && fwrite(st->b2, sizeof(float), n2, f) == n2;
    ok = ok && fwrite(st->s, sizeof(float), (size_t)(st->r * st->nb), f) == (size_t)(st->r * st->nb);
    ok = ok && fwrite(st->k1, 2, (size_t)(st->r * st->ng), f) == (size_t)(st->r * st->ng);
    if (st->K2) ok = ok && fwrite(st->k2, 2, (size_t)(st->r * st->ng), f) == (size_t)(st->r * st->ng);
    if (fclose(f) != 0) ok = 0;
    return ok ? 0 : -1;
}

/* returns 0 on success, 1 if the file does not exist, -1 if it is bad */
static int pq_state_read(const char *path, PqState *st)
{
    memset(st, 0, sizeof(*st));
    FILE *f = fopen(path, "rb");
    if (!f) return 1;
    uint32_t hdr[2];
    int64_t dims[3];
    int32_t cfg[3];
    if (fread(hdr, 4, 2, f) != 2 || hdr[0] != 0x51444946u || hdr[1] != 1u
            || fread(dims, 8, 3, f) != 3 || fread(cfg, 4, 3, f) != 3) { fclose(f); return -1; }
    st->r = dims[0]; st->c = dims[1]; st->d = cfg[0]; st->K = cfg[1]; st->K2 = cfg[2];
    if (st->d <= 0 || st->d > 16 || FID_BLOCK % st->d || st->c % FID_BLOCK || st->K <= 0
            || st->K > 65536 || st->K2 < 0 || st->K2 > 65536) { fclose(f); return -1; }
    st->nb = st->c / FID_BLOCK; st->nsub = FID_BLOCK / st->d; st->ng = st->c / st->d;
    size_t n1 = (size_t)st->nsub * st->K * st->d, n2 = (size_t)st->nsub * st->K2 * st->d;
    st->b1 = (float *)fid_alloc(n1 * sizeof(float));
    st->b2 = st->K2 ? (float *)fid_alloc(n2 * sizeof(float)) : NULL;
    st->s = (float *)fid_alloc((size_t)(st->r * st->nb) * sizeof(float));
    st->k1 = (uint16_t *)fid_alloc((size_t)(st->r * st->ng) * 2);
    st->k2 = st->K2 ? (uint16_t *)fid_alloc((size_t)(st->r * st->ng) * 2) : NULL;
    int ok = fread(st->b1, sizeof(float), n1, f) == n1;
    if (st->K2) ok = ok && fread(st->b2, sizeof(float), n2, f) == n2;
    ok = ok && fread(st->s, sizeof(float), (size_t)(st->r * st->nb), f) == (size_t)(st->r * st->nb);
    ok = ok && fread(st->k1, 2, (size_t)(st->r * st->ng), f) == (size_t)(st->r * st->ng);
    if (st->K2) ok = ok && fread(st->k2, 2, (size_t)(st->r * st->ng), f) == (size_t)(st->r * st->ng);
    fclose(f);
    if (ok) for (int64_t i = 0; i < st->r * st->ng; i++)
        if (st->k1[i] >= st->K || (st->K2 && st->k2[i] >= st->K2)) { ok = 0; break; }
    if (!ok) { free(st->b1); free(st->b2); free(st->s); free(st->k1); free(st->k2); memset(st, 0, sizeof(*st)); return -1; }
    return 0;
}

/* ---------- the whole tensor ---------- */

static void pq_state_free(PqState *st)
{
    free(st->b1); free(st->b2); free(st->s); free(st->k1); free(st->k2);
    memset(st, 0, sizeof(*st));
}

/* Quantize W (r x c). H (c x c, mean second moment) and U (its damped
 * inverse factor) may be NULL for weight fit. Q receives the
 * reconstruction. The state (books, scales, codes) is returned in st. */
static void pq_quantize(const float *W, int64_t r, int64_t c, const float *H,
                        const float *U, const PqCfg *cfg, float *Q, PqState *st,
                        PqStats *ps)
{
    double t0 = fid_now();
    memset(ps, 0, sizeof(*ps));
    for (int i = 0; i < 8; i++) ps->orel_stage[i] = -1;
    memset(st, 0, sizeof(*st));
    int d = cfg->d;
    st->r = r; st->c = c; st->d = d; st->K = cfg->K; st->K2 = cfg->K2;
    st->nb = c / FID_BLOCK; st->nsub = FID_BLOCK / d; st->ng = c / d;
    st->b1 = (float *)fid_alloc(sizeof(float) * st->nsub * cfg->K * d);
    st->b2 = cfg->K2 ? (float *)fid_alloc(sizeof(float) * st->nsub * cfg->K2 * d) : NULL;
    st->s = (float *)fid_alloc(sizeof(float) * r * st->nb);
    st->k1 = (uint16_t *)fid_calloc((size_t)r * st->ng, sizeof(uint16_t));
    st->k2 = cfg->K2 ? (uint16_t *)fid_calloc((size_t)r * st->ng, sizeof(uint16_t)) : NULL;
    float *hdiag = (float *)fid_alloc(sizeof(float) * c);
    double lam = 0;
    if (H && (cfg->fit_act || cfg->gptq || cfg->refit > 0 || cfg->scale_refit)) {
        double md = 0;
        for (int64_t i = 0; i < c; i++) { hdiag[i] = H[i * c + i]; md += hdiag[i]; }
        md /= c;
        lam = cfg->damp * md;
        for (int64_t i = 0; i < c; i++) hdiag[i] += (float)lam;
    } else for (int64_t i = 0; i < c; i++) hdiag[i] = 1.0f;
    pq_block_scales(W, r, c, st->s);
    if (cfg->ext_b1) {
        memcpy(st->b1, cfg->ext_b1, sizeof(float) * st->nsub * cfg->K * d);
        if (cfg->K2) memcpy(st->b2, cfg->ext_b2, sizeof(float) * st->nsub * cfg->K2 * d);
    } else {
        KmCtx k;
        memset(&k, 0, sizeof(k));
        k.W = W; k.s0 = st->s; k.hdiag = hdiag; k.r = r; k.c = c; k.nb = st->nb;
        k.nsamp = cfg->sample < r * st->nb ? cfg->sample : r * st->nb;
        k.d = d; k.K = cfg->K; k.iters = cfg->km_iters; k.act = cfg->fit_act && H;
        k.seed = cfg->seed; k.book = st->b1;
        fid_par_for(st->nsub, km_item, &k);
        if (cfg->K2) {
            k.K = cfg->K2; k.book = st->b2; k.base = st->b1; k.Kb = cfg->K;
            k.seed = cfg->seed ^ 0x5151;
            fid_par_for(st->nsub, km_item, &k);
        }
    }
    if (cfg->learn_only) { free(hdiag); ps->secs = fid_now() - t0; return; }
    double fw = (H && (cfg->gptq || cfg->refit > 0 || cfg->scale_refit)) ? pq_wquad(W, r, c, H, lam) : 0;
    double f = 0;
    int stage = 0;
    for (int round = 0; round < (cfg->rounds > 0 ? cfg->rounds : 1); round++) {
        if (cfg->gptq && U) pq_gptq(st, W, U);
        else {
            if (round > 0) pq_block_scales(W, r, c, st->s);
            AsCtx a; a.W = W; a.hdiag = hdiag; a.st = st; a.act = cfg->fit_act && H;
            fid_par_for(r, as_row, &a);
        }
        if (H && (cfg->gptq || cfg->refit > 0 || cfg->scale_refit)) {
            f = pq_objective(st, W, H, lam);
            if (stage < 8) ps->orel_stage[stage++] = f / fw;
            if (cfg->refit > 0) {
                pq_refit_books(st, W, H, lam, cfg->refit, &f);
                if (stage < 8) ps->orel_stage[stage++] = f / fw;
            }
            if (cfg->scale_refit) {
                pq_refit_scales(st, W, H, lam, &f);
                if (stage < 8) ps->orel_stage[stage++] = f / fw;
                if (cfg->refit > 0) {
                    pq_refit_books(st, W, H, lam, cfg->refit, &f);
                    if (stage < 8) ps->orel_stage[stage++] = f / fw;
                }
            }
        }
    }
    pq_finish(W, H, st, Q, ps);
    ps->bits = pq_bits(r, c, d, cfg->K, cfg->K2, cfg->ext_b1 == NULL);
    ps->secs = fid_now() - t0;
    free(hdiag);
}

#endif
