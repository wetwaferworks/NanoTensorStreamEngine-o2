/* sn_pqk.c - packed-matrix kernels (multi-vector, resident, DTC). See sn_pqk.h. Pure C99, no locks. */
#include "sn_pqk.h"
#include <stdlib.h>
#include <string.h>

#define SN_SLAB 1024u

static int seek64(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

/* ------------------------------------------------------------------ resident tensors */

uint64_t sn_res_bytes(const sn_pack *p, const sn_tensor *t)
{
    if (t->raw) return 0;
    return t->nblocks * (uint64_t)(p->nsub + 4) + p->book_floats * 4ull;
}

void sn_res_free(sn_pq_res *r)
{
    if (!r) return;
    free(r->idx); free(r->mn); free(r->sc); free(r->book);
    memset(r, 0, sizeof(*r));
}

int sn_res_load(const sn_pack *p, sn_rd *rd, const sn_tensor *t, sn_pq_res *out)
{
    memset(out, 0, sizeof(*out));
    if (t->raw || p->pq_d != 4) return -1;
    uint64_t nb = t->nblocks, g0 = t->block0;
    out->nblocks = nb;
    out->idx = (uint8_t *)malloc((size_t)(nb * p->nsub));
    out->mn = (uint16_t *)malloc((size_t)nb * 2);
    out->sc = (uint16_t *)malloc((size_t)nb * 2);
    out->book = (float *)malloc((size_t)p->book_floats * 4);
    int ok = out->idx && out->mn && out->sc && out->book;
    if (ok) ok = seek64(rd->fs, p->body_off + g0 * p->nsub) == 0 && fread(out->idx, p->nsub, (size_t)nb, rd->fs) == nb;
    if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * p->nsub + g0 * 2) == 0 && fread(out->mn, 2, (size_t)nb, rd->fs) == nb;
    if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * (p->nsub + 2ull) + g0 * 2) == 0 && fread(out->sc, 2, (size_t)nb, rd->fs) == nb;
    if (ok) ok = sn_load_book(p, rd, t, out->book) == 0;
    if (!ok) { sn_res_free(out); return -1; }
    out->bytes = sn_res_bytes(p, t);
    return 0;
}

/* The same bytes as sn_res_load, into buffers the caller owns (the GPU expert stream's pinned staging). */
int sn_res_read_into(const sn_pack *p, sn_rd *rd, const sn_tensor *t, uint8_t *idx, uint16_t *mn, uint16_t *sc, float *book)
{
    if (t->raw || p->pq_d != 4) return -1;
    uint64_t nb = t->nblocks, g0 = t->block0;
    int ok = seek64(rd->fs, p->body_off + g0 * p->nsub) == 0 && fread(idx, p->nsub, (size_t)nb, rd->fs) == nb;
    if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * p->nsub + g0 * 2) == 0 && fread(mn, 2, (size_t)nb, rd->fs) == nb;
    if (ok) ok = seek64(rd->fs, p->body_off + p->tot_blocks * (p->nsub + 2ull) + g0 * 2) == 0 && fread(sc, 2, (size_t)nb, rd->fs) == nb;
    if (ok) ok = sn_load_book(p, rd, t, book) == 0;
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------ DTC tables */

static int gcd_i(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

int sn_dtc_geom(const sn_pack *p, int cols, int *nph, int *ng)
{
    if (p->pq_d != 4 || p->block != 256 || cols <= 0 || (cols & 3)) return -1;
    int gph = gcd_i(cols, 256);
    if (nph) *nph = 256 / gph;
    if (ng) *ng = cols >> 2;
    return 0;
}

size_t sn_dtc_floats(const sn_pack *p, int cols)
{
    int nph, ng;
    if (sn_dtc_geom(p, cols, &nph, &ng) != 0) return 0;
    return (size_t)nph * (size_t)ng * p->entries;
}

void sn_dtc_build(const sn_pack *p, const float *book, int cols, const float *x, int g0, int g1, float *T)
{
    int nph, ng;
    if (sn_dtc_geom(p, cols, &nph, &ng) != 0) return;
    const uint32_t E = p->entries, nsub = p->nsub;
    const int gph = 256 / nph;
    for (int ph = 0; ph < nph; ph++) {
        const uint32_t sub0 = (uint32_t)((ph * gph) >> 2);
        for (int G = g0; G < g1; G++) {
            const float *c = book + (size_t)((sub0 + (uint32_t)G) % nsub) * E * 4;
            const float x0 = x[4 * G], x1 = x[4 * G + 1], x2 = x[4 * G + 2], x3 = x[4 * G + 3];
            float *Tk = T + ((size_t)ph * (size_t)ng + (size_t)G) * E;
            for (uint32_t k = 0; k < E; k++, c += 4)
                Tk[k] = c[0] * x0 + c[1] * x1 + c[2] * x2 + c[3] * x3;
        }
    }
}

/* Cost model, in rough cycles per 4-weight group: direct ~4.5, a table lookup ~1.0, a table entry ~0.5. */
int sn_dtc_pays(const sn_pack *p, int rows, int cols, int nv, int build_lanes)
{
    (void)nv;
    int nph, ng;
    if (sn_dtc_geom(p, cols, &nph, &ng) != 0 || p->entries != 256) return 0;
    if (build_lanes < 1) build_lanes = 1;
    double direct = 4.5 * (double)rows * ng;
    double dtc = 0.5 * (double)nph * ng * p->entries / (double)build_lanes + 1.0 * (double)rows * ng;
    return dtc < 0.85 * direct;
}

/* ------------------------------------------------------------------ kernels */

typedef struct {
    const sn_pack *p;
    const float *book;
    int cols, r0, nph, ng, gph;
    uint64_t flat0, flat1;
    const sn_mv *mv;
    double acc[SN_NV];
} pqk_st;

static inline float dtc_seg(const float *Tp, const uint8_t *ix, uint32_t n, uint32_t E)
{
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    uint32_t s = 0;
    for (; s + 4 <= n; s += 4) {
        a0 += Tp[(size_t)s * E + ix[s]];
        a1 += Tp[(size_t)(s + 1) * E + ix[s + 1]];
        a2 += Tp[(size_t)(s + 2) * E + ix[s + 2]];
        a3 += Tp[(size_t)(s + 3) * E + ix[s + 3]];
    }
    if (s < n) { a0 += Tp[(size_t)s * E + ix[s]]; s++; }
    if (s < n) { a1 += Tp[(size_t)s * E + ix[s]]; s++; }
    if (s < n) { a2 += Tp[(size_t)s * E + ix[s]]; }
    return (a0 + a1) + (a2 + a3);
}

static void pqk_blocks(pqk_st *S, uint64_t bs, uint64_t nb, const uint8_t *idx, const uint16_t *mn, const uint16_t *sc)
{
    const uint32_t B = S->p->block, nsub = S->p->nsub, E = S->p->entries;
    const int cols = S->cols, nv = S->mv->nv;
    const uint64_t ucols = (uint64_t)cols;
    for (uint64_t b = 0; b < nb; b++) {
        const uint64_t bf0 = (bs + b) * B;
        uint64_t cur = bf0 > S->flat0 ? bf0 : S->flat0;
        const uint64_t hi = bf0 + B < S->flat1 ? bf0 + B : S->flat1;
        const float m = sn_f16_lut[mn[b]], s = sn_f16_lut[sc[b]];
        const uint8_t *ix = idx + b * nsub;
        while (cur < hi) {
            const uint64_t row = cur / ucols, col = cur - row * ucols;
            const uint64_t row_end = (row + 1) * ucols;
            const uint64_t seg_end = row_end < hi ? row_end : hi;
            const uint32_t sub0 = (uint32_t)((cur - bf0) >> 2), sub1 = (uint32_t)((seg_end - bf0) >> 2);
            const uint64_t len = seg_end - cur;
            size_t phbase = 0;
            if (S->mv->T[0] && S->nph > 1) phbase = (size_t)(((row * ucols) & 255u) / (uint64_t)S->gph) * (size_t)S->ng * E;
            for (int v = 0; v < nv; v++) {
                float dotc;
                if (S->mv->T[v]) {
                    dotc = dtc_seg(S->mv->T[v] + phbase + (size_t)(col >> 2) * E, ix + sub0, sub1 - sub0, E);
                } else {
                    const float *xs = S->mv->x[v] + col;
                    dotc = 0.0f;
                    for (uint32_t sub = sub0; sub < sub1; sub++) {
                        const float *c = S->book + ((size_t)sub * E + ix[sub]) * 4;
                        uint32_t j = (sub - sub0) << 2;
                        dotc += c[0] * xs[j] + c[1] * xs[j + 1] + c[2] * xs[j + 2] + c[3] * xs[j + 3];
                    }
                }
                S->acc[v] += (double)m * (S->mv->P[v][col + len] - S->mv->P[v][col]) + (double)s * (double)dotc;
                if (seg_end == row_end) { S->mv->y[v][row - (uint64_t)S->r0] = (float)S->acc[v]; S->acc[v] = 0.0; }
            }
            cur = seg_end;
        }
    }
}

static int pqk_init(pqk_st *S, const sn_pack *p, const sn_tensor *t, const float *book, int cols, const sn_mv *mv, int r0, int r1)
{
    if (t->raw || p->pq_d != 4 || cols <= 0 || (cols & 3) || r1 <= r0 || !mv || mv->nv < 1 || mv->nv > SN_NV) return -1;
    if ((uint64_t)r1 * (uint64_t)cols > t->count) return -1;
    memset(S, 0, sizeof(*S));
    S->p = p; S->book = book; S->cols = cols; S->r0 = r0; S->mv = mv;
    S->flat0 = (uint64_t)r0 * (uint64_t)cols;
    S->flat1 = (uint64_t)r1 * (uint64_t)cols;
    if (mv->T[0]) {
        if (p->entries != 256 || sn_dtc_geom(p, cols, &S->nph, &S->ng) != 0) return -1;
        S->gph = 256 / S->nph;
        for (int v = 1; v < mv->nv; v++) if (!mv->T[v]) return -1;
    }
    return 0;
}

int sn_pqk_rows_res(const sn_pack *p, const sn_pq_res *res, const sn_tensor *t, int cols, const sn_mv *mv, int r0, int r1)
{
    pqk_st S;
    if (pqk_init(&S, p, t, res->book, cols, mv, r0, r1) != 0) return -1;
    const uint64_t b_lo = S.flat0 / p->block, b_hi = (S.flat1 + p->block - 1) / p->block;
    if (b_hi > res->nblocks) return -1;
    pqk_blocks(&S, b_lo, b_hi - b_lo, res->idx + b_lo * p->nsub, res->mn + b_lo, res->sc + b_lo);
    return 0;
}

int sn_pqk_rows(const sn_pack *p, sn_rd *r, const sn_tensor *t, const float *book, int cols, const sn_mv *mv, int r0, int r1)
{
    pqk_st S;
    if (pqk_init(&S, p, t, book, cols, mv, r0, r1) != 0) return -1;
    const uint32_t B = p->block, nsub = p->nsub;
    const uint64_t b_lo = S.flat0 / B, b_hi = (S.flat1 + B - 1) / B;
    uint8_t *idx = (uint8_t *)malloc((size_t)SN_SLAB * nsub);
    uint16_t *mn = (uint16_t *)malloc(SN_SLAB * 2), *sc = (uint16_t *)malloc(SN_SLAB * 2);
    if (!idx || !mn || !sc) { free(idx); free(mn); free(sc); return -1; }
    int rc = 0;
    for (uint64_t bs = b_lo; bs < b_hi && !rc; bs += SN_SLAB) {
        uint64_t be = bs + SN_SLAB < b_hi ? bs + SN_SLAB : b_hi, nb = be - bs;
        uint64_t g0 = t->block0 + bs;
        if (seek64(r->fs, p->body_off + g0 * nsub) != 0 || fread(idx, nsub, (size_t)nb, r->fs) != nb) { rc = -1; break; }
        if (seek64(r->fs, p->body_off + p->tot_blocks * nsub + g0 * 2) != 0 || fread(mn, 2, (size_t)nb, r->fs) != nb) { rc = -1; break; }
        if (seek64(r->fs, p->body_off + p->tot_blocks * (nsub + 2ull) + g0 * 2) != 0 || fread(sc, 2, (size_t)nb, r->fs) != nb) { rc = -1; break; }
        pqk_blocks(&S, bs, nb, idx, mn, sc);
    }
    free(idx); free(mn); free(sc);
    return rc;
}

int sn_rawk_rows(const sn_pack *p, sn_rd *r, const sn_tensor *t, int cols, const sn_mv *mv, int r0, int r1)
{
    (void)p;
    if (!t->raw || cols <= 0 || r1 <= r0 || !mv || mv->nv < 1 || mv->nv > SN_NV || (uint64_t)r1 * (uint64_t)cols > t->count) return -1;
    int rows_per = (1 << 20) / cols; if (rows_per < 1) rows_per = 1;
    uint16_t *buf = (uint16_t *)malloc((size_t)rows_per * (size_t)cols * 2);
    if (!buf) return -1;
    int rc = 0;
    for (int a = r0; a < r1 && !rc; a += rows_per) {
        int b = a + rows_per < r1 ? a + rows_per : r1;
        size_t n = (size_t)(b - a) * (size_t)cols;
        if (seek64(r->fs, t->raw_off + (uint64_t)a * (uint64_t)cols * 2) != 0 || fread(buf, 2, n, r->fs) != n) { rc = -1; break; }
        for (int row = a; row < b; row++) {
            const uint16_t *w = buf + (size_t)(row - a) * (size_t)cols;
            for (int v = 0; v < mv->nv; v++) {
                const float *x = mv->x[v];
                double acc = 0.0;
                for (int j = 0; j < cols; j++) acc += (double)sn_f16_lut[w[j]] * x[j];
                mv->y[v][row - r0] = (float)acc;
            }
        }
    }
    free(buf);
    return rc;
}
