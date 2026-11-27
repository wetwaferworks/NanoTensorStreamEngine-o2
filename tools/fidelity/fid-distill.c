/* fid-distill.c ... fidelity lab, method 5: a short distillation that
 * tunes only the codebooks and the frame scales of a quantized Sigma
 * model against the raw model's logits.
 *
 * Co-implemented by Cortana at JD's direction, 2026-11-27. Pure C99,
 * zero mutexes, bit-identical results for any --threads.
 *
 * Input: the raw model directory (for the small tensors and for any
 * tensor kept raw), a state directory written by fid-quant --state
 * (codes, books, scales of every quantized tensor), calibration token
 * sequences and the raw model's logits on them (fid-sigma --write-ref
 * with the same --gammas and --block). The codes never change; the
 * books (float16) and scales (float16) are trained with Adam on the
 * mean KL(raw || quantized) over the noised rows, with exact
 * gradients through the whole network (attention, RMSNorm, SwiGLU,
 * head), reconstructed weights rounded to bf16 in the forward pass as
 * the engine reads them (straight-through in the backward pass, as is
 * the bf16 key/value cache). The task and the arithmetic of the
 * forward pass are those of fid-sigma.
 *
 * Memory: the forward pass keeps each layer's input; the backward
 * pass recomputes one layer at a time. Steps take minibatches of
 * --batch sequences in order, cycling.
 *
 * Output: --out-state DIR receives the trained states (books and
 * scales rounded to float16); fid-quant --from-state DIR rebuilds a
 * variant model from them for scoring with fid-sigma.
 *
 * --check N: gradient check instead of training (no bf16 rounding of
 * weights or of the key/value cache, central finite differences of the
 * loss for N random parameters).
 *
 * Usage:
 *   fid-distill --dir RAW --state DIR --teacher IDS --ref F
 *               [--gammas 6,3,0,-3] [--block 32] [--seed 42]
 *               [--steps 8] [--batch 4] [--lr-books 1e-3]
 *               [--lr-scales 1e-3] [--out-state DIR] [--threads N]
 *               [--log FILE] [--check N]
 */
#include "fid-pq.h"

static int H, L, NH, NKV, HD = 128, FFN, V, ED;
static const double EPS = (double)1e-5f;
static int g_round = 1;

typedef struct { FidShard sh[8]; int ns; } Model;

static FidTensor *find_t(Model *m, const char *name, int *shard)
{
    for (int s = 0; s < m->ns; s++)
        for (int i = 0; i < m->sh[s].n; i++)
            if (!strcmp(m->sh[s].t[i].name, name)) { *shard = s; return &m->sh[s].t[i]; }
    return NULL;
}

static float *load_f(Model *m, const char *name, uint64_t need)
{
    int s = 0;
    FidTensor *t = find_t(m, name, &s);
    if (!t) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    if (need && t->count != need) { fprintf(stderr, "tensor %s has the wrong size\n", name); exit(1); }
    float *b = (float *)fid_alloc(t->count * sizeof(float));
    if (fid_tensor_read(&m->sh[s], t, 0, t->count, b) != 0) { fprintf(stderr, "read failed %s\n", name); exit(1); }
    return b;
}

/* ---------- parameters ---------- */

typedef struct {
    char name[160];
    int64_t r, c;
    int quant;
    PqState st;            /* quant: codes, books (trained), scales (trained) */
    float *s0;             /* initial scales: s = s0 * (1 + u) */
    float *u;
    float *raw;            /* !quant: fixed raw weights */
    double *gb1, *gb2, *gu;
    float *mb1, *vb1, *mb2, *vb2, *mu, *vu;   /* Adam moments */
} Par;

static Par *par_new(Model *m, const char *name, const char *sdir)
{
    Par *p = (Par *)fid_calloc(1, sizeof(Par));
    snprintf(p->name, sizeof(p->name), "%s", name);
    int s = 0;
    FidTensor *t = find_t(m, name, &s);
    if (!t || t->ndim != 2) { fprintf(stderr, "missing tensor %s\n", name); exit(1); }
    p->r = (int64_t)t->dims[0]; p->c = (int64_t)t->dims[1];
    char path[2200];
    snprintf(path, sizeof(path), "%s/%s.fidq", sdir, name);
    int rc = pq_state_read(path, &p->st);
    if (rc < 0) { fprintf(stderr, "bad state %s\n", path); exit(1); }
    if (rc == 1) { p->raw = load_f(m, name, 0); return p; }
    if (p->st.r != p->r || p->st.c != p->c) { fprintf(stderr, "state %s has the wrong shape\n", path); exit(1); }
    p->quant = 1;
    int64_t n1 = p->st.nsub * (int64_t)p->st.K * p->st.d, n2 = p->st.nsub * (int64_t)p->st.K2 * p->st.d;
    int64_t ns = p->r * p->st.nb;
    p->s0 = (float *)fid_alloc(ns * sizeof(float));
    memcpy(p->s0, p->st.s, ns * sizeof(float));
    p->u = (float *)fid_calloc(ns, sizeof(float));
    p->gb1 = (double *)fid_calloc(n1, sizeof(double));
    p->gb2 = n2 ? (double *)fid_calloc(n2, sizeof(double)) : NULL;
    p->gu = (double *)fid_calloc(ns, sizeof(double));
    p->mb1 = (float *)fid_calloc(n1, sizeof(float)); p->vb1 = (float *)fid_calloc(n1, sizeof(float));
    p->mb2 = n2 ? (float *)fid_calloc(n2, sizeof(float)) : NULL;
    p->vb2 = n2 ? (float *)fid_calloc(n2, sizeof(float)) : NULL;
    p->mu = (float *)fid_calloc(ns, sizeof(float)); p->vu = (float *)fid_calloc(ns, sizeof(float));
    return p;
}

/* full weight matrix as the forward pass uses it */
static float *par_weight(const Par *p)
{
    float *W = (float *)fid_alloc((size_t)(p->r * p->c) * sizeof(float));
    if (!p->quant) { memcpy(W, p->raw, (size_t)(p->r * p->c) * sizeof(float)); return W; }
    pq_recon_rows(&p->st, 0, p->r, 1, p->st.b1, p->st.b2, W);
    if (g_round)
        for (int64_t i = 0; i < p->r * p->c; i++) W[i] = fid_bf16_to_f32(fid_f32_to_bf16(W[i]));
    return W;
}

/* gradient of rows r0..r0+nr of W into the book and scale gradients.
 * Books: one item per subvector position (each item owns its book);
 * scales: one item per row. Fixed loop orders, so deterministic. */
typedef struct { Par *p; const float *dW; int64_t r0, nr; } PG;

static void pg_book(void *vc, int64_t sub)
{
    PG *g = (PG *)vc;
    Par *p = g->p;
    const PqState *st = &p->st;
    int d = st->d;
    double *gb1 = p->gb1 + sub * (int64_t)st->K * d;
    double *gb2 = st->K2 ? p->gb2 + sub * (int64_t)st->K2 * d : NULL;
    for (int64_t i = 0; i < g->nr; i++) {
        int64_t row = g->r0 + i;
        for (int64_t b = 0; b < st->nb; b++) {
            int64_t grp = b * st->nsub + sub, col0 = grp * d;
            double s = st->s[row * st->nb + b];
            const float *dw = g->dW + i * p->c + col0;
            double *e1 = gb1 + (int64_t)st->k1[row * st->ng + grp] * d;
            for (int j = 0; j < d; j++) e1[j] += s * dw[j];
            if (gb2) {
                double *e2 = gb2 + (int64_t)st->k2[row * st->ng + grp] * d;
                for (int j = 0; j < d; j++) e2[j] += s * dw[j];
            }
        }
    }
}

static void pg_scale(void *vc, int64_t i)
{
    PG *g = (PG *)vc;
    Par *p = g->p;
    const PqState *st = &p->st;
    int d = st->d;
    int64_t row = g->r0 + i;
    for (int64_t b = 0; b < st->nb; b++) {
        double acc = 0;
        for (int64_t sub = 0; sub < st->nsub; sub++) {
            int64_t grp = b * st->nsub + sub, col0 = grp * d;
            const float *c1 = st->b1 + (sub * (int64_t)st->K + st->k1[row * st->ng + grp]) * d;
            const float *c2 = st->K2 ? st->b2 + (sub * (int64_t)st->K2 + st->k2[row * st->ng + grp]) * d : NULL;
            const float *dw = g->dW + i * p->c + col0;
            for (int j = 0; j < d; j++) acc += (double)dw[j] * (c1[j] + (c2 ? c2[j] : 0.0f));
        }
        /* d loss / d u = d loss / d s * s0 */
        p->gu[row * st->nb + b] += acc * p->s0[row * st->nb + b];
    }
}

static void par_grad(Par *p, const float *dW, int64_t r0, int64_t nr)
{
    if (!p->quant) return;
    PG g = { p, dW, r0, nr };
    fid_par_for(p->st.nsub, pg_book, &g);
    fid_par_for(nr, pg_scale, &g);
}

static void par_zero(Par *p)
{
    if (!p->quant) return;
    memset(p->gb1, 0, sizeof(double) * p->st.nsub * p->st.K * p->st.d);
    if (p->gb2) memset(p->gb2, 0, sizeof(double) * p->st.nsub * p->st.K2 * p->st.d);
    memset(p->gu, 0, sizeof(double) * p->r * p->st.nb);
}

static void adam(float *x, float *m, float *v, const double *g, int64_t n, double lr, int t)
{
    const double b1 = 0.9, b2 = 0.999, eps = 1e-12;
    double c1 = 1.0 - pow(b1, t), c2 = 1.0 - pow(b2, t);
    for (int64_t i = 0; i < n; i++) {
        m[i] = (float)(b1 * m[i] + (1 - b1) * g[i]);
        v[i] = (float)(b2 * v[i] + (1 - b2) * g[i] * g[i]);
        double mh = m[i] / c1, vh = v[i] / c2;
        x[i] = (float)(x[i] - lr * mh / (sqrt(vh) + eps));
    }
}

static void par_step(Par *p, double lr_b, double lr_s, int t)
{
    if (!p->quant) return;
    int64_t n1 = p->st.nsub * (int64_t)p->st.K * p->st.d, n2 = p->st.nsub * (int64_t)p->st.K2 * p->st.d;
    int64_t ns = p->r * p->st.nb;
    adam(p->st.b1, p->mb1, p->vb1, p->gb1, n1, lr_b, t);
    if (n2) adam(p->st.b2, p->mb2, p->vb2, p->gb2, n2, lr_b, t);
    adam(p->u, p->mu, p->vu, p->gu, ns, lr_s, t);
    for (int64_t i = 0; i < ns; i++) p->st.s[i] = p->s0[i] * (1.0f + p->u[i]);
}

/* ---------- rows (as fid-sigma) ---------- */

typedef struct {
    int seq, pos, id, noisy, gi, blk;
    int64_t own_lo, own_hi, c_lo, c_hi;
} Row;

static Row *R;
static int64_t NR, NCL, NNO;
static int64_t *g_sbeg, *g_send;   /* per batch sequence: clean range */
static int64_t *g_nbeg, *g_nend;   /* per batch sequence: noisy range */
static int MAXK;

static float *g_cos, *g_sin;

static void rope_init(int max_pos)
{
    const double theta = 1000000.0, orig_max = 16384.0;
    const double factor = 4096.0 / orig_max;
    const double beta_fast = 32.0, beta_slow = 1.0;
    int half = HD / 2;
    double inv_freq[64];
    double corr = 2.0 * log(theta);
    double low_d = HD * log(orig_max / (beta_fast * 2.0 * 3.141592653589793)) / corr;
    double high_d = HD * log(orig_max / (beta_slow * 2.0 * 3.141592653589793)) / corr;
    int low = (int)floor(low_d); if (low < 0) low = 0;
    int high = (int)ceil(high_d); if (high > HD - 1) high = HD - 1;
    for (int i = 0; i < half; i++) {
        double pos_freq = pow(theta, (double)(2 * i) / HD);
        double inv_extrap = 1.0 / pos_freq, inv_interp = 1.0 / (factor * pos_freq);
        double ramp = ((double)i - low) / (high - low > 0.001 ? high - low : 0.001);
        if (ramp < 0.0) ramp = 0.0;
        if (ramp > 1.0) ramp = 1.0;
        double ef = 1.0 - ramp;
        inv_freq[i] = inv_interp * (1.0 - ef) + inv_extrap * ef;
    }
    g_cos = (float *)fid_alloc((size_t)max_pos * HD * sizeof(float));
    g_sin = (float *)fid_alloc((size_t)max_pos * HD * sizeof(float));
    for (int p = 0; p < max_pos; p++)
        for (int i = 0; i < half; i++) {
            double ang = inv_freq[i] * p;
            float c = (float)cos(ang), s = (float)sin(ang);
            g_cos[(size_t)p * HD + i] = c; g_cos[(size_t)p * HD + half + i] = c;
            g_sin[(size_t)p * HD + i] = s; g_sin[(size_t)p * HD + half + i] = s;
        }
}

static void rmsnorm_rows(float *out, float *inv_out, const float *x, const float *w, int64_t n, int dim)
{
    for (int64_t i = 0; i < n; i++) {
        const float *row = x + i * dim;
        float *o = out + i * dim;
        double ss = 0;
        for (int k = 0; k < dim; k++) ss += (double)row[k] * row[k];
        double inv = 1.0 / sqrt(ss / dim + EPS);
        if (inv_out) inv_out[i] = (float)inv;
        for (int k = 0; k < dim; k++) o[k] = (float)(row[k] * inv) * w[k];
    }
}

/* dx += d rmsnorm(x) given dy */
static void rmsnorm_back(float *dx, const float *x, const float *w, const float *inv,
                         const float *dy, int64_t n, int dim)
{
    for (int64_t i = 0; i < n; i++) {
        const float *xr = x + i * dim, *dr = dy + i * dim;
        double iv = inv[i], dot = 0;
        for (int k = 0; k < dim; k++) dot += (double)w[k] * dr[k] * xr[k];
        double c = iv * iv * iv * dot / dim;
        for (int k = 0; k < dim; k++) dx[i * dim + k] += (float)(iv * w[k] * dr[k] - xr[k] * c);
    }
}

static void rope_rows(float *X, int64_t n, int dim, int inverse)
{
    for (int64_t i = 0; i < n; i++) {
        const float *cs = g_cos + (size_t)R[i].pos * HD, *sn = g_sin + (size_t)R[i].pos * HD;
        for (int off = 0; off < dim; off += HD) {
            float *xh = X + i * dim + off, hold[64];
            for (int d = 0; d < HD / 2; d++) hold[d] = xh[d];
            for (int d = 0; d < HD / 2; d++) {
                float x1 = hold[d], x2 = xh[HD / 2 + d];
                if (!inverse) {
                    xh[d] = x1 * cs[d] - x2 * sn[d];
                    xh[HD / 2 + d] = x2 * cs[HD / 2 + d] + x1 * sn[HD / 2 + d];
                } else {
                    xh[d] = x1 * cs[d] + x2 * sn[HD / 2 + d];
                    xh[HD / 2 + d] = -x1 * sn[d] + x2 * cs[HD / 2 + d];
                }
            }
        }
    }
}

/* ---------- attention forward (stores probabilities) and backward ---------- */

typedef struct {
    const float *Q, *K, *Vv;
    const uint16_t *Kb, *Vb;
    float *Out, *P;
    const float *dOut;
    float *dQ, *DS, *dK, *dV;
} AttC;

static int64_t key_index(const Row *r, int64_t jr)
{
    int64_t nc = r->c_hi - r->c_lo;
    if (jr >= r->c_lo && jr < r->c_hi) return jr - r->c_lo;
    if (jr >= r->own_lo && jr < r->own_hi) return nc + (jr - r->own_lo);
    return -1;
}

static void att_fwd(void *vc, int64_t it)
{
    AttC *a = (AttC *)vc;
    int64_t i = it / NH;
    int h = (int)(it % NH);
    int kh = h / (NH / NKV), kvd = NKV * HD, qd = NH * HD;
    const Row *r = &R[i];
    const float *qv = a->Q + i * qd + h * HD;
    int64_t nc = r->c_hi - r->c_lo, tot = nc + (r->own_hi - r->own_lo);
    float *sc = a->P + it * (int64_t)MAXK;
    double mx = -1e30;
    for (int64_t j = 0; j < tot; j++) {
        double dot = 0.0;
        int allowed = 1;
        if (j < nc) {
            const uint16_t *kv = a->Kb + (r->c_lo + j) * kvd + kh * HD;
            const float *kf = a->K + (r->c_lo + j) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) dot += (double)qv[d] * (g_round ? fid_bf16_to_f32(kv[d]) : kf[d]);
        } else {
            int64_t jr = r->own_lo + (j - nc);
            if (!r->noisy && R[jr].pos > r->pos) allowed = 0;
            const float *kv = a->K + jr * kvd + kh * HD;
            if (allowed) for (int d = 0; d < HD; d++) dot += (double)qv[d] * kv[d];
        }
        if (!allowed) { sc[j] = -1e30f; continue; }
        double s = dot * (1.0 / sqrt((double)HD));
        sc[j] = (float)s;
        if (s > mx) mx = s;
    }
    double sum = 0.0;
    for (int64_t j = 0; j < tot; j++) {
        if (sc[j] <= -1e29f) { sc[j] = 0.0f; continue; }
        sc[j] = (float)exp((double)sc[j] - mx);
        sum += sc[j];
    }
    float *ov = a->Out + i * qd + h * HD;
    for (int d = 0; d < HD; d++) ov[d] = 0.0f;
    double inv = sum > 0.0 ? 1.0 / sum : 0.0;
    for (int64_t j = 0; j < tot; j++) {
        if (sc[j] == 0.0f) continue;
        double p = sc[j] * inv;
        if (j < nc) {
            const uint16_t *vv = a->Vb + (r->c_lo + j) * kvd + kh * HD;
            const float *vf = a->Vv + (r->c_lo + j) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) ov[d] += (float)(p * (g_round ? fid_bf16_to_f32(vv[d]) : vf[d]));
        } else {
            const float *vv = a->Vv + (r->own_lo + (j - nc)) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) ov[d] += (float)(p * vv[d]);
        }
        sc[j] = (float)p;   /* keep the probability */
    }
}

/* per query and head: dQ, and dS (scaled by 1/sqrt(HD)) for the keys */
static void att_bwd_q(void *vc, int64_t it)
{
    AttC *a = (AttC *)vc;
    int64_t i = it / NH;
    int h = (int)(it % NH);
    int kh = h / (NH / NKV), kvd = NKV * HD, qd = NH * HD;
    const Row *r = &R[i];
    int64_t nc = r->c_hi - r->c_lo, tot = nc + (r->own_hi - r->own_lo);
    const float *P = a->P + it * (int64_t)MAXK;
    float *DS = a->DS + it * (int64_t)MAXK;
    const float *dout = a->dOut + i * qd + h * HD;
    double sdp = 0.0;
    for (int64_t j = 0; j < tot; j++) {
        double dp = 0.0;
        if (P[j] != 0.0f) {
            if (j < nc) {
                const uint16_t *vv = a->Vb + (r->c_lo + j) * kvd + kh * HD;
            const float *vf = a->Vv + (r->c_lo + j) * kvd + kh * HD;
                for (int d = 0; d < HD; d++) dp += (double)dout[d] * (g_round ? fid_bf16_to_f32(vv[d]) : vf[d]);
            } else {
                const float *vv = a->Vv + (r->own_lo + (j - nc)) * kvd + kh * HD;
                for (int d = 0; d < HD; d++) dp += (double)dout[d] * vv[d];
            }
        }
        DS[j] = (float)dp;
        sdp += (double)P[j] * dp;
    }
    double scl = 1.0 / sqrt((double)HD);
    double dq[128];
    for (int d = 0; d < HD; d++) dq[d] = 0.0;
    for (int64_t j = 0; j < tot; j++) {
        double ds = (double)P[j] * ((double)DS[j] - sdp) * scl;
        DS[j] = (float)ds;
        if (ds == 0.0) continue;
        if (j < nc) {
            const uint16_t *kv = a->Kb + (r->c_lo + j) * kvd + kh * HD;
            const float *kf = a->K + (r->c_lo + j) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) dq[d] += ds * (g_round ? fid_bf16_to_f32(kv[d]) : kf[d]);
        } else {
            const float *kv = a->K + (r->own_lo + (j - nc)) * kvd + kh * HD;
            for (int d = 0; d < HD; d++) dq[d] += ds * kv[d];
        }
    }
    float *o = a->dQ + i * qd + h * HD;
    for (int d = 0; d < HD; d++) o[d] = (float)dq[d];
}

/* per key row and key head: gather dK and dV over every query that saw it */
static int64_t row_seq_index(int64_t jr);
static int g_nbatch;

static void att_bwd_kv(void *vc, int64_t it)
{
    AttC *a = (AttC *)vc;
    int64_t jr = it / NKV;
    int kh = (int)(it % NKV), groups = NH / NKV, kvd = NKV * HD, qd = NH * HD;
    int64_t bs = row_seq_index(jr);
    double dk[128], dv[128];
    for (int d = 0; d < HD; d++) { dk[d] = 0.0; dv[d] = 0.0; }
    for (int part = 0; part < 2; part++) {
        int64_t lo = part ? g_nbeg[bs] : g_sbeg[bs], hi = part ? g_nend[bs] : g_send[bs];
        for (int64_t i = lo; i < hi; i++) {
            int64_t idx = key_index(&R[i], jr);
            if (idx < 0) continue;
            for (int g = 0; g < groups; g++) {
                int h = kh * groups + g;
                int64_t qi = i * NH + h;
                float p = a->P[qi * MAXK + idx], ds = a->DS[qi * MAXK + idx];
                if (p == 0.0f && ds == 0.0f) continue;
                const float *qv = a->Q + i * qd + h * HD, *dout = a->dOut + i * qd + h * HD;
                for (int d = 0; d < HD; d++) { dk[d] += (double)ds * qv[d]; dv[d] += (double)p * dout[d]; }
            }
        }
    }
    for (int d = 0; d < HD; d++) {
        a->dK[jr * kvd + kh * HD + d] = (float)dk[d];
        a->dV[jr * kvd + kh * HD + d] = (float)dv[d];
    }
}

static int64_t row_seq_index(int64_t jr)
{
    for (int b = 0; b < g_nbatch; b++)
        if ((jr >= g_sbeg[b] && jr < g_send[b]) || (jr >= g_nbeg[b] && jr < g_nend[b])) return b;
    return 0;
}

/* ---------- token input ---------- */

typedef struct { int *ids; int n; } Seq;

static int load_seqs(const char *path, Seq **out)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int cap = 16, ns = 0, icap = 0, n = 0, have = 0, c;
    long v = 0;
    Seq *s = (Seq *)fid_calloc(cap, sizeof(Seq));
    int *ids = NULL;
    for (;;) {
        c = fgetc(f);
        if (c >= '0' && c <= '9') { v = v * 10 + (c - '0'); have = 1; continue; }
        if (have) {
            if (n == icap) { icap = icap ? icap * 2 : 1024; ids = (int *)realloc(ids, sizeof(int) * icap); }
            ids[n++] = (int)v; v = 0; have = 0;
        }
        if (c == '\n' || c == EOF) {
            if (n > 0) {
                if (ns == cap) { cap *= 2; s = (Seq *)realloc(s, sizeof(Seq) * cap); }
                s[ns].ids = ids; s[ns].n = n; ns++; ids = NULL; n = 0; icap = 0;
            }
            if (c == EOF) break;
        }
    }
    fclose(f);
    free(ids);
    *out = s;
    return ns;
}

static uint64_t t_mix(uint64_t x) { return fid_mix(x); }

static double t_normal(uint64_t key, uint64_t i)
{
    uint64_t a = t_mix(key ^ t_mix(2 * (i / 2)));
    uint64_t b = t_mix(key ^ t_mix(2 * (i / 2) + 1));
    double u1 = ((double)(a >> 11) + 0.5) * (1.0 / 9007199254740992.0);
    double u2 = (double)(b >> 11) * (1.0 / 9007199254740992.0);
    double r = sqrt(-2.0 * log(u1));
    double t = 2.0 * 3.14159265358979323846 * u2;
    return (i & 1) ? r * sin(t) : r * cos(t);
}

/* ---------- the model ---------- */

static Model M;
static Par **PL;          /* [L*7]: q k v o gate up down */
static Par *PHEAD;
static float **NIN, **NPOST, *NFIN;
static float *ENORM, *WIN, *WGAM;
static Seq *SEQS;
static int NSEQ, BLOCK, NG;
static double GAM[16];
static uint64_t SEED;
static int64_t *REF_OFF;    /* per sequence: first noisy row in the reference file */
static FILE *REFF;

/* batch state */
static float **HIN;       /* [L+1][NR*H] */
static double *ZB;        /* [NNO*ED] */
static int *BSEQ;

static void build_rows(const int *bseq, int nb)
{
    NCL = 0; NNO = 0;
    for (int b = 0; b < nb; b++) {
        int s = bseq[b], nbk = SEQS[s].n / BLOCK;
        NCL += (int64_t)nbk * BLOCK;
        NNO += (int64_t)(nbk - 1) * BLOCK * NG;
    }
    NR = NCL + NNO;
    free(R);
    R = (Row *)fid_calloc(NR, sizeof(Row));
    int64_t at = 0;
    for (int b = 0; b < nb; b++) {
        int s = bseq[b], nbk = SEQS[s].n / BLOCK;
        g_sbeg[b] = at; g_send[b] = at + (int64_t)nbk * BLOCK;
        for (int p = 0; p < nbk * BLOCK; p++) {
            Row *r = &R[at + p];
            r->seq = s; r->pos = p; r->id = SEQS[s].ids[p];
            int k = p / BLOCK;
            r->c_lo = at; r->c_hi = at + (int64_t)k * BLOCK;
            r->own_lo = at + (int64_t)k * BLOCK; r->own_hi = r->own_lo + BLOCK;
        }
        at += (int64_t)nbk * BLOCK;
    }
    for (int b = 0; b < nb; b++) {
        int s = bseq[b], nbk = SEQS[s].n / BLOCK;
        g_nbeg[b] = at;
        for (int k = 1; k < nbk; k++)
            for (int gi = 0; gi < NG; gi++) {
                int64_t lo = at;
                for (int i = 0; i < BLOCK; i++) {
                    Row *r = &R[at++];
                    r->seq = s; r->pos = k * BLOCK + i; r->id = SEQS[s].ids[k * BLOCK + i];
                    r->noisy = 1; r->gi = gi; r->blk = k;
                    r->c_lo = g_sbeg[b]; r->c_hi = g_sbeg[b] + (int64_t)k * BLOCK;
                    r->own_lo = lo; r->own_hi = lo + BLOCK;
                }
            }
        g_nend[b] = at;
    }
    g_nbatch = nb;
}

static void input_rows(float *h)
{
    int s0 = 0;
    FidTensor *emb = find_t(&M, "encoder.embed_tokens.weight", &s0);
    for (int64_t i = 0; i < NCL; i++)
        if (fid_tensor_read(&M.sh[s0], emb, (uint64_t)R[i].id * H, H, h + i * H) != 0) { fprintf(stderr, "embed read failed\n"); exit(1); }
    float *grow = (float *)fid_alloc(sizeof(float) * NG * H);
    for (int gi = 0; gi < NG; gi++) {
        float ge[64];
        for (int j = 0; j < 32; j++) {
            double fr = exp(-5.0 + (10.0 * j) / 31.0);
            ge[j] = (float)sin(fr * GAM[gi]); ge[32 + j] = (float)cos(fr * GAM[gi]);
        }
        for (int d = 0; d < H; d++) {
            double acc = 0;
            for (int k = 0; k < 64; k++) acc += (double)ge[k] * WGAM[d * 64 + k];
            grow[gi * H + d] = (float)acc;
        }
    }
    for (int64_t i = NCL; i < NR; i++) {
        Row *r = &R[i];
        double g = GAM[r->gi];
        double asq = 1.0 / (1.0 + exp(g)), ssq = 1.0 / (1.0 + exp(-g));
        double alpha = sqrt(asq), sigma = sqrt(ssq), zscale = 1.0 / sqrt(asq / ED + ssq);
        uint64_t key = t_mix(SEED ^ t_mix(((uint64_t)r->seq << 40) ^ ((uint64_t)r->blk << 20) ^ (uint64_t)r->gi));
        int ib = r->pos - r->blk * BLOCK;
        double *z = ZB + (i - NCL) * ED;
        float xin[64];
        for (int d = 0; d < ED; d++) {
            z[d] = alpha * ENORM[(size_t)r->id * ED + d] + sigma * t_normal(key, (uint64_t)ib * ED + d);
            xin[d] = (float)(z[d] * zscale);
        }
        for (int dd = 0; dd < H; dd++) {
            double acc = 0;
            for (int k = 0; k < ED; k++) acc += (double)xin[k] * WIN[dd * ED + k];
            h[i * H + dd] = (float)acc + grow[r->gi * H + dd];
        }
    }
    free(grow);
}

/* per-layer working set */
typedef struct {
    float *xn1, *Q, *K, *Vv, *att, *h2, *xn2, *G, *U, *A, *inv1, *inv2, *P;
    uint16_t *Kb, *Vb;
    float *W[7];
} LW;

static LW g_lw;

static void lw_alloc(void)
{
    int qd = NH * HD, kvd = NKV * HD;
    LW *w = &g_lw;
    memset(w, 0, sizeof(*w));
    w->xn1 = (float *)fid_alloc((size_t)NR * H * 4); w->Q = (float *)fid_alloc((size_t)NR * qd * 4);
    w->K = (float *)fid_alloc((size_t)NR * kvd * 4); w->Vv = (float *)fid_alloc((size_t)NR * kvd * 4);
    w->Kb = (uint16_t *)fid_alloc((size_t)NR * kvd * 2); w->Vb = (uint16_t *)fid_alloc((size_t)NR * kvd * 2);
    w->att = (float *)fid_alloc((size_t)NR * qd * 4); w->h2 = (float *)fid_alloc((size_t)NR * H * 4);
    w->xn2 = (float *)fid_alloc((size_t)NR * H * 4);
    w->G = (float *)fid_alloc((size_t)NR * FFN * 4); w->U = (float *)fid_alloc((size_t)NR * FFN * 4);
    w->A = (float *)fid_alloc((size_t)NR * FFN * 4);
    w->inv1 = (float *)fid_alloc((size_t)NR * 4); w->inv2 = (float *)fid_alloc((size_t)NR * 4);
    w->P = (float *)fid_alloc((size_t)NR * NH * MAXK * 4);
}

static void lw_free(void)
{
    LW *w = &g_lw;
    free(w->xn1); free(w->Q); free(w->K); free(w->Vv); free(w->Kb); free(w->Vb); free(w->att);
    free(w->h2); free(w->xn2); free(w->G); free(w->U); free(w->A); free(w->inv1); free(w->inv2); free(w->P);
}

/* forward of layer l from hin to hout (may alias? no: separate) */
static void layer_fwd(int l, const float *hin, float *hout)
{
    LW *w = &g_lw;
    int qd = NH * HD, kvd = NKV * HD;
    for (int k = 0; k < 7; k++) w->W[k] = par_weight(PL[l * 7 + k]);
    rmsnorm_rows(w->xn1, w->inv1, hin, NIN[l], NR, H);
    fid_gemm_nt(w->Q, qd, w->xn1, H, w->W[0], H, NR, H, qd);
    fid_gemm_nt(w->K, kvd, w->xn1, H, w->W[1], H, NR, H, kvd);
    fid_gemm_nt(w->Vv, kvd, w->xn1, H, w->W[2], H, NR, H, kvd);
    rope_rows(w->Q, NR, qd, 0);
    rope_rows(w->K, NR, kvd, 0);
    for (int64_t i = 0; i < NR * kvd; i++) { w->Kb[i] = fid_f32_to_bf16(w->K[i]); w->Vb[i] = fid_f32_to_bf16(w->Vv[i]); }
    AttC ac;
    memset(&ac, 0, sizeof(ac));
    ac.Q = w->Q; ac.K = w->K; ac.Vv = w->Vv; ac.Kb = w->Kb; ac.Vb = w->Vb; ac.Out = w->att; ac.P = w->P;
    fid_par_for(NR * NH, att_fwd, &ac);
    fid_gemm_nt(w->h2, H, w->att, qd, w->W[3], qd, NR, qd, H);
    for (int64_t i = 0; i < NR * H; i++) w->h2[i] += hin[i];
    rmsnorm_rows(w->xn2, w->inv2, w->h2, NPOST[l], NR, H);
    fid_gemm_nt(w->G, FFN, w->xn2, H, w->W[4], H, NR, H, FFN);
    fid_gemm_nt(w->U, FFN, w->xn2, H, w->W[5], H, NR, H, FFN);
    for (int64_t i = 0; i < NR * FFN; i++) w->A[i] = (w->G[i] / (1.0f + expf(-w->G[i]))) * w->U[i];
    fid_gemm_nt(hout, H, w->A, FFN, w->W[6], FFN, NR, FFN, H);
    for (int64_t i = 0; i < NR * H; i++) hout[i] += w->h2[i];
}

static void layer_free_w(void) { for (int k = 0; k < 7; k++) { free(g_lw.W[k]); g_lw.W[k] = NULL; } }

static float *transpose(const float *X, int64_t n, int64_t m)
{
    float *T = (float *)fid_alloc((size_t)(n * m) * sizeof(float));
    for (int64_t i0 = 0; i0 < n; i0 += 64)
        for (int64_t j0 = 0; j0 < m; j0 += 64)
            for (int64_t i = i0; i < n && i < i0 + 64; i++)
                for (int64_t j = j0; j < m && j < j0 + 64; j++) T[j * n + i] = X[i * m + j];
    return T;
}

/* dW = dY^T X for parameter p (rows out, cols in), fed to the book gradients */
static void weight_grad(Par *p, const float *dY, const float *X, int64_t out, int64_t in)
{
    if (!p->quant) return;
    float *dYT = transpose(dY, NR, out);
    float *dW = (float *)fid_calloc((size_t)(out * in), sizeof(float));
    fid_gemm_ex(dW, in, dYT, NR, X, in, out, NR, in, 0);
    par_grad(p, dW, 0, out);
    free(dW); free(dYT);
}

/* backward of layer l: dh holds d loss / d hout on entry, d loss / d hin
 * on exit. The working set holds this layer's forward pass. */
static void layer_bwd(int l, const float *hin, float *dh)
{
    LW *w = &g_lw;
    int qd = NH * HD, kvd = NKV * HD;
    /* down: hout = h2 + A Wd^T */
    float *dA = (float *)fid_calloc((size_t)NR * FFN, sizeof(float));
    fid_gemm(dA, FFN, dh, H, w->W[6], FFN, NR, H, FFN);
    weight_grad(PL[l * 7 + 6], dh, w->A, H, FFN);
    /* SwiGLU: dG into G's buffer, dU into U's (A no longer needed) */
    for (int64_t i = 0; i < NR * FFN; i++) {
        float g = w->G[i], u = w->U[i];
        float sg = 1.0f / (1.0f + expf(-g));
        float silu = g * sg;
        w->G[i] = dA[i] * u * (sg * (1.0f + g * (1.0f - sg)));
        w->U[i] = dA[i] * silu;
    }
    free(dA);
    float *dxn2 = (float *)fid_calloc((size_t)NR * H, sizeof(float));
    fid_gemm(dxn2, H, w->G, FFN, w->W[4], H, NR, FFN, H);
    fid_gemm(dxn2, H, w->U, FFN, w->W[5], H, NR, FFN, H);
    weight_grad(PL[l * 7 + 4], w->G, w->xn2, FFN, H);
    weight_grad(PL[l * 7 + 5], w->U, w->xn2, FFN, H);
    /* dh2 = dh + rmsnorm' (dh is reused as dh2) */
    rmsnorm_back(dh, w->h2, NPOST[l], w->inv2, dxn2, NR, H);
    free(dxn2);
    /* o: h2 = hin + att Wo^T */
    float *datt = (float *)fid_calloc((size_t)NR * qd, sizeof(float));
    fid_gemm(datt, qd, dh, H, w->W[3], qd, NR, H, qd);
    weight_grad(PL[l * 7 + 3], dh, w->att, H, qd);
    /* attention */
    float *dQ = (float *)fid_alloc((size_t)NR * qd * 4);
    float *dK = (float *)fid_alloc((size_t)NR * kvd * 4);
    float *dV = (float *)fid_alloc((size_t)NR * kvd * 4);
    float *DS = (float *)fid_alloc((size_t)NR * NH * MAXK * 4);
    AttC ac;
    memset(&ac, 0, sizeof(ac));
    ac.Q = w->Q; ac.K = w->K; ac.Vv = w->Vv; ac.Kb = w->Kb; ac.Vb = w->Vb; ac.P = w->P;
    ac.dOut = datt; ac.dQ = dQ; ac.DS = DS; ac.dK = dK; ac.dV = dV;
    fid_par_for(NR * NH, att_bwd_q, &ac);
    fid_par_for(NR * NKV, att_bwd_kv, &ac);
    free(DS); free(datt);
    rope_rows(dQ, NR, qd, 1);
    rope_rows(dK, NR, kvd, 1);
    float *dxn1 = (float *)fid_calloc((size_t)NR * H, sizeof(float));
    fid_gemm(dxn1, H, dQ, qd, w->W[0], H, NR, qd, H);
    fid_gemm(dxn1, H, dK, kvd, w->W[1], H, NR, kvd, H);
    fid_gemm(dxn1, H, dV, kvd, w->W[2], H, NR, kvd, H);
    weight_grad(PL[l * 7 + 0], dQ, w->xn1, qd, H);
    weight_grad(PL[l * 7 + 1], dK, w->xn1, kvd, H);
    weight_grad(PL[l * 7 + 2], dV, w->xn1, kvd, H);
    free(dQ); free(dK); free(dV);
    rmsnorm_back(dh, hin, NIN[l], w->inv1, dxn1, NR, H);
    free(dxn1);
}

/* ---------- loss and head ---------- */

#define HC 256
#define VT 4096

/* Forward the head on the final hidden rows; returns the summed KL and,
 * when dxf is not NULL, fills d loss / d xf and the head's gradient.
 * The loss is the mean over the noisy rows of KL(ref || q). */
static double head_pass(const float *xf, float *dxf, double *n_out, double *top1_out)
{
    Par *p = PHEAD;
    float *Wh = par_weight(p);           /* V x H */
    float *lg = (float *)fid_alloc((size_t)HC * V * 4);
    float *rl = (float *)fid_alloc((size_t)V * 4);
    float *dlT = dxf ? (float *)fid_alloc((size_t)VT * HC * 4) : NULL;
    float *dWt = dxf ? (float *)fid_alloc((size_t)VT * H * 4) : NULL;
    double kl_sum = 0, top1 = 0;
    for (int64_t r0 = 0; r0 < NNO; r0 += HC) {
        int64_t n = NNO - r0 < HC ? NNO - r0 : HC;
        fid_gemm_nt(lg, V, xf + r0 * H, H, Wh, H, n, H, V);
        for (int64_t i = 0; i < n; i++) {
            const Row *r = &R[NCL + r0 + i];
            double g = GAM[r->gi];
            double alpha = sqrt(1.0 / (1.0 + exp(g))), ssq = 1.0 / (1.0 + exp(-g));
            double prior = alpha / ssq;
            const double *z = ZB + (r0 + i) * ED;
            float zf[64];
            for (int d = 0; d < ED; d++) zf[d] = (float)z[d];
            float *l1 = lg + i * V;
            for (int v = 0; v < V; v++) {
                double acc = 0;
                for (int d = 0; d < ED; d++) acc += (double)zf[d] * ENORM[(size_t)v * ED + d];
                l1[v] += (float)(prior * (float)acc);
            }
            /* reference row: its place in the file */
            int b = 0;
            while (b < g_nbatch && !(NCL + r0 + i >= g_nbeg[b] && NCL + r0 + i < g_nend[b])) b++;
            int64_t fr = REF_OFF[r->seq] + (NCL + r0 + i - g_nbeg[b]);
            if (fid_seek(REFF, (uint64_t)fr * V * 4) != 0 || fread(rl, 4, V, REFF) != (size_t)V) {
                fprintf(stderr, "reference read failed\n"); exit(1); }
            double mq = -1e300, mr = -1e300;
            int aq = 0, ar = 0;
            for (int v = 0; v < V; v++) {
                if (l1[v] > mq) { mq = l1[v]; aq = v; }
                if (rl[v] > mr) { mr = rl[v]; ar = v; }
            }
            double sq = 0, sr = 0;
            for (int v = 0; v < V; v++) { sq += exp((double)l1[v] - mq); sr += exp((double)rl[v] - mr); }
            double lq = mq + log(sq), lr = mr + log(sr), kl = 0;
            for (int v = 0; v < V; v++) {
                double a = (double)rl[v] - lr, pr = exp(a);
                if (pr > 0) kl += pr * (a - ((double)l1[v] - lq));
                /* d KL / d logit = q - p_ref (scaled to the mean later) */
                l1[v] = (float)(exp((double)l1[v] - lq) - pr);
            }
            kl_sum += kl;
            top1 += (aq == ar);
        }
        if (!dxf) continue;
        double scale = 1.0 / (double)NNO;
        for (int64_t i = 0; i < n * V; i++) lg[i] = (float)(lg[i] * scale);
        /* d xf = dlogits Wh ; d Wh = dlogits^T xf, a vocabulary tile at a time */
        fid_gemm(dxf + r0 * H, H, lg, V, Wh, H, n, V, H);
        for (int v0 = 0; v0 < V; v0 += VT) {
            int nv = V - v0 < VT ? V - v0 : VT;
            for (int64_t i = 0; i < n; i++)
                for (int v = 0; v < nv; v++) dlT[(int64_t)v * n + i] = lg[i * V + v0 + v];
            memset(dWt, 0, (size_t)nv * H * 4);
            fid_gemm_ex(dWt, H, dlT, n, xf + r0 * H, H, nv, n, H, 0);
            par_grad(p, dWt, v0, nv);
        }
    }
    free(Wh); free(lg); free(rl); free(dlT); free(dWt);
    *n_out = (double)NNO;
    *top1_out = top1;
    return kl_sum;
}

/* full forward (and backward when grad) on the current batch; returns mean KL */
static double run_batch(int grad, double *top1_rate)
{
    float *h = HIN[0];
    input_rows(h);
    for (int l = 0; l < L; l++) {
        layer_fwd(l, HIN[l], HIN[l + 1]);
        layer_free_w();
    }
    float *xf = (float *)fid_alloc((size_t)NNO * H * 4);
    float *invf = (float *)fid_alloc((size_t)NNO * 4);
    rmsnorm_rows(xf, invf, HIN[L] + NCL * H, NFIN, NNO, H);
    float *dxf = grad ? (float *)fid_calloc((size_t)NNO * H, 4) : NULL;
    double n, t1;
    double kl = head_pass(xf, dxf, &n, &t1);
    if (top1_rate) *top1_rate = t1 / n;
    if (grad) {
        float *dh = (float *)fid_calloc((size_t)NR * H, 4);
        rmsnorm_back(dh + NCL * H, HIN[L] + NCL * H, NFIN, invf, dxf, NNO, H);
        for (int l = L - 1; l >= 0; l--) {
            float *hout = (float *)fid_alloc((size_t)NR * H * 4);
            layer_fwd(l, HIN[l], hout);   /* recompute this layer's working set */
            free(hout);
            layer_bwd(l, HIN[l], dh);
            layer_free_w();
        }
        free(dh);
    }
    free(xf); free(invf); free(dxf);
    return kl / n;
}

static double rand01(uint64_t *s) { *s = fid_mix(*s); return (double)(*s >> 11) * (1.0 / 9007199254740992.0); }

int main(int argc, char **argv)
{
    const char *dir = NULL, *sdir = NULL, *tpath = NULL, *rpath = NULL, *gam_s = "6,3,0,-3";
    const char *odir = NULL, *logp = NULL;
    int threads = 0, steps = 8, batch = 4, check = 0;
    double lr_b = 1e-3, lr_s = 1e-3;
    BLOCK = 32; SEED = 42;
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        int more = i + 1 < argc;
        if (!strcmp(a, "--dir") && more) dir = argv[++i];
        else if (!strcmp(a, "--state") && more) sdir = argv[++i];
        else if (!strcmp(a, "--teacher") && more) tpath = argv[++i];
        else if (!strcmp(a, "--ref") && more) rpath = argv[++i];
        else if (!strcmp(a, "--gammas") && more) gam_s = argv[++i];
        else if (!strcmp(a, "--block") && more) BLOCK = atoi(argv[++i]);
        else if (!strcmp(a, "--seed") && more) SEED = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(a, "--steps") && more) steps = atoi(argv[++i]);
        else if (!strcmp(a, "--batch") && more) batch = atoi(argv[++i]);
        else if (!strcmp(a, "--lr-books") && more) lr_b = atof(argv[++i]);
        else if (!strcmp(a, "--lr-scales") && more) lr_s = atof(argv[++i]);
        else if (!strcmp(a, "--out-state") && more) odir = argv[++i];
        else if (!strcmp(a, "--log") && more) logp = argv[++i];
        else if (!strcmp(a, "--threads") && more) threads = atoi(argv[++i]);
        else if (!strcmp(a, "--check") && more) check = atoi(argv[++i]);
        else { fprintf(stderr, "unknown or incomplete option %s\n", a); return 1; }
    }
    if (!dir || !sdir || !tpath || !rpath || (!odir && !check)) {
        fprintf(stderr, "usage: fid-distill --dir RAW --state DIR --teacher IDS --ref F --out-state DIR [options]\n");
        return 1;
    }
    g_fid_threads = threads > 0 ? threads : fid_cpu_count();
    if (check) g_round = 0;
    double t0 = fid_now();
    memset(&M, 0, sizeof(M));
    for (int s = 1; s <= 8; s++)
        for (int n = 1; n <= 8; n++) {
            char p[2200];
            snprintf(p, sizeof(p), "%s/model-%05d-of-%05d.safetensors", dir, s, n);
            FILE *f = fopen(p, "rb");
            if (!f) continue;
            fclose(f);
            if (fid_shard_open(&M.sh[M.ns], p) != 0) { fprintf(stderr, "cannot open %s\n", p); return 1; }
            M.ns++;
        }
    if (!M.ns) { fprintf(stderr, "no shards in %s\n", dir); return 1; }
    int sx = 0;
    FidTensor *te = find_t(&M, "encoder.embed_tokens.weight", &sx);
    FidTensor *tq = find_t(&M, "encoder.layers.0.self_attn.q_proj.weight", &sx);
    FidTensor *tk = find_t(&M, "encoder.layers.0.self_attn.k_proj.weight", &sx);
    FidTensor *tg = find_t(&M, "encoder.layers.0.mlp.gate_proj.weight", &sx);
    FidTensor *tE = find_t(&M, "replaid_embedding_matrix.matrix", &sx);
    if (!te || !tq || !tk || !tg || !tE) { fprintf(stderr, "not a Sigma-family model\n"); return 1; }
    V = (int)te->dims[0]; H = (int)te->dims[1]; NH = (int)tq->dims[0] / HD; NKV = (int)tk->dims[0] / HD;
    FFN = (int)tg->dims[0]; ED = (int)tE->dims[1];
    L = 0;
    for (int s = 0; s < M.ns; s++)
        for (int i = 0; i < M.sh[s].n; i++) {
            int li;
            if (sscanf(M.sh[s].t[i].name, "encoder.layers.%d.", &li) == 1 && li + 1 > L) L = li + 1;
        }
    {
        char b[256];
        snprintf(b, sizeof(b), "%s", gam_s);
        NG = 0;
        for (char *t = strtok(b, ","); t && NG < 16; t = strtok(NULL, ",")) GAM[NG++] = atof(t);
    }
    NSEQ = load_seqs(tpath, &SEQS);
    if (NSEQ <= 0) { fprintf(stderr, "no sequences in %s\n", tpath); return 1; }
    /* keep sequences with at least two blocks, in file order; reference offsets */
    REF_OFF = (int64_t *)fid_calloc(NSEQ, sizeof(int64_t));
    int *use = (int *)fid_alloc(sizeof(int) * NSEQ);
    int nuse = 0, maxlen = 0;
    int64_t off = 0;
    for (int s = 0; s < NSEQ; s++) {
        int nb = SEQS[s].n / BLOCK;
        REF_OFF[s] = off;
        if (nb < 2) continue;
        for (int i = 0; i < nb * BLOCK; i++)
            if (SEQS[s].ids[i] < 0 || SEQS[s].ids[i] >= V) { fprintf(stderr, "token id out of range\n"); return 1; }
        use[nuse++] = s;
        off += (int64_t)(nb - 1) * BLOCK * NG;
        if (nb * BLOCK > maxlen) maxlen = nb * BLOCK;
    }
    REFF = fopen(rpath, "rb");
    if (!REFF) { fprintf(stderr, "cannot open %s\n", rpath); return 1; }
    if (fid_seek(REFF, (uint64_t)off * V * 4 - 4) != 0 || fgetc(REFF) == EOF) {
        fprintf(stderr, "reference %s is shorter than these sequences and gammas need\n", rpath); return 1; }
    MAXK = maxlen;
    rope_init(maxlen + 8);
    /* small tensors */
    ENORM = load_f(&M, "replaid_embedding_matrix.matrix", (uint64_t)V * ED);
    for (int v = 0; v < V; v++) {
        double nrm = 0;
        for (int d = 0; d < ED; d++) nrm += (double)ENORM[v * ED + d] * ENORM[v * ED + d];
        double inv = 1.0 / (sqrt(nrm) + 1e-8);
        for (int d = 0; d < ED; d++) ENORM[v * ED + d] = (float)(ENORM[v * ED + d] * inv);
    }
    WIN = load_f(&M, "replaid_input_linear.weight", (uint64_t)H * ED);
    WGAM = load_f(&M, "replaid_gamma_linear.weight", (uint64_t)H * 64);
    NFIN = load_f(&M, "encoder.norm.weight", H);
    NIN = (float **)fid_calloc(L, sizeof(float *));
    NPOST = (float **)fid_calloc(L, sizeof(float *));
    PL = (Par **)fid_calloc((size_t)L * 7, sizeof(Par *));
    static const char *pn[7] = { "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
    int nquant = 0;
    for (int l = 0; l < L; l++) {
        char nm[256];
        snprintf(nm, sizeof(nm), "encoder.layers.%d.input_layernorm.weight", l); NIN[l] = load_f(&M, nm, H);
        snprintf(nm, sizeof(nm), "encoder.layers.%d.post_attention_layernorm.weight", l); NPOST[l] = load_f(&M, nm, H);
        for (int k = 0; k < 7; k++) {
            snprintf(nm, sizeof(nm), "encoder.layers.%d.%s.weight", l, pn[k]);
            PL[l * 7 + k] = par_new(&M, nm, sdir);
            nquant += PL[l * 7 + k]->quant;
        }
    }
    PHEAD = par_new(&M, "diffusion_head.weight", sdir);
    nquant += PHEAD->quant;
    if (!nquant) { fprintf(stderr, "no state files in %s\n", sdir); return 1; }
    int nb = batch < nuse ? batch : nuse;
    g_sbeg = (int64_t *)fid_calloc(nb, 8); g_send = (int64_t *)fid_calloc(nb, 8);
    g_nbeg = (int64_t *)fid_calloc(nb, 8); g_nend = (int64_t *)fid_calloc(nb, 8);
    BSEQ = (int *)fid_alloc(sizeof(int) * nb);
    FILE *lf = logp ? fopen(logp, "w") : NULL;
    printf("fid-distill: %d layers, %d quantized tensors, %d sequences (batch %d), gammas %s, threads %d, lr books %g scales %g\n",
           L, nquant, nuse, nb, gam_s, g_fid_threads, lr_b, lr_s);
    if (lf) fprintf(lf, "step\tbatch_first_seq\tkl_before\ttop1_before\tsecs\n");
    Par **all = (Par **)fid_alloc(sizeof(Par *) * (L * 7 + 1));
    for (int i = 0; i < L * 7; i++) all[i] = PL[i];
    all[L * 7] = PHEAD;
    int nall = L * 7 + 1;
    int cursor = 0;
    for (int step = 1; step <= (check ? 1 : steps); step++) {
        double ts = fid_now();
        for (int b = 0; b < nb; b++) BSEQ[b] = use[(cursor + b) % nuse];
        cursor = (cursor + nb) % nuse;
        build_rows(BSEQ, nb);
        HIN = (float **)fid_calloc(L + 1, sizeof(float *));
        for (int l = 0; l <= L; l++) HIN[l] = (float *)fid_alloc((size_t)NR * H * 4);
        ZB = (double *)fid_alloc((size_t)(NNO ? NNO : 1) * ED * sizeof(double));
        lw_alloc();
        for (int i = 0; i < nall; i++) par_zero(all[i]);
        double t1r;
        double kl = run_batch(1, &t1r);
        if (check) {
            /* finite differences on random book entries and scales */
            uint64_t rs = 12345;
            printf("check: loss %.9f\n", kl);
            for (int c = 0; c < check; c++) {
                Par *p;
                do { p = all[(int)(rand01(&rs) * nall)]; } while (!p->quant);
                int is_scale = rand01(&rs) < 0.3;
                int64_t n1 = p->st.nsub * (int64_t)p->st.K * p->st.d, ns = p->r * p->st.nb;
                int64_t k = (int64_t)(rand01(&rs) * (is_scale ? ns : n1));
                double ga = is_scale ? p->gu[k] : p->gb1[k];
                double e = 1e-2;
                double base = is_scale ? p->u[k] : p->st.b1[k], lp, lm;
                for (int sgn = 0; sgn < 2; sgn++) {
                    double val = base + (sgn ? -e : e);
                    if (is_scale) { p->u[k] = (float)val; p->st.s[k] = p->s0[k] * (1.0f + p->u[k]); }
                    else p->st.b1[k] = (float)val;
                    double l2 = run_batch(0, NULL);
                    if (sgn) lm = l2; else lp = l2;
                }
                if (is_scale) { p->u[k] = (float)base; p->st.s[k] = p->s0[k]; } else p->st.b1[k] = (float)base;
                double gn = (lp - lm) / (2 * e);
                printf("check %-44s %s[%lld] analytic %+.6e numeric %+.6e rel %.3e\n", p->name,
                       is_scale ? "scale" : "book", (long long)k, ga, gn,
                       fabs(ga - gn) / (fabs(gn) + fabs(ga) + 1e-20));
            }
        } else {
            for (int i = 0; i < nall; i++) par_step(all[i], lr_b, lr_s, step);
            printf("step %2d: seqs %d.., kl %.6f top1 %.4f (before the step), %.1f s, wall %.1f s\n",
                   step, BSEQ[0], kl, t1r, fid_now() - ts, fid_now() - t0);
            fflush(stdout);
            if (lf) { fprintf(lf, "%d\t%d\t%.6f\t%.4f\t%.1f\n", step, BSEQ[0], kl, t1r, fid_now() - ts); fflush(lf); }
        }
        lw_free();
        for (int l = 0; l <= L; l++) free(HIN[l]);
        free(HIN); free(ZB);
    }
    if (odir) {
        uint64_t dg = FID_FNV_SEED;
        for (int i = 0; i < nall; i++) {
            Par *p = all[i];
            if (!p->quant) continue;
            int64_t n1 = p->st.nsub * (int64_t)p->st.K * p->st.d, n2 = p->st.nsub * (int64_t)p->st.K2 * p->st.d;
            for (int64_t k = 0; k < n1; k++) p->st.b1[k] = fid_round_f16(p->st.b1[k]);
            for (int64_t k = 0; k < n2; k++) p->st.b2[k] = fid_round_f16(p->st.b2[k]);
            for (int64_t k = 0; k < p->r * p->st.nb; k++) p->st.s[k] = fid_round_f16(p->st.s[k]);
            dg = fid_fnv(p->st.b1, (size_t)n1 * 4, dg);
            dg = fid_fnv(p->st.s, (size_t)(p->r * p->st.nb) * 4, dg);
            char path[2200];
            snprintf(path, sizeof(path), "%s/%s.fidq", odir, p->name);
            if (pq_state_write(path, &p->st) != 0) { fprintf(stderr, "cannot write %s\n", path); return 1; }
        }
        printf("wrote %d states to %s, digest %016llx\n", nquant, odir, (unsigned long long)dg);
    }
    if (lf) fclose(lf);
    fclose(REFF);
    printf("wall %.1f s\n", fid_now() - t0);
    return 0;
}
