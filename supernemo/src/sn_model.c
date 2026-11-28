/* sn_model.c - see sn_model.h. Pure C99, zero mutexes (Little Nemo pool). */
#include "sn_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sn_moe.h"
#include "sn_pack.h"
#include "sn_cuda.h"
#include "sn_gemm.h"
#include "sn_pqk.h"
#include "../../littlenemo/src/attn.h"
#include "../../littlenemo/src/mamba.h"
#include "../../littlenemo/src/pool.h"

/* ------------------------------------------------------------ config */

static const char *find_after(const char *from, const char *key)
{
    char k[128]; snprintf(k, sizeof(k), "\"%s\"", key);
    const char *p = strstr(from, k);
    if (!p) return NULL;
    p = strchr(p + strlen(k), ':');
    return p ? p + 1 : NULL;
}

static int cfg_int(const char *text, const char *from, const char *key, int *out)
{
    (void)text;
    const char *p = find_after(from, key);
    if (!p) { fprintf(stderr, "sn_geom: config has no \"%s\"\n", key); return -1; }
    *out = (int)strtol(p, NULL, 10);
    return 0;
}

int sn_geom_load(sn_geom *g, const char *path)
{
    memset(g, 0, sizeof(*g));
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "sn_geom: cannot open %s\n", path); return -1; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *t = (char *)malloc((size_t)sz + 1);
    if (!t || fread(t, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(t); return -1; }
    t[sz] = 0; fclose(f);
    const char *L = strstr(t, "\"llm_config\"");
    if (!L) L = t;
    int rc = 0;
    int heads_kv = 0, norm_topk_i = 1;
    rc |= cfg_int(t, L, "hidden_size", &g->hidden);
    rc |= cfg_int(t, L, "vocab_size", &g->vocab);
    rc |= cfg_int(t, L, "mamba_num_heads", &g->m_heads);
    rc |= cfg_int(t, L, "mamba_head_dim", &g->m_head_dim);
    rc |= cfg_int(t, L, "ssm_state_size", &g->m_state);
    rc |= cfg_int(t, L, "n_groups", &g->m_groups);
    rc |= cfg_int(t, L, "num_attention_heads", &g->a_heads);
    rc |= cfg_int(t, L, "num_key_value_heads", &heads_kv);
    rc |= cfg_int(t, L, "head_dim", &g->a_hd);
    rc |= cfg_int(t, L, "n_routed_experts", &g->moe_experts);
    rc |= cfg_int(t, L, "num_experts_per_tok", &g->moe_topk);
    rc |= cfg_int(t, L, "moe_intermediate_size", &g->moe_inter);
    rc |= cfg_int(t, L, "moe_latent_size", &g->moe_latent);
    rc |= cfg_int(t, L, "moe_shared_expert_intermediate_size", &g->moe_shared);
    if (rc) { free(t); return -1; }
    g->a_kv = heads_kv;
    const char *p = find_after(L, "routed_scaling_factor");
    g->moe_scale = p ? (float)strtod(p, NULL) : 1.0f;
    p = find_after(L, "norm_topk_prob");
    if (p) { while (*p == ' ') p++; norm_topk_i = strncmp(p, "true", 4) == 0; }
    g->moe_norm_topk = norm_topk_i;
    p = find_after(L, "layer_norm_epsilon");
    g->eps = p ? (float)strtod(p, NULL) : 1e-5f;
    {   /* the MTP head: one module, an attention block then a MoE block */
        int nmtp = 0; const char *pm = find_after(L, "num_nextn_predict_layers"); if (pm) nmtp = (int)strtol(pm, NULL, 10);
        const char *mb = strstr(L, "\"mtp_layers_block_type\"");
        const char *mo = mb ? strchr(mb, '[') : NULL, *mc = mo ? strchr(mo, ']') : NULL;
        if (nmtp >= 1 && mo && mc) {
            const char *a1 = strstr(mo, "attention"), *e1 = strstr(mo, "moe");
            g->mtp = a1 && e1 && a1 < mc && e1 < mc && a1 < e1 ? 1 : 0;
        }
    }
    g->m_inner = g->m_heads * g->m_head_dim;
    g->m_conv_dim = g->m_inner + 2 * g->m_groups * g->m_state;
    g->m_in_proj = g->m_inner + g->m_conv_dim + g->m_heads;
    const char *lb = strstr(L, "\"layers_block_type\"");
    const char *open = lb ? strchr(lb, '[') : NULL, *close = open ? strchr(open, ']') : NULL;
    if (!open || !close) { fprintf(stderr, "sn_geom: no layers_block_type\n"); free(t); return -1; }
    int n = 0;
    for (const char *q = open; q < close; q++) {
        if (*q != '"') continue;
        const char *e = strchr(q + 1, '"');
        if (!e || e > close) break;
        char c = strncmp(q + 1, "mamba", 5) == 0 ? 'M' : strncmp(q + 1, "moe", 3) == 0 ? 'E' : strncmp(q + 1, "attention", 9) == 0 ? 'A' : '?';
        if (c == '?' || n >= SN_MAX_LAYERS) { fprintf(stderr, "sn_geom: bad layer type\n"); free(t); return -1; }
        g->types[n++] = c;
        q = e;
    }
    g->n_layers = n;
    free(t);
    if (n == 0 || g->m_groups <= 0 || g->m_heads % g->m_groups || g->m_inner % g->m_groups
        || g->a_heads % g->a_kv || g->moe_topk > g->moe_experts) {
        fprintf(stderr, "sn_geom: inconsistent geometry\n"); return -1;
    }
    return 0;
}

/* ------------------------------------------------------------- model */

/* The DTC-versus-direct choice must not depend on how many lanes are running, or results would differ between machines. */
#define SN_DTC_LANES 8

typedef struct {
    sn_rd rd;
    float *book;       /* entries*block floats */
    float *mid;        /* SN_NV * moe_inter */
    double *pmid;      /* SN_NV * (moe_inter + 1) */
    float *ew, *G, *U; /* batched prefill: expanded expert matrix, gathered inputs / outputs, hidden */
    size_t ew_cap, g_cap, u_cap;
    float *T; size_t t_cap;   /* DTC tables (grown on demand) */
    int ok;
} lane_t;

typedef struct {
    const sn_tensor *norm;              /* raw */
    float *norm_w;                      /* resident, hidden */
    /* mamba */
    const sn_tensor *in_proj, *out_proj;
    float *conv_w, *conv_b, *a_log, *dvec, *dt_bias, *gnorm_w;
    /* attention */
    const sn_tensor *q, *k, *v, *o;
    /* moe */
    const sn_tensor *fc1, *fc2, *sh_up, *sh_dn, *gate_t;
    uint16_t *gate_w;                   /* resident fp16 [E][hidden] */
    float *gate_bias;
    const sn_tensor **eup, **edn;       /* [E] */
    /* MTP head: the first layer carries the fusion (eh_proj, enorm, hnorm), the last the closing norm */
    const sn_tensor *eh;
    float *enorm_w, *hnorm_w, *fin_w;
} layer_t;

/* Hot-expert cache entry: an expert's two packed matrices kept in RAM (byte copies of the pack, so every kernel returns
 * identical bits from them) and optionally mirrored in VRAM. uses counts tokens that chose the expert (deterministic). */
typedef struct { sn_pq_res up, dn; unsigned char ram, gpu, pend; uint32_t uses; uint32_t epoch; int rpos; int pb; } ecent;   /* pb: pinned block + 1 (0: malloc'd copy) */

struct sn_model {
    sn_geom g;
    sn_pack pack;
    sn_rd main_rd;
    nemo_pool pool;
    int nthreads;
    lane_t *lanes;
    layer_t *L;
    int n_all;                          /* n_layers, plus the 2 MTP layers when the head is present */
    const sn_tensor *emb, *head, *normf_t;
    float *normf_w;
    nemo_mamba_state *mam;
    nemo_attn_cache *att;
    int max_pos, n_pos, state_valid;
    float *gsc;
    double *P;                          /* SN_NV * (maxcols + 1) prefix sums */
    int maxcols;
    float *book_shared;
    float *Tbuf; size_t tbuf_cap;       /* DTC tables of the dense matrices */
    int dtc;                            /* 0 off, 1 auto (default), 2 force where the geometry allows */
    sn_layer_cb on_layer; sn_moe_cb on_moe; void *hook_user;
    sn_cuda *cuda;
    sn_gemm *gm;
    /* Mamba snapshots for speculative rollback: snap_ssm[t][layer] is the state AFTER batch token t */
    float **snap_ssm[SN_NV], **snap_conv[SN_NV];
    int snap_n;
    /* MTP / speculative decoding */
    int has_mtp, mtp_on;
    float *hist_hn; int *hist_tok;      /* per position: post-norm_f hidden, token (-1 = an image embedding) */
    int mtp_done, mtp_slots;
    float *spec_logits;
    float *probe;                       /* optional: hidden of the last token after each layer (tests) */
    /* hot-expert cache */
    ecent *ec; int *rlist; int nrlist;  /* ec[layer * experts + expert]; rlist = RAM-resident keys */
    uint64_t ec_budget, ec_used, ec_reserved;
    uint32_t epoch;
    int gpu_tier;
    float *gmid; double *gpmid;         /* host scratch for GPU-served experts */
    uint64_t hits_ram, hits_gpu, misses, admits, evicts, gpu_puts, gpu_drops;
    long spec_cycles, spec_drafted, spec_accepted;
    /* GPU expert stream (sn_cuda_xs_*): every routed expert goes to the card compressed and is computed there */
    int xs_on;
    int *xs_oidx, *xs_list, *xs_slot; unsigned char *xs_done;
    int xs_nopf, xs_nopin, xs_nocap;    /* the three speed-ups below are on unless switched off (sn_model_stream_options, SN_XS_*=0) */
    int *xs_next; float *xs_ph, *xs_ptw; int *xs_ptop, *xs_pfx, *xs_pfl_keys; int xs_pfn, xs_pfli;   /* early fetch for the next MoE layer */
    uint64_t xs_direct, xs_pf_sent, xs_pf_right;
    struct sn_prof *prof; int prof_state;   /* optional profile (env SN_PROF=1); measurement only */
};

/* ------------------------------------------------------------------ profile
 * SN_PROF=1 turns it on. It times each kind of work, measures how well three cheap predictors would have
 * guessed the router's choice (same layer last token, next layer's router run one layer early, the layer's most
 * used experts so far), times the stages of the GPU expert stream, and counts MTP draft hits by position. It reads
 * state and writes only its own counters, so results are unchanged. */
enum { PC_NORM, PC_MIN, PC_MCORE, PC_MOUT, PC_QKV, PC_ACORE, PC_O, PC_ROUTE, PC_FC1, PC_EXP, PC_FC2, PC_SHARED, PC_HEAD, PC_N };
static const char *pc_name[PC_N] = { "norms and residual adds", "mamba in_proj", "mamba scan", "mamba out_proj", "attention q k v",
    "attention core", "attention o", "router", "moe fc1 (to latent)", "routed experts", "moe fc2 (from latent)", "shared expert", "lm_head" };
#define PROF_MISS 48
typedef struct sn_prof {
    double ms[2][PC_N]; long calls[2][PC_N], on_card[2][PC_N]; long tokens[2];
    int nl, E, K; int *next_moe, *prev, *early; unsigned char *has_prev, *has_early; uint32_t *cnt; float *tmp, *tw;
    long *n_tok, *n_prev, *n_early, *ov_prev, *ov_early, *ov_hot, *ov_union, *sz_union;
    double xs_wait, xs_prep, xs_issue, xs_end, xs_pref; long xs_layers, xs_hit, xs_mode[4], xs_direct;
    long pos_tried[SN_NV], pos_ok[SN_NV]; int miss_d[PROF_MISS], miss_a[PROF_MISS], miss_pos[PROF_MISS], nmiss;
} sn_prof;

static void prof_free(sn_prof *p);
static double pnow(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

static int lane_of(int n_items, int nthreads, int begin)
{
    int lanes = nthreads < n_items ? nthreads : n_items;
    for (int l = 0; l < lanes; l++)
        if ((int)((long)l * n_items / lanes) == begin) return l;
    return 0;
}

static void *xalloc(size_t n) { void *p = malloc(n ? n : 1); return p; }

static int dtc_wanted(const sn_model *m, int rows, int cols, int nv, int lanes)
{
    if (!m->dtc || sn_dtc_geom(&m->pack, cols, NULL, NULL) != 0 || m->pack.entries != 256) return 0;
    return m->dtc == 2 || sn_dtc_pays(&m->pack, rows, cols, nv, lanes);
}

/* ---- parallel dense linear on nv vectors: rows split across lanes; one pass over the packed data ---- */
typedef struct {
    sn_model *m; const sn_tensor *t; const float *book; int rows, cols, nv, ng; int fail;
    sn_mv mv;
    float *Y;
} lin_ctx;

static void lin_rows_job(void *vc, int b, int e)
{
    lin_ctx *c = (lin_ctx *)vc;
    sn_model *m = c->m;
    lane_t *ln = &m->lanes[lane_of(c->rows, m->nthreads, b)];
    sn_mv mv = c->mv;
    for (int v = 0; v < c->nv; v++) mv.y[v] = c->Y + (size_t)v * c->rows + b;
    int rc = c->t->raw ? sn_rawk_rows(&m->pack, &ln->rd, c->t, c->cols, &mv, b, e)
                       : sn_pqk_rows(&m->pack, &ln->rd, c->t, c->book, c->cols, &mv, b, e);
    if (rc) c->fail = 1;
}

static void lin_build_job(void *vc, int b, int e)
{
    lin_ctx *c = (lin_ctx *)vc;
    for (int i = b; i < e; ) {
        int v = i / c->ng, g0 = i - v * c->ng, g1 = c->ng - g0 < e - i ? c->ng : g0 + (e - i);
        sn_dtc_build(&c->m->pack, c->book, c->cols, c->mv.x[v], g0, g1, (float *)c->mv.T[v]);
        i += g1 - g0;
    }
}

static int lin_n(sn_model *m, const sn_tensor *t, const float *X, int nv, float *Y, int rows, int cols)
{
    if ((uint64_t)rows * (uint64_t)cols != t->count || nv < 1 || nv > SN_NV) return -1;
    /* resident on the GPU: same math from the packed form there; any failure falls through to the CPU lanes */
    if (m->cuda && sn_cuda_resident(m->cuda, &m->pack, t)) {
        int ok = 1;
        /* tall matrices: tables + lookups on the device (bit-identical to the CPU DTC kernel); short ones: the direct kernel */
        int dtc_gpu = !t->raw && m->dtc && (m->dtc == 2 || rows >= 4096) && cols <= 8192 && sn_dtc_geom(&m->pack, cols, NULL, NULL) == 0;
        for (int v = 0; v < nv && ok; v++) {
            const float *xv = X + (size_t)v * cols; float *yv = Y + (size_t)v * rows;
            if (dtc_gpu) {
                double *Pv = m->P + (size_t)v * ((size_t)m->maxcols + 1);
                Pv[0] = 0.0; for (int j = 0; j < cols; j++) Pv[j + 1] = Pv[j] + (double)xv[j];
                ok = sn_cuda_dtc_rows(m->cuda, &m->pack, t, cols, rows, xv, Pv, yv) == 0;
            } else ok = sn_cuda_matvec(m->cuda, &m->pack, t, cols, rows, xv, yv) == 0;
        }
        if (ok) return 0;
    }
    lin_ctx c; memset(&c, 0, sizeof c);
    c.m = m; c.t = t; c.rows = rows; c.cols = cols; c.nv = nv; c.Y = Y; c.mv.nv = nv;
    for (int v = 0; v < nv; v++) c.mv.x[v] = X + (size_t)v * cols;
    if (!t->raw) {
        if (sn_load_book(&m->pack, &m->main_rd, t, m->book_shared) != 0) return -1;
        c.book = m->book_shared;
        for (int v = 0; v < nv; v++) {
            double *Pv = m->P + (size_t)v * ((size_t)m->maxcols + 1);
            Pv[0] = 0.0;
            for (int j = 0; j < cols; j++) Pv[j + 1] = Pv[j] + (double)X[(size_t)v * cols + j];
            c.mv.P[v] = Pv;
        }
        if (dtc_wanted(m, rows, cols, nv, SN_DTC_LANES)) {
            size_t tf = sn_dtc_floats(&m->pack, cols), need_f = tf * (size_t)nv;
            if (m->tbuf_cap < need_f) { float *p = (float *)realloc(m->Tbuf, need_f * 4); if (!p) return -1; m->Tbuf = p; m->tbuf_cap = need_f; }
            for (int v = 0; v < nv; v++) c.mv.T[v] = m->Tbuf + (size_t)v * tf;
            c.ng = cols >> 2;
            nemo_pool_for(&m->pool, nv * c.ng, lin_build_job, &c);
        }
    }
    nemo_pool_for(&m->pool, rows, lin_rows_job, &c);
    return c.fail ? -1 : 0;
}

/* ---- MoE, fused form: experts split across lanes; each expert serves all the batch tokens that chose it ---- */
typedef struct {
    sn_model *m; const layer_t *l; const float *lat; const double *plat; float *eo;
    const int *used; int n_used; const int *start, *tok, *slot; int fail;
    int li; const int *emode; ecent *epend;   /* 0 stream, 1 RAM-resident, 2 served by the GPU already, 3 load a resident copy while computing */
    const unsigned char *xdone;               /* 1: the GPU expert stream already wrote this expert's outputs */
} fx_ctx;

/* y[v] = W xs[v] for one expert matrix, on this lane (tables built here, never shared) */
static int expert_mat(sn_model *m, lane_t *ln, const sn_tensor *t, const sn_pq_res *res, int rows, int cols, int nv,
                      const float *const *xs, const double *const *ps, float *const *ys)
{
    const float *book = ln->book;
    if (res) book = res->book;
    else if (sn_load_book(&m->pack, &ln->rd, t, ln->book) != 0) return -1;
    sn_mv mv = SN_MV_INIT(nv);
    for (int v = 0; v < nv; v++) { mv.x[v] = xs[v]; mv.P[v] = ps[v]; mv.y[v] = ys[v]; }
    if (dtc_wanted(m, rows, cols, nv, 1)) {
        size_t tf = sn_dtc_floats(&m->pack, cols), need_f = tf * (size_t)nv;
        if (ln->t_cap < need_f) { float *p = (float *)realloc(ln->T, need_f * 4); if (!p) return -1; ln->T = p; ln->t_cap = need_f; }
        for (int v = 0; v < nv; v++) { mv.T[v] = ln->T + (size_t)v * tf; sn_dtc_build(&m->pack, book, cols, xs[v], 0, cols >> 2, ln->T + (size_t)v * tf); }
    }
    return res ? sn_pqk_rows_res(&m->pack, res, t, cols, &mv, 0, rows) : sn_pqk_rows(&m->pack, &ln->rd, t, book, cols, &mv, 0, rows);
}

static void fx_job(void *vc, int b, int e)
{
    fx_ctx *c = (fx_ctx *)vc;
    sn_model *m = c->m;
    const sn_geom *g = &m->g;
    int Lt = g->moe_latent, I = g->moe_inter, K = g->moe_topk;
    lane_t *ln = &m->lanes[lane_of(c->n_used, m->nthreads, b)];
    for (int i = b; i < e; i++) {
        int ex = c->used[i], cnt = c->start[i + 1] - c->start[i];
        int mode = c->emode ? c->emode[i] : 0;
        if (mode == 2 || (c->xdone && c->xdone[i])) continue;
        const sn_tensor *up = c->l->eup[ex], *dn = c->l->edn[ex];
        const sn_pq_res *rup = NULL, *rdn = NULL;
        if (mode == 1) { const ecent *e = &m->ec[(size_t)c->li * g->moe_experts + (size_t)ex]; rup = &e->up; rdn = &e->dn; }
        else if (mode == 3) {
            ecent *pe = &c->epend[i];
            if (sn_res_load(&m->pack, &ln->rd, up, &pe->up) == 0 && sn_res_load(&m->pack, &ln->rd, dn, &pe->dn) == 0) { pe->pend = 1; rup = &pe->up; rdn = &pe->dn; }
            else { sn_res_free(&pe->up); sn_res_free(&pe->dn); pe->pend = 0; }
        }
        const float *xs[SN_NV], *mx[SN_NV]; const double *ps[SN_NV], *pm[SN_NV]; float *ym[SN_NV], *yo[SN_NV];
        for (int v = 0; v < cnt; v++) {
            int tk = c->tok[c->start[i] + v], sl = c->slot[c->start[i] + v];
            xs[v] = c->lat + (size_t)tk * Lt; ps[v] = c->plat + (size_t)tk * (Lt + 1);
            ym[v] = ln->mid + (size_t)v * I; mx[v] = ym[v]; pm[v] = ln->pmid + (size_t)v * (I + 1);
            yo[v] = c->eo + ((size_t)tk * K + (size_t)sl) * Lt;
        }
        if (expert_mat(m, ln, up, rup, I, Lt, cnt, xs, ps, ym) != 0) { c->fail = 1; return; }
        for (int v = 0; v < cnt; v++) {
            float *mid = ym[v]; double *pp = ln->pmid + (size_t)v * (I + 1);
            for (int j = 0; j < I; j++) mid[j] = sn_relu2(mid[j]);
            pp[0] = 0.0;
            for (int j = 0; j < I; j++) pp[j + 1] = pp[j] + (double)mid[j];
        }
        if (expert_mat(m, ln, dn, rdn, Lt, I, cnt, mx, pm, yo) != 0) { c->fail = 1; return; }
    }
}

typedef struct { const uint16_t *w; const float *x; float *s; int H; } gate_ctx;

static void gate_job(void *vc, int b, int e)
{
    gate_ctx *c = (gate_ctx *)vc;
    for (int ex = b; ex < e; ex++) {
        const uint16_t *w = c->w + (size_t)ex * (size_t)c->H;
        double a = 0.0;
        for (int j = 0; j < c->H; j++) a += (double)sn_f16_lut[w[j]] * c->x[j];
        c->s[ex] = sn_sigmoid((float)a);
    }
}

/* Router: sigmoid scores on the resident fp16 gate, choice on score + e_score_correction_bias, weights from the raw
 * scores, normalised, times routed_scaling_factor. top/tw receive moe_topk experts and weights. */
static int route_token(sn_model *m, int layer, const float *x_in, int *top, float *tw)
{
    const sn_geom *g = &m->g;
    const layer_t *l = &m->L[layer];
    int H = g->hidden, E = g->moe_experts, K = g->moe_topk;
    float *s = m->gsc;
    gate_ctx gc = { l->gate_w, x_in, s, H };
    nemo_pool_for(&m->pool, E, gate_job, &gc);
    unsigned char *taken = (unsigned char *)calloc((size_t)E, 1);
    if (!taken) return -1;
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int best = -1; float bv = 0;
        for (int e = 0; e < E; e++) {
            if (taken[e]) continue;
            float cv = s[e] + l->gate_bias[e];
            if (best < 0 || cv > bv) { best = e; bv = cv; }
        }
        taken[best] = 1; top[k] = best; tw[k] = s[best]; sum += s[best];
    }
    free(taken);
    if (g->moe_norm_topk) { float den = sum + 1e-20f; for (int k = 0; k < K; k++) tw[k] /= den; }
    for (int k = 0; k < K; k++) tw[k] *= g->moe_scale;
    return 0;
}

static float *load_f32(sn_model *m, const char *fmt, int a, uint64_t want, const sn_tensor **tout)
{
    char name[300]; snprintf(name, sizeof(name), fmt, a);
    const sn_tensor *t = sn_pack_find(&m->pack, name);
    if (!t) { fprintf(stderr, "sn_model: missing tensor %s\n", name); return NULL; }
    if (t->count != want) { fprintf(stderr, "sn_model: %s has %llu weights, expected %llu\n", name, (unsigned long long)t->count, (unsigned long long)want); return NULL; }
    float *o = (float *)xalloc(want * 4);
    if (!o || sn_expand(&m->pack, &m->main_rd, t, 0, want, o) != 0) { free(o); fprintf(stderr, "sn_model: cannot read %s\n", name); return NULL; }
    if (tout) *tout = t;
    return o;
}

static const sn_tensor *need(sn_model *m, uint64_t want, int raw, const char *fmt, int a, int b)
{
    char name[300]; snprintf(name, sizeof(name), fmt, a, b);
    const sn_tensor *t = sn_pack_find(&m->pack, name);
    if (!t) { fprintf(stderr, "sn_model: missing tensor %s\n", name); return NULL; }
    if (t->count != want) { fprintf(stderr, "sn_model: %s has %llu weights, expected %llu\n", name, (unsigned long long)t->count, (unsigned long long)want); return NULL; }
    if (raw >= 0 && t->raw != raw) { fprintf(stderr, "sn_model: %s is %s, expected %s\n", name, t->raw ? "raw" : "packed", raw ? "raw" : "packed"); return NULL; }
    if (!t->raw && !(m->pack.pq_d == 4 && m->pack.block == 256)) { fprintf(stderr, "sn_model: pack is not block 256 PQ d4\n"); return NULL; }
    return t;
}

void sn_model_close(sn_model *m)
{
    if (!m) return;
    prof_free(m->prof); m->prof = NULL;
    nemo_pool_shutdown(&m->pool);
    if (m->L) for (int i = 0; i < m->n_all; i++) {
        layer_t *l = &m->L[i];
        free(l->norm_w); free(l->conv_w); free(l->conv_b); free(l->a_log); free(l->dvec);
        free(l->dt_bias); free(l->gnorm_w); free(l->gate_w); free(l->gate_bias);
        free(l->eup); free(l->edn); free(l->enorm_w); free(l->hnorm_w); free(l->fin_w);
    }
    if (m->mam) for (int i = 0; i < m->n_all; i++) if (m->mam[i].ssm_state) nemo_mamba_state_free(&m->mam[i]);
    if (m->att) for (int i = 0; i < m->n_all; i++) if (m->att[i].k) nemo_attn_cache_free(&m->att[i]);
    for (int t = 0; t < m->snap_n; t++) {
        for (int i = 0; i < m->g.n_layers; i++) { if (m->snap_ssm[t]) free(m->snap_ssm[t][i]); if (m->snap_conv[t]) free(m->snap_conv[t][i]); }
        free(m->snap_ssm[t]); free(m->snap_conv[t]);
    }
    sn_cuda_close(m->cuda);
    sn_gemm_close(m->gm);
    if (m->lanes) for (int i = 0; i < m->nthreads; i++) { sn_rd_close(&m->lanes[i].rd); free(m->lanes[i].book); free(m->lanes[i].mid); free(m->lanes[i].pmid); free(m->lanes[i].ew); free(m->lanes[i].G); free(m->lanes[i].U); free(m->lanes[i].T); }
    sn_rd_close(&m->main_rd);
    free(m->lanes); free(m->L); free(m->mam); free(m->att); free(m->normf_w);
    free(m->gsc); free(m->P); free(m->book_shared); free(m->Tbuf);
    free(m->hist_hn); free(m->hist_tok); free(m->spec_logits);
    if (m->ec) { for (size_t k = 0; k < (size_t)m->n_all * (size_t)m->g.moe_experts; k++) if (!m->ec[k].pb) { sn_res_free(&m->ec[k].up); sn_res_free(&m->ec[k].dn); } }
    free(m->ec); free(m->rlist); free(m->gmid); free(m->gpmid);
    free(m->xs_oidx); free(m->xs_list); free(m->xs_slot); free(m->xs_done);
    free(m->xs_next); free(m->xs_ph); free(m->xs_ptw); free(m->xs_ptop); free(m->xs_pfx); free(m->xs_pfl_keys);
    sn_pack_close(&m->pack);
    free(m);
}

sn_model *sn_model_open(const char *prefix, const char *shapes, const char *config, int max_pos, int nthreads)
{
    sn_model *m = (sn_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
    if (sn_geom_load(&m->g, config) != 0) { free(m); return NULL; }
    const sn_geom *g = &m->g;
    if (sn_pack_open(&m->pack, prefix, shapes) != 0) { free(m); return NULL; }
    if (sn_rd_open(&m->pack, &m->main_rd) != 0) { fprintf(stderr, "sn_model: cannot open pack files\n"); sn_pack_close(&m->pack); free(m); return NULL; }
    if (nthreads <= 0) nthreads = nemo_cores();
    if (nemo_pool_init(&m->pool, nthreads) != 0) { sn_model_close(m); return NULL; }
    m->nthreads = nemo_pool_threads(&m->pool);
    m->max_pos = max_pos;
    int H = g->hidden, E = g->moe_experts;
    int want_mtp = g->mtp && sn_pack_find(&m->pack, "language_model.mtp.layers.0.eh_proj.weight") != NULL;
    m->n_all = g->n_layers + (want_mtp ? 2 : 0);
    m->dtc = 1;
    m->lanes = (lane_t *)calloc((size_t)m->nthreads, sizeof(lane_t));
    m->L = (layer_t *)calloc((size_t)m->n_all, sizeof(layer_t));
    m->mam = (nemo_mamba_state *)calloc((size_t)m->n_all, sizeof(nemo_mamba_state));
    m->att = (nemo_attn_cache *)calloc((size_t)m->n_all, sizeof(nemo_attn_cache));
    if (!m->lanes || !m->L || !m->mam || !m->att) { sn_model_close(m); return NULL; }
    for (int i = 0; i < m->nthreads; i++) {
        lane_t *ln = &m->lanes[i];
        if (sn_rd_open(&m->pack, &ln->rd) != 0) { fprintf(stderr, "sn_model: cannot open lane handles\n"); sn_model_close(m); return NULL; }
        ln->book = (float *)xalloc(m->pack.book_floats * 4);
        ln->mid = (float *)xalloc((size_t)SN_NV * g->moe_inter * 4 + 64);
        ln->pmid = (double *)xalloc((size_t)SN_NV * ((size_t)g->moe_inter + 1) * 8);
        if (!ln->book || !ln->mid || !ln->pmid) { sn_model_close(m); return NULL; }
    }
    int maxcols = H; if (g->m_inner > maxcols) maxcols = g->m_inner; if (g->moe_shared > maxcols) maxcols = g->moe_shared;
    if (g->moe_inter > maxcols) maxcols = g->moe_inter;
    if (g->a_heads * g->a_hd > maxcols) maxcols = g->a_heads * g->a_hd;
    if (want_mtp && 2 * H > maxcols) maxcols = 2 * H;
    m->maxcols = maxcols;
    m->gm = sn_gemm_open(&m->pack, &m->pool, m->nthreads, (size_t)maxcols);
    if (!m->gm) { sn_model_close(m); return NULL; }
    m->gsc = (float *)xalloc((size_t)E * 4);
    m->P = (double *)xalloc((size_t)SN_NV * ((size_t)maxcols + 1) * 8); m->book_shared = (float *)xalloc(m->pack.book_floats * 4);
    if (!m->gsc || !m->P || !m->book_shared) { sn_model_close(m); return NULL; }

    int bad = 0;
    m->emb = need(m, (uint64_t)g->vocab * H, -1, "language_model.backbone.embeddings.weight%.0d%.0d", 0, 0);
    m->head = need(m, (uint64_t)g->vocab * H, -1, "language_model.lm_head.weight%.0d%.0d", 0, 0);
    m->normf_w = load_f32(m, "language_model.backbone.norm_f.weight%.0d", 0, (uint64_t)H, &m->normf_t);
    if (!m->emb || !m->head || !m->normf_w) bad = 1;
    for (int i = 0; i < g->n_layers && !bad; i++) {
        layer_t *l = &m->L[i];
        l->norm_w = load_f32(m, "language_model.backbone.layers.%d.norm.weight", i, (uint64_t)H, &l->norm);
        if (!l->norm_w) { bad = 1; break; }
        char type = g->types[i];
        if (type == 'M') {
            l->in_proj = need(m, (uint64_t)g->m_in_proj * H, 0, "language_model.backbone.layers.%d.mixer.in_proj.weight%.0d", i, 0);
            l->out_proj = need(m, (uint64_t)H * g->m_inner, 0, "language_model.backbone.layers.%d.mixer.out_proj.weight%.0d", i, 0);
            l->conv_w = load_f32(m, "language_model.backbone.layers.%d.mixer.conv1d.weight", i, (uint64_t)g->m_conv_dim * 4, NULL);
            l->conv_b = load_f32(m, "language_model.backbone.layers.%d.mixer.conv1d.bias", i, (uint64_t)g->m_conv_dim, NULL);
            l->a_log = load_f32(m, "language_model.backbone.layers.%d.mixer.A_log", i, (uint64_t)g->m_heads, NULL);
            l->dvec = load_f32(m, "language_model.backbone.layers.%d.mixer.D", i, (uint64_t)g->m_heads, NULL);
            l->dt_bias = load_f32(m, "language_model.backbone.layers.%d.mixer.dt_bias", i, (uint64_t)g->m_heads, NULL);
            l->gnorm_w = load_f32(m, "language_model.backbone.layers.%d.mixer.norm.weight", i, (uint64_t)g->m_inner, NULL);
            if (!l->in_proj || !l->out_proj || !l->conv_w || !l->conv_b || !l->a_log || !l->dvec || !l->dt_bias || !l->gnorm_w) bad = 1;
            if (!bad && nemo_mamba_state_init(&m->mam[i], g->m_heads, g->m_head_dim, g->m_state, g->m_groups) != 0) bad = 1;
        } else if (type == 'A') {
            uint64_t qd = (uint64_t)g->a_heads * g->a_hd, kd = (uint64_t)g->a_kv * g->a_hd;
            l->q = need(m, qd * H, 0, "language_model.backbone.layers.%d.mixer.q_proj.weight%.0d", i, 0);
            l->k = need(m, kd * H, -1, "language_model.backbone.layers.%d.mixer.k_proj.weight%.0d", i, 0);
            l->v = need(m, kd * H, -1, "language_model.backbone.layers.%d.mixer.v_proj.weight%.0d", i, 0);
            l->o = need(m, (uint64_t)H * qd, 0, "language_model.backbone.layers.%d.mixer.o_proj.weight%.0d", i, 0);
            if (!l->q || !l->k || !l->v || !l->o) bad = 1;
            if (!bad && nemo_attn_cache_init(&m->att[i], g->a_heads, g->a_kv, g->a_hd, max_pos) != 0) bad = 1;
        } else {
            l->fc1 = need(m, (uint64_t)g->moe_latent * H, 0, "language_model.backbone.layers.%d.mixer.fc1_latent_proj.weight%.0d", i, 0);
            l->fc2 = need(m, (uint64_t)H * g->moe_latent, 0, "language_model.backbone.layers.%d.mixer.fc2_latent_proj.weight%.0d", i, 0);
            l->sh_up = need(m, (uint64_t)g->moe_shared * H, 0, "language_model.backbone.layers.%d.mixer.shared_experts.up_proj.weight%.0d", i, 0);
            l->sh_dn = need(m, (uint64_t)H * g->moe_shared, 0, "language_model.backbone.layers.%d.mixer.shared_experts.down_proj.weight%.0d", i, 0);
            l->gate_bias = load_f32(m, "language_model.backbone.layers.%d.mixer.gate.e_score_correction_bias", i, (uint64_t)E, NULL);
            l->gate_t = need(m, (uint64_t)E * H, 1, "language_model.backbone.layers.%d.mixer.gate.weight%.0d", i, 0);
            if (!l->fc1 || !l->fc2 || !l->sh_up || !l->sh_dn || !l->gate_bias || !l->gate_t) { bad = 1; break; }
            l->gate_w = (uint16_t *)xalloc((size_t)E * H * 2);
            if (!l->gate_w || sn_raw_read16(&m->pack, &m->main_rd, l->gate_t, 0, (uint64_t)E * H, l->gate_w) != 0) { bad = 1; break; }
            l->eup = (const sn_tensor **)xalloc((size_t)E * sizeof(void *));
            l->edn = (const sn_tensor **)xalloc((size_t)E * sizeof(void *));
            if (!l->eup || !l->edn) { bad = 1; break; }
            for (int e = 0; e < E && !bad; e++) {
                l->eup[e] = need(m, (uint64_t)g->moe_inter * g->moe_latent, 0, "language_model.backbone.layers.%d.mixer.experts.%d.up_proj.weight", i, e);
                l->edn[e] = need(m, (uint64_t)g->moe_latent * g->moe_inter, 0, "language_model.backbone.layers.%d.mixer.experts.%d.down_proj.weight", i, e);
                if (!l->eup[e] || !l->edn[e]) bad = 1;
            }
        }
    }
    if (!bad && want_mtp) {
        layer_t *a = &m->L[g->n_layers], *e2 = &m->L[g->n_layers + 1];
        uint64_t qd = (uint64_t)g->a_heads * g->a_hd, kd = (uint64_t)g->a_kv * g->a_hd;
        int mb = 0;
        a->norm_w = load_f32(m, "language_model.mtp.layers.0.norm.weight%.0d", 0, (uint64_t)H, &a->norm);
        a->enorm_w = load_f32(m, "language_model.mtp.layers.0.enorm.weight%.0d", 0, (uint64_t)H, NULL);
        a->hnorm_w = load_f32(m, "language_model.mtp.layers.0.hnorm.weight%.0d", 0, (uint64_t)H, NULL);
        a->eh = need(m, (uint64_t)H * 2 * H, 0, "language_model.mtp.layers.0.eh_proj.weight%.0d%.0d", 0, 0);
        a->q = need(m, qd * H, 0, "language_model.mtp.layers.0.mixer.q_proj.weight%.0d%.0d", 0, 0);
        a->k = need(m, kd * H, -1, "language_model.mtp.layers.0.mixer.k_proj.weight%.0d%.0d", 0, 0);
        a->v = need(m, kd * H, -1, "language_model.mtp.layers.0.mixer.v_proj.weight%.0d%.0d", 0, 0);
        a->o = need(m, (uint64_t)H * qd, 0, "language_model.mtp.layers.0.mixer.o_proj.weight%.0d%.0d", 0, 0);
        if (!a->norm_w || !a->enorm_w || !a->hnorm_w || !a->eh || !a->q || !a->k || !a->v || !a->o) mb = 1;
        if (!mb && nemo_attn_cache_init(&m->att[g->n_layers], g->a_heads, g->a_kv, g->a_hd, max_pos + SN_NV) != 0) mb = 1;
        e2->norm_w = load_f32(m, "language_model.mtp.layers.1.norm.weight%.0d", 0, (uint64_t)H, &e2->norm);
        e2->fin_w = load_f32(m, "language_model.mtp.layers.1.final_layernorm.weight%.0d", 0, (uint64_t)H, NULL);
        e2->fc1 = need(m, (uint64_t)g->moe_latent * H, 0, "language_model.mtp.layers.1.mixer.fc1_latent_proj.weight%.0d%.0d", 0, 0);
        e2->fc2 = need(m, (uint64_t)H * g->moe_latent, 0, "language_model.mtp.layers.1.mixer.fc2_latent_proj.weight%.0d%.0d", 0, 0);
        e2->sh_up = need(m, (uint64_t)g->moe_shared * H, 0, "language_model.mtp.layers.1.mixer.shared_experts.up_proj.weight%.0d%.0d", 0, 0);
        e2->sh_dn = need(m, (uint64_t)H * g->moe_shared, 0, "language_model.mtp.layers.1.mixer.shared_experts.down_proj.weight%.0d%.0d", 0, 0);
        e2->gate_bias = load_f32(m, "language_model.mtp.layers.1.mixer.gate.e_score_correction_bias%.0d", 0, (uint64_t)E, NULL);
        e2->gate_t = need(m, (uint64_t)E * H, 1, "language_model.mtp.layers.1.mixer.gate.weight%.0d%.0d", 0, 0);
        if (!e2->norm_w || !e2->fin_w || !e2->fc1 || !e2->fc2 || !e2->sh_up || !e2->sh_dn || !e2->gate_bias || !e2->gate_t) mb = 1;
        if (!mb) {
            e2->gate_w = (uint16_t *)xalloc((size_t)E * H * 2);
            e2->eup = (const sn_tensor **)xalloc((size_t)E * sizeof(void *));
            e2->edn = (const sn_tensor **)xalloc((size_t)E * sizeof(void *));
            if (!e2->gate_w || !e2->eup || !e2->edn || sn_raw_read16(&m->pack, &m->main_rd, e2->gate_t, 0, (uint64_t)E * H, e2->gate_w) != 0) mb = 1;
        }
        for (int e = 0; e < E && !mb; e++) {
            e2->eup[e] = need(m, (uint64_t)g->moe_inter * g->moe_latent, 0, "language_model.mtp.layers.1.mixer.experts.%d.up_proj.weight%.0d", e, 0);
            e2->edn[e] = need(m, (uint64_t)g->moe_latent * g->moe_inter, 0, "language_model.mtp.layers.1.mixer.experts.%d.down_proj.weight%.0d", e, 0);
            if (!e2->eup[e] || !e2->edn[e]) mb = 1;
        }
        if (mb) fprintf(stderr, "sn_model: the pack lacks a complete MTP head; speculative decoding is unavailable\n");
        else m->has_mtp = 1;
    }
    if (bad) { sn_model_close(m); return NULL; }
    m->state_valid = 1;
    return m;
}

const sn_geom *sn_model_geom(const sn_model *m) { return &m->g; }

/* ------------------------------------------------------------- batched prefill */

typedef struct {
    sn_model *m; const layer_t *l; const float *lat; float *eo;
    const int *used; int n_used; const int *start, *tok, *slot; int fail;
} pfx_ctx;

static void pfx_job(void *vc, int b, int e)
{
    pfx_ctx *c = (pfx_ctx *)vc;
    sn_model *m = c->m;
    const sn_geom *g = &m->g;
    int Lt = g->moe_latent, I = g->moe_inter, K = g->moe_topk;
    lane_t *ln = &m->lanes[lane_of(c->n_used, m->nthreads, b)];
    for (int i = b; i < e; i++) {
        int ex = c->used[i], cnt = c->start[i + 1] - c->start[i];
        const sn_tensor *up = c->l->eup[ex], *dn = c->l->edn[ex];
        size_t need_ew = (size_t)I * Lt, need_g = (size_t)cnt * (Lt > I ? Lt : I), need_u = (size_t)cnt * I;
        if (ln->ew_cap < need_ew) { float *p = (float *)realloc(ln->ew, need_ew * 4); if (!p) { c->fail = 1; return; } ln->ew = p; ln->ew_cap = need_ew; }
        if (ln->g_cap < need_g) { float *p = (float *)realloc(ln->G, need_g * 4); if (!p) { c->fail = 1; return; } ln->G = p; ln->g_cap = need_g; }
        if (ln->u_cap < need_u) { float *p = (float *)realloc(ln->U, need_u * 4); if (!p) { c->fail = 1; return; } ln->U = p; ln->u_cap = need_u; }
        for (int j = 0; j < cnt; j++) memcpy(ln->G + (size_t)j * Lt, c->lat + (size_t)c->tok[c->start[i] + j] * Lt, (size_t)Lt * 4);
        if ((!up->raw && sn_load_book(&m->pack, &ln->rd, up, ln->book) != 0)
            || sn_expand_book(&m->pack, &ln->rd, up, ln->book, 0, up->count, ln->ew) != 0) { c->fail = 1; return; }
        sn_gemm_rows(ln->ew, I, Lt, ln->G, cnt, NULL, 0, I, ln->U, 2);
        if ((!dn->raw && sn_load_book(&m->pack, &ln->rd, dn, ln->book) != 0)
            || sn_expand_book(&m->pack, &ln->rd, dn, ln->book, 0, dn->count, ln->ew) != 0) { c->fail = 1; return; }
        sn_gemm_rows(ln->ew, Lt, I, ln->U, cnt, NULL, 0, Lt, ln->G, 0);
        for (int j = 0; j < cnt; j++)
            memcpy(c->eo + ((size_t)c->tok[c->start[i] + j] * K + (size_t)c->slot[c->start[i] + j]) * Lt, ln->G + (size_t)j * Lt, (size_t)Lt * 4);
    }
}

#define PF_CHUNK 128

/* Workspace for n tokens moving through the layers together. */
typedef struct {
    float *X, *Hn, *Y, *Sh, *Pr, *In, *Q, *Kb, *Vb, *Lat, *Acc, *EO, *SM, *Tw;
    double *PL;
    int *Top, *cnt, *used, *start, *etok, *eslot, *fill;
    int *emode; ecent *epend;
} fws;

static void fws_close(fws *w)
{
    free(w->X); free(w->Hn); free(w->Y); free(w->Sh); free(w->Pr); free(w->In); free(w->Q); free(w->Kb); free(w->Vb); free(w->Lat);
    free(w->Acc); free(w->EO); free(w->SM); free(w->Tw); free(w->PL);
    free(w->Top); free(w->cnt); free(w->used); free(w->start); free(w->etok); free(w->eslot); free(w->fill);
    free(w->emode); free(w->epend);
    memset(w, 0, sizeof(*w));
}

static int fws_open(sn_model *m, fws *w, int n)
{
    const sn_geom *g = &m->g;
    int H = g->hidden, E = g->moe_experts, K = g->moe_topk, Lt = g->moe_latent;
    int qd = g->a_heads * g->a_hd, kd = g->a_kv * g->a_hd;
    int inner_w = g->m_inner > qd ? g->m_inner : qd;
    memset(w, 0, sizeof(*w));
    w->X = (float *)calloc((size_t)n * H, 4); w->Hn = (float *)calloc((size_t)n * H, 4); w->Y = (float *)calloc((size_t)n * H, 4); w->Sh = (float *)calloc((size_t)n * H, 4);
    w->Pr = (float *)calloc((size_t)n * g->m_in_proj, 4); w->In = (float *)calloc((size_t)n * inner_w, 4);
    w->Q = (float *)calloc((size_t)n * qd, 4); w->Kb = (float *)calloc((size_t)n * kd, 4); w->Vb = (float *)calloc((size_t)n * kd, 4);
    w->Lat = (float *)calloc((size_t)n * Lt, 4); w->Acc = (float *)calloc((size_t)n * Lt, 4); w->EO = (float *)calloc((size_t)n * K * Lt, 4);
    w->SM = (float *)calloc((size_t)n * g->moe_shared, 4); w->Tw = (float *)calloc((size_t)n * K, 4);
    w->PL = (double *)calloc((size_t)n * (Lt + 1), 8);
    w->Top = (int *)calloc((size_t)n * K, sizeof(int)); w->cnt = (int *)calloc((size_t)E, sizeof(int));
    w->used = (int *)calloc((size_t)E, sizeof(int)); w->start = (int *)calloc((size_t)E + 1, sizeof(int));
    w->etok = (int *)calloc((size_t)n * K, sizeof(int)); w->eslot = (int *)calloc((size_t)n * K, sizeof(int)); w->fill = (int *)calloc((size_t)E, sizeof(int));
    w->emode = (int *)calloc((size_t)E, sizeof(int)); w->epend = (ecent *)calloc((size_t)E, sizeof(ecent));
    if (!w->emode || !w->epend) { fws_close(w); return -1; }
    if (!w->X || !w->Hn || !w->Y || !w->Sh || !w->Pr || !w->In || !w->Q || !w->Kb || !w->Vb || !w->Lat || !w->Acc || !w->EO || !w->SM || !w->Tw
        || !w->PL || !w->Top || !w->cnt || !w->used || !w->start || !w->etok || !w->eslot || !w->fill) { fws_close(w); return -1; }
    return 0;
}

static int snap_ensure(sn_model *m, int cnt)
{
    const sn_geom *g = &m->g;
    for (int t = m->snap_n; t < cnt; t++) {
        m->snap_ssm[t] = (float **)calloc((size_t)g->n_layers, sizeof(float *));
        m->snap_conv[t] = (float **)calloc((size_t)g->n_layers, sizeof(float *));
        if (!m->snap_ssm[t] || !m->snap_conv[t]) return -1;
        for (int i = 0; i < g->n_layers; i++) if (g->types[i] == 'M') {
            m->snap_ssm[t][i] = (float *)xalloc((size_t)g->m_heads * g->m_head_dim * g->m_state * 4);
            m->snap_conv[t][i] = (float *)xalloc((size_t)g->m_conv_dim * 4 * 4);
            if (!m->snap_ssm[t][i] || !m->snap_conv[t][i]) return -1;
        }
        m->snap_n = t + 1;
    }
    return 0;
}

/* y = W x for n rows; fused = the multi-vector kernels (n <= SN_NV), else expand + GEMM (act 2 = relu squared). */
static int lin(sn_model *m, int fused, const sn_tensor *t, const float *X, int n, int cols, int rows, float *Y, int act)
{
    if (!fused) {
        /* prefill: a dense tensor resident on the card is expanded there and multiplied by the CPU-mirroring GEMM */
        if (m->xs_on && sn_cuda_resident(m->cuda, &m->pack, t) && !(cols & 7) && (act == 0 || act == 2)
            && sn_cuda_xs_dense_gemm(m->cuda, &m->pack, t, X, n, cols, rows, Y, act) == 0) return 0;
        return sn_gemm_linear(m->gm, t, NULL, X, n, cols, rows, Y, act);
    }
    int rc = lin_n(m, t, X, n, Y, rows, cols);
    if (!rc && act == 2) for (size_t j = 0; j < (size_t)n * (size_t)rows; j++) Y[j] = sn_relu2(Y[j]);
    return rc;
}


/* ------------------------------------------------------------- hot-expert cache */

static size_t ekey(const sn_model *m, int li, int ex) { return (size_t)li * (size_t)m->g.moe_experts + (size_t)ex; }

static void rlist_add(sn_model *m, size_t key) { m->ec[key].rpos = m->nrlist; m->rlist[m->nrlist++] = (int)key; }
static void rlist_del(sn_model *m, size_t key)
{
    int pos = m->ec[key].rpos, last = m->rlist[--m->nrlist];
    m->rlist[pos] = last; m->ec[last].rpos = pos;
}

/* Drop the packed copies of a cache entry (or of a pending one): a pinned block goes back to the pool, a malloc'd copy is freed. */
static void ec_release(sn_model *m, ecent *e)
{
    if (e->pb) { if (m->cuda) sn_cuda_xs_pool_put(m->cuda, e->pb - 1); e->pb = 0; memset(&e->up, 0, sizeof e->up); memset(&e->dn, 0, sizeof e->dn); }
    else { sn_res_free(&e->up); sn_res_free(&e->dn); }
}

static void cache_evict(sn_model *m, size_t key)
{
    ecent *e = &m->ec[key];
    int li = (int)(key / (size_t)m->g.moe_experts), ex = (int)(key % (size_t)m->g.moe_experts);
    const layer_t *l = &m->L[li];
    if (e->gpu && m->cuda) { sn_cuda_cache_drop(m->cuda, &m->pack, l->eup[ex]); sn_cuda_cache_drop(m->cuda, &m->pack, l->edn[ex]); e->gpu = 0; m->gpu_drops++; }
    m->ec_used -= e->up.bytes + e->dn.bytes;
    ec_release(m, e);
    e->ram = 0;
    rlist_del(m, key);
    m->evicts++;
}

/* Count the tokens that chose each expert (also during prefill, so popularity reflects the whole conversation). */
static void cache_count(sn_model *m, int li, fws *w, int nu)
{
    for (int i = 0; i < nu; i++) m->ec[ekey(m, li, w->used[i])].uses += (uint32_t)(w->start[i + 1] - w->start[i]);
}

/* Decide, before the parallel job, how each used expert is served; admission and eviction happen here, on the main thread,
 * from counts alone, so the cache state never depends on thread timing. */
static void cache_plan(sn_model *m, int li, fws *w, int nu)
{
    const sn_geom *g = &m->g;
    int gpu_ok = m->gpu_tier && dtc_wanted(m, g->moe_inter, g->moe_latent, 1, 1) && dtc_wanted(m, g->moe_latent, g->moe_inter, 1, 1);
    m->epoch++;
    for (int i = 0; i < nu; i++) m->ec[ekey(m, li, w->used[i])].epoch = m->epoch;     /* protected from eviction this step */
    for (int i = 0; i < nu; i++) {
        size_t key = ekey(m, li, w->used[i]);
        ecent *e = &m->ec[key];
        const layer_t *l = &m->L[li];
        w->emode[i] = 0;
        if (e->gpu && gpu_ok) { w->emode[i] = 2; m->hits_gpu++; continue; }
        if (e->ram) { w->emode[i] = 1; m->hits_ram++; continue; }
        m->misses++;
        uint64_t sz = sn_res_bytes(&m->pack, l->eup[w->used[i]]) + sn_res_bytes(&m->pack, l->edn[w->used[i]]);
        int fits = m->ec_used + m->ec_reserved + sz <= m->ec_budget;
        while (!fits && e->uses >= 3) {      /* full: displace the coldest resident expert that is colder than this one */
            int vi = -1; uint32_t vu = e->uses;
            for (int r = 0; r < m->nrlist; r++) {
                const ecent *c = &m->ec[m->rlist[r]];
                if (c->epoch != m->epoch && c->uses < vu) { vu = c->uses; vi = r; }
            }
            if (vi < 0) break;
            cache_evict(m, (size_t)m->rlist[vi]);
            fits = m->ec_used + m->ec_reserved + sz <= m->ec_budget;
        }
        if (fits) { m->ec_reserved += sz; w->emode[i] = 3; }
    }
}

/* After the job: adopt the copies the lanes loaded (in expert order), then promote hot RAM entries to the GPU. */
static void cache_adopt(sn_model *m, int li, fws *w, int nu, int failed)
{
    for (int i = 0; i < nu; i++) {
        if (w->emode[i] != 3) continue;
        size_t key = ekey(m, li, w->used[i]);
        ecent *pe = &w->epend[i];
        const layer_t *l = &m->L[li];
        m->ec_reserved -= sn_res_bytes(&m->pack, l->eup[w->used[i]]) + sn_res_bytes(&m->pack, l->edn[w->used[i]]);
        if (pe->pend && !failed) {
            ecent *e = &m->ec[key];
            e->up = pe->up; e->dn = pe->dn; e->ram = 1; e->pb = pe->pb; pe->pb = 0;
            memset(&pe->up, 0, sizeof pe->up); memset(&pe->dn, 0, sizeof pe->dn);
            m->ec_used += e->up.bytes + e->dn.bytes; m->admits++;
            rlist_add(m, key);
        } else ec_release(m, pe);
        pe->pend = 0;
    }
    if (!m->gpu_tier || failed) return;
    int promoted = 0;
    for (int i = 0; i < nu && promoted < 4; i++) {
        size_t key = ekey(m, li, w->used[i]);
        ecent *e = &m->ec[key];
        const layer_t *l = &m->L[li];
        if (!e->ram || e->gpu || e->uses < 2) continue;
        for (int attempt = 0; attempt < 2 && !e->gpu; attempt++) {
            if (sn_cuda_cache_put(m->cuda, &m->pack, l->eup[w->used[i]], &e->up) == 0) {
                if (sn_cuda_cache_put(m->cuda, &m->pack, l->edn[w->used[i]], &e->dn) == 0) { e->gpu = 1; m->gpu_puts++; promoted++; }
                else sn_cuda_cache_drop(m->cuda, &m->pack, l->eup[w->used[i]]);
            }
            if (e->gpu) break;
            int vi = -1; uint32_t vu = e->uses;      /* no room: displace the coldest GPU entry that is colder */
            for (int r = 0; r < m->nrlist; r++) {
                const ecent *c = &m->ec[m->rlist[r]];
                if (c->gpu && c->epoch != m->epoch && c->uses < vu) { vu = c->uses; vi = r; }
            }
            if (vi < 0) break;
            {
                size_t vk = (size_t)m->rlist[vi]; ecent *v = &m->ec[vk];
                int vli = (int)(vk / (size_t)m->g.moe_experts), vex = (int)(vk % (size_t)m->g.moe_experts);
                sn_cuda_cache_drop(m->cuda, &m->pack, m->L[vli].eup[vex]); sn_cuda_cache_drop(m->cuda, &m->pack, m->L[vli].edn[vex]);
                v->gpu = 0; m->gpu_drops++;
            }
        }
    }
}

/* Experts mirrored in VRAM are computed on the device (main thread, tokens one by one) before the lanes start; DTC on the
 * device is bit-identical to the CPU DTC kernel, so the output cannot tell. Any device failure leaves the expert to the lanes. */
static void cache_gpu_serve(sn_model *m, int li, fws *w, int nu, fx_ctx *fc)
{
    const sn_geom *g = &m->g;
    int Lt = g->moe_latent, I = g->moe_inter, K = g->moe_topk;
    if (!dtc_wanted(m, I, Lt, 1, 1) || !dtc_wanted(m, Lt, I, 1, 1)) return;
    const layer_t *l = &m->L[li];
    for (int i = 0; i < nu; i++) {
        if (w->emode[i] != 2) continue;
        int ex = w->used[i], ok = 1;
        for (int v = 0; v < w->start[i + 1] - w->start[i] && ok; v++) {
            int tk = fc->tok[w->start[i] + v], sl = fc->slot[w->start[i] + v];
            ok = sn_cuda_dtc_rows(m->cuda, &m->pack, l->eup[ex], Lt, I, w->Lat + (size_t)tk * Lt, w->PL + (size_t)tk * (Lt + 1), m->gmid) == 0;
            if (!ok) break;
            m->gpmid[0] = 0.0;
            for (int j = 0; j < I; j++) { m->gmid[j] = sn_relu2(m->gmid[j]); m->gpmid[j + 1] = m->gpmid[j] + (double)m->gmid[j]; }
            ok = sn_cuda_dtc_rows(m->cuda, &m->pack, l->edn[ex], I, Lt, m->gmid, m->gpmid, w->EO + ((size_t)tk * K + (size_t)sl) * Lt) == 0;
        }
        if (!ok) { w->emode[i] = m->ec[ekey(m, li, ex)].ram ? 1 : 0; }
    }
}

/* ------------------------------------------------------------- GPU expert stream
 * JD's standing CUDA design (2026-11-27): the routed experts of a layer travel to the card COMPRESSED (from the RAM cache's
 * packed copies, or straight from the pack), are expanded there with CUDA, multiplied there, and their VRAM slots are reused
 * for the next token. Experts still in a slot from an earlier token are not sent again. The lanes fill one half of the
 * pinned staging while the card copies and computes the other half. Decode uses the DTC kernels (bit-identical to the
 * CPU's), prefill expands into a float scratch and runs the GEMM that mirrors sn_gemm_rows. Any device failure leaves the
 * layer to the CPU lanes and turns the stream off with a note on stderr. */
typedef struct { sn_model *m; const layer_t *l; int li; fws *w; int b0, j0, cnt; int fail; } xprep_ctx;

static void xs_copy_res(sn_cuda *c, const sn_pack *p, int j, int which, const sn_pq_res *r)
{
    uint8_t *idx; uint16_t *mn, *sc; float *book;
    sn_cuda_xs_stage_ptrs(c, j, which, &idx, &mn, &sc, &book);
    memcpy(idx, r->idx, (size_t)(r->nblocks * p->nsub)); memcpy(mn, r->mn, (size_t)r->nblocks * 2);
    memcpy(sc, r->sc, (size_t)r->nblocks * 2); memcpy(book, r->book, (size_t)p->book_floats * 4);
}

static int xs_read(sn_cuda *c, const sn_pack *p, sn_rd *rd, int j, int which, const sn_tensor *t)
{
    uint8_t *idx; uint16_t *mn, *sc; float *book;
    sn_cuda_xs_stage_ptrs(c, j, which, &idx, &mn, &sc, &book);
    return sn_res_read_into(p, rd, t, idx, mn, sc, book);
}

static void xprep_job(void *vc, int b, int e)
{
    xprep_ctx *c = (xprep_ctx *)vc;
    sn_model *m = c->m;
    fws *w = c->w;
    lane_t *ln = &m->lanes[lane_of(c->cnt, m->nthreads, b)];
    for (int k = b; k < e; k++) {
        int i = m->xs_list[c->b0 + k], ex = w->used[i], j = c->j0 + k;
        const sn_tensor *up = c->l->eup[ex], *dn = c->l->edn[ex];
        int mode = w->emode[i];
        if (mode == 1) {
            const ecent *ce = &m->ec[(size_t)c->li * m->g.moe_experts + (size_t)ex];
            xs_copy_res(m->cuda, &m->pack, j, 0, &ce->up); xs_copy_res(m->cuda, &m->pack, j, 1, &ce->dn);
            continue;
        }
        if (mode == 3 && w->epend[i].pb) {   /* admitted into a pinned block: read the pack straight into it, DMA from it */
            ecent *pe = &w->epend[i];
            sn_pq_res *rr[2] = { &pe->up, &pe->dn }; const sn_tensor *tt[2] = { up, dn }; int ok = 1;
            for (int q = 0; q < 2 && ok; q++) {
                sn_cuda_xs_pool_ptrs(m->cuda, pe->pb - 1, q, &rr[q]->idx, &rr[q]->mn, &rr[q]->sc, &rr[q]->book);
                rr[q]->nblocks = tt[q]->nblocks; rr[q]->bytes = sn_res_bytes(&m->pack, tt[q]);
                ok = sn_res_read_into(&m->pack, &ln->rd, tt[q], rr[q]->idx, rr[q]->mn, rr[q]->sc, rr[q]->book) == 0;
            }
            if (ok) { pe->pend = 1; continue; }
            memset(&pe->up, 0, sizeof pe->up); memset(&pe->dn, 0, sizeof pe->dn); pe->pend = 0;   /* the main thread returns the block */
        } else if (mode == 3) {     /* the RAM cache admits this expert: load its packed copy once, stage from it */
            ecent *pe = &w->epend[i];
            if (sn_res_load(&m->pack, &ln->rd, up, &pe->up) == 0 && sn_res_load(&m->pack, &ln->rd, dn, &pe->dn) == 0) {
                pe->pend = 1;
                xs_copy_res(m->cuda, &m->pack, j, 0, &pe->up); xs_copy_res(m->cuda, &m->pack, j, 1, &pe->dn);
                continue;
            }
            sn_res_free(&pe->up); sn_res_free(&pe->dn); pe->pend = 0;
        }
        if (xs_read(m->cuda, &m->pack, &ln->rd, j, 0, up) != 0 || xs_read(m->cuda, &m->pack, &ln->rd, j, 1, dn) != 0) c->fail = 1;
    }
}

static int xs_compute(sn_model *m, fws *w, int i, int slot, int fused)
{
    int K = m->g.moe_topk;
    if (!fused) return sn_cuda_xs_gemm(m->cuda, slot, w->start[i], w->start[i + 1] - w->start[i]);
    for (int v = w->start[i]; v < w->start[i + 1]; v++)
        if (sn_cuda_xs_dtc(m->cuda, slot, w->etok[v], w->etok[v] * K + w->eslot[v]) != 0) return -1;
    return 0;
}

/* Early fetch (on by default): while the card works on this MoE layer, run the NEXT MoE layer's router on this layer's input
 * (right about two times in three on the real pack) and send the experts it picks ahead, so most of the next layer's experts
 * are already in their slots. It only decides what sits in which slot, never what is computed, so results are unchanged. */
typedef struct { sn_model *m; int nx, n, b0, j0; int fail; } xpf_ctx;

static void xpf_job(void *vc, int b, int e)
{
    xpf_ctx *c = (xpf_ctx *)vc;
    sn_model *m = c->m;
    const layer_t *l = &m->L[c->nx];
    lane_t *ln = &m->lanes[lane_of(c->n, m->nthreads, b)];
    for (int k = b; k < e; k++) {
        int ex = m->xs_pfx[c->b0 + k], j = c->j0 + k;
        const ecent *ce = m->ec ? &m->ec[(size_t)c->nx * m->g.moe_experts + (size_t)ex] : NULL;
        if (ce && ce->ram) { xs_copy_res(m->cuda, &m->pack, j, 0, &ce->up); xs_copy_res(m->cuda, &m->pack, j, 1, &ce->dn); continue; }
        if (xs_read(m->cuda, &m->pack, &ln->rd, j, 0, l->eup[ex]) != 0 || xs_read(m->cuda, &m->pack, &ln->rd, j, 1, l->edn[ex]) != 0) c->fail = 1;
    }
}

static int xs_prefetch(sn_model *m, int li, fws *w, int n, int *half)
{
    sn_cuda *c = m->cuda;
    const sn_geom *g = &m->g;
    int nx = m->xs_next[li], K = g->moe_topk, H = g->hidden, rc = 0;
    if (nx < 0) return 0;
    nemo_rmsnorm(w->X + (size_t)(n - 1) * H, m->L[nx].norm_w, m->xs_ph, H, g->eps);
    if (route_token(m, nx, m->xs_ph, m->xs_ptop, m->xs_ptw) != 0) return 0;
    m->xs_pfli = nx; m->xs_pfn = 0;
    int ns = 0;
    for (int k = 0; k < K && !rc; k++) {
        int ex = m->xs_ptop[k], key = (int)ekey(m, nx, ex);
        if (sn_cuda_xs_peek(c, key) >= 0) continue;            /* already on the card */
        const ecent *ce = m->ec ? &m->ec[(size_t)key] : NULL;
        if (ce && ce->ram && ce->pb) {                           /* pinned copy: one DMA, no CPU work */
            sn_cuda_xs_chunk_begin(c);
            int s = sn_cuda_xs_claim(c, key);
            if (s < 0 || sn_cuda_xs_upload_pool(c, ce->pb - 1, s, key) != 0) rc = -1;
            else { m->xs_pf_sent++; m->xs_direct++; m->xs_pfl_keys[m->xs_pfn++] = ex; }
        } else m->xs_pfx[ns++] = ex;
    }
    int C = sn_cuda_xs_chunk(c);
    for (int b = 0; b < ns && !rc; b += C, *half ^= 1) {
        int e = b + C < ns ? b + C : ns, cnt = e - b, j0 = *half * C;
        sn_cuda_xs_chunk_begin(c);
        if (sn_cuda_xs_stage_wait(c, j0, cnt) != 0) { rc = -1; break; }
        for (int k = b; k < e && !rc; k++) {
            int s = sn_cuda_xs_claim(c, (int)ekey(m, nx, m->xs_pfx[k]));
            if (s < 0) rc = -1; else m->xs_slot[k - b] = s;
        }
        if (rc) break;
        xpf_ctx pc = { m, nx, cnt, b, j0, 0 };
        nemo_pool_for(&m->pool, cnt, xpf_job, &pc);
        if (pc.fail) { rc = -1; break; }
        for (int k = 0; k < cnt && !rc; k++) {
            rc = sn_cuda_xs_upload(c, j0 + k, m->xs_slot[k], (int)ekey(m, nx, m->xs_pfx[b + k]));
            if (!rc) { m->xs_pf_sent++; m->xs_pfl_keys[m->xs_pfn++] = m->xs_pfx[b + k]; }
        }
    }
    return rc;
}

/* 0: every used expert of the layer computed on the card, outputs in w->EO and m->xs_done[i] = 1. -1: leave it to the CPU. */
static int xs_layer(sn_model *m, int li, fws *w, int n, int nu, int fused)
{
    sn_cuda *c = m->cuda;
    const sn_geom *g = &m->g;
    const layer_t *l = &m->L[li];
    int K = g->moe_topk, rc = 0, nmiss = 0;
    memset(m->xs_done, 0, (size_t)nu);
    if (n > PF_CHUNK) return -1;
    if (!fused) for (int i = 0; i < nu; i++) w->emode[i] = (m->ec && m->ec[ekey(m, li, w->used[i])].ram) ? 1 : 0;
    else if (!m->ec) for (int i = 0; i < nu; i++) w->emode[i] = 0;
    for (int i = 0; i < nu; i++) w->epend[i].pb = 0;
    if (m->xs_pfli == li) {           /* how many of the experts sent early did the router pick */
        for (int k = 0; k < m->xs_pfn; k++) if (w->cnt[m->xs_pfl_keys[k]]) m->xs_pf_right++;
        m->xs_pfli = -1; m->xs_pfn = 0;
    }
    for (int v = 0; v < n * K; v++) m->xs_oidx[v] = w->etok[v] * K + w->eslot[v];
    if (sn_cuda_xs_layer_begin(c, w->Lat, fused ? w->PL : NULL, n, w->etok, m->xs_oidx, n * K) != 0) rc = -1;
    sn_cuda_xs_chunk_begin(c);
    sn_prof *pf = m->prof; double q0 = pf ? pnow() : 0.0, q1;
    for (int i = 0; i < nu && !rc; i++) {
        int key = (int)ekey(m, li, w->used[i]);
        int s = sn_cuda_xs_find(c, key);
        if (s >= 0) rc = xs_compute(m, w, i, s, fused);
        else m->xs_list[nmiss++] = i;
    }
    if (pf) { q1 = pnow(); pf->xs_issue += q1 - q0; q0 = q1; pf->xs_layers++; pf->xs_hit += nu - nmiss;
              for (int k = 0; k < nmiss; k++) { int md = w->emode[m->xs_list[k]]; pf->xs_mode[md >= 0 && md < 4 ? md : 0]++; } }
    {   /* pinned RAM cache copies go up first: one DMA each, no CPU copy; the rest go through the staging halves */
        int keep = 0;
        for (int k = 0; k < nmiss && !rc; k++) {
            int i = m->xs_list[k], key = (int)ekey(m, li, w->used[i]);
            const ecent *ce = m->ec ? &m->ec[(size_t)key] : NULL;
            if (w->emode[i] == 1 && ce && ce->ram && ce->pb) {
                sn_cuda_xs_chunk_begin(c);
                int s = sn_cuda_xs_claim(c, key);
                if (s < 0) rc = -1;
                else rc = sn_cuda_xs_upload_pool(c, ce->pb - 1, s, key);
                if (!rc) rc = xs_compute(m, w, i, s, fused);
                if (!rc) { m->xs_direct++; if (pf) pf->xs_direct++; }
            } else m->xs_list[keep++] = i;
        }
        nmiss = keep;
        if (pf) { q1 = pnow(); pf->xs_issue += q1 - q0; q0 = q1; }
    }
    int C = sn_cuda_xs_chunk(c), half = 0;
    for (int b = 0; b < nmiss && !rc; b += C, half ^= 1) {
        int e = b + C < nmiss ? b + C : nmiss, cnt = e - b, j0 = half * C;
        sn_cuda_xs_chunk_begin(c);
        if (sn_cuda_xs_stage_wait(c, j0, cnt) != 0) { rc = -1; break; }
        if (pf) { q1 = pnow(); pf->xs_wait += q1 - q0; q0 = q1; }
        for (int k = b; k < e && !rc; k++) {
            int i = m->xs_list[k];
            int s = sn_cuda_xs_claim(c, (int)ekey(m, li, w->used[i]));
            if (s < 0) rc = -1; else m->xs_slot[k - b] = s;
            if (!rc && w->emode[i] == 3 && !m->xs_nopin) { int pb = sn_cuda_xs_pool_get(c); w->epend[i].pb = pb >= 0 ? pb + 1 : 0; }
        }
        if (rc) break;
        xprep_ctx pc = { m, l, li, w, b, j0, cnt, 0 };
        nemo_pool_for(&m->pool, cnt, xprep_job, &pc);
        if (pc.fail) { rc = -1; break; }
        if (pf) { q1 = pnow(); pf->xs_prep += q1 - q0; q0 = q1; }
        for (int k = b; k < e && !rc; k++) {
            int i = m->xs_list[k], s = m->xs_slot[k - b], key = (int)ekey(m, li, w->used[i]);
            ecent *pe = &w->epend[i];
            if (w->emode[i] == 3 && pe->pb && !pe->pend) { sn_cuda_xs_pool_put(c, pe->pb - 1); pe->pb = 0; }   /* read failed: staged instead */
            if (w->emode[i] == 3 && pe->pb) { rc = sn_cuda_xs_upload_pool(c, pe->pb - 1, s, key); if (!rc) { m->xs_direct++; if (pf) pf->xs_direct++; } }
            else rc = sn_cuda_xs_upload(c, j0 + (k - b), s, key);
            if (!rc) rc = xs_compute(m, w, i, s, fused);
        }
        if (pf) { q1 = pnow(); pf->xs_issue += q1 - q0; q0 = q1; }
    }
    if (!rc && fused && !m->xs_nopf && m->xs_next) {
        if (xs_prefetch(m, li, w, n, &half) != 0) rc = -1;
        if (pf) { q1 = pnow(); pf->xs_pref += q1 - q0; q0 = q1; }
    }
    if (!rc) rc = sn_cuda_xs_layer_end(c, w->EO, n * K);
    if (pf) pf->xs_end += pnow() - q0;
    if (rc) {
        sn_cuda_xs_abort(c);
        for (int i = 0; i < nu; i++) if (w->emode[i] == 3 && (w->epend[i].pend || w->epend[i].pb)) { ec_release(m, &w->epend[i]); w->epend[i].pend = 0; }
        fprintf(stderr, "sn_model: the GPU expert stream failed on layer %d; the CPU lanes take over from here\n", li);
        m->xs_on = 0;
        return -1;
    }
    memset(m->xs_done, 1, (size_t)nu);
    return 0;
}

int sn_model_gpu_experts(sn_model *m, int on, char *why, int whylen)
{
    why[0] = 0;
    if (!on) { m->xs_on = 0; return 0; }
    const sn_geom *g = &m->g;
    if (!m->cuda) { snprintf(why, (size_t)whylen, "no CUDA lane"); return -1; }
    if (!dtc_wanted(m, g->moe_inter, g->moe_latent, 1, 1) || !dtc_wanted(m, g->moe_latent, g->moe_inter, 1, 1)) {
        snprintf(why, (size_t)whylen, "DTC is off (or does not pay on the expert shapes), so the card's decode kernels would not match the CPU's"); return -1;
    }
    const layer_t *l = NULL;
    for (int i = 0; i < m->n_all && !l; i++) if (m->L[i].eup && m->L[i].eup[0]) l = &m->L[i];
    if (!l) { snprintf(why, (size_t)whylen, "no MoE layer"); return -1; }
    int E = g->moe_experts;
    if (!m->xs_oidx) {
        m->xs_oidx = (int *)xalloc((size_t)PF_CHUNK * g->moe_topk * sizeof(int));
        m->xs_list = (int *)xalloc((size_t)E * sizeof(int)); m->xs_slot = (int *)xalloc((size_t)E * sizeof(int));
        m->xs_done = (unsigned char *)xalloc((size_t)E);
        if (!m->xs_oidx || !m->xs_list || !m->xs_slot || !m->xs_done) { snprintf(why, (size_t)whylen, "out of memory"); return -1; }
    }
    {   /* switches for the three stream speed-ups (all on by default; tests and the engine's flags may turn them off) */
        const char *v;
        if ((v = getenv("SN_XS_PREFETCH")) && !strcmp(v, "0")) m->xs_nopf = 1;
        if ((v = getenv("SN_XS_PINNED")) && !strcmp(v, "0")) m->xs_nopin = 1;
        if ((v = getenv("SN_XS_SLOTCAP")) && !strcmp(v, "0")) m->xs_nocap = 1;
    }
    if (!m->xs_next) {
        int K = g->moe_topk;
        m->xs_next = (int *)xalloc((size_t)m->n_all * sizeof(int)); m->xs_ph = (float *)xalloc((size_t)g->hidden * 4);
        m->xs_ptw = (float *)xalloc((size_t)K * 4); m->xs_ptop = (int *)xalloc((size_t)K * sizeof(int));
        m->xs_pfx = (int *)xalloc((size_t)K * sizeof(int)); m->xs_pfl_keys = (int *)xalloc((size_t)K * sizeof(int));
        if (!m->xs_next || !m->xs_ph || !m->xs_ptw || !m->xs_ptop || !m->xs_pfx || !m->xs_pfl_keys) { snprintf(why, (size_t)whylen, "out of memory"); return -1; }
        for (int i = 0; i < m->n_all; i++) {      /* the next MoE layer of the main stack (none after the last, none for the MTP head) */
            m->xs_next[i] = -1;
            if (i < g->n_layers) for (int j = i + 1; j < g->n_layers; j++) if (m->L[j].eup) { m->xs_next[i] = j; break; }
        }
        m->xs_pfli = -1;
    }
    if (m->ec) {   /* reopening the stream frees the pinned pool: drop cache entries that live in it first */
        for (size_t k = 0; k < (size_t)m->n_all * (size_t)E; k++) if (m->ec[k].ram && m->ec[k].pb) cache_evict(m, k);
    }
    sn_xs_geom xg;
    memset(&xg, 0, sizeof xg);
    xg.Lt = g->moe_latent; xg.I = g->moe_inter; xg.K = g->moe_topk; xg.nmax = PF_CHUNK;
    xg.up_blocks = l->eup[0]->nblocks; xg.dn_blocks = l->edn[0]->nblocks;
    xg.nkeys = m->n_all * E; xg.simd = sn_gemm_simd();
    xg.stream_cap = m->xs_nocap ? 0 : 300;    /* measured on the real pack, card holding only the stream */
    for (int i = 0; i < m->n_all; i++) if (m->L[i].eup) for (int e = 0; e < E; e++)
        if (!m->L[i].eup[e] || !m->L[i].edn[e] || m->L[i].eup[e]->nblocks != xg.up_blocks || m->L[i].edn[e]->nblocks != xg.dn_blocks) {
            snprintf(why, (size_t)whylen, "experts of unequal size (layer %d expert %d)", i, e); return -1;
        }
    if (sn_cuda_xs_open(m->cuda, &m->pack, &xg, why, whylen) != 0) return -1;
    if (m->gpu_tier) { m->gpu_tier = 0; sn_cuda_set_cache_budget(m->cuda, 0); }
    if (m->ec && !m->xs_nopin) sn_cuda_xs_pool_limit(m->cuda, (long long)m->ec_budget);
    m->xs_on = 1;
    return 0;
}

void sn_model_stream_options(sn_model *m, int prefetch, int pinned_cache, int slot_cap)
{
    m->xs_nopf = !prefetch; m->xs_nopin = !pinned_cache; m->xs_nocap = !slot_cap;
}

int sn_model_gpu_experts_on(const sn_model *m) { return m->xs_on; }
void sn_model_gpu_experts_info(const sn_model *m, sn_xs_info *out)
{
    if (m->cuda) { sn_cuda_xs_note(m->cuda, m->xs_direct, m->xs_pf_sent, m->xs_pf_right); sn_cuda_xs_info(m->cuda, out); } else memset(out, 0, sizeof *out);
    out->on = m->xs_on;
}

static void prof_free(sn_prof *p)
{
    if (!p) return;
    free(p->next_moe); free(p->prev); free(p->early); free(p->has_prev); free(p->has_early); free(p->cnt); free(p->tmp); free(p->tw);
    free(p->n_tok);
    free(p);
}

static sn_prof *prof_get(sn_model *m)
{
    if (m->prof_state == 0) {
        const char *e = getenv("SN_PROF");
        m->prof_state = (e && *e && strcmp(e, "0")) ? 1 : -1;
        if (m->prof_state == 1) {
            const sn_geom *g = &m->g;
            int nl = g->n_layers, E = g->moe_experts, K = g->moe_topk;
            sn_prof *p = (sn_prof *)calloc(1, sizeof *p);
            if (p) {
                p->nl = nl; p->E = E; p->K = K;
                p->next_moe = (int *)calloc((size_t)nl, sizeof(int));
                p->prev = (int *)calloc((size_t)nl * K, sizeof(int)); p->early = (int *)calloc((size_t)nl * K, sizeof(int));
                p->has_prev = (unsigned char *)calloc((size_t)nl, 1); p->has_early = (unsigned char *)calloc((size_t)nl, 1);
                p->cnt = (uint32_t *)calloc((size_t)nl * E, 4); p->tmp = (float *)malloc((size_t)g->hidden * 4); p->tw = (float *)malloc((size_t)K * 4);
                p->n_tok = (long *)calloc((size_t)nl * 8, sizeof(long));
            }
            if (!p || !p->next_moe || !p->prev || !p->early || !p->has_prev || !p->has_early || !p->cnt || !p->tmp || !p->tw || !p->n_tok || E > 4096) {
                prof_free(p); m->prof_state = -1; return NULL;
            }
            p->n_prev = p->n_tok + nl; p->n_early = p->n_tok + 2 * nl; p->ov_prev = p->n_tok + 3 * nl; p->ov_early = p->n_tok + 4 * nl;
            p->ov_hot = p->n_tok + 5 * nl; p->ov_union = p->n_tok + 6 * nl; p->sz_union = p->n_tok + 7 * nl;
            for (int i = 0; i < nl; i++) {
                p->next_moe[i] = -1;
                for (int j = i + 1; j < nl; j++) if (g->types[j] != 'M' && g->types[j] != 'A') { p->next_moe[i] = j; break; }
            }
            m->prof = p;
        }
    }
    return m->prof;
}

static int pres(const sn_model *m, const sn_tensor *t) { return m->cuda && t && sn_cuda_resident(m->cuda, &m->pack, t); }

/* router predictability at MoE layer li (decode tokens), then the next MoE layer's router run one layer early */
static void prof_route(sn_model *m, sn_prof *p, int li, fws *w, int n, int fused)
{
    const sn_geom *g = &m->g;
    int E = p->E, K = p->K, H = g->hidden;
    if (li >= p->nl) return;
    uint32_t *cn = p->cnt + (size_t)li * E;
    int early_now = p->has_early[li]; p->has_early[li] = 0;
    if (fused && n == 1) {
        unsigned char inA[4096], inU[4096], hot[4096];
        memset(inA, 0, (size_t)E); memset(inU, 0, (size_t)E); memset(hot, 0, (size_t)E);
        for (int k = 0; k < K; k++) inA[w->Top[k]] = 1;
        p->n_tok[li]++;
        for (int k = 0; k < K; k++) {   /* the layer's K most used experts so far (ties: lower index) */
            int b = -1;
            for (int e = 0; e < E; e++) if (!hot[e] && (b < 0 || cn[e] > cn[b])) b = e;
            hot[b] = 1; p->ov_hot[li] += inA[b];
        }
        if (p->has_prev[li]) { p->n_prev[li]++; for (int k = 0; k < K; k++) p->ov_prev[li] += inA[p->prev[(size_t)li * K + k]]; }
        if (early_now) { p->n_early[li]++; for (int k = 0; k < K; k++) p->ov_early[li] += inA[p->early[(size_t)li * K + k]]; }
        if (early_now && p->has_prev[li]) {
            for (int k = 0; k < K; k++) { inU[p->prev[(size_t)li * K + k]] = 1; inU[p->early[(size_t)li * K + k]] = 1; }
            for (int e = 0; e < E; e++) if (inU[e]) { p->sz_union[li]++; p->ov_union[li] += inA[e]; }
        }
    }
    for (int t = 0; t < n; t++) for (int k = 0; k < K; k++) cn[w->Top[(size_t)t * K + k]]++;
    memcpy(p->prev + (size_t)li * K, w->Top + (size_t)(n - 1) * K, (size_t)K * sizeof(int)); p->has_prev[li] = 1;
    int nx = p->next_moe[li];
    if (nx >= 0 && fused && n == 1) {
        nemo_rmsnorm(w->X, m->L[nx].norm_w, p->tmp, H, g->eps);
        if (route_token(m, nx, p->tmp, p->early + (size_t)nx * K, p->tw) == 0) p->has_early[nx] = 1;
    }
}

#define PT0() double pt_ = m->prof ? pnow() : 0.0
#define PT(c, t) do { if (m->prof) { double q_ = pnow(); int f_ = fused != 0; m->prof->ms[f_][c] += q_ - pt_; m->prof->calls[f_][c]++; m->prof->on_card[f_][c] += pres(m, (t)); pt_ = q_; } } while (0)

/* One residual block (norm, mixer, add) for the n token rows in w->X. li indexes m->L / m->mam / m->att. */
static int run_layer(sn_model *m, fws *w, int li, char type, int n, int fused, int snap)
{
    const sn_geom *g = &m->g;
    const layer_t *l = &m->L[li];
    int H = g->hidden, E = g->moe_experts, K = g->moe_topk, Lt = g->moe_latent;
    int qd = g->a_heads * g->a_hd, kd = g->a_kv * g->a_hd;
    int inner_w = g->m_inner > qd ? g->m_inner : qd;
    int rc = 0;
    prof_get(m);
    PT0();
    for (int t = 0; t < n; t++) nemo_rmsnorm(w->X + (size_t)t * H, l->norm_w, w->Hn + (size_t)t * H, H, g->eps);
    PT(PC_NORM, NULL);
    if (type == 'M') {
        rc = lin(m, fused, l->in_proj, w->Hn, n, H, g->m_in_proj, w->Pr, 0);
        PT(PC_MIN, l->in_proj);
        for (int t = 0; t < n && !rc; t++) {
            rc = nemo_mamba_decode(&m->mam[li], w->Pr + (size_t)t * g->m_in_proj, 0, l->conv_w, l->conv_b, l->a_log, l->dvec, l->dt_bias, l->gnorm_w, g->m_groups, g->eps, w->In + (size_t)t * inner_w);
            if (!rc && snap && t < n - 1) {
                memcpy(m->snap_ssm[t][li], m->mam[li].ssm_state, (size_t)g->m_heads * g->m_head_dim * g->m_state * 4);
                memcpy(m->snap_conv[t][li], m->mam[li].conv_win, (size_t)g->m_conv_dim * 4 * 4);
            }
        }
        PT(PC_MCORE, NULL);
        if (!rc) {   /* the matrix kernels want densely packed rows of m_inner */
            if (inner_w != g->m_inner) for (int t = 0; t < n; t++) memmove(w->In + (size_t)t * g->m_inner, w->In + (size_t)t * inner_w, (size_t)g->m_inner * 4);
            rc = lin(m, fused, l->out_proj, w->In, n, g->m_inner, H, w->Y, 0);
            PT(PC_MOUT, l->out_proj);
        }
    } else if (type == 'A') {
        rc = lin(m, fused, l->q, w->Hn, n, H, qd, w->Q, 0);
        if (!rc) rc = lin(m, fused, l->k, w->Hn, n, H, kd, w->Kb, 0);
        if (!rc) rc = lin(m, fused, l->v, w->Hn, n, H, kd, w->Vb, 0);
        PT(PC_QKV, l->q);
        for (int t = 0; t < n && !rc; t++) {
            rc = nemo_attn_push(&m->att[li], w->Kb + (size_t)t * kd, w->Vb + (size_t)t * kd);
            if (!rc) rc = nemo_attn_decode(&m->att[li], w->Q + (size_t)t * qd, w->In + (size_t)t * qd);
        }
        PT(PC_ACORE, NULL);
        if (!rc) rc = lin(m, fused, l->o, w->In, n, qd, H, w->Y, 0);
        PT(PC_O, l->o);
    } else {
        for (int t = 0; t < n && !rc; t++) rc = route_token(m, li, w->Hn + (size_t)t * H, w->Top + (size_t)t * K, w->Tw + (size_t)t * K);
        if (rc) return rc;
        PT(PC_ROUTE, NULL);
        if (m->prof) { prof_route(m, m->prof, li, w, n, fused); pt_ = pnow(); }
        if (m->on_moe && li < g->n_layers) m->on_moe(m->hook_user, li, w->Top + (size_t)(n - 1) * K, K);
        rc = lin(m, fused, l->fc1, w->Hn, n, H, Lt, w->Lat, 0);
        if (rc) return rc;
        PT(PC_FC1, l->fc1);
        memset(w->cnt, 0, (size_t)E * sizeof(int));
        for (int t = 0; t < n; t++) for (int k = 0; k < K; k++) w->cnt[w->Top[(size_t)t * K + k]]++;
        int nu = 0; w->start[0] = 0;
        for (int e = 0; e < E; e++) if (w->cnt[e]) { w->used[nu] = e; w->start[nu + 1] = w->start[nu] + w->cnt[e]; w->fill[e] = w->start[nu]; nu++; }
        for (int t = 0; t < n; t++) for (int k = 0; k < K; k++) { int e = w->Top[(size_t)t * K + k]; w->etok[w->fill[e]] = t; w->eslot[w->fill[e]] = k; w->fill[e]++; }
        if (m->ec) cache_count(m, li, w, nu);
        if (fused) {
            for (int t = 0; t < n; t++) {
                double *Pv = w->PL + (size_t)t * (Lt + 1);
                Pv[0] = 0.0;
                for (int j = 0; j < Lt; j++) Pv[j + 1] = Pv[j] + (double)w->Lat[(size_t)t * Lt + j];
            }
            if (m->ec) cache_plan(m, li, w, nu);
            fx_ctx fc = { m, l, w->Lat, w->PL, w->EO, w->used, nu, w->start, w->etok, w->eslot, 0, li, w->emode, w->epend, NULL };
            if (m->ec && m->gpu_tier) cache_gpu_serve(m, li, w, nu, &fc);
            if (m->xs_on && xs_layer(m, li, w, n, nu, 1) == 0) fc.xdone = m->xs_done;
            else if (!m->ec) for (int i = 0; i < nu; i++) w->emode[i] = 0;
            nemo_pool_for(&m->pool, nu, fx_job, &fc);
            if (fc.fail) { if (m->ec) cache_adopt(m, li, w, nu, 1); return -1; }
            if (m->ec) cache_adopt(m, li, w, nu, 0);
        } else {
            if (!(m->xs_on && xs_layer(m, li, w, n, nu, 0) == 0)) {
                pfx_ctx pc = { m, l, w->Lat, w->EO, w->used, nu, w->start, w->etok, w->eslot, 0 };
                nemo_pool_for(&m->pool, nu, pfx_job, &pc);
                if (pc.fail) return -1;
            }
        }
        for (int t = 0; t < n; t++) {
            float *a = w->Acc + (size_t)t * Lt; memset(a, 0, (size_t)Lt * 4);
            for (int k = 0; k < K; k++) { float wt = w->Tw[(size_t)t * K + k]; const float *eo = w->EO + ((size_t)t * K + k) * Lt; for (int j = 0; j < Lt; j++) a[j] += wt * eo[j]; }
        }
        PT(PC_EXP, NULL);
        rc = lin(m, fused, l->fc2, w->Acc, n, Lt, H, w->Y, 0);
        PT(PC_FC2, l->fc2);
        if (!rc) rc = lin(m, fused, l->sh_up, w->Hn, n, H, g->moe_shared, w->SM, 2);
        if (!rc) rc = lin(m, fused, l->sh_dn, w->SM, n, g->moe_shared, H, w->Sh, 0);
        PT(PC_SHARED, l->sh_up);
        if (!rc) for (size_t j = 0; j < (size_t)n * H; j++) w->Y[j] += w->Sh[j];
    }
    if (rc) return rc;
    for (size_t j = 0; j < (size_t)n * H; j++) w->X[j] += w->Y[j];
    PT(PC_NORM, NULL);
    return 0;
}

typedef struct {
    const int *tok; const int *erow; const float *emb; int n;
    int fused;            /* the multi-vector fused kernels (n <= SN_NV); else expand + GEMM */
    float *logits_last;   /* logits of the last token, or NULL */
    float *logits_all;    /* n * vocab (needs fused), or NULL */
    int snap;             /* keep Mamba snapshots after tokens 0 .. n-2 (speculative rollback) */
} fwd_args;

static int fwd_chunk(sn_model *m, const fwd_args *a)
{
    const sn_geom *g = &m->g;
    int H = g->hidden, n = a->n, V = g->vocab;
    int rc = 0;
    fws w;
    float *hn = NULL;
    if (fws_open(m, &w, n) != 0) return -1;
    if (a->snap && snap_ensure(m, n - 1) != 0) { rc = -1; goto done; }
    for (int t = 0; t < n; t++) {
        if (a->erow && a->erow[t] >= 0) memcpy(w.X + (size_t)t * H, a->emb + (size_t)a->erow[t] * H, (size_t)H * 4);
        else if (a->tok[t] < 0 || a->tok[t] >= V || sn_expand(&m->pack, &m->main_rd, m->emb, (uint64_t)a->tok[t] * H, (uint64_t)H, w.X + (size_t)t * H) != 0) { rc = -1; goto done; }
    }
    for (int i = 0; i < g->n_layers && !rc; i++) {
        if (m->on_layer) m->on_layer(m->hook_user, i, g->types[i]);
        rc = run_layer(m, &w, i, g->types[i], n, a->fused, a->snap);
        if (!rc && m->probe) memcpy(m->probe + (size_t)i * H, w.X + (size_t)(n - 1) * H, (size_t)H * 4);
    }
    if (rc) { m->state_valid = 0; goto done; }
    int need_all = (a->logits_all != NULL) || m->hist_hn != NULL;
    if (need_all || a->logits_last) {
        hn = (float *)malloc((size_t)n * H * 4);
        if (!hn) { m->state_valid = 0; rc = -1; goto done; }
        for (int t = need_all ? 0 : n - 1; t < n; t++) nemo_rmsnorm(w.X + (size_t)t * H, m->normf_w, hn + (size_t)t * H, H, g->eps);
    }
    if (m->hist_hn && m->n_pos + n <= m->max_pos) {
        memcpy(m->hist_hn + (size_t)m->n_pos * H, hn, (size_t)n * H * 4);
        for (int t = 0; t < n; t++) m->hist_tok[m->n_pos + t] = (a->erow && a->erow[t] >= 0) ? -1 : a->tok[t];
    }
    double th_ = m->prof ? pnow() : 0.0;
    if (m->prof) m->prof->tokens[a->fused != 0] += n;
    if (a->logits_all) {
        if (lin_n(m, m->head, hn, n, a->logits_all, V, H) != 0) { m->state_valid = 0; rc = -1; goto done; }
        if (a->logits_last) memcpy(a->logits_last, a->logits_all + (size_t)(n - 1) * V, (size_t)V * 4);
    } else if (a->logits_last) {
        if (lin_n(m, m->head, hn + (size_t)(n - 1) * H, 1, a->logits_last, V, H) != 0) { m->state_valid = 0; rc = -1; goto done; }
    }
    if (m->prof && (a->logits_all || a->logits_last)) {
        int f = a->fused != 0; m->prof->ms[f][PC_HEAD] += pnow() - th_; m->prof->calls[f][PC_HEAD]++; if (pres(m, m->head)) m->prof->on_card[f][PC_HEAD]++;
    }
    m->n_pos += n;
done:
    free(hn);
    fws_close(&w);
    return rc ? -1 : 0;
}

int sn_model_prefill(sn_model *m, const int *tokens, const int *embed_row, const float *embeds, int n, float *logits)
{
    if (n <= 0) return 0;
    if (!m->state_valid) return -2;
    if (m->n_pos + n > m->max_pos) return -3;
    for (int at = 0; at < n; at += PF_CHUNK) {
        int c = n - at < PF_CHUNK ? n - at : PF_CHUNK;
        int last = at + c == n;
        fwd_args a = { tokens + at, embed_row ? embed_row + at : NULL, embeds, c, 0, last ? logits : NULL, NULL, 0 };
        if (fwd_chunk(m, &a) != 0) return -1;
    }
    return 0;
}

int sn_model_threads(const sn_model *m) { return m->nthreads; }
int sn_model_pos(const sn_model *m) { return m->n_pos; }
void sn_model_set_hooks(sn_model *m, sn_layer_cb a, sn_moe_cb b, void *u) { m->on_layer = a; m->on_moe = b; m->hook_user = u; }
void sn_model_set_probe(sn_model *m, float *buf) { m->probe = buf; }
void sn_model_set_dtc(sn_model *m, int mode) { m->dtc = mode < 0 ? 0 : mode > 2 ? 2 : mode; }
int sn_model_dtc(const sn_model *m) { return m->dtc; }

void sn_model_reset(sn_model *m)
{
    for (int i = 0; i < m->g.n_layers; i++) {
        if (m->g.types[i] == 'M') {
            nemo_mamba_state *s = &m->mam[i];
            memset(s->ssm_state, 0, (size_t)s->heads * s->head_dim * s->state_size * sizeof(float));
            memset(s->conv_win, 0, (size_t)s->conv_dim * 4 * sizeof(float));
        } else if (m->g.types[i] == 'A') m->att[i].len = 0;
    }
    if (m->has_mtp) m->att[m->g.n_layers].len = 0;
    m->mtp_done = 0; m->mtp_slots = 0;
    m->n_pos = 0; m->state_valid = 1;
}

/* ------------------------------------------------------------- step */

int sn_model_step(sn_model *m, int token, float *logits)
{
    if (token < 0 || token >= m->g.vocab) return -1;
    if (!m->state_valid) return -2;
    if (m->n_pos >= m->max_pos) return -3;
    fwd_args a = { &token, NULL, NULL, 1, 1, logits, NULL, 0 };
    return fwd_chunk(m, &a);
}

/* The same step, but the input is an embedding vector (hidden floats), e.g. a projected image patch. */
int sn_model_step_embed(sn_model *m, const float *embed, float *logits)
{
    int tok0 = 0, row0 = 0;
    if (!m->state_valid) return -2;
    if (m->n_pos >= m->max_pos) return -3;
    fwd_args a = { &tok0, &row0, embed, 1, 1, logits, NULL, 0 };
    return fwd_chunk(m, &a);
}

/* n (<= SN_NV) tokens in one fused pass over the packed data. Every token's logits are bit-identical to what n separate
 * sn_model_step calls would give. snap: keep Mamba snapshots so sn_model_rollback can drop the tail. */
int sn_model_step_n(sn_model *m, const int *tokens, int n, float *logits_all, int snap)
{
    if (n < 1 || n > SN_NV) return -1;
    if (!m->state_valid) return -2;
    if (m->n_pos + n > m->max_pos) return -3;
    for (int t = 0; t < n; t++) if (tokens[t] < 0 || tokens[t] >= m->g.vocab) return -1;
    fwd_args a = { tokens, NULL, NULL, n, 1, NULL, logits_all, snap };
    return fwd_chunk(m, &a);
}

/* After sn_model_step_n(..., snap=1) of n tokens that began at position base: keep only the first `keep` of them. */
int sn_model_rollback(sn_model *m, int base, int keep, int n)
{
    const sn_geom *g = &m->g;
    if (keep < 1 || keep > n || base < 0 || m->n_pos != base + n) return -1;
    if (keep == n) return 0;
    if (keep - 1 >= m->snap_n) return -1;
    for (int i = 0; i < g->n_layers; i++) {
        if (g->types[i] == 'M') {
            memcpy(m->mam[i].ssm_state, m->snap_ssm[keep - 1][i], (size_t)g->m_heads * g->m_head_dim * g->m_state * 4);
            memcpy(m->mam[i].conv_win, m->snap_conv[keep - 1][i], (size_t)g->m_conv_dim * 4 * 4);
        } else if (g->types[i] == 'A') m->att[i].len = base + keep;
    }
    m->n_pos = base + keep;
    return 0;
}

const sn_pack *sn_model_pack(const sn_model *m) { return &m->pack; }

/* Fill the GPU with the dense tensors (lm_head first, then each layer's mixers and the MoE's latent and shared
 * projections) until the budget is used. Routed experts stay on the CPU lanes (streamed from the pack). */
int sn_model_cuda(sn_model *m, long long budget_bytes, char *why, int whylen)
{
    if (m->cuda) return m->cuda ? sn_cuda_count(m->cuda) : 0;
    m->cuda = sn_cuda_open(budget_bytes, m->pack.n, why, whylen);
    if (!m->cuda) return -1;
    const sn_tensor *seq[8]; int ns;
    sn_cuda_load(m->cuda, &m->pack, &m->main_rd, m->head);
    for (int i = 0; i < m->g.n_layers; i++) {
        const layer_t *l = &m->L[i];
        ns = 0;
        if (m->g.types[i] == 'M') { seq[ns++] = l->in_proj; seq[ns++] = l->out_proj; }
        else if (m->g.types[i] == 'A') { seq[ns++] = l->q; seq[ns++] = l->k; seq[ns++] = l->v; seq[ns++] = l->o; }
        else { seq[ns++] = l->fc1; seq[ns++] = l->fc2; seq[ns++] = l->sh_up; seq[ns++] = l->sh_dn; }
        for (int k = 0; k < ns; k++) sn_cuda_load(m->cuda, &m->pack, &m->main_rd, seq[k]);
    }
    return sn_cuda_count(m->cuda);
}
const char *sn_model_accel(const sn_model *m) { return m->cuda && (sn_cuda_count(m->cuda) > 0 || m->xs_on) ? "cuda" : "cpu"; }
long long sn_model_gpu_bytes(const sn_model *m) { return m->cuda ? sn_cuda_used(m->cuda) : 0; }
int sn_model_gpu_count(const sn_model *m) { return m->cuda ? sn_cuda_count(m->cuda) : 0; }
const char *sn_model_gpu_name(const sn_model *m) { return m->cuda ? sn_cuda_name(m->cuda) : ""; }
nemo_pool *sn_model_pool(sn_model *m) { return &m->pool; }


/* Hot-expert cache: ram_bytes of RAM-resident experts (most-used first, by token count), and, when the CUDA lane is open,
 * vram_bytes of them mirrored in VRAM. Either may be 0. Contents never change what is computed. */
int sn_model_cache(sn_model *m, long long ram_bytes, long long vram_bytes, char *why, int whylen)
{
    if (ram_bytes <= 0) return 0;
    size_t n = (size_t)m->n_all * (size_t)m->g.moe_experts;
    if (!m->ec) {
        m->ec = (ecent *)calloc(n, sizeof(ecent)); m->rlist = (int *)calloc(n, sizeof(int));
        m->gmid = (float *)xalloc((size_t)m->g.moe_inter * 4 + 64); m->gpmid = (double *)xalloc(((size_t)m->g.moe_inter + 1) * 8);
        if (!m->ec || !m->rlist || !m->gmid || !m->gpmid) { snprintf(why, (size_t)whylen, "out of memory"); return -1; }
    }
    m->ec_budget = (uint64_t)ram_bytes;
    m->gpu_tier = 0;
    if (vram_bytes > 0) {
        if (!m->cuda) snprintf(why, (size_t)whylen, "no CUDA lane: the VRAM tier of the expert cache is off");
        else if (!m->dtc) snprintf(why, (size_t)whylen, "DTC is off: the VRAM tier of the expert cache needs it (bit-identical kernels)");
        else if (!dtc_wanted(m, m->g.moe_inter, m->g.moe_latent, 1, 1) || !dtc_wanted(m, m->g.moe_latent, m->g.moe_inter, 1, 1)) snprintf(why, (size_t)whylen, "DTC does not pay on the expert shapes: the VRAM tier is off");
        else { sn_cuda_set_cache_budget(m->cuda, vram_bytes); m->gpu_tier = 1; why[0] = 0; }
    }
    return 0;
}

void sn_model_cache_stats(const sn_model *m, sn_cache_stats *s)
{
    memset(s, 0, sizeof *s);
    s->hits_ram = m->hits_ram; s->hits_gpu = m->hits_gpu; s->misses = m->misses; s->admits = m->admits; s->evicts = m->evicts;
    s->ram_bytes = m->ec_used; s->ram_budget = m->ec_budget; s->ram_experts = m->nrlist;
    if (m->cuda) { s->gpu_bytes = (uint64_t)sn_cuda_cache_used(m->cuda); s->gpu_experts = sn_cuda_cache_count(m->cuda) / 2; }
    s->gpu_puts = m->gpu_puts; s->gpu_drops = m->gpu_drops;
}

/* ------------------------------------------------------------- MTP head and speculative decoding */

int sn_model_has_mtp(const sn_model *m) { return m->has_mtp; }
int sn_model_mtp_on(const sn_model *m) { return m->mtp_on; }

int sn_model_mtp_enable(sn_model *m, int on)
{
    if (!on) { m->mtp_on = 0; free(m->hist_hn); free(m->hist_tok); m->hist_hn = NULL; m->hist_tok = NULL; return 0; }
    if (!m->has_mtp) return -1;
    if (m->mtp_on) return 0;
    m->hist_hn = (float *)xalloc((size_t)m->max_pos * m->g.hidden * 4);
    m->hist_tok = (int *)xalloc((size_t)m->max_pos * sizeof(int));
    if (!m->hist_hn || !m->hist_tok) { free(m->hist_hn); free(m->hist_tok); m->hist_hn = NULL; m->hist_tok = NULL; return -1; }
    for (int i = 0; i < m->max_pos; i++) m->hist_tok[i] = -1;
    m->mtp_on = 1; m->mtp_done = 0; m->mtp_slots = 0;
    m->att[m->g.n_layers].len = 0;
    return 0;
}

/* The head on n (target hidden, next token) pairs, in order: fuse, one attention block, one MoE block, closing norm.
 * Fills hid_last (hidden floats) and logits_last (vocab floats) for the last pair when they are non-NULL. */
static int mtp_pairs(sn_model *m, const float *hn, const int *next, int n, int fused, float *logits_last, float *hid_last)
{
    const sn_geom *g = &m->g;
    int H = g->hidden, li0 = g->n_layers, li1 = g->n_layers + 1;
    const layer_t *l0 = &m->L[li0], *l1 = &m->L[li1];
    fws w; float *cat = NULL, *tmp = NULL, *out = NULL;
    int rc = 0;
    if (fws_open(m, &w, n) != 0) return -1;
    cat = (float *)malloc((size_t)n * 2 * H * 4); tmp = (float *)malloc((size_t)H * 4); out = (float *)malloc((size_t)n * H * 4);
    if (!cat || !tmp || !out) { rc = -1; goto done; }
    for (int t = 0; t < n; t++) {
        if (next[t] < 0 || next[t] >= g->vocab || sn_expand(&m->pack, &m->main_rd, m->emb, (uint64_t)next[t] * H, (uint64_t)H, tmp) != 0) { rc = -1; goto done; }
        nemo_rmsnorm(tmp, l0->enorm_w, cat + (size_t)t * 2 * H, H, g->eps);
        nemo_rmsnorm(hn + (size_t)t * H, l0->hnorm_w, cat + (size_t)t * 2 * H + H, H, g->eps);
    }
    rc = lin(m, fused, l0->eh, cat, n, 2 * H, H, w.X, 0);
    if (!rc) rc = run_layer(m, &w, li0, 'A', n, fused, 0);
    if (!rc) rc = run_layer(m, &w, li1, 'E', n, fused, 0);
    if (rc) goto done;
    for (int t = 0; t < n; t++) nemo_rmsnorm(w.X + (size_t)t * H, l1->fin_w, out + (size_t)t * H, H, g->eps);
    if (hid_last) memcpy(hid_last, out + (size_t)(n - 1) * H, (size_t)H * 4);
    if (logits_last && lin_n(m, m->head, out + (size_t)(n - 1) * H, 1, logits_last, g->vocab, H) != 0) rc = -1;
    m->mtp_slots += n;
done:
    free(cat); free(tmp); free(out);
    fws_close(&w);
    return rc;
}

/* Bring the head's attention cache up to date for every position q < upto whose next token is known. */
static int mtp_catchup(sn_model *m, int upto)
{
    int H = m->g.hidden;
    int q = m->mtp_done;
    while (q < upto) {
        int qs[PF_CHUNK], nq = 0;
        int lim = upto - q < PF_CHUNK ? upto - q : PF_CHUNK;
        for (int i = 0; i < lim; i++) if (m->hist_tok[q + i + 1] >= 0) qs[nq++] = q + i;
        if (nq > 0) {
            float *hn = (float *)malloc((size_t)nq * H * 4); int *nx = (int *)malloc((size_t)nq * sizeof(int));
            if (!hn || !nx) { free(hn); free(nx); return -1; }
            for (int i = 0; i < nq; i++) { memcpy(hn + (size_t)i * H, m->hist_hn + (size_t)qs[i] * H, (size_t)H * 4); nx[i] = m->hist_tok[qs[i] + 1]; }
            int rc = mtp_pairs(m, hn, nx, nq, nq <= SN_NV, NULL, NULL);
            free(hn); free(nx);
            if (rc) return -1;
        }
        q += lim;
    }
    m->mtp_done = upto > m->mtp_done ? upto : m->mtp_done;
    return 0;
}

static int argmax_f(const float *l, int n)
{
    int b = 0;
    for (int i = 1; i < n; i++) if (l[i] > l[b]) b = i;
    return b;
}

/* k greedy draft tokens that would follow `cur` (the chosen, not yet consumed, next token). */
static int mtp_draft(sn_model *m, int cur, int k, int *d)
{
    const sn_geom *g = &m->g;
    int H = g->hidden, V = g->vocab;
    if (m->n_pos < 1) return -1;
    if (mtp_catchup(m, m->n_pos - 1) != 0) return -1;
    float *logits = (float *)malloc((size_t)V * 4), *hid = (float *)malloc((size_t)H * 4);
    if (!logits || !hid) { free(logits); free(hid); return -1; }
    int rc = mtp_pairs(m, m->hist_hn + (size_t)(m->n_pos - 1) * H, &cur, 1, 1, logits, hid);
    if (!rc) { m->mtp_done = m->n_pos; d[0] = argmax_f(logits, V); }
    int keep = m->mtp_slots;
    for (int i = 1; i < k && !rc; i++) {
        float *hcopy = (float *)malloc((size_t)H * 4);
        if (!hcopy) { rc = -1; break; }
        memcpy(hcopy, hid, (size_t)H * 4);
        rc = mtp_pairs(m, hcopy, &d[i - 1], 1, 1, logits, hid);
        free(hcopy);
        if (!rc) d[i] = argmax_f(logits, V);
    }
    m->att[g->n_layers].len = keep; m->mtp_slots = keep;     /* the chained slots were speculative */
    free(logits); free(hid);
    return rc ? -1 : 0;
}

/* One speculative cycle. cur = the token already chosen from the last logits, not yet consumed. Consumes cur and
 * the accepted drafts; acc[0..ret) are the accepted tokens (they follow cur); logits_out is the logits after the last
 * consumed token, exactly as sn_model_step would leave them. Greedy: the result is token-identical to plain stepping.
 * A stop token (stop_a / stop_b) among the accepted ones ends the cycle there. Returns the accepted count, or -1. */
int sn_model_spec_cycle(sn_model *m, int cur, int kmax, int stop_a, int stop_b, int *acc, float *logits_out)
{
    if (kmax > SN_NV - 1) kmax = SN_NV - 1;
    if (m->n_pos + kmax + 1 > m->max_pos) kmax = m->max_pos - m->n_pos - 1;
    if (!m->mtp_on || kmax < 1) return sn_model_step(m, cur, logits_out) == 0 ? 0 : -1;
    int d[SN_NV];
    if (mtp_draft(m, cur, kmax, d) != 0) return sn_model_step(m, cur, logits_out) == 0 ? 0 : -1;
    return sn_model_spec_verify(m, cur, d, kmax, stop_a, stop_b, acc, logits_out);
}

/* Verify k given drafts after cur in one fused pass. Exposed so tests can drive the acceptance logic with chosen drafts. */
int sn_model_spec_verify(sn_model *m, int cur, const int *d, int kmax, int stop_a, int stop_b, int *acc, float *logits_out)
{
    int V = m->g.vocab, toks[SN_NV + 1];
    if (kmax < 1 || kmax > SN_NV - 1) return -1;
    if (!m->spec_logits) m->spec_logits = (float *)xalloc((size_t)SN_NV * V * 4);
    if (!m->spec_logits) return -1;
    int n = kmax + 1, base = m->n_pos;
    toks[0] = cur; for (int i = 0; i < kmax; i++) toks[i + 1] = d[i];
    if (sn_model_step_n(m, toks, n, m->spec_logits, 1) != 0) return -1;
    int j = 0;
    while (j < kmax && d[j] == argmax_f(m->spec_logits + (size_t)j * V, V)) j++;
    if (m->prof) {
        sn_prof *p = m->prof;
        for (int i = 0; i < kmax && i <= j; i++) { p->pos_tried[i]++; if (i < j) p->pos_ok[i]++; }
        if (j < kmax && p->nmiss < PROF_MISS) { p->miss_d[p->nmiss] = d[j]; p->miss_a[p->nmiss] = argmax_f(m->spec_logits + (size_t)j * V, V); p->miss_pos[p->nmiss++] = j; }
    }
    for (int i = 0; i < j; i++) if (d[i] == stop_a || d[i] == stop_b) { j = i + 1; break; }
    int keep = j + 1;
    if (keep < n && sn_model_rollback(m, base, keep, n) != 0) { m->state_valid = 0; return -1; }
    memcpy(logits_out, m->spec_logits + (size_t)(keep - 1) * V, (size_t)V * 4);
    for (int i = 0; i < j; i++) acc[i] = d[i];
    m->spec_cycles++; m->spec_drafted += kmax; m->spec_accepted += j;
    return j;
}
void sn_model_spec_stats(const sn_model *m, long *cycles, long *drafted, long *accepted)
{
    if (cycles) *cycles = m->spec_cycles;
    if (drafted) *drafted = m->spec_drafted;
    if (accepted) *accepted = m->spec_accepted;
}

void sn_model_prof_report(sn_model *m, FILE *f, void (*tokname)(void *u, int tok, char *buf, int len), void *u)
{
    sn_prof *p = prof_get(m);
    if (!p) return;
    const char *ph[2] = { "prefill", "decode" };
    for (int f2 = 1; f2 >= 0; f2--) {
        double tot = 0; for (int c = 0; c < PC_N; c++) tot += p->ms[f2][c];
        if (tot <= 0) continue;
        fprintf(f, "PROF %s: %ld tokens, %.0f ms in the layers and lm_head (%.1f ms per token)\n", ph[f2], p->tokens[f2], tot, p->tokens[f2] ? tot / p->tokens[f2] : 0.0);
        for (int c = 0; c < PC_N; c++) if (p->calls[f2][c])
            fprintf(f, "PROF %s %-24s %9.0f ms %5.1f%%  %7.1f ms/token  calls=%ld on_card=%ld\n", ph[f2], pc_name[c], p->ms[f2][c], 100.0 * p->ms[f2][c] / tot,
                    p->tokens[f2] ? p->ms[f2][c] / p->tokens[f2] : 0.0, p->calls[f2][c], p->on_card[f2][c]);
    }
    if (p->xs_layers) {
        long miss = p->xs_mode[0] + p->xs_mode[1] + p->xs_mode[2] + p->xs_mode[3];
        fprintf(f, "PROF stream: %ld layer passes; slot hits %ld, misses %ld (from RAM cache %ld, read into the cache %ld, read from the pack %ld)\n",
                p->xs_layers, p->xs_hit, miss, p->xs_mode[1], p->xs_mode[3], p->xs_mode[0] + p->xs_mode[2]);
        fprintf(f, "PROF stream time: staging wait %.0f ms | CPU fill of pinned staging %.0f ms | queueing copies and kernels %.0f ms | early fetch %.0f ms | waiting for the card at layer end %.0f ms\n",
                p->xs_wait, p->xs_prep, p->xs_issue, p->xs_pref, p->xs_end);
        fprintf(f, "PROF stream: %ld experts went up straight from pinned RAM cache blocks; early fetch sent %llu, router picked %llu of them\n",
                p->xs_direct, (unsigned long long)m->xs_pf_sent, (unsigned long long)m->xs_pf_right);
    }
    long a = 0, b = 0, c2 = 0, d = 0, n = 0, np = 0, ne = 0, su = 0;
    for (int li = 0; li < p->nl; li++) if (p->n_tok[li]) {
        n += p->n_tok[li]; np += p->n_prev[li]; ne += p->n_early[li]; a += p->ov_prev[li]; b += p->ov_early[li]; c2 += p->ov_hot[li]; d += p->ov_union[li]; su += p->sz_union[li];
        fprintf(f, "PROF route layer %2d: %ld tokens | same as last token %.1f%% | next-layer router run early %.1f%% | most used so far %.1f%% | both guesses together cover %.1f%% with %.1f experts\n",
                li, p->n_tok[li], p->n_prev[li] ? 100.0 * p->ov_prev[li] / (p->n_prev[li] * (double)p->K) : 0.0,
                p->n_early[li] ? 100.0 * p->ov_early[li] / (p->n_early[li] * (double)p->K) : 0.0, 100.0 * p->ov_hot[li] / (p->n_tok[li] * (double)p->K),
                p->n_early[li] ? 100.0 * p->ov_union[li] / (p->n_early[li] * (double)p->K) : 0.0, p->n_early[li] ? (double)p->sz_union[li] / p->n_early[li] : 0.0);
    }
    if (n) fprintf(f, "PROF route all layers: same as last token %.1f%% | next-layer router run early %.1f%% | most used so far %.1f%% | both guesses %.1f%% with %.1f experts (of %d picked)\n",
                   np ? 100.0 * a / (np * (double)p->K) : 0.0, ne ? 100.0 * b / (ne * (double)p->K) : 0.0, 100.0 * c2 / (n * (double)p->K),
                   ne ? 100.0 * d / (ne * (double)p->K) : 0.0, ne ? (double)su / ne : 0.0, p->K);
    if (p->pos_tried[0]) {
        fprintf(f, "PROF spec by draft position:");
        for (int i = 0; i < SN_NV && p->pos_tried[i]; i++) fprintf(f, " %d: %ld/%ld (%.0f%%)", i + 1, p->pos_ok[i], p->pos_tried[i], 100.0 * p->pos_ok[i] / p->pos_tried[i]);
        fprintf(f, "\n");
        for (int i = 0; i < p->nmiss; i++) {
            char bd[64] = "", ba[64] = "";
            if (tokname) { tokname(u, p->miss_d[i], bd, sizeof bd); tokname(u, p->miss_a[i], ba, sizeof ba); }
            fprintf(f, "PROF spec miss at position %d: drafted %d \"%s\", model wanted %d \"%s\"\n", p->miss_pos[i] + 1, p->miss_d[i], bd, p->miss_a[i], ba);
        }
    }
}
