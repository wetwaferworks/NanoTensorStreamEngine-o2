/* sn_model.c - see sn_model.h. Pure C99, zero mutexes (Little Nemo pool). */
#include "sn_model.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "sn_moe.h"
#include "sn_pack.h"
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

typedef struct {
    sn_rd rd;
    float *book;       /* entries*block floats */
    float *mid;        /* moe_inter */
    double *pmid;      /* moe_inter + 1 */
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
} layer_t;

struct sn_model {
    sn_geom g;
    sn_pack pack;
    sn_rd main_rd;
    nemo_pool pool;
    int nthreads;
    lane_t *lanes;
    layer_t *L;
    const sn_tensor *emb, *head, *normf_t;
    float *normf_w;
    nemo_mamba_state *mam;
    nemo_attn_cache *att;
    int max_pos, n_pos, state_valid;
    float *x, *h, *y, *proj, *inner, *qb, *kb, *vb, *lat, *eo, *acc, *sm, *sh, *gsc, *pf;
    double *P;
    float *book_shared;
    int *top; float *tw;
    sn_layer_cb on_layer; sn_moe_cb on_moe; void *hook_user;
};

static int lane_of(int n_items, int nthreads, int begin)
{
    int lanes = nthreads < n_items ? nthreads : n_items;
    for (int l = 0; l < lanes; l++)
        if ((int)((long)l * n_items / lanes) == begin) return l;
    return 0;
}

static void *xalloc(size_t n) { void *p = malloc(n ? n : 1); return p; }

/* ---- parallel dense linear: rows split across lanes ---- */
typedef struct {
    sn_model *m; const sn_tensor *t; const float *x; float *y; int rows, cols; int fail;
} lin_ctx;

static void lin_job(void *vc, int b, int e)
{
    lin_ctx *c = (lin_ctx *)vc;
    sn_model *m = c->m;
    lane_t *ln = &m->lanes[lane_of(c->rows, m->nthreads, b)];
    int rc = c->t->raw
        ? sn_raw_rows(&m->pack, &ln->rd, c->t, c->cols, c->x, b, e, c->y + b)
        : sn_pq_rows(&m->pack, &ln->rd, c->t, m->book_shared, c->cols, c->x, m->P, b, e, c->y + b);
    if (rc) c->fail = 1;
}

static int lin_par(sn_model *m, const sn_tensor *t, const float *x, float *y, int rows, int cols)
{
    if ((uint64_t)rows * (uint64_t)cols != t->count) return -1;
    if (!t->raw) {
        if (sn_load_book(&m->pack, &m->main_rd, t, m->book_shared) != 0) return -1;
        m->P[0] = 0.0;
        for (int j = 0; j < cols; j++) m->P[j + 1] = m->P[j] + (double)x[j];
    }
    lin_ctx c = { m, t, x, y, rows, cols, 0 };
    nemo_pool_for(&m->pool, rows, lin_job, &c);
    return c.fail ? -1 : 0;
}

/* ---- MoE: experts split across lanes ---- */
typedef struct {
    sn_model *m; const layer_t *l; const float *z; int k; float *eo; int fail;
} moe_ctx;

static void moe_job(void *vc, int b, int e)
{
    moe_ctx *c = (moe_ctx *)vc;
    sn_model *m = c->m;
    const sn_geom *g = &m->g;
    lane_t *ln = &m->lanes[lane_of(c->k, m->nthreads, b)];
    int L = g->moe_latent, I = g->moe_inter;
    for (int i = b; i < e; i++) {
        int ex = m->top[i];
        const sn_tensor *up = c->l->eup[ex], *dn = c->l->edn[ex];
        if (sn_load_book(&m->pack, &ln->rd, up, ln->book) != 0
            || sn_pq_rows(&m->pack, &ln->rd, up, ln->book, L, c->z, m->P + (size_t)0, 0, I, ln->mid) != 0) { c->fail = 1; return; }
        for (int j = 0; j < I; j++) ln->mid[j] = sn_relu2(ln->mid[j]);
        ln->pmid[0] = 0.0;
        for (int j = 0; j < I; j++) ln->pmid[j + 1] = ln->pmid[j] + (double)ln->mid[j];
        if (sn_load_book(&m->pack, &ln->rd, dn, ln->book) != 0
            || sn_pq_rows(&m->pack, &ln->rd, dn, ln->book, I, ln->mid, ln->pmid, 0, L, c->eo + (size_t)i * L) != 0) { c->fail = 1; return; }
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

static int moe_block(sn_model *m, int layer, const float *x_in, float *y_out)
{
    const sn_geom *g = &m->g;
    const layer_t *l = &m->L[layer];
    int H = g->hidden, E = g->moe_experts, K = g->moe_topk, Lt = g->moe_latent;
    /* router on the resident fp16 gate, experts split across lanes */
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
        taken[best] = 1; m->top[k] = best; m->tw[k] = s[best]; sum += s[best];
    }
    free(taken);
    if (g->moe_norm_topk) { float den = sum + 1e-20f; for (int k = 0; k < K; k++) m->tw[k] /= den; }
    for (int k = 0; k < K; k++) m->tw[k] *= g->moe_scale;
    if (m->on_moe) m->on_moe(m->hook_user, layer, m->top, K);
    /* latent input */
    if (lin_par(m, l->fc1, x_in, m->lat, Lt, H) != 0) return -1;
    /* P for z, shared by all experts, lives at the start of m->P */
    m->P[0] = 0.0;
    for (int j = 0; j < Lt; j++) m->P[j + 1] = m->P[j] + (double)m->lat[j];
    moe_ctx c = { m, l, m->lat, K, m->eo, 0 };
    nemo_pool_for(&m->pool, K, moe_job, &c);
    if (c.fail) return -1;
    for (int j = 0; j < Lt; j++) m->acc[j] = 0.0f;
    for (int k = 0; k < K; k++)
        for (int j = 0; j < Lt; j++) m->acc[j] += m->tw[k] * m->eo[(size_t)k * Lt + j];
    if (lin_par(m, l->fc2, m->acc, y_out, H, Lt) != 0) return -1;
    /* shared expert on the full-width input */
    if (lin_par(m, l->sh_up, x_in, m->sm, g->moe_shared, H) != 0) return -1;
    for (int j = 0; j < g->moe_shared; j++) m->sm[j] = sn_relu2(m->sm[j]);
    if (lin_par(m, l->sh_dn, m->sm, m->sh, H, g->moe_shared) != 0) return -1;
    for (int j = 0; j < H; j++) y_out[j] += m->sh[j];
    return 0;
}

/* ------------------------------------------------------------- open */

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
    nemo_pool_shutdown(&m->pool);
    if (m->L) for (int i = 0; i < m->g.n_layers; i++) {
        layer_t *l = &m->L[i];
        free(l->norm_w); free(l->conv_w); free(l->conv_b); free(l->a_log); free(l->dvec);
        free(l->dt_bias); free(l->gnorm_w); free(l->gate_w); free(l->gate_bias);
        free(l->eup); free(l->edn);
    }
    if (m->mam) for (int i = 0; i < m->g.n_layers; i++) if (m->mam[i].ssm_state) nemo_mamba_state_free(&m->mam[i]);
    if (m->att) for (int i = 0; i < m->g.n_layers; i++) if (m->att[i].k) nemo_attn_cache_free(&m->att[i]);
    if (m->lanes) for (int i = 0; i < m->nthreads; i++) { sn_rd_close(&m->lanes[i].rd); free(m->lanes[i].book); free(m->lanes[i].mid); free(m->lanes[i].pmid); }
    sn_rd_close(&m->main_rd);
    free(m->lanes); free(m->L); free(m->mam); free(m->att); free(m->normf_w);
    free(m->x); free(m->h); free(m->y); free(m->proj); free(m->inner); free(m->qb); free(m->kb); free(m->vb);
    free(m->lat); free(m->eo); free(m->acc); free(m->sm); free(m->sh); free(m->gsc); free(m->pf); free(m->P);
    free(m->book_shared); free(m->top); free(m->tw);
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
    m->lanes = (lane_t *)calloc((size_t)m->nthreads, sizeof(lane_t));
    m->L = (layer_t *)calloc((size_t)g->n_layers, sizeof(layer_t));
    m->mam = (nemo_mamba_state *)calloc((size_t)g->n_layers, sizeof(nemo_mamba_state));
    m->att = (nemo_attn_cache *)calloc((size_t)g->n_layers, sizeof(nemo_attn_cache));
    if (!m->lanes || !m->L || !m->mam || !m->att) { sn_model_close(m); return NULL; }
    for (int i = 0; i < m->nthreads; i++) {
        lane_t *ln = &m->lanes[i];
        if (sn_rd_open(&m->pack, &ln->rd) != 0) { fprintf(stderr, "sn_model: cannot open lane handles\n"); sn_model_close(m); return NULL; }
        ln->book = (float *)xalloc(m->pack.book_floats * 4);
        ln->mid = (float *)xalloc((size_t)g->moe_inter * 4 + 64);
        ln->pmid = (double *)xalloc(((size_t)g->moe_inter + 1) * 8);
        if (!ln->book || !ln->mid || !ln->pmid) { sn_model_close(m); return NULL; }
    }
    int maxcols = H; if (g->m_inner > maxcols) maxcols = g->m_inner; if (g->moe_shared > maxcols) maxcols = g->moe_shared;
    if (g->moe_inter > maxcols) maxcols = g->moe_inter;
    if (g->a_heads * g->a_hd > maxcols) maxcols = g->a_heads * g->a_hd;
    m->x = (float *)xalloc((size_t)H * 4); m->h = (float *)xalloc((size_t)H * 4); m->y = (float *)xalloc((size_t)H * 4);
    m->proj = (float *)xalloc((size_t)g->m_in_proj * 4); m->inner = (float *)xalloc((size_t)(g->m_inner > g->a_heads * g->a_hd ? g->m_inner : g->a_heads * g->a_hd) * 4);
    m->qb = (float *)xalloc((size_t)g->a_heads * g->a_hd * 4); m->kb = (float *)xalloc((size_t)g->a_kv * g->a_hd * 4); m->vb = (float *)xalloc((size_t)g->a_kv * g->a_hd * 4);
    m->lat = (float *)xalloc((size_t)g->moe_latent * 4); m->eo = (float *)xalloc((size_t)g->moe_topk * g->moe_latent * 4);
    m->acc = (float *)xalloc((size_t)g->moe_latent * 4); m->sm = (float *)xalloc((size_t)g->moe_shared * 4); m->sh = (float *)xalloc((size_t)H * 4);
    m->gsc = (float *)xalloc((size_t)E * 4); m->top = (int *)xalloc((size_t)g->moe_topk * sizeof(int)); m->tw = (float *)xalloc((size_t)g->moe_topk * 4);
    m->P = (double *)xalloc(((size_t)maxcols + 1) * 8); m->book_shared = (float *)xalloc(m->pack.book_floats * 4);
    if (!m->x || !m->h || !m->y || !m->proj || !m->inner || !m->qb || !m->kb || !m->vb || !m->lat || !m->eo || !m->acc
        || !m->sm || !m->sh || !m->gsc || !m->top || !m->tw || !m->P || !m->book_shared) { sn_model_close(m); return NULL; }

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
            l->k = need(m, kd * H, 0, "language_model.backbone.layers.%d.mixer.k_proj.weight%.0d", i, 0);
            l->v = need(m, kd * H, 0, "language_model.backbone.layers.%d.mixer.v_proj.weight%.0d", i, 0);
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
    if (bad) { sn_model_close(m); return NULL; }
    m->state_valid = 1;
    return m;
}

const sn_geom *sn_model_geom(const sn_model *m) { return &m->g; }
int sn_model_threads(const sn_model *m) { return m->nthreads; }
int sn_model_pos(const sn_model *m) { return m->n_pos; }
void sn_model_set_hooks(sn_model *m, sn_layer_cb a, sn_moe_cb b, void *u) { m->on_layer = a; m->on_moe = b; m->hook_user = u; }

void sn_model_reset(sn_model *m)
{
    for (int i = 0; i < m->g.n_layers; i++) {
        if (m->g.types[i] == 'M') {
            nemo_mamba_state *s = &m->mam[i];
            memset(s->ssm_state, 0, (size_t)s->heads * s->head_dim * s->state_size * sizeof(float));
            memset(s->conv_win, 0, (size_t)s->conv_dim * 4 * sizeof(float));
        } else if (m->g.types[i] == 'A') m->att[i].len = 0;
    }
    m->n_pos = 0; m->state_valid = 1;
}

/* ------------------------------------------------------------- step */

int sn_model_step(sn_model *m, int token, float *logits)
{
    const sn_geom *g = &m->g;
    int H = g->hidden;
    if (token < 0 || token >= g->vocab) return -1;
    if (!m->state_valid) return -2;
    if (m->n_pos >= m->max_pos) return -3;
    if (sn_expand(&m->pack, &m->main_rd, m->emb, (uint64_t)token * H, (uint64_t)H, m->x) != 0) return -1;
    for (int i = 0; i < g->n_layers; i++) {
        const layer_t *l = &m->L[i];
        char type = g->types[i];
        if (m->on_layer) m->on_layer(m->hook_user, i, type);
        nemo_rmsnorm(m->x, l->norm_w, m->h, H, g->eps);
        int rc = 0;
        if (type == 'M') {
            rc = lin_par(m, l->in_proj, m->h, m->proj, g->m_in_proj, H);
            if (!rc) rc = nemo_mamba_decode(&m->mam[i], m->proj, 0, l->conv_w, l->conv_b, l->a_log, l->dvec, l->dt_bias, l->gnorm_w, g->m_groups, g->eps, m->inner);
            if (!rc) rc = lin_par(m, l->out_proj, m->inner, m->y, H, g->m_inner);
        } else if (type == 'A') {
            int qd = g->a_heads * g->a_hd, kd = g->a_kv * g->a_hd;
            rc = lin_par(m, l->q, m->h, m->qb, qd, H);
            if (!rc) rc = lin_par(m, l->k, m->h, m->kb, kd, H);
            if (!rc) rc = lin_par(m, l->v, m->h, m->vb, kd, H);
            if (!rc) rc = nemo_attn_push(&m->att[i], m->kb, m->vb);
            if (!rc) rc = nemo_attn_decode(&m->att[i], m->qb, m->inner);
            if (!rc) rc = lin_par(m, l->o, m->inner, m->y, H, qd);
        } else {
            rc = moe_block(m, i, m->h, m->y);
        }
        if (rc) { m->state_valid = 0; return -1; }
        for (int j = 0; j < H; j++) m->x[j] += m->y[j];
    }
    m->n_pos++;
    if (logits) {
        nemo_rmsnorm(m->x, m->normf_w, m->h, H, g->eps);
        if (lin_par(m, m->head, m->h, logits, g->vocab, H) != 0) { m->state_valid = 0; return -1; }
    }
    return 0;
}
