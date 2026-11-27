/* test_sn_model.c - the assembled SuperNemo stack on a tiny pack.
 *
 *   test_sn_model PREFIX SHAPES.tsv CONFIG.json
 *
 * 1. Runs a token sequence through sn_model at 1, 3 and 12 lanes: logits must be bit-identical.
 * 2. Re-does the same sequence with an independent reference: every projection is expanded to a
 *    dense float matrix with sn_expand and multiplied naively in double, experts go through
 *    sn_moe_forward (itself checked against a double reference). The fused kernels and the
 *    pool must agree with it to a small relative error.
 * 3. State reset: replaying the sequence after sn_model_reset reproduces the logits exactly.
 * Exit 0 on success. Pure C99. */
#include "../src/sn_model.h"
#include "../src/sn_moe.h"
#include "../src/sn_pack.h"
#include "../../littlenemo/src/attn.h"
#include "../../littlenemo/src/mamba.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, m) do { checks++; if (!(c)) { fails++; printf("FAIL: %s\n", m); } } while (0)

static sn_pack P_; static sn_rd R_; static sn_geom G_;

static float *dense(const char *fmt, int a, int b, uint64_t *cnt)
{
    char name[300]; snprintf(name, sizeof(name), fmt, a, b);
    const sn_tensor *t = sn_pack_find(&P_, name);
    if (!t) { printf("reference: missing %s\n", name); exit(2); }
    float *o = (float *)malloc(t->count * 4);
    if (sn_expand(&P_, &R_, t, 0, t->count, o) != 0) { printf("reference: cannot expand %s\n", name); exit(2); }
    if (cnt) *cnt = t->count;
    return o;
}
static void mv(const float *W, int rows, int cols, const float *x, float *y)
{
    for (int r = 0; r < rows; r++) { double a = 0; for (int c = 0; c < cols; c++) a += (double)W[(size_t)r * cols + c] * x[c]; y[r] = (float)a; }
}

typedef struct { const float *up, *dn; } exp_t;
typedef struct { float **up, **dn; } exps_t;
static int fetch_ref(void *u, int e, const float **up, const float **down)
{ exps_t *x = (exps_t *)u; *up = x->up[e]; *down = x->dn[e]; return 0; }

/* reference stack */
typedef struct {
    nemo_mamba_state mam[16]; nemo_attn_cache att[16];
    float *norm[16], *conv_w[16], *conv_b[16], *a[16], *d[16], *dtb[16], *gn[16];
    float *in_proj[16], *out_proj[16], *q[16], *k[16], *v[16], *o[16];
    float *gate[16], *gbias[16], *fc1[16], *fc2[16], *su[16], *sd[16];
    exps_t ex[16];
    float *emb, *head, *normf;
} ref_t;

static float *dense_f(const char *fmt, int a) { return dense(fmt, a, 0, NULL); }

static void ref_open(ref_t *r, int max_pos)
{
    const sn_geom *g = &G_; int E = g->moe_experts;
    r->emb = dense("language_model.backbone.embeddings.weight%.0d%.0d", 0, 0, NULL);
    r->head = dense("language_model.lm_head.weight%.0d%.0d", 0, 0, NULL);
    r->normf = dense("language_model.backbone.norm_f.weight%.0d%.0d", 0, 0, NULL);
    for (int i = 0; i < g->n_layers; i++) {
        r->norm[i] = dense_f("language_model.backbone.layers.%d.norm.weight%.0d", i);
        if (g->types[i] == 'M') {
            r->in_proj[i] = dense_f("language_model.backbone.layers.%d.mixer.in_proj.weight%.0d", i);
            r->out_proj[i] = dense_f("language_model.backbone.layers.%d.mixer.out_proj.weight%.0d", i);
            r->conv_w[i] = dense_f("language_model.backbone.layers.%d.mixer.conv1d.weight%.0d", i);
            r->conv_b[i] = dense_f("language_model.backbone.layers.%d.mixer.conv1d.bias%.0d", i);
            r->a[i] = dense_f("language_model.backbone.layers.%d.mixer.A_log%.0d", i);
            r->d[i] = dense_f("language_model.backbone.layers.%d.mixer.D%.0d", i);
            r->dtb[i] = dense_f("language_model.backbone.layers.%d.mixer.dt_bias%.0d", i);
            r->gn[i] = dense_f("language_model.backbone.layers.%d.mixer.norm.weight%.0d", i);
            nemo_mamba_state_init(&r->mam[i], g->m_heads, g->m_head_dim, g->m_state, g->m_groups);
        } else if (g->types[i] == 'A') {
            r->q[i] = dense_f("language_model.backbone.layers.%d.mixer.q_proj.weight%.0d", i);
            r->k[i] = dense_f("language_model.backbone.layers.%d.mixer.k_proj.weight%.0d", i);
            r->v[i] = dense_f("language_model.backbone.layers.%d.mixer.v_proj.weight%.0d", i);
            r->o[i] = dense_f("language_model.backbone.layers.%d.mixer.o_proj.weight%.0d", i);
            nemo_attn_cache_init(&r->att[i], g->a_heads, g->a_kv, g->a_hd, max_pos);
        } else {
            r->gate[i] = dense_f("language_model.backbone.layers.%d.mixer.gate.weight%.0d", i);
            r->gbias[i] = dense_f("language_model.backbone.layers.%d.mixer.gate.e_score_correction_bias%.0d", i);
            r->fc1[i] = dense_f("language_model.backbone.layers.%d.mixer.fc1_latent_proj.weight%.0d", i);
            r->fc2[i] = dense_f("language_model.backbone.layers.%d.mixer.fc2_latent_proj.weight%.0d", i);
            r->su[i] = dense_f("language_model.backbone.layers.%d.mixer.shared_experts.up_proj.weight%.0d", i);
            r->sd[i] = dense_f("language_model.backbone.layers.%d.mixer.shared_experts.down_proj.weight%.0d", i);
            r->ex[i].up = (float **)calloc((size_t)E, sizeof(float *)); r->ex[i].dn = (float **)calloc((size_t)E, sizeof(float *));
            for (int e = 0; e < E; e++) {
                r->ex[i].up[e] = dense("language_model.backbone.layers.%d.mixer.experts.%d.up_proj.weight", i, e, NULL);
                r->ex[i].dn[e] = dense("language_model.backbone.layers.%d.mixer.experts.%d.down_proj.weight", i, e, NULL);
            }
        }
    }
}

static void ref_step(ref_t *r, int token, float *logits)
{
    const sn_geom *g = &G_; int H = g->hidden;
    float *x = (float *)malloc((size_t)H * 4), *h = (float *)malloc((size_t)H * 4), *y = (float *)malloc((size_t)H * 4);
    float *proj = (float *)malloc((size_t)g->m_in_proj * 4), *inner = (float *)malloc((size_t)g->m_inner * 4);
    memcpy(x, r->emb + (size_t)token * H, (size_t)H * 4);
    for (int i = 0; i < g->n_layers; i++) {
        nemo_rmsnorm(x, r->norm[i], h, H, g->eps);
        if (g->types[i] == 'M') {
            mv(r->in_proj[i], g->m_in_proj, H, h, proj);
            nemo_mamba_decode(&r->mam[i], proj, 0, r->conv_w[i], r->conv_b[i], r->a[i], r->d[i], r->dtb[i], r->gn[i], g->m_groups, g->eps, inner);
            mv(r->out_proj[i], H, g->m_inner, inner, y);
        } else if (g->types[i] == 'A') {
            int qd = g->a_heads * g->a_hd, kd = g->a_kv * g->a_hd;
            float *q = (float *)malloc((size_t)qd * 4), *k = (float *)malloc((size_t)kd * 4), *v = (float *)malloc((size_t)kd * 4), *ao = (float *)malloc((size_t)qd * 4);
            mv(r->q[i], qd, H, h, q); mv(r->k[i], kd, H, h, k); mv(r->v[i], kd, H, h, v);
            nemo_attn_push(&r->att[i], k, v); nemo_attn_decode(&r->att[i], q, ao);
            mv(r->o[i], H, qd, ao, y);
            free(q); free(k); free(v); free(ao);
        } else {
            sn_moe_cfg c = { H, g->moe_latent, g->moe_inter, g->moe_shared, g->moe_experts, g->moe_topk, g->moe_scale, g->moe_norm_topk };
            sn_moe_forward(&c, r->gate[i], r->gbias[i], r->fc1[i], r->fc2[i], r->su[i], r->sd[i], fetch_ref, &r->ex[i], h, y, NULL);
        }
        for (int j = 0; j < H; j++) x[j] += y[j];
    }
    nemo_rmsnorm(x, r->normf, h, H, g->eps);
    mv(r->head, g->vocab, H, h, logits);
    free(x); free(h); free(y); free(proj); free(inner);
}

static const int SEQ[] = { 5, 17, 3, 200, 42, 9, 311, 77 };
#define NSEQ ((int)(sizeof(SEQ) / sizeof(SEQ[0])))

static void run_model(const char *pre, const char *sh, const char *cfg, int lanes, float *out /* NSEQ*vocab */)
{
    sn_model *m = sn_model_open(pre, sh, cfg, 64, lanes);
    if (!m) { printf("cannot open model\n"); exit(2); }
    int V = sn_model_geom(m)->vocab;
    for (int t = 0; t < NSEQ; t++) CHECK(sn_model_step(m, SEQ[t], out + (size_t)t * V) == 0, "step ok");
    if (lanes == 3) {   /* reset determinism, on the 3-lane model */
        sn_model_reset(m);
        float *again = (float *)malloc((size_t)V * 4);
        int same = 1;
        for (int t = 0; t < NSEQ; t++) { sn_model_step(m, SEQ[t], again); if (memcmp(again, out + (size_t)t * V, (size_t)V * 4) != 0) same = 0; }
        CHECK(same, "replay after reset is bit-identical");
        free(again);
    }
    CHECK(sn_model_pos(m) >= NSEQ, "positions advance");
    sn_model_close(m);
}

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: test_sn_model PREFIX SHAPES CONFIG\n"); return 2; }
    if (sn_geom_load(&G_, argv[3]) != 0) return 2;
    CHECK(G_.n_layers == 5 && G_.m_in_proj == 772 && G_.m_conv_dim == 512, "geometry derived from config");
    int V = G_.vocab;
    float *l1 = (float *)malloc((size_t)NSEQ * V * 4), *l3 = (float *)malloc((size_t)NSEQ * V * 4), *l12 = (float *)malloc((size_t)NSEQ * V * 4);
    run_model(argv[1], argv[2], argv[3], 1, l1);
    run_model(argv[1], argv[2], argv[3], 3, l3);
    run_model(argv[1], argv[2], argv[3], 12, l12);
    CHECK(memcmp(l1, l3, (size_t)NSEQ * V * 4) == 0, "1 lane == 3 lanes, bit for bit");
    CHECK(memcmp(l1, l12, (size_t)NSEQ * V * 4) == 0, "1 lane == 12 lanes, bit for bit");
    int finite = 1; for (size_t i = 0; i < (size_t)NSEQ * V; i++) if (!isfinite(l1[i])) finite = 0;
    CHECK(finite, "logits finite");
    /* token sensitivity: different tokens give different logits */
    CHECK(memcmp(l1, l1 + V, (size_t)V * 4) != 0, "logits depend on the token");

    if (sn_pack_open(&P_, argv[1], argv[2]) != 0 || sn_rd_open(&P_, &R_) != 0) return 2;
    ref_t *r = (ref_t *)calloc(1, sizeof(*r));
    ref_open(r, 64);
    float *lr = (float *)malloc((size_t)V * 4);
    double worst = 0, scale = 0;
    int arg_ok = 1;
    for (int t = 0; t < NSEQ; t++) {
        ref_step(r, SEQ[t], lr);
        int am_r = 0, am_m = 0;
        for (int i = 0; i < V; i++) { double d = fabs((double)lr[i] - l1[(size_t)t * V + i]); if (d > worst) worst = d; if (fabs(lr[i]) > scale) scale = fabs(lr[i]); if (lr[i] > lr[am_r]) am_r = i; if (l1[(size_t)t * V + i] > l1[(size_t)t * V + am_m]) am_m = i; }
        if (am_r != am_m) arg_ok = 0;
    }
    printf("max |logit difference| vs dense reference: %.3e (max logit %.3f)\n", worst, scale);
    CHECK(worst <= 2e-3 * (scale + 1.0), "fused stack matches the dense reference");
    CHECK(arg_ok, "argmax agrees at every position");
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
