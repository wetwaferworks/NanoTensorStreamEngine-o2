/* forward.c - see forward.h and littlenemo/BLOCK-MATH-SPEC.md. */
#define _FILE_OFFSET_BITS 64 /* 64-bit off_t on Windows too (Archie P0-2) */
#include "forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

static const char LAYER_PATTERN[NEMO_LAYERS + 1] =
    "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-";

char nemo_layer_type(int i) {
    if (i < 0 || i >= NEMO_LAYERS) return '?';
    return LAYER_PATTERN[i];
}

#define EPS 1e-5f

/* Fetch helpers: every failure is a hard -1, never a silent zero. */
static uint16_t *get_bf16(const nemo_model *m, const char *name) {
    const nemo_tensor_info *t = nemo_find(m->cp, name);
    if (!t) return NULL;
    return nemo_load_bf16(m->cp, t);
}

static float *get_f32(const nemo_model *m, const char *name) {
    const nemo_tensor_info *t = nemo_find(m->cp, name);
    if (!t) return NULL;
    return nemo_load_f32(m->cp, t);
}

static void layer_name(char *buf, size_t n, int layer, const char *leaf) {
    snprintf(buf, n, "backbone.layers.%d.%s", layer, leaf);
}

/* --- Prepared layers: the dual path (JD, 2026-11-24) -----------------
 * A layer's tensors are fetched into a workspace ahead of use: while
 * layer n computes, the pool's prep thread fetches layer n+1, so the
 * compute path never waits on the disk. Preparation only moves the
 * fetch earlier; the math is untouched. With one lane the prep runs
 * inline and the flow is exactly the old serial one. */
typedef struct {
    float *norm_w;
    uint16_t *in_proj, *out_proj;  /* Mamba */
    float *conv_w, *conv_b, *a_log, *d, *dt_bias, *gnorm_w;
    uint16_t *up, *down;           /* MLP */
    uint16_t *wq, *wk, *wv, *wo;   /* attention */
    int rc;                        /* 0 once fully prepared */
} layer_ws;

static void ws_free(layer_ws *ws) {
    free(ws->norm_w);
    free(ws->in_proj);
    free(ws->out_proj);
    free(ws->conv_w);
    free(ws->conv_b);
    free(ws->a_log);
    free(ws->d);
    free(ws->dt_bias);
    free(ws->gnorm_w);
    free(ws->up);
    free(ws->down);
    free(ws->wq);
    free(ws->wk);
    free(ws->wv);
    free(ws->wo);
    memset(ws, 0, sizeof *ws);
    ws->rc = -1;
}

static int prepare_layer(const nemo_model *m, int layer, layer_ws *ws) {
    char name[96];
    ws_free(ws);
    layer_name(name, sizeof name, layer, "norm.weight");
    ws->norm_w = get_f32(m, name);
    if (!ws->norm_w) goto fail;
    char t = nemo_layer_type(layer);
    if (t == 'M') {
        layer_name(name, sizeof name, layer, "mixer.in_proj.weight");
        ws->in_proj = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.out_proj.weight");
        ws->out_proj = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.conv1d.weight");
        ws->conv_w = get_f32(m, name);
        layer_name(name, sizeof name, layer, "mixer.conv1d.bias");
        ws->conv_b = get_f32(m, name);
        layer_name(name, sizeof name, layer, "mixer.A_log");
        ws->a_log = get_f32(m, name);
        layer_name(name, sizeof name, layer, "mixer.D");
        ws->d = get_f32(m, name);
        layer_name(name, sizeof name, layer, "mixer.dt_bias");
        ws->dt_bias = get_f32(m, name);
        layer_name(name, sizeof name, layer, "mixer.norm.weight");
        ws->gnorm_w = get_f32(m, name);
        if (!ws->in_proj || !ws->out_proj || !ws->conv_w || !ws->conv_b ||
            !ws->a_log || !ws->d || !ws->dt_bias || !ws->gnorm_w)
            goto fail;
    } else if (t == '*') {
        layer_name(name, sizeof name, layer, "mixer.q_proj.weight");
        ws->wq = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.k_proj.weight");
        ws->wk = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.v_proj.weight");
        ws->wv = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.o_proj.weight");
        ws->wo = get_bf16(m, name);
        if (!ws->wq || !ws->wk || !ws->wv || !ws->wo) goto fail;
    } else {
        layer_name(name, sizeof name, layer, "mixer.up_proj.weight");
        ws->up = get_bf16(m, name);
        layer_name(name, sizeof name, layer, "mixer.down_proj.weight");
        ws->down = get_bf16(m, name);
        if (!ws->up || !ws->down) goto fail;
    }
    ws->rc = 0;
    return 0;
fail:
    ws_free(ws);
    return -1;
}

/* The prep thread's job ticket: one outstanding at a time, refilled
 * only after the previous job has been collected. */
typedef struct {
    nemo_model *m;
    int layer;
    layer_ws *ws;
} prep_ticket;

static void prepare_job(void *ctx) {
    prep_ticket *t = (prep_ticket *)ctx;
    t->ws->rc = prepare_layer(t->m, t->layer, t->ws);
}

/* Row-split projection over the pool. Small projections stay serial:
 * below the threshold the lane sync costs more than the math. */
typedef struct {
    const uint16_t *W;
    const float *x;
    float *y;
    int in;
} lin_ctx;

static void lin_chunk(void *ctx, int begin, int end) {
    lin_ctx *L = (lin_ctx *)ctx;
    nemo_linear_bf16_rows_auto(L->W, L->x, L->y, begin, end, L->in);
}

static void linear_par(nemo_model *m, const uint16_t *W, const float *x,
                       float *y, int out, int in) {
    if (m->nthreads <= 1 || out < 4096) {
        nemo_linear_bf16(W, x, y, out, in);
        return;
    }
    lin_ctx L;
    L.W = W;
    L.x = x;
    L.y = y;
    L.in = in;
    nemo_pool_for(&m->pool, out, lin_chunk, &L);
}

/* --- Whole-geometry validation and step preflight -------------------
 * One walk serves both: nemo_validate_model runs it before inference
 * begins, and every step runs it before any state moves, so a missing
 * or malformed tensor is a preparation failure, never a half
 * committed token (Gizmo review, 2026-11-24). Metadata only: the
 * walk reads the checkpoint index, never tensor data. */
#define DT_BF16 1  /* fetched as bf16: must be bf16 */
#define DT_FLOAT 2 /* fetched widened: bf16 or f32 both load */

static int check_tensor(const nemo_checkpoint *cp, const char *name,
                        const uint64_t *shape, int ndim, int dt,
                        char *bad, size_t badn, int *code) {
    const nemo_tensor_info *t = nemo_find(cp, name);
    if (!t) {
        if (code) *code = 1;
        if (bad) snprintf(bad, badn, "%s", name);
        return -1;
    }
    if (t->ndim != ndim) {
        if (code) *code = 2;
        if (bad) snprintf(bad, badn, "%s", name);
        return -1;
    }
    for (int i = 0; i < ndim; i++) {
        if (t->shape[i] != shape[i]) {
            if (code) *code = 2;
            if (bad) snprintf(bad, badn, "%s", name);
            return -1;
        }
    }
    int dtype_ok = (dt == DT_BF16) ? t->is_bf16 : (t->is_bf16 || t->is_f32);
    if (!dtype_ok) {
        if (code) *code = 3;
        if (bad) snprintf(bad, badn, "%s", name);
        return -1;
    }
    return 0;
}

static int validate_set(const nemo_checkpoint *cp, int layer_limit,
                        int need_head, char *bad, size_t badn, int *code) {
    static const uint64_t s_emb[2] = {NEMO_VOCAB, NEMO_HIDDEN};
    static const uint64_t s_vec[1] = {NEMO_HIDDEN};
    static const uint64_t s_inproj[2] = {22656, NEMO_HIDDEN};
    static const uint64_t s_outproj[2] = {NEMO_HIDDEN, 10240};
    static const uint64_t s_convw[3] = {12288, 1, 4};
    static const uint64_t s_convdim[1] = {12288};
    static const uint64_t s_heads[1] = {128};
    static const uint64_t s_gnorm[1] = {10240};
    static const uint64_t s_up[2] = {15680, NEMO_HIDDEN};
    static const uint64_t s_down[2] = {NEMO_HIDDEN, 15680};
    static const uint64_t s_q[2] = {5120, NEMO_HIDDEN};
    static const uint64_t s_kv[2] = {1024, NEMO_HIDDEN};
    static const uint64_t s_o[2] = {NEMO_HIDDEN, 5120};
    static const uint64_t s_head[2] = {NEMO_VOCAB, NEMO_HIDDEN};
    char name[96];
    if (check_tensor(cp, "backbone.embeddings.weight", s_emb, 2, DT_BF16,
                     bad, badn, code) != 0)
        return -1;
    for (int layer = 0; layer < layer_limit; layer++) {
        layer_name(name, sizeof name, layer, "norm.weight");
        if (check_tensor(cp, name, s_vec, 1, DT_FLOAT, bad, badn, code) !=
            0)
            return -1;
        char t = nemo_layer_type(layer);
        if (t == 'M') {
            layer_name(name, sizeof name, layer, "mixer.in_proj.weight");
            if (check_tensor(cp, name, s_inproj, 2, DT_BF16, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.conv1d.weight");
            if (check_tensor(cp, name, s_convw, 3, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.conv1d.bias");
            if (check_tensor(cp, name, s_convdim, 1, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.A_log");
            if (check_tensor(cp, name, s_heads, 1, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.D");
            if (check_tensor(cp, name, s_heads, 1, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.dt_bias");
            if (check_tensor(cp, name, s_heads, 1, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.norm.weight");
            if (check_tensor(cp, name, s_gnorm, 1, DT_FLOAT, bad, badn,
                             code) != 0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.out_proj.weight");
            if (check_tensor(cp, name, s_outproj, 2, DT_BF16, bad, badn,
                             code) != 0)
                return -1;
        } else if (t == '*') {
            layer_name(name, sizeof name, layer, "mixer.q_proj.weight");
            if (check_tensor(cp, name, s_q, 2, DT_BF16, bad, badn, code) !=
                0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.k_proj.weight");
            if (check_tensor(cp, name, s_kv, 2, DT_BF16, bad, badn, code) !=
                0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.v_proj.weight");
            if (check_tensor(cp, name, s_kv, 2, DT_BF16, bad, badn, code) !=
                0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.o_proj.weight");
            if (check_tensor(cp, name, s_o, 2, DT_BF16, bad, badn, code) !=
                0)
                return -1;
        } else {
            layer_name(name, sizeof name, layer, "mixer.up_proj.weight");
            if (check_tensor(cp, name, s_up, 2, DT_BF16, bad, badn, code) !=
                0)
                return -1;
            layer_name(name, sizeof name, layer, "mixer.down_proj.weight");
            if (check_tensor(cp, name, s_down, 2, DT_BF16, bad, badn,
                             code) != 0)
                return -1;
        }
    }
    if (need_head) {
        if (check_tensor(cp, "backbone.norm_f.weight", s_vec, 1, DT_FLOAT,
                         bad, badn, code) != 0)
            return -1;
        if (check_tensor(cp, "lm_head.weight", s_head, 2, DT_BF16, bad,
                         badn, code) != 0)
            return -1;
    }
    return 0;
}

int nemo_validate_model(const nemo_checkpoint *cp, int layer_limit,
                        int need_head, char *bad, size_t badn, int *code) {
    if (layer_limit <= 0 || layer_limit > NEMO_LAYERS)
        layer_limit = NEMO_LAYERS;
    return validate_set(cp, layer_limit, need_head, bad, badn, code);
}

int nemo_model_init(nemo_model *m, const nemo_checkpoint *cp,
                    int layer_limit, int max_pos) {
    return nemo_model_init_ex(m, cp, layer_limit, max_pos, 1);
}

int nemo_model_init_ex(nemo_model *m, const nemo_checkpoint *cp,
                       int layer_limit, int max_pos, int nthreads) {
    memset(m, 0, sizeof *m);
    if (nemo_pool_init(&m->pool, nthreads) != 0) return -1;
    m->nthreads = nemo_pool_threads(&m->pool);
    m->cp = cp;
    m->layer_limit = layer_limit > 0 ? layer_limit : NEMO_LAYERS;
    m->max_pos = max_pos;
    m->x = (float *)malloc(NEMO_HIDDEN * sizeof(float));
    m->h = (float *)malloc(NEMO_HIDDEN * sizeof(float));
    m->y = (float *)malloc(NEMO_HIDDEN * sizeof(float));
    m->proj = (float *)malloc(22656 * sizeof(float));
    m->inner = (float *)malloc(10240 * sizeof(float));
    m->mid = (float *)malloc(15680 * sizeof(float));
    m->qkv = (float *)malloc(5120 * sizeof(float));
    if (!m->x || !m->h || !m->y || !m->proj || !m->inner || !m->mid ||
        !m->qkv) {
        nemo_model_free(m);
        return -1;
    }
    for (int i = 0; i < m->layer_limit; i++) {
        char t = nemo_layer_type(i);
        if (t == 'M') {
            if (nemo_mamba_state_init(&m->mamba[i], 128, 80, 128, 8) != 0) {
                nemo_model_free(m);
                return -1;
            }
        } else if (t == '*') {
            if (nemo_attn_cache_init(&m->attn[i], 40, 8, 128, max_pos) !=
                0) {
                nemo_model_free(m);
                return -1;
            }
        }
    }
    m->state_valid = 1;
    return 0;
}

void nemo_model_free(nemo_model *m) {
    /* No step is running when the model is freed, so no prep job is
     * outstanding; shut the pool down before the buffers go. */
    nemo_pool_shutdown(&m->pool);
    for (int i = 0; i < NEMO_LAYERS; i++) {
        if (m->mamba[i].ssm_state) nemo_mamba_state_free(&m->mamba[i]);
        if (m->attn[i].k) nemo_attn_cache_free(&m->attn[i]);
    }
    free(m->x);
    free(m->h);
    free(m->y);
    free(m->proj);
    free(m->inner);
    free(m->mid);
    free(m->qkv);
    memset(m, 0, sizeof *m);
}

void nemo_model_reset(nemo_model *m) {
    for (int i = 0; i < m->layer_limit; i++) {
        char t = nemo_layer_type(i);
        if (t == 'M') {
            nemo_mamba_state *s = &m->mamba[i];
            memset(s->ssm_state, 0,
                   (size_t)s->heads * (size_t)s->head_dim *
                       (size_t)s->state_size * sizeof(float));
            memset(s->conv_win, 0,
                   (size_t)s->conv_dim * 4 * sizeof(float));
        } else if (t == '*') {
            m->attn[i].len = 0;
        }
    }
    m->n_pos = 0;
    m->state_valid = 1;
}

/* The mixers consume a prepared layer workspace; fetching happened
 * in prepare_layer, on the prep thread where the pool has one. */
static int run_mamba(nemo_model *m, int layer, const layer_ws *ws) {
    linear_par(m, ws->in_proj, m->h, m->proj, 22656, NEMO_HIDDEN);
    if (nemo_mamba_decode(&m->mamba[layer], m->proj, 0, ws->conv_w,
                          ws->conv_b, ws->a_log, ws->d, ws->dt_bias,
                          ws->gnorm_w, 8, EPS, m->inner) != 0)
        return -1;
    linear_par(m, ws->out_proj, m->inner, m->y, NEMO_HIDDEN, 10240);
    return 0;
}

static int run_mlp(nemo_model *m, const layer_ws *ws) {
    linear_par(m, ws->up, m->h, m->mid, 15680, NEMO_HIDDEN);
    for (int i = 0; i < 15680; i++) m->mid[i] = nemo_relu2(m->mid[i]);
    linear_par(m, ws->down, m->mid, m->y, NEMO_HIDDEN, 15680);
    return 0;
}

static int run_attn(nemo_model *m, int layer, const layer_ws *ws) {
    float kbuf[1024], vbuf[1024];
    linear_par(m, ws->wq, m->h, m->qkv, 5120, NEMO_HIDDEN);
    nemo_linear_bf16(ws->wk, m->h, kbuf, 1024, NEMO_HIDDEN);
    nemo_linear_bf16(ws->wv, m->h, vbuf, 1024, NEMO_HIDDEN);
    if (nemo_attn_push(&m->attn[layer], kbuf, vbuf) != 0) return -1;
    if (nemo_attn_decode(&m->attn[layer], m->qkv, m->inner) != 0)
        return -1;
    linear_par(m, ws->wo, m->inner, m->y, NEMO_HIDDEN, 5120);
    return 0;
}

int nemo_model_step_hidden(nemo_model *m, int token, float *hidden_out) {
    if (token < 0 || token >= NEMO_VOCAB) return -1;
    if (!m->state_valid) return -2;
    /* Preflight: every tensor this step will fetch must be present and
     * well formed before the first layer commits its state. A failure
     * here is a preparation failure; the state is untouched. */
    if (validate_set(m->cp, m->layer_limit, 0, NULL, 0, NULL) != 0)
        return -1;
    /* Context bound, also preflight (Archie P1-3): without it, a
     * full KV cache fails the step inside the first attention
     * layer, after the earlier layers have already mutated their
     * state, and the half-committed step invalidates the whole
     * conversation. Refused here, nothing has moved. */
    if (m->n_pos >= m->max_pos) return -1;
    const nemo_tensor_info *emb =
        nemo_find(m->cp, "backbone.embeddings.weight");
    if (!emb) return -1;
    /* Fetch just the embedding row for this token. */
    uint16_t *row = (uint16_t *)malloc(NEMO_HIDDEN * sizeof(uint16_t));
    if (!row) return -1;
    FILE *f = fopen(emb->shard_path, "rb");
    if (!f) {
        free(row);
        return -1;
    }
    uint64_t off = emb->data_off +
                   (uint64_t)token * NEMO_HIDDEN * sizeof(uint16_t);
    int ok = fseeko(f, (off_t)off, SEEK_SET) == 0 &&
             fread(row, sizeof(uint16_t), NEMO_HIDDEN, f) == NEMO_HIDDEN;
    fclose(f);
    if (!ok) {
        free(row);
        return -1;
    }
    for (int i = 0; i < NEMO_HIDDEN; i++)
        m->x[i] = nemo_bf16_to_f32(row[i]);
    free(row);

    /* The pipeline: layer n+1 is fetched while layer n computes.
     * Failure semantics are the pre-pipeline ones: a preparation
     * failure at any layer is a mid-step failure (the preflight has
     * already proven the tensors exist, so this can only be an I/O
     * or allocation failure), the state is invalid until reset, and
     * any outstanding prep is collected before returning. */
    layer_ws slots[2];
    memset(slots, 0, sizeof slots);
    slots[0].rc = -1;
    slots[1].rc = -1;
    prep_ticket ticket;
    ticket.m = m;
    ticket.layer = 0;
    ticket.ws = &slots[0];
    nemo_pool_prep(&m->pool, prepare_job, &ticket);
    int cur = 0;
    for (int layer = 0; layer < m->layer_limit; layer++) {
        nemo_pool_prep_wait(&m->pool);
        if (slots[cur].rc != 0) {
            m->state_valid = 0;
            ws_free(&slots[0]);
            ws_free(&slots[1]);
            return -1;
        }
        if (layer + 1 < m->layer_limit) {
            ticket.layer = layer + 1;
            ticket.ws = &slots[cur ^ 1];
            nemo_pool_prep(&m->pool, prepare_job, &ticket);
        }
        nemo_rmsnorm(m->x, slots[cur].norm_w, m->h, NEMO_HIDDEN, EPS);
        char t = nemo_layer_type(layer);
        if (m->on_layer) m->on_layer(m->on_layer_user, layer, t);
        int rc = t == 'M'   ? run_mamba(m, layer, &slots[cur])
                 : t == '*' ? run_attn(m, layer, &slots[cur])
                            : run_mlp(m, &slots[cur]);
        if (rc != 0) {
            nemo_pool_prep_wait(&m->pool);
            m->state_valid = 0;
            ws_free(&slots[0]);
            ws_free(&slots[1]);
            return -1;
        }
        for (int i = 0; i < NEMO_HIDDEN; i++) m->x[i] += m->y[i];
        ws_free(&slots[cur]);
        cur ^= 1;
    }
    memcpy(hidden_out, m->x, NEMO_HIDDEN * sizeof(float));
    m->n_pos++;
    return 0;
}

int nemo_model_step(nemo_model *m, int token, float *logits_out) {
    if (!m->state_valid) return -2;
    /* The head tensors are preflighted before the stack runs, so a
     * missing lm_head refuses the step before the token commits. */
    if (validate_set(m->cp, m->layer_limit, 1, NULL, 0, NULL) != 0)
        return -1;
    float hidden[NEMO_HIDDEN];
    int rc = nemo_model_step_hidden(m, token, hidden);
    if (rc != 0) return rc;
    float *nw = get_f32(m, "backbone.norm_f.weight");
    uint16_t *lm = get_bf16(m, "lm_head.weight");
    if (!nw || !lm) {
        /* The token committed coherently; only the readout failed.
         * The state stays valid (the engine resets the conversation
         * on a failed turn as its own policy). */
        free(nw);
        free(lm);
        return -1;
    }
    float normed[NEMO_HIDDEN];
    nemo_rmsnorm(hidden, nw, normed, NEMO_HIDDEN, EPS);
    free(nw);
    linear_par(m, lm, normed, logits_out, NEMO_VOCAB, NEMO_HIDDEN);
    free(lm);
    return 0;
}

/* ---- layer-major prefill (see forward.h) ---- */

int nemo_model_head(nemo_model *m, const float *hidden, float *logits_out) {
    float *nw = get_f32(m, "backbone.norm_f.weight");
    uint16_t *lm = get_bf16(m, "lm_head.weight");
    if (!nw || !lm) {
        free(nw);
        free(lm);
        return -1;
    }
    float normed[NEMO_HIDDEN];
    nemo_rmsnorm(hidden, nw, normed, NEMO_HIDDEN, EPS);
    free(nw);
    linear_par(m, lm, normed, logits_out, NEMO_VOCAB, NEMO_HIDDEN);
    free(lm);
    return 0;
}

int nemo_model_prefill(nemo_model *m, const int *tokens, int n,
                       float *hidden_out) {
    if (n < 1 || n > NEMO_PREFILL_CHUNK || !tokens) return -1;
    for (int i = 0; i < n; i++)
        if (tokens[i] < 0 || tokens[i] >= NEMO_VOCAB) return -1;
    if (!m->state_valid) return -2;
    if (validate_set(m->cp, m->layer_limit, 0, NULL, 0, NULL) != 0)
        return -1;
    if (m->n_pos + n > m->max_pos) return -1;
    const nemo_tensor_info *emb =
        nemo_find(m->cp, "backbone.embeddings.weight");
    if (!emb) return -1;
    float *X = (float *)malloc((size_t)n * NEMO_HIDDEN * sizeof(float));
    uint16_t *row = (uint16_t *)malloc(NEMO_HIDDEN * sizeof(uint16_t));
    if (!X || !row) {
        free(X);
        free(row);
        return -1;
    }
    FILE *f = fopen(emb->shard_path, "rb");
    if (!f) {
        free(X);
        free(row);
        return -1;
    }
    int ok = 1;
    for (int t = 0; t < n && ok; t++) {
        uint64_t off = emb->data_off +
                       (uint64_t)tokens[t] * NEMO_HIDDEN * sizeof(uint16_t);
        ok = fseeko(f, (off_t)off, SEEK_SET) == 0 &&
             fread(row, sizeof(uint16_t), NEMO_HIDDEN, f) == NEMO_HIDDEN;
        if (ok)
            for (int i = 0; i < NEMO_HIDDEN; i++)
                X[(size_t)t * NEMO_HIDDEN + i] = nemo_bf16_to_f32(row[i]);
    }
    fclose(f);
    free(row);
    if (!ok) {
        free(X);
        return -1;
    }

    layer_ws slots[2];
    memset(slots, 0, sizeof slots);
    slots[0].rc = -1;
    slots[1].rc = -1;
    prep_ticket ticket;
    ticket.m = m;
    ticket.layer = 0;
    ticket.ws = &slots[0];
    nemo_pool_prep(&m->pool, prepare_job, &ticket);
    int cur = 0;
    for (int layer = 0; layer < m->layer_limit; layer++) {
        nemo_pool_prep_wait(&m->pool);
        if (slots[cur].rc != 0) {
            m->state_valid = 0;
            ws_free(&slots[0]);
            ws_free(&slots[1]);
            free(X);
            return -1;
        }
        if (layer + 1 < m->layer_limit) {
            ticket.layer = layer + 1;
            ticket.ws = &slots[cur ^ 1];
            nemo_pool_prep(&m->pool, prepare_job, &ticket);
        }
        char t = nemo_layer_type(layer);
        if (m->on_layer) m->on_layer(m->on_layer_user, layer, t);
        for (int k = 0; k < n; k++) {
            float *xk = X + (size_t)k * NEMO_HIDDEN;
            memcpy(m->x, xk, NEMO_HIDDEN * sizeof(float));
            nemo_rmsnorm(m->x, slots[cur].norm_w, m->h, NEMO_HIDDEN, EPS);
            int rc = t == 'M'   ? run_mamba(m, layer, &slots[cur])
                     : t == '*' ? run_attn(m, layer, &slots[cur])
                                : run_mlp(m, &slots[cur]);
            if (rc != 0) {
                nemo_pool_prep_wait(&m->pool);
                m->state_valid = 0;
                ws_free(&slots[0]);
                ws_free(&slots[1]);
                free(X);
                return -1;
            }
            for (int i = 0; i < NEMO_HIDDEN; i++) m->x[i] += m->y[i];
            memcpy(xk, m->x, NEMO_HIDDEN * sizeof(float));
        }
        ws_free(&slots[cur]);
        cur ^= 1;
    }
    memcpy(hidden_out, X + (size_t)(n - 1) * NEMO_HIDDEN,
           NEMO_HIDDEN * sizeof(float));
    m->n_pos += n;
    free(X);
    return 0;
}
