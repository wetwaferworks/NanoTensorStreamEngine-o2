/* forward.c - see forward.h and littlenemo/BLOCK-MATH-SPEC.md. */
#include "forward.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn.h"

static const char PATTERN[NEMO_LAYERS + 1] =
    "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-";

char nemo_layer_type(int i) {
    if (i < 0 || i >= NEMO_LAYERS) return '?';
    return PATTERN[i];
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

int nemo_model_init(nemo_model *m, const nemo_checkpoint *cp,
                    int layer_limit, int max_pos) {
    memset(m, 0, sizeof *m);
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
    return 0;
}

void nemo_model_free(nemo_model *m) {
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
}

/* One Mamba-2 block's mixer (input h, output y), weights fetched live. */
static int run_mamba(nemo_model *m, int layer) {
    char name[96];
    layer_name(name, sizeof name, layer, "mixer.in_proj.weight");
    uint16_t *in_proj = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.out_proj.weight");
    uint16_t *out_proj = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.conv1d.weight");
    float *conv_w = get_f32(m, name);
    layer_name(name, sizeof name, layer, "mixer.conv1d.bias");
    float *conv_b = get_f32(m, name);
    layer_name(name, sizeof name, layer, "mixer.A_log");
    float *a_log = get_f32(m, name);
    layer_name(name, sizeof name, layer, "mixer.D");
    float *d = get_f32(m, name);
    layer_name(name, sizeof name, layer, "mixer.dt_bias");
    float *dt_bias = get_f32(m, name);
    layer_name(name, sizeof name, layer, "mixer.norm.weight");
    float *norm_w = get_f32(m, name);
    int ok = in_proj && out_proj && conv_w && conv_b && a_log && d &&
             dt_bias && norm_w;
    if (ok) {
        nemo_linear_bf16(in_proj, m->h, m->proj, 22656, NEMO_HIDDEN);
        if (nemo_mamba_decode(&m->mamba[layer], m->proj, 0, conv_w, conv_b,
                              a_log, d, dt_bias, norm_w, 8, EPS,
                              m->inner) != 0)
            ok = 0;
        else
            nemo_linear_bf16(out_proj, m->inner, m->y, NEMO_HIDDEN, 10240);
    }
    free(in_proj);
    free(out_proj);
    free(conv_w);
    free(conv_b);
    free(a_log);
    free(d);
    free(dt_bias);
    free(norm_w);
    return ok ? 0 : -1;
}

static int run_mlp(nemo_model *m, int layer) {
    char name[96];
    layer_name(name, sizeof name, layer, "mixer.up_proj.weight");
    uint16_t *up = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.down_proj.weight");
    uint16_t *down = get_bf16(m, name);
    int ok = up && down;
    if (ok) {
        nemo_linear_bf16(up, m->h, m->mid, 15680, NEMO_HIDDEN);
        for (int i = 0; i < 15680; i++) m->mid[i] = nemo_relu2(m->mid[i]);
        nemo_linear_bf16(down, m->mid, m->y, NEMO_HIDDEN, 15680);
    }
    free(up);
    free(down);
    return ok ? 0 : -1;
}

static int run_attn(nemo_model *m, int layer) {
    char name[96];
    layer_name(name, sizeof name, layer, "mixer.q_proj.weight");
    uint16_t *wq = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.k_proj.weight");
    uint16_t *wk = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.v_proj.weight");
    uint16_t *wv = get_bf16(m, name);
    layer_name(name, sizeof name, layer, "mixer.o_proj.weight");
    uint16_t *wo = get_bf16(m, name);
    int ok = wq && wk && wv && wo;
    if (ok) {
        float kbuf[1024], vbuf[1024];
        nemo_linear_bf16(wq, m->h, m->qkv, 5120, NEMO_HIDDEN);
        nemo_linear_bf16(wk, m->h, kbuf, 1024, NEMO_HIDDEN);
        nemo_linear_bf16(wv, m->h, vbuf, 1024, NEMO_HIDDEN);
        if (nemo_attn_push(&m->attn[layer], kbuf, vbuf) != 0) ok = 0;
        if (ok) {
            if (nemo_attn_decode(&m->attn[layer], m->qkv, m->inner) != 0)
                ok = 0;
            else
                nemo_linear_bf16(wo, m->inner, m->y, NEMO_HIDDEN, 5120);
        }
    }
    free(wq);
    free(wk);
    free(wv);
    free(wo);
    return ok ? 0 : -1;
}

int nemo_model_step_hidden(nemo_model *m, int token, float *hidden_out) {
    if (token < 0 || token >= NEMO_VOCAB) return -1;
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

    for (int layer = 0; layer < m->layer_limit; layer++) {
        char name[96];
        layer_name(name, sizeof name, layer, "norm.weight");
        float *nw = get_f32(m, name);
        if (!nw) return -1;
        nemo_rmsnorm(m->x, nw, m->h, NEMO_HIDDEN, EPS);
        free(nw);
        char t = nemo_layer_type(layer);
        if (m->on_layer) m->on_layer(m->on_layer_user, layer, t);
        int rc = t == 'M'   ? run_mamba(m, layer)
                 : t == '*' ? run_attn(m, layer)
                            : run_mlp(m, layer);
        if (rc != 0) return -1;
        for (int i = 0; i < NEMO_HIDDEN; i++) m->x[i] += m->y[i];
    }
    memcpy(hidden_out, m->x, NEMO_HIDDEN * sizeof(float));
    return 0;
}

int nemo_model_step(nemo_model *m, int token, float *logits_out) {
    float hidden[NEMO_HIDDEN];
    if (nemo_model_step_hidden(m, token, hidden) != 0) return -1;
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
    nemo_linear_bf16(lm, normed, logits_out, NEMO_VOCAB, NEMO_HIDDEN);
    free(lm);
    return 0;
}
