/* parakeet-c model: FastConformer encoder (dw_striding x8 subsampling,
 * rel-pos MHSA, conv module with BN), TDT / RNNT / CTC greedy decode.
 * Encoder weights stay bf16 in the memory-mapped checkpoint. */
#include "pk.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dirent.h>
#endif

/* ------------------------------------------------------------ loading */
static const pk_tensor *need(const pk_model *m, const char *name, int *err) {
    const pk_tensor *t = pk_st_get(&m->st, name);
    if (!t) { fprintf(stderr, "parakeet-c: missing tensor %s\n", name); *err = 1; }
    return t;
}
/* Same, but the tensor must hold exactly n values: everything later
 * indexes these vectors by the model dimensions, never by their size. */
static float *getf_n(const pk_model *m, const char *name, int64_t n, int *err) {
    const pk_tensor *t = need(m, name, err);
    if (!t) return NULL;
    if (pk_tensor_numel(t) != n) {
        fprintf(stderr, "parakeet-c: tensor %s has %lld values, expected %lld\n", name, (long long)pk_tensor_numel(t), (long long)n);
        *err = 1; return NULL;
    }
    return pk_tensor_f32(t);
}
static float *getf_opt(const pk_model *m, const char *name) {
    const pk_tensor *t = pk_st_get(&m->st, name); return t ? pk_tensor_f32(t) : NULL;
}
static void getlin(const pk_model *m, pk_lin *l, const char *wname, const char *bname, int *err) {
    memset(l, 0, sizeof *l);
    const pk_tensor *t = need(m, wname, err); if (!t) return;
    int64_t numel = pk_tensor_numel(t);
    if (t->ndim < 1 || t->shape[0] <= 0 || t->shape[0] > INT32_MAX || numel % t->shape[0] || numel / t->shape[0] > INT32_MAX
        || (t->dtype != PK_BF16 && t->dtype != PK_F32 && t->dtype != PK_F16)) {
        fprintf(stderr, "parakeet-c: weight %s has an unusable shape or type\n", wname); *err = 1; return;
    }
    l->N = (int)t->shape[0]; l->K = (int)(numel / t->shape[0]);
    if (t->dtype == PK_BF16) l->w = (const uint16_t *)t->data; else l->wf = pk_tensor_f32(t);
    l->b = NULL;
    if (bname) {
        const pk_tensor *b = pk_st_get(&m->st, bname);
        if (b && pk_tensor_numel(b) != l->N) { fprintf(stderr, "parakeet-c: bias %s does not match %s\n", bname, wname); *err = 1; return; }
        if (b) l->b = pk_tensor_f32(b);
    }
}
/* the tensor's dimension i, or -1 when it has fewer dimensions */
static int64_t dim(const pk_tensor *t, int i) { return t && i < t->ndim ? t->shape[i] : -1; }
static void lin(float *C, const float *A, const pk_lin *l, int M) {
    if (l->w) pk_gemm_bf16w(C, A, l->w, l->b, M, l->N, l->K);
    else pk_gemm_f32w(C, A, l->wf, l->b, M, l->N, l->K);
}

static int find_file(const char *dir, const char *suffix, char *out, int cap) {
#ifdef _WIN32
    char pat[1024]; snprintf(pat, sizeof pat, "%s\\*%s", dir, suffix);
    WIN32_FIND_DATAA fd; HANDLE h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return -1;
    snprintf(out, cap, "%s\\%s", dir, fd.cFileName); FindClose(h); return 0;
#else
    DIR *d = opendir(dir); if (!d) return -1; struct dirent *e; size_t sl = strlen(suffix);
    while ((e = readdir(d))) { size_t l = strlen(e->d_name); if (l >= sl && !strcmp(e->d_name + l - sl, suffix)) { snprintf(out, cap, "%s/%s", dir, e->d_name); closedir(d); return 0; } }
    closedir(d); return -1;
#endif
}

/* Dimension checks. The decoder works in fixed stack buffers (the LSTM
 * state, gates, joint vectors, attention query rows and the duration
 * table) and the encoder sizes its heap buffers from d_model and d_ff,
 * so a model whose shapes exceed or contradict those sizes is rejected
 * here, before any decoding. Split in stages because the loader learns
 * the dimensions in that order. Returns 0 when the model fits. */
int pk_model_check_dims(const pk_model *m, int stage) {
    const char *why = NULL;
    if (stage & PK_CHECK_ENCODER) {
        if (m->d_model <= 0 || m->d_model > PK_MAX_DIM || (m->d_model & 1)) why = "d_model out of range or odd";
        else if (m->n_heads <= 0 || m->d_model % m->n_heads) why = "heads do not divide d_model";
        else if (m->d_model / m->n_heads > PK_MAX_HEAD_DIM) why = "head width exceeds the attention buffer";
        else if (m->n_layers <= 0 || m->n_layers > PK_MAX_LAYERS) why = "layer count out of range";
        else if (m->d_ff <= 0 || m->d_ff > PK_MAX_FF) why = "d_ff out of range";
        else if (m->conv_ks <= 0 || !(m->conv_ks & 1) || m->conv_ks > 1023) why = "conv kernel must be odd and at most 1023";
        else if (m->sub_ch <= 0 || m->sub_ch > PK_MAX_DIM) why = "subsampling channels out of range";
        else if (m->feat_in <= 0 || m->feat_in > 8 * PK_MAX_DIM) why = "feature width out of range";
    }
    if (!why && (stage & PK_CHECK_DECODER)) {
        if (m->vocab < 1 || m->vocab > PK_MAX_VOCAB) why = "vocabulary out of range";
        else if (m->is_tdt == 2) {
            if (m->ctc.K != m->d_model) why = "CTC head does not take d_model inputs";
        } else {
            if (m->pred_hidden <= 0 || m->pred_hidden > PK_MAX_PRED) why = "prediction width exceeds the LSTM buffers";
            else if (m->pred_layers < 1 || m->pred_layers > 2) why = "LSTM layer count must be 1 or 2";
            else if (m->joint_hidden <= 0 || m->joint_hidden > PK_MAX_PRED) why = "joint width exceeds the joint buffers";
            else if (m->j_enc.K != m->d_model) why = "joint encoder input is not d_model";
            else if (m->j_pred.K != m->pred_hidden || m->j_pred.N != m->joint_hidden) why = "joint prediction projection does not match";
            else if (m->j_out.K != m->joint_hidden) why = "joint output input is not the joint width";
            else if (m->n_dur < 0 || m->n_dur > PK_MAX_DUR) why = "duration count out of range";
            else if (m->j_out.N != m->vocab + 1 + m->n_dur) why = "joint output is not vocab + blank + durations";
        }
    }
    if (why) { fprintf(stderr, "parakeet-c: model rejected: %s\n", why); return -1; }
    return 0;
}
/* every projection in one encoder layer must match d_model and d_ff */
int pk_model_check_layer(const pk_model *m, const pk_layer *L) {
    const int D = m->d_model, F = m->d_ff;
    struct { const pk_lin *l; int n, k; const char *name; } want[] = {
        {&L->ff1_1, F, D, "feed_forward1.linear1"}, {&L->ff1_2, D, F, "feed_forward1.linear2"},
        {&L->ff2_1, F, D, "feed_forward2.linear1"}, {&L->ff2_2, D, F, "feed_forward2.linear2"},
        {&L->q, D, D, "linear_q"}, {&L->k, D, D, "linear_k"}, {&L->v, D, D, "linear_v"}, {&L->o, D, D, "linear_out"},
        {&L->pos, D, D, "linear_pos"}, {&L->pw1, 2 * D, D, "pointwise_conv1"}, {&L->pw2, D, D, "pointwise_conv2"},
    };
    for (size_t i = 0; i < sizeof want / sizeof want[0]; i++)
        if (want[i].l->N != want[i].n || want[i].l->K != want[i].k) {
            fprintf(stderr, "parakeet-c: model rejected: %s is %dx%d, expected %dx%d\n", want[i].name, want[i].l->N, want[i].l->K, want[i].n, want[i].k);
            return -1;
        }
    return 0;
}

int pk_model_load(pk_model *m, const char *dir) {
    memset(m, 0, sizeof *m);
    char path[1024];
    if (find_file(dir, "model_weights.safetensors", path, sizeof path) && find_file(dir, ".safetensors", path, sizeof path)) {
        fprintf(stderr, "parakeet-c: no .safetensors in %s\n", dir); return -1;
    }
    int rc = pk_st_open(&m->st, path);
    if (rc) { fprintf(stderr, "parakeet-c: cannot map %s (%d)\n", path, rc); return -1; }
    int err = 0; char nm[200];
    const pk_tensor *t;
    /* dimensions */
    /* Every dimension is read from a tensor shape, which a hostile or
     * damaged file controls; pk_model_check_dims below rejects anything
     * the decode buffers cannot hold before any of it is used. */
    t = need(m, "encoder.pre_encode.out.weight", &err); if (err) return -1;
    if (dim(t, 0) <= 0 || dim(t, 0) > PK_MAX_DIM) { fprintf(stderr, "parakeet-c: bad d_model\n"); return -1; }
    m->d_model = (int)t->shape[0];
    t = need(m, "encoder.pre_encode.conv.0.weight", &err); if (err) return -1;
    if (dim(t, 0) <= 0 || dim(t, 0) > PK_MAX_DIM) { fprintf(stderr, "parakeet-c: bad subsampling channels\n"); return -1; }
    m->sub_ch = (int)t->shape[0];
    {
        int64_t per = pk_tensor_numel(pk_st_get(&m->st, "encoder.pre_encode.out.weight")) / m->d_model / m->sub_ch;
        if (per <= 0 || per > PK_MAX_DIM) { fprintf(stderr, "parakeet-c: bad feature width\n"); return -1; }
        m->feat_in = (int)per * 8;
    }
    while (m->n_layers < PK_MAX_LAYERS && (snprintf(nm, sizeof nm, "encoder.layers.%d.norm_out.weight", m->n_layers), pk_st_get(&m->st, nm))) m->n_layers++;
    t = need(m, "encoder.layers.0.self_attn.pos_bias_u", &err); if (err) return -1;
    if (dim(t, 0) <= 0 || dim(t, 0) > PK_MAX_DIM) { fprintf(stderr, "parakeet-c: bad head count\n"); return -1; }
    m->n_heads = (int)t->shape[0];
    t = need(m, "encoder.layers.0.feed_forward1.linear1.weight", &err); if (err) return -1;
    if (dim(t, 0) <= 0 || dim(t, 0) > PK_MAX_FF) { fprintf(stderr, "parakeet-c: bad d_ff\n"); return -1; }
    m->d_ff = (int)t->shape[0];
    t = need(m, "encoder.layers.0.conv.depthwise_conv.weight", &err); if (err) return -1;
    if (dim(t, 2) <= 0 || dim(t, 2) > 1023) { fprintf(stderr, "parakeet-c: bad conv kernel size\n"); return -1; }
    m->conv_ks = (int)t->shape[2];
    if (pk_model_check_dims(m, PK_CHECK_ENCODER)) return -1;
    const int64_t D = m->d_model, C = m->sub_ch;
    /* pre-encode */
    m->c0_w = getf_n(m, "encoder.pre_encode.conv.0.weight", C * 9, &err); m->c0_b = getf_n(m, "encoder.pre_encode.conv.0.bias", C, &err);
    m->c2_w = getf_n(m, "encoder.pre_encode.conv.2.weight", C * 9, &err); m->c2_b = getf_n(m, "encoder.pre_encode.conv.2.bias", C, &err);
    m->c3_w = getf_n(m, "encoder.pre_encode.conv.3.weight", C * C, &err); m->c3_b = getf_n(m, "encoder.pre_encode.conv.3.bias", C, &err);
    m->c5_w = getf_n(m, "encoder.pre_encode.conv.5.weight", C * 9, &err); m->c5_b = getf_n(m, "encoder.pre_encode.conv.5.bias", C, &err);
    m->c6_w = getf_n(m, "encoder.pre_encode.conv.6.weight", C * C, &err); m->c6_b = getf_n(m, "encoder.pre_encode.conv.6.bias", C, &err);
    getlin(m, &m->pre_out, "encoder.pre_encode.out.weight", "encoder.pre_encode.out.bias", &err);
    if (err) return -1;
    {   /* the flattened subsampler output is C * F3 wide, F3 = ceil(feat_in / 8) */
        int F1 = (m->feat_in - 1) / 2 + 1, F2 = (F1 - 1) / 2 + 1, F3 = (F2 - 1) / 2 + 1;
        if (m->pre_out.N != D || (int64_t)m->pre_out.K != C * F3) { fprintf(stderr, "parakeet-c: model rejected: pre_encode.out does not match the subsampler\n"); return -1; }
    }
    m->L = calloc(m->n_layers, sizeof(pk_layer));
    for (int i = 0; i < m->n_layers; i++) {
        pk_layer *L = &m->L[i];
#define P(s) (snprintf(nm, sizeof nm, "encoder.layers.%d.%s", i, s), nm)
#define PB(s) (snprintf(nmb, sizeof nmb, "encoder.layers.%d.%s", i, s), nmb)
        char nmb[200];
        L->ln_ff1_w = getf_n(m, P("norm_feed_forward1.weight"), D, &err); L->ln_ff1_b = getf_n(m, P("norm_feed_forward1.bias"), D, &err);
        L->ln_att_w = getf_n(m, P("norm_self_att.weight"), D, &err);     L->ln_att_b = getf_n(m, P("norm_self_att.bias"), D, &err);
        L->ln_conv_w = getf_n(m, P("norm_conv.weight"), D, &err);        L->ln_conv_b = getf_n(m, P("norm_conv.bias"), D, &err);
        L->ln_ff2_w = getf_n(m, P("norm_feed_forward2.weight"), D, &err); L->ln_ff2_b = getf_n(m, P("norm_feed_forward2.bias"), D, &err);
        L->ln_out_w = getf_n(m, P("norm_out.weight"), D, &err);          L->ln_out_b = getf_n(m, P("norm_out.bias"), D, &err);
        getlin(m, &L->ff1_1, P("feed_forward1.linear1.weight"), PB("feed_forward1.linear1.bias"), &err);
        getlin(m, &L->ff1_2, P("feed_forward1.linear2.weight"), PB("feed_forward1.linear2.bias"), &err);
        getlin(m, &L->ff2_1, P("feed_forward2.linear1.weight"), PB("feed_forward2.linear1.bias"), &err);
        getlin(m, &L->ff2_2, P("feed_forward2.linear2.weight"), PB("feed_forward2.linear2.bias"), &err);
        getlin(m, &L->q, P("self_attn.linear_q.weight"), PB("self_attn.linear_q.bias"), &err);
        getlin(m, &L->k, P("self_attn.linear_k.weight"), PB("self_attn.linear_k.bias"), &err);
        getlin(m, &L->v, P("self_attn.linear_v.weight"), PB("self_attn.linear_v.bias"), &err);
        getlin(m, &L->o, P("self_attn.linear_out.weight"), PB("self_attn.linear_out.bias"), &err);
        getlin(m, &L->pos, P("self_attn.linear_pos.weight"), NULL, &err);
        L->bias_u = getf_n(m, P("self_attn.pos_bias_u"), D, &err); L->bias_v = getf_n(m, P("self_attn.pos_bias_v"), D, &err);
        getlin(m, &L->pw1, P("conv.pointwise_conv1.weight"), PB("conv.pointwise_conv1.bias"), &err);
        getlin(m, &L->pw2, P("conv.pointwise_conv2.weight"), PB("conv.pointwise_conv2.bias"), &err);
        L->dw = getf_n(m, P("conv.depthwise_conv.weight"), D * m->conv_ks, &err);
        L->dw_b = getf_opt(m, P("conv.depthwise_conv.bias"));
        if (L->dw_b && pk_tensor_numel(pk_st_get(&m->st, P("conv.depthwise_conv.bias"))) != D) { fprintf(stderr, "parakeet-c: bad depthwise bias\n"); err = 1; }
        float *bw = getf_n(m, P("conv.batch_norm.weight"), D, &err), *bb = getf_n(m, P("conv.batch_norm.bias"), D, &err);
        float *bm = getf_n(m, P("conv.batch_norm.running_mean"), D, &err), *bv = getf_n(m, P("conv.batch_norm.running_var"), D, &err);
        if (err) return -1;
        if (pk_model_check_layer(m, L)) return -1;
        L->bn_scale = pk_alloc(sizeof(float) * m->d_model); L->bn_shift = pk_alloc(sizeof(float) * m->d_model);
        for (int c = 0; c < m->d_model; c++) {
            double s = bw[c] / sqrt((double)bv[c] + 1e-5);
            L->bn_scale[c] = (float)s; L->bn_shift[c] = (float)(bb[c] - bm[c] * s);
            if (L->dw_b) { L->bn_shift[c] += (float)(L->dw_b[c] * s); }
        }
        pk_free(bw); pk_free(bb); pk_free(bm); pk_free(bv);
#undef P
#undef PB
    }
    if (err) return -1;
    /* decoder / joint (TDT or RNNT) or CTC */
    if (pk_st_get(&m->st, "decoder.prediction.embed.weight")) {
        t = pk_st_get(&m->st, "decoder.prediction.embed.weight");
        if (t->ndim != 2 || dim(t, 0) < 2 || dim(t, 0) > PK_MAX_VOCAB + 1 || dim(t, 1) <= 0 || dim(t, 1) > PK_MAX_PRED) {
            fprintf(stderr, "parakeet-c: prediction embedding shape out of range\n"); return -1;
        }
        m->vocab = (int)t->shape[0] - 1; m->pred_hidden = (int)t->shape[1];
        m->embed = pk_tensor_f32(t);
        while (m->pred_layers < 2 && (snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_ih_l%d", m->pred_layers), pk_st_get(&m->st, nm))) m->pred_layers++;
        snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_ih_l%d", m->pred_layers);
        if (m->pred_layers == 2 && pk_st_get(&m->st, nm)) { fprintf(stderr, "parakeet-c: more than 2 LSTM layers is not supported\n"); return -1; }
        const int64_t PH = m->pred_hidden;
        for (int l = 0; l < m->pred_layers; l++) {
            snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_ih_l%d", l); m->lstm_wih[l] = getf_n(m, nm, 4 * PH * PH, &err);
            snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.weight_hh_l%d", l); m->lstm_whh[l] = getf_n(m, nm, 4 * PH * PH, &err);
            snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.bias_ih_l%d", l); float *b1 = getf_n(m, nm, 4 * PH, &err);
            snprintf(nm, sizeof nm, "decoder.prediction.dec_rnn.lstm.bias_hh_l%d", l); float *b2 = getf_n(m, nm, 4 * PH, &err);
            if (err) return -1;
            m->lstm_b[l] = pk_alloc(sizeof(float) * 4 * m->pred_hidden);
            for (int i = 0; i < 4 * m->pred_hidden; i++) m->lstm_b[l][i] = b1[i] + b2[i];
            pk_free(b1); pk_free(b2);
        }
        getlin(m, &m->j_enc, "joint.enc.weight", "joint.enc.bias", &err);
        getlin(m, &m->j_pred, "joint.pred.weight", "joint.pred.bias", &err);
        getlin(m, &m->j_out, "joint.joint_net.2.weight", "joint.joint_net.2.bias", &err);
        if (err) return -1;
        m->joint_hidden = m->j_enc.N;
        m->n_dur = m->j_out.N - (m->vocab + 1);
        m->is_tdt = m->n_dur > 0;
        if (pk_model_check_dims(m, PK_CHECK_DECODER)) return -1;
        for (int i = 0; i < m->n_dur; i++) m->durations[i] = i;
    } else if (pk_st_get(&m->st, "decoder.decoder_layers.0.weight")) {
        getlin(m, &m->ctc, "decoder.decoder_layers.0.weight", "decoder.decoder_layers.0.bias", &err);
        if (err) return -1;
        m->vocab = m->ctc.N - 1; m->is_tdt = 2;
        if (pk_model_check_dims(m, PK_CHECK_DECODER)) return -1;
    } else { fprintf(stderr, "parakeet-c: no decoder found\n"); return -1; }
    /* mel: use the checkpoint's own filterbank/window when present */
    t = pk_st_get(&m->st, "preprocessor.featurizer.fb");
    if (t && (t->ndim < 2 || dim(t, t->ndim - 2) != m->feat_in)) {
        fprintf(stderr, "parakeet-c: mel filterbank does not match the encoder's %d features\n", m->feat_in); return -1;
    }
    int nmel = m->feat_in;
    pk_mel_default_fb(&m->mel, nmel);
    if (t && t->shape[t->ndim - 1] == PK_NBIN) { pk_free(m->mel.fb); m->mel.fb = pk_tensor_f32(t); }
    t = pk_st_get(&m->st, "preprocessor.featurizer.window");
    if (t && pk_tensor_numel(t) == PK_WIN) { pk_free(m->mel.window); m->mel.window = pk_tensor_f32(t); }
    /* tokenizer */
    if (find_file(dir, "tokenizer.model", path, sizeof path) || pk_spm_load(path, &m->pieces, &m->n_pieces)) {
        fprintf(stderr, "parakeet-c: no SentencePiece tokenizer.model in %s\n", dir); return -1;
    }
    return 0;
}
void pk_model_free(pk_model *m) { pk_st_close(&m->st); /* process exit frees the rest */ }

/* ------------------------------------------------------------ row-parallel ops */
typedef struct {
    int op, T, D, D2;
    float *x, *y, *z; const float *w, *b, *w2;
    float scale;
} rows_job;
enum { OP_LN, OP_SWISH, OP_ADD_SCALED, OP_GLU, OP_DWCONV_BN_SWISH };
#define ROWB 4
static inline float sigm(float v) { return 1.f / (1.f + expf(-v)); }
static void rows_task(void *vc, int task) {
    const rows_job *j = vc;
    int t0 = task * ROWB, t1 = t0 + ROWB; if (t1 > j->T) t1 = j->T;
    const int D = j->D;
    for (int t = t0; t < t1; t++) {
        switch (j->op) {
        case OP_LN: { /* y = LN(x) */
            const float *x = j->x + (size_t)t * D; float *y = j->y + (size_t)t * D;
            double mu = 0, var = 0; for (int i = 0; i < D; i++) mu += x[i]; mu /= D;
            for (int i = 0; i < D; i++) { double d = x[i] - mu; var += d * d; } var /= D;
            float inv = (float)(1.0 / sqrt(var + 1e-5)), m = (float)mu;
            for (int i = 0; i < D; i++) y[i] = (x[i] - m) * inv * j->w[i] + j->b[i];
        } break;
        case OP_SWISH: { float *x = j->x + (size_t)t * D; for (int i = 0; i < D; i++) x[i] = x[i] * sigm(x[i]); } break;
        case OP_ADD_SCALED: { float *x = j->x + (size_t)t * D; const float *y = j->y + (size_t)t * D; for (int i = 0; i < D; i++) x[i] += j->scale * y[i]; } break;
        case OP_GLU: { const float *x = j->x + (size_t)t * 2 * D; float *y = j->y + (size_t)t * D; for (int i = 0; i < D; i++) y[i] = x[i] * sigm(x[D + i]); } break;
        case OP_DWCONV_BN_SWISH: { /* y[t] = swish(bn(sum_k w[c][k] x[t+k-pad][c])) */
            const int ks = j->D2, pad = (ks - 1) / 2; float *y = j->y + (size_t)t * D;
            for (int c = 0; c < D; c++) y[c] = 0;
            for (int k = 0; k < ks; k++) {
                int s = t + k - pad; if (s < 0 || s >= j->T) continue;
                const float *x = j->x + (size_t)s * D;
                for (int c = 0; c < D; c++) y[c] += j->w[c * ks + k] * x[c];
            }
            for (int c = 0; c < D; c++) { float v = y[c] * j->b[c] + j->w2[c]; y[c] = v * sigm(v); }
        } break;
        }
    }
}
static void rows(rows_job j) { pk_parallel_for((j.T + ROWB - 1) / ROWB, rows_task, &j); }

/* ------------------------------------------------------------ subsampling */
typedef struct { int mode; const float *in; float *out; const float *w, *b; int Tin, Fin, Tout, Fout, C; } sub_job;
static void sub_task(void *vc, int t) {
    const sub_job *j = vc; const int C = j->C;
    for (int f = 0; f < j->Fout; f++) {
        float *o = j->out + ((size_t)t * j->Fout + f) * C;
        for (int c = 0; c < C; c++) o[c] = j->b[c];
        for (int ki = 0; ki < 3; ki++) {
            int ti = 2 * t - 1 + ki; if (ti < 0 || ti >= j->Tin) continue;
            for (int kj = 0; kj < 3; kj++) {
                int fi = 2 * f - 1 + kj; if (fi < 0 || fi >= j->Fin) continue;
                if (j->mode == 0) { float v = j->in[(size_t)ti * j->Fin + fi]; for (int c = 0; c < C; c++) o[c] += j->w[c * 9 + ki * 3 + kj] * v; }
                else { const float *v = j->in + ((size_t)ti * j->Fin + fi) * C; for (int c = 0; c < C; c++) o[c] += j->w[c * 9 + ki * 3 + kj] * v[c]; }
            }
        }
        if (j->mode == 0) for (int c = 0; c < C; c++) if (o[c] < 0) o[c] = 0; /* conv.1 ReLU */
    }
}
typedef struct { float *x; int n; } relu_job;
static void relu_task(void *vc, int i) { relu_job *j = vc; int a = i * 4096, b = a + 4096; if (b > j->n) b = j->n; for (int k = a; k < b; k++) if (j->x[k] < 0) j->x[k] = 0; }
static void relu_all(float *x, int n) { relu_job j = { x, n }; pk_parallel_for((n + 4095) / 4096, relu_task, &j); }

/* ------------------------------------------------------------ attention */
typedef struct { const float *q, *k, *v, *p, *bu, *bv; float *out, *scr; int T, H, dk, D; } att_job;
#define ATT_QB 4
static void att_task(void *vc, int task) {
    const att_job *j = vc; const int T = j->T, dk = j->dk, D = j->D;
    int nqb = (T + ATT_QB - 1) / ATT_QB;
    int h = task / nqb, i0 = (task % nqb) * ATT_QB, i1 = i0 + ATT_QB; if (i1 > T) i1 = T;
    float *sc = j->scr + (size_t)task * T;
    const float isq = 1.f / sqrtf((float)dk);
    float qu[PK_MAX_HEAD_DIM], qv[PK_MAX_HEAD_DIM];
    for (int i = i0; i < i1; i++) {
        const float *qi = j->q + (size_t)i * D + h * dk;
        for (int d = 0; d < dk; d++) { qu[d] = qi[d] + j->bu[h * dk + d]; qv[d] = qi[d] + j->bv[h * dk + d]; }
        float mx = -1e30f;
        for (int jj = 0; jj < T; jj++) {
            const float *kj = j->k + (size_t)jj * D + h * dk;
            const float *pj = j->p + (size_t)(T - 1 - i + jj) * D + h * dk;
            float ac = 0, bd = 0;
            for (int d = 0; d < dk; d++) { ac += qu[d] * kj[d]; bd += qv[d] * pj[d]; }
            float s = (ac + bd) * isq; sc[jj] = s; if (s > mx) mx = s;
        }
        float sum = 0; for (int jj = 0; jj < T; jj++) { sc[jj] = expf(sc[jj] - mx); sum += sc[jj]; }
        float inv = 1.f / sum;
        float *o = j->out + (size_t)i * D + h * dk;
        for (int d = 0; d < dk; d++) o[d] = 0;
        for (int jj = 0; jj < T; jj++) { const float *vj = j->v + (size_t)jj * D + h * dk; float a = sc[jj] * inv; for (int d = 0; d < dk; d++) o[d] += a * vj[d]; }
    }
}

int pk_encode(const pk_model *m, const float *feats, int T, float **enc_out) {
    const int C = m->sub_ch, F0 = m->feat_in;
    int T1 = (T - 1) / 2 + 1, F1 = (F0 - 1) / 2 + 1;
    int T2 = (T1 - 1) / 2 + 1, F2 = (F1 - 1) / 2 + 1;
    int T3 = (T2 - 1) / 2 + 1, F3 = (F2 - 1) / 2 + 1;
    float *a1 = pk_alloc(sizeof(float) * (size_t)T1 * F1 * C);
    sub_job sj = { 0, feats, a1, m->c0_w, m->c0_b, T, F0, T1, F1, C };
    pk_parallel_for(T1, sub_task, &sj);
    float *a2 = pk_alloc(sizeof(float) * (size_t)T2 * F2 * C), *a3 = pk_alloc(sizeof(float) * (size_t)T2 * F2 * C);
    sj = (sub_job){ 1, a1, a2, m->c2_w, m->c2_b, T1, F1, T2, F2, C };
    pk_parallel_for(T2, sub_task, &sj);
    pk_gemm_f32w(a3, a2, m->c3_w, m->c3_b, T2 * F2, C, C); relu_all(a3, T2 * F2 * C);
    float *a4 = pk_alloc(sizeof(float) * (size_t)T3 * F3 * C), *a5 = pk_alloc(sizeof(float) * (size_t)T3 * F3 * C);
    sj = (sub_job){ 1, a3, a4, m->c5_w, m->c5_b, T2, F2, T3, F3, C };
    pk_parallel_for(T3, sub_task, &sj);
    pk_gemm_f32w(a5, a4, m->c6_w, m->c6_b, T3 * F3, C, C); relu_all(a5, T3 * F3 * C);
    /* flatten [t][f][c] -> [t][c*F3+f] */
    float *flat = pk_alloc(sizeof(float) * (size_t)T3 * C * F3);
    for (int t = 0; t < T3; t++) for (int f = 0; f < F3; f++) for (int c = 0; c < C; c++)
        flat[(size_t)t * C * F3 + c * F3 + f] = a5[((size_t)t * F3 + f) * C + c];
    pk_free(a1); pk_free(a2); pk_free(a3); pk_free(a4); pk_free(a5);
    const int D = m->d_model, Te = T3, H = m->n_heads, dk = D / H, DF = m->d_ff;
    float *x = pk_alloc(sizeof(float) * (size_t)Te * D);
    lin(x, flat, &m->pre_out, Te); pk_free(flat);
    /* relative positional encoding: positions Te-1 ... -(Te-1) */
    int NP = 2 * Te - 1;
    float *pe = pk_alloc(sizeof(float) * (size_t)NP * D), *pp = pk_alloc(sizeof(float) * (size_t)NP * D);
    for (int k = 0; k < NP; k++) {
        double pos = (double)(Te - 1 - k);
        for (int i = 0; i < D / 2; i++) {
            double div = exp((2.0 * i) * -(log(10000.0) / D));
            pe[(size_t)k * D + 2 * i] = (float)sin(pos * div); pe[(size_t)k * D + 2 * i + 1] = (float)cos(pos * div);
        }
    }
    float *ln = pk_alloc(sizeof(float) * (size_t)Te * D), *hb = pk_alloc(sizeof(float) * (size_t)Te * (DF > 2 * D ? DF : 2 * D));
    float *q = pk_alloc(sizeof(float) * (size_t)Te * D), *k = pk_alloc(sizeof(float) * (size_t)Te * D), *v = pk_alloc(sizeof(float) * (size_t)Te * D);
    float *att = pk_alloc(sizeof(float) * (size_t)Te * D), *tmp = pk_alloc(sizeof(float) * (size_t)Te * D);
    int natt = H * ((Te + ATT_QB - 1) / ATT_QB);
    float *scr = pk_alloc(sizeof(float) * (size_t)natt * Te);
    for (int li = 0; li < m->n_layers; li++) {
        const pk_layer *L = &m->L[li];
        /* FF1 (half step) */
        rows((rows_job){ OP_LN, Te, D, 0, x, ln, NULL, L->ln_ff1_w, L->ln_ff1_b, NULL, 0 });
        lin(hb, ln, &L->ff1_1, Te); rows((rows_job){ OP_SWISH, Te, DF, 0, hb, NULL, NULL, NULL, NULL, NULL, 0 });
        lin(tmp, hb, &L->ff1_2, Te); rows((rows_job){ OP_ADD_SCALED, Te, D, 0, x, tmp, NULL, NULL, NULL, NULL, 0.5f });
        /* rel-pos MHSA */
        rows((rows_job){ OP_LN, Te, D, 0, x, ln, NULL, L->ln_att_w, L->ln_att_b, NULL, 0 });
        lin(q, ln, &L->q, Te); lin(k, ln, &L->k, Te); lin(v, ln, &L->v, Te); lin(pp, pe, &L->pos, NP);
        att_job aj = { q, k, v, pp, L->bias_u, L->bias_v, att, scr, Te, H, dk, D };
        pk_parallel_for(natt, att_task, &aj);
        lin(tmp, att, &L->o, Te); rows((rows_job){ OP_ADD_SCALED, Te, D, 0, x, tmp, NULL, NULL, NULL, NULL, 1.0f });
        /* conv module */
        rows((rows_job){ OP_LN, Te, D, 0, x, ln, NULL, L->ln_conv_w, L->ln_conv_b, NULL, 0 });
        lin(hb, ln, &L->pw1, Te);
        rows((rows_job){ OP_GLU, Te, D, 0, hb, att, NULL, NULL, NULL, NULL, 0 });
        rows((rows_job){ OP_DWCONV_BN_SWISH, Te, D, m->conv_ks, att, ln, NULL, L->dw, L->bn_scale, L->bn_shift, 0 });
        lin(tmp, ln, &L->pw2, Te); rows((rows_job){ OP_ADD_SCALED, Te, D, 0, x, tmp, NULL, NULL, NULL, NULL, 1.0f });
        /* FF2 (half step) */
        rows((rows_job){ OP_LN, Te, D, 0, x, ln, NULL, L->ln_ff2_w, L->ln_ff2_b, NULL, 0 });
        lin(hb, ln, &L->ff2_1, Te); rows((rows_job){ OP_SWISH, Te, DF, 0, hb, NULL, NULL, NULL, NULL, NULL, 0 });
        lin(tmp, hb, &L->ff2_2, Te); rows((rows_job){ OP_ADD_SCALED, Te, D, 0, x, tmp, NULL, NULL, NULL, NULL, 0.5f });
        /* out norm */
        rows((rows_job){ OP_LN, Te, D, 0, x, ln, NULL, L->ln_out_w, L->ln_out_b, NULL, 0 });
        float *sw = x; x = ln; ln = sw;
    }
    pk_free(pe); pk_free(pp); pk_free(ln); pk_free(hb); pk_free(q); pk_free(k); pk_free(v); pk_free(att); pk_free(tmp); pk_free(scr);
    *enc_out = x;
    return Te;
}

/* ------------------------------------------------------------ decoding */
typedef struct { float h[2][PK_MAX_PRED], c[2][PK_MAX_PRED]; } lstm_state;
static void pred_step(const pk_model *m, int label, const lstm_state *in, lstm_state *out, float *g) {
    const int H = m->pred_hidden; float gates[4 * PK_MAX_PRED]; float x[PK_MAX_PRED];
    memcpy(x, m->embed + (size_t)label * H, sizeof(float) * H);
    for (int l = 0; l < m->pred_layers; l++) {
        float gh[4 * PK_MAX_PRED];
        pk_gemm_f32w(gates, x, m->lstm_wih[l], m->lstm_b[l], 1, 4 * H, H);
        pk_gemm_f32w(gh, in->h[l], m->lstm_whh[l], NULL, 1, 4 * H, H);
        for (int i = 0; i < 4 * H; i++) gates[i] += gh[i];
        for (int i = 0; i < H; i++) {
            float ig = sigm(gates[i]), fg = sigm(gates[H + i]), gg = tanhf(gates[2 * H + i]), og = sigm(gates[3 * H + i]);
            float c = fg * in->c[l][i] + ig * gg;
            out->c[l][i] = c; out->h[l][i] = og * tanhf(c);
        }
        memcpy(x, out->h[l], sizeof(float) * H);
    }
    memcpy(g, x, sizeof(float) * H);
}

int pk_decode_greedy(const pk_model *m, const float *enc, int Te, int **tok_out, int **ts_out) {
    int cap = Te * 4 + 16, n = 0;
    int *tok = malloc(sizeof(int) * cap), *ts = malloc(sizeof(int) * cap);
    const int V = m->vocab, blank = V;
    if (m->is_tdt == 2) { /* CTC */
        float *lg = pk_alloc(sizeof(float) * (size_t)Te * (V + 1));
        lin(lg, enc, &m->ctc, Te);
        int prev = -1;
        for (int t = 0; t < Te; t++) {
            const float *r = lg + (size_t)t * (V + 1); int best = 0; for (int i = 1; i <= V; i++) if (r[i] > r[best]) best = i;
            if (best != blank && best != prev) { tok[n] = best; ts[n] = t; n++; }
            prev = best;
        }
        pk_free(lg); *tok_out = tok; *ts_out = ts; return n;
    }
    const int J = m->joint_hidden, NO = m->j_out.N;
    float *f = pk_alloc(sizeof(float) * (size_t)Te * J);
    lin(f, enc, &m->j_enc, Te);
    static lstm_state st, st2;
    memset(&st, 0, sizeof st);
    float g[PK_MAX_PRED], gp[PK_MAX_PRED], hsum[PK_MAX_PRED], *logits = pk_alloc(sizeof(float) * NO);
    int have_g = 0, last = blank;
    const int max_symbols = 10;
    int t = 0;
    while (t < Te) {
        int symbols = 0, need = 1, skip = 1;
        while (need && symbols < max_symbols) {
            if (!have_g) { pred_step(m, last, &st, &st2, g); lin(gp, g, &m->j_pred, 1); have_g = 1; }
            const float *ft = f + (size_t)t * J;
            for (int i = 0; i < J; i++) { float s = ft[i] + gp[i]; hsum[i] = s > 0 ? s : 0; }
            lin(logits, hsum, &m->j_out, 1);
            int k = 0; for (int i = 1; i <= V; i++) if (logits[i] > logits[k]) k = i;
            if (m->is_tdt) {
                int dk = 0; for (int i = 1; i < m->n_dur; i++) if (logits[V + 1 + i] > logits[V + 1 + dk]) dk = i;
                skip = m->durations[dk];
            } else skip = (k == blank) ? 1 : 0;
            if (k != blank) {
                if (n == cap) { cap *= 2; tok = realloc(tok, sizeof(int) * cap); ts = realloc(ts, sizeof(int) * cap); }
                tok[n] = k; ts[n] = t; n++;
                st = st2; last = k; have_g = 0;
            }
            symbols++;
            t += skip;
            need = (skip == 0);
        }
        if (need && symbols >= max_symbols) t += 1;
    }
    pk_free(f); pk_free(logits);
    *tok_out = tok; *ts_out = ts; return n;
}

char *pk_detok(const pk_model *m, const int *tok, int n) { return pk_spm_detok(m->pieces, m->n_pieces, tok, n); }

/* ------------------------------------------------------------ pipeline */
static char *transcribe_one(const pk_model *m, const float *x, int n, pk_timing *tm) {
    double t0 = pk_now();
    float *feats = NULL; int T = pk_mel_compute(&m->mel, x, n, &feats);
    double t1 = pk_now();
    if (T < 2) { pk_free(feats); return calloc(1, 1); }
    float *enc = NULL; int Te = pk_encode(m, feats, T, &enc);
    double t2 = pk_now();
    int *tok, *ts; int nt = pk_decode_greedy(m, enc, Te, &tok, &ts);
    double t3 = pk_now();
    char *s = pk_detok(m, tok, nt);
    if (tm) { tm->mel_s += t1 - t0; tm->enc_s += t2 - t1; tm->dec_s += t3 - t2; tm->frames += Te; tm->tokens += nt; }
    free(tok); free(ts); pk_free(feats); pk_free(enc);
    return s;
}

char *pk_transcribe(const pk_model *m, const float *x, int n, pk_timing *tm) {
    double t0 = pk_now();
    if (tm) { memset(tm, 0, sizeof *tm); tm->audio_s = (double)n / PK_SR; }
    const int maxlen = 30 * PK_SR, minlen = 20 * PK_SR, win = PK_SR / 20;
    char *out = calloc(1, 1); size_t ol = 0;
    int pos = 0;
    while (pos < n) {
        int len = n - pos;
        if (len > maxlen + 5 * PK_SR) { /* cut at the quietest 50 ms between 20 s and 30 s */
            double best = 1e30; int cut = maxlen;
            for (int s = minlen; s + win <= maxlen; s += win / 2) {
                double e = 0; for (int i = 0; i < win; i++) { float v = x[pos + s + i]; e += v * v; }
                if (e < best) { best = e; cut = s + win / 2; }
            }
            len = cut;
        }
        char *s = transcribe_one(m, x + pos, len, tm);
        size_t sl = strlen(s);
        if (sl) { out = realloc(out, ol + sl + 2); if (ol) out[ol++] = ' '; memcpy(out + ol, s, sl + 1); ol += sl; }
        free(s);
        pos += len;
    }
    if (tm) tm->total_s = pk_now() - t0;
    return out;
}
