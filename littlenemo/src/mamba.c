/* mamba.c - see mamba.h and BLOCK-MATH-SPEC.md. */
#include "mamba.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float nemo_bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

float nemo_silu(float v) { return v / (1.0f + expf(-v)); }

float nemo_softplus(float v) {
    if (v > 20.0f) return v;
    return log1pf(expf(v));
}

void nemo_rmsnorm(const float *x, const float *w, float *out, int n, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * x[i];
    float rstd = (float)(1.0 / sqrt(ss / n + eps));
    for (int i = 0; i < n; i++) out[i] = x[i] * rstd * w[i];
}

void nemo_rmsnorm_gated(const float *x, const float *z, const float *w,
                        float *out, int n, int group_size, float eps) {
    /* Defensive floor: a non-positive group size would hang this loop
     * forever. Callers validate divisibility (nemo_mamba_decode does);
     * this guard keeps the primitive itself un-hangable. */
    if (group_size <= 0 || n <= 0) return;
    for (int base = 0; base < n; base += group_size) {
        double ss = 0.0;
        for (int i = 0; i < group_size; i++) {
            float g = x[base + i] * nemo_silu(z[base + i]);
            out[base + i] = g; /* stash gated value, scaled below */
            ss += (double)g * g;
        }
        float rstd = (float)(1.0 / sqrt(ss / group_size + eps));
        for (int i = 0; i < group_size; i++)
            out[base + i] = out[base + i] * rstd * w[base + i];
    }
}

void nemo_conv1d_step(float *win, const float *u, const float *weight,
                      const float *bias, float *out, int channels) {
    for (int c = 0; c < channels; c++) {
        float *w4 = win + (size_t)c * 4;
        w4[0] = w4[1]; w4[1] = w4[2]; w4[2] = w4[3]; w4[3] = u[c];
        double acc = bias ? bias[c] : 0.0;
        for (int k = 0; k < 4; k++) acc += (double)w4[k] * weight[(size_t)c * 4 + k];
        out[c] = nemo_silu((float)acc);
    }
}

int nemo_mamba_state_init(nemo_mamba_state *s, int heads, int head_dim,
                          int state_size, int groups) {
    memset(s, 0, sizeof *s);
    /* Geometry is validated here, at the door: the SSM step divides
     * heads by groups and indexes B/C per group, so a zero or
     * non-dividing group count must never get a state. */
    if (heads <= 0 || head_dim <= 0 || state_size <= 0 || groups <= 0)
        return -1;
    if (heads % groups != 0) return -1;
    if ((int64_t)heads * head_dim > INT32_MAX / state_size) return -1;
    s->heads = heads;
    s->head_dim = head_dim;
    s->state_size = state_size;
    s->groups = groups;
    s->inner = heads * head_dim;
    s->conv_dim = s->inner + 2 * groups * state_size;
    s->ssm_state = calloc((size_t)heads * head_dim * state_size, sizeof(float));
    s->conv_win = calloc((size_t)s->conv_dim * 4, sizeof(float));
    if (!s->ssm_state || !s->conv_win) {
        free(s->ssm_state);
        free(s->conv_win);
        s->ssm_state = s->conv_win = NULL;
        return -1;
    }
    return 0;
}

void nemo_mamba_state_free(nemo_mamba_state *s) {
    free(s->ssm_state);
    free(s->conv_win);
    s->ssm_state = s->conv_win = NULL;
}

void nemo_ssm_step(nemo_mamba_state *s, const float *x, const float *B,
                   const float *C, const float *dt_raw, const float *A_log,
                   const float *D, const float *dt_bias, float *y) {
    /* state_init validates this; the guard keeps a hand-built state
     * from dividing by zero here. */
    if (s->groups <= 0 || s->heads % s->groups != 0) return;
    int heads = s->heads, hd = s->head_dim, ns = s->state_size;
    int per_group = heads / s->groups;
    for (int h = 0; h < heads; h++) {
        int g = h / per_group;
        const float *Bg = B + (size_t)g * ns;
        const float *Cg = C + (size_t)g * ns;
        float dt = nemo_softplus(dt_raw[h] + dt_bias[h]);
        float A = -expf(A_log[h]);
        float decay = expf(dt * A);
        for (int d = 0; d < hd; d++) {
            float xv = x[(size_t)h * hd + d];
            float *st = s->ssm_state + ((size_t)h * hd + d) * ns;
            double acc = 0.0;
            for (int n = 0; n < ns; n++) {
                float v = st[n] * decay + dt * Bg[n] * xv;
                st[n] = v;
                acc += (double)Cg[n] * v;
            }
            y[(size_t)h * hd + d] = (float)acc + D[h] * xv;
        }
    }
}

int nemo_mamba_decode(nemo_mamba_state *s, const float *proj, int d_mlp,
                      const float *conv_w, const float *conv_b,
                      const float *A_log, const float *D,
                      const float *dt_bias, const float *norm_w,
                      int norm_groups, float eps, float *out) {
    /* norm_groups reaches a division below: validate before it can
     * divide by zero or hand the gated norm a non-dividing group. */
    if (norm_groups <= 0 || s->inner % norm_groups != 0) return -1;
    const float *z = proj + 2 * d_mlp;
    const float *u = z + s->inner;
    const float *dt_raw = u + s->conv_dim;
    float *v = malloc((size_t)s->conv_dim * sizeof(float));
    float *y = malloc((size_t)s->inner * sizeof(float));
    if (!v || !y) { free(v); free(y); return -1; }
    nemo_conv1d_step(s->conv_win, u, conv_w, conv_b, v, s->conv_dim);
    const float *x = v;
    const float *B = v + s->inner;
    const float *C = B + (size_t)s->groups * s->state_size;
    nemo_ssm_step(s, x, B, C, dt_raw, A_log, D, dt_bias, y);
    nemo_rmsnorm_gated(y, z, norm_w, out, s->inner, s->inner / norm_groups, eps);
    free(v);
    free(y);
    return 0;
}

float nemo_relu2(float v) { return v > 0.0f ? v * v : 0.0f; }
