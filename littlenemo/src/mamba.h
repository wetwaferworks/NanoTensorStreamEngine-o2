/* mamba.h - Little Nemo block math (Nemotron-H), pure C.
 * Implements littlenemo/BLOCK-MATH-SPEC.md. All math in fp32; bf16 weights
 * are widened exactly (bf16 -> fp32 is a pure bit shift). Dimensions are
 * parameters, never hard-coded, so tests can run tiny shapes.
 */
#ifndef NEMO_MAMBA_H
#define NEMO_MAMBA_H

#include <stdint.h>

float nemo_bf16_to_f32(uint16_t b);
float nemo_silu(float v);
float nemo_softplus(float v);

/* Plain RMSNorm over the full row: out = x * rstd * w,
 * rstd = 1 / sqrt(mean(x^2) + eps), statistics in fp32. */
void nemo_rmsnorm(const float *x, const float *w, float *out, int n, float eps);

/* Gated RMSNorm, norm_before_gate = False (Nemotron-H's setting):
 * g = x * silu(z) first, then RMSNorm in groups of group_size, then * w. */
void nemo_rmsnorm_gated(const float *x, const float *z, const float *w,
                        float *out, int n, int group_size, float eps);

/* Depthwise causal conv, one decode step. win holds the last 4 inputs per
 * channel (oldest first); u is pushed, then
 * out[c] = silu( sum_k win[k] * weight[c*4 + k] + bias[c] ). */
void nemo_conv1d_step(float *win, const float *u, const float *weight,
                      const float *bias, float *out, int channels);

/* Mamba-2 per-layer decode state. */
typedef struct {
    int heads;       /* 128 in the Nano */
    int head_dim;    /* 80 */
    int state_size;  /* 128 */
    int groups;      /* 8 (B and C are per group, broadcast to heads) */
    int inner;       /* heads * head_dim = 10240 */
    int conv_dim;    /* inner + 2 * groups * state_size = 12288 */
    float *ssm_state;  /* [heads][head_dim][state_size] */
    float *conv_win;   /* [conv_dim][4] */
} nemo_mamba_state;

int nemo_mamba_state_init(nemo_mamba_state *s, int heads, int head_dim,
                          int state_size, int groups);
void nemo_mamba_state_free(nemo_mamba_state *s);

/* One SSM decode step (spec: state updates BEFORE the C readout).
 * x, B, C are the post-conv splits; B and C are [groups * state_size].
 * dt_raw is [heads]. y receives [inner], before the gated norm. */
void nemo_ssm_step(nemo_mamba_state *s, const float *x, const float *B,
                   const float *C, const float *dt_raw, const float *A_log,
                   const float *D, const float *dt_bias, float *y);

/* Full Mamba-2 mixer decode for one token, up to (not including) out_proj.
 * proj is the in_proj output [2*d_mlp + inner + conv_dim + heads];
 * d_mlp is 0 for the Nano but kept as a parameter.
 * params: conv weight [conv_dim * 4], conv bias [conv_dim],
 * A_log [heads], D [heads], dt_bias [heads], norm weight [inner].
 * out receives [inner] after the gated norm. Returns 0 on success,
 * -1 on invalid geometry (norm_groups must be >= 1 and divide inner)
 * or allocation failure: the error is reported, never silent. */
int nemo_mamba_decode(nemo_mamba_state *s, const float *proj, int d_mlp,
                      const float *conv_w, const float *conv_b,
                      const float *A_log, const float *D,
                      const float *dt_bias, const float *norm_w,
                      int norm_groups, float eps, float *out);

/* MLP block activation: relu2(v) = relu(v)^2. */
float nemo_relu2(float v);

#endif
