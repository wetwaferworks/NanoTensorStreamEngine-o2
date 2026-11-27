/* sn_moe.c - see sn_moe.h. Pure C99, no locks, no globals. */
#include "sn_moe.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float sn_relu2(float v) { return v > 0.0f ? v * v : 0.0f; }
float sn_sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

void sn_matvec(const float *W, int rows, int cols, const float *x, float *y)
{
    for (int r = 0; r < rows; r++) {
        const float *w = W + (size_t)r * (size_t)cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}

int sn_moe_route(const sn_moe_cfg *c, const float *gate_w, const float *bias,
                 const float *x, int *top_idx, float *top_w)
{
    int E = c->n_experts, K = c->topk;
    if (E <= 0 || K <= 0 || K > E || c->hidden <= 0) return -1;
    float *s = (float *)malloc(sizeof(float) * (size_t)E);
    float *ch = (float *)malloc(sizeof(float) * (size_t)E);
    if (!s || !ch) { free(s); free(ch); return -1; }
    for (int e = 0; e < E; e++) {
        float acc = 0.0f;
        const float *w = gate_w + (size_t)e * (size_t)c->hidden;
        for (int i = 0; i < c->hidden; i++) acc += w[i] * x[i];
        s[e] = sn_sigmoid(acc);
        ch[e] = s[e] + bias[e];
    }
    /* Selection: K passes of "largest not yet taken", ties to the lower
     * index. O(E*K) is trivial next to the expert work. */
    unsigned char *taken = (unsigned char *)calloc((size_t)E, 1);
    if (!taken) { free(s); free(ch); return -1; }
    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int best = -1;
        for (int e = 0; e < E; e++) {
            if (taken[e]) continue;
            if (best < 0 || ch[e] > ch[best]) best = e;
        }
        taken[best] = 1;
        top_idx[k] = best;
        top_w[k] = s[best];
        sum += s[best];
    }
    if (c->norm_topk) {
        float den = sum + 1e-20f;
        for (int k = 0; k < K; k++) top_w[k] /= den;
    }
    for (int k = 0; k < K; k++) top_w[k] *= c->scale;
    free(s); free(ch); free(taken);
    return 0;
}

int sn_moe_forward(const sn_moe_cfg *c,
                   const float *gate_w, const float *bias,
                   const float *fc1_latent, const float *fc2_latent,
                   const float *shared_up, const float *shared_down,
                   sn_expert_fetch fetch, void *fetch_user,
                   const float *x, float *y, int *top_idx_out)
{
    int K = c->topk, H = c->hidden, L = c->latent, I = c->inter, S = c->shared_inter;
    int *top = (int *)malloc(sizeof(int) * (size_t)K);
    float *tw = (float *)malloc(sizeof(float) * (size_t)K);
    float *z = (float *)malloc(sizeof(float) * (size_t)L);
    float *acc = (float *)calloc((size_t)L, sizeof(float));
    float *mid = (float *)malloc(sizeof(float) * (size_t)I);
    float *eo = (float *)malloc(sizeof(float) * (size_t)L);
    float *sm = (float *)malloc(sizeof(float) * (size_t)S);
    float *sh = (float *)malloc(sizeof(float) * (size_t)H);
    int rc = 0;
    if (!top || !tw || !z || !acc || !mid || !eo || !sm || !sh) { rc = -1; goto done; }
    if (sn_moe_route(c, gate_w, bias, x, top, tw) != 0) { rc = -2; goto done; }
    if (top_idx_out) memcpy(top_idx_out, top, sizeof(int) * (size_t)K);
    sn_matvec(fc1_latent, L, H, x, z);
    for (int k = 0; k < K; k++) {
        const float *up, *down;
        if (fetch(fetch_user, top[k], &up, &down) != 0) { rc = -3; goto done; }
        sn_matvec(up, I, L, z, mid);
        for (int i = 0; i < I; i++) mid[i] = sn_relu2(mid[i]);
        sn_matvec(down, L, I, mid, eo);
        for (int i = 0; i < L; i++) acc[i] += tw[k] * eo[i];
    }
    sn_matvec(fc2_latent, H, L, acc, y);
    sn_matvec(shared_up, S, H, x, sm);
    for (int i = 0; i < S; i++) sm[i] = sn_relu2(sm[i]);
    sn_matvec(shared_down, H, S, sm, sh);
    for (int i = 0; i < H; i++) y[i] += sh[i];
done:
    free(top); free(tw); free(z); free(acc); free(mid); free(eo); free(sm); free(sh);
    return rc;
}
