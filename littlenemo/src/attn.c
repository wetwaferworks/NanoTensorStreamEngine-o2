/* attn.c - see attn.h and littlenemo/BLOCK-MATH-SPEC.md. */
#include "attn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "mamba.h" /* nemo_bf16_to_f32 */

uint16_t nemo_f32_to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, sizeof u);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0; /* NaN -> bf16 quiet NaN (exp all-ones); 0x7e00 would be a finite 2^125 */
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7fffu + lsb;
    return (uint16_t)(u >> 16);
}

int nemo_attn_cache_init(nemo_attn_cache *c, int n_q, int n_kv, int d,
                         int max_pos) {
    if (n_q <= 0 || n_kv <= 0 || d <= 0 || max_pos <= 0) return -1;
    if (n_kv > n_q || n_q % n_kv != 0) return -1;
    c->n_q = n_q;
    c->n_kv = n_kv;
    c->d = d;
    c->max_pos = max_pos;
    c->len = 0;
    size_t n = (size_t)max_pos * (size_t)n_kv * (size_t)d;
    c->k = (uint16_t *)malloc(n * sizeof(uint16_t));
    c->v = (uint16_t *)malloc(n * sizeof(uint16_t));
    if (!c->k || !c->v) {
        free(c->k);
        free(c->v);
        c->k = c->v = NULL;
        return -1;
    }
    return 0;
}

void nemo_attn_cache_free(nemo_attn_cache *c) {
    free(c->k);
    free(c->v);
    c->k = c->v = NULL;
    c->len = 0;
}

int nemo_attn_push(nemo_attn_cache *c, const float *k, const float *v) {
    if (c->len >= c->max_pos) return -1;
    size_t base = (size_t)c->len * (size_t)c->n_kv * (size_t)c->d;
    int n = c->n_kv * c->d;
    for (int i = 0; i < n; i++) {
        c->k[base + (size_t)i] = nemo_f32_to_bf16(k[i]);
        c->v[base + (size_t)i] = nemo_f32_to_bf16(v[i]);
    }
    c->len++;
    return 0;
}

int nemo_attn_decode(const nemo_attn_cache *c, const float *q, float *out) {
    const int group = c->n_q / c->n_kv;
    const double scale = 1.0 / sqrt((double)c->d);
    const int row = c->n_kv * c->d;
    double *outd = (double *)malloc((size_t)c->d * sizeof(double));
    if (!outd) return -1;
    for (int h = 0; h < c->n_q; h++) {
        const int kv = h / group;
        const float *qh = q + (size_t)h * (size_t)c->d;
        /* Pass 1: the softmax max, scores in position order. */
        double mx = -INFINITY;
        for (int t = 0; t < c->len; t++) {
            const uint16_t *kt =
                c->k + (size_t)t * (size_t)row + (size_t)kv * (size_t)c->d;
            double s = 0.0;
            for (int i = 0; i < c->d; i++)
                s += (double)qh[i] * (double)nemo_bf16_to_f32(kt[i]);
            s *= scale;
            if (s > mx) mx = s;
        }
        /* Pass 2: probabilities and the weighted sum, same order, so
         * the result is deterministic. */
        double denom = 0.0;
        for (int i = 0; i < c->d; i++) outd[i] = 0.0;
        for (int t = 0; t < c->len; t++) {
            const uint16_t *kt =
                c->k + (size_t)t * (size_t)row + (size_t)kv * (size_t)c->d;
            const uint16_t *vt =
                c->v + (size_t)t * (size_t)row + (size_t)kv * (size_t)c->d;
            double s = 0.0;
            for (int i = 0; i < c->d; i++)
                s += (double)qh[i] * (double)nemo_bf16_to_f32(kt[i]);
            double p = exp(s * scale - mx);
            denom += p;
            for (int i = 0; i < c->d; i++)
                outd[i] += p * (double)nemo_bf16_to_f32(vt[i]);
        }
        float *oh = out + (size_t)h * (size_t)c->d;
        if (denom != denom) {
            /* NaN in, NaN out: poison propagates so the verification
             * harness sees it, exactly as the NQDC probes ruled. */
            for (int i = 0; i < c->d; i++) oh[i] = (float)NAN;
        } else if (denom > 0.0) {
            for (int i = 0; i < c->d; i++) oh[i] = (float)(outd[i] / denom);
        } else {
            for (int i = 0; i < c->d; i++) oh[i] = 0.0f;
        }
    }
    free(outd);
    return 0;
}
