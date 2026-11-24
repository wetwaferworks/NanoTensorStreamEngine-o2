/* attn.h - Little Nemo attention decode (Nemotron-H), pure C.
 * Implements the attention entry of littlenemo/BLOCK-MATH-SPEC.md:
 * grouped-query attention, NoPE (no positional encoding), causal by
 * construction (decode attends only over positions already pushed),
 * scale 1/sqrt(head_dim). KV cache stores bf16 (widened exactly for
 * math); scores, softmax, and weighted sums accumulate in double (the
 * house discipline from the NanoQuant engines), results in fp32.
 * Dimensions are parameters, never hard-coded, so tests run tiny shapes.
 */
#ifndef NEMO_ATTN_H
#define NEMO_ATTN_H

#include <stdint.h>

/* fp32 -> bf16, round to nearest even (the exact inverse of widening
 * for every value bf16 can hold). */
uint16_t nemo_f32_to_bf16(float v);

/* KV cache for one attention layer. k and v are [max_pos][n_kv * d]
 * bf16. len is the number of positions pushed so far. */
typedef struct {
    int n_q;       /* query heads (40 in the Nano) */
    int n_kv;      /* KV heads (8); each serves n_q / n_kv query heads */
    int d;         /* head_dim (128) */
    int max_pos;
    int len;
    uint16_t *k;
    uint16_t *v;
} nemo_attn_cache;

int nemo_attn_cache_init(nemo_attn_cache *c, int n_q, int n_kv, int d,
                         int max_pos);
void nemo_attn_cache_free(nemo_attn_cache *c);

/* Append one position. k and v are fp32 [n_kv * d]. Returns 0, or -1
 * when the cache is full (nothing written). */
int nemo_attn_push(nemo_attn_cache *c, const float *k, const float *v);

/* One decode step for the position just pushed: q is fp32 [n_q * d],
 * out receives fp32 [n_q * d] (the o_proj input). Returns 0, or -1 on
 * allocation failure. NaN discipline: a NaN score propagates to the
 * output as NaN, matching the NanoQuant probes... corruption is
 * surfaced, never silently zeroed. */
int nemo_attn_decode(const nemo_attn_cache *c, const float *q, float *out);

#endif
