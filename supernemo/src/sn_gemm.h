/* sn_gemm.h - dense float32 matrix products from packed weights, for prefill and the vision tower (pure C99, no locks).
 *
 * Y[n][r] = act(bias[r] + sum_k X[n][k] * W[r][k]) for N tokens at once. W is read from the pack and expanded one slab of
 * 32 rows at a time per lane (the book is loaded once per lane per call), so a weight matrix is expanded once for the whole
 * batch instead of once per token. AVX2/FMA kernels are chosen at run time with a scalar fallback. Each output is computed
 * the same way whatever the lane count, so results are bit-identical on 1 or 12 lanes.
 *
 * Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#ifndef SN_GEMM_H
#define SN_GEMM_H

#include "sn_pack.h"
#include "../../littlenemo/src/pool.h"

typedef struct sn_gemm sn_gemm;

/* max_cols: the largest K of any matrix it will multiply. nthreads = lanes of the borrowed pool. */
sn_gemm *sn_gemm_open(const sn_pack *p, nemo_pool *pool, int nthreads, size_t max_cols);
void sn_gemm_close(sn_gemm *g);

/* act: 0 none, 1 GELU (erf), 2 ReLU squared. bias may be NULL. K must be a multiple of 8.
 * Returns 0; -1 read failure; -2 bad shape. */
int sn_gemm_linear(sn_gemm *g, const sn_tensor *t, const float *bias, const float *X, int N, int K, int R, float *Y, int act);

/* The kernel on an already-expanded matrix W[rows][K]: writes Y[n * R + r0 + r]. */
void sn_gemm_rows(const float *W, int rows, int K, const float *X, int N, const float *bias, int r0, int R, float *Y, int act);

/* 1 when sn_gemm_rows runs its AVX2/FMA kernel on this CPU, 0 for the scalar one (the GPU expert stream mirrors whichever). */
int sn_gemm_simd(void);

/* Small helpers shared with the vision attention (n % 8 == 0 for the AVX2 path). */
float sn_dot(const float *a, const float *b, int n);
void sn_axpy(float *y, float a, const float *x, int n);

#endif
