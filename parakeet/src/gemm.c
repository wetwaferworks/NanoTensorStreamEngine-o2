/* parakeet-c GEMM: C[M,N] = A[M,K] * W[N,K]^T + bias.
 * W is the PyTorch Linear layout (row = output feature). bf16 weights are
 * read straight from the memory-mapped safetensors file and widened in
 * registers (bf16 -> f32 is a 16-bit shift), so the encoder needs no
 * conversion pass and no extra RAM. AVX2+FMA 4x2 dot-product micro kernel,
 * split over the pool by blocks of 16 output features. */
#include "pk.h"
#include <string.h>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define PK_AVX2 1
#endif

#define NB 16

typedef struct { float *C; const float *A; const void *W; const float *bias; int M, N, K, bf16; } gemm_job;

#ifdef PK_AVX2
static inline float hsum(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehdup_ps(lo); lo = _mm_add_ps(lo, sh);
    sh = _mm_movehl_ps(sh, lo); lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}
static inline __m256 ldw_bf16(const uint16_t *p) {
    __m256i x = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p));
    return _mm256_castsi256_ps(_mm256_slli_epi32(x, 16));
}
static inline __m256 ldw_f32(const float *p) { return _mm256_loadu_ps(p); }
#endif

static inline float wat(const gemm_job *j, int n, int k) {
    if (j->bf16) return pk_bf16_to_f32(((const uint16_t *)j->W)[(size_t)n * j->K + k]);
    return ((const float *)j->W)[(size_t)n * j->K + k];
}

static void gemm_task(void *vctx, int task) {
    const gemm_job *j = vctx;
    const int K = j->K, M = j->M;
    int n0 = task * NB, n1 = n0 + NB; if (n1 > j->N) n1 = j->N;
    int m = 0;
#ifdef PK_AVX2
    const int K8 = K & ~7;
    for (; m + 4 <= M; m += 4) {
        const float *a0 = j->A + (size_t)m * K, *a1 = a0 + K, *a2 = a1 + K, *a3 = a2 + K;
        int n = n0;
        for (; n + 2 <= n1; n += 2) {
            __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00;
            if (j->bf16) {
                const uint16_t *w0 = (const uint16_t *)j->W + (size_t)n * K, *w1 = w0 + K;
                for (int k = 0; k < K8; k += 8) {
                    __m256 b0 = ldw_bf16(w0 + k), b1 = ldw_bf16(w1 + k), a;
                    a = _mm256_loadu_ps(a0 + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
                    a = _mm256_loadu_ps(a1 + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
                    a = _mm256_loadu_ps(a2 + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
                    a = _mm256_loadu_ps(a3 + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
                }
            } else {
                const float *w0 = (const float *)j->W + (size_t)n * K, *w1 = w0 + K;
                for (int k = 0; k < K8; k += 8) {
                    __m256 b0 = ldw_f32(w0 + k), b1 = ldw_f32(w1 + k), a;
                    a = _mm256_loadu_ps(a0 + k); c00 = _mm256_fmadd_ps(a, b0, c00); c01 = _mm256_fmadd_ps(a, b1, c01);
                    a = _mm256_loadu_ps(a1 + k); c10 = _mm256_fmadd_ps(a, b0, c10); c11 = _mm256_fmadd_ps(a, b1, c11);
                    a = _mm256_loadu_ps(a2 + k); c20 = _mm256_fmadd_ps(a, b0, c20); c21 = _mm256_fmadd_ps(a, b1, c21);
                    a = _mm256_loadu_ps(a3 + k); c30 = _mm256_fmadd_ps(a, b0, c30); c31 = _mm256_fmadd_ps(a, b1, c31);
                }
            }
            float s[8] = { hsum(c00), hsum(c01), hsum(c10), hsum(c11), hsum(c20), hsum(c21), hsum(c30), hsum(c31) };
            for (int k = K8; k < K; k++) {
                float b0 = wat(j, n, k), b1 = wat(j, n + 1, k);
                s[0] += a0[k] * b0; s[1] += a0[k] * b1; s[2] += a1[k] * b0; s[3] += a1[k] * b1;
                s[4] += a2[k] * b0; s[5] += a2[k] * b1; s[6] += a3[k] * b0; s[7] += a3[k] * b1;
            }
            float bb0 = j->bias ? j->bias[n] : 0.f, bb1 = j->bias ? j->bias[n + 1] : 0.f;
            float *c = j->C + (size_t)m * j->N + n;
            c[0] = s[0] + bb0; c[1] = s[1] + bb1; c += j->N;
            c[0] = s[2] + bb0; c[1] = s[3] + bb1; c += j->N;
            c[0] = s[4] + bb0; c[1] = s[5] + bb1; c += j->N;
            c[0] = s[6] + bb0; c[1] = s[7] + bb1;
        }
        for (; n < n1; n++) {
            for (int r = 0; r < 4; r++) {
                const float *a = j->A + (size_t)(m + r) * K; float s = 0;
                for (int k = 0; k < K; k++) s += a[k] * wat(j, n, k);
                j->C[(size_t)(m + r) * j->N + n] = s + (j->bias ? j->bias[n] : 0.f);
            }
        }
    }
    /* remaining rows: 1 x 2 kernel */
    for (; m < M; m++) {
        const float *a = j->A + (size_t)m * K;
        for (int n = n0; n < n1; n++) {
            __m256 c0 = _mm256_setzero_ps();
            int k = 0;
            if (j->bf16) { const uint16_t *w = (const uint16_t *)j->W + (size_t)n * K; for (; k < K8; k += 8) c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), ldw_bf16(w + k), c0); }
            else { const float *w = (const float *)j->W + (size_t)n * K; for (; k < K8; k += 8) c0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), ldw_f32(w + k), c0); }
            float s = hsum(c0);
            for (; k < K; k++) s += a[k] * wat(j, n, k);
            j->C[(size_t)m * j->N + n] = s + (j->bias ? j->bias[n] : 0.f);
        }
    }
#else
    for (; m < M; m++) {
        const float *a = j->A + (size_t)m * K;
        for (int n = n0; n < n1; n++) {
            float s = 0; for (int k = 0; k < K; k++) s += a[k] * wat(j, n, k);
            j->C[(size_t)m * j->N + n] = s + (j->bias ? j->bias[n] : 0.f);
        }
    }
#endif
}

static void gemm_run(float *C, const float *A, const void *W, const float *bias, int M, int N, int K, int bf16) {
    gemm_job j = { C, A, W, bias, M, N, K, bf16 };
    pk_parallel_for((N + NB - 1) / NB, gemm_task, &j);
}
void pk_gemm_f32w(float *C, const float *A, const float *W, const float *bias, int M, int N, int K) { gemm_run(C, A, W, bias, M, N, K, 0); }
void pk_gemm_bf16w(float *C, const float *A, const uint16_t *W, const float *bias, int M, int N, int K) { gemm_run(C, A, W, bias, M, N, K, 1); }
void pk_gemm_ref(float *C, const float *A, const float *W, const float *bias, int M, int N, int K) {
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        double s = bias ? bias[n] : 0; for (int k = 0; k < K; k++) s += (double)A[(size_t)m * K + k] * W[(size_t)n * K + k];
        C[(size_t)m * N + n] = (float)s;
    }
}
