/* nn_simd.c - the AVX2 twin of the bf16 row kernel, opt-in
 * only. This is the ONE translation unit the build scripts
 * compile with -mavx2 on x86-64 (Codex Astra review H9,
 * 2026-11-25): the scalar kernel, the dispatcher, and the
 * CPUID gate live in nn.c at the baseline instruction set,
 * so the fallback path contains no AVX instructions and a
 * CPU without AVX support can take it safely. Before the
 * split, nn.c carried everything under -mavx2 and disassembly
 * showed VEX-encoded instructions inside the "scalar" loop
 * itself. See nn.h for the kernel contract. */
#include "nn.h"

#include <stddef.h>

#include "mamba.h" /* nemo_bf16_to_f32 */

#if defined(__AVX2__) && (defined(__x86_64__) || defined(__i386__))
#define NEMO_HAVE_AVX2 1
#include <immintrin.h>

static unsigned long g_kernel_calls = 0;

static void rows_avx2(const uint16_t *W, const float *x, float *y,
                      int begin, int end, int in) {
    for (int o = begin; o < end; o++) {
        const uint16_t *row = W + (size_t)o * (size_t)in;
        __m256d acc = _mm256_setzero_pd();
        int i = 0;
        for (; i + 8 <= in; i += 8) {
            /* 8 bf16 values widen to 8 fp32 (a 16-bit shift), then
             * to two vectors of 4 doubles; x converts alongside. */
            __m128i h = _mm_loadu_si128((const __m128i *)(row + i));
            __m128i lo32 = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
            __m128i hi32 = _mm_slli_epi32(
                _mm_cvtepu16_epi32(_mm_srli_si128(h, 8)), 16);
            __m256d w0 = _mm256_cvtps_pd(_mm_castsi128_ps(lo32));
            __m256d w1 = _mm256_cvtps_pd(_mm_castsi128_ps(hi32));
            __m256d x0 = _mm256_cvtps_pd(_mm_loadu_ps(x + i));
            __m256d x1 = _mm256_cvtps_pd(_mm_loadu_ps(x + i + 4));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(w0, x0));
            acc = _mm256_add_pd(acc, _mm256_mul_pd(w1, x1));
        }
        /* Fixed combine order, then the scalar tail: deterministic. */
        double a[4];
        _mm256_storeu_pd(a, acc);
        double sum = (a[0] + a[1]) + (a[2] + a[3]);
        for (; i < in; i++) sum += (double)nemo_bf16_to_f32(row[i]) * x[i];
        y[o] = (float)sum;
    }
}
#endif

/* Test observation hook (Codex Astra review M-R2, 2026-11-25):
 * how many times the AVX2 kernel has actually been entered.
 * Production dispatch never reads it. The SIMD suite needs it
 * because no output comparison can prove dispatch: the kernel
 * is bit-exact against the scalar path by design, so "the
 * entry point selected the kernel" is observable only here. */
unsigned long nemo_simd_kernel_calls(void) {
#if defined(NEMO_HAVE_AVX2)
    return g_kernel_calls;
#else
    return 0;
#endif
}

void nemo_linear_bf16_rows_simd(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in) {
    if (in <= 0) return;
#if defined(NEMO_HAVE_AVX2)
    if (nemo_simd_available()) {
        g_kernel_calls++;
        rows_avx2(W, x, y, begin, end, in);
        return;
    }
#endif
    nemo_linear_bf16_rows(W, x, y, begin, end, in);
}
