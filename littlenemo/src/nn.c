/* nn.c - see nn.h. */
#include "nn.h"

#include <stddef.h>

#include "mamba.h" /* nemo_bf16_to_f32 */

void nemo_linear_bf16_rows(const uint16_t *W, const float *x, float *y,
                           int begin, int end, int in) {
    if (in <= 0) return;
    for (int o = begin; o < end; o++) {
        const uint16_t *row = W + (size_t)o * (size_t)in;
        double acc = 0.0;
        for (int i = 0; i < in; i++) acc += (double)nemo_bf16_to_f32(row[i]) * x[i];
        y[o] = (float)acc;
    }
}

void nemo_linear_bf16(const uint16_t *W, const float *x, float *y, int out, int in) {
    if (out <= 0) return;
    nemo_linear_bf16_rows(W, x, y, 0, out, in);
}

/* ---- SIMD (AVX2) twin, opt-in only. See nn.h for the contract.
 * This file is the one translation unit the build scripts compile
 * with -mavx2 on x86-64; the rest of the engine stays at the
 * baseline instruction set so the binary still runs on older
 * hardware, where the CPUID check below keeps it on the scalar
 * path. */

static int simd_opt_in = 0;

void nemo_set_simd(int on) { simd_opt_in = on ? 1 : 0; }

#if defined(__AVX2__) && (defined(__x86_64__) || defined(__i386__))
#define NEMO_HAVE_AVX2 1
#include <cpuid.h>
#include <immintrin.h>

int nemo_simd_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached;
    cached = 0;
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) &&
        (ecx & bit_OSXSAVE) && (ecx & bit_AVX)) {
        unsigned int lo, hi;
        __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
        if ((lo & 0x6) == 0x6 && /* xmm + ymm state enabled */
            __get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) &&
            (ebx & bit_AVX2))
            cached = 1;
    }
    return cached;
}

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
#else
int nemo_simd_available(void) { return 0; }
#endif

int nemo_simd_active(void) { return simd_opt_in && nemo_simd_available(); }

void nemo_linear_bf16_rows_simd(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in) {
    if (in <= 0) return;
#if defined(NEMO_HAVE_AVX2)
    if (nemo_simd_available()) {
        rows_avx2(W, x, y, begin, end, in);
        return;
    }
#endif
    nemo_linear_bf16_rows(W, x, y, begin, end, in);
}

void nemo_linear_bf16_rows_auto(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in) {
    if (nemo_simd_active())
        nemo_linear_bf16_rows_simd(W, x, y, begin, end, in);
    else
        nemo_linear_bf16_rows(W, x, y, begin, end, in);
}
