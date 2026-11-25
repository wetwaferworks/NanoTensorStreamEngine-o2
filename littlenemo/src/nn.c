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
    /* Dispatch through the auto rows call so the opt-in AVX2 kernel
     * is reachable from every projection, including single-thread
     * runs and sub-4096 outputs (Archie review P0-1, 2026-11-24:
     * this entry point previously called the scalar rows directly,
     * which silently bypassed the kernel on those paths). */
    nemo_linear_bf16_rows_auto(W, x, y, 0, out, in);
}

/* ---- SIMD dispatch, opt-in only. See nn.h for the contract.
 * This translation unit is compiled at the BASELINE
 * instruction set in both build scripts (Codex Astra review
 * H9, 2026-11-25): the scalar kernel above, this dispatcher,
 * and the CPUID gate must be safe on a CPU with no AVX at
 * all, because the scalar path is the fallback such a CPU
 * takes. The AVX2 kernel itself lives in nn_simd.c, the one
 * unit built with -mavx2. The CPUID check below executes no
 * AVX instructions (cpuid, plus xgetbv only after the OSXSAVE
 * bit says it is safe), so it is baseline-clean too. */

static int simd_opt_in = 0;

void nemo_set_simd(int on) { simd_opt_in = on ? 1 : 0; }

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>

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
#else
int nemo_simd_available(void) { return 0; }
#endif

int nemo_simd_active(void) { return simd_opt_in && nemo_simd_available(); }

void nemo_linear_bf16_rows_auto(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in) {
    if (nemo_simd_active())
        nemo_linear_bf16_rows_simd(W, x, y, begin, end, in);
    else
        nemo_linear_bf16_rows(W, x, y, begin, end, in);
}
