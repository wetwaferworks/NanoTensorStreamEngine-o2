/* nn.h - Little Nemo dense ops, pure C. */
#ifndef NEMO_NN_H
#define NEMO_NN_H

#include <stdint.h>

/* y[o] = sum_i W[o * in + i] * x[i], W stored bf16 row-major [out x in],
 * accumulation in double (the house discipline from the NanoQuant engines),
 * result in fp32. No bias: Nemotron-H projections are bias-free. */
void nemo_linear_bf16(const uint16_t *W, const float *x, float *y, int out, int in);

/* The same math for rows [begin, end) only. A row's dot product is
 * accumulated in the same order whatever range it is computed in,
 * so splitting rows across lanes is bit-identical to the full call.
 * nemo_linear_bf16 is exactly rows_auto(0, out): the scalar rows
 * unless SIMD is opted in and available (Archie review P0-1). */
void nemo_linear_bf16_rows(const uint16_t *W, const float *x, float *y,
                           int begin, int end, int in);

/* SIMD twin of the rows call. Same products, accumulated in a fixed
 * blocked order (groups of four doubles), so it is deterministic
 * run to run and machine to machine wherever it is used, but it may
 * differ from the scalar order in the last bits. It is never
 * selected silently: nemo_set_simd(1) opts in (the engine's --simd
 * flag, or simd = true in littlenemo.ini), the READY line reports
 * the active state, and without AVX2 in the build and on the CPU
 * the call is exactly the scalar rows call. */
void nemo_linear_bf16_rows_simd(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in);

/* The dispatching rows call the forward pass uses: the SIMD twin
 * when opted in and available, the scalar rows otherwise. */
void nemo_linear_bf16_rows_auto(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in);

void nemo_set_simd(int on);
int nemo_simd_available(void); /* an AVX2 kernel exists and the CPU has it */
int nemo_simd_active(void);    /* opted in and available */

#endif
