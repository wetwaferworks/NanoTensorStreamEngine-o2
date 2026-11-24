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
 * so splitting rows across lanes is bit-identical to the full call;
 * nemo_linear_bf16 is exactly rows(0, out). */
void nemo_linear_bf16_rows(const uint16_t *W, const float *x, float *y,
                           int begin, int end, int in);

#endif
