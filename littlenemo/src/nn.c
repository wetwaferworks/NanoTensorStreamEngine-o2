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
