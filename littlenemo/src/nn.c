/* nn.c - see nn.h. */
#include "nn.h"

#include <stddef.h>

#include "mamba.h" /* nemo_bf16_to_f32 */

void nemo_linear_bf16(const uint16_t *W, const float *x, float *y, int out, int in) {
    if (out <= 0 || in <= 0) return;
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * (size_t)in;
        double acc = 0.0;
        for (int i = 0; i < in; i++) acc += (double)nemo_bf16_to_f32(row[i]) * x[i];
        y[o] = (float)acc;
    }
}
