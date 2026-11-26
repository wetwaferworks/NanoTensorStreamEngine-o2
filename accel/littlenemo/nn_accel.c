/* nn_accel.c - Little Nemo projection routed through an accel lane.
 *
 * Co-implemented by Cortana. Little Nemo itself is Elfie's; its
 * sources are NOT modified. This file is linked INSTEAD of the two
 * public dispatch entry points of nn.c, which the accel build
 * compiles a second time with those two names renamed
 * (-Dnemo_linear_bf16=nemo_cpu_linear_bf16, and the same for
 * _rows_auto), so the golden scalar kernel stays exactly Elfie's
 * code and remains the fallback for every call.
 *
 * Routing: full projections (nemo_linear_bf16) at or above
 * NEMO_ACCEL_MIN_MACS go to the GPU when the switch is on and a
 * lane is up; ranged calls from the thread pool
 * (nemo_linear_bf16_rows_auto) always stay on the CPU, so the
 * threaded engine never calls a device from several threads.
 * Any GPU refusal or fault: the CPU kernel computes the call.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "../../littlenemo/src/nn.h"
#include "../include/ntse_accel.h"

void nemo_cpu_linear_bf16(const uint16_t *W, const float *x, float *y, int out, int in);
void nemo_cpu_linear_bf16_rows_auto(const uint16_t *W, const float *x, float *y,
                                    int begin, int end, int in);

#define NEMO_ACCEL_MIN_MACS (1 << 20)

static ntse_accel *g_acc = NULL;
static int g_on = 0;
static unsigned long long g_gpu_calls = 0, g_cpu_calls = 0;

void nemo_accel_bind(ntse_accel *a, int on) { g_acc = a; g_on = on; }
unsigned long long nemo_accel_gpu_calls(void) { return g_gpu_calls; }
unsigned long long nemo_accel_cpu_calls(void) { return g_cpu_calls; }
void nemo_accel_reset_counts(void) { g_gpu_calls = g_cpu_calls = 0; }

void nemo_linear_bf16(const uint16_t *W, const float *x, float *y, int out, int in)
{
    if (g_on && g_acc && out > 0 && in > 0
        && (long long)out * in >= NEMO_ACCEL_MIN_MACS) {
        ntse_lane used = NTSE_LANE_CPU;
        if (ntse_accel_linear_bf16(g_acc, W, x, y, out, in, &used) == NTSE_OK
            && used != NTSE_LANE_CPU) {
            g_gpu_calls++;
            return;
        }
    }
    g_cpu_calls++;
    nemo_cpu_linear_bf16(W, x, y, out, in);
}

void nemo_linear_bf16_rows_auto(const uint16_t *W, const float *x, float *y,
                                int begin, int end, int in)
{
    nemo_cpu_linear_bf16_rows_auto(W, x, y, begin, end, in);
}
