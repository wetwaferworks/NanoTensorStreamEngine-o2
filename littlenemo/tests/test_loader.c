/* test_loader.c - loader and linear checks.
 * Part 1 is synthetic (hand-computed). Part 2 runs against a real
 * checkpoint directory given as argv[1] and prints fingerprints that
 * become regression anchors for the weight path.
 */
#include "../src/loader.h"
#include "../src/mamba.h"
#include "../src/nn.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
static void check(const char *name, double got, double want, double tol) {
    /* !(<=) rather than (>): a NaN got must FAIL, not slip past
     * the comparison (Codex Astra review M-R3, 2026-11-25). */
    if (!(fabs(got - want) <= tol)) {
        printf("FAIL %-28s got %.7f want %.7f\n", name, got, want);
        fails++;
    } else {
        printf("ok   %-28s %.7f\n", name, got);
    }
}

static uint16_t f32_to_bf16(float f) {
    /* round-to-nearest-even, the standard f32 -> bf16 conversion */
    union { float f; uint32_t u; } v = { f };
    if ((v.u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u; /* NaN -> bf16 quiet NaN (Archie P0-5, guard from littlenemo/src/attn.c) */
    v.u += 0x7FFFu + ((v.u >> 16) & 1u);
    return (uint16_t)(v.u >> 16);
}

int main(int argc, char **argv) {
    /* Synthetic linear: W = [[1,2],[3,4]], x = [5,6] -> y = [17,39]. */
    {
        uint16_t W[4] = { f32_to_bf16(1), f32_to_bf16(2), f32_to_bf16(3), f32_to_bf16(4) };
        float x[2] = {5, 6}, y[2];
        nemo_linear_bf16(W, x, y, 2, 2);
        check("linear [0]", y[0], 17.0, 1e-6);
        check("linear [1]", y[1], 39.0, 1e-6);
        /* bf16(0.1) widened is exactly 0.10009765625 */
        uint16_t W1[1] = { f32_to_bf16(0.1f) };
        float x1[1] = {1.0f}, y1[1];
        nemo_linear_bf16(W1, x1, y1, 1, 1);
        check("linear bf16 0.1", y1[0], 0.10009765625, 1e-9);
    }

    if (argc < 2) {
        printf("RESULT: %s (synthetic only)\n", fails ? "FAIL" : "ALL PASS");
        return fails ? 1 : 0;
    }

    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        printf("FAIL checkpoint open %s\n", argv[1]);
        return 1;
    }
    printf("info indexed %d tensors from %s\n", cp.n, argv[1]);

    const nemo_tensor_info *alog = nemo_find(&cp, "backbone.layers.0.mixer.A_log");
    if (!alog) { printf("FAIL A_log not found\n"); return 1; }
    check("A_log count", (double)alog->count, 128.0, 0.0);
    float *a = nemo_load_f32(&cp, alog);
    if (!a) { printf("FAIL A_log load\n"); return 1; }
    double mn = a[0], mx = a[0], mean = 0.0;
    for (int i = 0; i < 128; i++) {
        if (a[i] < mn) mn = a[i];
        if (a[i] > mx) mx = a[i];
        mean += a[i];
    }
    printf("info A_log min %.6f max %.6f mean %.6f fnv %016llx\n",
           mn, mx, mean / 128.0, (unsigned long long)nemo_tensor_fnv(&cp, alog));
    free(a);

    const nemo_tensor_info *q = nemo_find(&cp, "backbone.layers.21.mixer.q_proj.weight");
    if (!q) { printf("FAIL q_proj(21) not found\n"); return 1; }
    check("q_proj(21) shape0", (double)q->shape[0], 5120.0, 0.0);
    check("q_proj(21) shape1", (double)q->shape[1], 4480.0, 0.0);
    float *qw = nemo_load_f32(&cp, q);
    if (!qw) { printf("FAIL q_proj load\n"); return 1; }
    double ss = 0.0;
    for (uint64_t i = 0; i < q->count; i++) ss += (double)qw[i] * qw[i];
    printf("info q_proj(21) rms %.6f fnv %016llx\n",
           sqrt(ss / (double)q->count), (unsigned long long)nemo_tensor_fnv(&cp, q));
    free(qw);

    nemo_checkpoint_close(&cp);
    printf(fails ? "RESULT: %d FAILURES\n" : "RESULT: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
