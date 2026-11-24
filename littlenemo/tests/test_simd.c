/* test_simd.c - the opt-in SIMD path, held to two standards.
 * (1) It must be deterministic: the same SIMD run across a reset
 *     is memcmp-identical to itself.
 * (2) It must stay faithful to the scalar path: the hidden states
 *     may drift in the last bits (the accumulation order differs
 *     by design), and this suite measures the drift on real
 *     weights and requires it to stay tiny.
 * The scalar path remains the golden reference; SIMD is a speed
 * mode, never a silent replacement.
 *
 *     test_simd <checkpoint-dir> [layers]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/forward.h"
#include "../src/nn.h"

static int checks = 0, failures = 0;

static void expect_true(const char *name, int cond) {
    checks++;
    if (!cond) {
        failures++;
        printf("FAIL %s\n", name);
    } else {
        printf("ok   %s\n", name);
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: test_simd <checkpoint-dir> [layers]\n");
        return 1;
    }
    int layers = argc > 2 ? atoi(argv[2]) : 15;
    expect_true("AVX2 kernel available on this machine",
                nemo_simd_available() == 1);
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        printf("FAIL checkpoint open\n");
        return 1;
    }
    nemo_model a, b;
    if (nemo_model_init(&a, &cp, layers, 16) != 0 ||
        nemo_model_init(&b, &cp, layers, 16) != 0) {
        printf("FAIL model init\n");
        return 1;
    }
    static float ha[NEMO_HIDDEN], hb[NEMO_HIDDEN];
    const int seq[3] = {42, 1000, 7};
    int rc = 0;
    nemo_set_simd(0);
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&a, seq[i], ha);
    nemo_set_simd(1);
    expect_true("simd activates when asked", nemo_simd_active() == 1);
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&b, seq[i], hb);
    expect_true("three tokens through both paths", rc == 0);

    double maxd = 0.0, sumsq = 0.0;
    for (int i = 0; i < NEMO_HIDDEN; i++) {
        double d = fabs((double)ha[i] - (double)hb[i]);
        if (d > maxd) maxd = d;
        sumsq += d * d;
    }
    printf("     drift vs scalar: max abs %.3e, RMS %.3e\n", maxd,
           sqrt(sumsq / NEMO_HIDDEN));
    expect_true("drift stays tiny (< 1e-5 max abs)", maxd < 1e-5);

    nemo_model_reset(&b);
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&b, seq[i], hb);
    expect_true("simd rerun after reset", rc == 0);
    static float hb2[NEMO_HIDDEN];
    nemo_model_reset(&b);
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&b, seq[i], hb2);
    expect_true("simd deterministic across reset",
                rc == 0 && memcmp(hb, hb2, sizeof hb) == 0);

    nemo_set_simd(0);
    nemo_model_free(&a);
    nemo_model_free(&b);
    nemo_checkpoint_close(&cp);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
