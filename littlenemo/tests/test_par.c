/* test_par.c - the dual path must not change a single bit.
 * Steps two models over the same truncated stack on real weights:
 * one with a single lane (the serial reference) and one with N lanes
 * (row-split projections plus the layer-ahead prep thread), and
 * requires the hidden states to be memcmp-identical after every
 * token. Also requires the parallel model to be deterministic
 * across a reset, and a different token to diverge.
 *
 *     test_par <checkpoint-dir> [lanes] [layers]
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/forward.h"

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
        printf("usage: test_par <checkpoint-dir> [lanes] [layers]\n");
        return 1;
    }
    int lanes = argc > 2 ? atoi(argv[2]) : 4;
    int layers = argc > 3 ? atoi(argv[3]) : 15;
    if (lanes < 2) lanes = 2;
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        printf("FAIL checkpoint open\n");
        return 1;
    }
    nemo_model a, b;
    if (nemo_model_init_ex(&a, &cp, layers, 16, 1) != 0) {
        printf("FAIL serial model init\n");
        return 1;
    }
    if (nemo_model_init_ex(&b, &cp, layers, 16, lanes) != 0) {
        printf("FAIL parallel model init\n");
        return 1;
    }
    expect_true("parallel pool reports its lanes", b.nthreads == lanes);

    static float ha[NEMO_HIDDEN], hb[NEMO_HIDDEN];
    const int seq[3] = {42, 1000, 7};
    int same = 1, rc = 0;
    for (int i = 0; i < 3; i++) {
        rc |= nemo_model_step_hidden(&a, seq[i], ha);
        rc |= nemo_model_step_hidden(&b, seq[i], hb);
        if (memcmp(ha, hb, sizeof ha) != 0) same = 0;
    }
    expect_true("three tokens through both models", rc == 0);
    expect_true("parallel hidden bit-identical to serial", same);
    double mag = 0.0;
    for (int i = 0; i < NEMO_HIDDEN; i++) mag += (double)ha[i] * ha[i];
    printf("     hidden RMS %.6f (serial and parallel)\n",
           sqrt(mag / NEMO_HIDDEN));

    nemo_model_reset(&b);
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&b, seq[i], hb);
    expect_true("parallel rerun after reset", rc == 0);
    expect_true("parallel deterministic across reset",
                memcmp(ha, hb, sizeof ha) == 0);

    nemo_model_reset(&b);
    rc = nemo_model_step_hidden(&b, 43, hb);
    expect_true("different token runs on parallel", rc == 0);
    expect_true("different token, different state",
                memcmp(ha, hb, sizeof ha) != 0);

    nemo_model_free(&a);
    nemo_model_free(&b);
    nemo_checkpoint_close(&cp);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
