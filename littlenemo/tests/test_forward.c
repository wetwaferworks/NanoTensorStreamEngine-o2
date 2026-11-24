/* test_forward.c - truncated-stack smoke over REAL checkpoint weights.
 * Runs tokens through layers 0..14 (all three block types, including
 * the attention layer at 14) using the shards present today, and checks
 * the plumbing properties: every hidden value finite, the same token
 * sequence after reset reproduces the hidden state bit for bit, and a
 * different token gives a different state. This is NOT a model-quality
 * test: the stack is truncated and there is no final norm or lm_head.
 *
 *     test_forward <checkpoint-dir>
 */
#include <math.h>
#include <stdio.h>
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
        printf("usage: test_forward <checkpoint-dir>\n");
        return 1;
    }
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) {
        printf("FAIL checkpoint open\n");
        return 1;
    }
    nemo_model m;
    if (nemo_model_init(&m, &cp, 15, 16) != 0) {
        printf("FAIL model init\n");
        return 1;
    }
    static float h1[NEMO_HIDDEN], h2[NEMO_HIDDEN], h3[NEMO_HIDDEN];
    const int seq[3] = {42, 1000, 7};
    int rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&m, seq[i], h1);
    expect_true("three tokens through 15 layers", rc == 0);
    int finite = 1;
    double mag = 0.0;
    for (int i = 0; i < NEMO_HIDDEN; i++) {
        if (!isfinite(h1[i])) finite = 0;
        mag += (double)h1[i] * h1[i];
    }
    expect_true("hidden state finite", finite);
    expect_true("hidden state nonzero", mag > 0.0);
    printf("     hidden RMS %.6f\n", sqrt(mag / NEMO_HIDDEN));

    nemo_model_reset(&m);
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&m, seq[i], h2);
    expect_true("rerun after reset", rc == 0);
    expect_true("deterministic across reset",
                memcmp(h1, h2, sizeof h1) == 0);

    nemo_model_reset(&m);
    rc = nemo_model_step_hidden(&m, 43, h3);
    expect_true("different token runs", rc == 0);
    expect_true("different token, different state",
                memcmp(h1, h3, sizeof h1) != 0);

    nemo_model_free(&m);
    nemo_checkpoint_close(&cp);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
