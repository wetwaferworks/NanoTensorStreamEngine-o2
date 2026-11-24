/* test_guards.c - the fix-pass guards from CODE-REVIEW-2026-11-24.md:
 * geometry validation at the mamba entry points, the gated norm's
 * anti-hang floor, and the attention NaN-propagation discipline.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/attn.h"
#include "../src/mamba.h"

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

int main(void) {
    nemo_mamba_state s;
    expect_true("state_init rejects groups 0",
                nemo_mamba_state_init(&s, 128, 80, 128, 0) == -1);
    expect_true("state_init rejects non-dividing groups",
                nemo_mamba_state_init(&s, 6, 4, 2, 4) == -1);
    expect_true("state_init rejects heads 0",
                nemo_mamba_state_init(&s, 0, 80, 128, 8) == -1);
    expect_true("state_init accepts Nano geometry",
                nemo_mamba_state_init(&s, 128, 80, 128, 8) == 0);
    nemo_mamba_state_free(&s);

    /* Tiny decode: heads 2, head_dim 4 (inner 8), state 2, groups 1,
     * conv_dim 8 + 2*1*2 = 12, proj = inner 8 + conv 12 + heads 2. */
    nemo_mamba_state t;
    if (nemo_mamba_state_init(&t, 2, 4, 2, 1) != 0) {
        printf("FAIL tiny state init\n");
        return 1;
    }
    float proj[22] = {0};
    float conv_w[48] = {0}, conv_b[12] = {0};
    float a_log[2] = {0}, d[2] = {1, 1}, dtb[2] = {0};
    float nw[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    float out[8];
    expect_true("decode rejects norm_groups 0",
                nemo_mamba_decode(&t, proj, 0, conv_w, conv_b, a_log, d,
                                  dtb, nw, 0, 1e-5f, out) == -1);
    expect_true("decode rejects non-dividing norm_groups",
                nemo_mamba_decode(&t, proj, 0, conv_w, conv_b, a_log, d,
                                  dtb, nw, 3, 1e-5f, out) == -1);
    expect_true("decode accepts dividing norm_groups",
                nemo_mamba_decode(&t, proj, 0, conv_w, conv_b, a_log, d,
                                  dtb, nw, 2, 1e-5f, out) == 0);
    nemo_mamba_state_free(&t);

    /* The gated norm must return, not hang, on group_size 0. Reaching
     * the next statement is the check. */
    float x[4] = {1, 2, 3, 4}, z[4] = {0}, w[4] = {1, 1, 1, 1}, o[4];
    nemo_rmsnorm_gated(x, z, w, o, 4, 0, 1e-5f);
    expect_true("rmsnorm_gated group_size 0 returns", 1);

    /* Attention: a NaN key must poison the output, not zero it. */
    nemo_attn_cache c;
    if (nemo_attn_cache_init(&c, 1, 1, 2, 4) != 0) {
        printf("FAIL attn init\n");
        return 1;
    }
    const float k_nan[2] = {NAN, 0.0f}, v[2] = {1.0f, 2.0f};
    const float q[2] = {1.0f, 0.0f};
    float ao[2];
    if (nemo_attn_push(&c, k_nan, v) != 0) {
        printf("FAIL attn push\n");
        return 1;
    }
    expect_true("attn decode returns 0",
                nemo_attn_decode(&c, q, ao) == 0);
    expect_true("attn NaN propagates", isnan(ao[0]) && isnan(ao[1]));
    nemo_attn_cache_free(&c);

    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
