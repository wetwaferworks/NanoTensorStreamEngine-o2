/* test_attn.c - hand-computed checks for the attention decode kernel.
 * Tiny shapes (2 query heads sharing 1 KV head, head_dim 2) with closed
 * forms: scores are set to ln 2 and ln 3 via q = sqrt(2) * ln(k) on one
 * axis, so the softmax weights are exact rationals (2/3, 1/3), (1/4, 3/4).
 * All cached values are bf16-exact small numbers.
 */
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "../src/attn.h"
#include "../src/mamba.h" /* nemo_bf16_to_f32 */

static int checks = 0, failures = 0;

static void expect_close(const char *name, double got, double want,
                         double tol) {
    checks++;
    /* !(<=) rather than (>): a NaN got must FAIL, not slip past
     * the comparison (Codex Astra review M-R3, 2026-11-25). */
    if (!(fabs(got - want) <= tol)) {
        failures++;
        printf("FAIL %s: got %.9g want %.9g\n", name, got, want);
    } else {
        printf("ok   %s = %.9g\n", name, got);
    }
}

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
    /* bf16 conversion: exact values round-trip. */
    const float exact[] = {1.0f, 0.5f, -2.0f, 3.0f, 5.0f, 0.25f, -0.75f};
    int rt = 1;
    for (unsigned i = 0; i < sizeof exact / sizeof *exact; i++)
        if (nemo_bf16_to_f32(nemo_f32_to_bf16(exact[i])) != exact[i]) rt = 0;
    expect_true("bf16 exact round trips", rt);
    expect_close("bf16 1.1 rounds to nearest",
                 nemo_bf16_to_f32(nemo_f32_to_bf16(1.1f)), 1.1015625, 1e-9);

    /* Geometry guards. */
    nemo_attn_cache bad;
    expect_true("init rejects non-divisible heads",
                nemo_attn_cache_init(&bad, 3, 2, 2, 4) == -1);

    nemo_attn_cache c;
    if (nemo_attn_cache_init(&c, 2, 1, 2, 4) != 0) {
        printf("FAIL cache init\n");
        return 1;
    }
    /* Position 1: k = (1, 0), v = (2, 0). */
    const float k1[2] = {1.0f, 0.0f}, v1[2] = {2.0f, 0.0f};
    expect_true("push 1", nemo_attn_push(&c, k1, v1) == 0);

    /* One position: the output is v itself, both heads. */
    const float q_any[4] = {1.0f, 1.0f, -3.0f, 0.5f};
    float out[4];
    nemo_attn_decode(&c, q_any, out);
    expect_close("single pos head0 x", out[0], 2.0, 1e-6);
    expect_close("single pos head0 y", out[1], 0.0, 1e-6);
    expect_close("single pos head1 x", out[2], 2.0, 1e-6);
    expect_close("single pos head1 y", out[3], 0.0, 1e-6);

    /* Position 2: k = (0, 1), v = (0, 4). */
    const float k2[2] = {0.0f, 1.0f}, v2[2] = {0.0f, 4.0f};
    expect_true("push 2", nemo_attn_push(&c, k2, v2) == 0);

    /* Head 0: q = (sqrt2 ln2, 0) -> scores (ln2, 0) -> weights (2/3, 1/3)
     * -> out = (2*(2/3), 4*(1/3)) = (4/3, 4/3).
     * Head 1: q = (0, sqrt2 ln3) -> scores (0, ln3) -> weights (1/4, 3/4)
     * -> out = (2*(1/4), 4*(3/4)) = (1/2, 3). */
    const float q[4] = {(float)(sqrt(2.0) * log(2.0)), 0.0f, 0.0f,
                        (float)(sqrt(2.0) * log(3.0))};
    nemo_attn_decode(&c, q, out);
    expect_close("gqa head0 x", out[0], 4.0 / 3.0, 1e-6);
    expect_close("gqa head0 y", out[1], 4.0 / 3.0, 1e-6);
    expect_close("gqa head1 x", out[2], 0.5, 1e-6);
    expect_close("gqa head1 y", out[3], 3.0, 1e-6);

    /* Full cache refuses cleanly. */
    const float k3[2] = {1.0f, 1.0f}, v3[2] = {1.0f, 1.0f};
    expect_true("push 3", nemo_attn_push(&c, k3, v3) == 0);
    expect_true("push 4", nemo_attn_push(&c, k3, v3) == 0);
    expect_true("push past full refused", nemo_attn_push(&c, k3, v3) == -1);
    expect_true("len stays at max", c.len == 4);
    nemo_attn_cache_free(&c);

    /* Real shape smoke: 40/8/128, three positions, determinism. */
    nemo_attn_cache big;
    if (nemo_attn_cache_init(&big, 40, 8, 128, 8) != 0) {
        printf("FAIL big cache init\n");
        return 1;
    }
    float kk[8 * 128], vv[8 * 128], qq[40 * 128], o1[40 * 128], o2[40 * 128];
    for (int p = 0; p < 3; p++) {
        for (int i = 0; i < 8 * 128; i++) {
            kk[i] = (float)((i * 7 + p * 13) % 17 - 8) * 0.25f;
            vv[i] = (float)((i * 5 + p * 11) % 13 - 6) * 0.25f;
        }
        if (nemo_attn_push(&big, kk, vv) != 0) {
            printf("FAIL big push\n");
            return 1;
        }
    }
    for (int i = 0; i < 40 * 128; i++)
        qq[i] = (float)((i * 3) % 9 - 4) * 0.25f;
    nemo_attn_decode(&big, qq, o1);
    nemo_attn_decode(&big, qq, o2);
    expect_true("real shape deterministic",
                memcmp(o1, o2, sizeof o1) == 0);
    int finite = 1;
    for (int i = 0; i < 40 * 128; i++)
        if (!isfinite(o1[i])) finite = 0;
    expect_true("real shape finite", finite);
    nemo_attn_cache_free(&big);

    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
