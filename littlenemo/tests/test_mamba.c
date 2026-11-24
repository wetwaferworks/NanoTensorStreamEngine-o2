/* test_mamba.c - synthetic checks for the block math in src/mamba.c.
 * Expected values are hand-computed from BLOCK-MATH-SPEC.md, not from the
 * implementation. Exit 0 only if every check passes.
 */
#include "../src/mamba.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fails = 0;
static void check(const char *name, double got, double want, double tol) {
    if (fabs(got - want) > tol) {
        printf("FAIL %-28s got %.7f want %.7f\n", name, got, want);
        fails++;
    } else {
        printf("ok   %-28s %.7f\n", name, got);
    }
}

int main(void) {
    /* T6: bf16 widening is exact */
    check("bf16 0x3F80", nemo_bf16_to_f32(0x3F80), 1.0, 0.0);
    check("bf16 0xC000", nemo_bf16_to_f32(0xC000), -2.0, 0.0);

    /* T5: relu2 */
    check("relu2(3)", nemo_relu2(3.0f), 9.0, 0.0);
    check("relu2(-2)", nemo_relu2(-2.0f), 0.0, 0.0);
    check("relu2(0.5)", nemo_relu2(0.5f), 0.25, 0.0);

    /* T1: conv window invariant, one channel, w = [1 2 3 4], bias 0.
     * An impulse walks the taps: silu(4), silu(3), silu(2), silu(1), 0. */
    {
        float win[4] = {0, 0, 0, 0};
        float w[4] = {1, 2, 3, 4};
        float out = 0, u;
        u = 1; nemo_conv1d_step(win, &u, w, NULL, &out, 1);
        check("conv impulse tap4", out, 3.928055, 1e-4);
        u = 0; nemo_conv1d_step(win, &u, w, NULL, &out, 1);
        check("conv impulse tap3", out, 2.857722, 1e-4);
        u = 0; nemo_conv1d_step(win, &u, w, NULL, &out, 1);
        check("conv impulse tap2", out, 1.761594, 1e-4);
        u = 0; nemo_conv1d_step(win, &u, w, NULL, &out, 1);
        check("conv impulse tap1", out, 0.731059, 1e-4);
        u = 0; nemo_conv1d_step(win, &u, w, NULL, &out, 1);
        check("conv impulse gone", out, 0.0, 1e-6);
    }

    /* T2: SSM step. heads 4, head_dim 2, state 3, groups 2.
     * A_log = 0 gives A = -1; dt_raw + dt_bias = 0 gives dt = ln 2,
     * decay = exp(-ln 2) = 0.5. B = C = 1, D = 0, x = 1.
     * step 1: state = ln 2 per element, y = 3 ln 2 = 2.0794415
     * step 2: state = 1.5 ln 2, y = 4.5 ln 2 = 3.1191623 */
    {
        nemo_mamba_state s;
        if (nemo_mamba_state_init(&s, 4, 2, 3, 2) != 0) { printf("FAIL init\n"); return 1; }
        float x[8], B[6], C[6], dt_raw[4] = {0, 0, 0, 0}, A_log[4] = {0, 0, 0, 0};
        float D[4] = {0, 0, 0, 0}, dt_bias[4] = {0, 0, 0, 0}, y[8];
        for (int i = 0; i < 8; i++) x[i] = 1.0f;
        for (int i = 0; i < 6; i++) { B[i] = 1.0f; C[i] = 1.0f; }
        nemo_ssm_step(&s, x, B, C, dt_raw, A_log, D, dt_bias, y);
        check("ssm step1 y[0]", y[0], 2.0794415, 1e-4);
        check("ssm step1 y[7]", y[7], 2.0794415, 1e-4);
        nemo_ssm_step(&s, x, B, C, dt_raw, A_log, D, dt_bias, y);
        check("ssm step2 y[0]", y[0], 3.1191623, 1e-4);
        nemo_mamba_state_free(&s);
    }

    /* T4: D skip with zero state and C = 0: y = D * x exactly. */
    {
        nemo_mamba_state s;
        if (nemo_mamba_state_init(&s, 1, 1, 2, 1) != 0) { printf("FAIL init\n"); return 1; }
        float x[1] = {3.0f}, B[2] = {5, 7}, C[2] = {0, 0}, dt_raw[1] = {0.25f};
        float A_log[1] = {0.0f}, D[1] = {2.0f}, dt_bias[1] = {0.0f}, y[1];
        nemo_ssm_step(&s, x, B, C, dt_raw, A_log, D, dt_bias, y);
        check("ssm D skip", y[0], 6.0, 1e-6);
        nemo_mamba_state_free(&s);
    }

    /* T3: gated norm. z = 0 kills the gate (silu(0) = 0).
     * z = 20: g = [20 40 60 80], mean square 3000, rstd = 1/sqrt(3000). */
    {
        float x[4] = {1, 2, 3, 4}, z0[4] = {0, 0, 0, 0}, z2[4] = {20, 20, 20, 20};
        float w[4] = {1, 1, 1, 1}, out[4];
        nemo_rmsnorm_gated(x, z0, w, out, 4, 4, 1e-5f);
        check("gated norm z=0", out[2], 0.0, 1e-6);
        nemo_rmsnorm_gated(x, z2, w, out, 4, 4, 1e-5f);
        check("gated norm [0]", out[0], 0.3651484, 1e-4);
        check("gated norm [3]", out[3], 1.4605935, 1e-4);
    }

    /* T7: full decode composition is finite and deterministic. */
    {
        nemo_mamba_state a, b;
        if (nemo_mamba_state_init(&a, 4, 2, 3, 2) != 0) return 1;
        if (nemo_mamba_state_init(&b, 4, 2, 3, 2) != 0) return 1;
        int inner = a.inner, conv_dim = a.conv_dim;
        float proj[32], conv_w[80], conv_b[20], A_log[4], D[4], dt_bias[4], norm_w[8];
        float oa[8], ob[8];
        for (int i = 0; i < 32; i++) proj[i] = 0.05f * (float)((i * 7) % 5) - 0.05f;
        for (int i = 0; i < conv_dim * 4; i++) conv_w[i] = 0.01f * (float)((i * 3) % 7) - 0.03f;
        for (int i = 0; i < conv_dim; i++) conv_b[i] = 0.001f * (float)(i % 3);
        for (int i = 0; i < 4; i++) { A_log[i] = 0.1f * i; D[i] = 0.5f; dt_bias[i] = -0.2f; }
        for (int i = 0; i < inner; i++) norm_w[i] = 1.0f;
        nemo_mamba_decode(&a, proj, 0, conv_w, conv_b, A_log, D, dt_bias, norm_w, 2, 1e-5f, oa);
        nemo_mamba_decode(&b, proj, 0, conv_w, conv_b, A_log, D, dt_bias, norm_w, 2, 1e-5f, ob);
        int same = memcmp(oa, ob, sizeof oa) == 0;
        int finite = 1;
        for (int i = 0; i < inner; i++) if (!isfinite(oa[i])) finite = 0;
        check("decode deterministic", same, 1.0, 0.0);
        check("decode finite", finite, 1.0, 0.0);
        nemo_mamba_state_free(&a);
        nemo_mamba_state_free(&b);
    }

    printf(fails ? "RESULT: %d FAILURES\n" : "RESULT: ALL PASS\n", fails);
    return fails ? 1 : 0;
}
