/* test_simd.c - the opt-in SIMD path, held to four standards.
 * (1) It must be deterministic: the same SIMD run across a reset
 *     is memcmp-identical to itself.
 * (2) It must stay faithful to the scalar path: the hidden states
 *     may drift in the last bits (the accumulation order differs
 *     by design), and this suite measures the drift on real
 *     weights and requires it to stay tiny.
 * (3) The kernel itself must be exercised directly, on synthetic
 *     data, and the nemo_linear_bf16 entry point must provably
 *     dispatch to it (Archie review P0-1, 2026-11-24: the first
 *     version of this suite only ever compared scalar to scalar,
 *     because both models ran one thread and the entry point
 *     called the scalar rows directly).
 * (4) A multi-lane SIMD forward must be bit-identical to the
 *     1-lane SIMD forward: rows are independent and each row's
 *     accumulation order is fixed, whatever lane computes it.
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

/* Truncating f32 -> bf16 is fine for synthetic data. */
static uint16_t f32_to_bf16_trunc(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)(bits >> 16);
}

/* Standard (3): the kernel, called directly, on synthetic data.
 * out = 37 (odd row count), in = 130 (two-element tail past the
 * last full group of 8). Values come from a fixed LCG so the
 * fixture is identical on every run and machine. */
static void synthetic_kernel_test(void) {
    enum { TOUT = 37, TIN = 130 };
    static uint16_t W[TOUT * TIN];
    static float x[TIN], y_scalar[TOUT], y_simd[TOUT], y_sub_s[TOUT],
        y_sub_v[TOUT], y_full_off[TOUT], y_full_on[TOUT];
    uint32_t s = 0x12345678u;
    for (int i = 0; i < TOUT * TIN; i++) {
        s = s * 1664525u + 1013904223u;
        W[i] = f32_to_bf16_trunc(((s >> 9) & 0xFFFF) / 32768.0f - 1.0f);
    }
    for (int i = 0; i < TIN; i++) {
        s = s * 1664525u + 1013904223u;
        x[i] = ((s >> 9) & 0xFFFF) / 32768.0f - 1.0f;
    }
    nemo_linear_bf16_rows(W, x, y_scalar, 0, TOUT, TIN);
    nemo_linear_bf16_rows_simd(W, x, y_simd, 0, TOUT, TIN);
    double maxd = 0.0;
    for (int o = 0; o < TOUT; o++) {
        double d = fabs((double)y_scalar[o] - (double)y_simd[o]);
        if (d > maxd) maxd = d;
    }
    printf("     synthetic kernel drift: max abs %.3e\n", maxd);
    expect_true("synthetic rows_simd matches rows (< 1e-9)", maxd < 1e-9);

    /* A row range must not change a row's result: compute rows
     * [5, 20) both ways and compare just those rows. */
    nemo_linear_bf16_rows(W, x, y_sub_s, 5, 20, TIN);
    nemo_linear_bf16_rows_simd(W, x, y_sub_v, 5, 20, TIN);
    maxd = 0.0;
    for (int o = 5; o < 20; o++) {
        double d = fabs((double)y_sub_s[o] - (double)y_sub_v[o]);
        if (d > maxd) maxd = d;
    }
    expect_true("synthetic rows_simd sub-range matches (< 1e-9)",
                maxd < 1e-9);

    /* The dispatch regression (P0-1): with SIMD opted in, the
     * full-call entry point must produce exactly the direct
     * kernel's output; with it off, exactly the scalar output. */
    nemo_set_simd(0);
    nemo_linear_bf16(W, x, y_full_off, TOUT, TIN);
    nemo_set_simd(1);
    nemo_linear_bf16(W, x, y_full_on, TOUT, TIN);
    expect_true("nemo_linear_bf16 off == scalar rows",
                memcmp(y_full_off, y_scalar, sizeof y_scalar) == 0);
    expect_true("nemo_linear_bf16 on == direct rows_simd",
                memcmp(y_full_on, y_simd, sizeof y_simd) == 0);
    nemo_set_simd(0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("usage: test_simd <checkpoint-dir> [layers]\n");
        return 1;
    }
    int layers = argc > 2 ? atoi(argv[2]) : 15;
    expect_true("AVX2 kernel available on this machine",
                nemo_simd_available() == 1);
    synthetic_kernel_test();
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

    /* Standard (4): the same SIMD forward on 4 lanes must be
     * bit-identical to the 1-lane run above (hb still holds it):
     * each row's accumulation order is fixed, whatever lane
     * computes the row. */
    nemo_model c;
    if (nemo_model_init_ex(&c, &cp, layers, 16, 4) != 0) {
        printf("FAIL multi-lane model init\n");
        return 1;
    }
    static float hc[NEMO_HIDDEN];
    rc = 0;
    for (int i = 0; i < 3; i++) rc |= nemo_model_step_hidden(&c, seq[i], hc);
    expect_true("multi-lane simd forward runs", rc == 0);
    expect_true("multi-lane simd == 1-lane simd (bit-identical)",
                memcmp(hc, hb, sizeof hb) == 0);
    nemo_model_free(&c);

    nemo_set_simd(0);
    nemo_model_free(&a);
    nemo_model_free(&b);
    nemo_checkpoint_close(&cp);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
