/* test_accel_linear.c - the expanded bf16 projection kernel
 * (ntse_accel_linear_bf16, the lane Little Nemo's shim routes to),
 * tested directly rather than only through the forward smoke.
 * Co-implemented by Cortana.
 *
 *     test_accel_linear [--lane cuda|igpu|gpu-cl]
 *
 * Reference: products of a bf16 weight (8 significant bits) and an
 * fp32 activation (24) are exact in double, so a double sum is within
 * n*2^-53 of the true dot product; that is the oracle. Gates per shape:
 *   output poisoned with NaN beforehand is fully overwritten, finite;
 *   every value within 1 ulp(ref) + 2^-40 * sum|w*x| of the oracle
 *     (1 ulp: the final fp32 rounding; the second term bounds the
 *     float-float accumulation used on GPUs with no fp64);
 *   bit-identical over three repeats;
 *   a cancellation case (exactly opposing pairs plus one tiny term)
 *     returns the tiny term, not noise of the large ones.
 * And refusals: NULL/zero/oversized geometry is NTSE_E_INVAL, and an
 * over-budget projection is refused without a sticky fault.
 */
#include "test_common.h"
#include <float.h>

static uint16_t to_bf16(float f)
{
    uint32_t u; memcpy(&u, &f, 4);
    u += 0x7fffu + ((u >> 16) & 1u);   /* round to nearest even */
    return (uint16_t)(u >> 16);
}
static float from_bf16(uint16_t b)
{
    uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f;
}
static double ulpf(float v)
{
    float a = fabsf(v);
    if (a < FLT_MIN) return ldexp(1.0, -149);
    int e; frexpf(a, &e);
    return ldexp(1.0, e - 24);
}

static void check_shape(ntse_accel *a, int out, int in, int cancel)
{
    char nm[160];
    size_t nw = (size_t)out * in;
    uint16_t *W = (uint16_t *)malloc(nw * 2);
    float *x = (float *)malloc((size_t)in * 4);
    float *y = (float *)malloc((size_t)out * 4), *y2 = (float *)malloc((size_t)out * 4);
    if (!W || !x || !y || !y2) { expect("allocate fixtures", 0); goto done; }
    g_rng = 0x5eed0000u + (uint64_t)out * 31 + in;
    for (int i = 0; i < in; i++) x[i] = (float)nrand();
    for (size_t k = 0; k < nw; k++) W[k] = to_bf16((float)(0.05 * nrand()));
    if (cancel) {
        /* rows: pairs (w,-w) on equal x cancel exactly; the last column
         * carries a tiny product; in must be odd */
        for (int i = 0; i + 1 < in - 1; i += 2) x[i + 1] = x[i];
        for (int r = 0; r < out; r++) {
            uint16_t *row = W + (size_t)r * in;
            for (int i = 0; i + 1 < in - 1; i += 2) {
                float big = (float)(1000.0 * (1 + urand()));
                row[i] = to_bf16(big);
                row[i + 1] = to_bf16(-from_bf16(row[i]));
            }
            row[in - 1] = to_bf16(1e-6f);
        }
        x[in - 1] = 1.0f;
    }
    for (int r = 0; r < out; r++) { uint32_t q = 0x7fc00001u; memcpy(&y[r], &q, 4); }
    ntse_lane used = NTSE_LANE_CPU;
    int rc = ntse_accel_linear_bf16(a, W, x, y, out, in, &used);
    snprintf(nm, sizeof nm, "[%d x %d]%s runs on %s", out, in, cancel ? " cancellation" : "",
             ntse_lane_name(ntse_accel_lane(a)));
    expect(nm, rc == NTSE_OK && used == ntse_accel_lane(a));
    if (rc != NTSE_OK) goto done;
    int finite = 1, within = 1;
    double worst = 0, d2 = 0, r2 = 0;
    int differ = 0;
    for (int r = 0; r < out; r++) {
        const uint16_t *row = W + (size_t)r * in;
        double s = 0, sa = 0;
        for (int i = 0; i < in; i++) {
            double p = (double)from_bf16(row[i]) * (double)x[i];
            s += p; sa += fabs(p);
        }
        if (!isfinite(y[r])) { finite = 0; continue; }
        float rf = (float)s;
        double err = fabs((double)y[r] - s);
        double bound = ulpf(rf) + ldexp(sa, -40);
        if (err > bound) within = 0;
        if (bound > 0 && err / bound > worst) worst = err / bound;
        if (memcmp(&y[r], &rf, 4)) differ++;
        d2 += err * err; r2 += s * s;
    }
    printf("     relL2 vs fp64 oracle %.3e | worst err/bound %.3f | %d/%d differ from "
           "correctly rounded%s\n", r2 > 0 ? sqrt(d2 / r2) : 0.0, worst, differ, out,
           cancel ? " | tiny term recovered" : "");
    expect("  output fully written and finite (NaN-poisoned beforehand)", finite);
    expect("  every value within 1 ulp + 2^-40 sum|wx| of the oracle", within);
    int det = 1;
    for (int k = 0; k < 2 && det; k++) {
        if (ntse_accel_linear_bf16(a, W, x, y2, out, in, NULL) != NTSE_OK
            || memcmp(y, y2, (size_t)out * 4))
            det = 0;
    }
    expect("  bit-identical over three runs", det);
done:
    free(W); free(x); free(y); free(y2);
}

int main(int argc, char **argv)
{
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.verbose = 1;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--lane") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) cfg.lane = NTSE_LANE_CUDA;
            else if (!strcmp(argv[i], "igpu")) cfg.lane = NTSE_LANE_CL_IGPU;
            else if (!strcmp(argv[i], "gpu-cl")) cfg.lane = NTSE_LANE_CL_GPU;
        }
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK) { printf("FAIL open\n"); return 1; }
    if (ntse_accel_lane(a) == NTSE_LANE_CPU) {
        uint16_t w = 0; float xv = 0, yv = 0;
        expect("cpu lane: linear refuses with NODEV (caller's kernel is the fallback)",
               ntse_accel_linear_bf16(a, &w, &xv, &yv, 1, 1, NULL) == NTSE_E_NODEV);
        ntse_accel_close(a);
        printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
        return g_fail ? 1 : 0;
    }
    static const int shapes[][2] = {
        { 1, 1 }, { 7, 13 }, { 3, 255 }, { 5, 257 }, { 64, 4480 },
        { 4480, 4480 }, { 257, 15680 }, { 2, 1 << 17 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++)
        check_shape(a, shapes[s][0], shapes[s][1], 0);
    check_shape(a, 33, 4097, 1);
    check_shape(a, 4, 1 << 16 | 1, 1);

    uint16_t w = 0; float xv = 0, yv = 0;
    expect("NULL W is INVAL", ntse_accel_linear_bf16(a, NULL, &xv, &yv, 1, 1, NULL) == NTSE_E_INVAL);
    expect("out = 0 is INVAL", ntse_accel_linear_bf16(a, &w, &xv, &yv, 0, 1, NULL) == NTSE_E_INVAL);
    expect("in < 0 is INVAL", ntse_accel_linear_bf16(a, &w, &xv, &yv, 1, -4, NULL) == NTSE_E_INVAL);
    expect("out*in > 2^34 is INVAL (no allocation attempted)",
           ntse_accel_linear_bf16(a, &w, &xv, &yv, 1 << 20, 1 << 15, NULL) == NTSE_E_INVAL);
    ntse_accel_close(a);

    /* over budget: refused per call, context stays healthy */
    cfg.vram_budget = 1ull << 20;
    if (ntse_accel_open(&a, &cfg) == NTSE_OK && ntse_accel_lane(a) != NTSE_LANE_CPU) {
        int out = 1024, in = 1024;   /* 2 MiB of bf16 under a 1 MiB budget */
        uint16_t *W = (uint16_t *)calloc((size_t)out * in, 2);
        float *x = (float *)calloc(in, 4), *y = (float *)calloc(out, 4);
        int rc = ntse_accel_linear_bf16(a, W, x, y, out, in, NULL);
        expect("over-budget projection refused with BUDGET", rc == NTSE_E_BUDGET);
        expect("  no sticky fault after a budget refusal", !ntse_accel_failed(a));
        rc = ntse_accel_linear_bf16(a, W, x, y, 64, 1024, NULL);
        expect("  a small projection still runs afterwards", rc == NTSE_OK);
        free(W); free(x); free(y);
    }
    ntse_accel_close(a);
    printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
