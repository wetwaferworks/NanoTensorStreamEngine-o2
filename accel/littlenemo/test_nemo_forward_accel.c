/* test_nemo_forward_accel.c - Little Nemo's truncated forward smoke
 * (the same 15 layers, tokens and checks as littlenemo/tests/
 * test_forward.c) with the projections on an accel lane, compared
 * against the CPU run of the same binary and the frozen anchor.
 * Co-implemented by Cortana; Little Nemo is Elfie's.
 *
 *     test_nemo_forward_accel <checkpoint-dir> [--lane cuda|igpu|gpu-cl]
 *
 * Gates:
 *   CPU run reproduces the frozen hidden RMS 0.520647 (printed %.6f,
 *     so |RMS - 0.520647| <= 5e-7: the anchor's own print precision);
 *   GPU run is finite, deterministic across a reset, and its RMS
 *     matches the anchor to the same print precision;
 *   GPU vs CPU hidden state: rel L2 <= 1e-5 (the projections sum in a
 *     different, fixed order; double accumulation on the GPU, so the
 *     expected drift is near 1e-7; measured and printed);
 *   at least one projection actually ran on the GPU.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../littlenemo/src/forward.h"
#include "../include/ntse_accel.h"

void nemo_accel_bind(ntse_accel *a, int on);
unsigned long long nemo_accel_gpu_calls(void);
unsigned long long nemo_accel_cpu_calls(void);
void nemo_accel_reset_counts(void);

static int checks = 0, failures = 0;
static void expect(const char *name, int ok)
{
    checks++;
    if (!ok) failures++;
    printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    fflush(stdout);
}

static int run(nemo_model *m, const int *seq, int n, float *h)
{
    nemo_model_reset(m);
    int rc = 0;
    for (int i = 0; i < n; i++) rc |= nemo_model_step_hidden(m, seq[i], h);
    return rc;
}

static double rms(const float *h)
{
    double s = 0;
    for (int i = 0; i < NEMO_HIDDEN; i++) s += (double)h[i] * h[i];
    return sqrt(s / NEMO_HIDDEN);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        printf("usage: test_nemo_forward_accel <checkpoint-dir> [--lane L]\n");
        return 1;
    }
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.verbose = 1;
    for (int i = 2; i < argc; i++)
        if (!strcmp(argv[i], "--lane") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) cfg.lane = NTSE_LANE_CUDA;
            else if (!strcmp(argv[i], "igpu")) cfg.lane = NTSE_LANE_CL_IGPU;
            else if (!strcmp(argv[i], "gpu-cl")) cfg.lane = NTSE_LANE_CL_GPU;
        }
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK) { printf("FAIL accel open\n"); return 1; }
    printf("lane %s: %s\n", ntse_lane_name(ntse_accel_lane(a)), ntse_accel_device(a));

    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) { printf("FAIL checkpoint open\n"); return 1; }
    nemo_model m;
    if (nemo_model_init(&m, &cp, 15, 16) != 0) { printf("FAIL model init\n"); return 1; }
    static float hc[NEMO_HIDDEN], hg[NEMO_HIDDEN], hg2[NEMO_HIDDEN];
    const int seq[3] = { 42, 1000, 7 };

    nemo_accel_bind(a, 0);
    int rc = run(&m, seq, 3, hc);
    double rc_rms = rms(hc);
    expect("cpu: three tokens through 15 layers", rc == 0);
    printf("     cpu hidden RMS %.6f (anchor 0.520647)\n", rc_rms);
    expect("cpu: frozen anchor RMS 0.520647 holds", fabs(rc_rms - 0.520647) <= 5e-7);

    if (ntse_accel_lane(a) == NTSE_LANE_CPU) {
        printf("SKIP gpu forward: no GPU lane present\n");
    } else {
        nemo_accel_bind(a, 1);
        nemo_accel_reset_counts();
        rc = run(&m, seq, 3, hg);
        unsigned long long gcalls = nemo_accel_gpu_calls(), ccalls = nemo_accel_cpu_calls();
        double g_rms = rms(hg);
        int finite = 1;
        for (int i = 0; i < NEMO_HIDDEN; i++) if (!isfinite(hg[i])) finite = 0;
        expect("gpu: three tokens through 15 layers", rc == 0);
        expect("gpu: hidden state finite", finite);
        printf("     gpu hidden RMS %.6f; projections on gpu %llu, on cpu %llu\n",
               g_rms, gcalls, ccalls);
        expect("gpu: projections actually ran on the device", gcalls > 0);
        expect("gpu: RMS matches the anchor at its print precision",
               fabs(g_rms - 0.520647) <= 5e-7);
        double d2 = 0, r2 = 0, mx = 0;
        unsigned long long differ = 0;
        for (int i = 0; i < NEMO_HIDDEN; i++) {
            double d = (double)hg[i] - hc[i];
            d2 += d * d; r2 += (double)hc[i] * hc[i];
            if (fabs(d) > mx) mx = fabs(d);
            if (memcmp(&hg[i], &hc[i], 4)) differ++;
        }
        double rl2 = sqrt(d2 / r2);
        printf("     gpu vs cpu hidden: relL2 %.3e, max_abs %.3e, %llu of %d values "
               "differ in any bit\n", rl2, mx, differ, NEMO_HIDDEN);
        expect("gpu vs cpu hidden relL2 <= 1e-5", rl2 <= 1e-5);
        rc = run(&m, seq, 3, hg2);
        expect("gpu: deterministic across reset", rc == 0 && memcmp(hg, hg2, sizeof hg) == 0);
        expect("gpu: no sticky device fault", !ntse_accel_failed(a));
        nemo_accel_bind(a, 0);
        rc = run(&m, seq, 3, hg2);
        expect("cpu after gpu: anchor bits unchanged", rc == 0 && memcmp(hc, hg2, sizeof hc) == 0);
    }
    nemo_model_free(&m);
    nemo_checkpoint_close(&cp);
    ntse_accel_close(a);
    printf("RESULT: %d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
