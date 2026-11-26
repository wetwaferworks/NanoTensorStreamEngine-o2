/* test_accel_guard.c - hostile inputs, budget, and the no-GPU
 * fallback. Co-implemented by Cortana.
 *
 * Part 1 (any machine): malformed tensors are refused with the
 * right code on the CPU path, before any device is touched.
 * Part 2 (any machine): NTSE_ACCEL_DISABLE=1 forces the CPU lane;
 * every operator and the stream then produce exactly the CPU
 * reference's bits, and the bf16 projection reports NODEV so the
 * caller uses its own kernel.
 * Part 3 (when a GPU lane is present): a 1 MiB budget makes a large
 * tensor fall back to the CPU, with correct output, and a budget
 * override above the hardware ceiling is clamped.
 */
#define _POSIX_C_SOURCE 200809L
#include "test_common.h"

#ifdef _WIN32
#include <windows.h>
static void set_env(const char *k, const char *v) { SetEnvironmentVariableA(k, v); _putenv_s(k, v ? v : ""); }
#else
static void set_env(const char *k, const char *v) { if (v) setenv(k, v, 1); else unsetenv(k); }
#endif

static void hostile(void)
{
    fixture f;
    fixture_make(&f, 8, 512, 256, 256, 256, 5);
    char why[200];
    expect("valid fixture validates", ntse_dtc_validate(&f.t, why, sizeof why) == NTSE_OK);

    ntse_dtc_tensor t = f.t;
    t.entries = 100;   /* idx bytes up to 255 now exceed entries */
    expect("index >= entries refused (INVAL)", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    printf("     reason: %s\n", why);

    uint16_t keep = f.mn[3];
    f.mn[3] = 0x7e00;  /* fp16 NaN */
    expect("NaN frame refused (NONFINITE)", ntse_dtc_validate(&f.t, why, sizeof why) == NTSE_E_NONFINITE);
    f.mn[3] = 0x7c00;  /* fp16 inf */
    expect("inf frame refused (NONFINITE)", ntse_dtc_validate(&f.t, why, sizeof why) == NTSE_E_NONFINITE);
    f.mn[3] = keep;

    float kb = f.book[10];
    f.book[10] = NAN;
    expect("NaN book refused (NONFINITE)", ntse_dtc_validate(&f.t, why, sizeof why) == NTSE_E_NONFINITE);
    f.book[10] = kb;

    t = f.t; t.cols = 500;
    expect("cols not a multiple of block: UNSUPPORTED", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_UNSUPPORTED);
    t = f.t; t.rows = 0;
    expect("zero rows refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    t = f.t; t.block = 0;
    expect("zero block refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    t = f.t; t.block = 8192;
    expect("block > 4096 refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    t = f.t; t.entries = 257;
    expect("entries > 256 refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    t = f.t; t.rows = 1ull << 31; t.cols = 1ull << 31;
    expect("rows*cols over the 2^40 cap refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    t = f.t; t.idx = NULL;
    expect("null index refused", ntse_dtc_validate(&t, why, sizeof why) == NTSE_E_INVAL);
    expect("null tensor refused", ntse_dtc_validate(NULL, why, sizeof why) == NTSE_E_INVAL);

    float x[1024], y[64];
    for (int i = 0; i < 1024; i++) x[i] = 1.0f;
    expect("cpu matvec batch 0 refused", ntse_cpu_matvec(&f.t, NTSE_OP_DTC_EAGER, x, 0, y) == NTSE_E_INVAL);
    expect("cpu matvec batch 65 refused", ntse_cpu_matvec(&f.t, NTSE_OP_DTC_EAGER, x, 65, y) == NTSE_E_INVAL);
    expect("cpu matvec bad op refused", ntse_cpu_matvec(&f.t, (ntse_op)9, x, 1, y) == NTSE_E_INVAL);
    fixture_free(&f);
}

static void no_gpu(void)
{
    set_env("NTSE_ACCEL_DISABLE", "1");
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.verbose = 1;
    ntse_accel *a = NULL;
    int rc = ntse_accel_open(&a, &cfg);
    expect("open with GPU disabled succeeds", rc == NTSE_OK && a);
    expect("lane is cpu", ntse_accel_lane(a) == NTSE_LANE_CPU);

    fixture f;
    fixture_make(&f, 333, 1024, 256, 256, 80, 77);
    float *x = activations(2 * 1024, 9);
    float *y = (float *)malloc(2 * 333 * 4), *ref = (float *)malloc(2 * 333 * 4);
    ntse_dtensor *d = NULL;
    expect("upload on cpu lane", ntse_accel_upload(a, &f.t, &d) == NTSE_OK && d);
    for (int op = 0; op <= NTSE_OP_DTC_LAZY; op++) {
        ntse_lane used = NTSE_LANE_CUDA;
        ntse_cpu_matvec(&f.t, (ntse_op)op, x, 2, ref);
        rc = ntse_accel_matvec(a, d, (ntse_op)op, x, 2, y, &used);
        char name[120];
        snprintf(name, sizeof name, "fallback %s is the CPU reference, bit for bit",
                 ntse_op_name((ntse_op)op));
        expect(name, rc == NTSE_OK && used == NTSE_LANE_CPU && memcmp(y, ref, 2 * 333 * 4) == 0);
    }
    ntse_stream_item it = { &f.t, x, y };
    ntse_stream_config sc = { 2, NTSE_OP_DTC_EAGER, 1 };
    ntse_stream_stats st;
    ntse_cpu_matvec(&f.t, NTSE_OP_DTC_EAGER, x, 2, ref);
    memset(y, 0, 2 * 333 * 4);
    rc = ntse_accel_stream(a, &it, 1, 2, &sc, &st);
    expect("stream on cpu lane matches the reference",
           rc == NTSE_OK && st.lane == NTSE_LANE_CPU && memcmp(y, ref, 2 * 333 * 4) == 0);
    uint16_t W[16] = { 0 };
    float xv[4] = { 1, 2, 3, 4 }, yv[4];
    expect("bf16 projection reports NODEV (caller keeps its kernel)",
           ntse_accel_linear_bf16(a, W, xv, yv, 4, 4, NULL) == NTSE_E_NODEV);
    ntse_accel_release(a, d);
    ntse_accel_close(a);

    cfg.allow_fallback = 0;
    cfg.lane = NTSE_LANE_CUDA;
    set_env("NTSE_ACCEL_DISABLE", NULL);
    set_env("NTSE_NVRTC_DLL", "ntse-no-such-nvrtc.dll");
    a = NULL;
    rc = ntse_accel_open(&a, &cfg);
    expect("CUDA lane with NVRTC missing and fallback off: open refuses",
           rc == NTSE_E_NODEV && a == NULL);
    cfg.allow_fallback = 1;
    rc = ntse_accel_open(&a, &cfg);
    expect("same, fallback on: opens on the cpu lane",
           rc == NTSE_OK && a && ntse_accel_lane(a) == NTSE_LANE_CPU);
    ntse_accel_close(a);
    set_env("NTSE_NVRTC_DLL", NULL);
    free(x); free(y); free(ref);
    fixture_free(&f);
}

static void budget(void)
{
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.verbose = 1;
    cfg.vram_budget = 1ull << 20;
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK || ntse_accel_lane(a) == NTSE_LANE_CPU) {
        printf("SKIP budget checks: no GPU lane present\n");
        ntse_accel_close(a);
        return;
    }
    expect("1 MiB budget honored", ntse_accel_budget(a) == (1ull << 20));
    /* (i) a compressed tensor larger than the budget: host-only handle */
    fixture f;
    fixture_make(&f, 8192, 4096, 256, 256, 90, 31);
    float *x = activations(4096, 3), *y = (float *)malloc(8192 * 4), *ref = (float *)malloc(8192 * 4);
    ntse_dtensor *d = NULL;
    int rc = ntse_accel_upload(a, &f.t, &d);
    expect("over-budget upload still yields a handle (host-only)", rc == NTSE_OK && d);
    ntse_lane used = NTSE_LANE_CUDA;
    ntse_cpu_matvec(&f.t, NTSE_OP_DTC_EAGER, x, 1, ref);
    rc = ntse_accel_matvec(a, d, NTSE_OP_DTC_EAGER, x, 1, y, &used);
    printf("     8192x4096 under a 1 MiB budget ran on %s\n", ntse_lane_name(used));
    expect("over-budget tensor runs on the CPU, exact",
           rc == NTSE_OK && used == NTSE_LANE_CPU && memcmp(y, ref, 8192 * 4) == 0);
    ntse_accel_release(a, d);
    fixture_free(&f);
    /* (ii) the compressed tensor fits, its expansion does not: DTC
     * runs on the device, EXPANDED falls back for that call only */
    fixture_make(&f, 1024, 1024, 256, 256, 90, 32);
    d = NULL;
    rc = ntse_accel_upload(a, &f.t, &d);
    ntse_cpu_matvec(&f.t, NTSE_OP_DTC_EAGER, x, 1, ref);
    rc |= ntse_accel_matvec(a, d, NTSE_OP_DTC_EAGER, x, 1, y, &used);
    expect("small compressed tensor runs DTC on the device under 1 MiB",
           rc == NTSE_OK && used != NTSE_LANE_CPU);
    ntse_cpu_matvec(&f.t, NTSE_OP_EXPANDED, x, 1, ref);
    rc = ntse_accel_matvec(a, d, NTSE_OP_EXPANDED, x, 1, y, &used);
    expect("its 4 MiB expansion exceeds the budget: that call runs on the CPU, exact",
           rc == NTSE_OK && used == NTSE_LANE_CPU && memcmp(y, ref, 1024 * 4) == 0);
    rc = ntse_accel_matvec(a, d, NTSE_OP_DTC_EAGER, x, 1, y, &used);
    expect("the next DTC call is back on the device (refusal not sticky)",
           rc == NTSE_OK && used != NTSE_LANE_CPU);
    expect("budget is never exceeded", ntse_accel_peak(a) <= ntse_accel_budget(a));
    expect("a budget refusal is not a sticky fault", !ntse_accel_failed(a));
    ntse_accel_release(a, d);
    ntse_accel_close(a);

    cfg.vram_budget = 1ull << 40;   /* 1 TiB: must clamp */
    a = NULL;
    ntse_accel_open(&a, &cfg);
    expect("over-ceiling budget clamped below 4 GiB", a && ntse_accel_budget(a) < (4ull << 30));
    printf("     clamped budget: %llu MiB\n", (unsigned long long)(ntse_accel_budget(a) >> 20));
    ntse_accel_close(a);
    free(x); free(y); free(ref);
    fixture_free(&f);
}

int main(void)
{
    hostile();
    no_gpu();
    budget();
    printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
