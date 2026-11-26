/* test_accel_kernels.c - per-kernel correctness of every present GPU
 * lane against the CPU reference, plus repeated-run determinism.
 * Co-implemented by Cortana for JD's acceleration lanes.
 *
 *   test_accel_kernels [--lane cuda|igpu|gpu-cl|all] [--big]
 *
 * For each lane present on this machine, each operator, and each
 * fixture shape: GPU y vs the CPU reference for the same operator
 * (max abs, max rel, rel L2, cosine, differing-bit count), both vs
 * the fp64 oracle, and three repeated GPU runs compared bit for bit.
 * Lanes that are absent are reported as SKIP, not as failures.
 */
#include "test_common.h"

typedef struct { uint64_t rows, cols; uint32_t block, entries, active, batch; } shape;

static const shape shapes[] = {
    { 64, 256, 256, 256, 256, 1 },
    { 1000, 1024, 256, 256, 96, 1 },
    { 1000, 1024, 256, 256, 96, 4 },
    { 300, 512, 64, 16, 16, 2 },
    { 17, 4096, 128, 200, 37, 1 },
    { 5, 768, 256, 256, 3, 3 },
    { 4096, 4096, 256, 256, 90, 1 },
};
static const shape big[] = {
    { 4480, 17920, 256, 256, 90, 1 },
    { 3072, 3072, 256, 256, 90, 8 },
};

static int run_lane(ntse_lane lane, int use_big)
{
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.lane = lane;
    cfg.allow_fallback = 0;
    cfg.verbose = 1;
    ntse_accel *a = NULL;
    int rc = ntse_accel_open(&a, &cfg);
    if (rc != NTSE_OK || ntse_accel_lane(a) != lane) {
        printf("SKIP lane %s: %s\n", ntse_lane_name(lane), ntse_accel_strerror(rc));
        ntse_accel_close(a);
        return 0;
    }
    printf("== lane %s: %s\n", ntse_lane_name(lane), ntse_accel_device(a));
    size_t ns = sizeof shapes / sizeof shapes[0];
    size_t nb = use_big ? sizeof big / sizeof big[0] : 0;
    for (size_t si = 0; si < ns + nb; si++) {
        const shape *s = si < ns ? &shapes[si] : &big[si - ns];
        fixture f;
        if (fixture_make(&f, s->rows, s->cols, s->block, s->entries, s->active,
                         1000 + si) != 0) {
            expect("fixture allocation", 0);
            continue;
        }
        uint64_t nx = (uint64_t)s->batch * s->cols, ny = (uint64_t)s->batch * s->rows;
        float *x = activations(nx, 7 + si);
        float *ref = (float *)malloc(ny * 4), *got = (float *)malloc(ny * 4),
              *again = (float *)malloc(ny * 4);
        double *orc = (double *)malloc(ny * 8);
        if (!x || !ref || !got || !again || !orc) {
            expect("buffer allocation", 0);
            goto next;
        }
        ntse_cpu_oracle(&f.t, x, s->batch, orc);
        ntse_dtensor *d = NULL;
        rc = ntse_accel_upload(a, &f.t, &d);
        char name[200];
        snprintf(name, sizeof name, "%s upload %llux%llu B%u K%u", ntse_lane_name(lane),
                 (unsigned long long)s->rows, (unsigned long long)s->cols, s->block,
                 s->entries);
        expect(name, rc == NTSE_OK && d);
        if (rc != NTSE_OK) goto next;
        for (int op = NTSE_OP_DTC_EAGER; op <= NTSE_OP_DTC_LAZY; op++) {
            ntse_lane used = NTSE_LANE_CPU;
            if (ntse_cpu_matvec(&f.t, (ntse_op)op, x, s->batch, ref) != NTSE_OK) {
                expect("cpu reference", 0);
                continue;
            }
            rc = ntse_accel_matvec(a, d, (ntse_op)op, x, s->batch, got, &used);
            snprintf(name, sizeof name, "%s %s %llux%llu B%u K%u act%u batch%u",
                     ntse_lane_name(lane), ntse_op_name((ntse_op)op),
                     (unsigned long long)s->rows, (unsigned long long)s->cols,
                     s->block, s->entries, s->active, s->batch);
            if (rc != NTSE_OK || used != lane) {
                printf("     %s: rc %d (%s), used %s\n", name, rc,
                       ntse_accel_strerror(rc), ntse_lane_name(used));
                expect(name, 0);
                continue;
            }
            metrics m = compare_f(got, ref, ny);
            double orl2 = rel_l2_vs_oracle(got, orc, ny);
            double crl2 = rel_l2_vs_oracle(ref, orc, ny);
            int det = 1;
            for (int rep = 0; rep < 3 && det; rep++) {
                if (ntse_accel_matvec(a, d, (ntse_op)op, x, s->batch, again, &used) != NTSE_OK
                    || memcmp(again, got, ny * 4) != 0)
                    det = 0;
            }
            double tol = op == NTSE_OP_EXPANDED ? TOL_EXP_RELL2 : TOL_DTC_RELL2;
            int ok = m.nonfinite == 0 && m.rel_l2 <= tol && m.cosine >= TOL_COS && det;
            expect(name, ok);
            print_metrics("vs CPU reference", &m);
            printf("     vs fp64 oracle: gpu relL2 %.3e, cpu relL2 %.3e; "
                   "deterministic x3 %s\n", orl2, crl2, det ? "yes" : "NO");
        }
        ntse_accel_release(a, d);
    next:
        free(x); free(ref); free(got); free(again); free(orc);
        fixture_free(&f);
    }
    printf("     peak device bytes charged: %llu of budget %llu\n",
           (unsigned long long)ntse_accel_peak(a), (unsigned long long)ntse_accel_budget(a));
    expect("no sticky device fault", !ntse_accel_failed(a));
    ntse_accel_close(a);
    return 1;
}

int main(int argc, char **argv)
{
    const char *which = "all";
    int use_big = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lane") && i + 1 < argc) which = argv[++i];
        else if (!strcmp(argv[i], "--big")) use_big = 1;
    }
    int ran = 0;
    if (!strcmp(which, "all") || !strcmp(which, "cuda")) ran += run_lane(NTSE_LANE_CUDA, use_big);
    if (!strcmp(which, "all") || !strcmp(which, "gpu-cl")) ran += run_lane(NTSE_LANE_CL_GPU, use_big);
    if (!strcmp(which, "all") || !strcmp(which, "igpu")) ran += run_lane(NTSE_LANE_CL_IGPU, use_big);
    printf("lanes exercised: %d\n", ran);
    printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
