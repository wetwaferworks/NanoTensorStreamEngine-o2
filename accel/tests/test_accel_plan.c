/* test_accel_plan.c - the prepared stream plan (Lane 2). Co-implemented
 * by Cortana; lane proposal Gizmo; NTSE/nq2 Elfie.
 *
 *     test_accel_plan [--lane cuda|igpu|gpu-cl]
 *
 * Gates: every (op, slots 1..4, batch 1 and 3, pinned and pageable
 * image) run is bit-identical to the resident path and to the CPU
 * kernel, reports zero ownership violations and the exact H2D byte
 * count; books are deduplicated by content; runs are deterministic;
 * a forced early recycle is refused; the GPU path reads the create-
 * time snapshot; pinned/device budgets and a CPU-only context give a
 * host-only plan with correct output and no sticky fault; hostile
 * and malformed input is NTSE_E_INVAL.
 */
#include "test_common.h"

enum { NT = 6 };
static fixture F[NT];
static ntse_dtc_tensor T[NT];
static const ntse_dtc_tensor *TP[NT];
static float *X[NT], *Y[NT], *R[NT], *Z[NT];
static float *bookcopy;

static void make(void)
{
    static const uint64_t g[NT][5] = {
        { 512, 1024, 256, 256, 90 }, { 3072, 3072, 256, 256, 90 },
        { 1000, 768, 256, 256, 40 }, { 64, 4096, 256, 256, 256 },
        { 300, 512, 128, 64, 64 },   { 2048, 2048, 256, 256, 256 },
    };
    for (int i = 0; i < NT; i++) {
        fixture_make(&F[i], g[i][0], g[i][1], (uint32_t)g[i][2], (uint32_t)g[i][3],
                     (uint32_t)g[i][4], 900 + i);
        T[i] = F[i].t;
        X[i] = activations(3 * g[i][1], 70 + i);
        Y[i] = (float *)malloc(3 * g[i][0] * 4);
        R[i] = (float *)malloc(3 * g[i][0] * 4);
        Z[i] = (float *)malloc(3 * g[i][0] * 4);
        TP[i] = &T[i];
    }
    /* books: 0,1,3,5 share one pointer; 2 carries an equal copy at
     * another address; 4 has its own geometry -> 2 distinct books */
    bookcopy = (float *)malloc(256 * 256 * 4);
    memcpy(bookcopy, T[0].book, 256 * 256 * 4);
    T[1].book = T[3].book = T[5].book = T[0].book;
    T[2].book = bookcopy;
}

static int ref_for(ntse_op op, uint32_t batch)
{
    for (int i = 0; i < NT; i++)
        if (ntse_cpu_matvec(&T[i], op, X[i], batch, R[i]) != NTSE_OK) return 0;
    return 1;
}

static int same(float *const *a, float *const *b, uint32_t batch)
{
    for (int i = 0; i < NT; i++)
        if (memcmp(a[i], b[i], (size_t)batch * T[i].rows * 4)) return 0;
    return 1;
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
    make();
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK) { printf("FAIL open\n"); return 1; }
    ntse_lane lane = ntse_accel_lane(a);
    printf("== plan suite on %s\n", ntse_lane_name(lane));
    const float *const *xc = (const float *const *)X;

    /* resident path for the same tensors, the twin to match */
    ntse_dtensor *D[NT];
    for (int i = 0; i < NT; i++) ntse_accel_upload(a, &T[i], &D[i]);

    static const ntse_op ops[4] = { NTSE_OP_DTC_EAGER, NTSE_OP_DTC_ACTIVE, NTSE_OP_DTC_DIRECT, NTSE_OP_DTC_LAZY };
    char nm[200];
    int all_ok = 1, cases = 0, leak = 0;
    for (int oi = 0; oi < 4; oi++) {
        for (uint32_t batch = 1; batch <= 3; batch += 2) {
            ref_for(ops[oi], batch);
            for (int i = 0; i < NT; i++) ntse_accel_matvec(a, D[i], ops[oi], X[i], batch, Z[i], NULL);
            int resident_ok = same(Z, R, batch);
            for (int pin = 1; pin >= 0; pin--) {
                for (uint32_t S = 1; S <= 4; S++) {
                    ntse_plan_config pc;
                    ntse_plan_config_default(&pc);
                    pc.slots = S; pc.op = ops[oi]; pc.max_batch = 3; pc.pin_image = pin;
                    ntse_plan *p = NULL;
                    uint64_t c0 = ntse_accel_charged(a);
                    int rc = ntse_accel_plan_create(a, TP, NT, &pc, &p);
                    ntse_plan_info info;
                    ntse_accel_plan_info_get(p, &info);
                    ntse_stream_stats st;
                    memset(Y[0], 0xff, 4);
                    int rr = rc == NTSE_OK ? ntse_accel_plan_run(a, p, xc, Y, batch, &st) : rc;
                    uint64_t want_h2d = 0;
                    int ok = rc == NTSE_OK && rr == NTSE_OK && same(Y, R, batch) && resident_ok
                             && st.ownership_violations == 0 && st.lane == lane;
                    if (lane != NTSE_LANE_CPU) {
                        ok = ok && info.on_device && info.pinned_image == pin && info.books == 2;
                        for (int i = 0; i < NT; i++) want_h2d += (uint64_t)batch * T[i].cols * 4;
                        ok = ok && st.h2d_bytes == info.image_bytes + want_h2d;
                    }
                    cases++;
                    if (!ok) {
                        all_ok = 0;
                        snprintf(nm, sizeof nm, "%s batch %u %s slots %u: rc %d/%d bit-exact %d "
                                 "viol %llu books %u h2d %llu (want %llu)", ntse_op_name(ops[oi]),
                                 batch, pin ? "pinned" : "pageable", S, rc, rr, same(Y, R, batch),
                                 (unsigned long long)st.ownership_violations, info.books,
                                 (unsigned long long)st.h2d_bytes,
                                 (unsigned long long)(info.image_bytes + want_h2d));
                        expect(nm, 0);
                    }
                    if (S == 2 && pin == 1 && batch == 1)
                        printf("     %s: image %llu B/token (+x), books %u resident (%llu B), "
                               "pinned %llu B, create %.1f ms, run %.3f ms\n",
                               ntse_op_name(ops[oi]), (unsigned long long)info.image_bytes,
                               info.books, (unsigned long long)info.book_bytes,
                               (unsigned long long)info.pinned_bytes, info.create_ms, st.wall_ms);
                    ntse_accel_plan_free(a, p);
                    if (ntse_accel_charged(a) != c0) leak = 1;
                }
            }
        }
    }
    expect("every plan returns exactly what it charged (no device leak)", !leak);
    snprintf(nm, sizeof nm, "%d configs (4 ops x batch 1,3 x pinned/pageable x slots 1..4) "
             "bit-identical to resident and CPU, 0 violations, exact H2D", cases);
    expect(nm, all_ok);

    /* determinism and the early-recycle refusal */
    {
        ntse_plan_config pc; ntse_plan_config_default(&pc); pc.slots = 3;
        ntse_plan *p = NULL;
        ntse_accel_plan_create(a, TP, NT, &pc, &p);
        ref_for(NTSE_OP_DTC_EAGER, 1);
        ntse_stream_stats st;
        int det = 1;
        for (int r = 0; r < 3; r++) {
            ntse_accel_plan_run(a, p, xc, Y, 1, &st);
            if (!same(Y, R, 1)) det = 0;
        }
        expect("three runs bit-identical", det);
        if (lane != NTSE_LANE_CPU) {
            ntse_accel_test_force_early_recycle(a, 1);
            int rc = ntse_accel_plan_run(a, p, xc, Y, 1, &st);
            ntse_accel_test_force_early_recycle(a, 0);
            printf("     early recycle: rc %d, violations %llu\n", rc,
                   (unsigned long long)st.ownership_violations);
            expect("forced early recycle refused (1 violation), output still exact",
                   rc == NTSE_OK && st.ownership_violations == 1 && same(Y, R, 1));
            expect("no sticky fault after the refusal", !ntse_accel_failed(a));
            /* snapshot: the device path never reads the caller's arrays again */
            uint8_t saved = F[1].idx[7];
            F[1].idx[7] = (uint8_t)(saved ^ 1);
            ntse_accel_plan_run(a, p, xc, Y, 1, &st);
            F[1].idx[7] = saved;
            expect("device path uses the create-time snapshot", same(Y, R, 1));
        }
        ntse_accel_plan_free(a, p);
    }

    /* refusals */
    {
        ntse_plan_config pc; ntse_plan_config_default(&pc);
        ntse_plan *p = NULL;
        expect("NULL tensors INVAL", ntse_accel_plan_create(a, NULL, NT, &pc, &p) == NTSE_E_INVAL);
        expect("n = 0 INVAL", ntse_accel_plan_create(a, TP, 0, &pc, &p) == NTSE_E_INVAL);
        pc.slots = 0; expect("slots 0 INVAL", ntse_accel_plan_create(a, TP, NT, &pc, &p) == NTSE_E_INVAL);
        pc.slots = 5; expect("slots 5 INVAL", ntse_accel_plan_create(a, TP, NT, &pc, &p) == NTSE_E_INVAL);
        pc.slots = 2; pc.max_batch = 65;
        expect("max_batch 65 INVAL", ntse_accel_plan_create(a, TP, NT, &pc, &p) == NTSE_E_INVAL);
        pc.max_batch = 1; pc.op = NTSE_OP_EXPANDED;
        expect("expanded op INVAL", ntse_accel_plan_create(a, TP, NT, &pc, &p) == NTSE_E_INVAL);
        pc.op = NTSE_OP_DTC_EAGER;
        uint8_t saved = F[3].idx[11];
        F[3].idx[11] = 255; T[3].entries = 200;   /* idx >= entries */
        int rc = ntse_accel_plan_create(a, TP, NT, &pc, &p);
        T[3].entries = 256; F[3].idx[11] = saved;
        expect("hostile tensor in the list refused, no plan", rc == NTSE_E_INVAL && p == NULL);
        ntse_accel_plan_create(a, TP, NT, &pc, &p);
        expect("run batch > max_batch INVAL", ntse_accel_plan_run(a, p, xc, Y, 2, NULL) == NTSE_E_INVAL);
        float *Xn[NT]; memcpy(Xn, X, sizeof Xn); Xn[2] = NULL;
        expect("NULL activation INVAL",
               ntse_accel_plan_run(a, p, (const float *const *)Xn, Y, 1, NULL) == NTSE_E_INVAL);
        ntse_accel_plan_free(a, p);
    }

    /* pinned budget too small for the slots -> host-only, still correct */
    if (lane != NTSE_LANE_CPU) {
        ntse_plan_config pc; ntse_plan_config_default(&pc); pc.pin_budget = 4096;
        ntse_plan *p = NULL;
        int rc = ntse_accel_plan_create(a, TP, NT, &pc, &p);
        ntse_plan_info info; ntse_accel_plan_info_get(p, &info);
        ntse_stream_stats st;
        ref_for(NTSE_OP_DTC_EAGER, 1);
        int rr = ntse_accel_plan_run(a, p, xc, Y, 1, &st);
        expect("4 KiB pinned budget: host-only plan, CPU output exact, no sticky fault",
               rc == NTSE_OK && !info.on_device && rr == NTSE_OK && st.lane == NTSE_LANE_CPU
               && same(Y, R, 1) && !ntse_accel_failed(a));
        ntse_accel_plan_free(a, p);
    }
    for (int i = 0; i < NT; i++) ntse_accel_release(a, D[i]);
    ntse_accel_close(a);

    /* device budget too small -> host-only plan */
    if (lane != NTSE_LANE_CPU) {
        cfg.vram_budget = 1ull << 20;
        ntse_accel_open(&a, &cfg);
        ntse_plan_config pc; ntse_plan_config_default(&pc);
        ntse_plan *p = NULL;
        int rc = ntse_accel_plan_create(a, TP, NT, &pc, &p);
        ntse_plan_info info; ntse_accel_plan_info_get(p, &info);
        ref_for(NTSE_OP_DTC_EAGER, 1);
        int rr = ntse_accel_plan_run(a, p, xc, Y, 1, NULL);
        expect("1 MiB device budget: host-only plan, CPU output exact, no sticky fault",
               rc == NTSE_OK && !info.on_device && rr == NTSE_OK && same(Y, R, 1)
               && !ntse_accel_failed(a));
        ntse_accel_plan_free(a, p);
        ntse_accel_close(a);
    }
    for (int i = 0; i < NT; i++) { fixture_free(&F[i]); free(X[i]); free(Y[i]); free(R[i]); free(Z[i]); }
    free(bookcopy);
    printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
