/* test_accel_stream.c - the GPU side of the Tensor Handshake.
 * Co-implemented by Cortana; the Handshake design is JD's, the
 * engines are Elfie's, the lane proposal Gizmo's.
 *
 * Streams a sequence of differently shaped tensors through 1, 2
 * and 3 pinned slots, inline and with the stager thread, for each
 * DTC op, and checks:
 *   - every output equals the CPU reference (gate: rel L2), and
 *     equals the resident (non-streamed) GPU result bit for bit;
 *   - zero ownership violations on every configuration;
 *   - repeated streams are bit-identical (determinism);
 *   - the early-recycle test hook is refused by the guard (exactly
 *     one violation refused) and the output is still correct.
 */
#include "test_common.h"

#define NI 12

int main(int argc, char **argv)
{
    ntse_lane lane = NTSE_LANE_AUTO;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lane") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) lane = NTSE_LANE_CUDA;
            else if (!strcmp(argv[i], "igpu")) lane = NTSE_LANE_CL_IGPU;
            else if (!strcmp(argv[i], "gpu-cl")) lane = NTSE_LANE_CL_GPU;
            else if (!strcmp(argv[i], "cpu")) lane = NTSE_LANE_CPU;
        }
    }
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.lane = lane;
    cfg.verbose = 1;
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK) { printf("FAIL open\n"); return 1; }
    printf("== stream lane %s: %s\n", ntse_lane_name(ntse_accel_lane(a)), ntse_accel_device(a));

    const uint32_t batch = 2;
    fixture f[NI];
    float *x[NI], *ref[NI], *res[NI], *y[NI];
    ntse_stream_item items[NI];
    for (int i = 0; i < NI; i++) {
        uint64_t rows = 128 + 97 * (uint64_t)i, cols = 256 * (1 + (i % 5));
        if (fixture_make(&f[i], rows, cols, 256, 256, 60 + i, 500 + i) != 0) {
            printf("FAIL fixture\n");
            return 1;
        }
        x[i] = activations(batch * cols, 900 + i);
        ref[i] = (float *)malloc(batch * rows * 4);
        res[i] = (float *)malloc(batch * rows * 4);
        y[i] = (float *)malloc(batch * rows * 4);
        items[i].t = &f[i].t; items[i].x = x[i]; items[i].y = y[i];
    }
    const ntse_op ops[3] = { NTSE_OP_DTC_EAGER, NTSE_OP_DTC_ACTIVE, NTSE_OP_DTC_DIRECT };
    for (int oi = 0; oi < 3; oi++) {
        ntse_op op = ops[oi];
        /* references: CPU kernel and resident GPU result per item */
        for (int i = 0; i < NI; i++) {
            ntse_cpu_matvec(&f[i].t, op, x[i], batch, ref[i]);
            ntse_dtensor *d = NULL;
            ntse_accel_upload(a, &f[i].t, &d);
            ntse_accel_matvec(a, d, op, x[i], batch, res[i], NULL);
            ntse_accel_release(a, d);
        }
        for (uint32_t slots = 1; slots <= 3; slots++) {
            for (int thr = 0; thr <= 1; thr++) {
                for (int i = 0; i < NI; i++) memset(y[i], 0xff, batch * f[i].t.rows * 4);
                ntse_stream_config sc = { slots, op, thr };
                ntse_stream_stats st;
                int rc = ntse_accel_stream(a, items, NI, batch, &sc, &st);
                double worst = 0;
                int same_resident = 1;
                for (int i = 0; i < NI; i++) {
                    uint64_t ny = batch * f[i].t.rows;
                    metrics m = compare_f(y[i], ref[i], ny);
                    if (m.nonfinite) worst = 1e9;
                    if (m.rel_l2 > worst) worst = m.rel_l2;
                    if (memcmp(y[i], res[i], ny * 4) != 0) same_resident = 0;
                }
                char name[200];
                snprintf(name, sizeof name, "%s slots %u stager %s: output = reference, "
                         "0 violations", ntse_op_name(op), slots, thr ? "thread" : "inline");
                expect(name, rc == NTSE_OK && worst <= TOL_DTC_RELL2
                       && st.ownership_violations == 0);
                printf("     lane %s worst relL2 %.3e, identical to resident %s, "
                       "h2d %llu B, slot %llu B, stager waits %llu, driver waits %llu, "
                       "%.2f ms\n", ntse_lane_name(st.lane), worst,
                       same_resident ? "yes" : "NO",
                       (unsigned long long)st.h2d_bytes, (unsigned long long)st.slot_bytes,
                       (unsigned long long)st.stager_waits, (unsigned long long)st.driver_waits,
                       st.wall_ms);
                expect("streamed output identical to resident GPU output", same_resident);
            }
        }
        /* determinism: two triple-buffered threaded streams */
        ntse_stream_config sc = { 3, op, 1 };
        ntse_accel_stream(a, items, NI, batch, &sc, NULL);
        float *keep[NI];
        for (int i = 0; i < NI; i++) {
            keep[i] = (float *)malloc(batch * f[i].t.rows * 4);
            memcpy(keep[i], y[i], batch * f[i].t.rows * 4);
        }
        ntse_accel_stream(a, items, NI, batch, &sc, NULL);
        int det = 1;
        for (int i = 0; i < NI; i++) {
            if (memcmp(keep[i], y[i], batch * f[i].t.rows * 4) != 0) det = 0;
            free(keep[i]);
        }
        char name[120];
        snprintf(name, sizeof name, "%s repeated streams bit-identical", ntse_op_name(op));
        expect(name, det);
    }
    if (ntse_accel_lane(a) != NTSE_LANE_CPU) {
        ntse_accel_test_force_early_recycle(a, 1);
        ntse_stream_config sc = { 2, NTSE_OP_DTC_EAGER, 1 };
        ntse_stream_stats st;
        for (int i = 0; i < NI; i++)
            ntse_cpu_matvec(&f[i].t, NTSE_OP_DTC_EAGER, x[i], batch, ref[i]);
        int rc = ntse_accel_stream(a, items, NI, batch, &sc, &st);
        double worst = 0;
        for (int i = 0; i < NI; i++) {
            metrics m = compare_f(y[i], ref[i], batch * f[i].t.rows);
            if (m.rel_l2 > worst || m.nonfinite) worst = m.nonfinite ? 1e9 : m.rel_l2;
        }
        expect("early recycle attempt refused by the guard (1 violation refused)",
               rc == NTSE_OK && st.ownership_violations == 1);
        expect("output still correct after the refused recycle", worst <= TOL_DTC_RELL2);
    } else {
        printf("SKIP early-recycle guard: cpu lane has no device buffers\n");
    }
    expect("no sticky device fault", !ntse_accel_failed(a));
    for (int i = 0; i < NI; i++) {
        free(x[i]); free(ref[i]); free(res[i]); free(y[i]);
        fixture_free(&f[i]);
    }
    ntse_accel_close(a);
    printf("RESULT: %d/%d checks passed\n", g_checks - g_fail, g_checks);
    return g_fail ? 1 : 0;
}
