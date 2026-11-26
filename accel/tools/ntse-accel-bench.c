/* ntse-accel-bench.c - benchmark receipts for the accel lanes.
 * Co-implemented by Cortana for JD's research lanes (Gizmo's lane
 * 1-4 measurement list). Every timing is end-to-end per call
 * (activation upload, kernels, result download) unless marked.
 *
 *   ntse-accel-bench [--lane L] [--reps N] [--cpu] [--sections abcdp]
 *                    [--pack v1.nq2 v1.nq2book shard...]
 *
 * Sections:  a operators on model-shaped tensors
 *            b break-even map over output rows M (fixed N, K, B)
 *            c activation batch sweep
 *            d dictionary size / block sweep
 *            s streaming (Tensor Handshake) vs resident, slots 1..3
 *            p full pass over every packed 2-D tensor of a real v1
 *              pack (resident, then streamed): weight-side compute
 *              per token, the linear-layer ceiling only
 * Output lines starting with "R\t" are TSV receipts.
 */
#define main nq2dc_probe_main
#include "../../tools/nq2-dc.c"
#undef main
#include "../include/ntse_accel.h"
#include "../tests/test_common.h"

#ifdef _WIN32
#include <windows.h>
static double tms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
}
#else
static double tms(void) { return now_ms(); }
#endif

static int g_reps = 20, g_cpu = 0;

static int cmpd(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* median of reps timings of one matvec; returns -1 on error */
static double time_op(ntse_accel *a, ntse_dtensor *d, ntse_op op, const float *x,
                      uint32_t batch, float *y, int reps, ntse_lane *used)
{
    double t[256];
    if (reps > 256) reps = 256;
    if (ntse_accel_matvec(a, d, op, x, batch, y, used) != NTSE_OK) return -1;  /* warm */
    for (int r = 0; r < reps; r++) {
        double t0 = tms();
        if (ntse_accel_matvec(a, d, op, x, batch, y, used) != NTSE_OK) return -1;
        t[r] = tms() - t0;
    }
    qsort(t, (size_t)reps, sizeof t[0], cmpd);
    return t[reps / 2];
}

static double time_cpu(const ntse_dtc_tensor *t, ntse_op op, const float *x,
                       uint32_t batch, float *y, int reps)
{
    double best[64];
    if (reps > 64) reps = 64;
    for (int r = 0; r < reps; r++) {
        double t0 = tms();
        ntse_cpu_matvec(t, op, x, batch, y);
        best[r] = tms() - t0;
    }
    qsort(best, (size_t)reps, sizeof best[0], cmpd);
    return best[reps / 2];
}

static void one_shape(ntse_accel *a, const char *sec, uint64_t M, uint64_t N,
                      uint32_t B, uint32_t K, uint32_t active, uint32_t batch)
{
    fixture f;
    if (fixture_make(&f, M, N, B, K, active, 17 + M + N + K) != 0) return;
    float *x = activations((uint64_t)batch * N, 3);
    float *y = (float *)malloc((uint64_t)batch * M * 4), *ref = (float *)malloc((uint64_t)batch * M * 4);
    ntse_dtensor *d = NULL;
    if (!x || !y || !ref || ntse_accel_upload(a, &f.t, &d) != NTSE_OK) goto out;
    for (int op = 0; op <= NTSE_OP_DTC_LAZY; op++) {
        ntse_lane used = NTSE_LANE_CPU;
        double g = time_op(a, d, (ntse_op)op, x, batch, y, g_reps, &used);
        double c = -1;
        ntse_cpu_matvec(&f.t, (ntse_op)op, x, batch, ref);
        metrics m = compare_f(y, ref, (uint64_t)batch * M);
        if (g_cpu) c = time_cpu(&f.t, (ntse_op)op, x, batch, ref, op == NTSE_OP_EXPANDED ? 3 : 5);
        double gw = (double)M * N * batch / (g > 0 ? g * 1e-3 : 1) / 1e9;
        printf("R\t%s\t%s\t%s\tM=%llu\tN=%llu\tB=%u\tK=%u\tact=%u\tbatch=%u\t"
               "ms=%.4f\tGweights/s=%.2f\tcpu_ms=%.4f\trelL2=%.3e\tdiffer=%llu\n",
               sec, ntse_lane_name(used), ntse_op_name((ntse_op)op),
               (unsigned long long)M, (unsigned long long)N, B, K, active, batch,
               g, gw, c, m.rel_l2, (unsigned long long)m.differ);
        fflush(stdout);
    }
out:
    ntse_accel_release(a, d);
    free(x); free(y); free(ref);
    fixture_free(&f);
}

static void sec_stream(ntse_accel *a)
{
    enum { NS = 24 };
    fixture f[NS];
    float *x[NS], *y[NS];
    ntse_stream_item it[NS];
    for (int i = 0; i < NS; i++) {
        fixture_make(&f[i], 4096, 4096, 256, 256, 90, 300 + i);
        x[i] = activations(4096, 40 + i);
        y[i] = (float *)malloc(4096 * 4);
        it[i].t = &f[i].t; it[i].x = x[i]; it[i].y = y[i];
    }
    /* resident: all uploaded once, then one matvec each */
    ntse_dtensor *d[NS];
    int up = 1;
    for (int i = 0; i < NS; i++) if (ntse_accel_upload(a, &f[i].t, &d[i]) != NTSE_OK) up = 0;
    if (up) {
        double ts[16];
        for (int r = 0; r < 9; r++) {
            double t0 = tms();
            for (int i = 0; i < NS; i++) ntse_accel_matvec(a, d[i], NTSE_OP_DTC_EAGER, x[i], 1, y[i], NULL);
            ts[r] = tms() - t0;
        }
        qsort(ts, 9, sizeof ts[0], cmpd);
        printf("R\ts\t%s\tresident\ttensors=%d\t4096x4096\tms_per_pass=%.3f\tpeak_dev=%llu\n",
               ntse_lane_name(ntse_accel_lane(a)), NS, ts[4],
               (unsigned long long)ntse_accel_peak(a));
    }
    for (int i = 0; i < NS; i++) ntse_accel_release(a, d[i]);
    for (uint32_t slots = 1; slots <= 3; slots++) {
        for (int thr = 0; thr <= 1; thr++) {
            ntse_stream_config sc = { slots, NTSE_OP_DTC_EAGER, thr };
            ntse_stream_stats st;
            double ts[16];
            int ok = 1;
            for (int r = 0; r < 5; r++) {
                if (ntse_accel_stream(a, it, NS, 1, &sc, &st) != NTSE_OK) ok = 0;
                ts[r] = st.wall_ms;
            }
            qsort(ts, 5, sizeof ts[0], cmpd);
            printf("R\ts\t%s\tstream\tslots=%u\tstager=%s\ttensors=%d\tms_per_pass=%.3f\t"
                   "h2d_bytes=%llu\th2d_per_tensor=%llu\tviolations=%llu\tstager_waits=%llu\t"
                   "driver_waits=%llu\tok=%d\n", ntse_lane_name(st.lane), slots,
                   thr ? "thread" : "inline", NS, ts[2], (unsigned long long)st.h2d_bytes,
                   (unsigned long long)(st.h2d_bytes / NS),
                   (unsigned long long)st.ownership_violations,
                   (unsigned long long)st.stager_waits, (unsigned long long)st.driver_waits, ok);
            fflush(stdout);
        }
    }
    for (int i = 0; i < NS; i++) { free(x[i]); free(y[i]); fixture_free(&f[i]); }
}

/* Section p: every packed 2-D tensor of a v1 pack, in stream order. */
static void sec_pass(ntse_accel *a, int npos, const char **pos)
{
    FILE *sf = fopen(pos[0], "rb");
    if (!sf) { printf("SKIP p: cannot open pack\n"); return; }
    fseeko(sf, 0, SEEK_END); off_t ssz = ftello(sf); fseeko(sf, 0, SEEK_SET);
    uint8_t *stream = (uint8_t *)malloc((size_t)ssz);
    if (!stream || fread(stream, 1, (size_t)ssz, sf) != (size_t)ssz) { fclose(sf); return; }
    fclose(sf);
    FILE *bf = fopen(pos[1], "rb");
    if (!bf) return;
    uint32_t block = rd32(stream + 8), entries = rd32(stream + 12);
    uint64_t n_blocks = rd64(stream + 24);
    if (memcmp(stream, "NQ2\0", 4) || rd32(stream + 4) != 1 || block == 0 || block > 4096
        || entries == 0 || entries > 256 || (uint64_t)ssz != 32 + 5 * n_blocks) {
        printf("SKIP p: not a valid v1 pack\n");
        fclose(bf); free(stream);
        return;
    }
    float *book = (float *)malloc((size_t)entries * block * 4);
    uint8_t hdr[12];
    if (fread(hdr, 1, 12, bf) != 12 || memcmp(hdr, "NQ2B", 4) || rd32(hdr + 4) != block
        || rd32(hdr + 8) != entries
        || fread(book, 4, (size_t)entries * block, bf) != (size_t)entries * block) {
        printf("SKIP p: book mismatch\n");
        fclose(bf); free(stream); free(book);
        return;
    }
    fclose(bf);
    enum { MAXT = 1024 };
    static ntse_dtc_tensor ts[MAXT];
    static uint16_t *mns[MAXT], *scs[MAXT];
    int nt = 0;
    uint64_t cursor = 0, weights = 0, skipped_w = 0;
    int skipped = 0;
    for (int s = 2; s < npos; s++) {
        Shard sh;
        if (shard_open(&sh, pos[s]) != 0) { printf("SKIP p: shard refused\n"); return; }
        for (int ti = 0; ti < sh.n_tensors; ti++) {
            TensorInfo *t = &sh.tensors[ti];
            uint64_t nb = t->count / block;
            /* embedding tables are row lookups per token, not matvecs:
             * leaving them in would overstate per-token weight work */
            int lookup = strstr(t->name, "embed_tokens") || strstr(t->name, "embedding_matrix");
            if (lookup && t->ndims == 2) { skipped++; skipped_w += t->count; }
            if (!lookup && t->ndims == 2 && t->dims[1] % block == 0 && t->count % block == 0
                && nt < MAXT && cursor + nb <= n_blocks) {
                uint16_t *mn = (uint16_t *)malloc(nb * 2), *sc = (uint16_t *)malloc(nb * 2);
                for (uint64_t b = 0; b < nb; b++) {
                    mn[b] = rd16(stream + 32 + n_blocks + 2 * (cursor + b));
                    sc[b] = rd16(stream + 32 + 3 * n_blocks + 2 * (cursor + b));
                }
                ntse_dtc_tensor tt = { t->dims[0], t->dims[1], block, entries, book,
                                       stream + 32 + cursor, mn, sc };
                mns[nt] = mn; scs[nt] = sc; ts[nt++] = tt;
                weights += t->count;
            }
            cursor += nb;
        }
        fclose(sh.file);
        shard_free_meta(&sh);
    }
    if (cursor != n_blocks) { printf("SKIP p: walk does not reconcile\n"); return; }
    float **x = (float **)calloc((size_t)nt, sizeof(float *)), **y = (float **)calloc((size_t)nt, sizeof(float *));
    ntse_stream_item *it = (ntse_stream_item *)calloc((size_t)nt, sizeof *it);
    for (int i = 0; i < nt; i++) {
        x[i] = activations(ts[i].cols, 11 + i);
        y[i] = (float *)malloc(ts[i].rows * 4);
        it[i].t = &ts[i]; it[i].x = x[i]; it[i].y = y[i];
    }
    printf("     pass: %d packed 2-D tensors, %.3f G weights (%d embedding lookups, "
           "%.3f G weights, excluded)\n", nt, weights / 1e9, skipped, skipped_w / 1e9);
    /* CPU reference outputs once, for the divergence column */
    float **ref = (float **)calloc((size_t)nt, sizeof(float *));
    for (int i = 0; i < nt; i++) {
        ref[i] = (float *)malloc(ts[i].rows * 4);
        ntse_cpu_matvec(&ts[i], NTSE_OP_DTC_EAGER, x[i], 1, ref[i]);
    }
    ntse_dtensor **d = (ntse_dtensor **)calloc((size_t)nt, sizeof *d);
    int all = 1;
    for (int i = 0; i < nt; i++) if (ntse_accel_upload(a, &ts[i], &d[i]) != NTSE_OK) all = 0;
    printf("     resident device bytes: %llu (budget %llu)\n",
           (unsigned long long)ntse_accel_charged(a), (unsigned long long)ntse_accel_budget(a));
    for (int op = 0; op <= NTSE_OP_DTC_LAZY && all; op++) {
        if (op == NTSE_OP_EXPANDED) continue;
        double tt[16];
        int on_cpu = 0;   /* tensors this pass ran on the CPU (host-only or refused) */
        uint64_t h0 = 0;
        /* untimed warm-up pass: the first timed op otherwise pays the
         * GPU's clock ramp from idle (seen: 15.1 vs 12.2 ms) */
        for (int i = 0; i < nt; i++) ntse_accel_matvec(a, d[i], (ntse_op)op, x[i], 1, y[i], NULL);
        for (int r = 0; r < 5; r++) {
            if (r == 4) h0 = ntse_accel_h2d_bytes(a);
            int cpu_r = 0;
            double t0 = tms();
            for (int i = 0; i < nt; i++) {
                ntse_lane used = NTSE_LANE_CPU;
                ntse_accel_matvec(a, d[i], (ntse_op)op, x[i], 1, y[i], &used);
                if (used == NTSE_LANE_CPU) cpu_r++;
            }
            tt[r] = tms() - t0;
            if (cpu_r > on_cpu) on_cpu = cpu_r;
        }
        uint64_t h2d_pass = ntse_accel_h2d_bytes(a) - h0;
        uint64_t differ = 0; double worst = 0;
        for (int i = 0; i < nt; i++) {
            metrics m = compare_f(y[i], ref[i], ts[i].rows);
            differ += m.differ; if (m.rel_l2 > worst) worst = m.rel_l2;
        }
        qsort(tt, 5, sizeof tt[0], cmpd);
        printf("R\tp\t%s\tresident\t%s\ttensors=%d\ton_cpu=%d\tGweights=%.3f\tms_per_pass=%.3f\t"
               "linear_tok_s_ceiling=%.2f\th2d_bytes_per_token=%llu\tdiffer_vs_cpu=%llu\t"
               "worst_relL2=%.3e\n", ntse_lane_name(ntse_accel_lane(a)),
               ntse_op_name((ntse_op)op), nt, on_cpu, weights / 1e9, tt[2], 1000.0 / tt[2],
               (unsigned long long)h2d_pass, (unsigned long long)differ, worst);
        fflush(stdout);
    }
    for (int i = 0; i < nt; i++) ntse_accel_release(a, d[i]);
    for (uint32_t slots = 2; slots <= 3; slots++) {
        ntse_stream_config sc = { slots, NTSE_OP_DTC_EAGER, 1 };
        ntse_stream_stats st;
        double tt[8];
        int src = 0;
        for (int r = 0; r < 3; r++) {
            src |= ntse_accel_stream(a, it, (uint32_t)nt, 1, &sc, &st);
            tt[r] = st.wall_ms;
        }
        uint64_t differ = 0;
        for (int i = 0; i < nt; i++) differ += compare_f(y[i], ref[i], ts[i].rows).differ;
        qsort(tt, 3, sizeof tt[0], cmpd);
        printf("R\tp\t%s\tstream\tdtc-eager\tslots=%u\ttensors=%d\tms_per_pass=%.3f\t"
               "linear_tok_s_ceiling=%.2f\th2d_bytes_per_token=%llu\tviolations=%llu\t"
               "differ_vs_cpu=%llu\trc=%d\n",
               ntse_lane_name(st.lane), slots, nt, tt[1], 1000.0 / tt[1],
               (unsigned long long)st.h2d_bytes, (unsigned long long)st.ownership_violations,
               (unsigned long long)differ, src);
        fflush(stdout);
    }
    /* Lane 2: prepared plans (layout once, books resident once) */
    if (ntse_accel_lane(a) != NTSE_LANE_CPU) {
        double hp = 0, hg = 0, dp = 0;
        if (ntse_accel_probe_bandwidth(a, 64ull << 20, &hp, &hg, &dp) == NTSE_OK)
            printf("R\tp\t%s\tpcie\t64MiB\th2d_pinned_GBs=%.2f\th2d_pageable_GBs=%.2f\t"
                   "d2h_pinned_GBs=%.2f\n", ntse_lane_name(ntse_accel_lane(a)), hp, hg, dp);
        const ntse_dtc_tensor **tp = (const ntse_dtc_tensor **)calloc((size_t)nt, sizeof *tp);
        for (int i = 0; i < nt; i++) tp[i] = &ts[i];
        static const ntse_op pops[3] = { NTSE_OP_DTC_EAGER, NTSE_OP_DTC_ACTIVE, NTSE_OP_DTC_LAZY };
        for (int pin = 1; pin >= 0; pin--)
            for (int oi = 0; oi < 3; oi++)
                for (uint32_t slots = 1; slots <= 3; slots++) {
                    ntse_plan_config pc;
                    ntse_plan_config_default(&pc);
                    pc.slots = slots; pc.op = pops[oi]; pc.pin_image = pin;
                    ntse_plan *pl = NULL;
                    if (ntse_accel_plan_create(a, tp, (uint32_t)nt, &pc, &pl) != NTSE_OK) continue;
                    ntse_plan_info info;
                    ntse_accel_plan_info_get(pl, &info);
                    ntse_stream_stats st;
                    double tt[8];
                    int prc = 0;
                    ntse_accel_plan_run(a, pl, (const float *const *)x, y, 1, &st);  /* warm */
                    for (int r = 0; r < 7; r++) {
                        double t0 = tms();
                        prc |= ntse_accel_plan_run(a, pl, (const float *const *)x, y, 1, &st);
                        tt[r] = tms() - t0;
                    }
                    qsort(tt, 7, sizeof tt[0], cmpd);
                    uint64_t differ = 0;
                    for (int i = 0; i < nt; i++) differ += compare_f(y[i], ref[i], ts[i].rows).differ;
                    printf("R\tp\t%s\tplan\t%s\t%s\tslots=%u\ttensors=%d\tms_per_pass=%.3f\t"
                           "linear_tok_s_ceiling=%.2f\th2d_bytes_per_token=%llu\tbooks=%u\t"
                           "book_bytes_resident=%llu\tpinned_bytes=%llu\tslot_dev_bytes=%llu\tcreate_ms=%.1f\t"
                           "on_device=%d\tviolations=%llu\tdiffer_vs_cpu=%llu\trc=%d\n",
                           ntse_lane_name(st.lane), ntse_op_name(pops[oi]),
                           info.pinned_image ? "pinned-image" : "pageable-image", slots, nt,
                           tt[3], 1000.0 / tt[3], (unsigned long long)st.h2d_bytes, info.books,
                           (unsigned long long)info.book_bytes, (unsigned long long)info.pinned_bytes,
                           (unsigned long long)info.slot_dev_bytes,
                           info.create_ms, info.on_device,
                           (unsigned long long)st.ownership_violations,
                           (unsigned long long)differ, prc);
                    fflush(stdout);
                    ntse_accel_plan_free(a, pl);
                }
        free(tp);
    }
    if (g_cpu) {
        double t0 = tms();
        for (int i = 0; i < nt; i++) ntse_cpu_matvec(&ts[i], NTSE_OP_DTC_EAGER, x[i], 1, y[i]);
        double c = tms() - t0;
        printf("R\tp\tcpu\treference\tdtc-eager\ttensors=%d\tms_per_pass=%.3f\t"
               "linear_tok_s_ceiling=%.2f\t(single thread)\n", nt, c, 1000.0 / c);
    }
    for (int i = 0; i < nt; i++) { free(x[i]); free(y[i]); free(ref[i]); free(mns[i]); free(scs[i]); }
    free(ref); free(x); free(y); free(it); free(d); free(stream); free(book);
}

int main(int argc, char **argv)
{
    ntse_accel_config cfg;
    ntse_accel_config_default(&cfg);
    cfg.verbose = 1;
    const char *secs = "abcds";
    const char *pos[24];
    int npos = 0, pack = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--lane") && i + 1 < argc) {
            i++;
            if (!strcmp(argv[i], "cuda")) cfg.lane = NTSE_LANE_CUDA;
            else if (!strcmp(argv[i], "igpu")) cfg.lane = NTSE_LANE_CL_IGPU;
            else if (!strcmp(argv[i], "gpu-cl")) cfg.lane = NTSE_LANE_CL_GPU;
            else if (!strcmp(argv[i], "cpu")) cfg.lane = NTSE_LANE_CPU;
        } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu")) g_cpu = 1;
        else if (!strcmp(argv[i], "--sections") && i + 1 < argc) secs = argv[++i];
        else if (!strcmp(argv[i], "--pack")) pack = 1;
        else if (pack && npos < 24) pos[npos++] = argv[i];
    }
    if (g_reps < 1) g_reps = 1;
    ntse_accel *a = NULL;
    if (ntse_accel_open(&a, &cfg) != NTSE_OK) return 1;
    printf("R\thw\t%s\t%s\tbudget_MiB=%llu\n", ntse_lane_name(ntse_accel_lane(a)),
           ntse_accel_device(a), (unsigned long long)(ntse_accel_budget(a) >> 20));
    if (strchr(secs, 'a')) {
        /* Sigma/Little Nemo-like projection shapes */
        one_shape(a, "a", 3072, 3072, 256, 256, 90, 1);
        one_shape(a, "a", 8192, 3072, 256, 256, 90, 1);
        one_shape(a, "a", 3072, 8192, 256, 256, 90, 1);
        one_shape(a, "a", 4096, 4096, 256, 256, 90, 1);
        one_shape(a, "a", 14336, 4096, 256, 256, 90, 1);
    }
    if (strchr(secs, 'b')) {
        static const uint64_t Ms[] = { 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
        for (size_t i = 0; i < sizeof Ms / sizeof Ms[0]; i++)
            one_shape(a, "b", Ms[i], 4096, 256, 256, 256, 1);
    }
    if (strchr(secs, 'c')) {
        static const uint32_t bs[] = { 1, 2, 4, 8, 16, 32, 64 };
        for (size_t i = 0; i < sizeof bs / sizeof bs[0]; i++)
            one_shape(a, "c", 4096, 4096, 256, 256, 90, bs[i]);
    }
    if (strchr(secs, 'd')) {
        static const uint32_t Bs[] = { 64, 128, 256, 512 };
        static const uint32_t Ks[] = { 16, 64, 256 };
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 3; j++)
                one_shape(a, "d", 4096, 4096, Bs[i], Ks[j], Ks[j], 1);
    }
    if (strchr(secs, 's')) sec_stream(a);
    if (strchr(secs, 'p') && npos >= 3) sec_pass(a, npos, pos);
    printf("R\tpeak\tdevice_bytes_charged=%llu\tsticky_fault=%d\n",
           (unsigned long long)ntse_accel_peak(a), ntse_accel_failed(a));
    ntse_accel_close(a);
    return 0;
}
