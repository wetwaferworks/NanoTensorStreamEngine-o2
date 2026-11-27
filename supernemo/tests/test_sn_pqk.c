/* test_sn_pqk.c - multi-vector / resident / DTC kernels on real-shaped packed tensors.
 *   test_sn_pqk PREFIX SHAPES [name-substring ...]
 * Checks: direct multi-vector == single-vector == the legacy sn_pq_rows (bit-identical); DTC vs direct (float noise);
 * DTC independent of vector count, row split and file/RAM source (bit-identical); and prints single-lane timings. */
#include "../src/sn_pqk.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
static uint64_t rng_s = 88172645463325252ull;
static float rnd(void) { rng_s ^= rng_s << 13; rng_s ^= rng_s >> 7; rng_s ^= rng_s << 17; return (float)((double)(rng_s >> 11) / 9007199254740992.0 * 2.0 - 1.0); }

static int bits_equal(const float *a, const float *b, size_t n) { return memcmp(a, b, n * sizeof(float)) == 0; }
static double relerr(const float *a, const float *b, size_t n)
{
    double d = 0, s = 0;
    for (size_t i = 0; i < n; i++) { double e = (double)a[i] - b[i]; d += e * e; s += (double)b[i] * b[i]; }
    return sqrt(d / (s > 0 ? s : 1));
}

static void prefix(const float *x, int cols, double *P) { P[0] = 0; for (int j = 0; j < cols; j++) P[j + 1] = P[j] + (double)x[j]; }

static void one(const sn_pack *p, sn_rd *rd, const sn_tensor *t, int cols, int bench)
{
    int rows = (int)(t->count / (uint64_t)cols);
    printf("-- %s  [%d x %d]\n", t->name, rows, cols);
    int nph = 0, ng = 0;
    CHECK(sn_dtc_geom(p, cols, &nph, &ng) == 0, "geom");
    float *book = (float *)malloc(p->book_floats * 4);
    CHECK(sn_load_book(p, rd, t, book) == 0, "book");
    enum { NV = 3 };
    float *x[NV]; double *P[NV]; float *yd[NV], *yl[NV], *yt[NV], *yt1[NV], *yr[NV];
    for (int v = 0; v < NV; v++) {
        x[v] = (float *)malloc((size_t)cols * 4); P[v] = (double *)malloc(((size_t)cols + 1) * 8);
        for (int j = 0; j < cols; j++) x[v][j] = rnd();
        prefix(x[v], cols, P[v]);
        yd[v] = (float *)malloc((size_t)rows * 4); yl[v] = (float *)malloc((size_t)rows * 4);
        yt[v] = (float *)malloc((size_t)rows * 4); yt1[v] = (float *)malloc((size_t)rows * 4); yr[v] = (float *)malloc((size_t)rows * 4);
    }
    size_t tf = sn_dtc_floats(p, cols);
    float *T[NV];
    for (int v = 0; v < NV; v++) { T[v] = (float *)malloc(tf * 4); sn_dtc_build(p, book, cols, x[v], 0, ng, T[v]); }
    /* a table built in two group ranges equals one built whole */
    float *T2 = (float *)malloc(tf * 4);
    sn_dtc_build(p, book, cols, x[0], 0, ng / 3, T2); sn_dtc_build(p, book, cols, x[0], ng / 3, ng, T2);
    CHECK(bits_equal(T2, T[0], tf), "split table build differs");
    free(T2);

    sn_mv md = SN_MV_INIT(NV), mt = SN_MV_INIT(NV);
    for (int v = 0; v < NV; v++) { md.x[v] = x[v]; md.P[v] = P[v]; md.y[v] = yd[v]; mt.x[v] = x[v]; mt.P[v] = P[v]; mt.T[v] = T[v]; mt.y[v] = yt[v]; }
    CHECK(sn_pqk_rows(p, rd, t, book, cols, &md, 0, rows) == 0, "direct nv=3");
    /* legacy kernel, single vector */
    for (int v = 0; v < NV; v++) {
        CHECK(sn_pq_rows(p, rd, t, book, cols, x[v], P[v], 0, rows, yl[v]) == 0, "legacy");
        CHECK(bits_equal(yd[v], yl[v], (size_t)rows), "direct multi-vector %d != legacy single", v);
    }
    CHECK(sn_pqk_rows(p, rd, t, book, cols, &mt, 0, rows) == 0, "dtc nv=3");
    for (int v = 0; v < NV; v++) {
        double e = relerr(yt[v], yd[v], (size_t)rows);
        CHECK(e < 1e-4, "dtc vs direct rel %g", e);
        if (v == 0) printf("   dtc vs direct rel L2 %.3g\n", e);
        sn_mv m1 = SN_MV_INIT(1); m1.x[0] = x[v]; m1.P[0] = P[v]; m1.T[0] = T[v]; m1.y[0] = yt1[v];
        CHECK(sn_pqk_rows(p, rd, t, book, cols, &m1, 0, rows) == 0, "dtc nv=1");
        CHECK(bits_equal(yt[v], yt1[v], (size_t)rows), "dtc vector %d depends on nv", v);
    }
    /* row split in three odd pieces, DTC and direct */
    {
        int c1 = rows / 3 + 1, c2 = 2 * rows / 3 + 3;
        int cut[4] = { 0, c1, c2, rows };
        for (int pass = 0; pass < 2; pass++) {
            for (int s = 0; s < 3; s++) {
                sn_mv ms = SN_MV_INIT(NV);
                for (int v = 0; v < NV; v++) { ms.x[v] = x[v]; ms.P[v] = P[v]; if (pass) ms.T[v] = T[v]; ms.y[v] = yr[v] + cut[s]; }
                CHECK(sn_pqk_rows(p, rd, t, book, cols, &ms, cut[s], cut[s + 1]) == 0, "split");
            }
            for (int v = 0; v < NV; v++) CHECK(bits_equal(yr[v], pass ? yt[v] : yd[v], (size_t)rows), "row split differs (pass %d)", pass);
        }
    }
    /* resident copy */
    {
        sn_pq_res res;
        CHECK(sn_res_load(p, rd, t, &res) == 0, "res load");
        for (int pass = 0; pass < 2; pass++) {
            sn_mv mr = SN_MV_INIT(NV);
            for (int v = 0; v < NV; v++) { mr.x[v] = x[v]; mr.P[v] = P[v]; if (pass) mr.T[v] = T[v]; mr.y[v] = yr[v]; }
            CHECK(sn_pqk_rows_res(p, &res, t, cols, &mr, 0, rows) == 0, "res rows");
            for (int v = 0; v < NV; v++) CHECK(bits_equal(yr[v], pass ? yt[v] : yd[v], (size_t)rows), "resident differs (pass %d)", pass);
        }
        CHECK(res.bytes == sn_res_bytes(p, t), "res bytes");
        sn_res_free(&res);
    }
    if (bench) {
        sn_mv m1d = SN_MV_INIT(1), m1t = SN_MV_INIT(1);
        m1d.x[0] = x[0]; m1d.P[0] = P[0]; m1d.y[0] = yd[0];
        m1t.x[0] = x[0]; m1t.P[0] = P[0]; m1t.T[0] = T[0]; m1t.y[0] = yt[0];
        sn_pq_res res; sn_res_load(p, rd, t, &res);
        int reps = (int)((double)(1 << 26) / ((double)t->count / 4.0)) + 1;
        double t0 = now_ms();
        for (int i = 0; i < reps; i++) sn_pqk_rows_res(p, &res, t, cols, &m1d, 0, rows);
        double td = (now_ms() - t0) / reps;
        t0 = now_ms();
        for (int i = 0; i < reps; i++) sn_pqk_rows_res(p, &res, t, cols, &m1t, 0, rows);
        double tl = (now_ms() - t0) / reps;
        t0 = now_ms();
        for (int i = 0; i < reps; i++) sn_dtc_build(p, book, cols, x[0], 0, ng, T[0]);
        double tb = (now_ms() - t0) / reps;
        printf("   1 lane, RAM-resident: direct %.3f ms | DTC lookups %.3f ms + table build %.3f ms = %.3f ms  (x%.2f)  pays=%d\n",
               td, tl, tb, tl + tb, td / (tl + tb), sn_dtc_pays(p, rows, cols, 1, 1));
        sn_res_free(&res);
    }
    for (int v = 0; v < NV; v++) { free(x[v]); free(P[v]); free(yd[v]); free(yl[v]); free(yt[v]); free(yt1[v]); free(yr[v]); free(T[v]); }
    free(book);
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_sn_pqk PREFIX SHAPES [name-substring ...] [--bench]\n"); return 2; }
    int bench = 0;
    sn_pack p;
    if (sn_pack_open(&p, argv[1], argv[2]) != 0) return 2;
    sn_rd rd;
    if (sn_rd_open(&p, &rd) != 0) return 2;
    for (int i = 0; i < p.n; i++) {
        const sn_tensor *t = &p.t[i];
        if (t->raw || t->cols == 0) continue;
        int nnames = 0; for (int a = 3; a < argc; a++) if (strcmp(argv[a], "--bench")) nnames++;
        int want = nnames == 0;
        for (int a = 3; a < argc; a++) { if (!strcmp(argv[a], "--bench")) bench = 1; else if (strstr(t->name, argv[a])) want = 1; }
        if (want) one(&p, &rd, t, (int)t->cols, bench);
    }
    printf("%d checks, %d failed\n", checks, fails);
    sn_rd_close(&rd); sn_pack_close(&p);
    return fails ? 1 : 0;
}
