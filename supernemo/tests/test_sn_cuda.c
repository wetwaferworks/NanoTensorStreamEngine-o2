/* test_sn_cuda.c - GPU matvec against the CPU fused kernel on every tensor of a pack that the lane can take.
 *   test_sn_cuda PACK SHAPES [budget_mb] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
#include "../src/sn_cuda.h"
#include "../src/sn_pqk.h"

static unsigned rs = 5;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((float)((rs >> 8) & 0xffff) / 32768.0f - 1.0f); }

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_sn_cuda PACK SHAPES [budget_mb]\n"); return 2; }
    static sn_pack pk; sn_rd rd;
    if (sn_pack_open(&pk, argv[1], argv[2]) != 0 || sn_rd_open(&pk, &rd) != 0) return 2;
    char why[400];
    sn_cuda *c = sn_cuda_open((long long)(argc > 3 ? atoi(argv[3]) : 1500) << 20, pk.n, why, sizeof why);
    if (!c) { printf("no CUDA lane: %s\n", why); return 3; }
    printf("GPU: %s\n", sn_cuda_name(c));
    float *book = (float *)malloc(pk.book_floats * 4);
    int tested = 0, bad = 0, skipped = 0, dtc_exact = 0, cached = 0; double worst = 0;
    for (int i = 0; i < pk.n; i++) {
        const sn_tensor *t = &pk.t[i];
        int cols = (int)t->cols; long rows = cols ? (long)(t->count / (uint64_t)cols) : 0;
        if (cols < 8 || (cols & 3) || rows < 2 || rows > 140000 || cols > 12000 || t->count != (uint64_t)cols * (uint64_t)rows) { skipped++; continue; }
        if (sn_cuda_load(c, &pk, &rd, t) != 0) { skipped++; continue; }
        float *x = (float *)malloc((size_t)cols * 4), *yc = (float *)malloc((size_t)rows * 4), *yg = (float *)malloc((size_t)rows * 4);
        double *P = (double *)malloc(((size_t)cols + 1) * 8);
        for (int k = 0; k < cols; k++) x[k] = rnd();
        P[0] = 0; for (int k = 0; k < cols; k++) P[k + 1] = P[k] + x[k];
        int rc = t->raw ? sn_raw_rows(&pk, &rd, t, cols, x, 0, (int)rows, yc)
                        : (sn_load_book(&pk, &rd, t, book) || sn_pq_rows(&pk, &rd, t, book, cols, x, P, 0, (int)rows, yc));
        if (rc) { printf("CPU reference failed for %s\n", t->name); bad++; goto next; }
        if (sn_cuda_matvec(c, &pk, t, cols, (int)rows, x, yg) != 0) { printf("GPU matvec failed for %s\n", t->name); bad++; goto next; }
        double num = 0, den = 0;
        for (long r = 0; r < rows; r++) { double d = (double)yg[r] - yc[r]; num += d * d; den += (double)yc[r] * yc[r]; }
        double rel = sqrt(num / (den > 0 ? den : 1));
        if (rel > worst) worst = rel;
        if (rel > 1e-4) { printf("MISMATCH %s: rel %.3g\n", t->name, rel); bad++; }
        tested++;
        if (!t->raw && cols <= 8192 && sn_dtc_geom(&pk, cols, NULL, NULL) == 0) {
            /* DTC: the GPU tables+lookups must equal the CPU DTC kernel BIT FOR BIT (dense-resident and cache-resident) */
            size_t tf = sn_dtc_floats(&pk, cols);
            float *T = (float *)malloc(tf * 4), *ydc = (float *)malloc((size_t)rows * 4), *ydg = (float *)malloc((size_t)rows * 4);
            sn_dtc_build(&pk, book, cols, x, 0, cols >> 2, T);
            sn_mv mv = SN_MV_INIT(1); mv.x[0] = x; mv.P[0] = P; mv.T[0] = T; mv.y[0] = ydc;
            int ok = sn_pqk_rows(&pk, &rd, t, book, cols, &mv, 0, (int)rows) == 0 && sn_cuda_dtc_rows(c, &pk, t, cols, (int)rows, x, P, ydg) == 0;
            if (!ok || memcmp(ydc, ydg, (size_t)rows * 4) != 0) { printf("DTC GPU != CPU (bits) for %s\n", t->name); bad++; }
            else {
                dtc_exact++;
                int reps = 20; double t0 = now_ms();
                for (int q = 0; q < reps; q++) sn_cuda_matvec(c, &pk, t, cols, (int)rows, x, ydg);
                double t_old = (now_ms() - t0) / reps; t0 = now_ms();
                for (int q = 0; q < reps; q++) sn_cuda_dtc_rows(c, &pk, t, cols, (int)rows, x, P, ydg);
                double t_dtc = (now_ms() - t0) / reps; t0 = now_ms();
                for (int q = 0; q < reps; q++) { sn_dtc_build(&pk, book, cols, x, 0, cols >> 2, T); sn_pqk_rows(&pk, &rd, t, book, cols, &mv, 0, (int)rows); }
                double t_cpu = (now_ms() - t0) / reps;
                printf("   %-62s [%d x %d]  GPU direct %.3f ms | GPU DTC %.3f ms | CPU DTC 1 lane (file-backed) %.2f ms\n", t->name + (strlen(t->name) > 62 ? strlen(t->name) - 62 : 0), (int)rows, cols, t_old, t_dtc, t_cpu);
            }
            free(T); free(ydc); free(ydg);
        }
next:
        free(x); free(yc); free(yg); free(P);
    }
    printf("DTC on the GPU bit-identical to the CPU DTC kernel on %d tensors\n", dtc_exact);
    /* cache tier: tensors copied up from a RAM-resident copy under their own budget, DTC bit-exact against the CPU */
    {
        sn_cuda *c2 = sn_cuda_open(0, pk.n, why, sizeof why);
        if (c2) {
            sn_cuda_set_cache_budget(c2, 700ll << 20);
            for (int i = 0; i < pk.n; i++) {
                const sn_tensor *t = &pk.t[i];
                int cols = (int)t->cols; long rows = cols ? (long)(t->count / (uint64_t)cols) : 0;
                if (t->raw || cols < 8 || (cols & 3) || cols > 8192 || rows < 2 || rows > 20000 || t->count != (uint64_t)cols * (uint64_t)rows || sn_dtc_geom(&pk, cols, NULL, NULL) != 0) continue;
                sn_pq_res res;
                if (sn_res_load(&pk, &rd, t, &res) != 0) { printf("res load failed for %s\n", t->name); bad++; continue; }
                if (sn_cuda_cache_put(c2, &pk, t, &res) != 0) { sn_res_free(&res); continue; }
                float *x = (float *)malloc((size_t)cols * 4), *y1 = (float *)malloc((size_t)rows * 4), *y2 = (float *)malloc((size_t)rows * 4);
                double *P = (double *)malloc(((size_t)cols + 1) * 8); size_t tf = sn_dtc_floats(&pk, cols); float *T = (float *)malloc(tf * 4);
                for (int k = 0; k < cols; k++) x[k] = rnd();
                P[0] = 0; for (int k = 0; k < cols; k++) P[k + 1] = P[k] + x[k];
                sn_dtc_build(&pk, res.book, cols, x, 0, cols >> 2, T);
                sn_mv mv = SN_MV_INIT(1); mv.x[0] = x; mv.P[0] = P; mv.T[0] = T; mv.y[0] = y1;
                if (sn_pqk_rows_res(&pk, &res, t, cols, &mv, 0, (int)rows) != 0 || sn_cuda_dtc_rows(c2, &pk, t, cols, (int)rows, x, P, y2) != 0 || memcmp(y1, y2, (size_t)rows * 4) != 0) { printf("cache-tier DTC != CPU for %s\n", t->name); bad++; }
                else cached++;
                sn_cuda_cache_drop(c2, &pk, t);
                free(x); free(y1); free(y2); free(P); free(T); sn_res_free(&res);
            }
            printf("cache tier: %d tensors uploaded from RAM copies, DTC bit-identical to the CPU on all; %lld bytes left resident after the drops\n", cached, sn_cuda_cache_used(c2));
            if (sn_cuda_cache_used(c2) != 0) bad++;
            sn_cuda_close(c2);
        }
    }
    printf("%d tensors checked on the GPU (%d left to the CPU), worst relative L2 vs the CPU kernel %.3g, %.1f MB resident, %d failed\n",
           tested, skipped, worst, sn_cuda_used(c) / 1048576.0, bad);
    sn_cuda_close(c);
    return bad ? 1 : 0;
}
