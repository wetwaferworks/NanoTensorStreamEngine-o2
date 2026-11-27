/* test_sn_cuda.c - GPU matvec against the CPU fused kernel on every tensor of a pack that the lane can take.
 *   test_sn_cuda PACK SHAPES [budget_mb] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../src/sn_cuda.h"

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
    int tested = 0, bad = 0, skipped = 0; double worst = 0;
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
next:
        free(x); free(yc); free(yg); free(P);
    }
    printf("%d tensors checked on the GPU (%d left to the CPU), worst relative L2 vs the CPU kernel %.3g, %.1f MB resident, %d failed\n",
           tested, skipped, worst, sn_cuda_used(c) / 1048576.0, bad);
    sn_cuda_close(c);
    return bad ? 1 : 0;
}
