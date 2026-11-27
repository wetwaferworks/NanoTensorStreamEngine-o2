/* test_nemo_prefill.c - the batched prefill (nemo_gpu_prefill) must leave
 * the GPU state bit-identical to stepping the same tokens one at a time.
 * For DTC and, when available, FULL: run a prompt both ways, then step
 * one more token with logits and compare the logits bit for bit; also a
 * second turn on top of the first (prefill at pos > 0) and a prompt that
 * spans more than one chunk. Prints the prefill and stepping rates.
 * Usage: test_nemo_prefill <ckpt> <pack.nq2> [tokens] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_s(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart / (double)f.QuadPart; }
#else
#include <time.h>
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (double)ts.tv_sec + ts.tv_nsec * 1e-9; }
#endif
#include "nemo_gpu.h"

#define VOC 131072
static int g_fail;

static int run(nemo_gpu *g, const int *ids, int n, int batched, float *logits, double *secs) {
    double t0 = now_s();
    if (batched) { if (nemo_gpu_prefill(g, ids, n - 1)) return -1; }
    else for (int i = 0; i + 1 < n; i++) if (nemo_gpu_step(g, ids[i], NULL)) return -1;
    if (nemo_gpu_step(g, ids[n - 1], NULL)) return -1;   /* sync point, not timed apart */
    *secs = now_s() - t0;
    return nemo_gpu_logits(g, logits);
}

static void compare(const char *what, const float *a, const float *b) {
    long diff = 0; int am = 0, bm = 0;
    for (int i = 0; i < VOC; i++) {
        if (memcmp(&a[i], &b[i], 4)) diff++;
        if (a[i] > a[am]) am = i;
        if (b[i] > b[bm]) bm = i;
    }
    printf("%-34s %s  (%ld of %d logits differ; argmax %d vs %d)\n", what, diff ? "FAIL" : "PASS bit-exact", diff, VOC, am, bm);
    if (diff) g_fail++;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_nemo_prefill <ckpt> <pack.nq2> [tokens]\n"); return 2; }
    int n = argc > 3 ? atoi(argv[3]) : 77;
    if (n < 2 || n > 1000) n = 77;
    int *ids = malloc(sizeof(int) * (n + 16));
    unsigned s = 12345u;
    for (int i = 0; i < n + 16; i++) { s = s * 1103515245u + 12345u; ids[i] = 1000 + (int)((s >> 8) % 30000u); }
    ids[0] = 1;
    char why[400];
    nemo_gpu *g = NULL;
    if (nemo_gpu_open(&g, argv[1], argv[2], 0, 2048, why, sizeof why)) { printf("FAIL gpu open: %s\n", why); return 1; }
    printf("prefill chunk %d, %d tokens\n", nemo_gpu_prefill_chunk(g), n);
    if (!nemo_gpu_prefill_chunk(g)) { printf("FAIL batched prefill not available\n"); return 1; }
    float *la = malloc(4u * VOC), *lb = malloc(4u * VOC);
    int full_ok = nemo_gpu_full_available(g, NULL, NULL, 0);
    for (int mode = 0; mode < 2; mode++) {
        if (mode == 1 && !full_ok) { printf("FULL unavailable: skipped\n"); break; }
        nemo_gpu_set_full(g, mode);
        const char *mn = mode ? "FULL" : "DTC";
        double ts = 0, tb = 0, t2;
        nemo_gpu_reset(g);
        if (run(g, ids, n, 0, la, &ts)) { printf("FAIL %s stepped run: device error\n", mn); return 1; }
        nemo_gpu_reset(g);
        if (run(g, ids, n, 1, lb, &tb)) { printf("FAIL %s batched run: device error\n", mn); return 1; }
        char w[80]; snprintf(w, sizeof w, "%s prompt of %d", mn, n);
        compare(w, la, lb);
        printf("  %s: stepped %.1f tok/s, batched %.1f tok/s (%.2fx)\n", mn, n / ts, n / tb, ts / tb);
        /* second turn on top of the first: prefill at pos > 0 */
        nemo_gpu_reset(g);
        if (run(g, ids, n, 0, la, &t2) || run(g, ids + n, 16, 0, la, &t2)) { printf("FAIL %s turn 2 stepped\n", mn); return 1; }
        nemo_gpu_reset(g);
        if (run(g, ids, n, 1, lb, &t2) || run(g, ids + n, 16, 1, lb, &t2)) { printf("FAIL %s turn 2 batched\n", mn); return 1; }
        snprintf(w, sizeof w, "%s second turn (16 at pos %d)", mn, n);
        compare(w, la, lb);
    }
    nemo_gpu_set_full(g, 0);
    nemo_gpu_close(g);
    printf("%s\n", g_fail ? "FAIL" : "PASS");
    return g_fail ? 1 : 0;
}
