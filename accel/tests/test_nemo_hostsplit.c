/* test_nemo_hostsplit.c - where a decode step's time goes, host versus GPU,
 * and how the host-side work scales with 1, 2, 4 and 8 CPU lanes
 * (the zero-mutex nemo_pool).
 * Usage: test_nemo_hostsplit <ckpt> <pack.nq2> [steps] */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
static double now_ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart * 1000.0 / (double)f.QuadPart; }
#else
#include <time.h>
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
#endif
#include "nemo_gpu.h"
#include "nemo_predict.h"
#include "pool.h"

#define VOC 131072
static int argmax(const float *z) { int a = 0; for (int i = 1; i < VOC; i++) if (z[i] > z[a]) a = i; return a; }

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_nemo_hostsplit <ckpt> <pack.nq2> [steps]\n"); return 2; }
    int steps = argc > 3 ? atoi(argv[3]) : 32;
    char why[400]; nemo_gpu *g = NULL;
    if (nemo_gpu_open(&g, argv[1], argv[2], 0, 1024, why, sizeof why)) { printf("FAIL gpu open: %s\n", why); return 1; }
    static int ids[16] = {1, 1784, 6290, 1307, 7771, 1395, 1044, 1278, 2001, 3002, 1500, 1600, 1700, 1800, 1900, 2100};
    float *lg = malloc(4u * VOC);
    if (nemo_gpu_prefill(g, ids, 16)) { printf("FAIL prefill\n"); return 1; }
    double t_step = 0, t_head = 0, t_conf = 0, t_arg = 0;
    nemo_confidence k;
    int tok = 2200;
    for (int s = 0; s < steps; s++) {
        double a = now_ms();
        if (nemo_gpu_step(g, tok, NULL)) { printf("FAIL step\n"); return 1; }   /* layers + sync */
        double b = now_ms();
        if (nemo_gpu_logits(g, lg)) { printf("FAIL logits\n"); return 1; }    /* head + 512 KiB copy */
        double c = now_ms();
        nemo_confidence_of(lg, VOC, &k);
        double d = now_ms();
        tok = argmax(lg);
        double e = now_ms();
        if (s) { t_step += b - a; t_head += c - b; t_conf += d - c; t_arg += e - d; }  /* step 0 is warm-up */
    }
    int n = steps - 1;
    double tot = (t_step + t_head + t_conf + t_arg) / n;
    printf("decode step split over %d steps (ms per token): layers+sync %.2f, lm_head+copy %.2f, confidence %.3f, argmax %.3f; total %.2f (%.1f tok/s)\n",
           n, t_step / n, t_head / n, t_conf / n, t_arg / n, tot, 1000.0 / tot);
    printf("host share of a step: %.1f %%\n", 100.0 * (t_conf + t_arg) / n / tot);
    /* host lanes: confidence of the same logits, 200 reps per lane count */
    nemo_confidence ref; nemo_confidence_of(lg, VOC, &ref);
    int lanes[4] = {1, 2, 4, 8};
    double base = 0;
    for (int li = 0; li < 4; li++) {
        nemo_pool p; memset(&p, 0, sizeof p);
        if (nemo_pool_init(&p, lanes[li])) { printf("FAIL pool init %d\n", lanes[li]); return 1; }
        nemo_confidence kk; int same = 1;
        nemo_confidence_of_pool(lg, VOC, &kk, &p);   /* wake the lanes */
        double a = now_ms();
        for (int r = 0; r < 200; r++) {
            nemo_confidence_of_pool(lg, VOC, &kk, &p);
            if (kk.top1 != ref.top1 || memcmp(&kk.entropy, &ref.entropy, 8) || memcmp(&kk.margin, &ref.margin, 8)) same = 0;
        }
        double ms = (now_ms() - a) / 200;
        if (li == 0) base = ms;
        printf("confidence on %d lane%s: %.3f ms (%.2fx), %s\n", lanes[li], lanes[li] > 1 ? "s" : "", ms, base / ms, same ? "bit-identical to 1 lane" : "DIFFERS");
        nemo_pool_shutdown(&p);
    }
    nemo_gpu_close(g);
    return 0;
}
