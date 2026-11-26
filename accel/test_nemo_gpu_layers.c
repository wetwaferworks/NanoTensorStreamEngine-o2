/* test_nemo_gpu_layers.c - per-layer verification of the Little Nemo
 * GPU runtime against the CPU reference forward (littlenemo/src) on
 * the SAME weights: the CPU runs the checkpoint the pack was unpacked
 * into (bf16, bit-identical to the GPU's on-the-fly decode), the GPU
 * runs the .nq2 pack directly.
 * Usage: test_nemo_gpu_layers <unpacked_ckpt> <pack.nq2> [tokens] [layers]
 * Reports, per layer entry (residual stream), the relative L2 and max
 * abs difference, and for the logits the argmax agreement. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "forward.h"
#include "loader.h"
#include "nemo_gpu.h"

#define HID 4480
static float *g_cpu;  /* [layers+1][HID] for the current token */
static float *g_gpu;
static int g_layers;

static void cpu_hook(void *u, int layer, char t) {
    nemo_model *m = (nemo_model *)u;
    (void)t;
    memcpy(g_cpu + (size_t)layer * HID, m->x, HID * sizeof(float));
}
static void gpu_hook(void *u, int layer, const float *x) {
    (void)u;
    memcpy(g_gpu + (size_t)layer * HID, x, HID * sizeof(float));
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: test_nemo_gpu_layers <ckpt> <pack.nq2> [tokens] [layers]\n"); return 2; }
    int ntok = argc > 3 ? atoi(argv[3]) : 4;
    g_layers = argc > 4 ? atoi(argv[4]) : 56;
    static const int toks[8] = {1, 1784, 6290, 1307, 7771, 1395, 1044, 1278};
    if (ntok < 1 || ntok > 8) ntok = 4;
    nemo_checkpoint cp;
    if (nemo_checkpoint_open(&cp, argv[1]) != 0) { printf("FAIL open ckpt\n"); return 1; }
    nemo_model m;
    if (nemo_model_init_ex(&m, &cp, g_layers, 64, 11) != 0) { printf("FAIL cpu init\n"); return 1; }
    m.on_layer = cpu_hook;
    m.on_layer_user = &m;
    char why[400];
    nemo_gpu *g = NULL;
    if (nemo_gpu_open(&g, argv[1], argv[2], g_layers, 64, why, sizeof why) != 0) { printf("FAIL gpu open: %s\n", why); return 1; }
    nemo_gpu_info inf;
    nemo_gpu_info_get(g, &inf);
    printf("device %s sm_%d%d, VRAM total %.0f MiB, free at open %.0f MiB, ours %.0f MiB (pack %.0f MiB, lm_head %.0f MiB %s, kernel %s)\n",
           inf.device, inf.cc_major, inf.cc_minor, inf.vram_total / 1048576.0, inf.vram_free_at_open / 1048576.0,
           inf.vram_used_by_us / 1048576.0, inf.pack_bytes / 1048576.0, inf.head_bytes / 1048576.0, inf.head_int8 ? "int8 per-row" : "bf16", inf.exact ? "exact" : "fast");
    nemo_gpu_set_layer_hook(g, gpu_hook, NULL);
    g_cpu = calloc((size_t)(g_layers + 1) * HID, sizeof(float));
    g_gpu = calloc((size_t)(g_layers + 1) * HID, sizeof(float));
    float *lc = malloc(sizeof(float) * NEMO_VOCAB), *lg = malloc(sizeof(float) * NEMO_VOCAB);
    float hid[HID];
    double worst[3] = {0, 0, 0}; /* by layer type of the layer just run: M, -, * */
    double worst_final = 0, worst_abs = 0;
    int fails = 0, top1 = 0;
    for (int ti = 0; ti < ntok; ti++) {
        if (nemo_model_step_hidden(&m, toks[ti], hid) != 0) { printf("FAIL cpu step\n"); return 1; }
        memcpy(g_cpu + (size_t)g_layers * HID, hid, sizeof hid);
        if (nemo_model_head(&m, hid, lc) != 0) { printf("FAIL cpu head\n"); return 1; }
        if (nemo_gpu_step(g, toks[ti], lg) != 0) { printf("FAIL gpu step\n"); return 1; }
        printf("token %d (id %d):\n", ti, toks[ti]);
        for (int l = 1; l <= g_layers; l++) {
            const float *a = g_cpu + (size_t)l * HID, *b = g_gpu + (size_t)l * HID;
            double num = 0, den = 0, mx = 0; int same = 0;
            for (int i = 0; i < HID; i++) {
                double d = (double)a[i] - b[i];
                num += d * d; den += (double)a[i] * a[i];
                if (fabs(d) > mx) mx = fabs(d);
                if (a[i] == b[i]) same++;
            }
            double rel = den > 0 ? sqrt(num / den) : sqrt(num);
            char t = nemo_layer_type(l - 1);
            int ix = t == 'M' ? 0 : t == '-' ? 1 : 2;
            if (rel > worst[ix]) worst[ix] = rel;
            if (mx > worst_abs) worst_abs = mx;
            if (l == g_layers && rel > worst_final) worst_final = rel;
            if (l <= 3 || l == 15 || l == g_layers || rel > 1e-3)
                printf("  after layer %2d (%c): rel_l2 %.3e max_abs %.3e bit-identical %d/%d\n", l - 1, t, rel, mx, same, HID);
            if (!(rel < 1e-3)) fails++;
        }
        int ac = 0, ag = 0;
        double lnum = 0, lden = 0;
        for (int i = 0; i < NEMO_VOCAB; i++) {
            if (lc[i] > lc[ac]) ac = i;
            if (lg[i] > lg[ag]) ag = i;
            double d = (double)lc[i] - lg[i]; lnum += d * d; lden += (double)lc[i] * lc[i];
        }
        printf("  logits: cpu argmax %d gpu argmax %d, rel_l2 %.3e\n", ac, ag, sqrt(lnum / lden));
        if (ac == ag) top1++;
    }
    printf("worst residual rel_l2 after: mamba %.3e, mlp %.3e, attention %.3e; final hidden %.3e; worst max_abs %.3e\n",
           worst[0], worst[1], worst[2], worst_final, worst_abs);
    printf("argmax agreement %d/%d\n", top1, ntok);
    printf("RESULT: %s (tolerance rel_l2 < 1e-3 per layer, argmax equal)\n", (fails == 0 && top1 == ntok) ? "PASS" : "FAIL");
    nemo_gpu_close(g);
    nemo_model_free(&m);
    nemo_checkpoint_close(&cp);
    return (fails == 0 && top1 == ntok) ? 0 : 1;
}
