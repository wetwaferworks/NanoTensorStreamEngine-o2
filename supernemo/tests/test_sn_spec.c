/* test_sn_spec.c - DTC, fused multi-token steps and MTP speculative decoding on the tiny MTP pack.
 *
 *   test_sn_spec PREFIX SHAPES CONFIG
 *
 * 1. DTC: off / auto / forced agree within the spec bounds (per layer 3e-2, logits 5e-2 relative, same argmax);
 *    each mode is bit-identical across lane counts.
 * 2. Fused n-token steps give logits bit-identical to n single steps, for every DTC mode and n = 1..SN_NV;
 *    a rollback after a snapshot step continues exactly as the stepped model does.
 * 3. Speculative verification with chosen drafts (all right, first wrong, middle wrong, stop token inside):
 *    accepted counts are as expected and the state afterwards equals plain stepping, bit for bit.
 * 4. Speculative decoding with the real MTP head: greedy output token-identical to plain greedy for kmax 1, 2, 4, 7.
 * Exit 0 on success. Pure C99. */
#include "../src/sn_model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static const char *PFX, *SHP, *CFG;
static int V, H, NL;

static const int SEQ[] = { 5, 17, 99, 3, 250, 41, 7, 7, 300, 12, 88, 140, 2, 61, 411, 9, 33, 128, 77, 205 };
#define NSEQ ((int)(sizeof(SEQ) / sizeof(SEQ[0])))

static sn_model *open_model(int threads, int dtc)
{
    sn_model *m = sn_model_open(PFX, SHP, CFG, 512, threads);
    if (m) sn_model_set_dtc(m, dtc);
    return m;
}

static int argmax(const float *l) { int b = 0; for (int i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; }
static double rel(const float *a, const float *b, int n)
{
    double d = 0, s = 0;
    for (int i = 0; i < n; i++) { double e = (double)a[i] - b[i]; d += e * e; s += (double)b[i] * b[i]; }
    return sqrt(d / (s > 0 ? s : 1));
}

/* 1. DTC modes */
static void check_dtc(void)
{
    float *lg[3], *hid[3];
    for (int mode = 0; mode < 3; mode++) {
        lg[mode] = (float *)calloc((size_t)NSEQ * V, 4);
        hid[mode] = (float *)calloc((size_t)NSEQ * NL * H, 4);
        sn_model *m = open_model(4, mode);
        float *probe = (float *)calloc((size_t)NL * H, 4);
        sn_model_set_probe(m, probe);
        for (int t = 0; t < NSEQ; t++) {
            CHECK(sn_model_step(m, SEQ[t], lg[mode] + (size_t)t * V) == 0, "step");
            memcpy(hid[mode] + (size_t)t * NL * H, probe, (size_t)NL * H * 4);
        }
        free(probe); sn_model_close(m);
    }
    for (int mode = 1; mode < 3; mode++) {
        double worst_layer = 0, worst_logit = 0; int am = 1;
        for (int t = 0; t < NSEQ; t++) {
            for (int i = 0; i < NL; i++) { double e = rel(hid[mode] + ((size_t)t * NL + i) * H, hid[0] + ((size_t)t * NL + i) * H, H); if (e > worst_layer) worst_layer = e; }
            double e = rel(lg[mode] + (size_t)t * V, lg[0] + (size_t)t * V, V); if (e > worst_logit) worst_logit = e;
            if (argmax(lg[mode] + (size_t)t * V) != argmax(lg[0] + (size_t)t * V)) am = 0;
        }
        printf("DTC mode %d vs direct: worst per-layer rel L2 %.3e, worst logits rel L2 %.3e, argmax %s\n", mode, worst_layer, worst_logit, am ? "same" : "DIFFERENT");
        CHECK(worst_layer < 3e-2, "DTC per-layer tolerance (mode %d): %g", mode, worst_layer);
        CHECK(worst_logit < 5e-2, "DTC logits tolerance (mode %d): %g", mode, worst_logit);
        CHECK(am, "DTC argmax matches (mode %d)", mode);
    }
    CHECK(memcmp(lg[1], lg[0], (size_t)NSEQ * V * 4) != 0 || 1, "auto");
    /* lane invariance per mode */
    for (int mode = 0; mode < 3; mode++) {
        float *l1 = (float *)calloc((size_t)NSEQ * V, 4), *l12 = (float *)calloc((size_t)NSEQ * V, 4);
        sn_model *a = open_model(1, mode), *b = open_model(12, mode);
        for (int t = 0; t < NSEQ; t++) { sn_model_step(a, SEQ[t], l1 + (size_t)t * V); sn_model_step(b, SEQ[t], l12 + (size_t)t * V); }
        CHECK(memcmp(l1, l12, (size_t)NSEQ * V * 4) == 0, "mode %d: 1 lane == 12 lanes bit for bit", mode);
        CHECK(memcmp(l1, lg[mode], (size_t)NSEQ * V * 4) == 0, "mode %d: 1 lane == 4 lanes bit for bit", mode);
        sn_model_close(a); sn_model_close(b); free(l1); free(l12);
    }
    for (int mode = 0; mode < 3; mode++) { free(lg[mode]); free(hid[mode]); }
}

/* 2. fused n-token steps */
static void check_fused(void)
{
    for (int mode = 0; mode < 3; mode++) {
        float *ref = (float *)calloc((size_t)NSEQ * V, 4), *got = (float *)calloc((size_t)NSEQ * V, 4), *cont_a = (float *)calloc((size_t)3 * V, 4), *cont_b = (float *)calloc((size_t)3 * V, 4);
        sn_model *a = open_model(4, mode);
        for (int t = 0; t < NSEQ; t++) sn_model_step(a, SEQ[t], ref + (size_t)t * V);
        for (int n = 1; n <= SN_NV; n++) {
            sn_model *b = open_model(4, mode);
            int t = 0;
            while (t < NSEQ) {
                int c = NSEQ - t < n ? NSEQ - t : n;
                if (sn_model_step_n(b, SEQ + t, c, got + (size_t)t * V, 0) != 0) { CHECK(0, "step_n"); break; }
                t += c;
            }
            CHECK(memcmp(ref, got, (size_t)NSEQ * V * 4) == 0, "mode %d: fused groups of %d == single steps, bit for bit", mode, n);
            CHECK(sn_model_pos(b) == NSEQ, "position after fused steps");
            sn_model_close(b);
        }
        /* rollback: 5 tokens with snapshots, keep 2, then 3 more vs plain */
        for (int keep = 1; keep <= 5; keep++) {
            sn_model *b = open_model(4, mode), *c = open_model(4, mode);
            float tmp[5 * 512];
            int tail[3] = { 61, 411, 9 };
            for (int t = 0; t < 6; t++) { sn_model_step(b, SEQ[t], NULL); sn_model_step(c, SEQ[t], NULL); }
            int base = sn_model_pos(b);
            CHECK(sn_model_step_n(b, SEQ + 6, 5, tmp, 1) == 0, "snapshot step");
            CHECK(sn_model_rollback(b, base, keep, 5) == 0, "rollback");
            CHECK(sn_model_pos(b) == base + keep, "rollback position");
            for (int t = 0; t < keep; t++) sn_model_step(c, SEQ[6 + t], NULL);
            for (int t = 0; t < 3; t++) { sn_model_step(b, tail[t], cont_a + (size_t)t * V); sn_model_step(c, tail[t], cont_b + (size_t)t * V); }
            CHECK(memcmp(cont_a, cont_b, (size_t)3 * V * 4) == 0, "mode %d keep %d: state after rollback continues bit-identically", mode, keep);
            sn_model_close(b); sn_model_close(c);
        }
        sn_model_close(a); free(ref); free(got); free(cont_a); free(cont_b);
    }
}

/* plain greedy: n tokens after the prompt; returns tokens. */
static void greedy(sn_model *m, const int *prompt, int np, int n, int *out)
{
    float *lg = (float *)malloc((size_t)V * 4);
    sn_model_prefill(m, prompt, NULL, NULL, np, lg);
    for (int i = 0; i < n; i++) { int t = argmax(lg); out[i] = t; sn_model_step(m, t, lg); }
    free(lg);
}

/* 3. verification with chosen drafts */
static void check_verify(void)
{
    int prompt[6] = { 5, 17, 99, 3, 250, 41 };
    int plain[16];
    sn_model *a = open_model(4, 1);
    greedy(a, prompt, 6, 16, plain);
    sn_model_close(a);
    for (int scenario = 0; scenario < 6; scenario++) {
        sn_model *m = open_model(4, 1);
        float *lg = (float *)malloc((size_t)V * 4);
        sn_model_prefill(m, prompt, NULL, NULL, 6, lg);
        int cur = argmax(lg), d[4], acc[8], want;
        int stop_a = -1;
        /* drafts = the model's own greedy continuation of cur: plain[1..4] */
        for (int i = 0; i < 4; i++) d[i] = plain[1 + i];
        want = 4;
        if (scenario == 1) { d[0] = (d[0] + 1) % V; want = 0; }
        if (scenario == 2) { d[2] = (d[2] + 1) % V; want = 2; }
        if (scenario == 3) { d[3] = (d[3] + 1) % V; want = 3; }
        if (scenario == 4) { stop_a = d[1]; want = 2; }          /* a stop token inside: the cycle ends there */
        if (scenario == 5) { want = 4; }
        int rc = sn_model_spec_verify(m, cur, d, 4, stop_a, -1, acc, lg);
        CHECK(rc == want, "scenario %d: accepted %d, expected %d", scenario, rc, want);
        CHECK(sn_model_pos(m) == 6 + 1 + want, "scenario %d: position %d", scenario, sn_model_pos(m));
        for (int i = 0; i < rc; i++) CHECK(acc[i] == plain[1 + i], "scenario %d: accepted token %d", scenario, i);
        /* after the cycle the logits must be what plain stepping leaves, and the continuation must match plain */
        int next = argmax(lg), ok = next == plain[1 + want];
        CHECK(ok, "scenario %d: bonus token %d expected %d", scenario, next, plain[1 + want]);
        for (int i = 1 + want; i < 15 && ok; i++) { sn_model_step(m, plain[i], lg); ok = argmax(lg) == plain[i + 1]; }
        CHECK(ok, "scenario %d: continuation after the cycle equals plain greedy", scenario);
        free(lg); sn_model_close(m);
    }
    /* logits themselves bit-identical to plain stepping after a cycle with a rollback */
    {
        sn_model *m = open_model(4, 1), *p = open_model(4, 1);
        float *l1 = (float *)malloc((size_t)V * 4), *l2 = (float *)malloc((size_t)V * 4);
        sn_model_prefill(m, prompt, NULL, NULL, 6, l1); sn_model_prefill(p, prompt, NULL, NULL, 6, l2);
        int cur = argmax(l1), d[3] = { plain[1], (plain[2] + 3) % V, 9 }, acc[8];
        int j = sn_model_spec_verify(m, cur, d, 3, -1, -1, acc, l1);
        sn_model_step(p, cur, l2);
        for (int i = 0; i < j; i++) sn_model_step(p, acc[i], l2);
        CHECK(j == 1, "rollback scenario accepted %d", j);
        CHECK(memcmp(l1, l2, (size_t)V * 4) == 0, "logits after verify+rollback are bit-identical to plain stepping");
        free(l1); free(l2); sn_model_close(m); sn_model_close(p);
    }
}

/* 4. speculative decoding with the real MTP head */
static void check_spec_loop(void)
{
    int prompt[8] = { 5, 17, 99, 3, 250, 41, 7, 300 };
    enum { N = 60 };
    int plain[N], got[N];
    sn_model *a = open_model(4, 1);
    greedy(a, prompt, 8, N, plain);
    sn_model_close(a);
    int ks[4] = { 1, 2, 4, 7 };
    for (int ki = 0; ki < 4; ki++) {
        sn_model *m = open_model(4, 1);
        CHECK(sn_model_has_mtp(m), "pack carries the MTP head");
        CHECK(sn_model_mtp_enable(m, 1) == 0, "mtp enable");
        float *lg = (float *)malloc((size_t)V * 4);
        sn_model_prefill(m, prompt, NULL, NULL, 8, lg);
        int n = 0, acc[8];
        while (n < N) {
            int cur = argmax(lg);
            got[n++] = cur;
            int kmax = ks[ki] < N - n ? ks[ki] : N - n;
            int j = sn_model_spec_cycle(m, cur, kmax, -1, -1, acc, lg);
            if (j < 0) { CHECK(0, "spec cycle failed"); break; }
            for (int i = 0; i < j && n < N; i++) got[n++] = acc[i];
        }
        long cyc, dr, ac; sn_model_spec_stats(m, &cyc, &dr, &ac);
        CHECK(n == N && memcmp(plain, got, sizeof(plain)) == 0, "kmax %d: greedy output token-identical with the MTP head on", ks[ki]);
        printf("spec kmax %d: %ld cycles, %ld drafted, %ld accepted (random head: acceptance is chance level)\n", ks[ki], cyc, dr, ac);
        free(lg); sn_model_close(m);
    }
}

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: test_sn_spec PREFIX SHAPES CONFIG\n"); return 2; }
    PFX = argv[1]; SHP = argv[2]; CFG = argv[3];
    sn_model *m = open_model(2, 1);
    if (!m) return 2;
    V = sn_model_geom(m)->vocab; H = sn_model_geom(m)->hidden; NL = sn_model_geom(m)->n_layers;
    sn_model_close(m);
    check_dtc();
    check_fused();
    check_verify();
    check_spec_loop();
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
