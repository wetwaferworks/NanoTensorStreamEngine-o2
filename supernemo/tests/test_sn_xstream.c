/* test_sn_xstream.c - the GPU expert stream against the CPU, bit for bit.
 *
 *   test_sn_xstream PACK SHAPES [CONFIG] [--kernels-only | --model-only] [--dense-mb N] [--experts N]
 *                   [--prompt N] [--steps N] [--reps N] [--threads LIST] [--model-dense-mb N]
 *
 * Kernel checks (any pack):
 *   1. on-card expansion of resident dense tensors (packed and raw) == sn_expand, byte for byte
 *   2. prefill GEMM on resident dense tensors (expand on the card + mirrored GEMM) == sn_expand + sn_gemm_rows
 *   3. experts staged from the pack into pinned RAM, sent to a slot: on-card expansion == sn_expand
 *   4. decode (DTC from the slot, 1 and 3 tokens) == the CPU DTC path of fx_job
 *   5. prefill (expand + GEMM from the slot, 7 tokens) == the CPU path of pfx_job
 * Model checks (CONFIG given): every logit of a prefill, greedy steps and a 4-token fused verify is hashed for
 *   CPU lanes only / expert stream only (no dense GPU tensors) / at each thread count / dense GPU lane / dense GPU lane +
 *   expert stream; the stream must not change a single bit against the run without it.
 * Pure C99, no locks. Direction by JD. Co-implementation by Cortana, built with Grok Bot by SpaceX. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../src/sn_cuda.h"
#include "../src/sn_pqk.h"
#include "../src/sn_gemm.h"
#include "../src/sn_moe.h"
#include "../src/sn_model.h"

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
static unsigned rs = 7;
static float rnd(void) { rs = rs * 1664525u + 1013904223u; return ((float)((rs >> 8) & 0xffff) / 32768.0f - 1.0f); }
static int checks, fails;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

/* expert tensors of one layer: "<base>.experts.<e>.up_proj.weight" */
typedef struct { char base[300]; } elayer;

static int find_expert_layers(const sn_pack *p, elayer *out, int max)
{
    int n = 0;
    for (int i = 0; i < p->n && n < max; i++) {
        const char *s = strstr(p->t[i].name, ".experts.0.up_proj.weight");
        if (!s) continue;
        snprintf(out[n].base, sizeof out[n].base, "%.*s", (int)(s - p->t[i].name), p->t[i].name);
        n++;
    }
    return n;
}

static const sn_tensor *etensor(const sn_pack *p, const elayer *l, int e, int dn)
{
    char nm[400]; snprintf(nm, sizeof nm, "%s.experts.%d.%s_proj.weight", l->base, e, dn ? "down" : "up");
    return sn_pack_find(p, nm);
}

/* the CPU decode path of fx_job for one expert and one token (DTC: dtc_wanted is required by the stream) */
static int cpu_expert_dtc(const sn_pack *p, const sn_pq_res *ru, const sn_pq_res *rd, const sn_tensor *tu, const sn_tensor *td,
                          int Lt, int I, const float *x, const double *P, float *out)
{
    float *mid = (float *)malloc((size_t)I * 4), *T = (float *)malloc(sn_dtc_floats(p, Lt > I ? (int)Lt : I) * 4 + sn_dtc_floats(p, I) * 4 + sn_dtc_floats(p, Lt) * 4);
    double *pm = (double *)malloc(((size_t)I + 1) * 8);
    int rc = -1;
    if (!mid || !T || !pm) goto done;
    sn_mv mv = SN_MV_INIT(1);
    sn_dtc_build(p, ru->book, Lt, x, 0, Lt >> 2, T);
    mv.x[0] = x; mv.P[0] = P; mv.T[0] = T; mv.y[0] = mid;
    if (sn_pqk_rows_res(p, ru, tu, Lt, &mv, 0, I) != 0) goto done;
    pm[0] = 0.0;
    for (int j = 0; j < I; j++) { mid[j] = sn_relu2(mid[j]); pm[j + 1] = pm[j] + (double)mid[j]; }
    sn_dtc_build(p, rd->book, I, mid, 0, I >> 2, T);
    sn_mv m2 = SN_MV_INIT(1); m2.x[0] = mid; m2.P[0] = pm; m2.T[0] = T; m2.y[0] = out;
    rc = sn_pqk_rows_res(p, rd, td, I, &m2, 0, Lt);
done:
    free(mid); free(T); free(pm);
    return rc;
}

static int stage_expert(sn_cuda *c, const sn_pack *p, sn_rd *rd, int j, const sn_tensor *tu, const sn_tensor *td)
{
    uint8_t *idx; uint16_t *mn, *sc; float *book;
    sn_cuda_xs_stage_ptrs(c, j, 0, &idx, &mn, &sc, &book);
    if (sn_res_read_into(p, rd, tu, idx, mn, sc, book) != 0) return -1;
    sn_cuda_xs_stage_ptrs(c, j, 1, &idx, &mn, &sc, &book);
    return sn_res_read_into(p, rd, td, idx, mn, sc, book);
}

static void kernels(const char *pfx, const char *shp, long dense_mb, int n_experts)
{
    static sn_pack pk; sn_rd rd;
    if (sn_pack_open(&pk, pfx, shp) != 0 || sn_rd_open(&pk, &rd) != 0) { fails++; printf("FAIL: open pack\n"); return; }
    elayer el[4]; int nel = find_expert_layers(&pk, el, 4);
    if (!nel) { printf("no routed experts in this pack\n"); fails++; return; }
    const sn_tensor *tu0 = etensor(&pk, &el[0], 0, 0), *td0 = etensor(&pk, &el[0], 0, 1);
    int Lt = (int)tu0->cols, I = (int)(tu0->count / tu0->cols);
    printf("experts: %d layer(s) found, up %d x %d, down %d x %d\n", nel, I, Lt, Lt, I);
    if (!sn_dtc_pays(&pk, I, Lt, 1, 1) || !sn_dtc_pays(&pk, Lt, I, 1, 1)) printf("note: DTC does not pay on these shapes; the engine would keep the stream off\n");
    char why[600];
    sn_cuda *c = sn_cuda_open((long long)dense_mb << 20, pk.n, why, sizeof why);
    if (!c) { printf("no CUDA lane: %s\n", why); fails++; return; }
    printf("GPU: %s\n", sn_cuda_name(c));
    /* dense tensors first (they take their budget), then the stream sizes itself on what is left */
    int loaded = 0;   /* dense tensors made resident */
    for (int i = 0; i < pk.n; i++) {
        const sn_tensor *t = &pk.t[i];
        if (strstr(t->name, ".experts.") || t->cols < 8 || (t->cols & 7) || t->count % t->cols || t->count / t->cols < 2 || t->count > (40ull << 20)) continue;
        if (sn_cuda_load(c, &pk, &rd, t) == 0) loaded++;
    }
    sn_xs_geom g; memset(&g, 0, sizeof g);
    int Etot = 0; while (etensor(&pk, &el[0], Etot, 0)) Etot++;
    g.Lt = Lt; g.I = I; g.K = 4; g.nmax = 16; g.up_blocks = tu0->nblocks; g.dn_blocks = td0->nblocks; g.nkeys = nel * Etot > 64 ? nel * Etot : 64; g.simd = sn_gemm_simd();
    if (sn_cuda_xs_open(c, &pk, &g, why, sizeof why) != 0) { printf("FAIL: stream open: %s\n", why); fails++; sn_cuda_close(c); return; }
    sn_xs_info xi; sn_cuda_xs_info(c, &xi);
    printf("dense tensors resident on the card: %d\n", loaded);
    printf("stream: %d slots of %.2f MB, %d staging; sized from %s; CPU GEMM kernel: %s\n", xi.slots, xi.slot_bytes / 1048576.0, xi.stage, xi.sized_by, g.simd ? "AVX2/FMA" : "scalar");
    /* 1 + 2: dense tensors resident on the card */
    int ex_ok = 0, ex_raw = 0, gm_ok = 0;
    float *book = (float *)malloc(pk.book_floats * 4);
    for (int i = 0; i < pk.n; i++) {
        const sn_tensor *t = &pk.t[i];
        if (!sn_cuda_resident(c, &pk, t)) continue;
        int K = (int)t->cols, R = (int)(t->count / t->cols);
        uint64_t n = t->count, first = 0;
        if (n > (4u << 20)) { n = 4u << 20; first = ((t->count - n) / 2) & ~3ull; }
        float *a = (float *)malloc((size_t)n * 4), *b = (float *)malloc((size_t)n * 4);
        int ok = a && b && sn_expand(&pk, &rd, t, first, n, a) == 0 && sn_cuda_expand(c, &pk, t, -1, 0, first, n, b) == 0;
        CHECK(ok && memcmp(a, b, (size_t)n * 4) == 0, "on-card expansion != sn_expand for %s", t->name);
        if (ok && !memcmp(a, b, (size_t)n * 4)) { ex_ok++; if (t->raw) ex_raw++; }
        free(a); free(b);
        if (t->count > (24ull << 20)) continue;          /* keep the CPU reference GEMM affordable */
        int N = (i & 1) ? 37 : 5, act = (i & 2) ? 2 : 0;
        float *W = (float *)malloc((size_t)t->count * 4), *X = (float *)malloc((size_t)N * K * 4);
        float *Y1 = (float *)malloc((size_t)N * R * 4), *Y2 = (float *)malloc((size_t)N * R * 4);
        for (size_t q = 0; q < (size_t)N * K; q++) X[q] = rnd();
        ok = W && X && Y1 && Y2 && sn_expand(&pk, &rd, t, 0, t->count, W) == 0;
        if (ok) sn_gemm_rows(W, R, K, X, N, NULL, 0, R, Y1, act);
        ok = ok && sn_cuda_xs_dense_gemm(c, &pk, t, X, N, K, R, Y2, act) == 0;
        CHECK(ok && memcmp(Y1, Y2, (size_t)N * R * 4) == 0, "prefill GEMM on the card != CPU for %s (N %d act %d)", t->name, N, act);
        if (ok && !memcmp(Y1, Y2, (size_t)N * R * 4)) gm_ok++;
        free(W); free(X); free(Y1); free(Y2);
    }
    printf("1. on-card expansion byte-identical to sn_expand on %d resident tensors (%d raw fp16, %d packed)\n", ex_ok, ex_raw, ex_ok - ex_raw);
    printf("2. prefill GEMM (expanded on the card) bit-identical to the CPU on %d tensors\n", gm_ok);
    /* 3, 4, 5: experts through slots */
    int ne = 0, dec_ok = 0, pre_ok = 0, slot_ok = 0;
    double t_gpu = 0, t_cpu = 0;
    for (int li = 0; li < nel; li++) for (int k = 0; k < n_experts; k++) {
        int e = (k * 7 + li * 3) % Etot;
        const sn_tensor *tu = etensor(&pk, &el[li], e, 0), *td = etensor(&pk, &el[li], e, 1);
        if (!tu || !td) continue;
        int key = li * Etot + e, j = ne % sn_cuda_xs_nstage(c);
        sn_cuda_xs_chunk_begin(c);
        if (sn_cuda_xs_stage_wait(c, j, 1) != 0 || stage_expert(c, &pk, &rd, j, tu, td) != 0) { CHECK(0, "staging failed"); continue; }
        int slot = sn_cuda_xs_claim(c, key);
        if (slot < 0 || sn_cuda_xs_upload(c, j, slot, key) != 0) { CHECK(0, "upload failed"); continue; }
        ne++;
        for (int w = 0; w < 2; w++) {
            const sn_tensor *t = w ? td : tu;
            float *a = (float *)malloc((size_t)t->count * 4), *b = (float *)malloc((size_t)t->count * 4);
            int ok = sn_expand(&pk, &rd, t, 0, t->count, a) == 0 && sn_cuda_expand(c, &pk, NULL, slot, w, 0, t->count, b) == 0;
            CHECK(ok && !memcmp(a, b, (size_t)t->count * 4), "slot expansion != sn_expand for %s", t->name);
            if (ok && !memcmp(a, b, (size_t)t->count * 4)) slot_ok++;
            free(a); free(b);
        }
        sn_pq_res ru, rdn;
        if (sn_res_load(&pk, &rd, tu, &ru) != 0 || sn_res_load(&pk, &rd, td, &rdn) != 0) { CHECK(0, "res load"); continue; }
        for (int nt = 1; nt <= 3; nt += 2) {
            float *lat = (float *)malloc((size_t)nt * Lt * 4), *eo1 = (float *)malloc((size_t)nt * Lt * 4), *eo2 = (float *)malloc((size_t)nt * Lt * 4);
            double *pl = (double *)malloc((size_t)nt * (Lt + 1) * 8);
            int tok[3] = { 0, 1, 2 }, oi[3] = { 0, 1, 2 };
            for (int q = 0; q < nt * Lt; q++) lat[q] = rnd();
            for (int v = 0; v < nt; v++) { double *P = pl + (size_t)v * (Lt + 1); P[0] = 0; for (int q = 0; q < Lt; q++) P[q + 1] = P[q] + (double)lat[(size_t)v * Lt + q]; }
            double t0 = now_ms();
            int ok = 1;
            for (int v = 0; v < nt && ok; v++) ok = cpu_expert_dtc(&pk, &ru, &rdn, tu, td, Lt, I, lat + (size_t)v * Lt, pl + (size_t)v * (Lt + 1), eo1 + (size_t)v * Lt) == 0;
            t_cpu += now_ms() - t0; t0 = now_ms();
            ok = ok && sn_cuda_xs_layer_begin(c, lat, pl, nt, tok, oi, nt) == 0;
            for (int v = 0; v < nt && ok; v++) ok = sn_cuda_xs_dtc(c, slot, v, v) == 0;
            ok = ok && sn_cuda_xs_layer_end(c, eo2, nt) == 0;
            t_gpu += now_ms() - t0;
            CHECK(ok && !memcmp(eo1, eo2, (size_t)nt * Lt * 4), "decode on the card != CPU DTC (layer %d expert %d, %d tokens)", li, e, nt);
            if (ok && !memcmp(eo1, eo2, (size_t)nt * Lt * 4)) dec_ok++;
            free(lat); free(eo1); free(eo2); free(pl);
        }
        {   /* prefill: 7 tokens gathered through the token list, outputs scattered through the row list */
            int nt = 7, tok[7] = { 3, 0, 6, 1, 5, 2, 4 }, oi[7] = { 13, 2, 5, 0, 9, 11, 7 };
            float *lat = (float *)malloc((size_t)nt * Lt * 4), *G = (float *)malloc((size_t)nt * Lt * 4), *U = (float *)malloc((size_t)nt * I * 4);
            float *Wu = (float *)malloc((size_t)tu->count * 4), *Wd = (float *)malloc((size_t)td->count * 4);
            float *eo1 = (float *)calloc((size_t)16 * Lt, 4), *eo2 = (float *)calloc((size_t)16 * Lt, 4), *Go = (float *)malloc((size_t)nt * Lt * 4);
            for (int q = 0; q < nt * Lt; q++) lat[q] = rnd();
            for (int v = 0; v < nt; v++) memcpy(G + (size_t)v * Lt, lat + (size_t)tok[v] * Lt, (size_t)Lt * 4);
            int ok = sn_expand(&pk, &rd, tu, 0, tu->count, Wu) == 0 && sn_expand(&pk, &rd, td, 0, td->count, Wd) == 0;
            if (ok) {
                sn_gemm_rows(Wu, I, Lt, G, nt, NULL, 0, I, U, 2);
                sn_gemm_rows(Wd, Lt, I, U, nt, NULL, 0, Lt, Go, 0);
                for (int v = 0; v < nt; v++) memcpy(eo1 + (size_t)oi[v] * Lt, Go + (size_t)v * Lt, (size_t)Lt * 4);
            }
            ok = ok && sn_cuda_xs_layer_begin(c, lat, NULL, nt, tok, oi, nt) == 0 && sn_cuda_xs_gemm(c, slot, 0, nt) == 0 && sn_cuda_xs_layer_end(c, eo2, 16) == 0;
            int same = 1;
            for (int v = 0; v < nt; v++) if (memcmp(eo1 + (size_t)oi[v] * Lt, eo2 + (size_t)oi[v] * Lt, (size_t)Lt * 4)) same = 0;
            CHECK(ok && same, "prefill on the card != CPU expand + GEMM (layer %d expert %d)", li, e);
            if (ok && same) pre_ok++;
            free(lat); free(G); free(U); free(Wu); free(Wd); free(eo1); free(eo2); free(Go);
        }
        sn_res_free(&ru); sn_res_free(&rdn);
    }
    printf("3. %d expert matrices staged, sent and expanded on the card: byte-identical to sn_expand on %d\n", 2 * ne, slot_ok);
    printf("4. decode through slots (1 and 3 tokens): bit-identical to the CPU DTC path on %d of %d runs\n", dec_ok, 2 * ne);
    printf("5. prefill through slots (7 tokens, gathered and scattered): bit-identical to the CPU on %d of %d experts\n", pre_ok, ne);
    printf("   (first-touch timings, not a benchmark: CPU DTC 1 lane %.1f ms, card %.1f ms over the decode checks)\n", t_cpu, t_gpu);
    {   /* LRU: claim more keys than slots; the oldest go, a key seen again is found */
        int ns = xi.slots, ok = 1;
        sn_cuda_xs_chunk_begin(c);
        int s0 = sn_cuda_xs_find(c, 0);
        (void)s0;
        for (int k = 0; k < ns + 2 && ok; k++) {
            int key = 32 + (k % 32);
            sn_cuda_xs_chunk_begin(c);
            int s = sn_cuda_xs_claim(c, key);
            ok = s >= 0 && sn_cuda_xs_stage_wait(c, 0, 1) == 0 && stage_expert(c, &pk, &rd, 0, tu0, td0) == 0 && sn_cuda_xs_upload(c, 0, s, key) == 0;
        }
        CHECK(ok, "slot reuse past capacity failed");
        sn_cuda_xs_chunk_begin(c);
        CHECK(ns < 3 || sn_cuda_xs_find(c, 32 + ((ns + 1) % 32)) >= 0, "the most recent key is not found in its slot");
    }
    free(book);
    sn_cuda_close(c);
    sn_rd_close(&rd);
    sn_pack_close(&pk);
}

/* ------------------------------------------------------------------ whole model */
static int V;
static int argmax(const float *l) { int b = 0; for (int i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; }

static uint64_t model_run(const char *pfx, const char *shp, const char *cfg, int threads, long dense_mb, int stream, long ram_mb,
                          int prompt_n, int steps, int reps, int *ok, sn_xs_info *xo, double *ms)
{
    *ok = 0;
    if (xo) memset(xo, 0, sizeof *xo);
    double t0 = now_ms();
    sn_model *m = sn_model_open(pfx, shp, cfg, 512, threads);
    if (!m) { *ms = now_ms() - t0; return 0; }
    sn_model_set_dtc(m, 1);
    char why[600] = "";
    if (dense_mb >= 0 && sn_model_cuda(m, (long long)dense_mb << 20, why, sizeof why) < 0) { printf("   no CUDA lane: %s\n", why); sn_model_close(m); return 0; }
    if (ram_mb > 0) sn_model_cache(m, (long long)ram_mb << 20, 0, why, sizeof why);
    if (stream && sn_model_gpu_experts(m, 1, why, sizeof why) != 0) { printf("   stream off: %s\n", why); sn_model_close(m); return 0; }
    V = sn_model_geom(m)->vocab;
    int mtp = sn_model_has_mtp(m) && sn_model_mtp_enable(m, 1) == 0;     /* MTP layers and speculative cycles too */
    float *lg = (float *)malloc((size_t)8 * V * 4);
    int *prompt = (int *)malloc((size_t)prompt_n * sizeof(int));
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < prompt_n; i++) prompt[i] = (i * 37 + 11) % V;
    for (int rep = 0; rep < reps; rep++) {
        sn_model_reset(m);
        if (sn_model_prefill(m, prompt, NULL, NULL, prompt_n, lg) != 0) goto out;
        h = fnv(h, lg, (size_t)V * 4);
        for (int i = 0; i < steps; i++) { if (sn_model_step(m, argmax(lg), lg) != 0) goto out; h = fnv(h, lg, (size_t)V * 4); }
        {
            int base = sn_model_pos(m), t = argmax(lg), toks[4] = { t, (t + 5) % V, (t + 9) % V, (t + 13) % V };
            if (sn_model_step_n(m, toks, 4, lg + V, 1) != 0) goto out;
            h = fnv(h, lg + V, (size_t)4 * V * 4);
            sn_model_rollback(m, base, 2, 4);
            if (sn_model_step(m, toks[0] ^ 1, lg) != 0) goto out;
            h = fnv(h, lg, (size_t)V * 4);
        }
        for (int i = 0; mtp && i < 8; i++) {
            int acc[8] = { 0 };
            int j = sn_model_spec_cycle(m, argmax(lg), 3, -1, -1, acc, lg);
            if (j < 0) goto out;
            h = fnv(h, &j, sizeof j); h = fnv(h, acc, (size_t)j * sizeof(int)); h = fnv(h, lg, (size_t)V * 4);
        }
    }
    *ok = 1;
out:
    if (xo) sn_model_gpu_experts_info(m, xo);
    free(lg); free(prompt); sn_model_close(m);
    *ms = now_ms() - t0;
    return h;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: test_sn_xstream PACK SHAPES [CONFIG] [--kernels-only|--model-only] [--dense-mb N] [--experts N] [--prompt N] [--steps N] [--reps N] [--threads LIST] [--model-dense-mb N]\n"); return 2; }
    const char *cfg = NULL; int kern = 1, model = 1, n_experts = 6, prompt = 24, steps = 24, reps = 2; long dense_mb = 300, mdense = 200;
    char tl[64] = "4,1,12";
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--kernels-only")) model = 0;
        else if (!strcmp(argv[i], "--model-only")) kern = 0;
        else if (!strcmp(argv[i], "--dense-mb") && i + 1 < argc) dense_mb = atol(argv[++i]);
        else if (!strcmp(argv[i], "--model-dense-mb") && i + 1 < argc) mdense = atol(argv[++i]);
        else if (!strcmp(argv[i], "--experts") && i + 1 < argc) n_experts = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--prompt") && i + 1 < argc) prompt = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc) steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) snprintf(tl, sizeof tl, "%s", argv[++i]);
        else if (argv[i][0] != '-') cfg = argv[i];
    }
    if (kern) kernels(argv[1], argv[2], dense_mb, n_experts);
    if (model && cfg) {
        int th[8], nth = 0; char *p = tl;
        while (*p && nth < 8) { th[nth++] = atoi(p); while (*p && *p != ',') p++; if (*p) p++; }
        int ok; double ms; sn_xs_info xi;
        printf("model: prompt %d tokens, %d greedy steps, one 4-token fused verify with rollback (logits hashed)\n", prompt, steps);
        uint64_t h_cpu = model_run(argv[1], argv[2], cfg, th[0], -1, 0, 64, prompt, steps, reps, &ok, NULL, &ms);
        CHECK(ok, "CPU run failed");
        printf("  CPU lanes, %2d threads                    : %016llx  (%.1f s)\n", th[0], (unsigned long long)h_cpu, ms / 1000);
        for (int k = 0; k < nth; k++) {
            uint64_t h = model_run(argv[1], argv[2], cfg, th[k], 0, 1, 64, prompt, steps, reps, &ok, &xi, &ms);
            CHECK(ok, "stream run failed (%d threads)", th[k]);
            printf("  expert stream only, %2d threads           : %016llx  (%.1f s; %d slots, %llu sent, %llu slot hits, %llu decode / %llu prefill experts on the card)\n",
                   th[k], (unsigned long long)h, ms / 1000, xi.slots, (unsigned long long)xi.loads, (unsigned long long)xi.hits, (unsigned long long)xi.experts_dtc, (unsigned long long)xi.experts_gemm);
            CHECK(h == h_cpu, "the expert stream changes the output against the CPU lanes (%d threads)", th[k]);
            CHECK(ok && xi.experts_dtc > 0 && xi.experts_gemm > 0, "the card did not compute the experts");
        }
        if (th[0] != 1 || nth > 0) {
            uint64_t h1 = model_run(argv[1], argv[2], cfg, nth > 1 ? th[1] : 1, -1, 0, 64, prompt, steps, reps, &ok, NULL, &ms);
            printf("  CPU lanes, %2d threads                    : %016llx  (%.1f s)\n", nth > 1 ? th[1] : 1, (unsigned long long)h1, ms / 1000);
            CHECK(h1 == h_cpu, "CPU lanes differ across thread counts");
        }
        uint64_t h_dense = model_run(argv[1], argv[2], cfg, th[0], mdense, 0, 64, prompt, steps, reps, &ok, NULL, &ms);
        printf("  dense GPU lane %ld MB, no stream (today)  : %016llx  (%.1f s)\n", mdense, (unsigned long long)h_dense, ms / 1000);
        uint64_t h_ds = model_run(argv[1], argv[2], cfg, th[0], mdense, 1, 64, prompt, steps, reps, &ok, &xi, &ms);
        printf("  dense GPU lane %ld MB + expert stream     : %016llx  (%.1f s; %d slots)\n", mdense, (unsigned long long)h_ds, ms / 1000, xi.slots);
        CHECK(ok && h_ds == h_dense, "the expert stream changes the output of the dense GPU lane");
        if (h_dense != h_cpu) printf("  note: the dense GPU lane's own float kernels differ from the CPU lanes (known, unchanged by this work)\n");
    }
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
