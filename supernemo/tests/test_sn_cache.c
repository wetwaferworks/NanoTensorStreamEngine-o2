/* test_sn_cache.c - the hot-expert cache must be invisible in the output.
 *
 *   test_sn_cache PREFIX SHAPES CONFIG
 *
 * The same token stream (prompt prefill, plain decode steps, fused verify steps with rollback) is run with the cache off,
 * with a roomy RAM cache, with a tiny RAM cache that has to evict, and (when a CUDA lane exists) with a tiny VRAM tier on
 * top; at 1 and 12 lanes. Every logit of every run is hashed; all hashes must be identical. Pure C99. */
#include "../src/sn_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { fails++; printf("FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint64_t fnv(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}

static int V;
static int argmax(const float *l) { int b = 0; for (int i = 1; i < V; i++) if (l[i] > l[b]) b = i; return b; }

static uint64_t run(const char *pfx, const char *shp, const char *cfg, int threads, long long ram, long long vram, int dtc, int gpu_lane, sn_cache_stats *st, int *gpu_used)
{
    sn_model *m = sn_model_open(pfx, shp, cfg, 512, threads);
    if (!m) { fails++; printf("FAIL: open\n"); return 0; }
    sn_model_set_dtc(m, dtc);
    char why[300] = "";
    if (gpu_lane) {
        if (sn_model_cuda(m, 200ll << 20, why, sizeof why) < 0) { *gpu_used = -1; sn_model_close(m); return 0; }
    }
    sn_model_cache(m, ram, vram, why, sizeof why);
    if (why[0]) printf("   (%s)\n", why);
    V = sn_model_geom(m)->vocab;
    float *lg = (float *)malloc((size_t)8 * V * 4);
    uint64_t h = 1469598103934665603ull;
    int prompt[24]; for (int i = 0; i < 24; i++) prompt[i] = (i * 37 + 11) % V;
    for (int rep = 0; rep < 2; rep++) {          /* a second conversation after reset: the cache keeps its experts */
        sn_model_reset(m);
        sn_model_prefill(m, prompt, NULL, NULL, 24, lg); h = fnv(h, lg, (size_t)V * 4);
        for (int i = 0; i < 40; i++) {
            int t = argmax(lg);
            sn_model_step(m, t, lg); h = fnv(h, lg, (size_t)V * 4);
        }
        for (int i = 0; i < 12; i++) {        /* fused verify with a rollback in the middle */
            int base = sn_model_pos(m), t = argmax(lg), toks[4] = { t, (t + 5) % V, (t + 9) % V, (t + 13) % V };
            sn_model_step_n(m, toks, 4, lg + V, 1); h = fnv(h, lg + V, (size_t)4 * V * 4);
            sn_model_rollback(m, base, 1 + i % 3, 4);
            sn_model_step(m, toks[0] ^ 1, lg); h = fnv(h, lg, (size_t)V * 4);
        }
    }
    sn_model_cache_stats(m, st);
    *gpu_used = (int)st->gpu_experts;
    free(lg); sn_model_close(m);
    return h;
}

int main(int argc, char **argv)
{
    if (argc != 4) { fprintf(stderr, "usage: test_sn_cache PREFIX SHAPES CONFIG\n"); return 2; }
    for (int dtc = 0; dtc <= 1; dtc++) {
        sn_cache_stats s0, s1, s2, s3, s4; int g0, g1, g2, g3 = 0, g4;
        printf("DTC %s\n", dtc ? "auto" : "off");
        uint64_t h0 = run(argv[1], argv[2], argv[3], 4, 0, 0, dtc, 0, &s0, &g0);
        uint64_t h1 = run(argv[1], argv[2], argv[3], 4, 512ll << 20, 0, dtc, 0, &s1, &g1);
        uint64_t h2 = run(argv[1], argv[2], argv[3], 4, 3ll << 20, 0, dtc, 0, &s2, &g2);
        uint64_t h4 = run(argv[1], argv[2], argv[3], 1, 3ll << 20, 0, dtc, 0, &s4, &g4);
        printf("  off          : hash %016llx\n", (unsigned long long)h0);
        printf("  RAM 512 MB   : hash %016llx  hits %llu misses %llu admits %llu resident %d experts (%.1f MB)\n", (unsigned long long)h1, (unsigned long long)s1.hits_ram, (unsigned long long)s1.misses, (unsigned long long)s1.admits, s1.ram_experts, s1.ram_bytes / 1048576.0);
        printf("  RAM 3 MB     : hash %016llx  hits %llu misses %llu admits %llu evicts %llu resident %d experts (%.1f MB)\n", (unsigned long long)h2, (unsigned long long)s2.hits_ram, (unsigned long long)s2.misses, (unsigned long long)s2.admits, (unsigned long long)s2.evicts, s2.ram_experts, s2.ram_bytes / 1048576.0);
        printf("  RAM 3 MB, 1 lane: hash %016llx\n", (unsigned long long)h4);
        CHECK(h1 == h0, "roomy RAM cache changes the output (dtc %d)", dtc);
        CHECK(h2 == h0, "evicting RAM cache changes the output (dtc %d)", dtc);
        CHECK(h4 == h0, "1 lane with the cache differs (dtc %d)", dtc);
        CHECK(s1.hits_ram > 0 && s1.admits > 0, "the roomy cache was used");
        CHECK(s2.evicts > 0, "the tiny cache evicted");
        CHECK(s2.ram_bytes <= s2.ram_budget, "the RAM budget is respected");
        sn_cache_stats sb; int gb = 0;
        uint64_t hb = run(argv[1], argv[2], argv[3], 4, 3ll << 20, 0, dtc, 1, &sb, &gb);       /* dense GPU lane on, VRAM expert tier off */
        uint64_t h3 = run(argv[1], argv[2], argv[3], 4, 3ll << 20, 2ll << 20, dtc, 1, &s3, &g3);
        if (g3 < 0) printf("  no CUDA lane here: VRAM tier not exercised\n");
        else {
            printf("  RAM 3 MB + VRAM 2 MB: hash %016llx  gpu hits %llu puts %llu drops %llu, %d experts (%.1f MB) in VRAM\n", (unsigned long long)h3, (unsigned long long)s3.hits_gpu, (unsigned long long)s3.gpu_puts, (unsigned long long)s3.gpu_drops, s3.gpu_experts, s3.gpu_bytes / 1048576.0);
            printf("  (dense GPU lane on, VRAM expert tier off: hash %016llx)\n", (unsigned long long)hb);
            if (dtc) {
                CHECK(s3.gpu_puts > 0 && s3.hits_gpu > 0, "the VRAM tier was used");
                CHECK(h3 == hb, "the VRAM expert tier changes the output");
            } else printf("  (DTC off: the VRAM tier stays off by design)\n");
            /* the dense GPU lane has its own float noise vs the CPU; compare VRAM tier on/off with the same dense lane */
        }
    }
    printf("%d checks, %d failed\n", checks, fails);
    return fails ? 1 : 0;
}
