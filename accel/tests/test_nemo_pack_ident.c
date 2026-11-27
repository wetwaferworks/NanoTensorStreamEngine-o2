/* test_nemo_pack_ident.c - synthetic format-v3 pack identity checks
 * for pack_open() / nemo_gpu_pack_check(). No device, no weights
 * from disk; the test writes its own tiny pack. */
#include "nemo_gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define MKDIR(p) _mkdir(p)
#else
#include <sys/stat.h>
#include <unistd.h>
#define MKDIR(p) mkdir((p), 0700)
#endif

static int passes, fails;
#define CHECK(c, ...) do { if (c) { passes++; printf("PASS "); } else { fails++; printf("FAIL "); } printf(__VA_ARGS__); printf("\n"); } while (0)

#define FNV_SEED 14695981039346656037ull
static uint64_t fnv(const void *data, size_t len, uint64_t h) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}
static int wr32(FILE *f, uint32_t v) {
    uint8_t b[4] = { (uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24) };
    return fwrite(b, 1, 4, f) == 4 ? 0 : -1;
}
static int wr64(FILE *f, uint64_t v) { return wr32(f, (uint32_t)v) || wr32(f, (uint32_t)(v >> 32)); }

enum { BLOCK = 256, ENTRIES = 256, PQ_D = 4, IDXW = BLOCK / PQ_D, NBLOCKS = 2 };

/* kind: "packed" or "raw". end_mode: 0 = "end" after tensors, 1 = missing,
 * 2 = gate lines then end, 3 = end then gate lines. xor_sd/xor_bd flip a
 * digest bit when non-zero. */
static int write_pack(const char *dir, const char *kind, int end_mode,
                      uint64_t xor_sd, uint64_t xor_bd, char *pack_path, size_t npath) {
    char stream[512], book[512], man[512];
    snprintf(stream, sizeof stream, "%s/ident.nq2", dir);
    snprintf(book, sizeof book, "%s/ident.nq2book", dir);
    snprintf(man, sizeof man, "%s/ident.nq2manifest", dir);
    snprintf(pack_path, npath, "%s", stream);

    uint32_t flags = 1, ver = 3;
    uint64_t total = (uint64_t)NBLOCKS * BLOCK, gen = 0x1234abcd5678ull;
    uint64_t header = 4 + 4 + 4 + 4 + 8 + 8 + 4 + 8 + 8 + 4 + 4; /* 1 segment */

    uint8_t *idx = (uint8_t *)malloc((size_t)NBLOCKS * IDXW);
    uint8_t *mn = (uint8_t *)malloc((size_t)NBLOCKS * 2);
    uint8_t *sc = (uint8_t *)malloc((size_t)NBLOCKS * 2);
    float *bk = (float *)malloc((size_t)ENTRIES * BLOCK * sizeof(float));
    if (!idx || !mn || !sc || !bk) { free(idx); free(mn); free(sc); free(bk); return -1; }
    for (size_t i = 0; i < (size_t)NBLOCKS * IDXW; i++) idx[i] = (uint8_t)(i * 17u + 3u);
    for (size_t i = 0; i < (size_t)NBLOCKS * 2; i++) { mn[i] = (uint8_t)(0xA0 + i); sc[i] = (uint8_t)(0x50 + i); }
    for (size_t i = 0; i < (size_t)ENTRIES * BLOCK; i++) bk[i] = (float)((int)i % 13) * 0.125f;

    uint64_t sd = FNV_SEED;
    sd = fnv(idx, (size_t)NBLOCKS * IDXW, sd);
    sd = fnv(mn, (size_t)NBLOCKS * 2, sd);
    sd = fnv(sc, (size_t)NBLOCKS * 2, sd);
    uint64_t bd = fnv(bk, (size_t)ENTRIES * BLOCK * sizeof(float), FNV_SEED);
    sd ^= xor_sd;
    bd ^= xor_bd;

    FILE *s = fopen(stream, "wb");
    FILE *b = fopen(book, "wb");
    FILE *m = fopen(man, "w");
    int bad = !s || !b || !m;
    bad |= !s ? 0 : (fwrite("NQ2\0", 1, 4, s) != 4 || wr32(s, ver) || wr32(s, BLOCK) || wr32(s, ENTRIES)
        || wr64(s, total) || wr64(s, NBLOCKS) || wr32(s, 1) || wr64(s, NBLOCKS)
        || wr64(s, gen) || wr32(s, flags) || wr32(s, PQ_D));
    if (!bad && (uint64_t)ftell(s) != header) bad = 1;
    bad |= !s ? 0 : (fwrite(idx, 1, (size_t)NBLOCKS * IDXW, s) != (size_t)NBLOCKS * IDXW
        || fwrite(mn, 1, (size_t)NBLOCKS * 2, s) != (size_t)NBLOCKS * 2
        || fwrite(sc, 1, (size_t)NBLOCKS * 2, s) != (size_t)NBLOCKS * 2);
    bad |= !b ? 0 : (fwrite("NQ2B", 1, 4, b) != 4 || wr32(b, ver) || wr32(b, BLOCK) || wr32(b, ENTRIES)
        || wr32(b, 1) || wr64(b, gen) || wr32(b, PQ_D)
        || fwrite(bk, sizeof(float), (size_t)ENTRIES * BLOCK, b) != (size_t)ENTRIES * BLOCK);
    if (!bad && m) {
        fprintf(m, "NQ2MANIFEST 1\n");
        fprintf(m, "generation %016llx\n", (unsigned long long)gen);
        fprintf(m, "tensors 1\n");
        fprintf(m, "tensor w %llu %llu %u %016llx %016llx %s\n",
                0ull, (unsigned long long)NBLOCKS, BLOCK,
                (unsigned long long)sd, (unsigned long long)bd, kind);
        if (end_mode == 2) fprintf(m, "gate 1\ngate-gen %016llx\n", (unsigned long long)gen);
        if (end_mode != 1) fprintf(m, "end\n");
        if (end_mode == 3) fprintf(m, "gate 1\ngate-gen %016llx\n", (unsigned long long)gen);
    }
    if (s && fclose(s)) bad = 1;
    if (b && fclose(b)) bad = 1;
    if (m && fclose(m)) bad = 1;
    free(idx); free(mn); free(sc); free(bk);
    return bad ? -1 : 0;
}

static int expect_rc(const char *pack, int want, const char *label) {
    char why[400];
    why[0] = 0;
    int rc = nemo_gpu_pack_check(pack, why, sizeof why);
    int ok = (want == 0) ? rc == 0 : rc != 0;
    CHECK(ok, "%s (rc=%d%s%s)", label, rc, why[0] ? " " : "", why);
    return ok;
}

int main(void) {
    char dir[512], pack[512];
#ifdef _WIN32
    snprintf(dir, sizeof dir, "%s\\ntse-r6-pack-ident", getenv("TEMP") ? getenv("TEMP") : ".");
#else
    snprintf(dir, sizeof dir, "/tmp/ntse-r6-pack-ident");
#endif
    MKDIR(dir);

    if (write_pack(dir, "packed", 0, 0, 0, pack, sizeof pack)) {
        CHECK(0, "write valid synthetic pack");
        printf("%d PASS / %d FAIL\n", passes, fails);
        return 1;
    }
    expect_rc(pack, 0, "valid packed manifest with end");

    if (write_pack(dir, "packed", 2, 0, 0, pack, sizeof pack) == 0)
        expect_rc(pack, 0, "valid pack with gate lines before end");
    if (write_pack(dir, "packed", 3, 0, 0, pack, sizeof pack) == 0)
        expect_rc(pack, 0, "valid pack with gate lines after end");

    if (write_pack(dir, "raw", 0, 0, 0, pack, sizeof pack) == 0)
        expect_rc(pack, -1, "kind raw is refused");
    if (write_pack(dir, "packed", 1, 0, 0, pack, sizeof pack) == 0)
        expect_rc(pack, -1, "missing trailing end line is refused");
    if (write_pack(dir, "packed", 0, 0, 1, pack, sizeof pack) == 0)
        expect_rc(pack, -1, "wrong book digest is refused");
    if (write_pack(dir, "packed", 0, 1, 0, pack, sizeof pack) == 0)
        expect_rc(pack, -1, "wrong segment digest is refused");

    printf("%d PASS / %d FAIL\n", passes, fails);
    return fails ? 1 : 0;
}
