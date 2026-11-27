/* sn_packcheck.c - verify a SuperNemo pack against its source shards.
 *
 *   sn_packcheck PREFIX SHAPES.tsv SOURCE_DIR [name-substring ...]
 *
 * For every pack tensor whose name contains one of the substrings (or all,
 * when none is given), reads the source bf16/f32 tensor through the
 * checkpoint's model.safetensors.index.json, expands the pack tensor, and
 * prints the relative L2 error. Also checks that a mid-tensor partial
 * expansion equals the same slice of the full expansion. Exit 1 on any
 * structural failure (the relative L2 itself is reported, not judged).
 * Pure C99, no locks. */
#include "../src/sn_pack.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int seek64(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

static char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char *b = (char *)malloc((size_t)sz + 1);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return NULL; }
    b[sz] = 0; fclose(f); if (n) *n = (size_t)sz; return b;
}

/* shard file name for a tensor, from the index json */
static int shard_of(const char *idx, const char *name, char *out, size_t cap)
{
    char key[1100]; snprintf(key, sizeof(key), "\"%s\"", name);
    const char *p = strstr(idx, key); if (!p) return -1;
    p = strchr(p + strlen(key), ':'); if (!p) return -1;
    p = strchr(p, '"'); if (!p) return -1;
    const char *q = strchr(p + 1, '"'); if (!q || (size_t)(q - p - 1) >= cap) return -1;
    memcpy(out, p + 1, (size_t)(q - p - 1)); out[q - p - 1] = 0; return 0;
}

/* read tensor name from a shard as float32; returns count or 0 */
static uint64_t read_src(const char *shard, const char *name, float **out)
{
    FILE *f = fopen(shard, "rb"); if (!f) return 0;
    uint8_t lb[8]; if (fread(lb, 1, 8, f) != 8) { fclose(f); return 0; }
    uint64_t hl = 0; for (int i = 0; i < 8; i++) hl |= (uint64_t)lb[i] << (8 * i);
    char *h = (char *)malloc(hl + 1); if (fread(h, 1, hl, f) != hl) { free(h); fclose(f); return 0; } h[hl] = 0;
    char key[1100]; snprintf(key, sizeof(key), "\"%s\"", name);
    char *k = strstr(h, key); if (!k) { free(h); fclose(f); return 0; }
    char *dt = strstr(k, "\"dtype\""); char *off = strstr(k, "\"data_offsets\"");
    if (!dt || !off) { free(h); fclose(f); return 0; }
    int bf = strncmp(strchr(dt + 7, '"') + 1, "BF16", 4) == 0;
    int f32 = strncmp(strchr(dt + 7, '"') + 1, "F32", 3) == 0;
    int f16 = strncmp(strchr(dt + 7, '"') + 1, "F16", 3) == 0;
    char *br = strchr(off, '['); uint64_t a = strtoull(br + 1, NULL, 10), b = strtoull(strchr(br, ',') + 1, NULL, 10);
    uint64_t esz = f32 ? 4 : 2, cnt = (b - a) / esz;
    uint8_t *raw = (uint8_t *)malloc(b - a);
    seek64(f, 8 + hl + a);
    if (fread(raw, 1, b - a, f) != b - a) { free(raw); free(h); fclose(f); return 0; }
    float *o = (float *)malloc(cnt * 4);
    for (uint64_t i = 0; i < cnt; i++) {
        if (f32) memcpy(&o[i], raw + i * 4, 4);
        else if (bf) { uint32_t v = (uint32_t)(raw[i * 2] | (raw[i * 2 + 1] << 8)) << 16; memcpy(&o[i], &v, 4); }
        else if (f16) o[i] = sn_f16_to_f32((uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8)));
    }
    free(raw); free(h); fclose(f); *out = o; return cnt;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: sn_packcheck PREFIX SHAPES.tsv SOURCE_DIR [substr ...]\n"); return 2; }
    sn_pack p; sn_rd r;
    if (sn_pack_open(&p, argv[1], argv[2]) != 0) return 1;
    if (sn_rd_open(&p, &r) != 0) { fprintf(stderr, "cannot open pack files\n"); return 1; }
    printf("pack: %d tensors, block %u, entries %u, pq d%u, generation %016llx\n", p.n, p.block, p.entries, p.pq_d, (unsigned long long)p.gen);
    char ipath[1100]; snprintf(ipath, sizeof(ipath), "%s/model.safetensors.index.json", argv[3]);
    char *idx = slurp(ipath, NULL); if (!idx) { fprintf(stderr, "cannot read %s\n", ipath); return 1; }
    int bad = 0, nchk = 0;
    for (int i = 0; i < p.n; i++) {
        const sn_tensor *t = &p.t[i];
        int want = argc == 4;
        for (int a = 4; a < argc; a++) if (strstr(t->name, argv[a])) want = 1;
        if (!want) continue;
        char sh[300], spath[1500];
        if (shard_of(idx, t->name, sh, sizeof(sh)) != 0) { printf("%s: not in source index\n", t->name); bad = 1; continue; }
        snprintf(spath, sizeof(spath), "%s/%s", argv[3], sh);
        float *src = NULL; uint64_t sc = read_src(spath, t->name, &src);
        if (sc != t->count) { printf("%s: source count %llu != pack count %llu\n", t->name, (unsigned long long)sc, (unsigned long long)t->count); bad = 1; free(src); continue; }
        float *full = (float *)malloc(t->count * 4);
        if (sn_expand(&p, &r, t, 0, t->count, full) != 0) { printf("%s: expand failed\n", t->name); bad = 1; free(src); free(full); continue; }
        double e2 = 0, r2 = 0; for (uint64_t k = 0; k < t->count; k++) { double d = (double)src[k] - full[k]; e2 += d * d; r2 += (double)src[k] * src[k]; }
        /* partial expansion at an unaligned offset */
        int part_ok = 1;
        if (t->count > 1000) {
            uint64_t f0 = 333, n = t->count / 3 + 7; if (f0 + n > t->count) n = t->count - f0;
            float *part = (float *)malloc(n * 4);
            if (sn_expand(&p, &r, t, f0, n, part) != 0) part_ok = 0;
            else for (uint64_t k = 0; k < n; k++) if (part[k] != full[f0 + k]) { part_ok = 0; break; }
            free(part);
        }
        printf("%-72s %s %10llu weights  relL2 %.4f  partial-slice %s\n", t->name, t->raw ? "raw   " : "packed", (unsigned long long)t->count, r2 > 0 ? sqrt(e2 / r2) : 0.0, part_ok ? "ok" : "MISMATCH");
        if (!part_ok) bad = 1;
        nchk++; free(src); free(full);
    }
    printf("%d tensors checked%s\n", nchk, bad ? ", FAILURES" : "");
    sn_rd_close(&r); sn_pack_close(&p); free(idx);
    return bad ? 1 : 0;
}
