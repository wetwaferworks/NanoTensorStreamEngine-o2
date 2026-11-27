/* nq2-merge-refine.c - build an expanded NanoTensor pack, pure C.
 *
 * Appends the refinement tensors of a second v3 PQ pack ("<name>.refine",
 * packed from W_original - W_decoded by nq2-quantize with the base
 * recipe) to a base v3 PQ pack and writes a NEW pack: stream, book and
 * manifest under a new prefix. The base pack is only read.
 *
 * Every base tensor keeps its exact bytes; the manifest digests are
 * recomputed from the bytes copied and must equal the inputs' digests,
 * and the base generation id is recomputed with the converter's formula
 * and must equal the stored one before anything is published. The new
 * generation id is the same content digest over the merged tensor list,
 * so the expanded pack is reproducible.
 *
 * Usage: nq2-merge-refine BASE.nq2 REFINE.nq2 OUT_PREFIX [--only a,b,...]
 *
 * --only keeps just the refinement tensors whose names contain one of the
 * comma-separated parts (as nq2-quantize --filter matches), copied byte for
 * byte. nq2-quantize's sampling depends on a tensor's place in its scope, so
 * re-quantizing a subset can give different bytes; taking the subset from the
 * pack that was measured keeps the shipped tensors identical to the measured
 * ones.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEED 14695981039346656037ull
static uint64_t fnv(const void *p, size_t n, uint64_t h) {
    const uint8_t *b = (const uint8_t *)p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static uint64_t fnv32(uint32_t v, uint64_t h) { uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)}; return fnv(b, 4, h); }
static int rd32(FILE *f, uint32_t *v) { uint8_t b[4]; if (fread(b, 1, 4, f) != 4) return -1; *v = b[0] | (b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24); return 0; }
static int rd64(FILE *f, uint64_t *v) { uint32_t lo, hi; if (rd32(f, &lo) || rd32(f, &hi)) return -1; *v = lo | ((uint64_t)hi << 32); return 0; }
static int wr32(FILE *f, uint32_t v) { uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)}; return fwrite(b, 1, 4, f) == 4 ? 0 : -1; }
static int wr64(FILE *f, uint64_t v) { return wr32(f, (uint32_t)v) || wr32(f, (uint32_t)(v >> 32)); }

typedef struct {
    FILE *s, *b;
    uint32_t block, entries, flags, pq, nseg;
    uint64_t total, nblocks, gen, header, book_hdr;
    int nt;
    char (*name)[256];
    uint64_t *b0, *nb, *sd, *bd;
} pk;

static int pk_open(pk *k, const char *path) {
    memset(k, 0, sizeof *k);
    k->s = fopen(path, "rb");
    if (!k->s) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char mg[4]; uint32_t ver; uint64_t sg;
    if (fread(mg, 1, 4, k->s) != 4 || memcmp(mg, "NQ2\0", 4) || rd32(k->s, &ver) || ver != 3 || rd32(k->s, &k->block)
        || rd32(k->s, &k->entries) || rd64(k->s, &k->total) || rd64(k->s, &k->nblocks) || rd32(k->s, &k->nseg) || k->nseg == 0) {
        fprintf(stderr, "%s is not a v3 stream\n", path); return -1;
    }
    for (uint32_t i = 0; i < k->nseg; i++) if (rd64(k->s, &sg)) return -1;
    if (rd64(k->s, &k->gen) || rd32(k->s, &k->flags) || k->flags != 1u || rd32(k->s, &k->pq) || k->pq != 4
        || k->block != 256 || k->entries != 256) {
        fprintf(stderr, "%s: need a PQ d4, K 256, block 256 pack with no raw sections\n", path); return -1;
    }
    k->header = (uint64_t)ftello(k->s);
    char mp[1200]; snprintf(mp, sizeof mp, "%s", path);
    size_t L = strlen(mp);
    if (L < 5 || strcmp(mp + L - 4, ".nq2")) { fprintf(stderr, "pack name must end in .nq2\n"); return -1; }
    strcpy(mp + L - 4, ".nq2manifest");
    FILE *m = fopen(mp, "r");
    if (!m) { fprintf(stderr, "manifest %s missing\n", mp); return -1; }
    char line[1200]; unsigned long long g; int nt;
    if (!fgets(line, sizeof line, m) || strcmp(line, "NQ2MANIFEST 1\n") || !fgets(line, sizeof line, m)
        || sscanf(line, "generation %llx", &g) != 1 || g != k->gen || !fgets(line, sizeof line, m)
        || sscanf(line, "tensors %d", &nt) != 1 || nt != (int)k->nseg) { fclose(m); fprintf(stderr, "manifest does not describe %s\n", path); return -1; }
    k->nt = nt;
    k->name = malloc(sizeof(*k->name) * (size_t)nt);
    k->b0 = malloc(8 * (size_t)nt); k->nb = malloc(8 * (size_t)nt); k->sd = malloc(8 * (size_t)nt); k->bd = malloc(8 * (size_t)nt);
    if (!k->name || !k->b0 || !k->nb || !k->sd || !k->bd) return -1;
    uint64_t cur = 0;
    for (int i = 0; i < nt; i++) {
        unsigned long long b0, nb, sd, bd; unsigned blk; char kind[16];
        if (!fgets(line, sizeof line, m) || sscanf(line, "tensor %255s %llu %llu %u %llx %llx %15s", k->name[i], &b0, &nb, &blk, &sd, &bd, kind) != 7
            || b0 != cur || blk != 256 || strcmp(kind, "packed")) { fclose(m); fprintf(stderr, "%s: manifest line %d unusable\n", path, i); return -1; }
        k->b0[i] = b0; k->nb[i] = nb; k->sd[i] = sd; k->bd[i] = bd; cur += nb;
    }
    fclose(m);
    if (cur != k->nblocks) { fprintf(stderr, "%s: block walk mismatch\n", path); return -1; }
    strcpy(mp + L - 4, ".nq2book");
    k->b = fopen(mp, "rb");
    uint32_t bver, bblk, bent, bbooks, bpq; uint64_t bgen;
    if (!k->b || fread(mg, 1, 4, k->b) != 4 || memcmp(mg, "NQ2B", 4) || rd32(k->b, &bver) || bver != 3 || rd32(k->b, &bblk)
        || rd32(k->b, &bent) || rd32(k->b, &bbooks) || rd64(k->b, &bgen) || rd32(k->b, &bpq)
        || bblk != 256 || bent != 256 || bgen != k->gen || bpq != 4 || bbooks != (uint32_t)nt) {
        fprintf(stderr, "%s: book does not pair with the stream\n", path); return -1;
    }
    k->book_hdr = (uint64_t)ftello(k->b);
    return 0;
}

static uint8_t g_buf[1 << 20];
/* copy len bytes src@soff -> dst@doff (dst may be NULL), hashing into *h1 and *h2 */
static int xfer(FILE *src, uint64_t soff, FILE *dst, uint64_t doff, uint64_t len, uint64_t *h1, uint64_t *h2) {
    if (fseeko(src, (off_t)soff, SEEK_SET)) return -1;
    if (dst && fseeko(dst, (off_t)doff, SEEK_SET)) return -1;
    while (len) {
        size_t n = len > sizeof g_buf ? sizeof g_buf : (size_t)len;
        if (fread(g_buf, 1, n, src) != n) return -1;
        if (h1) *h1 = fnv(g_buf, n, *h1);
        if (h2) *h2 = fnv(g_buf, n, *h2);
        if (dst && fwrite(g_buf, 1, n, dst) != n) return -1;
        len -= n;
    }
    return 0;
}

/* One pass over the merged tensor list (base tensors, then refine
 * tensors): copy every tensor's book and stream bytes into place, check
 * them against the input manifest digests, and digest the generation id
 * exactly as nq2-quantize does (geometry, then per tensor its name, book,
 * indices, frame minima, frame scales, in list order). */
typedef struct { pk *k; int i; } ref_t;
static const uint64_t BSZ = 256ull * 256 * 4;
static uint64_t geom(int n) {
    uint64_t g = SEED;
    g = fnv32(256, g); g = fnv32(256, g); g = fnv32((uint32_t)n, g); g = fnv32(1, g); g = fnv32(4, g);
    return g;
}
static int pass(ref_t *list, int n, uint64_t NB, FILE *os, uint64_t oh, FILE *ob, uint64_t obh, uint64_t *gen_out) {
    uint64_t g = geom(n), ob0 = 0;
    for (int t = 0; t < n; t++) {
        pk *k = list[t].k; int i = list[t].i;
        uint64_t nb = k->nb[i], bh = SEED, sh = SEED;
        g = fnv(k->name[i], strlen(k->name[i]), g);
        if (xfer(k->b, k->book_hdr + (uint64_t)i * BSZ, ob, obh + (uint64_t)t * BSZ, BSZ, &g, &bh)
            || xfer(k->s, k->header + k->b0[i] * 64, os, oh + ob0 * 64, nb * 64, &g, &sh)
            || xfer(k->s, k->header + k->nblocks * 64 + 2 * k->b0[i], os, oh + NB * 64 + 2 * ob0, nb * 2, &g, &sh)
            || xfer(k->s, k->header + k->nblocks * 66 + 2 * k->b0[i], os, oh + NB * 66 + 2 * ob0, nb * 2, &g, &sh)) return -1;
        if (sh != k->sd[i] || bh != k->bd[i]) { fprintf(stderr, "tensor %s: bytes do not match its manifest digests\n", k->name[i]); return -1; }
        ob0 += nb;
    }
    *gen_out = g;
    return 0;
}
/* The base pack's own generation, recomputed (proves the formula). */
static int base_gen(pk *k, uint64_t *out) {
    uint64_t g = geom(k->nt);
    for (int i = 0; i < k->nt; i++) {
        g = fnv(k->name[i], strlen(k->name[i]), g);
        if (xfer(k->b, k->book_hdr + (uint64_t)i * BSZ, NULL, 0, BSZ, &g, NULL)
            || xfer(k->s, k->header + k->b0[i] * 64, NULL, 0, k->nb[i] * 64, &g, NULL)
            || xfer(k->s, k->header + k->nblocks * 64 + 2 * k->b0[i], NULL, 0, k->nb[i] * 2, &g, NULL)
            || xfer(k->s, k->header + k->nblocks * 66 + 2 * k->b0[i], NULL, 0, k->nb[i] * 2, &g, NULL)) return -1;
    }
    *out = g;
    return 0;
}

static int in_list(const char *name, const char *list) {
    if (!list) return 1;
    char buf[2048];
    snprintf(buf, sizeof buf, "%s", list);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ","))
        if (*tok && strstr(name, tok)) return 1;
    return 0;
}

int main(int argc, char **argv) {
    const char *only = NULL;
    if (argc == 6 && !strcmp(argv[4], "--only")) only = argv[5];
    else if (argc != 4) { fprintf(stderr, "usage: nq2-merge-refine BASE.nq2 REFINE.nq2 OUT_PREFIX [--only a,b,...]\n"); return 2; }
    pk A, R;
    if (pk_open(&A, argv[1]) || pk_open(&R, argv[2])) return 1;
    for (int i = 0; i < R.nt; i++) {
        if (!in_list(R.name[i], only)) continue;
        size_t L = strlen(R.name[i]);
        if (L < 8 || strcmp(R.name[i] + L - 7, ".refine")) { fprintf(stderr, "refine tensor %s is not named <name>.refine\n", R.name[i]); return 1; }
        char bn[256]; memcpy(bn, R.name[i], L - 7); bn[L - 7] = 0;
        int found = -1;
        for (int j = 0; j < A.nt; j++) {
            if (!strcmp(A.name[j], bn)) found = j;
            if (!strcmp(A.name[j], R.name[i])) { fprintf(stderr, "%s is already in the base pack\n", R.name[i]); return 1; }
        }
        if (found < 0 || A.nb[found] != R.nb[i]) { fprintf(stderr, "%s: no base tensor of the same size\n", R.name[i]); return 1; }
    }
    int nr = 0;
    uint64_t RB = 0;
    for (int i = 0; i < R.nt; i++) if (in_list(R.name[i], only)) { nr++; RB += R.nb[i]; }
    if (nr == 0) { fprintf(stderr, "no refinement tensor selected\n"); return 1; }
    int n = A.nt + nr;
    ref_t *list = malloc(sizeof(ref_t) * (size_t)n);
    if (!list) return 1;
    for (int i = 0; i < A.nt; i++) { list[i].k = &A; list[i].i = i; }
    for (int i = 0, t = A.nt; i < R.nt; i++) if (in_list(R.name[i], only)) { list[t].k = &R; list[t].i = i; t++; }
    uint64_t NB = A.nblocks + RB;
    char sp[1100], bp[1100], mp[1100], st[1120], bt[1120];
    snprintf(sp, sizeof sp, "%s.nq2", argv[3]); snprintf(bp, sizeof bp, "%s.nq2book", argv[3]); snprintf(mp, sizeof mp, "%s.nq2manifest", argv[3]);
    snprintf(st, sizeof st, "%s.tmp", sp); snprintf(bt, sizeof bt, "%s.tmp", bp);
    { FILE *e = fopen(sp, "rb"); if (e) { fclose(e); fprintf(stderr, "%s exists; refusing to overwrite\n", sp); return 1; } }
    FILE *os = fopen(st, "wb+"), *ob = fopen(bt, "wb+");
    if (!os || !ob) { fprintf(stderr, "cannot write the output\n"); return 1; }
    uint64_t oh = 4 + 4 + 4 + 4 + 8 + 8 + 4 + 8ull * (uint64_t)n + 8 + 4 + 4;
    uint64_t obh = 4 + 4 + 4 + 4 + 4 + 8 + 4;
    uint64_t gen = 0, bgen = 0, rgen = 0;
    if (base_gen(&A, &bgen) || base_gen(&R, &rgen)) { fprintf(stderr, "read failed\n"); return 1; }
    if (bgen != A.gen || rgen != R.gen) {
        fprintf(stderr, "recomputed base generation %016llx != stored %016llx; formula mismatch, nothing published\n",
                (unsigned long long)bgen, (unsigned long long)A.gen);
        return 1;
    }
    if (pass(list, n, NB, os, oh, ob, obh, &gen)) { fprintf(stderr, "merge failed; nothing published\n"); return 1; }
    /* headers */
    if (fseeko(os, 0, SEEK_SET) || fwrite("NQ2\0", 1, 4, os) != 4 || wr32(os, 3) || wr32(os, 256) || wr32(os, 256)
        || wr64(os, A.total + RB * 256) || wr64(os, NB) || wr32(os, (uint32_t)n)) return 1;
    for (int t = 0; t < n; t++) if (wr64(os, list[t].k->nb[list[t].i])) return 1;
    if (wr64(os, gen) || wr32(os, 1) || wr32(os, 4)) return 1;
    if (fseeko(ob, 0, SEEK_SET) || fwrite("NQ2B", 1, 4, ob) != 4 || wr32(ob, 3) || wr32(ob, 256) || wr32(ob, 256)
        || wr32(ob, (uint32_t)n) || wr64(ob, gen) || wr32(ob, 4)) return 1;
    if (fclose(os) || fclose(ob)) return 1;
    FILE *mf = fopen(mp, "w");
    if (!mf) return 1;
    fprintf(mf, "NQ2MANIFEST 1\ngeneration %016llx\ntensors %d\n", (unsigned long long)gen, n);
    uint64_t b0 = 0;
    for (int t = 0; t < n; t++) {
        pk *k = list[t].k; int i = list[t].i;
        fprintf(mf, "tensor %s %llu %llu 256 %016llx %016llx packed\n", k->name[i], (unsigned long long)b0,
                (unsigned long long)k->nb[i], (unsigned long long)k->sd[i], (unsigned long long)k->bd[i]);
        b0 += k->nb[i];
    }
    fprintf(mf, "expanded from %016llx with %d refinement tensors from %016llx\nend\n",
            (unsigned long long)A.gen, nr, (unsigned long long)R.gen);
    if (fclose(mf)) return 1;
    if (rename(bt, bp) || rename(st, sp)) { fprintf(stderr, "publish failed\n"); return 1; }
    printf("expanded pack %s: %d tensors (%d base + %d refine), %llu blocks, generation %016llx\n",
           sp, n, A.nt, nr, (unsigned long long)NB, (unsigned long long)gen);
    printf("refinement stream %.1f MiB (+ %.1f MiB fp32 books); base generation %016llx verified\n",
           (double)RB * 68 / 1048576.0, (double)nr * 256 * 256 * 4 / 1048576.0, (unsigned long long)A.gen);
    return 0;
}
