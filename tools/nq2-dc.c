/* nq2-dc.c ... NanoQuant Dictionary Compute (NQDC) baseline probe.
 *
 * Implemented by Elfie (E.L.F.I.E.). Direction and executive
 * planning by JD. Built to answer the first questions of the
 * NQDC specification (Gizmo, 2026-11-24): does the dictionary
 * algebra check out against our actual decoder, and what does
 * it cost and buy on a real tensor?
 *
 * The claim under test: with weights stored as
 *     w[r][jB+i] = s[r][j] * C[idx[r][j]][i] + m[r][j]
 * the product y = W x can be computed without reconstructing W:
 *     T[j][k] = sum_i C[k][i] * x[jB+i],   A[j] = sum_i x[jB+i]
 *     y[r]    = sum_j ( s[r][j] * T[j][idx[r][j]] + m[r][j] * A[j] )
 *
 * Tests:
 *   DC1 synthetic fixture vs an fp64 oracle over the same
 *       compressed representation (agreement isolates algebra).
 *   DC2/DC4 a real projection tensor from the shipped pack:
 *       NQDC vs the decoder's own reconstruction vs the fp64
 *       oracle vs the raw bf16 weights; per-segment active
 *       index histogram (reuse); timings for dense, materialized,
 *       fused and dictionary paths.
 *
 * Usage: nq2-dc <model.nq2> <model.nq2book> <shard> [more shards]
 *        [--match substr]   (default "q_proj")
 *
 * Plain hosted C99. Build: cc -O2 -std=c99 -o nq2-dc
 * tools/nq2-dc.c -lm
 */

#define _FILE_OFFSET_BITS 64 /* 64-bit off_t on Windows too (Archie P0-2) */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static float half_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while (!(mant & 0x400u)) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            bits = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

static float bf16_to_float(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

static uint32_t rd32(const uint8_t *p)
{
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t rd64(const uint8_t *p)
{
    return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32);
}

/* Byte-wise 16-bit load: the pack's half sections are not
 * guaranteed 2-byte aligned in the stream buffer, so they must
 * never be read through a uint16_t pointer (Archie P0-4). */
static uint16_t rd16(const uint8_t *p)
{
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint64_t rng_state = 42;
static uint64_t splitmix(void)
{
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}
static double randn(void)   /* Box-Muller, deterministic */
{
    double u1 = ((splitmix() >> 11) + 1) / 9007199254740992.0;
    double u2 = (splitmix() >> 11) / 9007199254740992.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

typedef struct { double rel_l2, max_abs, cosine; uint64_t nonfinite; } Cmp;
/* Hardened per Gizmo's adversarial review, 2026-11-24:
 * - a zero-norm reference no longer scores a nonzero candidate
 *   as a perfect match (rel L2 is INFINITY, not 0);
 * - nonfinite values in either input are counted and poison
 *   every metric, so a NaN difference can never slip past the
 *   maximum-error or agreement checks unnoticed. */
static Cmp compare(const double *a, const double *b, uint64_t n)
{
    double d2 = 0, r2 = 0, dot = 0, na = 0, nb = 0, mx = 0;
    uint64_t nf = 0;
    for (uint64_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) { nf++; continue; }
        double d = a[i] - b[i];
        d2 += d * d; r2 += b[i] * b[i];
        dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i];
        if (fabs(d) > mx) mx = fabs(d);
    }
    Cmp c;
    c.nonfinite = nf;
    if (nf) { c.rel_l2 = NAN; c.max_abs = NAN; c.cosine = NAN; return c; }
    c.rel_l2 = r2 > 0 ? sqrt(d2 / r2) : (d2 > 0 ? INFINITY : 0);
    c.max_abs = mx;
    c.cosine = (na > 0 && nb > 0) ? dot / sqrt(na * nb) : 0;
    return c;
}
static int cmp_dbl(const void *a, const void *b)
{
    double d = *(const double *)a - *(const double *)b;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

/* ---------------- DC1: synthetic fixture ---------------- */

static void dc1(void)
{
    const int B = 256, K = 256, M = 384, N = 1024, J = N / B;
    float *C  = malloc((size_t)K * B * 4);
    uint8_t *idx = malloc((size_t)M * J);
    float *sc = malloc((size_t)M * J * 4);
    float *mn = malloc((size_t)M * J * 4);
    float *x  = malloc((size_t)N * 4);
    rng_state = 1234;
    for (int i = 0; i < K * B; i++) C[i] = (float)(randn() * 0.1);
    for (int i = 0; i < M * J; i++) {
        idx[i] = (uint8_t)(splitmix() & 0xff);
        sc[i]  = (float)(0.01 + 0.02 * ((splitmix() >> 8) % 1000) / 1000.0);
        mn[i]  = (float)(randn() * 0.005);
    }
    for (int i = 0; i < N; i++) x[i] = (float)randn();

    double *y_or  = calloc(M, 8), *y_rec = calloc(M, 8),
           *y_dc  = calloc(M, 8);
    float *T = malloc((size_t)J * K * 4), *A = malloc((size_t)J * 4);
    /* oracle: fp64 over the compressed representation */
    for (int r = 0; r < M; r++)
        for (int j = 0; j < J; j++) {
            double acc = 0;
            const float *proto = C + (size_t)idx[r * J + j] * B;
            for (int i = 0; i < B; i++)
                acc += ((double)sc[r * J + j] * proto[i]
                        + mn[r * J + j]) * x[j * B + i];
            y_or[r] += acc;
        }
    /* decoder path: fp32 reconstruct, fp32 dot, same order */
    for (int r = 0; r < M; r++) {
        float acc = 0;
        for (int j = 0; j < J; j++) {
            const float *proto = C + (size_t)idx[r * J + j] * B;
            for (int i = 0; i < B; i++) {
                float w = mn[r * J + j] + sc[r * J + j] * proto[i];
                acc += w * x[j * B + i];
            }
        }
        y_rec[r] = acc;
    }
    /* NQDC: fp32 tables, fp32 lookup sum */
    for (int j = 0; j < J; j++) {
        float a = 0;
        for (int i = 0; i < B; i++) a += x[j * B + i];
        A[j] = a;
        for (int k = 0; k < K; k++) {
            float acc = 0;
            for (int i = 0; i < B; i++) acc += C[k * B + i] * x[j * B + i];
            T[j * K + k] = acc;
        }
    }
    for (int r = 0; r < M; r++) {
        float acc = 0;
        for (int j = 0; j < J; j++)
            acc += sc[r * J + j] * T[j * K + idx[r * J + j]]
                 + mn[r * J + j] * A[j];
        y_dc[r] = acc;
    }
    Cmp c1 = compare(y_dc, y_or, M), c2 = compare(y_rec, y_or, M),
        c3 = compare(y_dc, y_rec, M);
    printf("DC1 synthetic (M=%d N=%d B=%d K=%d):\n", M, N, B, K);
    printf("  NQDC  vs fp64 oracle : rel L2 %.3e  max abs %.3e  "
           "cos %.9f  nonfinite %llu\n", c1.rel_l2, c1.max_abs,
           c1.cosine, (unsigned long long)c1.nonfinite);
    printf("  recon vs fp64 oracle : rel L2 %.3e  max abs %.3e  "
           "cos %.9f  nonfinite %llu\n", c2.rel_l2, c2.max_abs,
           c2.cosine, (unsigned long long)c2.nonfinite);
    printf("  NQDC  vs recon       : rel L2 %.3e  max abs %.3e"
           "  nonfinite %llu\n", c3.rel_l2, c3.max_abs,
           (unsigned long long)c3.nonfinite);
    free(C); free(idx); free(sc); free(mn); free(x);
    free(y_or); free(y_rec); free(y_dc); free(T); free(A);
}

/* ---------------- safetensors walk (as nq2-errprobe) -------- */

enum { DT_BF16, DT_F16, DT_F32 };
typedef struct {
    char *name; int dtype; uint64_t count, begin, end;
    int ndims; uint64_t dims[4];
} TensorInfo;
typedef struct { FILE *file; uint64_t data_start;
                 TensorInfo *tensors; int n_tensors; } Shard;

static int shard_open(Shard *sh, const char *path)
{
    sh->file = fopen(path, "rb");
    if (!sh->file) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    uint8_t lenbuf[8];
    if (fread(lenbuf, 1, 8, sh->file) != 8) return -1;
    uint64_t hlen = 0;
    for (int i = 0; i < 8; i++) hlen |= (uint64_t)lenbuf[i] << (8 * i);
    if (hlen == 0 || hlen > (1ull << 28)) return -1;
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr) return -1;
    if (fread(hdr, 1, hlen, sh->file) != hlen) { free(hdr); return -1; }
    hdr[hlen] = '\0';
    sh->data_start = 8 + hlen;
    int cap = 256;
    sh->tensors = (TensorInfo *)malloc(sizeof(TensorInfo) * cap);
    sh->n_tensors = 0;
    const char *p = hdr;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        const char *ks = ++p;
        while (*p && *p != '"') p++;
        if (!*p) break;
        size_t klen = (size_t)(p - ks);
        p++;
        while (*p && *p != ':' && *p != '{') p++;
        if (*p != ':') continue;
        p++;
        while (*p && *p != '{') p++;
        if (!*p) break;
        const char *obj = p;
        int depth = 0;
        do { if (*p == '{') depth++; else if (*p == '}') depth--; p++; }
        while (*p && depth > 0);
        char key[512];
        if (klen >= sizeof(key)) continue;
        memcpy(key, ks, klen); key[klen] = '\0';
        if (strcmp(key, "__metadata__") == 0) continue;
        const char *dt = strstr(obj, "\"dtype\"");
        const char *shp = strstr(obj, "\"shape\"");
        const char *off = strstr(obj, "\"data_offsets\"");
        if (!dt || !shp || !off) continue;
        char dbuf[8] = { 0 };
        const char *q1 = strchr(dt + 8, '"');
        if (!q1) continue;
        const char *q2 = strchr(q1 + 1, '"');
        if (!q2 || (size_t)(q2 - q1 - 1) >= sizeof(dbuf)) continue;
        memcpy(dbuf, q1 + 1, (size_t)(q2 - q1 - 1));
        int dtype;
        if (strcmp(dbuf, "BF16") == 0) dtype = DT_BF16;
        else if (strcmp(dbuf, "F16") == 0) dtype = DT_F16;
        else if (strcmp(dbuf, "F32") == 0) dtype = DT_F32;
        else continue;
        uint64_t count = 1; int ndims = 0; uint64_t dims[4] = {0,0,0,0};
        const char *b = strchr(shp, '[');
        const char *e = b ? strchr(b, ']') : NULL;
        if (!b || !e) continue;
        char sbuf[256];
        size_t sl = (size_t)(e - b - 1);
        if (sl >= sizeof(sbuf)) continue;
        memcpy(sbuf, b + 1, sl); sbuf[sl] = '\0';
        if (*sbuf) {
            char *save = NULL;
            for (char *tok = strtok_r(sbuf, ",", &save); tok;
                 tok = strtok_r(NULL, ",", &save)) {
                uint64_t d = strtoull(tok, NULL, 10);
                count *= d;
                if (ndims < 4) dims[ndims++] = d;
            }
        }
        const char *ob = strchr(off, '[');
        if (!ob) continue;
        uint64_t begin = strtoull(ob + 1, NULL, 10);
        const char *comma = strchr(ob, ',');
        if (!comma) continue;
        uint64_t end = strtoull(comma + 1, NULL, 10);
        if (sh->n_tensors == cap) {
            cap *= 2;
            sh->tensors = (TensorInfo *)realloc(sh->tensors,
                                                sizeof(TensorInfo) * cap);
        }
        TensorInfo *t = &sh->tensors[sh->n_tensors++];
        t->name = (char *)malloc(strlen(key) + 1);
        strcpy(t->name, key);
        t->dtype = dtype; t->count = count;
        t->begin = begin; t->end = end;
        t->ndims = ndims;
        memcpy(t->dims, dims, sizeof(dims));
    }
    free(hdr);
    return 0;
}

static void shard_free_meta(Shard *sh)
{
    for (int i = 0; i < sh->n_tensors; i++) free(sh->tensors[i].name);
    free(sh->tensors);
    sh->tensors = NULL; sh->n_tensors = 0;
}

static int tensor_read(Shard *sh, TensorInfo *t, float *dst)
{
    uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
    uint8_t *raw = (uint8_t *)malloc(t->count * esz);
    if (!raw) return -1;
    if (fseeko(sh->file, (off_t)(sh->data_start + t->begin), SEEK_SET) != 0)
        { free(raw); return -1; }
    if (fread(raw, esz, t->count, sh->file) != t->count) {
        free(raw); return -1;
    }
    for (uint64_t i = 0; i < t->count; i++) {
        if (t->dtype == DT_F32) memcpy(&dst[i], raw + i * 4, 4);
        else {
            uint16_t h = (uint16_t)(raw[i * 2] | (raw[i * 2 + 1] << 8));
            dst[i] = t->dtype == DT_BF16 ? bf16_to_float(h)
                                         : half_to_float(h);
        }
    }
    free(raw);
    return 0;
}

/* ---------------- DC2/DC4: real tensor ---------------- */

int main(int argc, char **argv)
{
    const char *match = "q_proj";
    int first_shard = 3;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--match") == 0 && i + 1 < argc) match = argv[++i];
    if (argc < 4) {
        fprintf(stderr, "usage: nq2-dc model.nq2 model.nq2book "
                "shard [more shards] [--match substr]\n");
        return 1;
    }
    /* locate shard args (everything after argv[2] that is not a flag) */
    const char *shards[16]; int n_shards = 0;
    for (int i = 3; i < argc && n_shards < 16; i++) {
        if (strcmp(argv[i], "--match") == 0) { i++; continue; }
        shards[n_shards++] = argv[i];
    }
    (void)first_shard;

    dc1();

    /* pack into memory */
    FILE *sf = fopen(argv[1], "rb");
    if (!sf) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    /* Size kept in off_t: a long ftell narrows a stream above
     * 2 GiB on Windows (Archie Round 3 P2-2, 2026-11-25). */
    fseeko(sf, 0, SEEK_END); off_t ssz = ftello(sf); fseeko(sf, 0, SEEK_SET);
    if (ssz < 0) { fprintf(stderr, "cannot size %s\n", argv[1]); return 1; }
    uint8_t *stream = (uint8_t *)malloc((size_t)ssz);
    if (!stream || fread(stream, 1, (size_t)ssz, sf) != (size_t)ssz) return 1;
    fclose(sf);
    if (ssz < 36 || memcmp(stream, "NQ2\0", 4) != 0) {
        fprintf(stderr, "not an .nq2 stream\n"); return 1;
    }
    uint32_t version = rd32(stream + 4);
    uint32_t block = rd32(stream + 8), entries = rd32(stream + 12);
    uint64_t n_blocks = rd64(stream + 24);
    if (version != 1) { fprintf(stderr, "probe reads v1 packs\n"); return 1; }
    /* Geometry validation before any offset is trusted
     * (Gizmo's adversarial review, 2026-11-24): bounds on block
     * and book geometry, an overflow guard on the section
     * arithmetic, and an exact stream-size reconciliation, the
     * same discipline as the hardened engine reader. */
    /* The v1 index encoding is one byte, so entries above 256
     * are unaddressable (Gizmo's AppZero S14: entry limits must
     * agree with the actual index encoding). */
    if (block == 0 || block > 4096 || entries == 0 || entries > 256) {
        fprintf(stderr, "nq2-dc: reject: unsupported geometry "
                "(block %u, entries %u)\n", block, entries);
        return 1;
    }
    if (n_blocks > (UINT64_MAX - 32) / 5
        || (uint64_t)ssz != 32 + 5 * n_blocks) {
        fprintf(stderr, "nq2-dc: reject: stream size %lld does not "
                "equal header + 5 * n_blocks (%llu)\n", (long long)ssz,
                (unsigned long long)n_blocks);
        return 1;
    }
    const uint8_t *idx_sec = stream + 32;
    const uint8_t *mn_sec = stream + 32 + n_blocks;
    const uint8_t *sc_sec = stream + 32 + 3 * n_blocks;
    FILE *bf = fopen(argv[2], "rb");
    if (!bf) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fseeko(bf, 0, SEEK_END); off_t bsz = ftello(bf); fseeko(bf, 0, SEEK_SET);
    if (bsz < 0) { fprintf(stderr, "cannot size %s\n", argv[2]); return 1; }
    uint8_t *bookf = (uint8_t *)malloc((size_t)bsz);
    if (!bookf || fread(bookf, 1, (size_t)bsz, bf) != (size_t)bsz) return 1;
    fclose(bf);
    if ((uint64_t)bsz != 12 + (uint64_t)entries * block * 4) {
        fprintf(stderr, "nq2-dc: reject: book size %lld does not "
                "equal 12 + entries * block * 4\n", (long long)bsz);
        free(stream); free(bookf);
        return 1;
    }
    if (memcmp(bookf, "NQ2B", 4) != 0 || rd32(bookf + 4) != block
        || rd32(bookf + 8) != entries) {
        fprintf(stderr, "nq2-dc: reject: book header does not "
                "match the stream (magic, block, entries)\n");
        free(stream); free(bookf);
        return 1;
    }
    const float *book = (const float *)(bookf + 12);
    /* Every index byte must name a book entry (review follow-up:
     * the timed paths trust indices; validate them once here). */
    for (uint64_t b = 0; b < n_blocks; b++) {
        if (idx_sec[b] >= entries) {
            fprintf(stderr, "nq2-dc: reject: index byte %u at block "
                    "%llu exceeds entries %u\n", idx_sec[b],
                    (unsigned long long)b, entries);
            free(stream); free(bookf);
            return 1;
        }
    }

    /* --checksum: whole-model expansion identity against the
     * independent runner (Gizmo's AppZero S14). Recomputes the
     * FNV-1a 64 hash of the full fp32 expansion in stream order,
     * exactly as tools/nq2-run.c defines it, and compares with
     * the recorded runner checksum for the shipped pack. */
    int want_checksum = 0;
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], "--checksum") == 0) want_checksum = 1;
    if (want_checksum) {
        uint64_t h = 1469598103934665603ull;
        float wbuf[4096];
        for (uint64_t b = 0; b < n_blocks; b++) {
            float m2 = half_to_float(rd16(mn_sec + 2 * b));
            float s2 = half_to_float(rd16(sc_sec + 2 * b));
            const float *proto = book + (size_t)idx_sec[b] * block;
            for (uint32_t i = 0; i < block; i++)
                wbuf[i] = m2 + s2 * proto[i];
            const uint8_t *bytes = (const uint8_t *)wbuf;
            for (uint64_t i = 0; i < (uint64_t)block * 4; i++) {
                h ^= bytes[i];
                h *= 1099511628211ull;
            }
        }
        printf("whole-model expansion checksum (FNV-1a 64): "
               "%016llx\n", (unsigned long long)h);
        printf("recorded nq2-run checksum for this pack:   "
               "7bd9557531df1ab7  ->  %s\n",
               h == 0x7bd9557531df1ab7ull
                   ? "MATCH (independent decoders agree exactly)"
                   : "MISMATCH");
        free(stream); free(bookf);
        return h == 0x7bd9557531df1ab7ull ? 0 : 1;
    }

    /* walk to the target tensor */
    TensorInfo target = { 0 }; Shard tsh = { 0 }; int found = 0, tsh_idx = -1;
    uint64_t cursor = 0, target_block0 = 0;
    for (int s = 0; s < n_shards; s++) {
        Shard sh;
        if (shard_open(&sh, shards[s]) != 0) {
            free(stream); free(bookf);
            return 1;
        }
        for (int ti = 0; ti < sh.n_tensors; ti++) {
            TensorInfo *t = &sh.tensors[ti];
            uint64_t tnb = t->count / block;
            if (!found && strstr(t->name, match) && t->ndims == 2
                    && t->dims[1] % block == 0 && t->count % block == 0) {
                target = *t; tsh = sh; found = 1; tsh_idx = s;
                target_block0 = cursor;
            }
            cursor += tnb;
        }
        if (s != tsh_idx) { fclose(sh.file); shard_free_meta(&sh); }
    }
    if (!found) { fprintf(stderr, "no 2-D tensor matching '%s'\n", match);
        free(stream); free(bookf);
        return 1; }
    /* The walk must reconcile with the header exactly, and the
     * target's segment must lie inside the stream (the ordering
     * cross-check Gizmo's review asked for; the weight-space
     * fingerprint below verifies the segment's identity). */
    if (cursor != n_blocks) {
        fprintf(stderr, "nq2-dc: reject: tensor walk totals %llu "
                "blocks, header says %llu\n",
                (unsigned long long)cursor,
                (unsigned long long)n_blocks);
        free(stream); free(bookf);
        return 1;
    }
    uint64_t M = target.dims[0], N = target.dims[1], J = N / block;
    uint64_t nb = target.count / block;
    if (target_block0 + nb > n_blocks) {
        fprintf(stderr, "nq2-dc: reject: target segment overruns "
                "the stream\n");
        return 1;
    }
    printf("\nDC2/DC4 real tensor: %s  [%llu x %llu], %llu blocks, "
           "J=%llu segments\n", target.name,
           (unsigned long long)M, (unsigned long long)N,
           (unsigned long long)nb, (unsigned long long)J);

    float *raw = (float *)malloc(target.count * 4);
    if (tensor_read(&tsh, &target, raw) != 0) return 1;
    /* Weight-space fingerprint: reconstruct this segment and
     * compare against the raw tensor in weight space. If the
     * walk were misaligned by even one block, the figure would
     * sit near sqrt(2) (uncorrelated); at the right offset it
     * reproduces the packer's recorded per-tensor error (see
     * results/errprobe-baseline-2026-11-23.txt). */
    {
        double d2 = 0, r2 = 0;
        const uint8_t *tidx = idx_sec + target_block0;
        for (uint64_t b = 0; b < nb; b++) {
            float m2 = half_to_float(rd16(mn_sec + 2 * (target_block0 + b)));
            float s2 = half_to_float(rd16(sc_sec + 2 * (target_block0 + b)));
            const float *proto = book + (size_t)tidx[b] * block;
            for (uint32_t i = 0; i < block; i++) {
                double w = (double)m2 + (double)s2 * proto[i];
                double d = w - raw[b * block + i];
                d2 += d * d; r2 += (double)raw[b * block + i]
                                 * raw[b * block + i];
            }
        }
        printf("  weight-space fingerprint: reconstruction rel L2 "
               "%.4f vs the raw tensor\n", r2 > 0 ? sqrt(d2 / r2) : 0);
    }
    const uint8_t *idx = idx_sec + target_block0;
    float *mn = (float *)malloc(nb * 4), *sc = (float *)malloc(nb * 4);
    for (uint64_t b = 0; b < nb; b++) {
        mn[b] = half_to_float(rd16(mn_sec + 2 * (target_block0 + b)));
        sc[b] = half_to_float(rd16(sc_sec + 2 * (target_block0 + b)));
    }
    float *x = (float *)malloc(N * 4);
    rng_state = 42;
    for (uint64_t i = 0; i < N; i++) x[i] = (float)randn();

    double *y_or = calloc(M, 8), *y_raw = calloc(M, 8);
    float *y_rec = calloc(M, 4), *y_fus = calloc(M, 4),
          *y_dc = calloc(M, 4);
    float *T = (float *)malloc(J * entries * 4), *A = (float *)malloc(J * 4);
    float *What = (float *)malloc(target.count * 4);

    /* oracle + raw truth (fp64) */
    for (uint64_t r = 0; r < M; r++)
        for (uint64_t j = 0; j < J; j++) {
            uint64_t b = r * J + j;
            const float *proto = book + (size_t)idx[b] * block;
            double acc = 0, racc = 0;
            for (uint32_t i = 0; i < block; i++) {
                acc += ((double)sc[b] * proto[i] + mn[b]) * x[j * block + i];
                racc += (double)raw[r * N + j * block + i] * x[j * block + i];
            }
            y_or[r] += acc; y_raw[r] += racc;
        }
    /* reuse histogram: distinct indices per segment */
    {
        uint64_t tot_u = 0, full = 0, minu = entries, maxu = 0;
        uint8_t seen[256];
        for (uint64_t j = 0; j < J; j++) {
            memset(seen, 0, sizeof(seen));
            uint64_t u = 0;
            for (uint64_t r = 0; r < M; r++)
                if (!seen[idx[r * J + j]]) { seen[idx[r * J + j]] = 1; u++; }
            tot_u += u;
            if (u == entries) full++;
            if (u < minu) minu = u;
            if (u > maxu) maxu = u;
        }
        printf("  active indices per segment: mean %.1f of %u "
               "(min %llu, max %llu); segments using the full book: "
               "%llu/%llu; mean reuse %.1f rows per entry\n",
               (double)tot_u / J, entries,
               (unsigned long long)minu, (unsigned long long)maxu,
               (unsigned long long)full, (unsigned long long)J,
               (double)M / ((double)tot_u / J));
    }
    /* timed paths */
    const int REPS = 15;
    double t_dense = 1e18, t_mat_r = 1e18, t_mat_g = 1e18,
           t_fus = 1e18, t_dc_b = 1e18, t_dc_a = 1e18;
    /* Paired per-repetition totals (Gizmo's review: stage
     * minima may come from different repetitions and must not
     * be summed as if they were one run). */
    double pair_mat[64], pair_dc[64];
    for (int rep = 0; rep < REPS; rep++) {
        double t0, t1, t2;
        t0 = now_ms();                        /* dense raw GEMV */
        for (uint64_t r = 0; r < M; r++) {
            float acc = 0;
            for (uint64_t i = 0; i < N; i++) acc += raw[r * N + i] * x[i];
            y_fus[r] = acc;                   /* scratch (checked once) */
        }
        t1 = now_ms(); if (t1 - t0 < t_dense) t_dense = t1 - t0;
        t0 = now_ms();                        /* materialize W^ */
        for (uint64_t b = 0; b < nb; b++) {
            const float *proto = book + (size_t)idx[b] * block;
            float *w = What + b * block;
            for (uint32_t i = 0; i < block; i++)
                w[i] = mn[b] + sc[b] * proto[i];
        }
        t1 = now_ms();
        for (uint64_t r = 0; r < M; r++) {
            float acc = 0;
            for (uint64_t i = 0; i < N; i++) acc += What[r * N + i] * x[i];
            y_rec[r] = acc;
        }
        t2 = now_ms();
        if (t1 - t0 < t_mat_r) t_mat_r = t1 - t0;
        if (t2 - t1 < t_mat_g) t_mat_g = t2 - t1;
        pair_mat[rep] = t2 - t0;
        t0 = now_ms();                        /* fused */
        for (uint64_t r = 0; r < M; r++) {
            float acc = 0;
            for (uint64_t j = 0; j < J; j++) {
                uint64_t b = r * J + j;
                const float *proto = book + (size_t)idx[b] * block;
                for (uint32_t i = 0; i < block; i++)
                    acc += (mn[b] + sc[b] * proto[i]) * x[j * block + i];
            }
            y_fus[r] = acc;
        }
        t1 = now_ms(); if (t1 - t0 < t_fus) t_fus = t1 - t0;
        t0 = now_ms();                        /* NQDC build */
        for (uint64_t j = 0; j < J; j++) {
            float a = 0;
            for (uint32_t i = 0; i < block; i++) a += x[j * block + i];
            A[j] = a;
            for (uint32_t k = 0; k < entries; k++) {
                float acc = 0;
                const float *proto = book + (size_t)k * block;
                for (uint32_t i = 0; i < block; i++)
                    acc += proto[i] * x[j * block + i];
                T[j * entries + k] = acc;
            }
        }
        t1 = now_ms();                        /* NQDC apply */
        for (uint64_t r = 0; r < M; r++) {
            float acc = 0;
            for (uint64_t j = 0; j < J; j++) {
                uint64_t b = r * J + j;
                acc += sc[b] * T[j * entries + idx[b]] + mn[b] * A[j];
            }
            y_dc[r] = acc;
        }
        t2 = now_ms();
        if (t1 - t0 < t_dc_b) t_dc_b = t1 - t0;
        if (t2 - t1 < t_dc_a) t_dc_a = t2 - t1;
        pair_dc[rep] = t2 - t0;
    }
    /* DC3: active-mask and lazy table construction */
    {
        float *Tc = (float *)malloc(J * entries * 4);
        int16_t *slot_of = (int16_t *)malloc(J * entries * 2);
        uint32_t *nact = (uint32_t *)malloc(J * 4);
        float *y_act = (float *)calloc(M, 4), *y_lzy = (float *)calloc(M, 4);
        float *Tl = (float *)malloc(J * entries * 4);
        uint8_t *built = (uint8_t *)calloc(J * entries, 1);
        double t_mask = 1e18, t_ab = 1e18, t_aa = 1e18, t_lz = 1e18;
        double pair_act[64];
        for (int rep = 0; rep < REPS; rep++) {
            double t0 = now_ms();
            for (uint64_t j = 0; j < J; j++) {
                int16_t *mp = slot_of + j * entries;
                for (uint32_t k = 0; k < entries; k++) mp[k] = -1;
                uint32_t na = 0;
                for (uint64_t r = 0; r < M; r++) {
                    uint8_t id = idx[r * J + j];
                    if (mp[id] < 0) mp[id] = (int16_t)na++;
                }
                nact[j] = na;
            }
            double t1 = now_ms();
            for (uint64_t j = 0; j < J; j++) {
                int16_t *mp = slot_of + j * entries;
                for (uint32_t k = 0; k < entries; k++) {
                    if (mp[k] < 0) continue;
                    float acc = 0;
                    const float *proto = book + (size_t)k * block;
                    for (uint32_t i = 0; i < block; i++)
                        acc += proto[i] * x[j * block + i];
                    Tc[j * entries + mp[k]] = acc;
                }
            }
            double t2 = now_ms();
            for (uint64_t r = 0; r < M; r++) {
                float acc = 0;
                for (uint64_t j = 0; j < J; j++) {
                    uint64_t b = r * J + j;
                    acc += sc[b] * Tc[j * entries + slot_of[j * entries + idx[b]]]
                         + mn[b] * A[j];
                }
                y_act[r] = acc;
            }
            double t3 = now_ms();
            pair_act[rep] = t3 - t0;
            if (t1 - t0 < t_mask) t_mask = t1 - t0;
            if (t2 - t1 < t_ab) t_ab = t2 - t1;
            if (t3 - t2 < t_aa) t_aa = t3 - t2;
            /* lazy: build each entry on first touch during apply */
            memset(built, 0, J * entries);
            t0 = now_ms();
            for (uint64_t r = 0; r < M; r++) {
                float acc = 0;
                for (uint64_t j = 0; j < J; j++) {
                    uint64_t b = r * J + j;
                    uint32_t k = idx[b];
                    if (!built[j * entries + k]) {
                        float a2 = 0;
                        const float *proto = book + (size_t)k * block;
                        for (uint32_t i = 0; i < block; i++)
                            a2 += proto[i] * x[j * block + i];
                        Tl[j * entries + k] = a2;
                        built[j * entries + k] = 1;
                    }
                    acc += sc[b] * Tl[j * entries + k] + mn[b] * A[j];
                }
                y_lzy[r] = acc;
            }
            t1 = now_ms(); if (t1 - t0 < t_lz) t_lz = t1 - t0;
        }
        double mxa = 0, mxl = 0;
        for (uint64_t r = 0; r < M; r++) {
            double d1 = fabs((double)y_act[r] - y_dc[r]);
            double d2 = fabs((double)y_lzy[r] - y_dc[r]);
            if (d1 > mxa) mxa = d1;
            if (d2 > mxl) mxl = d2;
        }
        double s_act[64];
        memcpy(s_act, pair_act, sizeof(double) * REPS);
        qsort(s_act, REPS, sizeof(double), cmp_dbl);
        printf("  DC3 construction policies (best of %d, contended):\n",
               REPS);
        printf("    eager : build %.3f + apply %.3f (stage minima); "
               "paired totals below\n", t_dc_b, t_dc_a);
        printf("    active: mask %.3f + build %.3f + apply %.3f "
               "(stage minima); paired best %.3f, median %.3f ms "
               "(max |diff vs eager| %.3e)\n",
               t_mask, t_ab, t_aa, s_act[0], s_act[REPS / 2], mxa);
        printf("    lazy  : interleaved total %.3f ms "
               "(max |diff vs eager| %.3e)\n", t_lz, mxl);
        free(Tc); free(slot_of); free(nact);
        free(y_act); free(y_lzy); free(Tl); free(built);
    }
    double *d_rec = calloc(M, 8), *d_dc = calloc(M, 8),
           *d_fus = calloc(M, 8);
    for (uint64_t r = 0; r < M; r++) {
        d_rec[r] = y_rec[r]; d_dc[r] = y_dc[r]; d_fus[r] = y_fus[r];
    }
    Cmp a1 = compare(d_dc, y_or, M), a2 = compare(d_rec, y_or, M),
        a3 = compare(d_dc, d_rec, M), a4 = compare(d_rec, y_raw, M),
        a5 = compare(d_fus, y_or, M);
    printf("  NQDC  vs fp64 oracle : rel L2 %.3e  max abs %.3e  "
           "cos %.9f  nonfinite %llu\n", a1.rel_l2, a1.max_abs,
           a1.cosine, (unsigned long long)a1.nonfinite);
    printf("  recon vs fp64 oracle : rel L2 %.3e  max abs %.3e  "
           "cos %.9f  nonfinite %llu\n", a2.rel_l2, a2.max_abs,
           a2.cosine, (unsigned long long)a2.nonfinite);
    printf("  NQDC  vs recon       : rel L2 %.3e  max abs %.3e"
           "  nonfinite %llu\n", a3.rel_l2, a3.max_abs,
           (unsigned long long)a3.nonfinite);
    /* T6 (CODE-REVIEW-2026-11-24): the fused path's output is
     * compared, not merely timed. */
    printf("  fused vs fp64 oracle : rel L2 %.3e  max abs %.3e  "
           "cos %.9f  nonfinite %llu\n", a5.rel_l2, a5.max_abs,
           a5.cosine, (unsigned long long)a5.nonfinite);
    printf("  recon vs RAW weights : rel L2 %.4f (the representation's "
           "own distortion)\n", a4.rel_l2);
    {
        double s_m[64], s_d[64];
        memcpy(s_m, pair_mat, sizeof(double) * REPS);
        memcpy(s_d, pair_dc, sizeof(double) * REPS);
        qsort(s_m, REPS, sizeof(double), cmp_dbl);
        qsort(s_d, REPS, sizeof(double), cmp_dbl);
        printf("  timing, best of %d, CONTENDED bench (ms); stage "
               "minima may come from different repetitions, paired "
               "totals are single-run:\n", REPS);
        printf("    dense raw GEMV        : %8.3f\n", t_dense);
        printf("    materialize + GEMV    : %8.3f + %.3f (stages); "
               "paired best %.3f, median %.3f\n",
               t_mat_r, t_mat_g, s_m[0], s_m[REPS / 2]);
        printf("    fused reconstruct-dot : %8.3f\n", t_fus);
        printf("    NQDC build + apply    : %8.3f + %.3f (stages); "
               "paired best %.3f, median %.3f\n",
               t_dc_b, t_dc_a, s_d[0], s_d[REPS / 2]);
    }
    printf("  arithmetic: conventional %llu MACs; NQDC table %llu + "
           "lookups %llu\n", (unsigned long long)(M * N),
           (unsigned long long)(entries * N),
           (unsigned long long)(M * J));
    /* Leak cleanup (Gizmo's AppZero S14): release everything so
     * the sanitizer suite can run with leak detection on. */
    free(stream); free(bookf); free(raw); free(What);
    free(T); free(A); free(mn); free(sc); free(x);
    free(y_or); free(y_raw); free(y_rec); free(y_fus); free(y_dc);
    free(d_rec); free(d_dc); free(d_fus);
    fclose(tsh.file); shard_free_meta(&tsh);
    return 0;
}
