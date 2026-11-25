/* nq2-errprobe.c ... NanoQuatTwo per-tensor error breakdown.
 *
 * Implemented by Elfie (E.L.F.I.E.). Direction and executive
 * planning by JD. Built to answer the question Gizmo's
 * engineering handoff of 2026-11-23 (sections 7 and 13) puts
 * to the headline fidelity number: a single relative L2 for a
 * whole model hides everything... is the error dominated by a
 * subset of tensors, by outliers, by frame behavior, or is it
 * broadly distributed? This tool expands a pack tensor by
 * tensor against the raw shards and reports each tensor's
 * share of the total squared error.
 *
 * The walk order replicates the quantizer's packing order
 * (shards in argument order, tensors in header order), so the
 * block cursor lands on the same segments the quantizer wrote.
 *
 * Usage: nq2-errprobe model.nq2 model.nq2book shard.safetensors
 *        [more shards...]
 *
 * Plain hosted C99. Build: cc -O2 -std=c99 -o nq2-errprobe
 * tools/nq2-errprobe.c -lm
 */

#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

static int seek_to(FILE *f, uint64_t off)
{
#ifdef _WIN32
    return _fseeki64(f, (long long)off, SEEK_SET);
#else
    return fseeko(f, (off_t)off, SEEK_SET);
#endif
}

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
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static float bf16_to_float(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
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

/* ---------------- safetensors (header walk only) ------------ */

enum { DT_BF16, DT_F16, DT_F32 };

typedef struct {
    char *name;
    int dtype;
    uint64_t count;
    uint64_t begin, end;
} TensorInfo;

typedef struct {
    FILE *file;
    uint64_t data_start;
    TensorInfo *tensors;
    int n_tensors;
} Shard;

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
    /* Minimal object walk: find "name": { ... } entries, the
     * same tolerant scan the quantizer uses. */
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
        do {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            p++;
        } while (*p && depth > 0);
        char key[512];
        if (klen >= sizeof(key)) continue;
        memcpy(key, ks, klen);
        key[klen] = '\0';
        if (strcmp(key, "__metadata__") == 0) continue;
        const char *dt = strstr(obj, "\"dtype\"");
        const char *shp = strstr(obj, "\"shape\"");
        const char *off = strstr(obj, "\"data_offsets\"");
        if (!dt || !shp || !off || dt > obj + (p - obj)) continue;
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
        else {
            fprintf(stderr, "warning: unrecognized dtype '%s' on "
                    "%s, skipped\n", dbuf, key);
            continue;
        }
        uint64_t count = 1;
        const char *b = strchr(shp, '[');
        const char *e = b ? strchr(b, ']') : NULL;
        if (!b || !e) continue;
        char sbuf[256];
        size_t sl = (size_t)(e - b - 1);
        if (sl >= sizeof(sbuf)) continue;
        memcpy(sbuf, b + 1, sl);
        sbuf[sl] = '\0';
        if (!*sbuf) count = 1;
        else {
            char *save = NULL;
            for (char *tok = strtok_r(sbuf, ",", &save); tok;
                 tok = strtok_r(NULL, ",", &save))
                count *= strtoull(tok, NULL, 10);
        }
        const char *ob = strchr(off, '[');
        if (!ob) continue;
        uint64_t begin = strtoull(ob + 1, NULL, 10);
        const char *comma = strchr(ob, ',');
        if (!comma) continue;
        uint64_t end = strtoull(comma + 1, NULL, 10);
        if (sh->n_tensors == cap) {
            cap *= 2;
            sh->tensors = (TensorInfo *)realloc(
                sh->tensors, sizeof(TensorInfo) * cap);
        }
        TensorInfo *t = &sh->tensors[sh->n_tensors++];
        t->name = (char *)malloc(strlen(key) + 1);
        strcpy(t->name, key);
        t->dtype = dtype;
        t->count = count;
        t->begin = begin;
        t->end = end;
    }
    free(hdr);
    return 0;
}

static int tensor_read(Shard *sh, TensorInfo *t, uint64_t start,
                       uint64_t count, float *dst)
{
    uint64_t esz = t->dtype == DT_F32 ? 4 : 2;
    uint64_t byte_off = sh->data_start + t->begin + start * esz;
    uint8_t *raw = (uint8_t *)malloc(count * esz);
    if (!raw) return -1;
    if (seek_to(sh->file, byte_off) != 0) { free(raw); return -1; }
    if (fread(raw, esz, count, sh->file) != count) {
        free(raw);
        return -1;
    }
    for (uint64_t i = 0; i < count; i++) {
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

/* ---------------- results table ---------------- */

typedef struct {
    char *name;
    uint64_t weights;   /* packed weights (blocks * block) */
    uint64_t params;    /* true tensor element count */
    double err2, ref2;
} Row;

static int by_err2(const void *a, const void *b)
{
    double d = ((const Row *)b)->err2 - ((const Row *)a)->err2;
    return d > 0 ? 1 : d < 0 ? -1 : 0;
}

static int by_rel(const void *a, const void *b)
{
    double ra = ((const Row *)a)->ref2 > 0
        ? sqrt(((const Row *)a)->err2 / ((const Row *)a)->ref2) : 0;
    double rb = ((const Row *)b)->ref2 > 0
        ? sqrt(((const Row *)b)->err2 / ((const Row *)b)->ref2) : 0;
    return rb > ra ? 1 : rb < ra ? -1 : 0;
}

int main(int argc, char **argv)
{
    int print_all = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--all") == 0) {
            print_all = 1;
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--; i--;
        }
    }
    if (argc < 4) {
        fprintf(stderr, "usage: nq2-errprobe [--all] model.nq2 "
                "model.nq2book shard.safetensors [more shards...]\n");
        return 1;
    }
    /* stream into memory */
    FILE *sf = fopen(argv[1], "rb");
    if (!sf) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    fseek(sf, 0, SEEK_END);
    long ssz = ftell(sf);
    fseek(sf, 0, SEEK_SET);
    uint8_t *stream = (uint8_t *)malloc((size_t)ssz);
    if (!stream || fread(stream, 1, (size_t)ssz, sf) != (size_t)ssz) {
        fprintf(stderr, "cannot read stream\n");
        return 1;
    }
    fclose(sf);
    if (ssz < 36 || memcmp(stream, "NQ2\0", 4) != 0) {
        fprintf(stderr, "not an .nq2 stream\n");
        return 1;
    }
    uint32_t version = rd32(stream + 4);
    uint32_t block   = rd32(stream + 8);
    uint32_t entries = rd32(stream + 12);
    uint64_t n_blocks = rd64(stream + 24);
    uint64_t header = 32;
    const uint8_t *seg_table = NULL;
    uint32_t n_books = 1;
    if (version == 2) {
        n_books = rd32(stream + 32);
        /* Byte pointer, read via rd64: stream + 36 is only
         * 4-byte aligned (Archie P0-4). */
        seg_table = stream + 36;
        header = 36 + (uint64_t)n_books * 8;
    }
    /* T5 hardening (CODE-REVIEW-2026-11-24): validate the pack
     * geometry before the section pointers are computed. */
    if (block == 0 || block > 4096 || entries == 0 || entries > 256
        || n_blocks > (UINT64_MAX - header) / 5
        || header + 5 * n_blocks > (uint64_t)ssz) {
        fprintf(stderr, "reject: unsupported or inconsistent pack "
                "geometry (block %u, entries %u)\n", block, entries);
        return 1;
    }
    const uint8_t *idx_sec = stream + header;
    const uint8_t *mn_sec = stream + header + n_blocks;
    const uint8_t *sc_sec = stream + header + 3 * n_blocks;

    /* book into memory */
    FILE *bf = fopen(argv[2], "rb");
    if (!bf) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fseek(bf, 0, SEEK_END);
    long bsz = ftell(bf);
    fseek(bf, 0, SEEK_SET);
    uint8_t *bookf = (uint8_t *)malloc((size_t)bsz);
    if (!bookf || fread(bookf, 1, (size_t)bsz, bf) != (size_t)bsz) {
        fprintf(stderr, "cannot read book\n");
        return 1;
    }
    fclose(bf);
    if (bsz < 12 || memcmp(bookf, "NQ2B", 4) != 0) {
        fprintf(stderr, "not an .nq2book\n");
        return 1;
    }
    /* The book must actually hold n_books * entries * block
     * floats past its header (Archie P0-4). */
    {
        uint64_t book_off = version == 2 ? 20 : 12;
        uint64_t need = book_off
            + (uint64_t)n_books * entries * block * 4;
        if ((uint64_t)bsz < need) {
            fprintf(stderr, "reject: book is %ld bytes, pack "
                    "geometry needs %llu\n", bsz,
                    (unsigned long long)need);
            return 1;
        }
    }
    const float *books = (const float *)(bookf + (version == 2
                                                   ? 20 : 12));

    Row *rows = NULL;
    int n_rows = 0, cap_rows = 0;
    uint64_t cursor = 0;       /* block cursor into the stream */
    uint32_t book_idx = 0;
    double tot_err = 0.0, tot_ref = 0.0;
    uint64_t fallback_weights = 0;
    float *raw = (float *)malloc((size_t)block * 4096 * sizeof(float));
    if (!raw) { fprintf(stderr, "out of memory\n"); return 1; }

    for (int s = 3; s < argc; s++) {
        Shard sh;
        if (shard_open(&sh, argv[s]) != 0) return 1;
        for (int ti = 0; ti < sh.n_tensors; ti++) {
            TensorInfo *t = &sh.tensors[ti];
            uint64_t nb = t->count / block;
            if (n_rows == cap_rows) {
                cap_rows = cap_rows ? cap_rows * 2 : 512;
                rows = (Row *)realloc(rows, sizeof(Row) * cap_rows);
            }
            Row *r = &rows[n_rows++];
            r->name = t->name;
            r->weights = nb * block;
            r->params = t->count;
            r->err2 = 0.0;
            r->ref2 = 0.0;
            if (nb == 0) {
                fallback_weights += t->count;
                continue;
            }
            if (version == 2) {
                if (book_idx >= n_books) {
                    fprintf(stderr, "more tensor segments than "
                            "books in the stream\n");
                    return 1;
                }
                if (rd64(seg_table + 8 * (uint64_t)book_idx) != nb) {
                    fprintf(stderr, "segment table mismatch at %s\n",
                            t->name);
                    return 1;
                }
            }
            const float *book = books
                + (size_t)(version == 2 ? book_idx : 0)
                * entries * block;
            if (version == 2) book_idx++;
            for (uint64_t base = 0; base < nb; base += 4096) {
                uint64_t want = nb - base;
                if (want > 4096) want = 4096;
                if (tensor_read(&sh, t, base * block,
                                want * block, raw) != 0) {
                    fprintf(stderr, "read failed on %s\n", t->name);
                    return 1;
                }
                for (uint64_t b = 0; b < want; b++) {
                    uint64_t at = cursor + base + b;
                    /* A stream index outside the book would read
                     * a neighbouring book's prototypes, or past
                     * the allocation (Archie P0-4). */
                    if (idx_sec[at] >= entries) {
                        fprintf(stderr, "reject: stream index %u "
                                "outside book of %u entries\n",
                                idx_sec[at], entries);
                        return 1;
                    }
                    float mn = half_to_float(rd16(mn_sec + 2 * at));
                    float sc = half_to_float(rd16(sc_sec + 2 * at));
                    const float *proto = book
                        + (size_t)idx_sec[at] * block;
                    const float *rw = raw + b * block;
                    for (uint32_t i = 0; i < block; i++) {
                        double rec = (double)mn
                            + (double)sc * proto[i];
                        double d = (double)rw[i] - rec;
                        r->err2 += d * d;
                        r->ref2 += (double)rw[i] * rw[i];
                    }
                }
            }
            cursor += nb;
            tot_err += r->err2;
            tot_ref += r->ref2;
        }
        fclose(sh.file);
    }
    if (cursor != n_blocks) {
        fprintf(stderr, "walk consumed %llu blocks, stream holds "
                "%llu (scope mismatch)\n",
                (unsigned long long)cursor,
                (unsigned long long)n_blocks);
        return 1;
    }

    printf("per-tensor error breakdown: %s\n", argv[1]);
    printf("total relative L2: %.4f over %d tensors "
           "(fallback weights unpacked: %llu)\n",
           sqrt(tot_err / tot_ref), n_rows,
           (unsigned long long)fallback_weights);

    qsort(rows, n_rows, sizeof(Row), by_err2);
    printf("\ntop 15 tensors by share of total squared error:\n");
    double cum = 0.0;
    for (int i = 0; i < n_rows && i < 15; i++) {
        double share = 100.0 * rows[i].err2 / tot_err;
        cum += share;
        printf("  %-52s rel L2 %.4f  err share %5.2f%%  "
               "(cum %.1f%%)\n",
               rows[i].name,
               rows[i].ref2 > 0
                   ? sqrt(rows[i].err2 / rows[i].ref2) : 0.0,
               share, cum);
    }
    double top10 = 0.0;
    for (int i = 0; i < n_rows && i < 10; i++)
        top10 += rows[i].err2 / tot_err;

    if (print_all) {
        double fcum = 0.0;
        printf("\nfull ranked table (all tensors, by share of total "
               "squared error):\n");
        printf("%4s  %-52s %12s %8s %8s %8s  %s\n",
               "rank", "tensor", "params", "rel L2", "err %",
               "cum %", "representation");
        for (int i = 0; i < n_rows; i++) {
            double share = tot_err > 0
                ? 100.0 * rows[i].err2 / tot_err : 0.0;
            fcum += share;
            printf("%4d  %-52.52s %12llu %8.4f %8.2f %8.2f  %s\n",
                   i + 1, rows[i].name,
                   (unsigned long long)rows[i].params,
                   rows[i].ref2 > 0
                       ? sqrt(rows[i].err2 / rows[i].ref2) : 0.0,
                   share, fcum,
                   rows[i].weights
                       ? "global book, block 256"
                       : "raw fallback (unpacked)");
        }
    }

    qsort(rows, n_rows, sizeof(Row), by_rel);
    printf("\nworst 10 tensors by their own relative L2:\n");
    for (int i = 0; i < n_rows && i < 10; i++)
        printf("  %-52s rel L2 %.4f  (%llu weights)\n",
               rows[i].name,
               rows[i].ref2 > 0
                   ? sqrt(rows[i].err2 / rows[i].ref2) : 0.0,
               (unsigned long long)rows[i].weights);
    printf("\nbest 5 tensors by their own relative L2:\n");
    for (int i = n_rows - 5; i < n_rows; i++) {
        if (i < 0) continue;
        printf("  %-52s rel L2 %.4f  (%llu weights)\n",
               rows[i].name,
               rows[i].ref2 > 0
                   ? sqrt(rows[i].err2 / rows[i].ref2) : 0.0,
               (unsigned long long)rows[i].weights);
    }

    /* rows are sorted by rel L2 descending; packed tensors
     * (weights > 0) sit above the zero-weight fallback rows,
     * so the median packed tensor is at index m/2. */
    int m = 0;
    for (int i = 0; i < n_rows; i++) if (rows[i].weights) m++;
    double median_rel = 0.0;
    if (m > 0 && rows[m / 2].ref2 > 0)
        median_rel = sqrt(rows[m / 2].err2 / rows[m / 2].ref2);
    printf("\nsummary: top-10 tensors hold %.1f%% of the squared "
           "error;\nmedian per-tensor rel L2 over %d packed "
           "tensors: %.4f\n",
           100.0 * top10, m, median_rel);
    return 0;
}
