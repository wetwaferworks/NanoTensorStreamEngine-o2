/*
 * nq2-tensorstat.c - NanoQuatTwo tensor statistics for failure diagnosis.
 *
 * Why do some tensors collapse under min-max block normalization while
 * others survive? This tool measures the shape of a tensor's blocks:
 * how much of each block's energy sits in its single most extreme
 * element (outlier dominance), how wide block ranges run relative to
 * the tensor's RMS, and the distribution's kurtosis. A block whose
 * energy is one spike and 255 whispers cannot be represented by a
 * shared prototype after min-max stretching; these numbers show it.
 *
 * Usage:
 *   nq2-tensorstat [--block N] [--row N] shard.safetensors [...]
 *                  -- tensor.name [more names...]
 *
 * Pure C99, hosted, no external dependencies beyond libm.
 * Implementation: Elfie. Direction: JD. Outside review: Gizmo
 * (independent investigation of the worst tensors was his ask).
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

/* ---------- little helpers (same conventions as the other tools) --- */

static uint64_t rd64(const uint8_t *p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

static float bf16_to_f32(uint16_t h)
{
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static float f16_to_f32(uint16_t h)
{
    int sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    float v;
    if (exp == 0) v = (float)man * 5.9604644775390625e-08f;
    else if (exp == 31) v = man ? NAN : INFINITY;
    else v = ldexpf(1.0f + (float)man / 1024.0f, exp - 15);
    return sign ? -v : v;
}

/* ---------- minimal safetensors reader (one tensor at a time) ------ */

typedef struct {
    char name[512];         /* owned copy: the parser's key buffer
                             * is a stack local (Archie P2-4) */
    int dtype;              /* 0 = F32, 1 = BF16, 2 = F16 */
    uint64_t count;
    uint64_t begin, end;    /* absolute file offsets of the data */
} TInfo;

static int find_tensor(const char *path, const char *want,
                       TInfo *out, FILE **keep_open)
{
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    uint8_t lb[8];
    if (fread(lb, 1, 8, f) != 8) { fclose(f); return -1; }
    uint64_t hlen = rd64(lb);
    if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); return -1; }
    char *hdr = (char *)malloc(hlen + 1);
    if (!hdr || fread(hdr, 1, hlen, f) != hlen) { fclose(f); return -1; }
    hdr[hlen] = 0;
    const char *p = hdr;
    int found = -1;
    while (*p) {
        while (*p && *p != '"') p++;
        if (!*p) break;
        const char *ks = ++p;
        while (*p && *p != '"') p++;
        size_t klen = (size_t)(p - ks);
        if (!*p) break;
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
        key[klen] = 0;
        if (strcmp(key, "__metadata__") == 0) continue;
        if (strcmp(key, want) != 0) continue;
        char dbuf[8] = {0};
        const char *dt = strstr(obj, "\"dtype\"");
        if (!dt) continue;
        const char *q1 = strchr(dt + 7, '"');
        if (!q1) continue;
        const char *q2 = strchr(q1 + 1, '"');
        if (!q2 || (size_t)(q2 - q1 - 1) >= sizeof(dbuf)) continue;
        memcpy(dbuf, q1 + 1, (size_t)(q2 - q1 - 1));
        int dtype = -1;
        if (strcmp(dbuf, "F32") == 0) dtype = 0;
        else if (strcmp(dbuf, "BF16") == 0) dtype = 1;
        else if (strcmp(dbuf, "F16") == 0) dtype = 2;
        if (dtype < 0) continue;
        const char *shp = strstr(obj, "\"shape\"");
        if (!shp) continue;
        const char *b = strchr(shp, '[');
        const char *e = b ? strchr(b, ']') : NULL;
        if (!b || !e) continue;
        char sbuf[256];
        size_t sl = (size_t)(e - b - 1);
        if (sl >= sizeof(sbuf)) continue;
        memcpy(sbuf, b + 1, sl);
        sbuf[sl] = 0;
        uint64_t count = 1;
        char *tok = strtok(sbuf, ", ");
        if (!tok) count = 0;
        while (tok) { count *= strtoull(tok, NULL, 10); tok = strtok(NULL, ", "); }
        const char *off = strstr(obj, "\"data_offsets\"");
        if (!off) continue;
        const char *ob = strchr(off, '[');
        const char *oe = ob ? strchr(ob, ']') : NULL;
        if (!ob || !oe) continue;
        char obuf[64];
        size_t ol = (size_t)(oe - ob - 1);
        if (ol >= sizeof(obuf)) continue;
        memcpy(obuf, ob + 1, ol);
        obuf[ol] = 0;
        uint64_t rel_b = strtoull(obuf, NULL, 10);
        const char *comma = strchr(obuf, ',');
        uint64_t rel_e = comma ? strtoull(comma + 1, NULL, 10) : 0;
        memcpy(out->name, key, klen + 1);
        out->dtype = dtype;
        out->count = count;
        out->begin = 8 + hlen + rel_b;
        out->end = 8 + hlen + rel_e;
        found = 0;
        *keep_open = f;
        break;
    }
    free(hdr);
    if (found != 0) fclose(f);
    return found;
}

static int read_range(FILE *f, const TInfo *t, uint64_t start,
                      uint64_t count, float *dst)
{
    int esz = t->dtype == 0 ? 4 : 2;
    uint64_t bytes = count * (uint64_t)esz;
    uint8_t *raw = (uint8_t *)malloc(bytes ? bytes : 1);
    if (!raw) return -1;
#ifdef _WIN32
    _fseeki64(f, (long long)(t->begin + start * (uint64_t)esz), SEEK_SET);
#else
    fseeko(f, (off_t)(t->begin + start * (uint64_t)esz), SEEK_SET);
#endif
    if (bytes && fread(raw, 1, bytes, f) != bytes) { free(raw); return -1; }
    for (uint64_t i = 0; i < count; i++) {
        if (t->dtype == 0) memcpy(&dst[i], raw + i * 4, 4);
        else if (t->dtype == 1) {
            uint16_t h;
            memcpy(&h, raw + i * 2, 2);
            dst[i] = bf16_to_f32(h);
        } else {
            uint16_t h;
            memcpy(&h, raw + i * 2, 2);
            dst[i] = f16_to_f32(h);
        }
    }
    free(raw);
    return 0;
}

/* ---------- statistics ---------------------------------------------- */

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y ? 1 : 0;
}

static void analyze(const char *path, const char *name,
                    uint64_t block, uint64_t row)
{
    TInfo t;
    FILE *f = NULL;
    if (find_tensor(path, name, &t, &f) != 0) {
        printf("%s: not found in %s\n", name, path);
        return;
    }
    uint64_t n = t.count;
    uint64_t nb = block ? n / block : 0;
    printf("== %s (%llu weights, dtype %s)\n", name,
           (unsigned long long)n,
           t.dtype == 0 ? "F32" : t.dtype == 1 ? "BF16" : "F16");
    if (nb == 0) { printf("   no full blocks of %llu\n",
           (unsigned long long)block); fclose(f); return; }

    double *ranges = (double *)malloc(sizeof(double) * nb);
    double *topshares = (double *)malloc(sizeof(double) * nb);
    float *buf = (float *)malloc(sizeof(float) * block);
    if (!ranges || !topshares || !buf) { fclose(f); return; }

    double sum = 0, sum2 = 0, sum4 = 0, maxabs = 0;
    double dom_blocks = 0;      /* blocks where top-1 share > 50% */
    uint64_t done = 0;
    for (uint64_t b = 0; b < nb; b++) {
        if (read_range(f, &t, b * block, block, buf) != 0) break;
        float mn = buf[0], mx = buf[0];
        for (uint64_t i = 0; i < block; i++) {
            if (buf[i] < mn) mn = buf[i];
            if (buf[i] > mx) mx = buf[i];
        }
        double e = 0, top = 0;
        for (uint64_t i = 0; i < block; i++) {
            double d = (double)buf[i] - mn;
            double d2 = d * d;
            e += d2;
            if (d2 > top) top = d2;
            double v = buf[i];
            sum += v; sum2 += v * v; sum4 += v * v * v * v;
            if (fabs(v) > maxabs) maxabs = fabs(v);
        }
        ranges[b] = (double)mx - mn;
        topshares[b] = e > 0 ? top / e : 0;
        if (topshares[b] > 0.5) dom_blocks += 1;
        done++;
    }
    double mean = sum / (block * done);
    double var = sum2 / (block * done) - mean * mean;
    double rms = sqrt(sum2 / (block * done));
    double kurt = (sum4 / (block * done)) / (var * var) - 3.0;
    qsort(ranges, done, sizeof(double), cmp_d);
    qsort(topshares, done, sizeof(double), cmp_d);
    printf("   mean %.6g  RMS %.6g  max|w| %.6g  excess kurtosis %.2f\n",
           mean, rms, maxabs, kurt);
    printf("   blocks of %llu: %llu\n", (unsigned long long)block,
           (unsigned long long)done);
    double range_sum = 0;
    for (uint64_t i = 0; i < done; i++) range_sum += ranges[i];
    if (done == 0) {
        printf("   no complete blocks read (truncated tensor?)\n");
    } else {
        printf("   block range / RMS: mean %.2f  p50 %.2f  p99 %.2f  max %.2f\n",
               range_sum / done / rms,
               ranges[done / 2] / rms, ranges[(done * 99) / 100] / rms,
               ranges[done - 1] / rms);
        printf("   top-1 element share of block energy (after min shift): "
               "p50 %.3f  p90 %.3f  p99 %.3f\n",
               topshares[done / 2], topshares[(done * 90) / 100],
               topshares[(done * 99) / 100]);
    }
    printf("   outlier-dominated blocks (top-1 share > 50%%): %.1f%%\n",
           100.0 * dom_blocks / done);

    if (row && n % row == 0) {
        uint64_t nr = n / row;
        double *norms = (double *)malloc(sizeof(double) * nr);
        float *rbuf = (float *)malloc(sizeof(float) * row);
        if (norms && rbuf) {
            double nmin = 1e300, nmax = 0, nsum = 0;
            for (uint64_t r = 0; r < nr; r++) {
                if (read_range(f, &t, r * row, row, rbuf) != 0) break;
                double s = 0;
                for (uint64_t i = 0; i < row; i++) s += (double)rbuf[i] * rbuf[i];
                norms[r] = sqrt(s);
                if (norms[r] < nmin) nmin = norms[r];
                if (norms[r] > nmax) nmax = norms[r];
                nsum += norms[r];
            }
            printf("   rows of %llu: %llu; row-norm mean %.6g  min %.6g  "
                   "max %.6g  max/min %.2f\n",
                   (unsigned long long)row, (unsigned long long)nr,
                   nsum / nr, nmin, nmax, nmin > 0 ? nmax / nmin : 0);
        }
        free(norms);
        free(rbuf);
    }
    free(ranges); free(topshares); free(buf);
    fclose(f);
}

int main(int argc, char **argv)
{
    uint64_t block = 256, row = 0;
    const char *shards[16];
    int n_shards = 0, i = 1, sep = 0;
    for (; i < argc; i++) {
        if (strcmp(argv[i], "--") == 0) { sep = i; break; }
        if (strcmp(argv[i], "--block") == 0 && i + 1 < argc)
            block = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--row") == 0 && i + 1 < argc)
            row = strtoull(argv[++i], NULL, 10);
        else if (n_shards < 16) shards[n_shards++] = argv[i];
    }
    if (!sep || n_shards == 0 || sep + 1 >= argc) {
        fprintf(stderr, "usage: nq2-tensorstat [--block N] [--row N] "
                "shard.safetensors [...] -- tensor.name [...]\n");
        return 1;
    }
    for (int tn = sep + 1; tn < argc; tn++) {
        int done = 0;
        for (int s = 0; s < n_shards && !done; s++) {
            TInfo probe; FILE *pf = NULL;
            if (find_tensor(shards[s], argv[tn], &probe, &pf) == 0) {
                fclose(pf);
                analyze(shards[s], argv[tn], block, row);
                done = 1;
            }
        }
        if (!done) printf("%s: not found in any shard\n", argv[tn]);
    }
    return 0;
}
