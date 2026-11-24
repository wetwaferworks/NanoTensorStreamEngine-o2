/*
 * nq2-fused.c - NanoQuatTwo Normalization Research, Test C.
 *
 * Fused reconstruction: can the consuming operation (here a
 * matvec, the shape of every projection in the engine) eat the
 * packed form directly, reconstructing each block's weights on
 * the fly inside the dot product, instead of materializing the
 * expanded tensor first?
 *
 * Two paths over the same in-memory pack of one real tensor:
 *   A materialize: expand the whole tensor to fp32, then matvec.
 *   B fused:       matvec reads indices + frames + book and
 *                  reconstructs each element as it is consumed.
 * Both accumulate in the same order, so the outputs must agree
 * bit for bit; the test reports the max abs difference (expect
 * exactly 0), both timings, and the memory each path holds.
 *
 * VQ core copied from nq2-quantize.c, as in the sibling tools.
 * Direction: JD. Outside review: Gizmo. Implementation: Elfie.
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

static double now_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static uint64_t rng_state = 0x2545F4914F6CDD1Dull;
static uint64_t rng_next(void)
{
    uint64_t x = rng_state;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    rng_state = x;
    return x * 0x2545F4914F6CDD1Dull;
}
static double dsqrt(double x)
{
    if (x <= 0.0) return 0.0;
    double y = x > 1.0 ? x : 1.0;
    for (int i = 0; i < 60; i++) y = 0.5 * (y + x / y);
    return y;
}
static uint16_t float_to_half(float f)
{
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 112;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = mant >> shift;
        if ((mant >> (shift - 1)) & 1u
                && ((half & 1u) || (mant & ((1u << (shift - 1)) - 1))))
            half++;
        return (uint16_t)(sign | half);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0));
    if (mant & 0x1000u && ((mant & 0x2000u) || (mant & 0xfffu))) mant += 0x2000u;
    if (mant & 0x800000u) { mant = 0; exp++; if (exp >= 31) return (uint16_t)(sign | 0x7c00u); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}
static float f16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) bits = sign;
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

static void normalize_block(const float *raw, uint32_t block,
                            float *out, float *mn_out, float *sc_out)
{
    float mn = raw[0], mx = raw[0];
    for (uint32_t i = 1; i < block; i++) {
        if (raw[i] < mn) mn = raw[i];
        if (raw[i] > mx) mx = raw[i];
    }
    float sc = mx - mn;
    if (sc == 0.0f) sc = 1.0f;
    for (uint32_t i = 0; i < block; i++) out[i] = (raw[i] - mn) / sc;
    *mn_out = mn;
    *sc_out = sc;
}

static int nearest_centroid(const float *cent, uint32_t entries,
                            uint32_t block, const float *x)
{
    int best = 0;
    double best_err = 1e30;
    for (uint32_t k = 0; k < entries; k++) {
        const float *c = cent + (size_t)k * block;
        double err = 0.0;
        for (uint32_t i = 0; i < block; i++) {
            double d = (double)x[i] - c[i];
            err += d * d;
        }
        if (err < best_err) { best_err = err; best = (int)k; }
    }
    return best;
}

static void learn_book(const float *sample, uint64_t sample_n,
                       uint32_t block, uint32_t entries, int iters,
                       float *cent)
{
    for (uint32_t k = 0; k < entries; k++) {
        uint64_t src = rng_next() % sample_n;
        memcpy(cent + (size_t)k * block, sample + src * block,
               block * sizeof(float));
    }
    double *sums = (double *)calloc((size_t)entries * block, sizeof(double));
    uint64_t *counts = (uint64_t *)calloc(entries, sizeof(uint64_t));
    if (!sums || !counts) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int it = 0; it < iters; it++) {
        memset(sums, 0, (size_t)entries * block * sizeof(double));
        memset(counts, 0, entries * sizeof(uint64_t));
        for (uint64_t b = 0; b < sample_n; b++) {
            const float *x = sample + b * block;
            int k = nearest_centroid(cent, entries, block, x);
            counts[k]++;
            double *sum = sums + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) sum[i] += x[i];
        }
        for (uint32_t k = 0; k < entries; k++) {
            if (!counts[k]) continue;
            float *c = cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++)
                c[i] = (float)(sums[(size_t)k * block + i] / (double)counts[k]);
        }
    }
    free(sums);
    free(counts);
}

static uint64_t rd64le(const uint8_t *p)
{
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

/* loader returns shape too (rows, cols) */
static float *load_tensor2(const char **paths, int n_paths,
                           const char *name, uint64_t *count_out,
                           uint64_t *rows_out, uint64_t *cols_out)
{
    for (int s = 0; s < n_paths; s++) {
        FILE *f = fopen(paths[s], "rb");
        if (!f) continue;
        uint8_t lb[8];
        if (fread(lb, 1, 8, f) != 8) { fclose(f); continue; }
        uint64_t hlen = rd64le(lb);
        char *hdr = (char *)malloc(hlen + 1);
        if (!hdr || fread(hdr, 1, hlen, f) != hlen) { fclose(f); continue; }
        hdr[hlen] = 0;
        char pat[512];
        snprintf(pat, sizeof(pat), "\"%s\"", name);
        const char *hit = strstr(hdr, pat);
        if (!hit) { free(hdr); fclose(f); continue; }
        const char *obj = strchr(hit + strlen(pat), '{');
        if (!obj) { free(hdr); fclose(f); continue; }
        char dtype[16] = "";
        const char *dp = strstr(obj, "\"dtype\"");
        if (dp) sscanf(dp, "\"dtype\": \"%15[^\"]\"", dtype);
        uint64_t dims[4] = {0, 0, 0, 0};
        int nd = 0;
        uint64_t count = 1;
        const char *shp = strstr(obj, "\"shape\"");
        if (shp) {
            const char *b = strchr(shp, '[');
            const char *e = b ? strchr(b, ']') : NULL;
            if (b && e) {
                char buf[256];
                size_t l = (size_t)(e - b - 1);
                if (l >= sizeof(buf)) l = sizeof(buf) - 1;
                memcpy(buf, b + 1, l);
                buf[l] = 0;
                count = 1;
                char *tok = strtok(buf, ", ");
                if (!tok) count = 0;
                while (tok && nd < 4) {
                    dims[nd++] = strtoull(tok, NULL, 10);
                    count *= dims[nd - 1];
                    tok = strtok(NULL, ", ");
                }
            }
        }
        uint64_t o0 = 0, o1 = 0;
        const char *off = strstr(obj, "\"data_offsets\"");
        if (off) {
            const char *b = strchr(off, '[');
            if (b) sscanf(b + 1, "%llu, %llu",
                          (unsigned long long *)&o0,
                          (unsigned long long *)&o1);
        }
        free(hdr);
        if (count == 0 || o1 <= o0 || nd != 2) { fclose(f); continue; }
        uint64_t nbytes = o1 - o0;
        uint8_t *raw = (uint8_t *)malloc(nbytes);
        float *out = (float *)malloc(count * sizeof(float));
        if (!raw || !out) { fclose(f); return NULL; }
        fseeko(f, (off_t)(8 + hlen + o0), SEEK_SET);
        if (fread(raw, 1, nbytes, f) != nbytes) { fclose(f); return NULL; }
        fclose(f);
        if (strcmp(dtype, "BF16") == 0)
            for (uint64_t i = 0; i < count; i++)
                out[i] = bf16_to_float((uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)));
        else if (strcmp(dtype, "F16") == 0)
            for (uint64_t i = 0; i < count; i++)
                out[i] = f16_to_float((uint16_t)(raw[2 * i] | (raw[2 * i + 1] << 8)));
        else if (strcmp(dtype, "F32") == 0)
            memcpy(out, raw, count * sizeof(float));
        else { free(raw); free(out); continue; }
        free(raw);
        *count_out = count;
        *rows_out = dims[0];
        *cols_out = dims[1];
        return out;
    }
    return NULL;
}

#define BLOCK 256
#define KENT 256

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint64_t sample_cap = 8192;
    int iters = 8, passes = 100;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc) passes = atoi(argv[++i]);
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) { fprintf(stderr, "usage: nq2-fused shard...\n"); return 1; }
    uint64_t count = 0, rows = 0, cols = 0;
    float *data = load_tensor2(paths, n_paths, tensor, &count, &rows, &cols);
    if (!data) { fprintf(stderr, "tensor %s not found (needs 2D shape)\n", tensor); return 1; }
    if (cols % BLOCK != 0) { fprintf(stderr, "cols %llu not a multiple of block\n", (unsigned long long)cols); return 1; }
    uint64_t nblocks = count / BLOCK;
    printf("tensor: %s  [%llu x %llu], %llu blocks of %d\n",
           tensor, (unsigned long long)rows, (unsigned long long)cols,
           (unsigned long long)nblocks, BLOCK);
    /* pack in memory */
    if (sample_cap > nblocks) sample_cap = nblocks;
    float *sample = (float *)malloc(sample_cap * BLOCK * sizeof(float));
    float *tmp = (float *)malloc(BLOCK * sizeof(float));
    uint64_t sn = 0, seen = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        float mn, sc;
        normalize_block(data + b * BLOCK, BLOCK, tmp, &mn, &sc);
        if (sn < sample_cap) { memcpy(sample + sn * BLOCK, tmp, BLOCK * sizeof(float)); sn++; }
        else {
            uint64_t j = rng_next() % (seen + 1);
            if (j < sample_cap) memcpy(sample + j * BLOCK, tmp, BLOCK * sizeof(float));
        }
        seen++;
    }
    float *book = (float *)malloc((size_t)KENT * BLOCK * sizeof(float));
    learn_book(sample, sn, BLOCK, KENT, iters, book);
    free(sample);
    uint8_t *idx = (uint8_t *)malloc(nblocks);
    float *frames = (float *)malloc(nblocks * 2 * sizeof(float));
    float *expanded = (float *)malloc(count * sizeof(float));
    for (uint64_t b = 0; b < nblocks; b++) {
        float mn, sc;
        normalize_block(data + b * BLOCK, BLOCK, tmp, &mn, &sc);
        int k = nearest_centroid(book, KENT, BLOCK, tmp);
        idx[b] = (uint8_t)k;
        float rmn = f16_to_float(float_to_half(mn));
        float rsc = f16_to_float(float_to_half(sc));
        frames[2 * b] = rmn;
        frames[2 * b + 1] = rsc;
        const float *c = book + (size_t)k * BLOCK;
        for (uint32_t i = 0; i < BLOCK; i++)
            expanded[b * BLOCK + i] = rmn + rsc * c[i];
    }
    free(data); free(tmp);
    /* deterministic input vector */
    float *x = (float *)malloc(cols * sizeof(float));
    for (uint64_t i = 0; i < cols; i++)
        x[i] = (float)((int)(rng_next() % 2000) - 1000) / 1000.0f;
    float *yA = (float *)calloc(rows, sizeof(float));
    float *yB = (float *)calloc(rows, sizeof(float));
    /* path A: matvec over the expanded tensor */
    double tA = now_s();
    for (int p = 0; p < passes; p++) {
        memset(yA, 0, rows * sizeof(float));
        for (uint64_t r = 0; r < rows; r++) {
            const float *w = expanded + r * cols;
            double acc = 0.0;
            for (uint64_t i = 0; i < cols; i++) acc += (double)w[i] * x[i];
            yA[r] = (float)acc;
        }
    }
    tA = now_s() - tA;
    /* path B: fused matvec over the packed form */
    double tB = now_s();
    uint64_t bpr = cols / BLOCK;   /* blocks per row */
    for (int p = 0; p < passes; p++) {
        memset(yB, 0, rows * sizeof(float));
        for (uint64_t r = 0; r < rows; r++) {
            double acc = 0.0;
            for (uint64_t bb = 0; bb < bpr; bb++) {
                uint64_t b = r * bpr + bb;
                float rmn = frames[2 * b], rsc = frames[2 * b + 1];
                const float *c = book + (size_t)idx[b] * BLOCK;
                const float *xv = x + bb * BLOCK;
                for (uint32_t i = 0; i < BLOCK; i++)
                    acc += (double)(rmn + rsc * c[i]) * xv[i];
            }
            yB[r] = (float)acc;
        }
    }
    tB = now_s() - tB;
    double md = 0.0;
    for (uint64_t r = 0; r < rows; r++) {
        double d = fabs((double)yA[r] - (double)yB[r]);
        if (d > md) md = d;
    }
    uint64_t packed_bytes = nblocks + nblocks * 4 + (uint64_t)KENT * BLOCK * 4;
    printf("path A materialize+matvec: %.3f s for %d passes (%.1f M weights/s)\n",
           tA, passes, (double)count * passes / tA / 1e6);
    printf("path B fused matvec:        %.3f s for %d passes (%.1f M weights/s)\n",
           tB, passes, (double)count * passes / tB / 1e6);
    printf("output agreement: max abs diff %.3e (same accumulation order)\n", md);
    printf("memory: expanded %.1f MB held by A; packed %.2f MB held by B "
           "(indices + frames + book)\n",
           (double)count * 4 / 1e6, (double)packed_bytes / 1e6);
    free(x); free(yA); free(yB); free(idx); free(frames); free(expanded); free(book);
    return 0;
}
