/*
 * nq2-framed.c - NanoQuatTwo Normalization Research, Test D.
 *
 * RMS frame precision: the production frame stores the block RMS
 * as fp16 (relative rounding up to about 2^-11). Test D asks what
 * that rounding costs: same tensor, same book, same assignments,
 * reconstruction with the fp16-rounded scale against the same
 * reconstruction with the scale kept in fp32. Reports both
 * relative L2 values, the delta, and the artifact price of the
 * wider frame (+2 bytes per block for an fp32 scale).
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

/* RMS normalization only (this test's subject) */
static void normalize_rms(const float *raw, uint32_t block,
                          float *out, float *sc_out)
{
    double s = 0.0;
    for (uint32_t i = 0; i < block; i++) s += (double)raw[i] * raw[i];
    float sc = (float)dsqrt(s / block);
    if (sc == 0.0f) sc = 1.0f;
    for (uint32_t i = 0; i < block; i++) out[i] = raw[i] / sc;
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
    if (sample_n == 0) { fprintf(stderr, "learn_book: no samples\n"); exit(1); }
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

static float *load_tensor(const char **paths, int n_paths,
                          const char *name, uint64_t *count_out)
{
    for (int s = 0; s < n_paths; s++) {
        FILE *f = fopen(paths[s], "rb");
        if (!f) continue;
        uint8_t lb[8];
        if (fread(lb, 1, 8, f) != 8) { fclose(f); continue; }
        uint64_t hlen = rd64le(lb);
        if (hlen == 0 || hlen > (1ull << 28)) { fclose(f); continue; }
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
                while (tok) { count *= strtoull(tok, NULL, 10); tok = strtok(NULL, ", "); }
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
        if (count == 0 || o1 <= o0) { fclose(f); continue; }
        uint64_t nbytes = o1 - o0;
        /* T3 hardening (CODE-REVIEW-2026-11-24): the shape-derived
         * count and the data_offsets span must agree exactly for
         * the declared dtype, and count * 4 must not wrap. */
        {
            uint64_t elem = strcmp(dtype, "F32") == 0 ? 4 : 2;
            if (count > UINT64_MAX / sizeof(float)
                || nbytes != count * elem) { fclose(f); continue; }
        }
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
        return out;
    }
    return NULL;
}

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint32_t block = 32, K = 256;
    uint64_t sample_cap = 8192;
    int iters = 8;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) { fprintf(stderr, "usage: nq2-framed shard...\n"); return 1; }
    uint64_t count = 0;
    float *data = load_tensor(paths, n_paths, tensor, &count);
    if (!data) { fprintf(stderr, "tensor %s not found\n", tensor); return 1; }
    if (block == 0) { fprintf(stderr, "--block must be at least 1\n"); return 1; }
    uint64_t nblocks = count / block;
    printf("tensor: %s\nweights: %llu in %llu blocks of %u, K %u, RMS norm\n",
           tensor, (unsigned long long)(nblocks * block),
           (unsigned long long)nblocks, block, K);
    double t0 = now_s();
    if (sample_cap > nblocks) sample_cap = nblocks;
    float *sample = (float *)malloc(sample_cap * block * sizeof(float));
    float *tmp = (float *)malloc(block * sizeof(float));
    uint64_t sn = 0, seen = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        float sc;
        normalize_rms(data + b * block, block, tmp, &sc);
        if (sn < sample_cap) { memcpy(sample + sn * block, tmp, block * sizeof(float)); sn++; }
        else {
            uint64_t j = rng_next() % (seen + 1);
            if (j < sample_cap) memcpy(sample + j * block, tmp, block * sizeof(float));
        }
        seen++;
    }
    float *cent = (float *)malloc((size_t)K * block * sizeof(float));
    learn_book(sample, sn, block, K, iters, cent);
    free(sample);
    double err16 = 0.0, err32 = 0.0, ref2 = 0.0, cross = 0.0;
    for (uint64_t b = 0; b < nblocks; b++) {
        const float *w = data + b * block;
        float sc;
        normalize_rms(w, block, tmp, &sc);
        int k = nearest_centroid(cent, K, block, tmp);
        const float *c = cent + (size_t)k * block;
        float sc16 = f16_to_float(float_to_half(sc));
        for (uint32_t i = 0; i < block; i++) {
            double r16 = (double)sc16 * c[i];
            double r32 = (double)sc * c[i];
            double d16 = (double)w[i] - r16;
            double d32 = (double)w[i] - r32;
            err16 += d16 * d16;
            err32 += d32 * d32;
            ref2 += (double)w[i] * (double)w[i];
            double dc = r16 - r32;
            cross += dc * dc;
        }
    }
    double l16 = dsqrt(err16 / ref2), l32 = dsqrt(err32 / ref2);
    printf("  RMS scale stored fp16: relative L2 %.6f\n", l16);
    printf("  RMS scale stored fp32: relative L2 %.6f\n", l32);
    printf("  delta: %.6f (fp16 frame costs %.4f%% of the error)\n",
           l16 - l32, 100.0 * (l16 - l32) / l32);
    printf("  the two reconstructions differ from each other by rel L2 %.6f\n",
           dsqrt(cross / ref2));
    printf("  price of fp32 scale: +2 bytes/block = +%.4f bpw at block %u; "
           "encode %.1f s\n",
           16.0 / block, block, now_s() - t0);
    free(tmp); free(cent); free(data);
    return 0;
}
