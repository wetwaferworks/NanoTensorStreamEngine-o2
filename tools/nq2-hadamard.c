/*
 * nq2-hadamard.c - NanoQuatTwo Round 3, Priority 1 experiment.
 *
 * Does a randomized Hadamard transform, applied inside each block
 * before vector quantization, improve reconstruction fidelity at
 * the preserved baseline (block 256, K 256)? The transform is
 * orthogonal, so L2 error in transformed space equals L2 error in
 * original space up to float rounding; both are reported.
 *
 * The inverse transform is tested two ways at decode time:
 *   naive:  apply the inverse (signs, then FWHT) to every
 *           reconstructed block, paying O(n log n) per block;
 *   folded: migrate the inverse into the book at encode time.
 *           The reconstruction is linear in the frame, so
 *           W = D H (mn * 1 + sc * proto) = sc * (D H proto)
 *           + mn * (D H 1): store book' = D H proto per entry and
 *           one constant vector v = D H 1 per tensor, and decode
 *           costs the same FMA loop as the baseline plus one
 *           vector add. The fold is verified element-for-element
 *           against the naive inverse.
 *
 * The VQ core (normalize, nearest, k-means, rounding) is copied
 * from nq2-quantize.c so the experiment measures the production
 * algorithm, not a sketch of it. Sampling and k-means draw from
 * the same fixed seed in both paths, so the A/B differs only by
 * the transform. Transform seed is recorded in the output.
 *
 * Self-contained and independently testable, per the Round 3
 * engineering constraint: nothing here touches the production
 * reader or the shipped baseline.
 *
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

/* ---- copied VQ core (from nq2-quantize.c) ------------------------- */

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

static int g_norm_rms = 0;
static void normalize_block(const float *raw, uint32_t block,
                            float *out, float *mn_out, float *sc_out)
{
    if (g_norm_rms) {
        double s = 0.0;
        for (uint32_t i = 0; i < block; i++) s += (double)raw[i] * raw[i];
        float sc = (float)dsqrt(s / block);
        if (sc == 0.0f) sc = 1.0f;
        for (uint32_t i = 0; i < block; i++) out[i] = raw[i] / sc;
        *mn_out = 0.0f;
        *sc_out = sc;
        return;
    }
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

/* ---- tensor loading (one named tensor, whole, as fp32) ------------ */

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
        /* find "name" then its object */
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

/* ---- Hadamard machinery -------------------------------------------- */

/* Normalized Walsh-Hadamard transform, in place; orthogonal and
 * symmetric, so it is its own inverse. n must be a power of two. */
static void fwht(float *a, uint32_t n)
{
    for (uint32_t len = 1; len < n; len <<= 1)
        for (uint32_t i = 0; i < n; i += len << 1)
            for (uint32_t j = 0; j < len; j++) {
                float u = a[i + j], v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
    float s = 1.0f / sqrtf((float)n);
    for (uint32_t i = 0; i < n; i++) a[i] *= s;
}

/* forward: t = H (D w); inverse: w = D (H t) */
static void xform_fwd(const float *w, const float *sgn, uint32_t n, float *t)
{
    for (uint32_t i = 0; i < n; i++) t[i] = w[i] * sgn[i];
    fwht(t, n);
}
static void xform_inv(const float *t, const float *sgn, uint32_t n, float *w)
{
    for (uint32_t i = 0; i < n; i++) w[i] = t[i];
    fwht(w, n);
    for (uint32_t i = 0; i < n; i++) w[i] *= sgn[i];
}

/* ---- the experiment ------------------------------------------------- */

typedef struct {
    double rel_l2;        /* original space */
    double encode_s;
    double decode_mws;    /* reconstruction only, working space */
    double decode_inv_mws;/* + naive inverse per block (xform only) */
    double decode_fold_mws;/* folded book (xform only) */
    double fold_maxdiff;
    double fold_rel_l2;
} Result;

static Result run_vq(const float *orig, const float *work, uint64_t nblocks,
                     uint32_t block, uint32_t entries, uint64_t sample_cap,
                     int iters, int xform, const float *sgn)
{
    Result R;
    memset(&R, 0, sizeof(R));
    double t0 = now_s();
    /* sample (reservoir, identical policy both paths) */
    if (sample_cap > nblocks) sample_cap = nblocks;
    float *sample = (float *)malloc(sample_cap * block * sizeof(float));
    float *tmp = (float *)malloc(block * sizeof(float));
    uint64_t sn = 0, seen = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        float mn, sc;
        normalize_block(work + b * block, block, tmp, &mn, &sc);
        if (sn < sample_cap) {
            memcpy(sample + sn * block, tmp, block * sizeof(float));
            sn++;
        } else {
            uint64_t j = rng_next() % (seen + 1);
            if (j < sample_cap)
                memcpy(sample + j * block, tmp, block * sizeof(float));
        }
        seen++;
    }
    float *cent = (float *)malloc((size_t)entries * block * sizeof(float));
    learn_book(sample, sn, block, entries, iters, cent);
    free(sample);
    /* folded book: book' = D (H proto) per entry, v = D (H 1) */
    float *bookp = NULL, *vvec = NULL;
    if (xform) {
        bookp = (float *)malloc((size_t)entries * block * sizeof(float));
        vvec = (float *)malloc(block * sizeof(float));
        for (uint32_t e = 0; e < entries; e++) {
            float *dst = bookp + (size_t)e * block;
            memcpy(dst, cent + (size_t)e * block, block * sizeof(float));
            fwht(dst, block);
            for (uint32_t i = 0; i < block; i++) dst[i] *= sgn[i];
        }
        for (uint32_t i = 0; i < block; i++) vvec[i] = 1.0f;
        fwht(vvec, block);
        for (uint32_t i = 0; i < block; i++) vvec[i] *= sgn[i];
    }
    /* assignment + fidelity in original space */
    double err2 = 0.0, ref2 = 0.0, ferr2 = 0.0;
    float *rec = (float *)malloc(block * sizeof(float));
    float *recw = (float *)malloc(block * sizeof(float));
    for (uint64_t b = 0; b < nblocks; b++) {
        const float *w = work + b * block;
        float mn, sc;
        normalize_block(w, block, tmp, &mn, &sc);
        int k = nearest_centroid(cent, entries, block, tmp);
        float rmn = f16_to_float(float_to_half(mn));
        float rsc = f16_to_float(float_to_half(sc));
        const float *c = cent + (size_t)k * block;
        for (uint32_t i = 0; i < block; i++) rec[i] = rmn + rsc * c[i];
        const float *cmp = rec;
        if (xform) {
            xform_inv(rec, sgn, block, recw);
            cmp = recw;
            /* folded reconstruction of the same block */
            const float *bp = bookp + (size_t)k * block;
            double md = 0.0;
            for (uint32_t i = 0; i < block; i++) {
                double fv = (double)rsc * bp[i] + (double)rmn * vvec[i];
                double d = fv - (double)cmp[i];
                if (d < 0) d = -d;
                if (d > md) md = d;
                double dd = (double)orig[b * block + i] - fv;
                ferr2 += dd * dd;
            }
            if (md > R.fold_maxdiff) R.fold_maxdiff = md;
        }
        for (uint32_t i = 0; i < block; i++) {
            double d = (double)orig[b * block + i] - (double)cmp[i];
            err2 += d * d;
            ref2 += (double)orig[b * block + i] * (double)orig[b * block + i];
        }
    }
    R.rel_l2 = ref2 > 0 ? dsqrt(err2 / ref2) : 0.0;
    R.fold_rel_l2 = ref2 > 0 ? dsqrt(ferr2 / ref2) : 0.0;
    R.encode_s = now_s() - t0;
    /* decode timing: pure reconstruction passes */
    uint64_t total = nblocks * block;
    {
        double td = now_s();
        volatile float sink = 0.0f;
        for (uint64_t b = 0; b < nblocks; b++) {
            const float *w = work + b * block;
            float mn, sc;
            normalize_block(w, block, tmp, &mn, &sc);
            int k = nearest_centroid(cent, entries, block, tmp);
            float rmn = f16_to_float(float_to_half(mn));
            float rsc = f16_to_float(float_to_half(sc));
            const float *c = cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) rec[i] = rmn + rsc * c[i];
            sink += rec[0];
        }
        R.decode_mws = (double)total / (now_s() - td) / 1e6;
        (void)sink;
    }
    if (xform) {
        double td = now_s();
        volatile float sink = 0.0f;
        for (uint64_t b = 0; b < nblocks; b++) {
            const float *w = work + b * block;
            float mn, sc;
            normalize_block(w, block, tmp, &mn, &sc);
            int k = nearest_centroid(cent, entries, block, tmp);
            float rmn = f16_to_float(float_to_half(mn));
            float rsc = f16_to_float(float_to_half(sc));
            const float *c = cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) rec[i] = rmn + rsc * c[i];
            xform_inv(rec, sgn, block, recw);
            sink += recw[0];
        }
        R.decode_inv_mws = (double)total / (now_s() - td) / 1e6;
        td = now_s();
        for (uint64_t b = 0; b < nblocks; b++) {
            const float *w = work + b * block;
            float mn, sc;
            normalize_block(w, block, tmp, &mn, &sc);
            int k = nearest_centroid(cent, entries, block, tmp);
            float rmn = f16_to_float(float_to_half(mn));
            float rsc = f16_to_float(float_to_half(sc));
            const float *bp = bookp + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++)
                rec[i] = rsc * bp[i] + rmn * vvec[i];
            sink += rec[0];
        }
        R.decode_fold_mws = (double)total / (now_s() - td) / 1e6;
        (void)sink;
    }
    free(tmp); free(rec); free(recw); free(cent);
    free(bookp); free(vvec);
    return R;
}

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint32_t block = 256, entries = 256;
    uint64_t sample_cap = 8192;
    int iters = 8;
    uint64_t seed = 42;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) block = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--entries") == 0 && i + 1 < argc) entries = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--sample-blocks") == 0 && i + 1 < argc) sample_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) { const char *m = argv[++i]; g_norm_rms = strcmp(m, "rms") == 0; }
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) {
        fprintf(stderr, "usage: nq2-hadamard [--tensor NAME] [--seed N] shard...\n");
        return 1;
    }
    if (block == 0 || (block & (block - 1)) != 0 || entries == 0) {
        fprintf(stderr, "block must be a nonzero power of two for "
                "FWHT and entries at least 1\n");
        return 1;
    }
    uint64_t count = 0;
    float *data = load_tensor(paths, n_paths, tensor, &count);
    if (!data) { fprintf(stderr, "tensor %s not found\n", tensor); return 1; }
    uint64_t nblocks = count / block;
    printf("tensor: %s\nweights: %llu in %llu blocks of %u, K %u, norm %s\n",
           tensor, (unsigned long long)(nblocks * block),
           (unsigned long long)nblocks, block, entries,
           g_norm_rms ? "rms" : "minmax");
    printf("transform seed: %llu (sign pattern, per tensor)\n",
           (unsigned long long)seed);

    /* bare transform round-trip check */
    {
        float *sgn = (float *)malloc(block * sizeof(float));
        uint64_t saved = rng_state;
        rng_state = seed * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
        for (uint32_t i = 0; i < block; i++)
            sgn[i] = (rng_next() & 1) ? 1.0f : -1.0f;
        rng_state = saved;
        float *t = (float *)malloc(block * sizeof(float));
        float *bk = (float *)malloc(block * sizeof(float));
        double md = 0.0;
        for (uint64_t b = 0; b < nblocks && b < 64; b++) {
            xform_fwd(data + b * block, sgn, block, t);
            xform_inv(t, sgn, block, bk);
            for (uint32_t i = 0; i < block; i++) {
                double d = fabs((double)bk[i] - (double)data[b * block + i]);
                if (d > md) md = d;
            }
        }
        printf("bare transform round trip: max abs error %.3e over 64 blocks\n", md);
        /* build the transformed working copy */
        float *tdata = (float *)malloc(nblocks * block * sizeof(float));
        for (uint64_t b = 0; b < nblocks; b++)
            xform_fwd(data + b * block, sgn, block, tdata + b * block);
        /* baseline */
        rng_state = 0x2545F4914F6CDD1Dull;
        Result base = run_vq(data, data, nblocks, block, entries,
                             sample_cap, iters, 0, NULL);
        printf("\nbaseline (no transform):\n");
        printf("  relative L2: %.4f\n  encode: %.2f s; decode: %.1f M weights/s\n",
               base.rel_l2, base.encode_s, base.decode_mws);
        rng_state = 0x2545F4914F6CDD1Dull;
        Result had = run_vq(data, tdata, nblocks, block, entries,
                            sample_cap, iters, 1, sgn);
        printf("\nhadamard (randomized signs + normalized FWHT per block):\n");
        printf("  relative L2 (original space): %.4f\n", had.rel_l2);
        printf("  encode: %.2f s; decode working space: %.1f M weights/s\n",
               had.encode_s, had.decode_mws);
        printf("  decode + naive inverse: %.1f M weights/s\n", had.decode_inv_mws);
        printf("  decode folded book: %.1f M weights/s; folded rel L2 %.4f; "
               "fold vs naive max abs diff %.3e\n",
               had.decode_fold_mws, had.fold_rel_l2, had.fold_maxdiff);
        printf("\nverdict: baseline %.4f -> hadamard %.4f (delta %.4f)\n",
               base.rel_l2, had.rel_l2, had.rel_l2 - base.rel_l2);
        free(tdata); free(sgn); free(t); free(bk);
    }
    free(data);
    return 0;
}
