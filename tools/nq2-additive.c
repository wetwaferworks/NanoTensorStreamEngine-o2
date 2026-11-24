/*
 * nq2-additive.c - NanoQuatTwo Round 3, Priority 3 experiment.
 *
 * Joint additive codebooks vs sequential residual quantization.
 * Both methods use two books of K 256 over block 256 in the SAME
 * one-frame formulation (one minmax frame per block; both books
 * live in normalized space; reconstruction = frame applied to
 * A[i] + B[j]). Identical stream cost: 2 index bytes + 4 frame
 * bytes per block.
 *
 *   sequential: train A on the blocks; train B on the residuals
 *               of A's assignments. Done. (The production
 *               --residual experiment, restated in one frame.)
 *   joint:      start from the sequential solution, then
 *               alternate: coordinate-descent assignment sweeps
 *               (re-pick i given j, re-pick j given i) and book
 *               updates (A_i = mean of x - B_j over its blocks,
 *               B_j = mean of x - A_i over its blocks).
 *
 * Multiple initialization seeds (directive: determine whether
 * initialization sensitivity is limiting fidelity). Sampling
 * positions are seed-independent; only k-means init draws vary.
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

static float *load_tensor(const char **paths, int n_paths,
                          const char *name, uint64_t *count_out)
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

#define BLOCK 256
#define KENT 256

/* assignment by coordinate descent: pick i for x, then j for the
 * residual, then optionally re-pick each given the other */
static void assign_pair(const float *x, const float *A, const float *B,
                        int sweeps, int *oi, int *oj, float *resid_out)
{
    int i = nearest_centroid(A, KENT, BLOCK, x);
    float r[BLOCK];
    for (uint32_t t = 0; t < BLOCK; t++) r[t] = x[t] - A[(size_t)i * BLOCK + t];
    int j = nearest_centroid(B, KENT, BLOCK, r);
    for (int s = 0; s < sweeps; s++) {
        for (uint32_t t = 0; t < BLOCK; t++) r[t] = x[t] - B[(size_t)j * BLOCK + t];
        i = nearest_centroid(A, KENT, BLOCK, r);
        for (uint32_t t = 0; t < BLOCK; t++) r[t] = x[t] - A[(size_t)i * BLOCK + t];
        j = nearest_centroid(B, KENT, BLOCK, r);
    }
    if (resid_out)
        for (uint32_t t = 0; t < BLOCK; t++)
            resid_out[t] = x[t] - A[(size_t)i * BLOCK + t] - B[(size_t)j * BLOCK + t];
    *oi = i;
    *oj = j;
}

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint64_t sample_cap = 8192;
    int iters = 8, rounds = 3;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--rounds") == 0 && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) { fprintf(stderr, "usage: nq2-additive shard...\n"); return 1; }
    uint64_t count = 0;
    float *data = load_tensor(paths, n_paths, tensor, &count);
    if (!data) { fprintf(stderr, "tensor %s not found\n", tensor); return 1; }
    uint64_t nblocks = count / BLOCK;
    printf("tensor: %s\nweights: %llu in %llu blocks of %d, two books K %d, "
           "one minmax frame\n",
           tensor, (unsigned long long)(nblocks * BLOCK),
           (unsigned long long)nblocks, BLOCK, KENT);
    /* sample once, seed-independent positions */
    if (sample_cap > nblocks) sample_cap = nblocks;
    float *sample = (float *)malloc(sample_cap * BLOCK * sizeof(float));
    float *tmp = (float *)malloc(BLOCK * sizeof(float));
    rng_state = 0x2545F4914F6CDD1Dull;
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
    const uint64_t seeds[3] = {42, 7, 1234};
    float *A = (float *)malloc((size_t)KENT * BLOCK * sizeof(float));
    float *B = (float *)malloc((size_t)KENT * BLOCK * sizeof(float));
    float *resid = (float *)malloc(sn * BLOCK * sizeof(float));
    int *ai = (int *)malloc(sn * sizeof(int));
    int *aj = (int *)malloc(sn * sizeof(int));
    for (int si = 0; si < 3; si++) {
        /* sequential: A on blocks, B on residuals of A's picks */
        rng_state = seeds[si] * 0x9E3779B97F4A7C15ull + 0x2545F4914F6CDD1Dull;
        learn_book(sample, sn, BLOCK, KENT, iters, A);
        for (uint64_t b = 0; b < sn; b++) {
            int i = nearest_centroid(A, KENT, BLOCK, sample + b * BLOCK);
            ai[b] = i;
            for (uint32_t t = 0; t < BLOCK; t++)
                resid[b * BLOCK + t] = sample[b * BLOCK + t] - A[(size_t)i * BLOCK + t];
        }
        rng_state = seeds[si] * 0xD1B54A32D192ED03ull + 0x2545F4914F6CDD1Dull;
        learn_book(resid, sn, BLOCK, KENT, iters, B);
        /* fidelity of the sequential pair on the full tensor */
        double err2 = 0.0, ref2 = 0.0;
        for (uint64_t b = 0; b < nblocks; b++) {
            const float *w = data + b * BLOCK;
            float mn, sc;
            normalize_block(w, BLOCK, tmp, &mn, &sc);
            int i2, j2;
            assign_pair(tmp, A, B, 0, &i2, &j2, NULL);
            float rmn = f16_to_float(float_to_half(mn));
            float rsc = f16_to_float(float_to_half(sc));
            for (uint32_t t = 0; t < BLOCK; t++) {
                double rec = (double)rmn + (double)rsc *
                    (A[(size_t)i2 * BLOCK + t] + B[(size_t)j2 * BLOCK + t]);
                double d = (double)w[t] - rec;
                err2 += d * d;
                ref2 += (double)w[t] * (double)w[t];
            }
        }
        double seq_l2 = dsqrt(err2 / ref2);
        /* joint: alternate assignment sweeps and book updates on
         * the sample, then re-measure on the full tensor */
        double jt0 = now_s();
        double *sums = (double *)malloc((size_t)KENT * BLOCK * sizeof(double));
        uint64_t *cnt = (uint64_t *)malloc(KENT * sizeof(uint64_t));
        for (int rd = 0; rd < rounds; rd++) {
            for (uint64_t b = 0; b < sn; b++)
                assign_pair(sample + b * BLOCK, A, B, 1, &ai[b], &aj[b], NULL);
            memset(sums, 0, (size_t)KENT * BLOCK * sizeof(double));
            memset(cnt, 0, KENT * sizeof(uint64_t));
            for (uint64_t b = 0; b < sn; b++) {
                double *sum = sums + (size_t)ai[b] * BLOCK;
                cnt[ai[b]]++;
                for (uint32_t t = 0; t < BLOCK; t++)
                    sum[t] += sample[b * BLOCK + t] - B[(size_t)aj[b] * BLOCK + t];
            }
            for (uint32_t k = 0; k < KENT; k++) {
                if (!cnt[k]) continue;
                float *c = A + (size_t)k * BLOCK;
                for (uint32_t t = 0; t < BLOCK; t++)
                    c[t] = (float)(sums[(size_t)k * BLOCK + t] / (double)cnt[k]);
            }
            memset(sums, 0, (size_t)KENT * BLOCK * sizeof(double));
            memset(cnt, 0, KENT * sizeof(uint64_t));
            for (uint64_t b = 0; b < sn; b++) {
                double *sum = sums + (size_t)aj[b] * BLOCK;
                cnt[aj[b]]++;
                for (uint32_t t = 0; t < BLOCK; t++)
                    sum[t] += sample[b * BLOCK + t] - A[(size_t)ai[b] * BLOCK + t];
            }
            for (uint32_t k = 0; k < KENT; k++) {
                if (!cnt[k]) continue;
                float *c = B + (size_t)k * BLOCK;
                for (uint32_t t = 0; t < BLOCK; t++)
                    c[t] = (float)(sums[(size_t)k * BLOCK + t] / (double)cnt[k]);
            }
        }
        free(sums); free(cnt);
        double joint_s = now_s() - jt0;
        err2 = 0.0; ref2 = 0.0;
        for (uint64_t b = 0; b < nblocks; b++) {
            const float *w = data + b * BLOCK;
            float mn, sc;
            normalize_block(w, BLOCK, tmp, &mn, &sc);
            int i2, j2;
            assign_pair(tmp, A, B, 1, &i2, &j2, NULL);
            float rmn = f16_to_float(float_to_half(mn));
            float rsc = f16_to_float(float_to_half(sc));
            for (uint32_t t = 0; t < BLOCK; t++) {
                double rec = (double)rmn + (double)rsc *
                    (A[(size_t)i2 * BLOCK + t] + B[(size_t)j2 * BLOCK + t]);
                double d = (double)w[t] - rec;
                err2 += d * d;
                ref2 += (double)w[t] * (double)w[t];
            }
        }
        double joint_l2 = dsqrt(err2 / ref2);
        printf("  seed %5llu: sequential %.4f   joint (%d rounds, %.0f s) %.4f   "
               "delta %.4f\n",
               (unsigned long long)seeds[si], seq_l2, rounds, joint_s,
               joint_l2, joint_l2 - seq_l2);
        fflush(stdout);
    }
    printf("rate: stream 6 bytes/block = 0.1875 bpw; books 2 x %d B; "
           "complete on this tensor %.4f bpw\n",
           KENT * BLOCK * 4,
           (double)(nblocks * 6 + 2ull * KENT * BLOCK * 4) * 8.0 / (double)(nblocks * BLOCK));
    free(sample); free(tmp); free(A); free(B); free(resid); free(ai); free(aj);
    free(data);
    return 0;
}
