/*
 * nq2-pq.c - NanoQuatTwo Round 3, Priority 2 experiment.
 *
 * Product quantization: each normalized 256-block is split into
 * subvectors of d dimensions (32 subspaces at d 8, 16 at d 16),
 * and each subspace gets its own k-means book of K entries. The
 * frame (fp16 min + fp16 scale per 256-block) is unchanged from
 * production. The representable set grows combinatorially
 * (K^(256/d) combinations) while the book bytes stay at
 * 256 * K * 4... the same bytes as the flat block-256 book at
 * the same K. That is the whole bet: coverage without book growth.
 *
 * Rates are reported COMPLETE (stream + books) in bits per
 * weight, byte-indexed as the current format stores them, with
 * the ideal bit-packed rate in parentheses where K < 256.
 *
 * The VQ core is copied from nq2-quantize.c (same discipline as
 * nq2-hadamard.c): same reservoir policy, same seed reset per
 * configuration, so configs differ only in representation.
 * Decode timing here is PURE reconstruction from stored indices
 * and frames (the hadamard tool's timing included assignment;
 * this one does not).
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

/* ---- tensor loading (same as nq2-hadamard.c) ----------------------- */

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

/* ---- the PQ experiment ---------------------------------------------- */

#define BLOCK 256

typedef struct {
    double rel_l2;
    double encode_s;
    double decode_mws;   /* pure reconstruction from stored indices */
    double bpw_complete; /* stream (byte indices) + books */
    double bpw_packed;   /* ideal bit-packed indices + books */
    uint64_t book_bytes;
    uint64_t stream_bytes;
} PQResult;

/* d = subvector dim (BLOCK means flat VQ), K = entries per book */
static PQResult run_pq(const float *data, uint64_t nblocks,
                       uint32_t d, uint32_t K, uint64_t sample_cap,
                       int iters)
{
    PQResult R;
    memset(&R, 0, sizeof(R));
    uint32_t nsub = BLOCK / d;
    double t0 = now_s();
    /* sample normalized blocks (identical policy every config) */
    if (sample_cap > nblocks) sample_cap = nblocks;
    float *sample = (float *)malloc(sample_cap * BLOCK * sizeof(float));
    float *tmp = (float *)malloc(BLOCK * sizeof(float));
    uint64_t sn = 0, seen = 0;
    for (uint64_t b = 0; b < nblocks; b++) {
        float mn, sc;
        normalize_block(data + b * BLOCK, BLOCK, tmp, &mn, &sc);
        if (sn < sample_cap) {
            memcpy(sample + sn * BLOCK, tmp, BLOCK * sizeof(float));
            sn++;
        } else {
            uint64_t j = rng_next() % (seen + 1);
            if (j < sample_cap)
                memcpy(sample + j * BLOCK, tmp, BLOCK * sizeof(float));
        }
        seen++;
    }
    /* learn one book per subspace */
    float *books = (float *)malloc((size_t)nsub * K * d * sizeof(float));
    float *sub_sample = (float *)malloc(sn * d * sizeof(float));
    for (uint32_t s = 0; s < nsub; s++) {
        for (uint64_t b = 0; b < sn; b++)
            memcpy(sub_sample + b * d, sample + b * BLOCK + s * d,
                   d * sizeof(float));
        learn_book(sub_sample, sn, d, K, iters, books + (size_t)s * K * d);
    }
    free(sub_sample);
    free(sample);
    R.book_bytes = (uint64_t)nsub * K * d * 4;
    /* assignment + fidelity; store indices and frames for decode */
    uint8_t *idx = (uint8_t *)malloc(nblocks * nsub);
    float *frames = (float *)malloc(nblocks * 2 * sizeof(float));
    double err2 = 0.0, ref2 = 0.0;
    float *rec = (float *)malloc(BLOCK * sizeof(float));
    for (uint64_t b = 0; b < nblocks; b++) {
        const float *w = data + b * BLOCK;
        float mn, sc;
        normalize_block(w, BLOCK, tmp, &mn, &sc);
        float rmn = f16_to_float(float_to_half(mn));
        float rsc = f16_to_float(float_to_half(sc));
        frames[2 * b] = rmn;
        frames[2 * b + 1] = rsc;
        for (uint32_t s = 0; s < nsub; s++) {
            int k = nearest_centroid(books + (size_t)s * K * d, K, d,
                                     tmp + s * d);
            idx[b * nsub + s] = (uint8_t)k;
            const float *c = books + ((size_t)s * K + k) * d;
            for (uint32_t i = 0; i < d; i++)
                rec[s * d + i] = rmn + rsc * c[i];
        }
        for (uint32_t i = 0; i < BLOCK; i++) {
            double dd = (double)w[i] - (double)rec[i];
            err2 += dd * dd;
            ref2 += (double)w[i] * (double)w[i];
        }
    }
    R.rel_l2 = ref2 > 0 ? dsqrt(err2 / ref2) : 0.0;
    R.encode_s = now_s() - t0;
    R.stream_bytes = nblocks * (nsub + 4);
    uint64_t weights = nblocks * BLOCK;
    R.bpw_complete = (double)(R.stream_bytes + R.book_bytes) * 8.0 / (double)weights;
    /* ideal packed: bits per index = log2(K) */
    int bits = 0;
    while ((1u << bits) < K) bits++;
    double packed_stream = (double)nblocks * ((double)nsub * bits / 8.0 + 4.0);
    R.bpw_packed = (packed_stream + (double)R.book_bytes) * 8.0 / (double)weights;
    /* pure decode timing from stored indices + frames */
    {
        double td = now_s();
        volatile float sink = 0.0f;
        for (uint64_t b = 0; b < nblocks; b++) {
            float rmn = frames[2 * b], rsc = frames[2 * b + 1];
            for (uint32_t s = 0; s < nsub; s++) {
                const float *c = books + ((size_t)s * K + idx[b * nsub + s]) * d;
                for (uint32_t i = 0; i < d; i++)
                    rec[s * d + i] = rmn + rsc * c[i];
            }
            sink += rec[0];
        }
        R.decode_mws = (double)weights / (now_s() - td) / 1e6;
        (void)sink;
    }
    free(idx); free(frames); free(rec); free(tmp); free(books);
    return R;
}

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint64_t sample_cap = 8192;
    int iters = 8;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) { const char *m = argv[++i]; g_norm_rms = strcmp(m, "rms") == 0; }
        else if (strcmp(argv[i], "--sample-blocks") == 0 && i + 1 < argc) sample_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) {
        fprintf(stderr, "usage: nq2-pq [--tensor NAME] shard...\n");
        return 1;
    }
    uint64_t count = 0;
    float *data = load_tensor(paths, n_paths, tensor, &count);
    if (!data) { fprintf(stderr, "tensor %s not found\n", tensor); return 1; }
    uint64_t nblocks = count / BLOCK;
    printf("tensor: %s\nweights: %llu in %llu blocks of %d (%s frame)\n",
           tensor, (unsigned long long)(nblocks * BLOCK),
           (unsigned long long)nblocks, BLOCK,
           g_norm_rms ? "rms" : "minmax");
    struct { uint32_t d, K; const char *label; } cfgs[] = {
        {256, 256, "flat VQ block 256 (baseline)"},
        {16, 256, "PQ d16 K256"},
        {8, 256, "PQ d8 K256"},
        {4, 256, "PQ d4 K256"},
        {16, 64, "PQ d16 K64"},
        {8, 64, "PQ d8 K64"},
        {4, 64, "PQ d4 K64"},
        {16, 16, "PQ d16 K16"},
        {8, 16, "PQ d8 K16"},
    };
    for (size_t c = 0; c < sizeof(cfgs) / sizeof(cfgs[0]); c++) {
        rng_state = 0x2545F4914F6CDD1Dull;
        PQResult r = run_pq(data, nblocks, cfgs[c].d, cfgs[c].K,
                            sample_cap, iters);
        printf("  %-30s rel L2 %.4f   complete %.4f bpw (packed %.4f)   "
               "books %llu B   encode %.1f s   decode %.1f M w/s\n",
               cfgs[c].label, r.rel_l2, r.bpw_complete, r.bpw_packed,
               (unsigned long long)r.book_bytes, r.encode_s, r.decode_mws);
        fflush(stdout);
    }
    free(data);
    return 0;
}
