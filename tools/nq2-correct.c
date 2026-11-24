/*
 * nq2-correct.c - Nano Quant Engine, Gizmo Round 4 P3 diagnostic:
 * the per-entry correction probe.
 *
 * Question (Elfie's proposal, Gizmo's Round 4 section 5): after
 * k-means converges, does a usable per-entry mean residual
 * survive, and can it be harvested for free?
 *
 * For converged k-means the centroid IS the mean of its assigned
 * vectors, so the normalized-space residual mean should sit near
 * zero. Corrections can still exist when training stops early,
 * when reconstruction happens in a different space than training
 * (a block scale multiplies the normalized error back into
 * weight space), or when the objective differs. This probe
 * measures the residual mean in BOTH spaces and compares, on
 * held-out blocks:
 *
 *   1. base      the book as trained
 *   2. tableN    book + per-entry correction table measured in
 *                normalized space, applied scaled by the block's
 *                reconstruction scale (costs a second book)
 *   3. tableW    book + per-entry correction table measured in
 *                reconstructed weight space, applied raw
 *                (costs a second book)
 *   4. folded    the normalized correction folded into the
 *                centroids themselves (zero extra bytes; must
 *                reproduce tableN's numbers)
 *   5. weighted  the book retrained with each sample weighted
 *                by the square of its block's reconstruction
 *                scale, which is the training objective that
 *                matches weight-space error (zero extra bytes)
 *
 * Discipline: even-indexed blocks form the training pool,
 * odd-indexed blocks the held-out pool; no block is ever in
 * both. Frames are rounded through fp16 exactly as production
 * stores them. The VQ core and tensor loader are copied from
 * nq2-pq.c, same reservoir policy, same seed discipline.
 *
 * Direction: Gizmo (Round 4 P3). Executive direction: JD.
 * Implementation: Elfie (E.L.F.I.E.).
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

/* ---- copied VQ core (from nq2-pq.c / nq2-quantize.c) --------------- */

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

/* k-means; when wts is non-NULL the centroid update is the
 * weighted mean (assignment is unchanged: a per-block weight is
 * constant across candidate centroids). */
static void learn_book_w(const float *sample, const double *wts,
                         uint64_t sample_n, uint32_t block,
                         uint32_t entries, int iters, float *cent)
{
    for (uint32_t k = 0; k < entries; k++) {
        uint64_t src = rng_next() % sample_n;
        memcpy(cent + (size_t)k * block, sample + src * block,
               block * sizeof(float));
    }
    double *sums = (double *)calloc((size_t)entries * block, sizeof(double));
    double *wsums = (double *)calloc(entries, sizeof(double));
    if (!sums || !wsums) { fprintf(stderr, "out of memory\n"); exit(1); }
    for (int it = 0; it < iters; it++) {
        memset(sums, 0, (size_t)entries * block * sizeof(double));
        memset(wsums, 0, entries * sizeof(double));
        for (uint64_t b = 0; b < sample_n; b++) {
            const float *x = sample + b * block;
            int k = nearest_centroid(cent, entries, block, x);
            double w = wts ? wts[b] : 1.0;
            wsums[k] += w;
            double *sum = sums + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++) sum[i] += w * x[i];
        }
        for (uint32_t k = 0; k < entries; k++) {
            if (wsums[k] <= 0.0) continue;
            float *c = cent + (size_t)k * block;
            for (uint32_t i = 0; i < block; i++)
                c[i] = (float)(sums[(size_t)k * block + i] / wsums[k]);
        }
    }
    free(sums);
    free(wsums);
}

/* ---- tensor loading (same as nq2-pq.c) ------------------------------ */

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

/* ---- the probe ------------------------------------------------------- */

#define BLOCK 256

/* Evaluate one variant on the held-out pool. mode: 0 = book
 * only, 1 = book + normalized correction scaled by block scale,
 * 2 = book + weight-space correction added raw. */
static double eval_variant(const float *cent, const float *corr,
                           int mode, const float *pool_raw,
                           const float *pool_norm, const float *pool_mn,
                           const float *pool_sc, uint64_t pool_n,
                           uint32_t entries)
{
    double err2 = 0.0, ref2 = 0.0;
    for (uint64_t b = 0; b < pool_n; b++) {
        const float *x = pool_norm + b * BLOCK;
        const float *w = pool_raw + b * BLOCK;
        int k = nearest_centroid(cent, entries, BLOCK, x);
        const float *c = cent + (size_t)k * BLOCK;
        float mn = pool_mn[b], sc = pool_sc[b];
        for (uint32_t i = 0; i < BLOCK; i++) {
            double rec = (double)mn + (double)sc * c[i];
            if (mode == 1) rec += (double)sc * corr[(size_t)k * BLOCK + i];
            else if (mode == 2) rec += corr[(size_t)k * BLOCK + i];
            double dd = (double)w[i] - rec;
            err2 += dd * dd;
            ref2 += (double)w[i] * (double)w[i];
        }
    }
    return ref2 > 0 ? dsqrt(err2 / ref2) : 0.0;
}

int main(int argc, char **argv)
{
    const char *tensor = "encoder.layers.0.self_attn.q_proj.weight";
    uint32_t entries = 256;
    uint64_t sample_cap = 8192, eval_cap = 8192;
    int iters = 25;
    const char *paths[16];
    int n_paths = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) tensor = argv[++i];
        else if (strcmp(argv[i], "--entries") == 0 && i + 1 < argc) entries = (uint32_t)strtoul(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--norm") == 0 && i + 1 < argc) { const char *m = argv[++i]; g_norm_rms = strcmp(m, "rms") == 0; }
        else if (strcmp(argv[i], "--sample-blocks") == 0 && i + 1 < argc) sample_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--eval-blocks") == 0 && i + 1 < argc) eval_cap = strtoull(argv[++i], NULL, 10);
        else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) iters = atoi(argv[++i]);
        else if (n_paths < 16) paths[n_paths++] = argv[i];
    }
    if (n_paths == 0) {
        fprintf(stderr, "usage: nq2-correct [--tensor NAME] [--norm rms|minmax] shard...\n");
        return 1;
    }
    uint64_t count = 0;
    float *data = load_tensor(paths, n_paths, tensor, &count);
    if (!data) { fprintf(stderr, "tensor %s not found\n", tensor); return 1; }
    uint64_t nblocks = count / BLOCK;
    uint64_t train_pool = (nblocks + 1) / 2, eval_pool = nblocks / 2;
    printf("tensor: %s\nweights: %llu in %llu blocks of %d (%s frame), "
           "K %u\ntrain pool %llu blocks (even), held-out pool %llu "
           "blocks (odd)\n",
           tensor, (unsigned long long)(nblocks * BLOCK),
           (unsigned long long)nblocks, BLOCK,
           g_norm_rms ? "rms" : "minmax", entries,
           (unsigned long long)train_pool, (unsigned long long)eval_pool);
    double t0 = now_s();

    /* training sample: reservoir over the even pool, with each
     * block's (fp16-rounded) frame kept beside it */
    if (sample_cap > train_pool) sample_cap = train_pool;
    float *sample = (float *)malloc(sample_cap * BLOCK * sizeof(float));
    float *s_mn = (float *)malloc(sample_cap * sizeof(float));
    float *s_sc = (float *)malloc(sample_cap * sizeof(float));
    float *tmp = (float *)malloc(BLOCK * sizeof(float));
    uint64_t sn = 0, seen = 0;
    for (uint64_t b = 0; b < nblocks; b += 2) {
        float mn, sc;
        normalize_block(data + b * BLOCK, BLOCK, tmp, &mn, &sc);
        mn = f16_to_float(float_to_half(mn));
        sc = f16_to_float(float_to_half(sc));
        if (sn < sample_cap) {
            memcpy(sample + sn * BLOCK, tmp, BLOCK * sizeof(float));
            s_mn[sn] = mn; s_sc[sn] = sc; sn++;
        } else {
            uint64_t j = rng_next() % (seen + 1);
            if (j < sample_cap) {
                memcpy(sample + j * BLOCK, tmp, BLOCK * sizeof(float));
                s_mn[j] = mn; s_sc[j] = sc;
            }
        }
        seen++;
    }
    /* held-out pool: stride-sampled odd blocks, raw + normalized */
    if (eval_cap > eval_pool) eval_cap = eval_pool;
    float *e_raw = (float *)malloc(eval_cap * BLOCK * sizeof(float));
    float *e_norm = (float *)malloc(eval_cap * BLOCK * sizeof(float));
    float *e_mn = (float *)malloc(eval_cap * sizeof(float));
    float *e_sc = (float *)malloc(eval_cap * sizeof(float));
    uint64_t stride = eval_pool / eval_cap;
    if (stride == 0) stride = 1;
    uint64_t en = 0;
    for (uint64_t j = 0; j < eval_pool && en < eval_cap; j += stride) {
        uint64_t b = 2 * j + 1;
        memcpy(e_raw + en * BLOCK, data + b * BLOCK, BLOCK * sizeof(float));
        float mn, sc;
        normalize_block(data + b * BLOCK, BLOCK, tmp, &mn, &sc);
        memcpy(e_norm + en * BLOCK, tmp, BLOCK * sizeof(float));
        e_mn[en] = f16_to_float(float_to_half(mn));
        e_sc[en] = f16_to_float(float_to_half(sc));
        en++;
    }
    printf("sampled: %llu train, %llu held-out; prep %.1f s\n",
           (unsigned long long)sn, (unsigned long long)en, now_s() - t0);

    /* 1. base book */
    rng_state = 0x2545F4914F6CDD1Dull;
    float *cent = (float *)malloc((size_t)entries * BLOCK * sizeof(float));
    learn_book_w(sample, NULL, sn, BLOCK, entries, iters, cent);

    /* residual means per entry, from the training sample:
     * normalized space and reconstructed weight space */
    double *rsum_n = (double *)calloc((size_t)entries * BLOCK, sizeof(double));
    double *rsum_w = (double *)calloc((size_t)entries * BLOCK, sizeof(double));
    uint64_t *rcnt = (uint64_t *)calloc(entries, sizeof(uint64_t));
    if (!rsum_n || !rsum_w || !rcnt) { fprintf(stderr, "oom\n"); return 1; }
    for (uint64_t b = 0; b < sn; b++) {
        const float *x = sample + b * BLOCK;
        /* the raw block behind this normalized sample is not
         * retained; weight-space residual uses the frame:
         * w = mn + sc * x exactly (normalization is invertible
         * up to the fp16 frame rounding already applied). */
        int k = nearest_centroid(cent, entries, BLOCK, x);
        rcnt[k]++;
        const float *c = cent + (size_t)k * BLOCK;
        for (uint32_t i = 0; i < BLOCK; i++) {
            double rn = (double)x[i] - c[i];
            rsum_n[(size_t)k * BLOCK + i] += rn;
            rsum_w[(size_t)k * BLOCK + i] += (double)s_sc[b] * rn;
        }
    }
    float *corr_n = (float *)malloc((size_t)entries * BLOCK * sizeof(float));
    float *corr_w = (float *)malloc((size_t)entries * BLOCK * sizeof(float));
    double mag_n = 0.0, mag_w = 0.0;
    uint64_t used = 0;
    for (uint32_t k = 0; k < entries; k++) {
        double acc_n = 0.0, acc_w = 0.0;
        for (uint32_t i = 0; i < BLOCK; i++) {
            double vn = rcnt[k] ? rsum_n[(size_t)k * BLOCK + i] / (double)rcnt[k] : 0.0;
            double vw = rcnt[k] ? rsum_w[(size_t)k * BLOCK + i] / (double)rcnt[k] : 0.0;
            corr_n[(size_t)k * BLOCK + i] = (float)vn;
            corr_w[(size_t)k * BLOCK + i] = (float)vw;
            acc_n += vn * vn;
            acc_w += vw * vw;
        }
        if (rcnt[k]) { used++; mag_n += dsqrt(acc_n); mag_w += dsqrt(acc_w); }
    }
    printf("entries used: %llu/%u; mean per-entry residual-mean norm: "
           "normalized %.6f, weight space %.6f\n",
           (unsigned long long)used, entries,
           used ? mag_n / (double)used : 0.0,
           used ? mag_w / (double)used : 0.0);

    /* folded book: centroids plus the normalized correction */
    float *cent_f = (float *)malloc((size_t)entries * BLOCK * sizeof(float));
    for (size_t i = 0; i < (size_t)entries * BLOCK; i++)
        cent_f[i] = cent[i] + corr_n[i];

    /* 5. weighted retrain: sample weights = block scale squared */
    double *wts = (double *)malloc(sn * sizeof(double));
    for (uint64_t b = 0; b < sn; b++) wts[b] = (double)s_sc[b] * (double)s_sc[b];
    rng_state = 0x2545F4914F6CDD1Dull;
    float *cent_w = (float *)malloc((size_t)entries * BLOCK * sizeof(float));
    learn_book_w(sample, wts, sn, BLOCK, entries, iters, cent_w);

    uint64_t book_bytes = (uint64_t)entries * BLOCK * 4;
    uint64_t stream_bytes = nblocks * 5;    /* index + fp16 frame */
    uint64_t weights = nblocks * BLOCK;
    double bpw1 = (double)(stream_bytes + book_bytes) * 8.0 / (double)weights;
    double bpw2 = (double)(stream_bytes + 2 * book_bytes) * 8.0 / (double)weights;

    struct { const char *label; const float *cent; const float *corr; int mode; double bpw; } vs[] = {
        { "1 base",            cent,   NULL,    0, bpw1 },
        { "2 tableN (norm)",   cent,   corr_n,  1, bpw2 },
        { "3 tableW (weight)", cent,   corr_w,  2, bpw2 },
        { "4 folded",          cent_f, NULL,    0, bpw1 },
        { "5 weighted retrain", cent_w, NULL,   0, bpw1 },
    };
    for (size_t v = 0; v < sizeof(vs) / sizeof(vs[0]); v++) {
        double t1 = now_s();
        double r = eval_variant(vs[v].cent, vs[v].corr, vs[v].mode,
                                e_raw, e_norm, e_mn, e_sc, en, entries);
        printf("  %-20s held-out rel L2 %.4f   complete %.4f bpw   "
               "(eval %.1f s)\n", vs[v].label, r, vs[v].bpw, now_s() - t1);
        fflush(stdout);
    }
    printf("total %.1f s\n", now_s() - t0);
    return 0;
}
